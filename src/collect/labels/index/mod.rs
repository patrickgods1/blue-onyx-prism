//! A background index of the folder under review, so `GET /v1/review/next` does not read every
//! frame on every Next or save (`docs/PRODUCTION_REVIEW.md`, "Review walk measurements"). Only that
//! route asks it ([`answer`]); [`super::queue::build`] stays the walk over the real files, for bulk
//! actions, previews and the CLI, and for a folder of under [`MIN_INDEXED_FRAMES`] frames.
//!
//! One index, of the current folder and wanted classes, on one thread that lives while the Review
//! tab is in use. The rows ([`row`]) hold everything `queue::build` reads of a frame, so the answer
//! ([`query`]) has to equal the walk's, compared whole by this module's tests. How it stays true:
//!
//! - **Own writes**: every label write marks its path ([`written`], under the label write lock), and
//!   the next query re-reads those files before it answers. Over [`DIRTY_MAX`] paths, their day
//!   folders go stale instead and the thread walks them at once; `layout::append_review` and
//!   `layout::rewrite_index` mark the queue and a day folder the same way.
//! - **Other processes and the collector**: the thread stats the folders every minute and walks what
//!   changed; every 30 minutes it reads everything again, which is what heals a file system with
//!   unreliable folder times. Rows keep serving meanwhile.
//! - **A write during a walk** is seen again after the swap, atomically with it ([`apply`]).
//! - **Deleted frames** retention got to first: the frames an answer shows are checked.
//!
//! An answer has [`Answer::indexing`] while rows it counts are still to come (a build, stale days:
//! never `done == total`), and otherwise holds every write of this process that finished before
//! the query began, but for frames of a day folder new since the last walk that the thread had not
//! walked within the query's wait.
//!
//! Locks, outermost first: `Index::walk` (one walk at a time), `Index::drain`, `Index::state`,
//! `Index::names`. [`HUB`] is a leaf: nothing is taken while holding it, so [`written`] taking it
//! under the label write lock cannot deadlock, and nothing here takes the label write lock. File
//! reads happen off `state`: what a drain or swap applies under it was read before.

mod answer;
mod apply;
mod query;
mod refresh;
mod row;
#[cfg(test)]
pub(crate) mod tests;
mod walk;

pub use answer::{Answer, Indexing};

use super::Frame;
use super::queue::{self, Filters, QueueSource};
use crate::sync::{LockExt as _, RwLockExt as _};
use row::{Derive, Encoder, Names, Row, Shard};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use walk::{DayStat, QueueStat};

/// A folder of fewer frames is walked: reading it costs less than an index is worth.
pub const MIN_INDEXED_FRAMES: usize = 2000;
/// Most written paths a query re-reads. Over this, their day folders go stale and the thread walks
/// them: a bulk action writes thousands of files, each write staying O(1).
pub const DIRTY_MAX: usize = 1000;
/// The index goes over this, and it is dropped for good (until a restart or another folder): the
/// route walks as if there were none. Twice the 64 MB budget of a 300,000 frame folder.
pub const INDEX_CEILING_BYTES: usize = 128 * 1024 * 1024;
/// How often the thread looks for what changed on disk.
const REFRESH_EVERY: Duration = Duration::from_secs(60);
/// How often everything is read again.
const FULL_EVERY: Duration = Duration::from_secs(30 * 60);
/// No query for this long: the index and its thread go.
const IDLE_AFTER: Duration = Duration::from_secs(10 * 60);
/// How long a query waits for a build, or stale days, before answering from the rows so far.
const BUILD_WAIT: Duration = Duration::from_millis(150);
/// Least time between two passes over stale days, while a bulk action keeps making them.
const STALE_PAUSE: Duration = Duration::from_millis(250);
/// How long a panicked index keeps a folder walking before it is tried again.
const AFTER_PANIC: Duration = Duration::from_secs(10 * 60);
/// How often a query goes round when the frames it would show are gone.
const RESOLVE_TRIES: usize = 8;

/// The times and limit of an index, apart so tests can shorten them.
#[derive(Debug, Clone, Copy)]
struct Timing {
    refresh: Duration,
    full: Duration,
    idle: Duration,
    wait: Duration,
    pause: Duration,
    ceiling: usize,
}

impl Timing {
    const REAL: Self = Self {
        refresh: REFRESH_EVERY,
        full: FULL_EVERY,
        idle: IDLE_AFTER,
        wait: BUILD_WAIT,
        pause: STALE_PAUSE,
        ceiling: INDEX_CEILING_BYTES,
    };
}

/// What queries read; swapped in by the walks.
#[derive(Debug, Default)]
struct State {
    /// By `<scene>/<camera>/<date>` of the frame's path.
    shards: HashMap<String, Shard>,
    /// The day folders as last walked.
    days: HashMap<String, DayStat>,
    queue: QueueStat,
    /// An index line names a frame of another folder: only a walk of everything is exact.
    foreign: bool,
}

impl State {
    /// `rel`'s shard and row.
    fn locate(&self, rel: &str) -> Option<(&Shard, usize)> {
        let (stem, json) = row::stem_of(rel)?;
        let shard = self.shards.get(queue::day_of(rel))?;
        Some((shard, shard.find(&stem, json)?))
    }

    fn row(&self, rel: &str) -> Option<&Row> {
        let (shard, at) = self.locate(rel)?;
        shard.rows.get(at)
    }

    /// `layout::damaged_lines`.
    fn damaged(&self, all_days: bool) -> u64 {
        let days: u64 = if all_days {
            self.days.values().map(|d| d.damaged).sum()
        } else {
            0
        };
        self.queue.damaged + days
    }

    fn rows(&self) -> u64 {
        self.shards.values().map(|s| s.rows.len() as u64).sum()
    }

    fn bytes(&self) -> usize {
        self.shards.values().map(Shard::bytes).sum::<usize>()
            + self.shards.capacity() * size_of::<(String, Shard)>()
            + self.days.capacity() * size_of::<(String, DayStat)>()
    }
}

/// The index of one folder for one wanted list.
#[derive(Debug)]
struct Index {
    dir: PathBuf,
    wanted: Option<(Vec<String>, Vec<String>)>,
    derive: Derive,
    timing: Timing,
    /// One walk at a time.
    walk: Mutex<()>,
    /// Held while what was read is applied: a query waits for a drain in progress, so a write that
    /// finished before it began is in what it reads.
    drain: Mutex<()>,
    state: RwLock<State>,
    names: RwLock<Names>,
    /// Set for good when the index stops being current.
    cancelled: AtomicBool,
    /// The first build is running.
    building: AtomicBool,
    /// A pass is walking stale days.
    catching_up: AtomicBool,
    /// Frames walked, and to walk, of the running build or stale pass.
    done: AtomicU64,
    total: AtomicU64,
    started: Instant,
    /// Milliseconds from `started` to the last query.
    last_query: AtomicU64,
}

impl Index {
    fn new(dir: &Path, wanted: Option<(Vec<String>, Vec<String>)>) -> Self {
        Self {
            derive: Derive::new(wanted.as_ref()),
            dir: dir.to_path_buf(),
            wanted,
            timing: timing(dir),
            walk: Mutex::new(()),
            drain: Mutex::new(()),
            state: RwLock::new(State::default()),
            names: RwLock::new(Names::default()),
            cancelled: AtomicBool::new(false),
            building: AtomicBool::new(true),
            catching_up: AtomicBool::new(false),
            done: AtomicU64::new(0),
            total: AtomicU64::new(0),
            started: Instant::now(),
            last_query: AtomicU64::new(0),
        }
    }

    fn is(&self, dir: &Path, wanted: Option<&(Vec<String>, Vec<String>)>) -> bool {
        self.dir == dir && self.wanted.as_ref() == wanted
    }

    fn encoder(&self) -> Encoder<'_> {
        Encoder {
            derive: &self.derive,
            names: &self.names,
        }
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn touch(&self) {
        let ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_query.store(ms, Ordering::Relaxed);
    }

    /// How long ago the last query was.
    fn idle_for(&self) -> Duration {
        self.started.elapsed().saturating_sub(Duration::from_millis(
            self.last_query.load(Ordering::Relaxed),
        ))
    }

    /// Frames still to come into the rows, or days known to be out of date.
    fn behind(&self, stale: bool) -> bool {
        stale || self.building.load(Ordering::Acquire) || self.catching_up.load(Ordering::Acquire)
    }

    fn bytes(&self) -> usize {
        self.state.read_recover().bytes() + self.names.read_recover().bytes()
    }

    /// How long a query waits for what is behind.
    fn wait(&self) -> Duration {
        #[cfg(test)]
        if tests::held(&self.dir) {
            return Duration::ZERO;
        }
        self.timing.wait
    }

    /// Now, for the stats of a walk: a test can move it on, so that nothing it wrote is within
    /// the racy window.
    fn now(&self) -> SystemTime {
        #[cfg(test)]
        if let Some(ahead) = tests::clock_ahead(&self.dir) {
            return SystemTime::now() + ahead;
        }
        SystemTime::now()
    }

    /// Wait until nothing is behind, or until `until`.
    fn settle(self: &Arc<Self>, until: Instant) {
        let mut hub = HUB.lock_recover();
        loop {
            let left = until.saturating_duration_since(Instant::now());
            if self.cancelled() || !self.behind(hub.stale(self)) || left.is_zero() {
                return;
            }
            hub = CHANGED
                .wait_timeout(hub, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// The progress an answer shows: frames whose rows are current, of the frames there are. A
    /// build counts the frames walked; a pass over stale days counts the rows outside them, and so
    /// do stale days still waiting for the thread. None when nothing is behind, and when every
    /// frame counted is current: the last day of a build is in and only its swap is left, or the
    /// stale days hold no row (a label file of a frame no line names). A query has waited for
    /// those, and its drain applied this process's writes to every row there is.
    fn indexing(self: &Arc<Self>) -> Option<Indexing> {
        let stale: Vec<String> = {
            let mut hub = HUB.lock_recover();
            hub.of(self)
                .map(|c| c.log.stale.iter().cloned().collect())
                .unwrap_or_default()
        };
        let (done, total) =
            if self.building.load(Ordering::Acquire) || self.catching_up.load(Ordering::Acquire) {
                (
                    self.done.load(Ordering::Acquire),
                    self.total.load(Ordering::Acquire),
                )
            } else if !stale.is_empty() {
                let state = self.state.read_recover();
                let behind: u64 = stale
                    .iter()
                    .filter_map(|day| state.shards.get(day))
                    .map(|s| s.rows.len() as u64)
                    .sum();
                let rows = state.rows();
                (rows.saturating_sub(behind), rows)
            } else {
                return None;
            };
        (done < total).then_some(Indexing { done, total })
    }

    /// The answer of the rows as they are, with the frames it shows.
    fn read(self: &Arc<Self>, f: &Filters, after: Option<&str>) -> (Answer, Vec<String>) {
        // Taken first: an answer without it must have nothing behind it.
        let indexing = self.indexing();
        let state = self.state.read_recover();
        let names = self.names.read_recover();
        let (mut answer, shown) = query::answer(&state, &names, f, after);
        answer.indexing = indexing;
        (answer, shown)
    }

    /// Whichever frames of `shown` are gone.
    fn missing(&self, shown: Vec<String>) -> Vec<String> {
        shown
            .into_iter()
            .filter(|rel| Frame::resolve(&self.dir, rel).is_none())
            .collect()
    }
}

/// Writes one reader has not seen yet.
#[derive(Debug, Default)]
struct Writes {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    rels: HashSet<String>,
    queue: bool,
}

/// What changed since it was last looked at, in the current folder.
#[derive(Debug)]
struct Log {
    root: PathBuf,
    /// `root` canonical: label paths come either way.
    canon: Option<PathBuf>,
    /// For the next query.
    pending: Writes,
    /// Days to walk.
    stale: HashSet<String>,
    /// For the walk in progress, seen again at its swap.
    since: Option<Writes>,
}

impl Log {
    fn new(dir: &Path) -> Self {
        Self {
            root: dir.to_path_buf(),
            canon: dir.canonicalize().ok(),
            pending: Writes::default(),
            stale: HashSet::new(),
            since: None,
        }
    }

    /// `path` from the root of the folder, when it is in it.
    fn inside<'a>(&self, path: &'a Path) -> Option<&'a Path> {
        path.strip_prefix(&self.root)
            .ok()
            .or_else(|| self.canon.as_ref().and_then(|c| path.strip_prefix(c).ok()))
    }

    /// The frame of the label file at `path`, when it is one of this folder.
    fn rel_of(&self, path: &Path) -> Option<String> {
        let mut parts = self.inside(path)?.iter().map(std::ffi::OsStr::to_str);
        let (scene, camera, date, file) = (
            parts.next()??,
            parts.next()??,
            parts.next()??,
            parts.next()??,
        );
        if parts.next().is_some() {
            return None;
        }
        let stem = file.strip_suffix(super::LABELS_SUFFIX)?;
        Some(format!("{scene}/{camera}/{date}/{stem}.jpg"))
    }

    /// Note `rel` in `writes`, or its day as stale when that holds [`DIRTY_MAX`].
    fn note(writes: &mut Writes, stale: &mut HashSet<String>, rel: &str) {
        if writes.rels.len() < DIRTY_MAX || writes.rels.contains(rel) {
            writes.rels.insert(rel.to_string());
        } else {
            stale.insert(queue::day_of(rel).to_string());
        }
    }

    fn written(&mut self, rel: &str) {
        Self::note(&mut self.pending, &mut self.stale, rel);
        if let Some(since) = &mut self.since {
            Self::note(since, &mut self.stale, rel);
        }
    }
}

/// The one index and what is known about it.
#[derive(Debug)]
struct Current {
    index: Arc<Index>,
    log: Log,
}

/// A folder that is walked for now, with no index to try.
#[derive(Debug)]
struct Refused {
    dir: PathBuf,
    /// None: until a restart.
    until: Option<Instant>,
}

/// All that lives past a call. A leaf lock: see the module's first lines.
#[derive(Debug)]
struct Hub {
    /// At most one: a query of another folder or wanted list drops it ([`same_group`]).
    current: Vec<Current>,
    /// At most one folder per group.
    refused: Vec<Refused>,
}

impl Hub {
    fn of(&mut self, index: &Index) -> Option<&mut Current> {
        self.current
            .iter_mut()
            .find(|c| std::ptr::eq(Arc::as_ptr(&c.index), index))
    }

    /// `index` has days to walk.
    fn stale(&mut self, index: &Index) -> bool {
        self.of(index).is_some_and(|c| !c.log.stale.is_empty())
    }

    /// Whether `dir` may have an index now. Asking for a folder ends the refusal of another one:
    /// a folder change gives an index another try.
    fn allows(&mut self, dir: &Path) -> bool {
        self.refused.retain(|r| {
            (r.dir == dir || !same_group(&r.dir, dir)) && r.until.is_none_or(|u| Instant::now() < u)
        });
        !self.refused.iter().any(|r| r.dir == dir)
    }

    /// Walk `dir` for now: until `until`, None = until a restart or another folder.
    fn refuse(&mut self, dir: &Path, until: Option<Instant>) {
        self.refused.retain(|r| r.dir != dir);
        self.refused.push(Refused {
            dir: dir.to_path_buf(),
            until,
        });
    }

    /// Stop the indexes `drop` picks, and wake their threads and whoever waits for them.
    fn clear_where(&mut self, drop: impl Fn(&Index) -> bool) {
        let mut stopped = false;
        self.current.retain(|c| {
            let goes = drop(&c.index);
            if goes {
                c.index.cancelled.store(true, Ordering::Release);
                stopped = true;
            }
            !goes
        });
        if stopped {
            CHANGED.notify_all();
        }
    }

    /// Stop `index`, if it is still current.
    fn remove(&mut self, index: &Index) {
        self.clear_where(|i| std::ptr::eq(i, index));
    }

    /// What [`open`] has to do about `dir` and `f` now: Some(the index to use, or None to walk),
    /// or None when a new index is to be made. Drops the index of another folder or wanted list.
    #[allow(clippy::option_option, reason = "use this / walk / make one")]
    fn current_for(&mut self, dir: &Path, f: &Filters) -> Option<Option<Arc<Index>>> {
        if let Some(c) = self
            .current
            .iter()
            .find(|c| c.index.is(dir, f.wanted.as_ref()))
        {
            return Some(Some(c.index.clone()));
        }
        self.clear_where(|i| same_group(&i.dir, dir));
        if !self.allows(dir) {
            return Some(None);
        }
        None
    }

    /// Stop every index.
    fn clear(&mut self) {
        self.clear_where(|_| true);
    }
}

static HUB: Mutex<Hub> = Mutex::new(Hub {
    current: Vec::new(),
    refused: Vec::new(),
});
/// Rung when a build ends or days go stale; waits on [`HUB`].
static CHANGED: Condvar = Condvar::new();

/// Whether `a` and `b` share the one index: always outside tests, which run side by side, each
/// folder its own unless a test puts folders together ([`tests::force_together`]).
fn same_group(a: &Path, b: &Path) -> bool {
    #[cfg(test)]
    return tests::same_group(a, b);
    #[cfg(not(test))]
    {
        let _ = (a, b);
        true
    }
}

/// A label file at `path` changed or went: its frame is read again by the next query. Runs under
/// the label write lock: no file work, O(1), and nothing at all in a process with no index.
pub fn written(path: &Path) {
    let mut hub = HUB.lock_recover();
    for c in &mut hub.current {
        let Some(rel) = c.log.rel_of(path) else {
            continue;
        };
        let before = c.log.stale.len();
        c.log.written(&rel);
        if c.log.stale.len() != before {
            CHANGED.notify_all();
        }
    }
}

/// `dir`'s `review-queue.jsonl` gained lines: the next query reads it again.
pub fn queue_written(dir: &Path) {
    let mut hub = HUB.lock_recover();
    for c in &mut hub.current {
        if c.log.root == dir || c.log.canon.as_deref() == Some(dir) {
            c.log.pending.queue = true;
            if let Some(since) = &mut c.log.since {
                since.queue = true;
            }
        }
    }
}

/// The index file of the day folder `day` was rewritten: the thread walks it again now.
pub fn day_written(day: &Path) {
    let mut hub = HUB.lock_recover();
    for c in &mut hub.current {
        if c.log.inside(day).is_some()
            && let Some(key) = walk::day_key(day)
        {
            c.log.stale.insert(key);
            CHANGED.notify_all();
        }
    }
}

/// The index's bytes, from capacities; None with no index.
pub fn bytes() -> Option<usize> {
    let index = HUB.lock_recover().current.first()?.index.clone();
    Some(index.bytes())
}

/// Drop the index, whatever it is of.
pub fn forget() {
    HUB.lock_recover().clear();
}

/// The times in force for an index of `dir`.
fn timing(dir: &Path) -> Timing {
    #[cfg(test)]
    if let Some(t) = tests::timing(dir) {
        return t;
    }
    let _ = dir;
    Timing::REAL
}

/// Whether a folder of `frames` frames takes an index.
fn takes_index(frames: usize) -> bool {
    frames >= MIN_INDEXED_FRAMES
}

/// The index for this query, started if it takes one.
fn open(dir: &Path, f: &Filters) -> Option<Arc<Index>> {
    // Tests run side by side and there is one index: only a test that asks gets it.
    #[cfg(test)]
    if !tests::forced(dir) {
        return None;
    }
    if let Some(index) = HUB.lock_recover().current_for(dir, f) {
        return index;
    }
    let frames = queue::candidates(dir, QueueSource::All).len();
    if !takes_index(frames) && !cfg!(test) {
        return None;
    }
    let mut hub = HUB.lock_recover();
    if let Some(index) = hub.current_for(dir, f) {
        return index;
    }
    let index = Arc::new(Index::new(dir, f.wanted.clone()));
    // Known now: the first answer, before the thread has read a line, says how many are to come.
    index.total.store(frames as u64, Ordering::Release);
    let thread = index.clone();
    let spawned = std::thread::Builder::new()
        .name("review-index".into())
        .spawn(move || refresh::run(thread));
    if let Err(e) = spawned {
        tracing::warn!("review index: no thread for it ({e}), so Next walks the folder");
        return None;
    }
    hub.current.push(Current {
        index: index.clone(),
        log: Log::new(dir),
    });
    Some(index)
}

/// The answer to `GET /v1/review/next` from the index. None: walk ([`Answer::walk`]), because the
/// folder is small, its index was dropped or there is no folder.
pub fn answer(dir: &Path, f: &Filters, after: Option<&str>) -> Option<Answer> {
    let out = answer_of(dir, f, after);
    #[cfg(test)]
    tests::note_served(dir, out.is_some());
    out
}

fn answer_of(dir: &Path, f: &Filters, after: Option<&str>) -> Option<Answer> {
    if !dir.is_dir() {
        HUB.lock_recover().clear_where(|i| i.dir == dir);
        return None;
    }
    let index = open(dir, f)?;
    index.touch();
    let until = Instant::now() + index.wait();
    index.settle(until);
    index.drain();
    // A write to a frame the rows do not have yet made its day stale: that is waited for too.
    index.settle(until);
    let mut tries = RESOLVE_TRIES;
    loop {
        let (answer, shown) = index.read(f, after);
        if index.cancelled() {
            return None;
        }
        let gone = index.missing(shown);
        tries = tries.saturating_sub(1);
        if gone.is_empty() || tries == 0 {
            tracing::debug!("review index: {} bytes", index.bytes());
            return Some(answer);
        }
        index.mark_gone(&gone);
    }
}
