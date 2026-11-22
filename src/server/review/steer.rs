//! What steers the Review queue's Recommended order from outside its rows ([`snapshot`], one
//! [`Steer`] per `GET /v1/review/next`, handed to the review index and the walk alike): the
//! measured recall of the newest trained models ([`recall`]), the folder's parked spots
//! ([`spots`]) and the camera days held out for good ([`heldout_listed`]). Nothing here runs a walk
//! on the request path, and nothing here writes a file.
//!
//! Held-out days. The `camera/date` keys of the review folder's `heldout-days.json`
//! ([`crate::collect::labels::heldout::load`], read only: never written, locked or moved aside
//! here), one cached entry keyed on the file's (path, modification time, length) like the recall's.
//! A missing file lists nothing; a newer or damaged one lists nothing too, with a warning once per
//! change of the key, and the order then holds out the hash split's days alone (an export refuses
//! such a file, so no export follows it until it is fixed). A list read again that is the same
//! keeps its `Arc`.
//!
//! Recall. A [`Measured`] from `runs.json` through [`crate::training::bench::measured_recall`],
//! one cached entry keyed on `runs.json`'s (path, modification time, length): a request costs one
//! `stat` while nothing changed, and the table is rebuilt only after the training queue saved a run
//! (it rewrites the whole file). Records come from the started queue, else from the file through
//! the pure [`crate::training::task::read_records`] ([`super::super::training::records_read_only`]):
//! a damaged `runs.json` is no measured run here, and is never renamed, moved aside or written (the
//! queue deals with it when it starts). A missing file, or one without measured runs, is None,
//! logged once per change of the key; never an error to the request.
//!
//! Spots. The static spots of the folder's reviewed frames ([`SpotCache`]), by the export's rules
//! ([`crate::collect::labels::need::spot_obs`]: no reviewed frame with a pending box or a label off
//! the class list): from the review index's reviewed rows when the folder has a built index
//! ([`crate::collect::labels::index::reviewed_obs`], no label file read), rebuilt when those rows
//! changed and the last build is [`SPOTS_INDEXED_EVERY`] old; neutral (empty) while a folder that
//! takes an index has none built yet; else walked ([`crate::collect::labels::queue::reviewed_obs`],
//! a small folder) at most every [`SPOTS_EVERY`]. Both make the observations through the one
//! function, at the same whole-pixel geometry, so a spot near [`statics::SPOT_IOU`] is the same
//! either way. A rebuild that finds the same table serves the old `Arc` on, so the index's
//! per-shard need cache keeps its parts. A request serves the last table built for the folder and
//! class list, empty (so neutral: no box counts as parked) until the first is in; so the order may
//! shift a little when a new table lands. Builds run on one background thread at a time; one asked
//! for while another runs is remembered (the latest) and run after it; a build that panics keeps
//! the old table and waits its interval before the next try (a guard). In tests the build runs on
//! the request's thread at every ask that is due, the intervals zero, so an answer never depends
//! on a thread's timing; the decision ([`refresh_due`]) and the builder ([`build`]) are the
//! service's.

use crate::collect::labels::need::Steer;
use crate::collect::labels::recall::Measured;
use crate::collect::labels::statics;
use crate::server::AppState;
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Least time between two builds of a folder's spot table without an index.
pub const SPOTS_EVERY: Duration = Duration::from_secs(10 * 60);
/// Least time between two builds from the index's rows, while they change.
pub const SPOTS_INDEXED_EVERY: Duration = Duration::from_secs(60);

/// What steers the Recommended order of the folder under review `dir` for the class list
/// (`classes`, `hidden`) now: one snapshot for the whole request.
pub(super) fn snapshot(
    state: &AppState,
    dir: &Path,
    classes: &[String],
    hidden: &[String],
) -> Arc<Steer> {
    Arc::new(Steer {
        recall: recall(state),
        spots: spots(dir, classes, hidden),
        heldout_listed: Some(heldout_listed(dir)),
    })
}

/// A spot table to build: for which folder and class list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Job {
    /// The folder and class list.
    pub key: String,
    /// The index rows' stamp when asked (None: no built index).
    pub stamp: Option<u64>,
    pub dir: PathBuf,
    pub classes: Vec<String>,
    pub hidden: Vec<String>,
}

/// The table last built for one key.
#[derive(Debug)]
struct SpotSlot {
    /// The stamp of the rows it was built from (None: walked, or neutral).
    stamp: Option<u64>,
    /// When it was built (or a build of it last failed).
    at: Instant,
    table: Arc<statics::Table>,
    /// Dropped by a rescan ([`forget_spots`]), or built from files read before one: due without
    /// waiting its interval, once its rows changed (or at once without an index).
    forgotten: bool,
}

/// Keys kept at once (folders and class lists): one is in use, the others are switches back.
const SLOTS: usize = 4;

#[derive(Debug, Default)]
struct Slots {
    slots: HashMap<String, SpotSlot>,
    building: bool,
    /// The latest build asked for while one ran.
    queued: Option<Job>,
    /// Bumped by [`forget_spots`]: a build that began before keeps its table forgotten.
    epoch: u64,
}

impl Slots {
    /// Keep `table` for `key`, built from rows of `stamp`: the old `Arc` when it is the same
    /// table. At most [`SLOTS`] keys, the oldest let go.
    fn store(&mut self, key: &str, stamp: Option<u64>, table: statics::Table) {
        let table = match self.slots.get(key) {
            Some(old) if *old.table == table => old.table.clone(),
            _ => Arc::new(table),
        };
        self.slots.insert(
            key.to_string(),
            SpotSlot {
                stamp,
                at: Instant::now(),
                table,
                forgotten: false,
            },
        );
        while self.slots.len() > SLOTS {
            let Some(oldest) = self
                .slots
                .iter()
                .min_by_key(|(_, s)| s.at)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.slots.remove(&oldest);
        }
    }

    /// The table of `job`'s key now (empty when none), and whether a build of it is due.
    fn ask(&self, job: &Job, every: Duration) -> (Arc<statics::Table>, bool) {
        let slot = self.slots.get(&job.key);
        let table = slot.map(|s| s.table.clone()).unwrap_or_default();
        // A forgotten table is as old as can be: due when its rows changed, or without rows.
        let age = |s: &SpotSlot| {
            if s.forgotten {
                Duration::MAX
            } else {
                s.at.elapsed()
            }
        };
        let due = refresh_due(slot.map(|s| (s.stamp, age(s))), job.stamp, every);
        (table, due)
    }

    /// Forget the tables of the folder `dir` ([`forget_spots`]).
    fn forget(&mut self, dir: &Path) {
        self.epoch = self.epoch.wrapping_add(1);
        let prefix = format!("{}\u{1f}", dir.display());
        for (key, slot) in &mut self.slots {
            if key.starts_with(&prefix) {
                slot.forgotten = true;
            }
        }
    }
}

/// Whether the table of a key is to be built again: `built` = the stamp it was built from and its
/// age (None: never built), `stamp` = the rows' stamp now (None: no built index). Never built:
/// yes. Younger than `every`: no. Else, with no stamp (a walked folder) yes, and with one, yes when
/// the rows changed since.
pub(super) fn refresh_due(
    built: Option<(Option<u64>, Duration)>,
    stamp: Option<u64>,
    every: Duration,
) -> bool {
    match built {
        None => true,
        Some((_, age)) if age < every => false,
        Some((was, _)) => stamp.is_none() || was != stamp,
    }
}

/// The spot tables of the process (see the module docs): one build at a time, the latest asked
/// for run next, a panicking build waiting its interval.
#[derive(Debug)]
pub(super) struct SpotCache {
    slots: Mutex<Option<Slots>>,
}

/// On a build that did not finish (a panic): no longer building, and the key's last try is now,
/// its old table and stamp kept, so the next try waits `every`.
struct Building<'a> {
    cache: &'a SpotCache,
    key: String,
    done: bool,
}

impl Drop for Building<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let mut guard = self.cache.slots.lock_recover();
        let s = guard.get_or_insert_with(Slots::default);
        s.building = false;
        s.queued = None;
        match s.slots.get_mut(&self.key) {
            Some(slot) => slot.at = Instant::now(),
            None => {
                s.slots.insert(
                    self.key.clone(),
                    SpotSlot {
                        stamp: None,
                        at: Instant::now(),
                        table: Arc::default(),
                        forgotten: false,
                    },
                );
            }
        }
    }
}

/// What builds a table: the stamp of the rows it came from and the table.
pub(super) type Builder<'a> = &'a dyn Fn(&Job) -> (Option<u64>, statics::Table);

impl SpotCache {
    pub(super) const fn new() -> Self {
        Self {
            slots: Mutex::new(None),
        }
    }

    /// The table to serve for `job` now, and the job to build (the caller runs it with
    /// [`Self::run`] on a thread) when one is due ([`refresh_due`]). While a build runs, `job` is
    /// remembered for after it.
    pub(super) fn get(&self, job: Job, every: Duration) -> (Arc<statics::Table>, Option<Job>) {
        let mut guard = self.slots.lock_recover();
        let s = guard.get_or_insert_with(Slots::default);
        let (table, due) = s.ask(&job, every);
        if !due {
            return (table, None);
        }
        if s.building {
            s.queued = Some(job);
            return (table, None);
        }
        s.building = true;
        (table, Some(job))
    }

    /// Build `job` with `build`, then the latest job asked for meanwhile, until none is left.
    pub(super) fn run(&self, mut job: Job, build: Builder<'_>) {
        loop {
            let mut guard = Building {
                cache: self,
                key: job.key.clone(),
                done: false,
            };
            let epoch = self
                .slots
                .lock_recover()
                .get_or_insert_with(Slots::default)
                .epoch;
            let (stamp, table) = build(&job);
            let next = {
                let mut g = self.slots.lock_recover();
                let s = g.get_or_insert_with(Slots::default);
                s.store(&job.key, stamp, table);
                if s.epoch != epoch
                    && let Some(slot) = s.slots.get_mut(&job.key)
                {
                    // A rescan during the build: what it read may be from before.
                    slot.forgotten = true;
                }
                let next = s.queued.take().filter(|q| *q != job);
                if next.is_none() {
                    s.building = false;
                }
                next
            };
            guard.done = true;
            drop(guard);
            match next {
                Some(n) => job = n,
                None => return,
            }
        }
    }

    /// [`Self::get`] and, when due, the build on this thread (tests: no other build is waited
    /// for), then the key's table.
    #[cfg(test)]
    pub(super) fn get_now(
        &self,
        job: Job,
        every: Duration,
        build: Builder<'_>,
    ) -> Arc<statics::Table> {
        let due = {
            let mut guard = self.slots.lock_recover();
            guard.get_or_insert_with(Slots::default).ask(&job, every).1
        };
        if due {
            let (stamp, table) = build(&job);
            self.slots
                .lock_recover()
                .get_or_insert_with(Slots::default)
                .store(&job.key, stamp, table);
        }
        self.table_of(&job.key).unwrap_or_default()
    }

    /// Forget the tables of `dir` ([`forget_spots`]).
    fn forget(&self, dir: &Path) {
        self.slots
            .lock_recover()
            .get_or_insert_with(Slots::default)
            .forget(dir);
    }

    /// The table last built for `key`, if any.
    #[cfg(test)]
    fn table_of(&self, key: &str) -> Option<Arc<statics::Table>> {
        self.slots
            .lock_recover()
            .as_ref()
            .and_then(|s| s.slots.get(key))
            .map(|s| s.table.clone())
    }
}

/// The spot table of `job` and the stamp of the rows it came from (see the module docs): the
/// index's rows when it is built, neutral while a folder that takes an index has none yet, else a
/// walk.
pub(super) fn build(job: &Job) -> (Option<u64>, statics::Table) {
    use crate::collect::labels::{index, queue};
    let wanted = (job.classes.clone(), job.hidden.clone());
    if let Some((stamp, obs)) = index::reviewed_obs(&job.dir, Some(&wanted)) {
        return (Some(stamp), statics::Table::build(obs));
    }
    if index::building(&job.dir, Some(&wanted)) {
        return (None, statics::Table::default());
    }
    let all = queue::candidates(&job.dir, queue::QueueSource::All);
    if index::takes_index(all.len()) {
        // Its index comes with the first Next: a walk of a big folder is not this table's.
        return (None, statics::Table::default());
    }
    let w = crate::collect::labels::bulk::Wanted {
        classes: &job.classes,
        hidden: &job.hidden,
    };
    (
        None,
        statics::Table::build(queue::reviewed_obs(&job.dir, all, &w)),
    )
}

static SPOTS: SpotCache = SpotCache::new();

/// The folder `dir` was rescanned: its spot tables are due again without waiting their
/// interval: a walked folder's at the next Next, an indexed one's once the index's walk of
/// everything changed its rows. Each keeps serving its table until the new one is in; a build
/// running now stores what it found, still due ([`SpotSlot::forgotten`]).
pub(super) fn forget_spots(dir: &Path) {
    SPOTS.forget(dir);
}

/// The job of `dir` and the class list now.
fn job_of(dir: &Path, classes: &[String], hidden: &[String]) -> Job {
    let wanted = (classes.to_vec(), hidden.to_vec());
    let stamp = crate::collect::labels::index::rows_version(dir, Some(&wanted));
    Job {
        key: format!(
            "{}\u{1f}{}\u{1f}{}",
            dir.display(),
            classes.join("\u{1e}"),
            hidden.join("\u{1e}")
        ),
        stamp,
        dir: dir.to_path_buf(),
        classes: wanted.0,
        hidden: wanted.1,
    }
}

/// The interval of `job` (see the module docs; zero in tests).
fn every_of(job: &Job) -> Duration {
    if cfg!(test) {
        Duration::ZERO
    } else if job.stamp.is_some() {
        SPOTS_INDEXED_EVERY
    } else {
        SPOTS_EVERY
    }
}

/// The parked spots of `dir`'s reviewed frames for the class list, built on the request's thread
/// in tests (see the module docs).
#[cfg(test)]
pub(super) fn spots(dir: &Path, classes: &[String], hidden: &[String]) -> Arc<statics::Table> {
    let job = job_of(dir, classes, hidden);
    let every = every_of(&job);
    SPOTS.get_now(job, every, &build)
}

/// The parked spots of `dir`'s reviewed frames for the class list (see the module docs).
#[cfg(not(test))]
pub(super) fn spots(dir: &Path, classes: &[String], hidden: &[String]) -> Arc<statics::Table> {
    let job = job_of(dir, classes, hidden);
    let every = every_of(&job);
    let (table, due) = SPOTS.get(job, every);
    if let Some(job) = due {
        let key = job.key.clone();
        let spawned = std::thread::Builder::new()
            .name("review-spots".into())
            .spawn(move || {
                // A walk for the page: never ahead of detections for the CPU.
                crate::backend::priority::lower_this_thread(
                    crate::backend::priority::ThreadRole::Background,
                );
                SPOTS.run(job, &build);
            });
        if let Err(e) = spawned {
            tracing::warn!(
                "review queue: no thread to find the parked spots ({e}), so the order counts \
                 every box as unparked for now; it tries again on the next Next"
            );
            // As a build that failed: not building, the next try after the interval.
            drop(Building {
                cache: &SPOTS,
                key,
                done: false,
            });
        }
    }
    table
}

/// What the cached table was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Key {
    path: PathBuf,
    modified: Option<SystemTime>,
    len: u64,
}

/// The one cached table (None = the file was missing or measured nothing).
struct Entry {
    key: Option<Key>,
    path: PathBuf,
    value: Option<Arc<Measured>>,
}

static CACHE: Mutex<Option<Entry>> = Mutex::new(None);

/// The measured recall of the training runs of the current config (see the module docs).
pub fn recall(state: &AppState) -> Option<Arc<Measured>> {
    let path = crate::training::runs_file(&state.config_read());
    cached(&CACHE, &path, || {
        super::super::training::records_read_only(state)
    })
}

/// The key of `path` now; None when it cannot be read (missing).
fn key_of(path: &Path) -> Option<Key> {
    let m = std::fs::metadata(path).ok().filter(|m| m.is_file())?;
    Some(Key {
        path: path.to_path_buf(),
        modified: m.modified().ok(),
        len: m.len(),
    })
}

/// [`recall`] over `cache` for the `runs.json` at `path`, reading the records with `read` only
/// when the key changed.
fn cached(
    cache: &Mutex<Option<Entry>>,
    path: &Path,
    read: impl FnOnce() -> Vec<RunRecord>,
) -> Option<Arc<Measured>> {
    let key = key_of(path);
    let mut c = cache.lock_recover();
    if let Some(e) = c.as_ref()
        && e.path == path
        && e.key == key
    {
        return e.value.clone();
    }
    let value = match &key {
        None => {
            tracing::debug!(
                "no {}: the Review queue has no measured recall to steer by",
                path.display()
            );
            None
        }
        Some(_) => {
            let v = crate::training::bench::measured_recall(&read()).map(Arc::new);
            if v.is_none() {
                tracing::debug!(
                    "{} holds no finished run with per-class held-out recall: the Review queue \
                     has no measured recall to steer by",
                    path.display()
                );
            }
            v
        }
    };
    *c = Some(Entry {
        key,
        path: path.to_path_buf(),
        value: value.clone(),
    });
    value
}

/// The one cached list of held-out days.
struct Listed {
    key: Option<Key>,
    path: PathBuf,
    value: Arc<BTreeSet<String>>,
}

static LISTED: Mutex<Option<Listed>> = Mutex::new(None);

/// The camera days `heldout-days.json` of the review folder `dir` lists (see the module docs).
pub fn heldout_listed(dir: &Path) -> Arc<BTreeSet<String>> {
    listed_in(&LISTED, dir)
}

/// [`heldout_listed`] over `cache`.
fn listed_in(cache: &Mutex<Option<Listed>>, dir: &Path) -> Arc<BTreeSet<String>> {
    use crate::collect::labels::heldout;
    let path = dir.join(heldout::FILE);
    let key = key_of(&path);
    let mut c = cache.lock_recover();
    if let Some(e) = c.as_ref()
        && e.path == path
        && e.key == key
    {
        return e.value.clone();
    }
    let read = match &key {
        None => BTreeSet::new(),
        Some(_) => match heldout::load(dir) {
            Ok(file) => file.keys(),
            Err(e) => {
                tracing::warn!(
                    "review queue: {e:#}. Until it reads, the Recommended order holds out only the \
                     days the export's hash split does"
                );
                BTreeSet::new()
            }
        },
    };
    let value = match c.as_ref() {
        Some(e) if e.path == path && *e.value == read => e.value.clone(),
        _ => Arc::new(read),
    };
    *c = Some(Listed {
        key,
        path,
        value: value.clone(),
    });
    value
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::benchmark::ClassRecall;
    use crate::resources::catalog::TrainFamily;
    use crate::training::job::{BenchSummary, RecallCells, TrainOutput, TrainRequest};
    use crate::training::recipe::TrainRecipe;
    use crate::training::service::RunState;
    use std::cell::Cell;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop_steer_{name}_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A finished run measured over every scene (person: `gt` objects, `missed` missed).
    fn run(id: &str, gt: usize, missed: usize) -> serde_json::Value {
        let mut cells = RecallCells::default();
        cells
            .by_scene
            .insert("all".into(), vec![ClassRecall::new("person", gt, missed)]);
        cells
            .by_scene
            .insert("day".into(), vec![ClassRecall::new("person", gt, missed)]);
        let b = BenchSummary {
            heldout_fingerprint: Some("fp".into()),
            recall_cells: cells,
            ..BenchSummary::default()
        };
        let r = RunRecord {
            id: id.into(),
            name: id.into(),
            created: "2026-11-01T10:00:00Z".into(),
            request: TrainRequest {
                recipe: TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap(),
                datasets: Vec::new(),
                include_public: false,
                scene: Default::default(),
                classes: None,
                rename: Default::default(),
                benchmark_sets: Vec::new(),
                benchmark_devices: Vec::new(),
                allow_large: false,
                install_dir: None,
                gpu: None,
                excluded: Default::default(),
                balance: Default::default(),
                compare_group: None,
                train_seed: None,
                ignore_memory_check: false,
            },
            device: "cpu".into(),
            state: RunState::Done,
            stage: None,
            step: 8,
            steps: 8,
            percent: 100,
            detail: String::new(),
            progress: None,
            log: Vec::new(),
            dataset: None,
            output: Some(TrainOutput {
                name: id.into(),
                onnx: PathBuf::from(format!("/models/{id}.onnx")),
                yaml: PathBuf::from(format!("/models/{id}.yaml")),
                provenance: PathBuf::new(),
                family: crate::model::ModelFamilyKind::Yolo26,
                classes: vec!["person".into()],
                license: String::new(),
                heldout: None,
                device: String::new(),
                train_seconds: 0.0,
            }),
            registered: None,
            benchmark: Some(b),
            benchmark_error: None,
            baseline: None,
            error: None,
            finished: Some("2026-11-01T11:00:00Z".into()),
            run_dir: PathBuf::from(format!("/training/{id}")),
            retrying: false,
            resources: None,
        };
        serde_json::to_value(r).unwrap()
    }

    fn write(path: &Path, runs: &[serde_json::Value]) {
        std::fs::write(path, serde_json::to_vec(runs).unwrap()).unwrap();
    }

    fn read(path: &Path) -> Vec<RunRecord> {
        crate::training::service::load_bare_records(path)
    }

    #[test]
    fn the_table_is_cached_until_runs_json_changes() {
        let dir = temp("cache");
        let path = dir.join("runs.json");
        let cache = Mutex::new(None);
        let reads = Cell::new(0);
        let get = || {
            cached(&cache, &path, || {
                reads.set(reads.get() + 1);
                read(&path)
            })
        };
        // Missing: None, and nothing read.
        assert!(get().is_none());
        assert_eq!(reads.get(), 0);
        write(&path, &[run("a", 40, 4)]);
        // The record format must parse, or this test proves nothing.
        assert_eq!(read(&path).len(), 1);
        let first = get().unwrap();
        assert_eq!(reads.get(), 1);
        assert_eq!(first.overall.as_ref().unwrap().run.id, "a");
        // Unchanged: the same table, not read again.
        let again = get().unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(reads.get(), 1);
        // Changed (another length): read again.
        write(&path, &[run("a", 40, 4), run("bb", 50, 5)]);
        let next = get().unwrap();
        assert_eq!(reads.get(), 2);
        assert_eq!(next.overall.as_ref().unwrap().class("person").gt, 50);
        // Removed: None.
        std::fs::remove_file(&path).unwrap();
        assert!(get().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_or_empty_runs_json_is_none() {
        let dir = temp("damaged");
        let path = dir.join("runs.json");
        let cache = Mutex::new(None);
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(cached(&cache, &path, || read(&path)).is_none());
        std::fs::write(&path, b"[]").unwrap();
        assert!(cached(&cache, &path, || read(&path)).is_none());
        // A run without recall cells measures nothing.
        let mut r = run("a", 40, 4);
        r["benchmark"]["recall_cells"] = serde_json::json!({});
        write(&path, &[r]);
        assert!(cached(&cache, &path, || read(&path)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The listed held-out days: none without the file, the file's keys cached until it changes
    /// (the same `Arc` while unchanged, and for the same list read again), none for a newer or
    /// damaged file; and the file is only read: byte for byte, nothing beside it.
    #[test]
    fn the_listed_days_are_cached_until_the_file_changes() {
        use crate::collect::labels::heldout::FILE;
        let dir = temp("listed");
        let cache = Mutex::new(None);
        let get = || listed_in(&cache, &dir);
        assert!(get().is_empty());
        let day = |camera: &str, date: &str| {
            serde_json::json!({"camera": camera, "date": date, "added": "2026-11-01T00:00:00Z",
                "reason": "hash"})
        };
        let write = |days: Vec<serde_json::Value>| {
            let v = serde_json::json!({"version": 1, "days": days});
            std::fs::write(dir.join(FILE), serde_json::to_vec_pretty(&v).unwrap()).unwrap();
        };
        write(vec![day("drive", "2026-10-21"), day("yard", "2026-10-22")]);
        let first = get();
        let want: BTreeSet<String> = ["drive/2026-10-21".to_string(), "yard/2026-10-22".into()]
            .into_iter()
            .collect();
        assert_eq!(*first, want);
        assert!(Arc::ptr_eq(&first, &get()), "unchanged: the same Arc");
        // Another length, the same days (one twice): the same list, the same Arc.
        write(vec![
            day("drive", "2026-10-21"),
            day("yard", "2026-10-22"),
            day("yard", "2026-10-22"),
        ]);
        assert!(Arc::ptr_eq(&first, &get()));
        write(vec![day("drive", "2026-10-21")]);
        assert_eq!(get().len(), 1);
        // Newer, damaged: nothing listed, and the file stays as it is.
        let newer = br#"{"version": 9, "days": []}"#;
        std::fs::write(dir.join(FILE), newer).unwrap();
        assert!(get().is_empty());
        std::fs::write(dir.join(FILE), b"{ not json").unwrap();
        assert!(get().is_empty());
        assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), b"{ not json");
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        assert_eq!(names, [FILE.to_string()], "nothing written beside it");
        std::fs::remove_file(dir.join(FILE)).unwrap();
        assert!(get().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Through the app state: a config whose training folder has no `runs.json` answers None.
    #[test]
    fn the_state_without_runs_is_none() {
        let state = crate::server::test_util::test_state();
        assert!(recall(&state).is_none());
    }

    /// The snapshot of a folder without reviewed frames steers by nothing: no recall, no spot.
    #[test]
    fn a_snapshot_of_a_new_folder_is_neutral() {
        let state = crate::server::test_util::test_state();
        let dir = temp("snapshot");
        let s = snapshot(&state, &dir, &["car".to_string()], &[]);
        assert!(s.recall.is_none());
        assert!(s.spots.is_empty());
        // Held-out steering is on, with no listed day.
        assert!(s.heldout_listed.as_ref().unwrap().is_empty());
        assert_eq!(SPOTS_EVERY, std::time::Duration::from_secs(600));
        assert_eq!(SPOTS_INDEXED_EVERY, std::time::Duration::from_secs(60));
        // Without an index the builder walks (nothing here), with no stamp.
        let (stamp, t) = build(&job_for(&dir, "k", None));
        assert!(stamp.is_none() && t.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn job_for(dir: &Path, key: &str, stamp: Option<u64>) -> Job {
        Job {
            key: key.into(),
            stamp,
            dir: dir.to_path_buf(),
            classes: vec!["car".into()],
            hidden: Vec::new(),
        }
    }

    /// A table of `n` made-up spots of camera `cam` (each a car parked on two days).
    fn table(cam: &str, n: usize) -> statics::Table {
        let obs = (0..n).flat_map(|s| {
            (0..12).map(move |i| statics::Obs {
                camera: cam.into(),
                frame: format!("{s}-{i}"),
                day: if i < 6 { "d1" } else { "d2" }.into(),
                scene: "day".into(),
                label: "car".into(),
                bbox: [0.2 * s as f32, 0.0, 0.2 * s as f32 + 0.1, 0.1],
            })
        });
        statics::Table::build(obs)
    }

    #[test]
    fn a_table_is_rebuilt_after_its_interval_or_when_its_rows_changed() {
        let (min, hour) = (Duration::from_secs(60), Duration::from_secs(3600));
        // Never built: due, whatever the stamp.
        assert!(refresh_due(None, None, hour));
        assert!(refresh_due(None, Some(3), hour));
        // Within the interval: never.
        assert!(!refresh_due(Some((None, min)), None, hour));
        assert!(!refresh_due(Some((Some(1), min)), Some(2), hour));
        // No index (a walked folder): after the interval, always (its files may have changed).
        assert!(refresh_due(Some((None, hour)), None, min));
        assert!(refresh_due(Some((None, min)), None, min));
        // With an index: after the interval, only when the rows changed.
        assert!(!refresh_due(Some((Some(1), hour)), Some(1), min));
        assert!(refresh_due(Some((Some(1), hour)), Some(2), min));
        // An index built since a walk (or the other way): a change.
        assert!(refresh_due(Some((None, hour)), Some(1), min));
        assert!(refresh_due(Some((Some(1), hour)), None, min));
    }

    #[test]
    fn the_spot_cache_builds_once_serves_the_last_and_keeps_an_unchanged_table() {
        let cache = SpotCache::new();
        let dir = PathBuf::from("/nowhere");
        let every = Duration::from_secs(3600);
        // Nothing built: empty, and a build is due.
        let (t, due) = cache.get(job_for(&dir, "a", Some(1)), every);
        assert!(t.is_empty());
        let due = due.unwrap();
        // While it runs, a second ask for another folder waits; one for the same is no build.
        assert!(cache.get(job_for(&dir, "b", Some(7)), every).1.is_none());
        assert!(cache.get(job_for(&dir, "a", Some(1)), every).1.is_none());
        // The build of `a`, then the latest asked for (`a` again, which is no new work).
        let built = std::cell::Cell::new(0);
        cache.run(due, &|j| {
            built.set(built.get() + 1);
            (j.stamp, table("cam", if j.key == "a" { 1 } else { 2 }))
        });
        assert_eq!(built.get(), 1, "the queued job was the one just built");
        let first = cache.table_of("a").unwrap();
        assert_eq!(first.len(), 1);
        // The same key: served, nothing due while the rows did not change, nor within `every`.
        let (t, due) = cache.get(job_for(&dir, "a", Some(1)), every);
        assert!(Arc::ptr_eq(&t, &first) && due.is_none());
        assert!(cache.get(job_for(&dir, "a", Some(2)), every).1.is_none());
        // Past `every` the changed rows rebuild, the old table serving meanwhile; the same table
        // again keeps its `Arc` (the index's need cache keys on it).
        let (t, due) = cache.get(job_for(&dir, "a", Some(2)), Duration::ZERO);
        assert!(Arc::ptr_eq(&t, &first));
        // Another key while that build runs: queued, built right after.
        assert!(cache.get(job_for(&dir, "b", Some(9)), every).1.is_none());
        let seen = std::cell::RefCell::new(Vec::new());
        cache.run(due.unwrap(), &|j| {
            seen.borrow_mut().push(j.key.clone());
            (j.stamp, table("cam", if j.key == "a" { 1 } else { 2 }))
        });
        assert_eq!(*seen.borrow(), ["a", "b"]);
        assert!(Arc::ptr_eq(&cache.table_of("a").unwrap(), &first));
        let (t, due) = cache.get(job_for(&dir, "b", Some(9)), every);
        assert_eq!(t.len(), 2);
        assert!(due.is_none());
        // A switch back serves the key's own table, nothing due.
        let (t, due) = cache.get(job_for(&dir, "a", Some(2)), every);
        assert!(Arc::ptr_eq(&t, &first) && due.is_none());
        // A table that changed is a new `Arc`.
        let (_, due) = cache.get(job_for(&dir, "a", Some(3)), Duration::ZERO);
        cache.run(due.unwrap(), &|j| (j.stamp, table("cam", 3)));
        assert_eq!(cache.table_of("a").unwrap().len(), 3);
        // At most SLOTS keys: the oldest goes.
        for k in ["c", "d", "e", "f"] {
            let (_, due) = cache.get(job_for(&dir, k, None), every);
            cache.run(due.unwrap(), &|_| (None, table("cam", 1)));
        }
        assert!(cache.table_of("b").is_none() && cache.table_of("f").is_some());
    }

    /// A rescan ([`forget_spots`]) makes the folder's tables due without their interval, the old
    /// table serving meanwhile; a build it overtook stores its table still due; other folders'
    /// tables are left alone; an indexed table waits for its rows to change.
    #[test]
    fn a_rescan_makes_its_folders_tables_due_and_a_build_it_overtook_stays_due() {
        let cache = SpotCache::new();
        let (dir, other) = (PathBuf::from("/rescanned"), PathBuf::from("/other"));
        let key = |d: &Path| format!("{}\u{1f}car\u{1f}", d.display());
        let job = |d: &Path, stamp: Option<u64>| job_for(d, &key(d), stamp);
        let every = Duration::from_secs(3600);
        for d in [&dir, &other] {
            let (_, due) = cache.get(job(d, None), every);
            cache.run(due.unwrap(), &|j| (j.stamp, table("cam", 1)));
        }
        assert!(cache.get(job(&dir, None), every).1.is_none(), "fresh");
        cache.forget(&dir);
        let (t, due) = cache.get(job(&dir, None), every);
        assert_eq!(t.len(), 1, "the old table serves");
        assert!(
            cache.get(job(&other, None), every).1.is_none(),
            "other folder"
        );
        // A rescan while that build runs: its table is in, and due again.
        cache.run(due.unwrap(), &|j| {
            cache.forget(&dir);
            (j.stamp, table("cam", 2))
        });
        assert_eq!(cache.table_of(&key(&dir)).unwrap().len(), 2);
        let (_, due) = cache.get(job(&dir, None), every);
        cache.run(due.unwrap(), &|j| (j.stamp, table("cam", 2)));
        assert!(cache.get(job(&dir, None), every).1.is_none(), "built after");
        // Indexed: the same rows make the same table; changed rows are due at once.
        let (_, due) = cache.get(job(&dir, Some(5)), Duration::ZERO);
        cache.run(due.unwrap(), &|j| (j.stamp, table("cam", 2)));
        cache.forget(&dir);
        assert!(cache.get(job(&dir, Some(5)), every).1.is_none());
        assert!(cache.get(job(&dir, Some(6)), every).1.is_some());
    }

    #[test]
    fn a_panicking_spot_build_waits_its_interval_and_keeps_the_old_table() {
        let cache = SpotCache::new();
        let dir = PathBuf::from("/nowhere");
        let hour = Duration::from_secs(3600);
        let panics = || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let due = cache.get(job_for(&dir, "a", Some(2)), Duration::ZERO).1;
                cache.run(due.expect("due"), &|_| panic!("a build that fails"));
            }))
            .is_err()
        };
        // A first build that panics: no table, and no new try within the interval.
        assert!(panics());
        let (t, due) = cache.get(job_for(&dir, "a", Some(2)), hour);
        assert!(t.is_empty() && due.is_none(), "no respawn on every request");
        // After it: a try again, which builds.
        let due = cache.get(job_for(&dir, "a", Some(2)), Duration::ZERO).1;
        cache.run(due.expect("due again"), &|j| (j.stamp, table("cam", 1)));
        let good = cache.table_of("a").unwrap();
        assert_eq!(good.len(), 1);
        // A later build that panics keeps the table it had, and waits again.
        assert!(panics());
        let (t, due) = cache.get(job_for(&dir, "a", Some(5)), hour);
        assert!(Arc::ptr_eq(&t, &good) && due.is_none());
        // The stamp kept is the one built, so the changed rows are still due after the interval.
        assert!(
            cache
                .get(job_for(&dir, "a", Some(5)), Duration::ZERO)
                .1
                .is_some()
        );
    }

    /// A car parked at one place on two days: 12 reviewed frames of camera `lot`, 640x480.
    fn parked_folder(dir: &Path) -> Vec<String> {
        use crate::collect::labels::{BoxState, Frame, LabelBox, LabelFile, Source, Status};
        (0..12u32)
            .map(|i| {
                let date = if i < 6 { "2026-10-20" } else { "2026-10-21" };
                let stem = format!("{}T{:02}0000000Z_p{i}", date.replace('-', ""), 1 + i);
                let day = dir.join("day").join("lot").join(date);
                std::fs::create_dir_all(&day).unwrap();
                image::RgbImage::from_pixel(8, 6, image::Rgb([90, 90, 90]))
                    .save(day.join(format!("{stem}.jpg")))
                    .unwrap();
                let mut r = crate::collect::record::empty_record();
                r.id = stem.clone();
                r.scene = "day".into();
                r.camera = "lot".into();
                r.timestamp = format!("{date}T{:02}:00:00Z", 1 + i);
                r.width = 640;
                r.height = 480;
                std::fs::write(
                    day.join(format!("{stem}.json")),
                    serde_json::to_vec(&r).unwrap(),
                )
                .unwrap();
                let rel = format!("day/lot/{date}/{stem}.jpg");
                crate::collect::layout::append_review(dir, std::slice::from_ref(&rel)).unwrap();
                let labels = LabelFile {
                    status: Status::Reviewed,
                    reviewed_at: Some("2026-10-22T00:00:00Z".into()),
                    boxes: vec![LabelBox {
                        label: "car".into(),
                        bbox: [64.4, 48.0, 192.0, 144.6],
                        source: Source::Human,
                        state: BoxState::Accepted,
                        ..crate::collect::labels::bulk::test_support::auto_box(
                            "car",
                            0.9,
                            crate::collect::labels::Triage::Review,
                        )
                    }],
                    ..LabelFile::default()
                };
                let frame = Frame::resolve(dir, &rel).unwrap();
                crate::collect::labels::write(&frame.labels, &labels).unwrap();
                rel
            })
            .collect()
    }

    /// The service's path without its thread: a walked folder's table, then the index's once it
    /// is built (the same table, so the same `Arc`), rebuilt when its rows change.
    #[test]
    fn the_spot_table_follows_the_index_rows() {
        use crate::collect::labels::{index, queue};
        let dir = temp("rows");
        let rels = parked_folder(&dir);
        let classes = vec!["car".to_string()];
        let cache = SpotCache::new();
        let get = || cache.get_now(job_of(&dir, &classes, &[]), Duration::ZERO, &build);
        // No index: walked, and every ask walks again (the interval is zero here).
        let walked = get();
        assert_eq!(walked.camera("lot").len(), 1, "{walked:?}");
        assert!(Arc::ptr_eq(&get(), &walked), "the same table, the same Arc");
        // An index of the folder: its rows give the same table, under a stamp.
        index::tests::force(&dir);
        let f = queue::Filters {
            wanted: Some((classes.clone(), Vec::new())),
            ..queue::Filters::default()
        };
        assert!(index::answer(&dir, &f, None, None).is_some());
        let stamp = index::rows_version(&dir, Some(&(classes.clone(), Vec::new()))).unwrap();
        assert_eq!(job_of(&dir, &classes, &[]).stamp, Some(stamp));
        let (built, t) = build(&job_of(&dir, &classes, &[]));
        assert_eq!((built, &t), (Some(stamp), &*walked));
        assert!(Arc::ptr_eq(&get(), &walked));
        // A frame reopened with a pending box leaves the spot's evidence (the export's rule).
        for rel in &rels[..4] {
            let frame = crate::collect::labels::Frame::resolve(&dir, rel).unwrap();
            crate::collect::labels::update(&frame.labels, |l| {
                let mut l = l.unwrap();
                if let Some(b) = l.boxes.first_mut() {
                    b.state = crate::collect::labels::BoxState::Pending;
                }
                Ok(Some(l))
            })
            .unwrap();
        }
        assert!(index::answer(&dir, &f, None, None).is_some());
        let moved = index::rows_version(&dir, Some(&(classes.clone(), Vec::new()))).unwrap();
        assert_ne!(moved, stamp);
        let after = get();
        assert!(!Arc::ptr_eq(&after, &walked));
        assert!(
            after.camera("lot").is_empty(),
            "8 frames are too few: {after:?}"
        );
        index::tests::unforce(&dir);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
