//! A run's per-epoch metrics ([`TrainProgress::history`]) on disk: `<run dir>/history.jsonl`,
//! appended while the run trains, so `runs.json` holds only the record (best epoch, latest
//! values, state) and is no longer rewritten with every epoch so far.
//!
//! Format: one JSON object per line, an [`EpochPoint`] (`epoch`, `elapsed_s`, `metrics`) holding
//! what changed in that epoch's row since the last line ([`HistoryLog::sync`]): the epoch event
//! writes the row, a later eval event of the same epoch adds its keys. A reader merges the lines
//! by epoch, later values winning ([`read`]). Lines that do not parse (a crash or a full disk
//! mid-append) are skipped and the next append starts on a fresh line
//! ([`crate::collect::layout::append_line`]), so a torn tail costs at most that row's last change.
//! A new attempt of the run (an out-of-memory restart from epoch 1) truncates the file: its
//! record forgets the earlier attempt's progress too.
//!
//! Readers ([`attach`]) put the file's epochs back on the copies of the records they hand out,
//! through a cache checked by file size and time ([`cached`]), so the page's poll reads a file
//! again only when it changed. A record of an older `runs.json` that still holds its history is
//! moved to the file once ([`migrate`]), never losing an epoch: the file is written (merged with
//! one already there) before the record drops it, and a record whose file cannot be written keeps
//! its history in `runs.json`.

use super::RunRecord;
use crate::collect::layout;
use crate::sync::LockExt;
use crate::training::progress::{EpochPoint, MAX_HISTORY, TrainProgress};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// File of a run's history in its run folder.
pub const HISTORY_FILE: &str = "history.jsonl";

/// Histories kept by [`cached`] (least recently used dropped first).
const CACHE_ENTRIES: usize = 256;

/// `<run dir>/history.jsonl`.
pub fn history_path(run_dir: &Path) -> PathBuf {
    run_dir.join(HISTORY_FILE)
}

/// Fold `lines` into rows by epoch, in epoch order: `elapsed_s` the largest, metrics of later
/// lines over earlier ones. Epoch 0 is no row (the progress never makes one); the last
/// [`MAX_HISTORY`] epochs are kept, like the progress.
fn merge_lines(lines: impl IntoIterator<Item = EpochPoint>) -> Vec<EpochPoint> {
    let mut rows: BTreeMap<u32, EpochPoint> = BTreeMap::new();
    for l in lines {
        if l.epoch == 0 {
            continue;
        }
        let row = rows.entry(l.epoch).or_insert_with(|| EpochPoint {
            epoch: l.epoch,
            ..EpochPoint::default()
        });
        if l.elapsed_s.is_finite() {
            row.elapsed_s = row.elapsed_s.max(l.elapsed_s);
        }
        row.metrics
            .extend(l.metrics.into_iter().filter(|(_, v)| v.is_finite()));
    }
    let skip = rows.len().saturating_sub(MAX_HISTORY);
    rows.into_values().skip(skip).collect()
}

/// What a history file holds, or why it could not be read.
fn read_file(path: &Path) -> Result<Vec<EpochPoint>, String> {
    let lines = layout::read_lines::<EpochPoint>(path);
    match lines.failed {
        Some(e) => Err(e),
        None => Ok(merge_lines(lines.lines)),
    }
}

/// The epochs in the history file `path`, merged ([`merge_lines`]); none when it is missing or
/// cannot be read.
pub fn read(path: &Path) -> Vec<EpochPoint> {
    read_file(path).unwrap_or_default()
}

struct Cached {
    len: u64,
    modified: Option<SystemTime>,
    points: Arc<Vec<EpochPoint>>,
    used: u64,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<PathBuf, Cached>,
    tick: u64,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// [`read`] through a process-wide cache: the file is parsed again only when its size or time
/// changed since the last read (every append changes the size).
pub fn cached(path: &Path) -> Arc<Vec<EpochPoint>> {
    let Ok(meta) = std::fs::metadata(path) else {
        if let Some(c) = CACHE.lock_recover().as_mut() {
            c.entries.remove(path);
        }
        return Arc::default();
    };
    let (len, modified) = (meta.len(), meta.modified().ok());
    {
        let mut guard = CACHE.lock_recover();
        let c = guard.get_or_insert_with(Cache::default);
        c.tick += 1;
        let tick = c.tick;
        if let Some(e) = c.entries.get_mut(path)
            && e.len == len
            && e.modified == modified
        {
            e.used = tick;
            return Arc::clone(&e.points);
        }
    }
    // Parsed outside the lock. A file that changes meanwhile has another size next time, so a
    // stale parse is read again then.
    let points = match read_file(path) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            tracing::debug!("training history {e}");
            return Arc::default();
        }
    };
    let mut guard = CACHE.lock_recover();
    let c = guard.get_or_insert_with(Cache::default);
    c.tick += 1;
    let used = c.tick;
    c.entries.insert(
        path.to_path_buf(),
        Cached {
            len,
            modified,
            points: Arc::clone(&points),
            used,
        },
    );
    while c.entries.len() > CACHE_ENTRIES {
        let Some(oldest) = c
            .entries
            .iter()
            .min_by_key(|(_, e)| e.used)
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        c.entries.remove(&oldest);
    }
    points
}

/// Put the history back on `r`, a reader's copy of a stored record: `live` (the running job's
/// epochs) when given, else its history file ([`cached`]). A record that still holds a history
/// (older `runs.json` not migrated yet, or a file that could not be written) keeps its own.
/// `losses` is rebuilt from the epochs' `loss` when the record holds none.
pub fn attach(r: &mut RunRecord, live: Option<&Arc<Vec<EpochPoint>>>) {
    let Some(p) = r.progress.as_mut() else {
        return;
    };
    if p.history.is_empty() {
        let h = match live {
            Some(h) => Arc::clone(h),
            None => cached(&history_path(&r.run_dir)),
        };
        p.history = h.as_ref().clone();
    }
    if p.losses.is_empty() {
        p.losses = p
            .history
            .iter()
            .filter_map(|e| e.metrics.get("loss").copied())
            .collect();
    }
}

/// The best epoch of an interrupted run from its epochs: its progress was saved up to
/// [`super::PROGRESS_SAVE_EVERY`] before it stopped, its history file at every epoch
/// ([`TrainProgress::recompute_best`]; the record's own history when it still holds one).
pub fn refresh_best(p: &mut TrainProgress, run_dir: &Path) {
    if !p.history.is_empty() {
        p.recompute_best();
        return;
    }
    let h = read(&history_path(run_dir));
    if h.is_empty() {
        return;
    }
    p.history = h;
    p.recompute_best();
    p.history.clear();
}

/// `points` as history lines, one full row each.
fn lines_of(points: &[EpochPoint]) -> String {
    let mut out = String::new();
    for p in points {
        if let Ok(l) = serde_json::to_string(p) {
            out.push_str(&l);
            out.push('\n');
        }
    }
    out
}

/// Move the history of records from an older `runs.json` into their history files: merged with
/// a file already there (its values win: it is the newer write), written atomically, and only
/// then dropped from the record (with `losses`, rebuilt from it on read). A record whose run
/// folder is gone, or whose file cannot be read or written, keeps its history (warned). Returns
/// how many records changed, so the caller saves `runs.json` once.
pub fn migrate(records: &mut [RunRecord]) -> usize {
    let mut moved = 0;
    for r in records {
        let Some(p) = r.progress.as_mut() else {
            continue;
        };
        if p.history.is_empty() {
            continue;
        }
        if !r.run_dir.is_dir() {
            tracing::debug!(
                run = %r.id,
                "run folder {} is gone: its epochs stay in runs.json",
                r.run_dir.display()
            );
            continue;
        }
        let path = history_path(&r.run_dir);
        let on_disk = match read_file(&path) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    run = %r.id,
                    "cannot read the training history {e}: the run's epochs stay in runs.json \
                     for now; check that no other program holds the file, then restart"
                );
                continue;
            }
        };
        let merged = merge_lines(p.history.iter().cloned().chain(on_disk.iter().cloned()));
        if merged != on_disk
            && let Err(e) = layout::write_atomic(&path, lines_of(&merged).as_bytes())
        {
            tracing::warn!(
                run = %r.id,
                "cannot write the training history {} ({e}): the run's epochs stay in \
                 runs.json; check free disk space and the permissions of its folder",
                path.display()
            );
            continue;
        }
        p.history.clear();
        p.losses.clear();
        moved += 1;
    }
    if moved > 0 {
        tracing::info!("moved the epoch history of {moved} run(s) out of runs.json");
    }
    moved
}

/// The running job's side of the history file: appends what changed in the progress' history
/// since the last call ([`Self::sync`]).
#[derive(Debug)]
pub struct HistoryLog {
    path: PathBuf,
    /// The rows as the file holds them (merged), in epoch order.
    written: Vec<EpochPoint>,
    /// The file was truncated for this attempt (the first write of a log does it).
    opened: bool,
    /// A write failed: nothing more is written, and the record keeps the history itself.
    failed: bool,
    /// Bytes appended so far (the measurement in the tests).
    bytes: u64,
    /// The [`TrainProgress::history_rev`] last brought to the file (None: none yet).
    seen: Option<u64>,
}

impl HistoryLog {
    pub fn new(run_dir: &Path) -> Self {
        Self {
            path: history_path(run_dir),
            written: Vec::new(),
            opened: false,
            failed: false,
            bytes: 0,
            seen: None,
        }
    }

    /// A write failed: the history is no longer complete on disk.
    pub fn failed(&self) -> bool {
        self.failed
    }

    /// Bytes this log wrote.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The record a reader of the service gets while the job runs: the progress without its
    /// lists, or all of it once the file is incomplete ([`Self::failed`]).
    pub fn stored(&self, tp: &TrainProgress) -> TrainProgress {
        if self.failed {
            tp.clone()
        } else {
            tp.without_history()
        }
    }

    /// Bring the file up to `tp`'s history: a line per row that is new or changed, with the
    /// changed metrics only, appended in one write; nothing to compare when its
    /// [`TrainProgress::history_rev`] was seen last time (most reports: a batch). A history that
    /// went back (a new attempt from epoch 1) starts the file over. Returns whether the history
    /// differs from what was last seen (the caller's live copy is then stale).
    pub fn sync(&mut self, tp: &TrainProgress) -> bool {
        let rev = tp.history_rev.0;
        if rev != 0 && self.seen == Some(rev) {
            return false;
        }
        self.seen = Some(rev);
        self.sync_rows(&tp.history)
    }

    /// [`Self::sync`] by comparing every row.
    fn sync_rows(&mut self, history: &[EpochPoint]) -> bool {
        let restarted = match (history.last(), self.written.last()) {
            (None, Some(_)) => true,
            (Some(h), Some(w)) => h.epoch < w.epoch,
            _ => false,
        };
        if restarted {
            self.written.clear();
            self.opened = false;
            if !self.failed {
                self.truncate();
            }
        }
        // Rows the progress dropped off the front (past MAX_HISTORY) stay in the file.
        if let Some(first) = history.first() {
            let drop = self.written.partition_point(|w| w.epoch < first.epoch);
            self.written.drain(..drop);
        }
        let mut lines = String::new();
        let mut changed = restarted;
        for p in history {
            let line = match self.written.binary_search_by_key(&p.epoch, |w| w.epoch) {
                Ok(i) => {
                    let Some(w) = self.written.get_mut(i) else {
                        continue;
                    };
                    let delta: BTreeMap<String, f64> = p
                        .metrics
                        .iter()
                        .filter(|(k, v)| w.metrics.get(*k) != Some(*v))
                        .map(|(k, v)| (k.clone(), *v))
                        .collect();
                    if delta.is_empty() && p.elapsed_s <= w.elapsed_s {
                        continue;
                    }
                    w.elapsed_s = w.elapsed_s.max(p.elapsed_s);
                    w.metrics.extend(delta.clone());
                    EpochPoint {
                        epoch: p.epoch,
                        elapsed_s: w.elapsed_s,
                        metrics: delta,
                    }
                }
                Err(i) => {
                    self.written.insert(i, p.clone());
                    p.clone()
                }
            };
            changed = true;
            if let Ok(l) = serde_json::to_string(&line) {
                lines.push_str(&l);
                lines.push('\n');
            }
        }
        if !lines.is_empty() && !self.failed {
            self.append(lines.trim_end_matches('\n'));
        }
        changed
    }

    /// Start the file over for a new attempt (or the job's first write).
    fn truncate(&mut self) {
        let made = self
            .path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::File::create(&self.path).map(drop));
        match made {
            Ok(()) => self.opened = true,
            Err(e) => self.fail(&e),
        }
    }

    fn append(&mut self, lines: &str) {
        if !self.opened {
            self.truncate();
            if self.failed {
                return;
            }
        }
        match layout::append_line(&self.path, lines) {
            Ok(()) => self.bytes += lines.len() as u64 + 1,
            Err(e) => self.fail(&e),
        }
    }

    fn fail(&mut self, e: &std::io::Error) {
        if !self.failed {
            tracing::warn!(
                "cannot write the training history {} ({e}): this run's epochs are kept in \
                 runs.json instead; check free disk space and the permissions of the run folder",
                self.path.display()
            );
        }
        self.failed = true;
    }
}

#[cfg(test)]
mod tests;
