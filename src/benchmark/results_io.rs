//! Writing the saved benchmark results safely: one writer at a time, whole files only.
//!
//! - **The results lock** ([`lock`]): every load-modify-save of `benchmark.json` and the files
//!   saved with it (`benchmark-gt.json`, `benchmark-preds.json`) holds it, never longer than
//!   that one load-modify-save (never across a run): a run's save of each
//!   model and its final tidy-up (service and CLI), a Re-score of Your cameras (route and
//!   `benchmark --rescore`), the re-rank of saved results for new settings and the removal of a
//!   model's results. Two parts, taken in this order and released in reverse:
//!   - a process-wide slot per results file ([`Holder`] says who holds it), waited for on a
//!     condition variable. A run's start ([`hold_for_start`]) refuses at once while a Re-score
//!     holds it (`POST /v1/benchmark` answers 409), and a Re-score checks under the lock that no
//!     run is active, so the two never overlap;
//!   - an fs4 lock on [`LOCK_FILE`] beside `benchmark.json`, so the service and a
//!     `blue-onyx-prism benchmark` command exclude each other. Waited for at most
//!     [`FILE_WAIT`] (a Re-score of a large review folder walks it for a while), then an error
//!     that names the file; a folder where the file cannot be created or locked (read-only, a
//!     network share) goes on under the process slot alone, with one warning per folder.
//! - **Whole files** ([`write_atomic`]): each save writes its own temporary file in the target's
//!   folder (`.<name>.<uuid>.tmp`, opened with `create_new`, so two writers never share one;
//!   [`crate::fsx::write_atomic_with`]), flushes it and renames it over the target: a reader sees
//!   the old file or the new one.

use anyhow::{Result, bail};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::sync::LockExt;

/// The lock file beside `benchmark.json`.
pub const LOCK_FILE: &str = ".benchmark.lock";
/// Longest a writer waits for another process's hold of [`LOCK_FILE`].
pub const FILE_WAIT: Duration = Duration::from_secs(600);
/// The longest pause between two tries of the file lock.
const FILE_POLL: Duration = Duration::from_millis(50);

/// Who holds a results file's slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holder {
    /// A load-modify-save (a run's save, a re-rank, a removal).
    Writer,
    /// A Re-score of Your cameras: a run may not start meanwhile.
    Rescore,
    /// A run being started (only while its state turns to running).
    Start,
}

#[derive(Debug, Default)]
struct Slot {
    held: Mutex<Option<Holder>>,
    freed: Condvar,
}

/// The slots by results file.
static SLOTS: Mutex<Option<HashMap<PathBuf, Arc<Slot>>>> = Mutex::new(None);
/// Folders whose [`LOCK_FILE`] could not be used (warned once).
static NO_FILE_LOCK: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);

fn slot_of(results: &Path) -> Arc<Slot> {
    let mut slots = SLOTS.lock_recover();
    slots
        .get_or_insert_with(HashMap::new)
        .entry(results.to_path_buf())
        .or_default()
        .clone()
}

/// A hold of a results file ([`lock`]); released on drop.
#[derive(Debug)]
pub struct ResultsLock {
    /// Before the slot: the file lock goes first.
    file: Option<std::fs::File>,
    slot: Arc<Slot>,
}

impl Drop for ResultsLock {
    fn drop(&mut self) {
        // Explicitly (a child process spawned now would share the open file until it execs).
        if let Some(f) = self.file.take()
            && let Err(e) = fs4::FileExt::unlock(&f)
        {
            tracing::debug!("benchmark: releasing {LOCK_FILE}: {e}");
        }
        *self.slot.held.lock_recover() = None;
        self.slot.freed.notify_all();
    }
}

/// A Re-score holds the results file: a run must not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rescoring;

impl std::fmt::Display for Rescoring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "the saved results are being re-scored against the current labels; start the \
             benchmark once that has finished",
        )
    }
}

impl std::error::Error for Rescoring {}

/// Take the slot of `results` for `holder`, waiting while another holds it; `refuse_rescore`:
/// return [`Rescoring`] at once while a Re-score holds it instead of waiting.
fn take_slot(results: &Path, holder: Holder, refuse_rescore: bool) -> Result<Arc<Slot>, Rescoring> {
    let slot = slot_of(results);
    let mut held = slot.held.lock_recover();
    loop {
        match *held {
            None => break,
            Some(Holder::Rescore) if refuse_rescore => return Err(Rescoring),
            Some(_) => {
                held = slot
                    .freed
                    .wait(held)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }
    *held = Some(holder);
    drop(held);
    Ok(slot)
}

/// Hold `results` (see the module docs): the process slot, then [`LOCK_FILE`] beside it. Errors
/// when another process holds the file for longer than [`FILE_WAIT`].
pub fn lock(results: &Path, holder: Holder) -> Result<ResultsLock> {
    let slot = take_slot(results, holder, false).map_err(anyhow::Error::new)?;
    // Released with the slot if the file lock fails.
    let mut guard = ResultsLock { file: None, slot };
    guard.file = file_lock(results, FILE_WAIT)?;
    Ok(guard)
}

/// The slot of `results` while a run starts (no file lock: a start writes nothing):
/// [`Rescoring`] at once while a Re-score holds it, else waits for another writer's save.
pub fn hold_for_start(results: &Path) -> Result<ResultsLock, Rescoring> {
    let slot = take_slot(results, Holder::Start, true)?;
    Ok(ResultsLock { file: None, slot })
}

/// The fs4 lock of [`LOCK_FILE`] beside `results`, waiting at most `wait`; None when the file
/// cannot be used (warned once per folder).
fn file_lock(results: &Path, wait: Duration) -> Result<Option<std::fs::File>> {
    let dir = results.parent().unwrap_or(Path::new("."));
    let path = dir.join(LOCK_FILE);
    let without = |why: String| {
        let mut warned = NO_FILE_LOCK.lock_recover();
        if warned
            .get_or_insert_with(HashSet::new)
            .insert(dir.to_path_buf())
        {
            tracing::warn!(
                "benchmark: {} {why}, so saves of the benchmark results here are serialized in \
                 this process only; a `blue-onyx-prism benchmark` command writing them at the \
                 same moment could overwrite one",
                path.display()
            );
        }
    };
    std::fs::create_dir_all(dir).ok();
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            without(format!("cannot be opened ({e})"));
            return Ok(None);
        }
    };
    let waited = crate::fsx::lock_wait_with(
        &file,
        crate::fsx::LockKind::Exclusive,
        crate::fsx::deadline_after(wait),
        crate::fsx::Pauses::doubling(Duration::from_millis(1), FILE_POLL),
        || true,
    );
    match waited {
        Ok(crate::fsx::Waited::Held) => Ok(Some(file)),
        Ok(crate::fsx::Waited::TimedOut | crate::fsx::Waited::Stopped) => bail!(
            "{} stayed locked by another process for {} s (a `blue-onyx-prism benchmark` \
             run or re-score?): the benchmark results were not saved. Let it finish, then \
             try again",
            path.display(),
            wait.as_secs()
        ),
        Err(e) => {
            without(format!("cannot be locked ({e})"));
            Ok(None)
        }
    }
}

/// Write `bytes` to `path` through a temporary file of its own in the same folder, flushed to
/// disk, then renamed over `path` ([`crate::fsx::write_atomic_with`], see the module docs). The
/// folder is created when missing.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let opts = crate::fsx::Options {
        create_parent: true,
        ..crate::fsx::Options::default()
    };
    crate::fsx::write_atomic_with(path, bytes, &opts).map_err(|e| match e.step {
        crate::fsx::Step::Rename => anyhow::anyhow!("writing {}: {}", path.display(), e.source),
        _ => anyhow::Error::new(e.source).context(format!("writing {}", e.temp.display())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-results-io-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// One writer at a time: a second waits for the first's release.
    #[test]
    fn writers_wait_for_each_other() {
        let dir = tmp();
        let results = dir.join("benchmark.json");
        let first = lock(&results, Holder::Writer).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let r2 = results.clone();
        let t = std::thread::spawn(move || {
            let _second = lock(&r2, Holder::Writer).unwrap();
            tx.send(Instant::now()).unwrap();
        });
        std::thread::sleep(Duration::from_millis(150));
        assert!(rx.try_recv().is_err(), "the second waits");
        let released = Instant::now();
        drop(first);
        let got = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(got >= released);
        t.join().unwrap();
        assert!(dir.join(LOCK_FILE).is_file());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A start refuses while a Re-score holds the file, and waits for a plain writer.
    #[test]
    fn a_start_refuses_during_a_rescore() {
        let dir = tmp();
        let results = dir.join("benchmark.json");
        let r = lock(&results, Holder::Rescore).unwrap();
        assert_eq!(hold_for_start(&results).unwrap_err(), Rescoring);
        drop(r);
        let s = hold_for_start(&results).unwrap();
        drop(s);
        let w = lock(&results, Holder::Writer).unwrap();
        let r2 = results.clone();
        let t = std::thread::spawn(move || hold_for_start(&r2).map(drop));
        std::thread::sleep(Duration::from_millis(100));
        drop(w);
        assert_eq!(t.join().unwrap(), Ok(()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Another holder of the lock file (another process: another open file) is waited for, then
    /// an error naming the file.
    #[test]
    fn the_file_lock_excludes_another_holder() {
        let dir = tmp();
        let results = dir.join("benchmark.json");
        let other = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE))
            .unwrap();
        fs4::FileExt::try_lock(&other).unwrap();
        let e = file_lock(&results, Duration::from_millis(200)).unwrap_err();
        assert!(format!("{e:#}").contains(LOCK_FILE), "{e:#}");
        fs4::FileExt::unlock(&other).unwrap();
        assert!(
            file_lock(&results, Duration::from_millis(200))
                .unwrap()
                .is_some()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
