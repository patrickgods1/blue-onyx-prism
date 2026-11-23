//! The benchmark breadcrumb: `<data root>/benchmark-breadcrumb.json` names the device and model a
//! run is about to use, flushed to the disk before that device starts, and goes when the run ends
//! with the process alive. A file still there at the next start therefore says what a run last
//! touched before the process or the machine went away; it does not say why.
//!
//! The service's runs and the `benchmark` CLI's sweeps write it alike. The running process holds
//! an OS lock on the file (released when it dies), so a run in another process, such as a CLI
//! sweep or a `train` CLI benchmarking its model, is never taken for one that was interrupted.
//! Nothing here can fail a benchmark: a data root that cannot be written costs the breadcrumb
//! and one warning.
//!
//! A planned stop (audit 2026-11 phase 1): a run that ends within the stop's wait drops its trail,
//! so nothing is reported. A run still busy when the service gives up on it (a compile cannot be
//! interrupted, and a device that hangs never returns) keeps its breadcrumb, marked by
//! [`Trail::mark_stopped`] with when the service stopped and how long the device had been busy: the
//! next start reports that the service stopped while the device was still working, never hides
//! it, since a hang in the driver looks exactly like a slow compile from here.

use crate::sync::LockExt;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// The file, under the data root.
pub const BREADCRUMB_FILE: &str = "benchmark-breadcrumb.json";

/// How long a run waits for a lock that looks held before it takes it for another run's: the
/// probes of [`left_behind`] and [`clear`] hold it for microseconds.
const HELD_WAIT: Duration = Duration::from_millis(300);
/// Opens a run tries when the path stops naming the file it locked.
const OPEN_TRIES: usize = 5;

/// What a run was about to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Breadcrumb {
    /// Canonical device spec.
    pub device: String,
    pub model: String,
    /// UTC, RFC 3339.
    pub time: String,
}

impl Breadcrumb {
    /// The note for the log and the Benchmark page.
    pub fn message(&self) -> String {
        format!(
            "The last benchmark did not end cleanly: its last recorded step was {} on {} at {}. \
             The service or the machine stopped during that run; this record cannot tell why. \
             If the machine hung, leave {} out of the next run.",
            self.model, self.device, self.time, self.device
        )
    }
}

/// The file: the breadcrumb, and what a planned stop added ([`Trail::mark_stopped`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Left {
    #[serde(flatten)]
    pub crumb: Breadcrumb,
    /// When the service stopped with the run still busy (UTC, RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_at: Option<String>,
    /// Seconds the device had been busy with the last step by then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busy_s: Option<u64>,
}

impl Left {
    /// The note for the log and the Benchmark page.
    pub fn message(&self) -> String {
        let c = &self.crumb;
        match &self.stopped_at {
            None => c.message(),
            Some(at) => format!(
                "The last benchmark did not end cleanly: the service was stopped at {at} while {} \
                 on {} was still running ({} s since {}), and it did not stop within the time \
                 the stop allows. A long compile on a slow GPU can do that; so can a device \
                 that hung. If the machine hung, leave {} out of the next run.",
                c.model,
                c.device,
                self.busy_s.unwrap_or(0),
                c.time,
                c.device
            ),
        }
    }
}

fn path_in(data_root: &Path) -> PathBuf {
    data_root.join(BREADCRUMB_FILE)
}

/// The breadcrumb of one run, removed when it is dropped (also by a panic: the process lives).
#[derive(Debug)]
pub(crate) struct Trail {
    path: PathBuf,
    state: Mutex<TrailState>,
}

#[derive(Debug)]
enum TrailState {
    /// No device started yet.
    Idle,
    /// Holds the file, locked, with the step last written and when.
    Open(File, Option<(Breadcrumb, std::time::Instant)>),
    /// Not writable, or held by a run in another process: no breadcrumb for this run.
    Off,
}

impl Trail {
    pub(crate) fn new(data_root: &Path) -> Self {
        Self {
            path: path_in(data_root),
            state: Mutex::new(TrailState::Idle),
        }
    }

    /// `model` is about to run on `device`: on the disk when this returns.
    pub(crate) fn note(&self, device: &str, model: &str) {
        let crumb = Breadcrumb {
            device: device.to_string(),
            model: model.to_string(),
            time: super::now(),
        };
        let mut state = self.state.lock_recover();
        if matches!(*state, TrailState::Idle) {
            *state = match self.open() {
                Ok(Some(file)) => TrailState::Open(file, None),
                Ok(None) => {
                    tracing::debug!("{} is held by another run", self.path.display());
                    TrailState::Off
                }
                Err(e) => {
                    self.warn(&e);
                    TrailState::Off
                }
            };
        }
        if let TrailState::Open(file, last) = &mut *state {
            let left = Left {
                crumb: crumb.clone(),
                stopped_at: None,
                busy_s: None,
            };
            if let Err(e) = write_crumb(file, &left) {
                self.warn(&e);
            }
            *last = Some((crumb, std::time::Instant::now()));
        }
    }

    /// The service is stopping and gives up on this run while it is still busy: add when, and how
    /// long its device had been at the last step, to the file (kept: see the module doc). Returns
    /// (device, model, seconds busy) for the log; None when the run holds no file or wrote no
    /// step. The file still goes if the run ends after all while the process lives.
    pub(crate) fn mark_stopped(&self) -> Option<(String, String, u64)> {
        let mut state = self.state.lock_recover();
        let TrailState::Open(file, Some((crumb, since))) = &mut *state else {
            return None;
        };
        let busy_s = since.elapsed().as_secs();
        let left = Left {
            crumb: crumb.clone(),
            stopped_at: Some(super::now()),
            busy_s: Some(busy_s),
        };
        if let Err(e) = write_crumb(file, &left) {
            self.warn(&e);
        }
        Some((crumb.device.clone(), crumb.model.clone(), busy_s))
    }

    /// The file, created and locked; None when another run holds it.
    fn open(&self) -> std::io::Result<Option<File>> {
        self.open_with(&|_| {})
    }

    /// [`Self::open`]; `between` runs after the file is opened and before it is locked (a test
    /// seam: a probe unlinking the path there).
    ///
    /// Two things keep a run from writing a breadcrumb nobody will find. A probe
    /// ([`left_behind`], [`clear`]) takes the same lock for a moment, so a held lock is waited
    /// for [`HELD_WAIT`] before it counts as another run's (a run that started during a page
    /// load's probe would otherwise write nothing for its whole life). And a probe removes the
    /// file while it holds the lock, so a handle opened just before may name an unlinked file:
    /// once locked, the path must still be this file, else it is opened again.
    fn open_with(&self, between: &dyn Fn(&Path)) -> std::io::Result<Option<File>> {
        for _ in 0..OPEN_TRIES {
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&self.path)?;
            between(&self.path);
            let waited = crate::fsx::lock_wait_with(
                &file,
                crate::fsx::LockKind::Exclusive,
                crate::fsx::deadline_after(HELD_WAIT),
                crate::fsx::Pauses::doubling(Duration::from_millis(1), Duration::from_millis(20)),
                || true,
            )?;
            if waited != crate::fsx::Waited::Held {
                return Ok(None);
            }
            if !names_file(&file, &self.path) {
                continue;
            }
            // A new name lives in its directory: flush that too. Directories cannot be opened
            // like this on Windows, where the journal covers the name.
            if let Some(dir) = self.path.parent()
                && let Ok(d) = File::open(dir)
            {
                let _ = d.sync_all();
            }
            return Ok(Some(file));
        }
        Err(std::io::Error::other(format!(
            "{} was removed again and again while it was being locked",
            self.path.display()
        )))
    }

    fn warn(&self, e: &std::io::Error) {
        tracing::warn!(
            "could not write {}: {e}. The benchmark goes on, but a hard hang will not leave the \
             last device on record; check that the data folder is writable",
            self.path.display()
        );
    }
}

/// Whether `path` still names the open `file` (a probe may have removed it, or removed it and
/// a newer run made a new one): same device and file number on Unix, the volume serial and
/// file index on Windows. False when the path is gone.
#[cfg(unix)]
fn names_file(file: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (file.metadata(), std::fs::metadata(path)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(windows)]
fn names_file(file: &File, path: &Path) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    /// `FILE_READ_ATTRIBUTES`; the lock holder may have the file open for more.
    const READ_ATTRIBUTES: u32 = 0x80;
    let Ok(other) = OpenOptions::new().access_mode(READ_ATTRIBUTES).open(path) else {
        return false;
    };
    match (
        winapi_util::file::information(file),
        winapi_util::file::information(&other),
    ) {
        (Ok(a), Ok(b)) => {
            a.volume_serial_number() == b.volume_serial_number() && a.file_index() == b.file_index()
        }
        _ => false,
    }
}

#[cfg(not(any(unix, windows)))]
fn names_file(_file: &File, _path: &Path) -> bool {
    true
}

fn write_crumb(file: &mut File, crumb: &Left) -> std::io::Result<()> {
    let text = serde_json::to_vec_pretty(crumb)?;
    file.set_len(0)?;
    file.rewind()?;
    file.write_all(&text)?;
    file.sync_all()
}

#[cfg(test)]
impl Trail {
    /// What the file says, read through the run's own handle: on Windows a lock is mandatory, so
    /// another handle cannot read a live run's file.
    pub(in crate::benchmark) fn recorded(&self) -> Breadcrumb {
        self.recorded_left().crumb
    }

    /// [`Self::recorded`] with what a stop added.
    pub(in crate::benchmark) fn recorded_left(&self) -> Left {
        let state = self.state.lock_recover();
        let TrailState::Open(file, _) = &*state else {
            panic!("no file open");
        };
        let mut file = file.try_clone().unwrap();
        file.rewind().unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        serde_json::from_str(&text).unwrap()
    }
}

impl Drop for Trail {
    fn drop(&mut self) {
        let state = std::mem::replace(&mut *self.state.lock_recover(), TrailState::Off);
        if let TrailState::Open(file, _) = state {
            // Under the lock, so nobody reads it as left behind between the two steps.
            let removed = std::fs::remove_file(&self.path);
            drop(file);
            if let Err(e) = removed
                && let Err(e2) = std::fs::remove_file(&self.path)
            {
                tracing::warn!(
                    "could not remove {}: {e} / {e2}; the next start will report this run as \
                     interrupted",
                    self.path.display()
                );
            }
        }
    }
}

/// The file `data_root` holds with no run behind it: None when absent, or held by a live run.
/// Unreadable content is dropped: there is nothing to report from it.
pub(crate) fn left_behind(data_root: &Path) -> Option<Left> {
    let path = path_in(data_root);
    let mut file = File::open(&path).ok()?;
    file.try_lock().ok()?;
    let mut text = String::new();
    let read = file.read_to_string(&mut text);
    match read.ok().and_then(|_| serde_json::from_str(&text).ok()) {
        Some(crumb) => Some(crumb),
        None => {
            // Still under the lock, so a run starting now never has its file removed.
            if std::fs::remove_file(&path).is_err() {
                drop(file);
                let _ = std::fs::remove_file(&path);
            }
            None
        }
    }
}

/// Remove what [`left_behind`] reported, unless a run took the file over since.
pub(super) fn clear(data_root: &Path) {
    let path = path_in(data_root);
    if let Ok(file) = File::open(&path)
        && file.try_lock().is_ok()
    {
        // Under the lock (as `Trail::drop` does): a run starting now never has its file removed.
        // Where that is refused, after the lock is released.
        if std::fs::remove_file(&path).is_err() {
            drop(file);
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!("could not remove {}: {e}", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-crumb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_run_that_ends_leaves_nothing() {
        let root = tmp();
        let trail = Trail::new(&root);
        assert!(!path_in(&root).exists(), "nothing before a device starts");
        trail.note("ort:cuda:0", "m");
        trail.note("openvino:gpu.0", "m");
        let crumb = trail.recorded();
        assert_eq!(
            (crumb.device.as_str(), crumb.model.as_str()),
            ("openvino:gpu.0", "m"),
            "the last device"
        );
        assert!(chrono::DateTime::parse_from_rfc3339(&crumb.time).is_ok());
        drop(trail);
        assert!(!path_in(&root).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    /// What a hard hang leaves: the file, unlocked once the process is gone (here: forgotten
    /// without the `Drop`, its handle closed).
    #[test]
    fn a_run_that_never_ended_is_left_behind_and_a_live_one_is_not() {
        let root = tmp();
        let trail = Trail::new(&root);
        trail.note("ort:directml:1", "IPcam-general");
        assert_eq!(left_behind(&root), None, "a live run holds it");
        clear(&root);
        assert!(path_in(&root).exists(), "clear leaves a live run's file");

        let TrailState::Open(file, _) =
            std::mem::replace(&mut *trail.state.lock_recover(), TrailState::Off)
        else {
            panic!("the trail holds the file");
        };
        drop(file);
        drop(trail);
        let crumb = left_behind(&root).expect("left behind");
        assert_eq!(crumb.stopped_at, None);
        assert_eq!(crumb.crumb.device, "ort:directml:1");
        let text = crumb.message();
        assert!(
            text.contains("IPcam-general") && text.contains("ort:directml:1"),
            "{text}"
        );
        assert!(text.contains("cannot tell why"), "{text}");
        clear(&root);
        assert_eq!(left_behind(&root), None);
        assert!(!path_in(&root).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    /// A run the service gave up on at a planned stop keeps its breadcrumb, which then says the
    /// service stopped while the device was busy (and still names the device).
    #[test]
    fn a_run_left_busy_at_a_stop_is_reported_as_such() {
        let root = tmp();
        let trail = Trail::new(&root);
        assert_eq!(trail.mark_stopped(), None, "no step yet");
        trail.note("openvino:gpu.0", "rtdetr");
        let (device, model, _) = trail.mark_stopped().expect("a step was running");
        assert_eq!(
            (device.as_str(), model.as_str()),
            ("openvino:gpu.0", "rtdetr")
        );
        let left = trail.recorded_left();
        assert!(
            left.stopped_at.is_some() && left.busy_s.is_some(),
            "{left:?}"
        );
        // The process exits with the run still busy: the file stays, unlocked.
        let TrailState::Open(file, _) =
            std::mem::replace(&mut *trail.state.lock_recover(), TrailState::Off)
        else {
            panic!("the trail holds the file");
        };
        drop(file);
        drop(trail);
        let left = left_behind(&root).expect("kept");
        let text = left.message();
        assert!(text.contains("the service was stopped at"), "{text}");
        assert!(text.contains("rtdetr on openvino:gpu.0"), "{text}");
        assert!(text.contains("leave openvino:gpu.0 out"), "{text}");
        assert!(!text.contains("cannot tell why"), "{text}");
        // An older file (no stop note) still reads.
        clear(&root);
        std::fs::write(
            path_in(&root),
            br#"{"device":"ort:cpu","model":"m","time":"2026-11-06T10:00:00Z"}"#,
        )
        .unwrap();
        let left = left_behind(&root).expect("read");
        assert!(left.stopped_at.is_none() && left.message().contains("cannot tell why"));
        // A run that ends after the mark, with the process alive, still removes it.
        clear(&root);
        let trail = Trail::new(&root);
        trail.note("ort:cpu", "m");
        trail.mark_stopped();
        drop(trail);
        assert!(!path_in(&root).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_second_run_stays_off_the_file_of_the_first() {
        let root = tmp();
        let first = Trail::new(&root);
        first.note("ort:cpu", "a");
        let second = Trail::new(&root);
        second.note("openvino:cpu", "b");
        assert_eq!(first.recorded().model, "a");
        drop(second);
        assert!(path_in(&root).exists(), "the second one removes nothing");
        drop(first);
        assert!(!path_in(&root).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_file_nobody_can_read_is_dropped_and_a_missing_folder_is_no_error() {
        let root = tmp();
        std::fs::write(path_in(&root), b"not json").unwrap();
        assert_eq!(left_behind(&root), None);
        assert!(!path_in(&root).exists());
        let trail = Trail::new(&root.join("gone"));
        trail.note("ort:cpu", "m");
        drop(trail);
        std::fs::remove_dir_all(&root).ok();
    }

    /// A probe (a page load's `clear` / `left_behind`) holding the lock for a moment while a run
    /// starts: the run waits and still writes its breadcrumb.
    #[test]
    fn a_run_starting_during_a_probe_still_writes() {
        let root = tmp();
        let probe = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path_in(&root))
            .unwrap();
        probe.try_lock().unwrap();
        let trail = Trail::new(&root);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(60));
                drop(probe);
            });
            trail.note("ort:cpu", "m");
        });
        assert_eq!(trail.recorded().device, "ort:cpu");
        drop(trail);
        std::fs::remove_dir_all(&root).ok();
    }

    /// A probe removed the file between the run's open and its lock: the run must not keep
    /// writing to the unlinked file; it opens the path again.
    #[test]
    fn a_path_removed_between_open_and_lock_is_opened_again() {
        let root = tmp();
        let trail = Trail::new(&root);
        let first = std::cell::Cell::new(true);
        let mut file = trail
            .open_with(&|p| {
                if first.replace(false) {
                    std::fs::remove_file(p).unwrap();
                }
            })
            .unwrap()
            .expect("locked");
        // Checked through the path, not through `names_file`: what the run writes must be what
        // a later start reads. Read once the handle is closed: on Windows its lock is mandatory,
        // so a read beside it fails.
        std::io::Write::write_all(&mut file, b"written by the run").unwrap();
        drop(file);
        assert_eq!(
            std::fs::read(path_in(&root)).ok().as_deref(),
            Some(&b"written by the run"[..]),
            "the run holds a file the path no longer names"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// `clear` removes the file under its lock: after it, a run can open a fresh one.
    #[test]
    fn clear_and_left_behind_remove_what_they_report() {
        let root = tmp();
        std::fs::write(
            path_in(&root),
            br#"{"device":"ort:cpu","model":"m","time":"2026-11-06T10:00:00Z"}"#,
        )
        .unwrap();
        assert!(left_behind(&root).is_some());
        clear(&root);
        assert!(!path_in(&root).exists());
        let trail = Trail::new(&root);
        trail.note("ort:cpu", "m");
        assert_eq!(trail.recorded().model, "m");
        drop(trail);
        std::fs::remove_dir_all(&root).ok();
    }
}
