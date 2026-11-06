//! The benchmark breadcrumb: `<data root>/benchmark-breadcrumb.json` names the device and model a
//! run is about to use, flushed to the disk before that device starts, and goes when the run ends
//! with the process alive. A file still there at the next start therefore says what a run last
//! touched before the process or the machine went away; it does not say why.
//!
//! The running process holds an OS lock on the file (released when it dies), so a run in another
//! process, such as a `train` CLI benchmarking its model, is never taken for one that was
//! interrupted. Nothing here can fail a benchmark: a data root that cannot be written costs the
//! breadcrumb and one warning.

use crate::sync::LockExt;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The file, under the data root.
pub const BREADCRUMB_FILE: &str = "benchmark-breadcrumb.json";

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

fn path_in(data_root: &Path) -> PathBuf {
    data_root.join(BREADCRUMB_FILE)
}

/// The breadcrumb of one run, removed when it is dropped (also by a panic: the process lives).
#[derive(Debug)]
pub(super) struct Trail {
    path: PathBuf,
    state: Mutex<TrailState>,
}

#[derive(Debug)]
enum TrailState {
    /// No device started yet.
    Idle,
    /// Holds the file, locked.
    Open(File),
    /// Not writable, or held by a run in another process: no breadcrumb for this run.
    Off,
}

impl Trail {
    pub(super) fn new(data_root: &Path) -> Self {
        Self {
            path: path_in(data_root),
            state: Mutex::new(TrailState::Idle),
        }
    }

    /// `model` is about to run on `device`: on the disk when this returns.
    pub(super) fn note(&self, device: &str, model: &str) {
        let crumb = Breadcrumb {
            device: device.to_string(),
            model: model.to_string(),
            time: super::now(),
        };
        let mut state = self.state.lock_recover();
        if matches!(*state, TrailState::Idle) {
            *state = match self.open() {
                Ok(Some(file)) => TrailState::Open(file),
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
        if let TrailState::Open(file) = &mut *state
            && let Err(e) = write_crumb(file, &crumb)
        {
            self.warn(&e);
        }
    }

    /// The file, created and locked; None when another run holds it.
    fn open(&self) -> std::io::Result<Option<File>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
        // A new name lives in its directory: flush that too. Directories cannot be opened like
        // this on Windows, where the journal covers the name.
        if let Some(dir) = self.path.parent()
            && let Ok(d) = File::open(dir)
        {
            let _ = d.sync_all();
        }
        Ok(Some(file))
    }

    fn warn(&self, e: &std::io::Error) {
        tracing::warn!(
            "could not write {}: {e}. The benchmark goes on, but a hard hang will not leave the \
             last device on record; check that the data folder is writable",
            self.path.display()
        );
    }
}

fn write_crumb(file: &mut File, crumb: &Breadcrumb) -> std::io::Result<()> {
    let text = serde_json::to_vec_pretty(crumb)?;
    file.set_len(0)?;
    file.rewind()?;
    file.write_all(&text)?;
    file.sync_all()
}

impl Drop for Trail {
    fn drop(&mut self) {
        let state = std::mem::replace(&mut *self.state.lock_recover(), TrailState::Off);
        if let TrailState::Open(file) = state {
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
pub(super) fn left_behind(data_root: &Path) -> Option<Breadcrumb> {
    let path = path_in(data_root);
    let mut file = File::open(&path).ok()?;
    file.try_lock().ok()?;
    let mut text = String::new();
    let read = file.read_to_string(&mut text);
    match read.ok().and_then(|_| serde_json::from_str(&text).ok()) {
        Some(crumb) => Some(crumb),
        None => {
            drop(file);
            let _ = std::fs::remove_file(&path);
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
        drop(file);
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!("could not remove {}: {e}", path.display());
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
        let crumb: Breadcrumb =
            serde_json::from_slice(&std::fs::read(path_in(&root)).unwrap()).unwrap();
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

        let TrailState::Open(file) =
            std::mem::replace(&mut *trail.state.lock_recover(), TrailState::Off)
        else {
            panic!("the trail holds the file");
        };
        drop(file);
        drop(trail);
        let crumb = left_behind(&root).expect("left behind");
        assert_eq!(crumb.device, "ort:directml:1");
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

    #[test]
    fn a_second_run_stays_off_the_file_of_the_first() {
        let root = tmp();
        let first = Trail::new(&root);
        first.note("ort:cpu", "a");
        let second = Trail::new(&root);
        second.note("openvino:cpu", "b");
        let crumb: Breadcrumb =
            serde_json::from_slice(&std::fs::read(path_in(&root)).unwrap()).unwrap();
        assert_eq!(crumb.model, "a");
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
}
