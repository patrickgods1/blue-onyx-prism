//! Busy markers: how a sweep tells a job's folder that is in use from one a crash left behind.
//!
//! A job that owns a folder (an auto-label, import or overlap check's `work`, a training run's
//! `.<name>.train-staging`) holds [`Busy`] for as long as it runs: a sibling file `<folder>.busy`,
//! locked (fs4) by the job's process. An orderly end deletes the marker before it lets the lock go;
//! a crash leaves it unlocked. [`probe`] reads that back:
//! - the marker is locked: the job runs (in this process or another), [`State::Live`];
//! - there is no marker: the job ended in order (its folder, when still there, was kept on purpose),
//!   or it is from a version without markers, [`State::Ended`];
//! - the marker is there and unlocked: the job died, [`State::Crashed`], whose [`Claim`] holds the
//!   lock so no job or second sweep can take the folder meanwhile.
//!
//! The marker lives beside the folder, never in it: a folder holding an open, locked file cannot
//! be removed on Windows.

use std::fs::File;
use std::path::{Path, PathBuf};

/// Suffix of a marker file: `<folder name>.busy`.
pub const SUFFIX: &str = ".busy";

/// The marker of `target`: `<target>.busy` beside it.
pub fn marker_of(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(SUFFIX);
    target.with_file_name(name)
}

/// The folder a marker file stands for, when `marker` is one.
pub fn target_of(marker: &Path) -> Option<PathBuf> {
    let name = marker.file_name()?.to_str()?;
    let stem = name.strip_suffix(SUFFIX).filter(|s| !s.is_empty())?;
    Some(marker.with_file_name(stem))
}

/// A job's hold on its folder, released (marker deleted, then unlocked) on drop.
#[derive(Debug)]
pub struct Busy {
    file: Option<File>,
    marker: PathBuf,
    target: PathBuf,
    remove_target: bool,
}

impl Busy {
    /// Hold `target` (which need not exist yet). Never fails: a marker that cannot be written
    /// only costs the sweep's view of this job (a crash then strands the folder as before), so it
    /// is logged and the job goes on.
    pub fn hold(target: &Path) -> Self {
        let marker = marker_of(target);
        let file = match take_marker(&marker) {
            Ok(mut f) => {
                use std::io::Write;
                let _ = f.set_len(0);
                let _ = writeln!(f, "{}", std::process::id());
                Some(f)
            }
            Err(e) => {
                tracing::warn!(
                    "could not hold the busy marker {} ({e}); if this job crashes, its folder \
                     stays until it is removed by hand",
                    marker.display()
                );
                None
            }
        };
        Self {
            file,
            marker,
            target: target.to_path_buf(),
            remove_target: false,
        }
    }

    /// Also remove the target folder on drop (a staging folder: nothing in it outlives its job).
    #[must_use]
    pub fn remove_target_on_drop(mut self) -> Self {
        self.remove_target = true;
        self
    }

    /// The marker file.
    pub fn marker(&self) -> &Path {
        &self.marker
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        if self.remove_target
            && let Err(e) = crate::resources::manager::remove_any(&self.target)
        {
            tracing::warn!("{e:#}; the next storage sweep removes it");
        }
        let Some(f) = self.file.take() else { return };
        // Deleted while still locked: a sweep that opened the marker meanwhile gets the lock only
        // after this, finds the file gone and leaves the folder alone.
        if let Err(e) = std::fs::remove_file(&self.marker)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("removing the busy marker {}: {e}", self.marker.display());
        }
        // Explicitly: a child spawned right now shares the open file until it execs.
        let _ = fs4::FileExt::unlock(&f);
    }
}

/// How long [`Busy::hold`] waits for a marker a sweep holds: a staging folder keeps its name
/// from run to run, so a sweep may be removing the last crash's folder of the same name.
const HOLD_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// Lock `f`, waiting up to `wait` while another holder has it.
fn lock_waiting(f: &File, wait: std::time::Duration) -> std::io::Result<()> {
    let start = std::time::Instant::now();
    loop {
        match fs4::FileExt::try_lock(f) {
            Ok(()) => return Ok(()),
            Err(fs4::TryLockError::WouldBlock) if start.elapsed() < wait => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(fs4::TryLockError::WouldBlock) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    format!("still held after {} s", wait.as_secs()),
                ));
            }
            Err(fs4::TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// Open and lock `marker`. A sweep that held it may delete it before letting go: the lock then
/// sits on a file no longer at that path, so it is opened again.
fn take_marker(marker: &Path) -> std::io::Result<File> {
    for _ in 0..3 {
        let f = open_marker(marker)?;
        lock_waiting(&f, HOLD_WAIT)?;
        if std::fs::symlink_metadata(marker).is_ok() {
            return Ok(f);
        }
        let _ = fs4::FileExt::unlock(&f);
    }
    Err(std::io::Error::other(
        "it was deleted under this job three times",
    ))
}

fn open_marker(marker: &Path) -> std::io::Result<File> {
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(marker)
}

/// What [`probe`] found.
#[derive(Debug)]
pub enum State {
    /// A running job holds the marker (or it cannot be looked at: treated the same).
    Live,
    /// No marker: nothing runs, nothing crashed.
    Ended,
    /// The job died; the claim holds its marker's lock.
    Crashed(Claim),
}

/// The lock of a crashed job's marker. [`Claim::finish`] deletes the marker once its folder is
/// dealt with; dropping the claim instead keeps it, so the next sweep tries again.
#[derive(Debug)]
pub struct Claim {
    file: File,
    marker: PathBuf,
}

impl Claim {
    /// The folder is gone or dealt with: delete the marker, then let the lock go.
    pub fn finish(self) {
        if let Err(e) = std::fs::remove_file(&self.marker)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("removing the busy marker {}: {e}", self.marker.display());
        }
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        let _ = fs4::FileExt::unlock(&self.file);
    }
}

/// Whether the job of `target` runs, ended or crashed (see the module docs).
pub fn probe(target: &Path) -> State {
    let marker = marker_of(target);
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&marker)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return State::Ended,
        // Another account's marker, a delete in progress (Windows): not this sweep's.
        Err(_) => return State::Live,
    };
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => {}
        Err(_) => return State::Live,
    }
    // The job may have ended between the open and the lock: it deletes the marker first.
    if std::fs::symlink_metadata(&marker).is_err() {
        let _ = fs4::FileExt::unlock(&file);
        return State::Ended;
    }
    State::Crashed(Claim { file, marker })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-busy-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn marker_names_round_trip() {
        let t = Path::new("/x/autolabel/20261107-1");
        assert_eq!(marker_of(t), Path::new("/x/autolabel/20261107-1.busy"));
        assert_eq!(target_of(&marker_of(t)).unwrap(), t);
        assert_eq!(target_of(Path::new("/x/a.json")), None);
        assert_eq!(target_of(Path::new("/x/.busy")), None);
    }

    #[test]
    fn a_held_marker_is_live_an_ended_one_is_gone_a_crashed_one_is_claimed() {
        let d = tmp("states");
        let work = d.join("job");
        std::fs::create_dir_all(&work).unwrap();
        assert!(matches!(probe(&work), State::Ended));
        let held = Busy::hold(&work);
        assert!(held.marker().is_file());
        // Another open file of the same marker (what another process does) sees it held.
        assert!(matches!(probe(&work), State::Live));
        drop(held);
        assert!(!marker_of(&work).exists());
        assert!(work.is_dir(), "an orderly end keeps what the job kept");
        assert!(matches!(probe(&work), State::Ended));

        // A crash: the marker stays, unlocked.
        std::fs::write(marker_of(&work), b"123\n").unwrap();
        let State::Crashed(claim) = probe(&work) else {
            panic!("a marker nobody holds is a crash")
        };
        // While claimed, nobody else gets it.
        assert!(matches!(probe(&work), State::Live));
        drop(claim);
        assert!(
            marker_of(&work).exists(),
            "an unfinished claim keeps the marker"
        );
        let State::Crashed(claim) = probe(&work) else {
            panic!("still crashed")
        };
        claim.finish();
        assert!(!marker_of(&work).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_staging_hold_removes_its_folder() {
        let d = tmp("staging");
        let staging = d.join(".m.train-staging");
        let held = Busy::hold(&staging).remove_target_on_drop();
        std::fs::create_dir_all(staging.join("sub")).unwrap();
        std::fs::write(staging.join("sub/x.onnx"), b"x").unwrap();
        drop(held);
        assert!(!staging.exists());
        assert!(!marker_of(&staging).exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
