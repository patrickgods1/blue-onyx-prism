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
//! A job that meant to remove its folder and could not ([`Busy::finish_removing`], the drop of a
//! [`Busy::remove_target_on_drop`] hold: a file still open, an antivirus scan on Windows) keeps
//! the marker too, unlocked, so the next sweep sees a crashed job and removes the folder; a deleted
//! marker would read as a folder kept on purpose, never swept.
//!
//! The marker lives beside the folder, never in it: a folder holding an open, locked file cannot
//! be removed on Windows.
//!
//! A job that panics is no orderly end: a [`Busy`] dropped while its thread unwinds keeps the
//! marker (and the folder, even with [`Busy::remove_target_on_drop`]) and only lets the lock go,
//! like [`Busy::abandon`], so the next sweep finds a crashed job and recovers or removes its folder
//! (an auto-label's finished proposals are written then, [`crate::storage::sweep`]).

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

    /// Let go of the folder but keep its marker: the job ended without removing a folder it
    /// meant to (a file still in use), so the next sweep sees a crashed job's folder and removes
    /// it ([`probe`] then answers [`State::Crashed`]).
    pub fn abandon(mut self) {
        self.remove_target = false;
        if let Some(f) = self.file.take() {
            let _ = fs4::FileExt::unlock(&f);
        }
    }

    /// The job is done with its folder: remove it, then release the hold. Whether the folder is
    /// gone. When the removal fails (a file still open: an antivirus scan, a child's handles not
    /// closed yet on Windows), the hold is [abandoned](Busy::abandon) instead, so the next storage
    /// sweep finds a crashed job's folder and removes it, rather than a folder kept on purpose that
    /// stays for good.
    pub fn finish_removing(self) -> bool {
        match crate::resources::manager::remove_any(&self.target) {
            Ok(()) => {
                drop(self);
                true
            }
            Err(e) => {
                tracing::warn!(
                    "{e:#}; a file in it is probably still in use; the next storage sweep \
                     removes it"
                );
                self.abandon();
                false
            }
        }
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        // Unwinding out of the job: it did not end in order, so the folder reads as crashed.
        if std::thread::panicking() {
            self.remove_target = false;
            if let Some(f) = self.file.take() {
                let _ = fs4::FileExt::unlock(&f);
            }
            return;
        }
        if self.remove_target
            && let Err(e) = crate::resources::manager::remove_any(&self.target)
        {
            tracing::warn!("{e:#}; the next storage sweep removes it");
            // Kept, unlocked: the sweep sees a crash and removes the folder (a deleted marker
            // would read as a folder kept on purpose).
            if let Some(f) = self.file.take() {
                let _ = fs4::FileExt::unlock(&f);
            }
            return;
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

/// Lock `f`, waiting up to `wait` while another holder has it ([`crate::fsx::lock_wait_with`],
/// every 20 ms); a holder that keeps it past `wait` is a `WouldBlock` error.
fn lock_waiting(f: &File, wait: std::time::Duration) -> std::io::Result<()> {
    let waited = crate::fsx::lock_wait_with(
        f,
        crate::fsx::LockKind::Exclusive,
        crate::fsx::deadline_after(wait),
        crate::fsx::Pauses::fixed(std::time::Duration::from_millis(20)),
        || true,
    )?;
    match waited {
        crate::fsx::Waited::Held => Ok(()),
        crate::fsx::Waited::TimedOut | crate::fsx::Waited::Stopped => Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!("still held after {} s", wait.as_secs()),
        )),
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

    /// A held lock is waited for, then a `WouldBlock` error naming the wait; a free one at once.
    #[test]
    fn lock_waiting_waits_then_gives_up() {
        let d = tmp("wait");
        let open = || open_marker(&d.join("m.busy")).unwrap();
        let held = open();
        fs4::FileExt::try_lock(&held).unwrap();
        let f = open();
        let started = std::time::Instant::now();
        let e = lock_waiting(&f, std::time::Duration::from_millis(60)).unwrap_err();
        assert!(started.elapsed() >= std::time::Duration::from_millis(60));
        assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock);
        assert!(e.to_string().starts_with("still held after"), "{e}");
        fs4::FileExt::unlock(&held).unwrap();
        lock_waiting(&f, std::time::Duration::ZERO).unwrap();
        std::fs::remove_dir_all(&d).ok();
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

    /// A job that panics keeps its marker (unlocked) and its folder, even a staging one: the
    /// sweep then sees a crash, never an orderly end that kept the folder on purpose.
    #[test]
    fn a_hold_dropped_by_a_panic_reads_as_crashed() {
        let d = tmp("panic");
        let work = d.join("job");
        let staging = d.join(".m.train-staging");
        let (w, s) = (work.clone(), staging.clone());
        let r = std::panic::catch_unwind(move || {
            let _busy = Busy::hold(&w);
            let _staging = Busy::hold(&s).remove_target_on_drop();
            std::fs::create_dir_all(w.join("sub")).unwrap();
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(w.join("sub/proposals.jsonl"), b"{}\n").unwrap();
            panic!("a bug in the job");
        });
        assert!(r.is_err());
        assert!(work.join("sub/proposals.jsonl").is_file(), "the work stays");
        assert!(staging.is_dir(), "a staging folder stays for the sweep");
        for t in [&work, &staging] {
            assert!(marker_of(t).is_file(), "the marker stays");
            let State::Crashed(claim) = probe(t) else {
                panic!("a panicked job's folder reads as crashed")
            };
            claim.finish();
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// `sub` made unwritable, so removing the folder above it fails; None when it still can be
    /// written (root ignores the mode): the test then has nothing to check.
    #[cfg(unix)]
    fn lock_down(sub: &Path) -> Option<()> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(sub).unwrap();
        std::fs::write(sub.join("f_012345.jpg"), b"x").unwrap();
        std::fs::set_permissions(sub, std::fs::Permissions::from_mode(0o555)).unwrap();
        if std::fs::write(sub.join("probe"), b"x").is_ok() {
            eprintln!("skipped: permissions are not enforced for this user");
            return None;
        }
        Some(())
    }

    #[cfg(unix)]
    fn unlock_down(sub: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(sub, std::fs::Permissions::from_mode(0o755));
    }

    /// A job done with its folder removes it and its marker; one whose removal fails keeps the
    /// marker, unlocked, so the next sweep sees a crashed job and removes the folder.
    #[test]
    fn finish_removing_removes_the_folder_or_leaves_it_to_the_sweep() {
        let d = tmp("finish");
        let work = d.join("job");
        let busy = Busy::hold(&work);
        std::fs::create_dir_all(work.join("sub")).unwrap();
        std::fs::write(work.join("sub/proposals.jsonl"), b"{}\n").unwrap();
        assert!(busy.finish_removing());
        assert!(!work.exists());
        assert!(!marker_of(&work).exists());
        assert!(matches!(probe(&work), State::Ended));

        #[cfg(unix)]
        {
            let sub = work.join("candidates");
            let busy = Busy::hold(&work);
            if lock_down(&sub).is_some() {
                assert!(!busy.finish_removing(), "the removal failed");
                assert!(work.is_dir());
                assert!(marker_of(&work).is_file(), "the marker stays");
                let State::Crashed(claim) = probe(&work) else {
                    panic!("a folder its job could not remove reads as crashed")
                };
                claim.finish();
            }
            unlock_down(&sub);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A staging folder that cannot be removed when its hold drops keeps its marker, unlocked.
    #[cfg(unix)]
    #[test]
    fn a_staging_folder_that_cannot_be_removed_reads_as_crashed() {
        let d = tmp("staging-stuck");
        let staging = d.join(".m.train-staging");
        let sub = staging.join("sub");
        let held = Busy::hold(&staging).remove_target_on_drop();
        if lock_down(&sub).is_some() {
            drop(held);
            assert!(staging.is_dir());
            let State::Crashed(claim) = probe(&staging) else {
                panic!("a staging folder its job could not remove reads as crashed")
            };
            claim.finish();
        }
        unlock_down(&sub);
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
