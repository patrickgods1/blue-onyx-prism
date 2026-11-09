//! `<auth>/.lock`: the one lock of the secrets folder. Every read-modify-write of a file in
//! `auth/` (accounts, the setup code, the admin token's move) holds it, in the service and in the
//! CLI alike, so two writers never lose each other's change. It is held for one read, change and
//! rename, never while a password is hashed (hash first, then lock, read again, apply, write).
//!
//! The pattern of `.labels.lock` (`collect::labels`): a zero-byte file never deleted, an `fs4`
//! exclusive lock polled with a short backoff, the lock released explicitly on drop. Unlike label
//! writes, which go on without a lock that cannot work, an account change fails closed: a lock
//! that cannot be taken within [`LOCK_WAIT`] is an error and nothing is written.
//!
//! The lock is per open file, so it also orders threads of one process (each call opens its own
//! handle) on every OS.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The lock file's name in `auth/`.
pub const LOCK_FILE: &str = ".lock";
/// Longest wait for the lock. A holder keeps it for milliseconds; running out means another
/// process stopped while holding it (a lock dies with its process, so that one is still running).
pub const LOCK_WAIT: Duration = Duration::from_secs(30);
/// Longest pause between tries of a held lock.
const LOCK_POLL: Duration = Duration::from_millis(2);

/// The held lock, released on drop.
#[derive(Debug)]
pub struct AuthLock {
    file: std::fs::File,
    path: PathBuf,
}

impl Drop for AuthLock {
    fn drop(&mut self) {
        // Explicitly: a child spawned right now would share the open file until it execs.
        if let Err(e) = fs4::FileExt::unlock(&self.file) {
            tracing::debug!("releasing {}: {e}", self.path.display());
        }
    }
}

/// Take `<auth>/.lock`, waiting at most [`LOCK_WAIT`]. `auth` must exist
/// ([`crate::private_dir::ensure`]).
pub fn lock(auth: &Path) -> Result<AuthLock> {
    lock_within(auth, LOCK_WAIT)
}

/// [`lock`] with its own longest wait.
pub fn lock_within(auth: &Path, wait: Duration) -> Result<AuthLock> {
    let path = auth.join(LOCK_FILE);
    let file = open(&path).with_context(|| {
        format!(
            "opening the lock {}; check that this runs as the service's account (the folder \
             is readable by it and administrators only)",
            path.display()
        )
    })?;
    let start = Instant::now();
    let mut pause = Duration::from_micros(100);
    loop {
        match fs4::FileExt::try_lock(&file) {
            Ok(()) => return Ok(AuthLock { file, path }),
            Err(fs4::TryLockError::WouldBlock) if start.elapsed() < wait => {
                std::thread::sleep(pause);
                pause = (pause * 2).min(LOCK_POLL);
            }
            Err(fs4::TryLockError::WouldBlock) => bail!(
                "{} stayed locked for {} s, so nothing was changed: another blue-onyx-prism \
                 process (the service or a command) is changing accounts and seems stuck. Try \
                 again; if it persists, restart the service",
                path.display(),
                wait.as_secs()
            ),
            Err(fs4::TryLockError::Error(e)) => bail!(
                "could not lock {} ({e}), so nothing was changed; keep the data folder on a \
                 local disk",
                path.display()
            ),
        }
    }
}

/// Open (creating) the lock file. Another account's file (root made it) locks through a read
/// handle too. Made by root, it is handed to the folder's owner, as every file in `auth/`.
fn open(path: &Path) -> std::io::Result<std::fs::File> {
    let rw = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path);
    let file = match rw {
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return std::fs::File::open(path).map_err(|_| e);
        }
        r => r?,
    };
    #[cfg(unix)]
    hand_over(path);
    Ok(file)
}

/// Run as root, the lock file goes to the owner of its folder (the service's account), which
/// could otherwise only lock it through a read handle. A failure only costs that.
#[cfg(unix)]
fn hand_over(path: &Path) {
    use std::os::unix::fs::{MetadataExt, chown};
    let (Ok(me), Some(Ok(dir))) = (
        std::fs::metadata(path),
        path.parent().map(std::fs::metadata),
    ) else {
        return;
    };
    if let Some((uid, gid)) = crate::resources::hf_token::chown_to(me.uid(), (dir.uid(), dir.gid()))
    {
        let _ = chown(path, Some(uid), Some(gid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_held_lock_makes_the_next_wait_then_fail_closed() {
        let dir = std::env::temp_dir().join(format!("bo_authlock_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let held = lock(&dir).unwrap();
        let started = Instant::now();
        let e = format!(
            "{:#}",
            lock_within(&dir, Duration::from_millis(100)).unwrap_err()
        );
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(e.contains("nothing was changed"), "{e}");
        drop(held);
        let again = lock_within(&dir, Duration::from_millis(100)).unwrap();
        drop(again);
        assert!(dir.join(LOCK_FILE).is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
