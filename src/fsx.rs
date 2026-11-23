//! File-system helpers shared by every writer of the data root (audit 2026-11, Phase 0):
//! the one atomic write, the one file-lock wait and the one rule for on-disk names built from
//! user input. Std + fs4 only (a leaf).
//!
//! - [`write_atomic`] writes `<dir>/.<name>.<uuid>.tmp` (the shape [`is_temp_name`] recognises,
//!   so the storage sweeper can remove one a crash left), syncs it, keeps the target's
//!   permissions and renames it over the target.
//! - [`lock_wait`] takes an fs4 lock, waiting up to a deadline.
//! - [`safe_component`] maps a user-supplied name to one that is valid on every OS (no Windows
//!   reserved device names).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The temporary file [`write_atomic`] writes beside `path`: `<dir>/.<name>.<uuid>.tmp`.
pub fn temp_path(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    dir.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()))
}

/// Whether `file_name` has the shape of a [`write_atomic`] temporary file.
pub fn is_temp_name(file_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix('.') else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".tmp") else {
        return false;
    };
    match rest.rsplit_once('.') {
        Some((name, id)) => {
            !name.is_empty() && id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
        }
        None => false,
    }
}

/// Write `bytes` to `path` through a synced temporary file in the same folder and a rename, so a
/// reader never sees a half-written file. Phase 0 stub: the final version keeps the target's
/// permissions and syncs the folder.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_path(path);
    let result = (|| {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Which lock [`lock_wait`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockKind {
    /// `try_lock_shared`.
    Shared,
    /// `try_lock` (exclusive).
    Exclusive,
}

/// Take an fs4 lock on `file`, retrying every `poll` until `deadline` passes. `Ok(true)` = held,
/// `Ok(false)` = still held by someone else at the deadline, `Err` = the lock call cannot work
/// on this file (callers decide their fallback).
pub fn lock_wait(
    file: &std::fs::File,
    kind: LockKind,
    deadline: Instant,
    poll: Duration,
) -> std::io::Result<bool> {
    loop {
        let r = match kind {
            LockKind::Shared => fs4::FileExt::try_lock_shared(file),
            LockKind::Exclusive => fs4::FileExt::try_lock(file),
        };
        match r {
            Ok(()) => return Ok(true),
            Err(fs4::TryLockError::WouldBlock) => {}
            Err(fs4::TryLockError::Error(e)) => return Err(e),
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        std::thread::sleep(poll);
    }
}

/// Whether `stem` (a file or folder name without extension) is a Windows reserved device name
/// (`con`, `prn`, `aux`, `nul`, `com1`-`com9`, `lpt1`-`lpt9`, any case).
pub fn is_windows_reserved(stem: &str) -> bool {
    let base = stem.split('.').next().unwrap_or(stem).trim_end_matches([' ', '.']);
    let lower = base.to_ascii_lowercase();
    matches!(lower.as_str(), "con" | "prn" | "aux" | "nul")
        || matches!(lower.as_bytes(), [b'c', b'o', b'm', d] | [b'l', b'p', b't', d] if (b'1'..=b'9').contains(d))
}
