//! Disk use and the cleanup of files that can be rebuilt or downloaded again.
//!
//! Automatic deletion here only ever touches rebuildable files: model caches, the staging folders
//! of training runs, the work folders of jobs that crashed (an auto-label's finished proposals are
//! first written to their frames, as a Cancel does) and duplicate or dangling review-queue lines.
//! Never labels, datasets, exports, models, training runs or their checkpoints (a failed run's
//! `work` is its retry), configs, or logs.
//!
//! - [`busy`]: the markers that tell a running job's folder from a crashed one's.
//! - [`caches`]: the last-use marker of a cache folder ([`touch`]) and the cache sweeps.
//! - [`sweep`]: the sweep at startup and once a day (run by [`crate::retention`]'s thread), and
//!   the Storage card's Clear actions.
//! - [`usage`]: the Storage card's sizes ("safe to clear" and "your data"), measured off the
//!   request thread and kept a short while.

pub mod busy;
pub mod caches;
pub mod sweep;
pub mod usage;

pub use busy::Busy;
pub use caches::touch;
pub use sweep::Settings;

use serde::Serialize;
use std::path::{Path, PathBuf};

/// One thing a sweep or a Clear deleted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Removed {
    pub path: PathBuf,
    /// What deleting it freed ([`crate::training::retry::disk_use`]; a file's length).
    pub bytes: u64,
    pub why: String,
}

/// Bytes deleting `p` frees: [`crate::training::retry::disk_use`] of a folder, a file's length.
pub fn size_of(p: &Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => crate::training::retry::disk_use(p),
        Ok(m) if m.is_file() => m.len(),
        _ => 0,
    }
}

/// Delete `p` (file or folder), log it with its size and `why`, and record it in `out`. False
/// (logged) when it could not be deleted.
pub(crate) fn remove_counted(p: &Path, why: &str, out: &mut Vec<Removed>) -> bool {
    let bytes = size_of(p);
    match crate::resources::manager::remove_any(p) {
        Ok(()) => {
            tracing::info!(
                "storage: removed {} ({}): {why}",
                p.display(),
                crate::resources::catalog::format_size(bytes)
            );
            out.push(Removed {
                path: p.to_path_buf(),
                bytes,
                why: why.to_string(),
            });
            true
        }
        Err(e) => {
            tracing::warn!(
                "storage: could not remove {} ({why}): {e:#}; it is tried again at the next \
                 sweep, or delete it by hand",
                p.display()
            );
            false
        }
    }
}

/// Sum of [`Removed::bytes`].
pub fn total(removed: &[Removed]) -> u64 {
    removed.iter().map(|r| r.bytes).sum()
}
