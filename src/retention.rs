//! Disk hygiene: retention of the saved-image folder and a low-free-space warning.
//!
//! [`plan_prune`] is the pure decision (which files go), [`prune_saved_images`] applies it to
//! one folder, and [`spawn_sweeper`] runs both periodically on a background thread, reading the
//! current settings each tick so a config change applies without a restart.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio_util::sync::CancellationToken;

/// Free space under the data root below which the sweeper warns, in MB.
pub const MIN_FREE_MB: u64 = 1024;
/// Time between sweeps.
pub const SWEEP_INTERVAL: Duration = Duration::from_secs(10 * 60);
/// Delay before the first sweep after startup.
pub const FIRST_SWEEP_DELAY: Duration = Duration::from_secs(30);
/// A persisting low-space condition is logged at most this often.
pub const FREE_SPACE_WARN_INTERVAL: Duration = Duration::from_secs(60 * 60);
/// Only files with these extensions (lowercase) are ever deleted by the pruner.
pub const PRUNABLE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

const BYTES_PER_MB: u64 = 1024 * 1024;
const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// Which of `entries` (path, modified time, size) to delete, oldest first: first everything
/// older than `max_age`, then the oldest of the rest until the total is at most `max_bytes`.
/// `None` disables the respective limit.
pub fn plan_prune(
    entries: &[(PathBuf, SystemTime, u64)],
    now: SystemTime,
    max_age: Option<Duration>,
    max_bytes: Option<u64>,
) -> Vec<PathBuf> {
    let mut sorted: Vec<&(PathBuf, SystemTime, u64)> = entries.iter().collect();
    sorted.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = Vec::new();
    let mut kept: Vec<&(PathBuf, SystemTime, u64)> = Vec::new();
    for e in sorted {
        let too_old =
            max_age.is_some_and(|age| now.duration_since(e.1).is_ok_and(|elapsed| elapsed > age));
        if too_old {
            out.push(e.0.clone());
        } else {
            kept.push(e);
        }
    }
    if let Some(cap) = max_bytes {
        let mut total: u64 = kept.iter().map(|e| e.2).fold(0, u64::saturating_add);
        for e in kept {
            if total <= cap {
                break;
            }
            total = total.saturating_sub(e.2);
            out.push(e.0.clone());
        }
    }
    out
}

/// What a prune pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneReport {
    /// Files deleted.
    pub deleted: usize,
    /// Bytes freed by the deleted files.
    pub freed_bytes: u64,
    /// Files that should have gone but could not be deleted.
    pub failed: usize,
    /// Prunable files left in the folder.
    pub remaining: usize,
}

fn is_prunable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| PRUNABLE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Delete the oldest jpg/jpeg/png files directly in `dir` (no recursion, symlinks and other
/// file types are never touched) until none is older than `max_age_days` and the total is at
/// most `max_mb`. 0 disables a limit. A missing folder is not an error.
pub fn prune_saved_images(dir: &Path, max_age_days: u32, max_mb: u64) -> Result<PruneReport> {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(PruneReport::default()),
        Err(e) => return Err(e).with_context(|| format!("listing {}", dir.display())),
    };
    let mut entries = Vec::new();
    for ent in rd.flatten() {
        // `DirEntry::file_type` / `metadata` do not follow symlinks.
        let Ok(ft) = ent.file_type() else { continue };
        if !ft.is_file() || !is_prunable(&ent.path()) {
            continue;
        }
        let Ok(meta) = ent.metadata() else { continue };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        entries.push((ent.path(), modified, meta.len()));
    }
    let age =
        (max_age_days > 0).then(|| Duration::from_secs(u64::from(max_age_days) * SECS_PER_DAY));
    let cap = (max_mb > 0).then(|| max_mb.saturating_mul(BYTES_PER_MB));
    let doomed = plan_prune(&entries, SystemTime::now(), age, cap);
    let mut report = PruneReport {
        remaining: entries.len(),
        ..PruneReport::default()
    };
    for path in doomed {
        let size = entries.iter().find(|e| e.0 == path).map_or(0, |e| e.2);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                report.deleted += 1;
                report.freed_bytes = report.freed_bytes.saturating_add(size);
                report.remaining = report.remaining.saturating_sub(1);
            }
            Err(e) => {
                report.failed += 1;
                tracing::debug!("retention: removing {}: {e}", path.display());
            }
        }
    }
    if report.deleted > 0 || report.failed > 0 {
        tracing::info!(
            "saved images in {}: deleted {} file(s) ({} MB), {} failed, {} left",
            dir.display(),
            report.deleted,
            report.freed_bytes / BYTES_PER_MB,
            report.failed,
            report.remaining
        );
    }
    Ok(report)
}

/// A warning text when the file system holding `data_root` has less than `min_free_mb` MB
/// available. None when there is enough, or when the space cannot be determined.
pub fn free_space_warning(data_root: &Path, min_free_mb: u64) -> Option<String> {
    // The root may not exist yet; ask about the nearest existing ancestor.
    let probe = data_root.ancestors().find(|p| p.exists())?;
    let free = fs4::available_space(probe).ok()?;
    low_space_message(data_root, free, min_free_mb)
}

fn low_space_message(data_root: &Path, free_bytes: u64, min_free_mb: u64) -> Option<String> {
    (free_bytes < min_free_mb.saturating_mul(BYTES_PER_MB)).then(|| {
        format!(
            "low disk space under {}: {} MB free (threshold {} MB); saved images, logs, caches and downloads may fail",
            data_root.display(),
            free_bytes / BYTES_PER_MB,
            min_free_mb
        )
    })
}

/// What the sweeper does on one tick, resolved from the current config.
#[derive(Debug, Clone)]
pub struct SweeperConfig {
    /// Saved-image folder (`save_image_path`); None = no pruning.
    pub save_image_dir: Option<PathBuf>,
    /// `save_image_max_age_days` (0 = unlimited).
    pub max_age_days: u32,
    /// `save_image_max_mb` (0 = unlimited).
    pub max_mb: u64,
    /// Directory whose free space is watched.
    pub data_root: PathBuf,
    /// Warn below this many MB free (normally [`MIN_FREE_MB`]).
    pub min_free_mb: u64,
}

/// True when a low-space warning may be logged now (never yet, or the last is old enough).
fn may_warn(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|t| now.saturating_duration_since(t) >= FREE_SPACE_WARN_INTERVAL)
}

/// One sweep: prune the saved images, then check free space (`last_warn` throttles the log).
fn sweep(cfg: &SweeperConfig, last_warn: &mut Option<Instant>) {
    if let Some(dir) = &cfg.save_image_dir
        && let Err(e) = prune_saved_images(dir, cfg.max_age_days, cfg.max_mb)
    {
        tracing::warn!("saved-image retention: {e:#}");
    }
    match free_space_warning(&cfg.data_root, cfg.min_free_mb) {
        Some(msg) => {
            let now = Instant::now();
            if may_warn(*last_warn, now) {
                tracing::warn!("{msg}");
                *last_warn = Some(now);
            }
        }
        None => *last_warn = None,
    }
}

/// Sleep `total` in short steps; false when `shutdown` was triggered meanwhile.
fn nap(total: Duration, shutdown: &CancellationToken) -> bool {
    let step = Duration::from_millis(500);
    let end = Instant::now() + total;
    while Instant::now() < end {
        if shutdown.is_cancelled() {
            return false;
        }
        std::thread::sleep(step.min(end.saturating_duration_since(Instant::now())));
    }
    !shutdown.is_cancelled()
}

/// Run [`prune_saved_images`] and the free-space check every [`SWEEP_INTERVAL`] (first after
/// [`FIRST_SWEEP_DELAY`]) until `shutdown`. `settings` is called on every tick, so config
/// changes apply at once.
#[allow(
    clippy::expect_used,
    reason = "the OS can create one more thread; without it the sweeper cannot exist, and it \
              is a best-effort helper, so a failure to spawn aborts startup loudly"
)]
pub fn spawn_sweeper(
    settings: Arc<dyn Fn() -> SweeperConfig + Send + Sync>,
    shutdown: CancellationToken,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("retention".into())
        .spawn(move || {
            let mut last_warn = None;
            let mut delay = FIRST_SWEEP_DELAY;
            while nap(delay, &shutdown) {
                sweep(&settings(), &mut last_warn);
                delay = SWEEP_INTERVAL;
            }
        })
        .expect("spawning the retention thread (OS thread limit or out of memory)")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// A fresh temp directory removed on drop.
    struct Tmp(PathBuf);
    impl Tmp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("bop-ret-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }
    fn e(name: &str, t: u64, size: u64) -> (PathBuf, SystemTime, u64) {
        (PathBuf::from(name), at(t), size)
    }

    #[test]
    fn constants() {
        assert_eq!(MIN_FREE_MB, 1024);
        assert_eq!(SWEEP_INTERVAL, Duration::from_secs(600));
        assert_eq!(FIRST_SWEEP_DELAY, Duration::from_secs(30));
        assert_eq!(FREE_SPACE_WARN_INTERVAL, Duration::from_secs(3600));
    }

    #[test]
    fn unlimited_deletes_nothing() {
        let v = [e("a", 1, 10), e("b", 2, 10)];
        assert!(plan_prune(&v, at(1_000_000), None, None).is_empty());
    }

    #[test]
    fn deletes_by_age_oldest_first() {
        let v = [e("new", 900, 1), e("old2", 20, 1), e("old1", 10, 1)];
        let out = plan_prune(&v, at(1000), Some(Duration::from_secs(500)), None);
        assert_eq!(out, vec![PathBuf::from("old1"), PathBuf::from("old2")]);
    }

    #[test]
    fn deletes_by_size_until_under_cap() {
        let v = [e("c", 3, 40), e("a", 1, 40), e("b", 2, 40)];
        let out = plan_prune(&v, at(10), None, Some(80));
        assert_eq!(out, vec![PathBuf::from("a")]);
        let out = plan_prune(&v, at(10), None, Some(30));
        assert_eq!(
            out,
            vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")]
        );
        assert!(plan_prune(&v, at(10), None, Some(120)).is_empty());
    }

    #[test]
    fn age_then_size_and_future_mtime_kept() {
        let v = [
            e("old", 1, 100),
            e("a", 900, 50),
            e("b", 950, 50),
            e("future", 5000, 50),
        ];
        let out = plan_prune(&v, at(1000), Some(Duration::from_secs(500)), Some(100));
        // "old" by age; then 150 bytes remain, the cap 100 drops the oldest kept ("a").
        assert_eq!(out, vec![PathBuf::from("old"), PathBuf::from("a")]);
    }

    #[test]
    fn prune_only_touches_images_directly_in_dir() {
        let d = Tmp::new();
        let p = d.path();
        for n in ["a.jpg", "b.JPEG", "c.png", "keep.txt", "keep.json"] {
            std::fs::write(p.join(n), vec![0u8; 1024 * 1024]).unwrap();
        }
        std::fs::create_dir(p.join("sub")).unwrap();
        std::fs::write(p.join("sub/x.jpg"), b"x").unwrap();
        // Everything is brand new: the age limit deletes nothing.
        let r = prune_saved_images(p, 1, 0).unwrap();
        assert_eq!(r.deleted, 0);
        assert_eq!(r.remaining, 3);
        // Size cap of 1 MB leaves the newest image; text files and the subfolder stay.
        let r = prune_saved_images(p, 0, 1).unwrap();
        assert_eq!(r.deleted, 2);
        assert_eq!(r.freed_bytes, 2 * 1024 * 1024);
        assert_eq!(r.remaining, 1);
        assert!(p.join("keep.txt").exists() && p.join("keep.json").exists());
        assert!(p.join("sub/x.jpg").exists());
        let jpgs = std::fs::read_dir(p)
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_file() && is_prunable(&e.path()))
            .count();
        assert_eq!(jpgs, 1);
    }

    #[cfg(unix)]
    #[test]
    fn prune_ignores_symlinks() {
        let d = Tmp::new();
        let outside = Tmp::new();
        std::fs::write(outside.path().join("t.jpg"), vec![0u8; 2 * 1024 * 1024]).unwrap();
        std::os::unix::fs::symlink(outside.path().join("t.jpg"), d.path().join("l.jpg")).unwrap();
        let r = prune_saved_images(d.path(), 0, 1).unwrap();
        assert_eq!(r.deleted, 0);
        assert!(outside.path().join("t.jpg").exists());
        assert!(d.path().join("l.jpg").symlink_metadata().is_ok());
    }

    #[test]
    fn missing_dir_is_fine() {
        let d = Tmp::new();
        let r = prune_saved_images(&d.path().join("nope"), 1, 1).unwrap();
        assert_eq!(r, PruneReport::default());
    }

    #[test]
    fn low_space_message_and_throttle() {
        let root = Path::new("/data");
        assert!(low_space_message(root, 10 * BYTES_PER_MB, 1024).is_some());
        assert!(low_space_message(root, 2048 * BYTES_PER_MB, 1024).is_none());
        assert!(low_space_message(root, 0, 0).is_none());
        let now = Instant::now();
        assert!(may_warn(None, now));
        assert!(!may_warn(Some(now), now));
        if let Some(later) = now.checked_add(FREE_SPACE_WARN_INTERVAL) {
            assert!(may_warn(Some(now), later));
        }
    }

    #[test]
    fn free_space_of_a_real_dir_is_known() {
        let d = Tmp::new();
        // Absurd threshold: always warns; zero threshold: never.
        assert!(free_space_warning(d.path(), u64::MAX / BYTES_PER_MB).is_some());
        assert!(free_space_warning(d.path(), 0).is_none());
        assert!(free_space_warning(&d.path().join("not/yet"), 0).is_none());
    }
}
