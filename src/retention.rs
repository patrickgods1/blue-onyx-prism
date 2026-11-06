//! Disk hygiene: retention of the saved-image folder and a low-free-space warning.
//!
//! [`plan_prune`] is the pure decision (which files go), [`prune_saved_images`] applies it to
//! one folder, [`prune_dataset`] to the dataset collector's folder tree ([`crate::collect`]),
//! and [`spawn_sweeper`] runs them periodically on a background thread, reading the current
//! settings each tick so a config change applies without a restart. The dataset is pruned on
//! the collector's own thread (its only writer): the sweeper asks it to.

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
    /// The dataset's size after this pass ([`prune_dataset`] only).
    pub usage: DatasetUsage,
}

/// How the dataset folder's frames (image + sidecar, label files not counted) divide up. The size
/// cap applies to `prunable_bytes`, the one figure the collector compares with it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct DatasetUsage {
    /// Frames retention may delete.
    pub prunable_bytes: u64,
    /// Reviewed frames and frames with a human decision
    /// ([`LabelFile::has_human_decision`](crate::collect::labels::LabelFile::has_human_decision)):
    /// never deleted, outside the cap.
    pub protected_bytes: u64,
    pub protected_frames: u64,
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

/// The frames of the dataset folder `dir` retention may delete, as (path, modified, bytes), and
/// how the folder divides up. With `clean`, temporary files of an interrupted write older than an
/// hour are removed on the way.
fn walk_dataset(
    dir: &Path,
    now: SystemTime,
    clean: bool,
) -> (Vec<(PathBuf, SystemTime, u64)>, DatasetUsage) {
    use crate::collect::{labels, layout};
    let mut entries = Vec::new();
    let mut usage = DatasetUsage::default();
    for day in layout::day_dirs(dir) {
        let Ok(rd) = std::fs::read_dir(&day) else {
            continue;
        };
        for ent in rd.flatten() {
            let Ok(ft) = ent.file_type() else { continue };
            if !ft.is_file() {
                continue;
            }
            let path = ent.path();
            let name = ent.file_name().to_string_lossy().into_owned();
            let Ok(meta) = ent.metadata() else { continue };
            let modified = meta.modified().unwrap_or(now);
            if name.starts_with('.') && name.ends_with(".tmp") {
                let stale = now
                    .duration_since(modified)
                    .is_ok_and(|d| d > Duration::from_secs(3600));
                if stale && clean {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
            // A run that was labelling a frame as it went leaves a label file behind.
            if let Some(stem) = name.strip_suffix(labels::LABELS_SUFFIX) {
                if clean && !day.join(format!("{stem}.jpg")).exists() {
                    labels::unless_protected(&path, || std::fs::remove_file(&path));
                }
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jpg") {
                continue;
            }
            let side_len = std::fs::symlink_metadata(path.with_extension("json"))
                .ok()
                .filter(|m| m.file_type().is_file())
                .map_or(0, |m| m.len());
            let size = meta.len().saturating_add(side_len);
            if labels::is_protected(&labels::path_of_frame(&path)) {
                usage.protected_bytes = usage.protected_bytes.saturating_add(size);
                usage.protected_frames += 1;
            } else {
                usage.prunable_bytes = usage.prunable_bytes.saturating_add(size);
                entries.push((path, modified, size));
            }
        }
    }
    (entries, usage)
}

/// How the frames of the dataset folder `dir` divide up (reads every label file; call it off the
/// async runtime). A missing folder is empty.
pub fn dataset_usage(dir: &Path) -> DatasetUsage {
    walk_dataset(dir, SystemTime::now(), false).1
}

/// Delete the oldest frames of the dataset folder `dir` (`<scene>/<camera>/<date>/<stem>.jpg`
/// with its `.json` sidecar and `.labels.json`, see `docs/DATASET.md`) until none is older than
/// `max_age_days` and the deletable frames total at most `max_bytes` (0 disables a limit), then
/// drop the deleted frames from their day's `index.jsonl` and remove emptied folders. Reviewed
/// frames and frames with a human decision are never deleted and do not count toward the cap; a
/// label file with nothing human on it (an auto-label run writes one even when it finds nothing)
/// goes with its frame. Only those names are touched (no symlinks, nothing outside the layout,
/// never `review-queue.jsonl`); temporary files of an interrupted write older than an hour go
/// too. A missing folder is not an error. Frames are written only by the collector thread, where
/// this runs; label files can change at any time, so each frame's is read again, under the label
/// write lock, right before the frame goes.
pub fn prune_dataset(dir: &Path, max_age_days: u32, max_bytes: u64) -> Result<PruneReport> {
    use crate::collect::{labels, layout};
    if !dir.is_dir() {
        return Ok(PruneReport::default());
    }
    let now = SystemTime::now();
    let (entries, mut usage) = walk_dataset(dir, now, true);
    let age =
        (max_age_days > 0).then(|| Duration::from_secs(u64::from(max_age_days) * SECS_PER_DAY));
    let cap = (max_bytes > 0).then_some(max_bytes);
    let doomed = plan_prune(&entries, now, age, cap);
    let mut report = PruneReport {
        remaining: entries.len(),
        ..PruneReport::default()
    };
    let mut touched: Vec<PathBuf> = Vec::new();
    for path in doomed {
        let size = entries.iter().find(|e| e.0 == path).map_or(0, |e| e.2);
        let label_file = labels::path_of_frame(&path);
        let removed = labels::unless_protected(&label_file, || {
            std::fs::remove_file(&path)?;
            for extra in [path.with_extension("json"), label_file.clone()] {
                if let Err(e) = std::fs::remove_file(&extra)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::debug!("retention: removing {}: {e}", extra.display());
                }
            }
            Ok::<(), std::io::Error>(())
        });
        match removed {
            // A reviewer got to it during this pass.
            None => {
                usage.prunable_bytes = usage.prunable_bytes.saturating_sub(size);
                usage.protected_bytes = usage.protected_bytes.saturating_add(size);
                usage.protected_frames += 1;
                report.remaining = report.remaining.saturating_sub(1);
            }
            Some(Ok(())) => {
                usage.prunable_bytes = usage.prunable_bytes.saturating_sub(size);
                report.deleted += 1;
                report.freed_bytes = report.freed_bytes.saturating_add(size);
                report.remaining = report.remaining.saturating_sub(1);
                if let Some(day) = path.parent()
                    && !touched.iter().any(|t| t == day)
                {
                    touched.push(day.to_path_buf());
                }
            }
            Some(Err(e)) => {
                report.failed += 1;
                tracing::debug!("retention: removing {}: {e}", path.display());
            }
        }
    }
    report.usage = usage;
    for day in &touched {
        if let Err(e) = layout::rewrite_index(day) {
            tracing::warn!("dataset retention: {e:#}");
        }
        // Emptied day, camera and scene folders (remove_dir fails on a non-empty one).
        let mut p = Some(day.as_path());
        for _ in 0..3 {
            let Some(d) = p else { break };
            if std::fs::remove_dir(d).is_err() {
                break;
            }
            p = d.parent();
        }
    }
    if report.deleted > 0 || report.failed > 0 {
        tracing::info!(
            "dataset in {}: deleted {} frame(s) ({} MB), {} failed, {} left",
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
    low_space_message(data_root, free_bytes(data_root)?, min_free_mb)
}

/// Bytes available to this process on the file system holding `path`, None when unknown. The path
/// may not exist yet: the nearest existing ancestor is asked.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let probe = path.ancestors().find(|p| p.exists())?;
    fs4::available_space(probe).ok()
}

/// The low-space warning for `free_bytes` under `data_root`, None at or above `min_free_mb`.
pub fn low_space_message(data_root: &Path, free_bytes: u64, min_free_mb: u64) -> Option<String> {
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
    /// The dataset collector's settings (`collect`): its folder is pruned on every tick (by
    /// the collector thread) when it exists, whether collection is on or not.
    pub dataset: Option<crate::collect::CollectSettings>,
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
    if let Some(ds) = &cfg.dataset {
        crate::collect::global().request_prune(ds.clone());
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

    /// A frame (`<stem>.jpg` + `.json` + its index line) in `<dir>/<rel_day>`, `age_days` old.
    fn dataset_frame(dir: &Path, rel_day: &str, stem: &str, size: usize, age_days: u64) {
        let day = dir.join(rel_day);
        std::fs::create_dir_all(&day).unwrap();
        let t = SystemTime::now() - Duration::from_secs(age_days * SECS_PER_DAY);
        for ext in ["jpg", "json"] {
            let p = day.join(format!("{stem}.{ext}"));
            std::fs::write(&p, vec![0u8; size]).unwrap();
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(t)
                .unwrap();
        }
        let line = crate::collect::record::IndexLine {
            id: stem.into(),
            timestamp: "t".into(),
            scene: "ir".into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: vec![crate::collect::Bucket::Empty],
            labels: Vec::new(),
            bytes: 2 * size as u64,
        };
        crate::collect::layout::append_line(
            &day.join(crate::collect::layout::INDEX_FILE),
            &serde_json::to_string(&line).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn dataset_prune_keeps_human_work_and_deletes_machine_labels_with_their_frame() {
        use crate::collect::labels::{self, LabelFile, Status};
        let d = Tmp::new();
        let p = d.path();
        let day = p.join("ir/cam/2026-01-01");
        let put = |stem: &str, f: Option<LabelFile>| {
            dataset_frame(p, "ir/cam/2026-01-01", stem, 100, 200);
            if let Some(f) = f {
                labels::write(&day.join(format!("{stem}.labels.json")), &f).unwrap();
            }
        };
        put(
            "reviewed",
            Some(LabelFile {
                status: Status::Reviewed,
                ..LabelFile::default()
            }),
        );
        put(
            "sent_back",
            Some(LabelFile {
                reviewed_at: Some("2026-01-02T00:00:00.000Z".into()),
                ..LabelFile::default()
            }),
        );
        // An auto-label run that found nothing.
        put("auto_empty", Some(LabelFile::default()));
        put("plain", None);
        dataset_frame(p, "ir/cam/2026-01-01", "garbled", 100, 200);
        std::fs::write(day.join("garbled.labels.json"), b"not json").unwrap();

        let u = dataset_usage(p);
        assert_eq!(
            (u.prunable_bytes, u.protected_bytes, u.protected_frames),
            (400, 600, 3)
        );
        let r = prune_dataset(p, 90, 1).unwrap();
        assert_eq!((r.deleted, r.remaining, r.failed), (2, 0, 0));
        assert_eq!(
            r.usage,
            DatasetUsage {
                prunable_bytes: 0,
                protected_bytes: 600,
                protected_frames: 3
            }
        );
        for f in [
            "reviewed.jpg",
            "reviewed.json",
            "reviewed.labels.json",
            "sent_back.jpg",
            "garbled.jpg",
        ] {
            assert!(day.join(f).exists(), "{f}");
        }
        for f in [
            "auto_empty.jpg",
            "auto_empty.json",
            "auto_empty.labels.json",
            "plain.jpg",
        ] {
            assert!(!day.join(f).exists(), "{f}");
        }
    }

    #[test]
    fn dataset_cap_counts_only_what_it_can_delete() {
        use crate::collect::labels::{self, LabelFile, Status};
        let d = Tmp::new();
        let p = d.path();
        let day = p.join("ir/cam/2026-01-01");
        for (stem, age) in [("a", 3), ("b", 2), ("c", 1)] {
            dataset_frame(p, "ir/cam/2026-01-01", stem, 100, age);
        }
        // Protected "a" is the oldest and bigger than the cap: it takes no room from b and c.
        let reviewed = LabelFile {
            status: Status::Reviewed,
            ..LabelFile::default()
        };
        labels::write(&day.join("a.labels.json"), &reviewed).unwrap();
        let r = prune_dataset(p, 0, 400).unwrap();
        assert_eq!(r.deleted, 0);
        assert_eq!(r.usage.prunable_bytes, 400);
        // A second pass at the same cap has nothing to do: the figure is at the cap, not over it.
        assert_eq!(prune_dataset(p, 0, 400).unwrap().deleted, 0);
    }

    #[test]
    fn dataset_prune_sweeps_up_machine_label_files_left_without_a_frame() {
        use crate::collect::labels::{self, LabelFile, Status};
        let d = Tmp::new();
        let p = d.path();
        dataset_frame(p, "ir/cam/2026-01-01", "kept", 100, 1);
        let day = p.join("ir/cam/2026-01-01");
        labels::write(&day.join("orphan.labels.json"), &LabelFile::default()).unwrap();
        let human = LabelFile {
            status: Status::Reviewed,
            ..LabelFile::default()
        };
        labels::write(&day.join("reviewed.labels.json"), &human).unwrap();
        prune_dataset(p, 0, 0).unwrap();
        assert!(!day.join("orphan.labels.json").exists());
        assert!(day.join("reviewed.labels.json").exists());
    }

    #[test]
    fn dataset_prune_by_age_and_size_keeps_the_index_in_step() {
        use crate::collect::layout::{INDEX_FILE, REVIEW_QUEUE_FILE, read_index};
        let d = Tmp::new();
        let p = d.path();
        dataset_frame(p, "ir/cam/2026-01-01", "old", 100, 200);
        dataset_frame(p, "ir/cam/2026-10-01", "a", 100, 3);
        dataset_frame(p, "ir/cam/2026-10-01", "b", 100, 2);
        dataset_frame(p, "day/cam/2026-10-02", "c", 100, 1);
        std::fs::write(p.join(REVIEW_QUEUE_FILE), b"{}\n").unwrap();
        std::fs::write(p.join("notes.jpg"), b"not in the layout").unwrap();
        std::fs::write(p.join("ir/cam/2026-10-01/keep.txt"), b"x").unwrap();
        // Age: only "old" goes, and its emptied day folder with it.
        let r = prune_dataset(p, 90, 0).unwrap();
        assert_eq!((r.deleted, r.remaining, r.freed_bytes), (1, 3, 200));
        assert!(!p.join("ir/cam/2026-01-01").exists());
        // Size: 3 frames of 200 bytes under a 450-byte cap: the oldest ("a") goes.
        let r = prune_dataset(p, 0, 450).unwrap();
        assert_eq!((r.deleted, r.remaining), (1, 2));
        let idx = read_index(&p.join("ir/cam/2026-10-01").join(INDEX_FILE));
        assert_eq!(idx.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["b"]);
        assert!(!p.join("ir/cam/2026-10-01/a.json").exists());
        assert!(p.join("ir/cam/2026-10-01/b.jpg").exists());
        // Files outside the layout stay.
        assert!(p.join(REVIEW_QUEUE_FILE).exists());
        assert!(p.join("notes.jpg").exists());
        assert!(p.join("ir/cam/2026-10-01/keep.txt").exists());
        assert_eq!(
            prune_dataset(&p.join("missing"), 1, 1).unwrap(),
            PruneReport::default()
        );
    }

    #[cfg(unix)]
    #[test]
    fn dataset_prune_ignores_symlinked_frames() {
        let d = Tmp::new();
        let outside = Tmp::new();
        std::fs::write(outside.path().join("t.jpg"), vec![0u8; 4096]).unwrap();
        let day = d.path().join("ir/cam/2026-10-01");
        std::fs::create_dir_all(&day).unwrap();
        std::os::unix::fs::symlink(outside.path().join("t.jpg"), day.join("l.jpg")).unwrap();
        let r = prune_dataset(d.path(), 0, 1).unwrap();
        assert_eq!(r.deleted, 0);
        assert!(outside.path().join("t.jpg").exists());
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
