//! Disk hygiene: retention of the saved-image folder and a low-free-space warning.
//!
//! [`plan_prune`] is the pure decision (which files go), [`prune_saved_images`] applies it to
//! one folder, [`prune_dataset`] to the dataset collector's folder tree ([`crate::collect`]),
//! and [`spawn_sweeper`] runs them periodically on a background thread, reading the current
//! settings each tick so a config change applies without a restart. The dataset is pruned on
//! the collector's own thread (the collector's writes and the prune never meet; an import is the
//! other writer, see [`prune_dataset`]): the sweeper asks it to. The same thread runs the
//! storage sweep of rebuildable files once a day ([`crate::storage::sweep`]).
//!
//! Crash repair (audit 2026-11 phase 2): the dataset prune's walk also finds what an interrupted
//! write or a killed pass left (stale temporary files, orphan sidecars, day folders whose index
//! names deleted frames, empty day folders) and puts it right ([`prune_dataset`]). Index rewrites
//! and folder removals hold the dataset's cross-process index lock, which the import's appends take
//! as well ([`crate::collect::layout::INDEX_LOCK_FILE`]).
//!
//! Stopping (audit 2026-11 phase 1): the dataset prune checks the collector's stop flag
//! ([`crate::collect::stop_requested`]) between two frames and handles one day at a time, its
//! `index.jsonl` rewritten as soon as that day's frames are gone, so a stop (or a kill) mid-prune
//! leaves no index line naming a deleted frame outside the day in hand. The storage sweep gets the
//! sweeper's cancel token and checks it between items. [`join_within`] is the bounded join of a
//! stopping thread.

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
    /// The pass ended early for a stop ([`prune_dataset_until`]); the rest goes at the next one.
    pub stopped: bool,
    /// Day folders an interrupted prune or write had left out of step, put right
    /// ([`prune_dataset`]: index lines of deleted frames dropped, emptied folders removed).
    pub repaired_days: usize,
    /// Sidecars whose image was gone, removed ([`prune_dataset`]).
    pub orphan_sidecars: usize,
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

/// How old a leftover of an interrupted write must be before a cleaning walk removes it (a
/// temporary file of [`crate::fsx::write_atomic`], an orphan sidecar): older than any write in
/// progress, so a live writer in this or another process never loses one.
pub const LEFTOVER_AGE: Duration = Duration::from_secs(60 * 60);

/// What [`walk_dataset`] found.
#[derive(Debug, Default)]
struct Walked {
    /// The frames retention may delete, as (path, modified, bytes).
    entries: Vec<(PathBuf, SystemTime, u64)>,
    usage: DatasetUsage,
    /// Day folders an interrupted prune or write left out of step (cleaning walks only): an
    /// `index.jsonl` naming a frame whose image is not in the folder, or nothing in the folder at
    /// all. [`prune_dataset`] rewrites their index and removes the emptied folders.
    repair: Vec<PathBuf>,
    /// Orphan sidecars removed (cleaning walks only).
    orphan_sidecars: usize,
}

fn is_stale(now: SystemTime, modified: SystemTime) -> bool {
    now.duration_since(modified).is_ok_and(|d| d > LEFTOVER_AGE)
}

/// The frames of the dataset folder `dir` retention may delete and how the folder divides up.
/// `stop` is asked between two day folders: once it says yes the walk ends there, incomplete
/// (the caller checks `stop` again).
///
/// With `clean` (audit 2026-11 phase 2), what an interrupted write or prune left is dealt with on
/// the way, each kind matched exactly:
/// - [`crate::fsx::write_atomic`]'s temporary files ([`crate::fsx::is_temp_name`]) older than
///   [`LEFTOVER_AGE`] are removed;
/// - a label file whose image is gone, unless it holds human work;
/// - an orphan sidecar: a regular file `<stem>.json` (never `*.labels.json`) whose stem is a frame
///   stem ([`layout::is_frame_stem`]), older than [`LEFTOVER_AGE`], with no `<stem>.jpg` on disk and
///   no protected label file. Images go in before their sidecar, so only a deletion that stopped
///   part way leaves one;
/// - the day folders to repair ([`Walked::repair`]) are listed, by reading their index.
///
/// A frame whose image and sidecar are in but whose index line is not (a kill between the two) is
/// not repaired: the caps still count and delete it, and adding its line would race the writer.
/// An image with no sidecar is left alone (the Review tab's Rescan folder lists it).
fn walk_dataset(dir: &Path, now: SystemTime, clean: bool, stop: &dyn Fn() -> bool) -> Walked {
    use crate::collect::{labels, layout};
    use std::collections::HashSet;
    let mut out = Walked::default();
    for day in layout::day_dirs(dir) {
        if stop() {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&day) else {
            continue;
        };
        // What is still in the folder, and the stems of its images.
        let mut left = 0usize;
        let mut images: HashSet<String> = HashSet::new();
        for ent in rd.flatten() {
            left += 1;
            let Ok(ft) = ent.file_type() else { continue };
            if !ft.is_file() {
                continue;
            }
            let path = ent.path();
            let name = ent.file_name().to_string_lossy().into_owned();
            let Ok(meta) = ent.metadata() else { continue };
            let modified = meta.modified().unwrap_or(now);
            if crate::fsx::is_temp_name(&name) {
                if clean && is_stale(now, modified) && std::fs::remove_file(&path).is_ok() {
                    left -= 1;
                }
                continue;
            }
            // A run that was labelling a frame as it went leaves a label file behind.
            if let Some(stem) = name.strip_suffix(labels::LABELS_SUFFIX) {
                if clean
                    && !day.join(format!("{stem}.jpg")).exists()
                    && matches!(
                        labels::unless_protected(&path, || std::fs::remove_file(&path)),
                        Some(Ok(()))
                    )
                {
                    left -= 1;
                }
                continue;
            }
            if let Some(stem) = name.strip_suffix(".json") {
                if clean
                    && layout::is_frame_stem(stem)
                    && is_stale(now, modified)
                    && orphan_sidecar(&day, stem)
                    && std::fs::remove_file(&path).is_ok()
                {
                    tracing::debug!("retention: removed the orphan sidecar {}", path.display());
                    out.orphan_sidecars += 1;
                    left -= 1;
                }
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jpg") {
                continue;
            }
            if let Some(stem) = name.strip_suffix(".jpg") {
                images.insert(stem.to_string());
            }
            let side_len = std::fs::symlink_metadata(path.with_extension("json"))
                .ok()
                .filter(|m| m.file_type().is_file())
                .map_or(0, |m| m.len());
            let size = meta.len().saturating_add(side_len);
            if labels::is_protected(&labels::path_of_frame(&path)) {
                out.usage.protected_bytes = out.usage.protected_bytes.saturating_add(size);
                out.usage.protected_frames += 1;
            } else {
                out.usage.prunable_bytes = out.usage.prunable_bytes.saturating_add(size);
                out.entries.push((path, modified, size));
            }
        }
        if clean && (left == 0 || names_a_missing_image(&day, &images)) {
            out.repair.push(day);
        }
    }
    out
}

/// Whether the sidecar `<day>/<stem>.json` has lost its frame: no `<stem>.jpg` (not even a
/// symlink) and no label file that retention must keep ([`labels::is_protected`]).
fn orphan_sidecar(day: &Path, stem: &str) -> bool {
    use crate::collect::labels;
    let jpg = day.join(format!("{stem}.jpg"));
    std::fs::symlink_metadata(&jpg).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        && !labels::is_protected(&labels::path_of_frame(&jpg))
}

/// Whether `day`'s index lists a frame whose image is not among `images` (the folder's listing).
/// An index that cannot be read names nothing here (the rewrite would refuse it anyway).
fn names_a_missing_image(day: &Path, images: &std::collections::HashSet<String>) -> bool {
    use crate::collect::{layout, record::IndexLine};
    let index = layout::read_lines::<IndexLine>(&day.join(layout::INDEX_FILE));
    index.failed.is_none() && index.lines.iter().any(|l| !images.contains(&l.id))
}

/// How the frames of the dataset folder `dir` divide up (reads every label file; call it off the
/// async runtime). A missing folder is empty.
pub fn dataset_usage(dir: &Path) -> DatasetUsage {
    walk_dataset(dir, SystemTime::now(), false, &|| false).usage
}

/// Delete the oldest frames of the dataset folder `dir` (`<scene>/<camera>/<date>/<stem>.jpg`
/// with its `.json` sidecar and `.labels.json`, see `docs/DATASET.md`) until none is older than
/// `max_age_days` and the deletable frames total at most `max_bytes` (0 disables a limit), then
/// drop the deleted frames from their day's `index.jsonl` and remove emptied folders. Reviewed
/// frames and frames with a human decision are never deleted and do not count toward the cap; a
/// label file with nothing human on it (an auto-label run writes one even when it finds nothing)
/// goes with its frame. Only those names are touched (no symlinks, nothing outside the layout,
/// never `review-queue.jsonl`).
///
/// Crash repair (audit 2026-11 phase 2): what an interrupted write or an earlier pass that was
/// killed left behind is put right on the way ([`walk_dataset`]): temporary files and orphan
/// sidecars older than [`LEFTOVER_AGE`] go, and each day folder whose index names a frame whose
/// image is gone, or that holds nothing, gets the same index rewrite and folder removal as a day
/// whose frames were just deleted. A missing folder is not an error.
///
/// Writers: the collector thread, where this runs, and an import (a job thread of the service or a
/// `train import` process) write frames and index lines into the same day folders. Each day's
/// index rewrite and the removal of its emptied folders run under the dataset's index lock
/// ([`crate::collect::layout::lock_index`]), which every index append takes too, so no appended
/// line is lost; an import whose freshly made, still empty day folder goes makes it again
/// ([`crate::collect::layout::write_in_day`]). Label files can change at any time, so each frame's
/// is read again, under the label write lock, right before the frame goes. Stops between two
/// frames when the collector thread running it is asked to stop
/// ([`crate::collect::stop_requested`]; never elsewhere).
pub fn prune_dataset(dir: &Path, max_age_days: u32, max_bytes: u64) -> Result<PruneReport> {
    prune_dataset_until(
        dir,
        max_age_days,
        max_bytes,
        &crate::collect::stop_requested,
    )
}

/// [`prune_dataset`], asking `stop` between two frames (and between two day folders of the walk):
/// once it says yes the pass ends with [`PruneReport::stopped`]. The doomed frames go one day
/// folder at a time, the days in the order of their oldest doomed frame, and each day's index is
/// rewritten (and its emptied folders removed) as soon as its frames are gone, before the next
/// day starts: a stop, or a kill, leaves at most the day in hand with index lines of deleted
/// frames, and the readers skip a line whose image is gone.
pub fn prune_dataset_until(
    dir: &Path,
    max_age_days: u32,
    max_bytes: u64,
    stop: &dyn Fn() -> bool,
) -> Result<PruneReport> {
    use crate::collect::labels;
    use std::collections::HashMap;
    if !dir.is_dir() {
        return Ok(PruneReport::default());
    }
    let now = SystemTime::now();
    let Walked {
        entries,
        mut usage,
        repair,
        orphan_sidecars,
    } = walk_dataset(dir, now, true, stop);
    if stop() {
        // The walk may be incomplete: a cap decided on it would be wrong.
        tracing::info!(
            "dataset retention of {} stopped before it deleted anything (the service is \
             stopping); the next pass applies the caps",
            dir.display()
        );
        return Ok(PruneReport {
            remaining: entries.len(),
            usage,
            stopped: true,
            orphan_sidecars,
            ..PruneReport::default()
        });
    }
    let age =
        (max_age_days > 0).then(|| Duration::from_secs(u64::from(max_age_days) * SECS_PER_DAY));
    let cap = (max_bytes > 0).then_some(max_bytes);
    let doomed = plan_prune(&entries, now, age, cap);
    let sizes: HashMap<&Path, u64> = entries.iter().map(|e| (e.0.as_path(), e.2)).collect();
    let mut report = PruneReport {
        remaining: entries.len(),
        orphan_sidecars,
        ..PruneReport::default()
    };
    let mut to_repair: std::collections::HashSet<&Path> =
        repair.iter().map(PathBuf::as_path).collect();
    // One day folder at a time, oldest doomed frame first.
    let mut days: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();
    let mut day_of: HashMap<PathBuf, usize> = HashMap::new();
    for path in doomed {
        let day = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let at = *day_of.entry(day.clone()).or_insert_with(|| {
            days.push((day, Vec::new()));
            days.len() - 1
        });
        if let Some((_, frames)) = days.get_mut(at) {
            frames.push(path);
        }
    }
    for (day, frames) in &days {
        let mut deleted_here = false;
        for path in frames {
            if stop() {
                report.stopped = true;
                break;
            }
            let size = sizes.get(path.as_path()).copied().unwrap_or(0);
            let label_file = labels::path_of_frame(path);
            let removed = labels::unless_protected(&label_file, || {
                std::fs::remove_file(path)?;
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
                    deleted_here = true;
                }
                Some(Err(e)) => {
                    report.failed += 1;
                    tracing::debug!("retention: removing {}: {e}", path.display());
                }
            }
        }
        let repairing = to_repair.remove(day.as_path());
        if (deleted_here || repairing) && finish_day(day) && repairing {
            report.repaired_days += 1;
        }
        if report.stopped {
            break;
        }
    }
    // The days left out of step that lost no frame in this pass, in walk order.
    for day in &repair {
        if report.stopped {
            break;
        }
        if !to_repair.contains(day.as_path()) {
            continue;
        }
        if stop() {
            report.stopped = true;
            break;
        }
        if finish_day(day) {
            report.repaired_days += 1;
        }
    }
    if report.repaired_days > 0 || report.orphan_sidecars > 0 {
        tracing::info!(
            "dataset in {}: repaired {} day folder(s) and removed {} orphan sidecar(s) left by an \
             interrupted write or retention pass (a crash or a forced stop): index lines of \
             deleted frames dropped, empty folders removed",
            dir.display(),
            report.repaired_days,
            report.orphan_sidecars
        );
    }
    report.usage = usage;
    if report.deleted > 0 || report.failed > 0 || report.stopped {
        tracing::info!(
            "dataset in {}: deleted {} frame(s) ({} MB), {} failed, {} left{}",
            dir.display(),
            report.deleted,
            report.freed_bytes / BYTES_PER_MB,
            report.failed,
            report.remaining,
            if report.stopped {
                "; stopped early because the service is stopping, the next pass goes on"
            } else {
                ""
            }
        );
    }
    Ok(report)
}

/// After frames of `day` were deleted (or the walk found it out of step): its index without them,
/// and the emptied day, camera and scene folders removed (`remove_dir` fails on a non-empty one),
/// all under the dataset's index lock, so an append of an import is never lost to the rewrite.
/// When the lock cannot be had (another process holds it past the wait, or the service is
/// stopping) the day is left as it is, and false: the next pass finds it out of step and finishes
/// it.
fn finish_day(day: &Path) -> bool {
    use crate::collect::layout;
    let lock = match layout::lock_index(day) {
        Ok(lock) => lock,
        Err(e) => {
            tracing::warn!("dataset retention: {e:#}");
            return false;
        }
    };
    if let Err(e) = layout::rewrite_index_locked(day, &lock) {
        tracing::warn!("dataset retention: {e:#}");
    }
    let mut p = Some(day);
    for _ in 0..3 {
        let Some(d) = p else { break };
        if std::fs::remove_dir(d).is_err() {
            break;
        }
        p = d.parent();
    }
    drop(lock);
    true
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
    /// The storage sweep ([`crate::storage::sweep`]): on the first tick, then once every
    /// [`crate::storage::sweep::INTERVAL`]. None = never.
    pub storage: Option<crate::storage::Settings>,
}

/// True when a low-space warning may be logged now (never yet, or the last is old enough).
fn may_warn(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|t| now.saturating_duration_since(t) >= FREE_SPACE_WARN_INTERVAL)
}

/// Whether the storage sweep is due: never run yet, or [`crate::storage::sweep::INTERVAL`] ago.
fn storage_due(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|t| now.saturating_duration_since(t) >= crate::storage::sweep::INTERVAL)
}

/// One sweep: prune the saved images, then check free space (`last_warn` throttles the log),
/// then the storage sweep when it is due (`last_storage`), which checks `shutdown` between items.
fn sweep(
    cfg: &SweeperConfig,
    last_warn: &mut Option<Instant>,
    last_storage: &mut Option<Instant>,
    shutdown: &CancellationToken,
) {
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
    if let Some(st) = &cfg.storage {
        let now = Instant::now();
        if storage_due(*last_storage, now) {
            *last_storage = Some(now);
            crate::storage::sweep::run(st, shutdown);
        }
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
/// [`FIRST_SWEEP_DELAY`]), and the storage sweep once a day, until `shutdown`. `settings` is called on every tick, so config
/// changes apply at once. The thread runs below normal priority
/// ([`crate::backend::priority::lower_this_thread`]). Once `shutdown` is cancelled the thread
/// ends within one item of the storage sweep or one saved-image prune; join it with
/// [`join_within`].
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
            crate::backend::priority::lower_this_thread(
                crate::backend::priority::ThreadRole::Background,
            );
            let mut last_warn = None;
            let mut last_storage = None;
            let mut delay = FIRST_SWEEP_DELAY;
            while nap(delay, &shutdown) {
                sweep(&settings(), &mut last_warn, &mut last_storage, &shutdown);
                delay = SWEEP_INTERVAL;
            }
        })
        .expect("spawning the retention thread (OS thread limit or out of memory)")
}

/// Join `handle` if its thread ends by `deadline` (polled every 10 ms; std has no timed join).
/// Err(handle) when it is still running then: the caller warns and leaves it (detached when the
/// handle drops). A thread that panicked counts as ended, with a warning naming `what`.
pub fn join_within(
    handle: std::thread::JoinHandle<()>,
    deadline: Instant,
    what: &str,
) -> std::result::Result<(), std::thread::JoinHandle<()>> {
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return Err(handle);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if handle.join().is_err() {
        tracing::warn!("the {what} thread panicked while stopping");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn the_storage_sweep_runs_first_then_once_a_day() {
        let now = Instant::now();
        assert!(storage_due(None, now));
        assert!(!storage_due(Some(now), now + SWEEP_INTERVAL));
        assert!(storage_due(
            Some(now),
            now + crate::storage::sweep::INTERVAL
        ));
    }

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

    /// Audit 2026-11 phase 1: a stop between two frames ends the pass there, and every day whose
    /// frames went has its index rewritten before the next day starts.
    #[test]
    fn a_stopped_prune_ends_between_frames_with_each_days_index_in_step() {
        use crate::collect::layout::{INDEX_FILE, read_index};
        let d = Tmp::new();
        let p = d.path();
        let (day1, day2) = (p.join("ir/cam/2026-01-01"), p.join("ir/cam/2026-01-02"));
        for (stem, age) in [("a", 30), ("b", 29), ("c", 28)] {
            dataset_frame(p, "ir/cam/2026-01-01", stem, 100, age);
        }
        for (stem, age) in [("d", 20), ("e", 19), ("f", 18)] {
            dataset_frame(p, "ir/cam/2026-01-02", stem, 100, age);
        }
        let ids = |day: &Path| -> Vec<String> {
            read_index(&day.join(INDEX_FILE))
                .into_iter()
                .map(|l| l.id)
                .collect()
        };
        // Stop once "b" is gone: "c" (same day) and the second day stay.
        let stop = || !day1.join("b.jpg").exists();
        let r = prune_dataset_until(p, 0, 1, &stop).unwrap();
        assert!(r.stopped);
        assert_eq!((r.deleted, r.remaining), (2, 4));
        assert_eq!(
            ids(&day1),
            ["c"],
            "the day in hand is rewritten at the stop"
        );
        assert_eq!(ids(&day2), ["d", "e", "f"]);
        assert!(day2.join("d.jpg").is_file());

        // A stop right after the first day: its index (and folder) are done before the next day.
        let seen_at_stop = std::cell::RefCell::new(None);
        let stop = || {
            let gone = !day1.join("c.jpg").exists();
            if gone && seen_at_stop.borrow().is_none() {
                *seen_at_stop.borrow_mut() = Some((day1.exists(), ids(&day2)));
            }
            gone
        };
        let r = prune_dataset_until(p, 0, 1, &stop).unwrap();
        assert!(r.stopped);
        assert_eq!(r.deleted, 1);
        assert_eq!(
            seen_at_stop.into_inner(),
            Some((false, vec!["d".to_string(), "e".into(), "f".into()])),
            "the emptied day was finished before the stop was asked again"
        );

        // A stop asked before anything: nothing goes, and the next pass does the rest.
        let r = prune_dataset_until(p, 0, 1, &|| true).unwrap();
        assert!(r.stopped);
        assert_eq!((r.deleted, r.remaining), (0, 0), "the walk itself stopped");
        let r = prune_dataset(p, 0, 1).unwrap();
        assert!(!r.stopped);
        assert_eq!(r.deleted, 3);
        assert!(!day2.exists());
    }

    /// The collector thread's flag reaches [`prune_dataset`] through the thread it runs on only.
    #[test]
    fn prune_dataset_follows_the_stop_flag_of_its_own_thread() {
        let d = Tmp::new();
        let p = d.path();
        dataset_frame(p, "ir/cam/2026-01-01", "a", 100, 3);
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
        crate::collect::bind_stop(Some(flag.clone()));
        let r = prune_dataset(p, 0, 1).unwrap();
        assert!(r.stopped && r.deleted == 0);
        // Another thread has no flag.
        let other = p.to_path_buf();
        let r = std::thread::spawn(move || prune_dataset(&other, 0, 1).unwrap())
            .join()
            .unwrap();
        assert!(!r.stopped && r.deleted == 1);
        crate::collect::bind_stop(None);
    }

    #[test]
    fn join_within_waits_for_a_thread_that_ends_and_gives_up_on_one_that_does_not() {
        let quick = std::thread::spawn(|| std::thread::sleep(Duration::from_millis(20)));
        assert!(join_within(quick, Instant::now() + Duration::from_secs(10), "t").is_ok());
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let stuck = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        let started = Instant::now();
        let back = join_within(stuck, Instant::now() + Duration::from_millis(50), "t").unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5));
        tx.send(()).unwrap();
        back.join().unwrap();
        let panics = std::thread::spawn(|| panic!("test panic"));
        assert!(join_within(panics, Instant::now() + Duration::from_secs(10), "t").is_ok());
    }

    /// Set the modified time of `p` to `ago` before now.
    fn age(p: &Path, ago: Duration) {
        std::fs::File::options()
            .write(true)
            .open(p)
            .unwrap()
            .set_modified(SystemTime::now() - ago)
            .unwrap();
    }

    const TWO_HOURS: Duration = Duration::from_secs(2 * 3600);
    /// Frame stems as the collector writes them.
    const S1: &str = "20260101T000000000Z_req1";
    const S2: &str = "20260101T000001000Z_req2";
    const S3: &str = "20260101T000002000Z_req3";

    /// Audit 2026-11 phase 2 (R:collect-4): a kill mid-prune (or a failed write) leaves index
    /// lines of deleted frames, orphan sidecars and empty day folders; one prune with no caps
    /// repairs them all, and leaves what may still be in use or holds human work.
    #[test]
    fn a_prune_repairs_what_an_interrupted_prune_or_write_left() {
        use crate::collect::labels::{self, LabelFile, Status};
        use crate::collect::layout::{INDEX_FILE, Summary, read_index};
        let d = Tmp::new();
        let p = d.path();
        // A day whose prune was killed after its images went: index, sidecars, no images.
        let gone_day = p.join("ir/cam/2026-01-01");
        dataset_frame(p, "ir/cam/2026-01-01", S1, 100, 1);
        dataset_frame(p, "ir/cam/2026-01-01", S2, 100, 1);
        for stem in [S1, S2] {
            std::fs::remove_file(gone_day.join(format!("{stem}.jpg"))).unwrap();
        }
        // A day that keeps a frame, with a dangling line and an orphan sidecar beside it.
        let kept_day = p.join("day/yard/2026-01-02");
        dataset_frame(p, "day/yard/2026-01-02", "kept", 100, 1);
        dataset_frame(p, "day/yard/2026-01-02", S1, 100, 1);
        std::fs::remove_file(kept_day.join(format!("{S1}.jpg"))).unwrap();
        // Left alone: a fresh orphan sidecar (its writer may be mid-way), one whose frame holds a
        // reviewed label file, a .json that is no frame's, and a reviewed label file.
        std::fs::write(kept_day.join(format!("{S2}.json")), b"{}").unwrap();
        std::fs::write(kept_day.join(format!("{S3}.json")), b"{}").unwrap();
        age(&kept_day.join(format!("{S3}.json")), TWO_HOURS);
        let reviewed = LabelFile {
            status: Status::Reviewed,
            ..LabelFile::default()
        };
        labels::write(&kept_day.join(format!("{S3}.labels.json")), &reviewed).unwrap();
        std::fs::write(kept_day.join("notes.json"), b"{}").unwrap();
        age(&kept_day.join("notes.json"), TWO_HOURS);
        // An empty day folder a failed write left, alone in its camera folder.
        let empty_day = p.join("night/porch/2026-01-03");
        std::fs::create_dir_all(&empty_day).unwrap();
        // Temporary files: write_atomic's exact shape goes when old, anything else stays.
        let temp = kept_day.join(format!(
            ".kept.jpg.{}.tmp",
            "0123456789abcdef0123456789abcdef"
        ));
        let fresh_temp = kept_day.join(format!(
            ".kept.json.{}.tmp",
            "fedcba9876543210fedcba9876543210"
        ));
        let lookalike = kept_day.join(".mine.tmp");
        for t in [&temp, &fresh_temp, &lookalike] {
            std::fs::write(t, b"t").unwrap();
        }
        age(&temp, TWO_HOURS);
        age(&lookalike, TWO_HOURS);

        assert_eq!(Summary::scan(p).frames, 4, "the dangling lines are counted");
        let r = prune_dataset(p, 0, 0).unwrap();
        assert_eq!((r.deleted, r.failed, r.repaired_days), (0, 0, 3), "{r:?}");
        assert_eq!(
            r.orphan_sidecars, 3,
            "two in the gone day, one beside 'kept'"
        );
        assert!(!gone_day.exists(), "the emptied day folder goes");
        assert!(!p.join("ir").exists(), "and its camera and scene folders");
        assert!(
            !p.join("night/porch").exists(),
            "the empty day and its camera"
        );
        let ids: Vec<String> = read_index(&kept_day.join(INDEX_FILE))
            .into_iter()
            .map(|l| l.id)
            .collect();
        assert_eq!(ids, ["kept"]);
        assert_eq!(Summary::scan(p).frames, 1);
        for f in [
            "kept.jpg".to_string(),
            "kept.json".into(),
            format!("{S2}.json"),
            format!("{S3}.json"),
            format!("{S3}.labels.json"),
            "notes.json".into(),
        ] {
            assert!(kept_day.join(&f).exists(), "{f} stays");
        }
        assert!(!kept_day.join(format!("{S1}.json")).exists());
        assert!(!temp.exists());
        assert!(fresh_temp.exists() && lookalike.exists());
        // Nothing left to repair: a second pass does nothing.
        let r = prune_dataset(p, 0, 0).unwrap();
        assert_eq!((r.repaired_days, r.orphan_sidecars), (0, 0));
    }

    /// The emptied-folder removal holds the index lock: an import that made the day folder and
    /// takes the lock to list its frame never sees it go under it.
    #[test]
    fn an_empty_day_is_not_removed_while_another_process_holds_the_index_lock() {
        use crate::collect::layout::{INDEX_FILE, INDEX_LOCK_FILE, read_index};
        let d = Tmp::new();
        let p = d.path();
        let day = p.join("ir/cam/2026-01-04");
        std::fs::create_dir_all(&day).unwrap();
        let held = std::fs::File::create(p.join(INDEX_LOCK_FILE)).unwrap();
        fs4::FileExt::try_lock(&held).unwrap();
        let prune = {
            let p = p.to_path_buf();
            std::thread::spawn(move || prune_dataset(&p, 0, 0).unwrap())
        };
        std::thread::sleep(Duration::from_millis(200));
        assert!(!prune.is_finished(), "the prune waits for the index lock");
        assert!(day.is_dir(), "nothing is removed meanwhile");
        // The other process writes its frame and lists it, then lets go.
        std::fs::write(day.join(format!("{S1}.jpg")), b"x").unwrap();
        std::fs::write(day.join(format!("{S1}.json")), b"{}").unwrap();
        let line = crate::collect::record::IndexLine {
            id: S1.into(),
            timestamp: "t".into(),
            scene: "ir".into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: vec![crate::collect::Bucket::Empty],
            labels: Vec::new(),
            bytes: 2,
        };
        crate::collect::layout::append_line(
            &day.join(INDEX_FILE),
            &serde_json::to_string(&line).unwrap(),
        )
        .unwrap();
        fs4::FileExt::unlock(&held).unwrap();
        let r = prune.join().unwrap();
        assert_eq!(r.repaired_days, 1);
        assert!(day.join(format!("{S1}.jpg")).is_file());
        assert_eq!(read_index(&day.join(INDEX_FILE)).len(), 1);
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
