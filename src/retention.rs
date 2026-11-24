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
//! sweeper's cancel token and checks it between items, the saved-image prune between two
//! deletions. [`join_within`] is the bounded join of a stopping thread.
//!
//! Idle cost (audit 2026-11 phase 6): the sweeper parks between ticks ([`crate::sync::nap`], one
//! wake-up per [`SWEEP_INTERVAL`]); [`plan_prune`] hands back the doomed entries themselves, so a
//! prune of n files is O(n log n); and a dataset walk takes the day folders that look unchanged
//! from a [`DayCache`] (two stats a day instead of a stat per file and a read per label file).
//! Whether the 10-minute tick walks the dataset at all is the collector's decision
//! ([`crate::collect::store::Store::prune_if_due`]).

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
/// A file retention may delete: (path, modified time, bytes).
type Entry = (PathBuf, SystemTime, u64);
const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// Which of `entries` (path, modified time, size) to delete, oldest first: first everything
/// older than `max_age`, then the oldest of the rest until the total is at most `max_bytes`.
/// `None` disables the respective limit. The entries themselves come back, so a caller reads each
/// one's size from it (audit 2026-11 phase 6: looking every doomed path up again in `entries` was
/// quadratic, minutes of a core on a large folder).
pub fn plan_prune(
    entries: &[(PathBuf, SystemTime, u64)],
    now: SystemTime,
    max_age: Option<Duration>,
    max_bytes: Option<u64>,
) -> Vec<&(PathBuf, SystemTime, u64)> {
    plan(entries, now, max_age, max_bytes)
        .into_iter()
        .map(|(_, e)| e)
        .collect()
}

/// [`plan_prune`] with each doomed entry's index in `entries`. O(n log n) in the entries.
fn plan(
    entries: &[(PathBuf, SystemTime, u64)],
    now: SystemTime,
    max_age: Option<Duration>,
    max_bytes: Option<u64>,
) -> Vec<(usize, &Entry)> {
    let mut sorted: Vec<(usize, &(PathBuf, SystemTime, u64))> =
        entries.iter().enumerate().collect();
    sorted.sort_by(|(_, a), (_, b)| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = Vec::new();
    let mut kept = Vec::new();
    for (i, e) in sorted {
        let too_old =
            max_age.is_some_and(|age| now.duration_since(e.1).is_ok_and(|elapsed| elapsed > age));
        if too_old {
            out.push((i, e));
        } else {
            kept.push((i, e));
        }
    }
    if let Some(cap) = max_bytes {
        let mut total: u64 = kept.iter().map(|(_, e)| e.2).fold(0, u64::saturating_add);
        for (i, e) in kept {
            if total <= cap {
                break;
            }
            total = total.saturating_sub(e.2);
            out.push((i, e));
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
    /// The pass ended early for a stop ([`prune_dataset_until`], [`prune_saved_images`]); the
    /// rest goes at the next one.
    pub stopped: bool,
    /// Day folders an interrupted prune or write had left out of step, put right
    /// ([`prune_dataset`]: index lines of deleted frames dropped, emptied folders removed).
    pub repaired_days: usize,
    /// Sidecars whose image was gone, removed ([`prune_dataset`]).
    pub orphan_sidecars: usize,
    /// The modified time of the oldest deletable frame left ([`prune_dataset`]; None when none
    /// is, or the pass stopped in its walk).
    pub oldest_prunable: Option<SystemTime>,
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
/// most `max_mb`. 0 disables a limit. A missing folder is not an error. `stop` is checked between
/// two deletions: once it is cancelled the pass ends with [`PruneReport::stopped`] (the service's
/// stop joins the sweeper thread running it), and the next pass goes on.
pub fn prune_saved_images(
    dir: &Path,
    max_age_days: u32,
    max_mb: u64,
    stop: &CancellationToken,
) -> Result<PruneReport> {
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
    let report = delete_planned(&entries, SystemTime::now(), age, cap, stop, &mut |p| {
        std::fs::remove_file(p)
    });
    if report.deleted > 0 || report.failed > 0 || report.stopped {
        tracing::info!(
            "saved images in {}: deleted {} file(s) ({} MB), {} failed, {} left{}",
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

/// [`prune_saved_images`]'s plan and deletions over `entries`, each doomed file removed with
/// `remove` and its size read from its entry; `stop` is checked before each removal.
fn delete_planned(
    entries: &[(PathBuf, SystemTime, u64)],
    now: SystemTime,
    max_age: Option<Duration>,
    max_bytes: Option<u64>,
    stop: &CancellationToken,
    remove: &mut dyn FnMut(&Path) -> std::io::Result<()>,
) -> PruneReport {
    let mut report = PruneReport {
        remaining: entries.len(),
        ..PruneReport::default()
    };
    for (path, _, size) in plan_prune(entries, now, max_age, max_bytes) {
        if stop.is_cancelled() {
            report.stopped = true;
            break;
        }
        match remove(path) {
            Ok(()) => {
                report.deleted += 1;
                report.freed_bytes = report.freed_bytes.saturating_add(*size);
                report.remaining = report.remaining.saturating_sub(1);
            }
            Err(e) => {
                report.failed += 1;
                tracing::debug!("retention: removing {}: {e}", path.display());
            }
        }
    }
    report
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

/// A folder or file time this close to the walk (or later than it) may still change unseen within
/// the file system's time resolution, so that day is not kept in a [`DayCache`] (the review index's
/// day stat uses the same window).
const RACY_WINDOW: Duration = Duration::from_secs(2);

/// How a day folder looked when it was walked: its own modified time and its `index.jsonl`'s length
/// and modified time. Every change a walk could see moves one of them: frames, sidecars, label files
/// and temporary files are created, renamed into place or removed in the folder (each moves the
/// folder's time), and index lines are appended (length and time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DayKey {
    folder: SystemTime,
    index: Option<(u64, Option<SystemTime>)>,
}

impl DayKey {
    /// The key of `day` now; None when it is too fresh to trust ([`RACY_WINDOW`]) or unreadable.
    fn of(day: &Path, now: SystemTime) -> Option<Self> {
        let folder = std::fs::metadata(day).ok()?.modified().ok()?;
        let index = std::fs::metadata(day.join(crate::collect::layout::INDEX_FILE))
            .ok()
            .map(|m| (m.len(), m.modified().ok()));
        let racy = |t: SystemTime| !now.duration_since(t).is_ok_and(|d| d >= RACY_WINDOW);
        let fresh = racy(folder) || index.is_some_and(|(_, t)| t.is_none_or(racy));
        (!fresh).then_some(Self { folder, index })
    }
}

/// One day folder's part of a walk.
#[derive(Debug, Clone, Default)]
struct DayWalk {
    entries: Vec<(PathBuf, SystemTime, u64)>,
    usage: DatasetUsage,
    /// Needs the repair of [`Walked::repair`].
    repair: bool,
    /// Walked with `clean` (only then is `repair` known).
    cleaned: bool,
}

/// What earlier walks found in each day folder of one dataset folder, reused while the day looks
/// unchanged ([`DayKey`]): a walk then costs two stats for such a day instead of a stat of every
/// file and a read of every label file (audit 2026-11 phase 6). The owner (the collector's
/// [`crate::collect::store::Store`]) clears it now and then ([`Self::clear`]) so a file system
/// whose folder times do not move (some network shares) is walked in full anyway. A stale entry
/// can only list a frame whose label file changed meanwhile: each frame's label file is read
/// again, under the label write lock, right before it goes, so a frame with human work is never
/// deleted on the cache's word.
#[derive(Debug, Default)]
pub struct DayCache {
    root: Option<PathBuf>,
    days: std::collections::HashMap<PathBuf, (DayKey, DayWalk)>,
    /// Days walked (not taken from the cache), for tests.
    walked: usize,
}

impl DayCache {
    /// Forget every day: the next walk reads them all.
    pub fn clear(&mut self) {
        self.root = None;
        self.days.clear();
    }

    /// Day folders read from disk by the walks so far (the rest came from the cache).
    pub fn days_walked(&self) -> usize {
        self.walked
    }

    /// Day folders kept.
    pub fn len(&self) -> usize {
        self.days.len()
    }

    /// Whether no day is kept.
    pub fn is_empty(&self) -> bool {
        self.days.is_empty()
    }

    /// Point the cache at `dir`, forgetting another folder's days.
    fn root_at(&mut self, dir: &Path) {
        if self.root.as_deref() != Some(dir) {
            self.days.clear();
            self.root = Some(dir.to_path_buf());
        }
    }
}

/// The frames of the dataset folder `dir` retention may delete and how the folder divides up.
/// `stop` is asked between two day folders: once it says yes the walk ends there, incomplete
/// (the caller checks `stop` again). A day folder that looks as it did at an earlier walk
/// ([`DayKey`]) is taken from `cache` (a cleaning walk takes only what a cleaning walk found);
/// the rest are read and kept there.
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
/// A day taken from the cache is not cleaned this time (its leftovers wait for the walk that
/// reads it again; the cache's owner clears it at least hourly).
///
/// A frame whose image and sidecar are in but whose index line is not (a kill between the two) is
/// not repaired: the caps still count and delete it, and adding its line would race the writer.
/// An image with no sidecar is left alone (the Review tab's Rescan folder lists it).
fn walk_dataset(
    dir: &Path,
    now: SystemTime,
    clean: bool,
    stop: &dyn Fn() -> bool,
    cache: &mut DayCache,
) -> Walked {
    use crate::collect::layout;
    cache.root_at(dir);
    let mut out = Walked::default();
    let mut seen = std::collections::HashSet::new();
    let mut complete = true;
    for day in layout::day_dirs(dir) {
        if stop() {
            complete = false;
            break;
        }
        let key = DayKey::of(&day, now);
        let cached = cache
            .days
            .get(&day)
            .filter(|(k, w)| Some(*k) == key && (w.cleaned || !clean))
            .map(|(_, w)| w.clone());
        let walk = match cached {
            Some(w) => w,
            None => {
                cache.walked = cache.walked.saturating_add(1);
                let w = walk_day(&day, now, clean, &mut out.orphan_sidecars);
                match key {
                    Some(k) => {
                        cache.days.insert(day.clone(), (k, w.clone()));
                    }
                    None => {
                        cache.days.remove(&day);
                    }
                }
                w
            }
        };
        out.usage.prunable_bytes = out
            .usage
            .prunable_bytes
            .saturating_add(walk.usage.prunable_bytes);
        out.usage.protected_bytes = out
            .usage
            .protected_bytes
            .saturating_add(walk.usage.protected_bytes);
        out.usage.protected_frames = out
            .usage
            .protected_frames
            .saturating_add(walk.usage.protected_frames);
        out.entries.extend(walk.entries);
        if clean && walk.repair {
            out.repair.push(day.clone());
        }
        seen.insert(day);
    }
    if complete {
        // Day folders that are gone.
        cache.days.retain(|d, _| seen.contains(d));
    }
    out
}

/// [`walk_dataset`]'s reading of one day folder.
fn walk_day(day: &Path, now: SystemTime, clean: bool, orphan_sidecars: &mut usize) -> DayWalk {
    use crate::collect::{labels, layout};
    use std::collections::HashSet;
    let mut out = DayWalk {
        cleaned: clean,
        ..DayWalk::default()
    };
    let Ok(rd) = std::fs::read_dir(day) else {
        return out;
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
                && orphan_sidecar(day, stem)
                && std::fs::remove_file(&path).is_ok()
            {
                tracing::debug!("retention: removed the orphan sidecar {}", path.display());
                *orphan_sidecars += 1;
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
    out.repair = clean && (left == 0 || names_a_missing_image(day, &images));
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
    dataset_survey(dir, &mut DayCache::default()).usage
}

/// What a walk of the dataset folder knows that decides whether retention has work there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Survey {
    pub usage: DatasetUsage,
    /// The modified time of the oldest frame retention may delete, None when there is none: the
    /// age cap has nothing to do before this frame is `max_age_days` old.
    pub oldest_prunable: Option<SystemTime>,
}

/// [`dataset_usage`] and the oldest deletable frame, days that look unchanged taken from `cache`
/// (nothing is cleaned or deleted).
pub fn dataset_survey(dir: &Path, cache: &mut DayCache) -> Survey {
    let w = walk_dataset(dir, SystemTime::now(), false, &|| false, cache);
    Survey {
        usage: w.usage,
        oldest_prunable: w.entries.iter().map(|e| e.1).min(),
    }
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
    prune_dataset_with(dir, max_age_days, max_bytes, stop, &mut DayCache::default())
}

/// [`prune_dataset_until`], the walk taking the days that look unchanged from `cache`
/// ([`DayCache`]); the days it deletes from are dropped from the cache, so the next walk reads
/// them again. [`PruneReport::oldest_prunable`] is the oldest deletable frame left.
pub fn prune_dataset_with(
    dir: &Path,
    max_age_days: u32,
    max_bytes: u64,
    stop: &dyn Fn() -> bool,
    cache: &mut DayCache,
) -> Result<PruneReport> {
    prune_dataset_at(dir, max_age_days, max_bytes, stop, cache, SystemTime::now())
}

/// [`prune_dataset_with`] as of `now` (tests move it past [`RACY_WINDOW`]).
fn prune_dataset_at(
    dir: &Path,
    max_age_days: u32,
    max_bytes: u64,
    stop: &dyn Fn() -> bool,
    cache: &mut DayCache,
    now: SystemTime,
) -> Result<PruneReport> {
    use crate::collect::labels;
    use std::collections::HashMap;
    if !dir.is_dir() {
        cache.clear();
        return Ok(PruneReport::default());
    }
    let Walked {
        entries,
        mut usage,
        repair,
        orphan_sidecars,
    } = walk_dataset(dir, now, true, stop, cache);
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
    let doomed = plan(&entries, now, age, cap);
    let mut report = PruneReport {
        remaining: entries.len(),
        orphan_sidecars,
        ..PruneReport::default()
    };
    // Entries no longer deletable after this pass: deleted, or protected meanwhile.
    let mut gone = vec![false; entries.len()];
    let mut to_repair: std::collections::HashSet<&Path> =
        repair.iter().map(PathBuf::as_path).collect();
    // One day folder at a time, oldest doomed frame first.
    let mut days: Vec<(&Path, Vec<(usize, &Entry)>)> = Vec::new();
    let mut day_of: HashMap<&Path, usize> = HashMap::new();
    for (i, e) in doomed {
        let day = e.0.parent().unwrap_or(Path::new(""));
        let at = *day_of.entry(day).or_insert_with(|| {
            days.push((day, Vec::new()));
            days.len() - 1
        });
        if let Some((_, frames)) = days.get_mut(at) {
            frames.push((i, e));
        }
    }
    for (day, frames) in &days {
        let mut deleted_here = false;
        // Its files change now: the next walk reads it again.
        cache.days.remove(*day);
        for &(i, (path, _, size)) in frames {
            if stop() {
                report.stopped = true;
                break;
            }
            let size = *size;
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
                    if let Some(g) = gone.get_mut(i) {
                        *g = true;
                    }
                }
                Some(Ok(())) => {
                    usage.prunable_bytes = usage.prunable_bytes.saturating_sub(size);
                    if let Some(g) = gone.get_mut(i) {
                        *g = true;
                    }
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
        let repairing = to_repair.remove(*day);
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
        cache.days.remove(day);
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
    report.oldest_prunable = entries
        .iter()
        .zip(&gone)
        .filter(|(_, gone)| !**gone)
        .map(|(e, _)| e.1)
        .min();
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
    /// The dataset collector's settings (`collect`): every tick asks the collector thread to
    /// apply the caps when the folder exists, whether collection is on or not; it walks the folder
    /// only when a cap may have work ([`crate::collect::store::Store::prune_if_due`]).
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
        && let Err(e) = prune_saved_images(dir, cfg.max_age_days, cfg.max_mb, shutdown)
    {
        tracing::warn!("saved-image retention: {e:#}");
    }
    if shutdown.is_cancelled() {
        return;
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

/// Run [`prune_saved_images`] and the free-space check every [`SWEEP_INTERVAL`] (first after
/// [`FIRST_SWEEP_DELAY`]), and the storage sweep once a day, until `shutdown`. `settings` is called on every tick, so config
/// changes apply at once. Between ticks the thread is parked ([`crate::sync::nap`]), so it wakes
/// once per interval. The thread runs below normal priority
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
            // `nap` parks the thread for the whole interval (no wake-up in between) and returns
            // false at once when `shutdown` or a parent token is cancelled.
            while crate::sync::nap(delay, &shutdown) && !shutdown.is_cancelled() {
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
    fn paths(v: Vec<&(PathBuf, SystemTime, u64)>) -> Vec<PathBuf> {
        v.into_iter().map(|e| e.0.clone()).collect()
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
        let out = paths(plan_prune(
            &v,
            at(1000),
            Some(Duration::from_secs(500)),
            None,
        ));
        assert_eq!(out, vec![PathBuf::from("old1"), PathBuf::from("old2")]);
    }

    #[test]
    fn deletes_by_size_until_under_cap() {
        let v = [e("c", 3, 40), e("a", 1, 40), e("b", 2, 40)];
        let out = paths(plan_prune(&v, at(10), None, Some(80)));
        assert_eq!(out, vec![PathBuf::from("a")]);
        let out = paths(plan_prune(&v, at(10), None, Some(30)));
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
        let out = paths(plan_prune(
            &v,
            at(1000),
            Some(Duration::from_secs(500)),
            Some(100),
        ));
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
        let r = prune_saved_images(p, 1, 0, &CancellationToken::new()).unwrap();
        assert_eq!(r.deleted, 0);
        assert_eq!(r.remaining, 3);
        // Size cap of 1 MB leaves the newest image; text files and the subfolder stay.
        let r = prune_saved_images(p, 0, 1, &CancellationToken::new()).unwrap();
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
        let r = prune_saved_images(d.path(), 0, 1, &CancellationToken::new()).unwrap();
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
        let r =
            prune_saved_images(&d.path().join("nope"), 1, 1, &CancellationToken::new()).unwrap();
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

    // --- Audit 2026-11 phase 6: prune cost, cancel, idle walks ---

    /// A day folder's label file marking the frame `<day>/<stem>.jpg` reviewed (human work).
    fn review(day: &Path, stem: &str) {
        use crate::collect::labels::{self, LabelFile, Status};
        labels::write(
            &labels::path_of_frame(&day.join(format!("{stem}.jpg"))),
            &LabelFile {
                status: Status::Reviewed,
                ..LabelFile::default()
            },
        )
        .unwrap();
    }

    /// Past [`RACY_WINDOW`] of everything the test just wrote, so its days may be cached.
    fn later() -> SystemTime {
        SystemTime::now() + Duration::from_secs(10)
    }

    /// RA-104 / QA-031: each deletion's freed bytes are its own entry's size, on mixed sizes.
    #[test]
    fn freed_bytes_are_the_sum_of_the_doomed_sizes() {
        let v = [
            e("a", 1, 7),
            e("b", 2, 300),
            e("c", 3, 11),
            e("d", 4, 5000),
            e("e", 5, 42),
            e("f", 6, 1),
        ];
        let doomed: u64 = plan_prune(&v, at(10), None, Some(5050))
            .iter()
            .map(|e| e.2)
            .sum();
        assert_eq!(doomed, 7 + 300 + 11);
        let r = delete_planned(
            &v,
            at(10),
            None,
            Some(5050),
            &CancellationToken::new(),
            &mut |_| Ok(()),
        );
        assert_eq!((r.deleted, r.freed_bytes, r.remaining), (3, doomed, 3));

        // The dataset: frames of mixed sizes, image + sidecar each.
        let d = Tmp::new();
        let p = d.path();
        for (stem, size, age) in [("a", 100, 9), ("b", 250, 8), ("c", 37, 7), ("d", 400, 6)] {
            dataset_frame(p, "ir/cam/2026-01-01", stem, size, age);
        }
        // 1574 bytes in all; a cap of 850 takes a, b and c (774 bytes), d's 800 stay.
        let r = prune_dataset(p, 0, 850).unwrap();
        assert_eq!((r.deleted, r.freed_bytes), (3, 2 * (100 + 250 + 37)));
        assert_eq!(r.usage.prunable_bytes, 800);
        assert!(p.join("ir/cam/2026-01-01/d.jpg").is_file());
    }

    /// RA-104: 200k files with 50k doomed is planned and applied at once (each doomed file looked
    /// up again in the whole list took about 1.25e9 path comparisons: minutes).
    #[test]
    fn a_large_prune_takes_no_quadratic_time() {
        let n: u64 = 200_000;
        let v: Vec<(PathBuf, SystemTime, u64)> = (0..n)
            .map(|i| {
                (
                    PathBuf::from(format!("saved/cam{}/{i:08}.jpg", i % 7)),
                    at(i),
                    10,
                )
            })
            .collect();
        let started = Instant::now();
        let mut removed = 0u64;
        let r = delete_planned(
            &v,
            at(n),
            None,
            Some(150_000 * 10),
            &CancellationToken::new(),
            &mut |_| {
                removed += 1;
                Ok(())
            },
        );
        let took = started.elapsed();
        assert_eq!(
            (r.deleted, r.freed_bytes, removed),
            (50_000, 500_000, 50_000)
        );
        assert!(took < Duration::from_secs(5), "took {took:?}");
    }

    /// RA-104: a cancel stops the saved-image prune between two deletions.
    #[test]
    fn a_cancel_stops_the_saved_image_prune_between_deletions() {
        let v: Vec<_> = (0..10).map(|i| e(&format!("{i}.jpg"), i, 1)).collect();
        let stop = CancellationToken::new();
        let mut removed = 0;
        let r = delete_planned(&v, at(100), None, Some(1), &stop, &mut |_| {
            removed += 1;
            if removed == 3 {
                stop.cancel();
            }
            Ok(())
        });
        assert!(r.stopped);
        assert_eq!((r.deleted, r.remaining, removed), (3, 7, 3));

        // Through the folder: a cancelled token deletes nothing.
        let d = Tmp::new();
        std::fs::write(d.path().join("a.jpg"), vec![0u8; 2 * 1024 * 1024]).unwrap();
        let r = prune_saved_images(d.path(), 0, 1, &stop).unwrap();
        assert!(r.stopped && r.deleted == 0);
        assert!(d.path().join("a.jpg").is_file());
    }

    /// RA-115: the sweeper parks for its whole (30 s) first delay and still ends within a second
    /// of a cancel, a parent token's included; nothing ran meanwhile.
    #[test]
    fn the_sweeper_stops_within_a_second_of_a_cancel() {
        let d = Tmp::new();
        let root = d.path().to_path_buf();
        let ticks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = ticks.clone();
        let parent = CancellationToken::new();
        let h = spawn_sweeper(
            Arc::new(move || {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                SweeperConfig {
                    save_image_dir: None,
                    max_age_days: 0,
                    max_mb: 0,
                    data_root: root.clone(),
                    min_free_mb: 0,
                    dataset: None,
                    storage: None,
                }
            }),
            parent.child_token(),
        );
        std::thread::sleep(Duration::from_millis(100));
        let cancelled = Instant::now();
        parent.cancel();
        assert!(
            join_within(h, cancelled + Duration::from_secs(1), "retention").is_ok(),
            "the sweeper was still napping a second after the cancel"
        );
        assert_eq!(ticks.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// RA-105: a walk takes the day folders that look unchanged from its cache, and reads again a
    /// day that changed (a review renames a label file into it) or lost frames to a prune.
    #[test]
    fn a_walk_reads_again_only_the_days_that_changed() {
        let d = Tmp::new();
        let p = d.path();
        let (day1, day2) = (p.join("ir/cam/2026-01-01"), p.join("ir/cam/2026-01-02"));
        dataset_frame(p, "ir/cam/2026-01-01", "a", 100, 5);
        dataset_frame(p, "ir/cam/2026-01-01", "b", 100, 5);
        dataset_frame(p, "ir/cam/2026-01-02", "c", 100, 4);
        let mut cache = DayCache::default();
        let w = walk_dataset(p, later(), true, &|| false, &mut cache);
        assert_eq!(
            (cache.days_walked(), cache.len(), w.entries.len()),
            (2, 2, 3)
        );
        let w = walk_dataset(p, later(), true, &|| false, &mut cache);
        assert_eq!(
            (cache.days_walked(), w.entries.len(), w.usage.prunable_bytes),
            (2, 3, 600),
            "nothing changed: nothing read"
        );
        // File times are coarse (NTFS about 15 ms, Linux one tick): step past them.
        std::thread::sleep(Duration::from_millis(50));
        review(&day1, "a");
        let w = walk_dataset(p, later(), true, &|| false, &mut cache);
        assert_eq!(
            cache.days_walked(),
            3,
            "only the reviewed day is read again"
        );
        assert_eq!((w.entries.len(), w.usage.protected_frames), (2, 1));
        // A walk as of now: the day changed within the racy window is read and not kept.
        let before = cache.days_walked();
        let w = walk_dataset(p, SystemTime::now(), true, &|| false, &mut cache);
        assert!(cache.days_walked() > before);
        assert_eq!(w.usage.protected_frames, 1);
        // A prune drops the day it deletes from; the other day stays cached.
        walk_dataset(p, later(), true, &|| false, &mut cache);
        let before = cache.days_walked();
        let r = prune_dataset_at(p, 0, 250, &|| false, &mut cache, later()).unwrap();
        assert_eq!(
            cache.days_walked(),
            before,
            "the pass's walk came from the cache"
        );
        assert_eq!(r.deleted, 1);
        assert!(!day1.join("b.jpg").exists() && day2.join("c.jpg").is_file());
        assert!(r.oldest_prunable.is_some(), "c is left");
        assert!(!cache.days.contains_key(&day1) && cache.days.contains_key(&day2));
        std::thread::sleep(Duration::from_millis(50));
        let w = walk_dataset(p, later(), true, &|| false, &mut cache);
        assert_eq!(cache.days_walked(), before + 1);
        assert_eq!(w.entries.len(), 1);
    }

    /// RA-105: a cache that missed a review (a file system whose folder times do not move) costs
    /// no reviewed frame: each frame's label file is read again right before it goes.
    #[test]
    fn a_stale_day_cache_never_costs_a_reviewed_frame() {
        let d = Tmp::new();
        let p = d.path();
        let day = p.join("ir/cam/2026-01-01");
        dataset_frame(p, "ir/cam/2026-01-01", "a", 100, 5);
        dataset_frame(p, "ir/cam/2026-01-01", "b", 100, 4);
        let mut cache = DayCache::default();
        walk_dataset(p, later(), true, &|| false, &mut cache);
        review(&day, "a");
        // The cache believes the day unchanged.
        let key = DayKey::of(&day, later()).unwrap();
        cache.days.get_mut(&day).unwrap().0 = key;
        let walked = cache.days_walked();
        let r = prune_dataset_at(p, 0, 1, &|| false, &mut cache, later()).unwrap();
        assert_eq!(cache.days_walked(), walked, "the stale entry was used");
        assert!(day.join("a.jpg").is_file(), "the reviewed frame stays");
        assert!(!day.join("b.jpg").exists());
        assert_eq!((r.deleted, r.usage.protected_frames), (1, 1));
        assert_eq!(r.oldest_prunable, None);
    }

    /// The oldest deletable frame left is reported, so the collector knows when the age cap has
    /// work again without walking.
    #[test]
    fn a_prune_reports_the_oldest_deletable_frame_left() {
        let d = Tmp::new();
        let p = d.path();
        dataset_frame(p, "ir/cam/2026-01-01", "old", 100, 100);
        dataset_frame(p, "ir/cam/2026-02-01", "mid", 100, 60);
        dataset_frame(p, "ir/cam/2026-03-01", "new", 100, 10);
        review(&p.join("ir/cam/2026-02-01"), "mid");
        let r = prune_dataset(p, 90, 0).unwrap();
        assert_eq!(r.deleted, 1);
        let oldest = r.oldest_prunable.unwrap();
        let age = SystemTime::now().duration_since(oldest).unwrap();
        assert!(
            age > Duration::from_secs(9 * SECS_PER_DAY)
                && age < Duration::from_secs(11 * SECS_PER_DAY),
            "{age:?}: the reviewed frame is not deletable, the deleted one is gone"
        );
        let s = dataset_survey(p, &mut DayCache::default());
        assert_eq!(s.oldest_prunable, Some(oldest));
        assert_eq!(s.usage, r.usage);
    }
}
