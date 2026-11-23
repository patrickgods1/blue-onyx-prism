//! Camera days held out for good: `<review dir>/heldout-days.json` ([`FILE`], docs/DATASET.md
//! "Export"). The split of an export ([`super::export::assign_splits`]) hashes each camera day, so
//! by chance a camera can have no held-out day at all, and a benchmark of a custom model then never
//! sees it. This list makes the held-out set grow only: a day once held out stays held out in every
//! later export of the default split, and a training run holds it out even from an older export
//! that still had it in train (`training::dataset::Options::frozen_heldout`).
//!
//! - [`decide`] (pure): the export's split is today's hash split plus every listed day; a camera
//!   with fewer than [`MIN_HELDOUT_DAYS_PER_CAMERA`] held-out days gets more of its days, those no
//!   training run has seen (`seen`, the days of every run's `train.txt` / `valid.txt`) first by
//!   lowest hash, never its last training day of the export. What it cannot reach is reported
//!   ([`Short`]).
//! - Only an export of the default split (default `holdout_fraction`, no named camera or day) adds
//!   days: its hash picks and the camera-minimum top-ups. Other exports use their own split plus the
//!   listed days and write nothing. A refine fold export or any other selection without a
//!   [`HeldoutPolicy`] never reads or writes the list.
//! - The file only grows. A file of a later version refuses the export as "newer"; a file that does
//!   not parse refuses it as "damaged" and is never moved aside or replaced: a shorter list would
//!   let held-out days flip to training.
//! - I/O ([`load`], [`update`], [`apply`]): one cross-process lock, [`LOCK_FILE`] at the review
//!   folder's root (fs4, after this process's mutex, like the review queue's), waited for up to
//!   [`LOCK_WAIT`], held from the read to the rename of one update (the decision only, never a walk
//!   of the frames), never taken under the label lock. A lock that cannot work (a read-only share)
//!   refuses only an update that has days to add; a lock still held after the wait refuses.

use super::export::{
    DEFAULT_HOLDOUT_FRACTION, Group, Split, SplitConfig, assign_splits, group_hash,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// The list, at the review folder's root.
pub const FILE: &str = "heldout-days.json";
/// Its cross-process lock (zero bytes, never deleted).
pub const LOCK_FILE: &str = ".heldout-days.lock";
/// Version written; a later one is refused ([`load`]).
pub const VERSION: u32 = 1;
/// Fewest held-out camera days wanted per camera of an export: one day of a camera says little
/// (a single day's weather and light), two give its score a second, unrelated look.
pub const MIN_HELDOUT_DAYS_PER_CAMERA: usize = 2;
/// Longest wait for [`LOCK_FILE`]. An update holds it for milliseconds; running out means a
/// stopped process, and the export is refused rather than decided on a list that may be changing.
pub const LOCK_WAIT: Duration = Duration::from_secs(30);

/// Why a day is listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reason {
    /// The default split's hash held it out ([`assign_splits`]).
    Hash,
    /// Added so that its camera reaches [`MIN_HELDOUT_DAYS_PER_CAMERA`].
    CameraMinimum,
}

/// One listed camera day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeldoutDay {
    pub camera: String,
    pub date: String,
    /// When it was listed (UTC, RFC 3339).
    pub added: String,
    pub reason: Reason,
}

impl HeldoutDay {
    pub fn group(&self) -> Group {
        (self.camera.clone(), self.date.clone())
    }
}

/// [`FILE`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeldoutFile {
    pub version: u32,
    /// In the order they were listed.
    pub days: Vec<HeldoutDay>,
}

impl Default for HeldoutFile {
    fn default() -> Self {
        Self {
            version: VERSION,
            days: Vec::new(),
        }
    }
}

impl HeldoutFile {
    /// The listed camera days.
    pub fn groups(&self) -> BTreeSet<Group> {
        self.days.iter().map(HeldoutDay::group).collect()
    }

    /// The `camera/day` keys of the listed days (`training::dataset::Options::frozen_heldout`).
    pub fn keys(&self) -> BTreeSet<String> {
        self.days
            .iter()
            .map(|d| format!("{}/{}", d.camera, d.date))
            .collect()
    }
}

/// How an export uses the list (`ExportRequest::heldout`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeldoutPolicy {
    /// The camera days every training run trained or validated on (`training::seen_days`); None =
    /// a run's lists did not read, so no camera-minimum day is added (one may have been trained on).
    pub seen: Option<BTreeSet<Group>>,
}

/// A camera of the export still under [`MIN_HELDOUT_DAYS_PER_CAMERA`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Short {
    pub camera: String,
    /// Its held-out days in this export.
    pub held: usize,
    pub wanted: usize,
    /// Why no more could be held out, with the next action.
    pub why: String,
}

/// What [`decide`] chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// The split of every group of the export.
    pub splits: BTreeMap<Group, Split>,
    /// Days to list, sorted: held out by this export's default split and not listed yet, or added
    /// for the camera minimum. Empty for an export of other split settings.
    pub added: Vec<(Group, Reason)>,
    /// Cameras under the minimum (default split only).
    pub short: Vec<Short>,
}

/// Whether `cfg` is the default split: the default fraction, no camera or day named. Only such an
/// export adds days to the list.
pub fn is_default_split(cfg: &SplitConfig) -> bool {
    cfg.holdout_cameras.is_empty()
        && cfg.holdout_days.is_empty()
        && (cfg.holdout_fraction - DEFAULT_HOLDOUT_FRACTION).abs() < f32::EPSILON
}

/// The split of `groups` (see the module doc): [`assign_splits`] with `cfg`, every `listed` day
/// held out, and for the default split the camera-minimum top-ups from days not in `seen`.
///
/// The "one group each side" guard of the hash split never moves a listed day to train: when the
/// listed days leave nothing in train, the highest-hash day that is not listed trains. Pure.
pub fn decide(
    groups: &BTreeSet<Group>,
    cfg: &SplitConfig,
    listed: &BTreeSet<Group>,
    seen: Option<&BTreeSet<Group>>,
) -> Decision {
    let mut splits = assign_splits(groups, cfg);
    for g in groups.intersection(listed) {
        splits.insert(g.clone(), Split::Val);
    }
    let explicit = !cfg.holdout_cameras.is_empty() || !cfg.holdout_days.is_empty();
    if !explicit
        && cfg.holdout_fraction > 0.0
        && groups.len() >= 2
        && !splits.values().any(|s| *s == Split::Train)
        && let Some(g) = groups
            .iter()
            .filter(|g| !listed.contains(*g))
            .max_by_key(|g| (group_hash(g), (*g).clone()))
    {
        splits.insert(g.clone(), Split::Train);
    }
    let default = is_default_split(cfg);
    if !default {
        return Decision {
            splits,
            added: Vec::new(),
            short: Vec::new(),
        };
    }
    let mut topped: BTreeSet<Group> = BTreeSet::new();
    let mut short = Vec::new();
    let cameras: BTreeSet<&String> = groups.iter().map(|g| &g.0).collect();
    for camera in cameras {
        let days: Vec<&Group> = groups.iter().filter(|g| &g.0 == camera).collect();
        let is = |g: &Group, s: Split| splits.get(g) == Some(&s);
        let mut held = days.iter().filter(|g| is(g, Split::Val)).count();
        let mut train: Vec<&Group> = days
            .iter()
            .copied()
            .filter(|g| is(g, Split::Train))
            .collect();
        train.sort_by_key(|g| (group_hash(g), (*g).clone()));
        if let Some(seen) = seen {
            for g in &train {
                if held >= MIN_HELDOUT_DAYS_PER_CAMERA {
                    break;
                }
                // Never the camera's last training day of the export.
                if days.len() - held <= 1 {
                    break;
                }
                if seen.contains(*g) {
                    continue;
                }
                topped.insert((*g).clone());
                splits.insert((*g).clone(), Split::Val);
                held += 1;
            }
        }
        if held >= MIN_HELDOUT_DAYS_PER_CAMERA {
            continue;
        }
        let left: Vec<&&Group> = train.iter().filter(|g| !topped.contains(**g)).collect();
        let why = match seen {
            None => "the frame lists of past training runs could not all be read, so no day was \
                     added (one may have been trained on); check the training folder and export \
                     again"
                .to_string(),
            Some(_) if left.is_empty() => format!(
                "only {} reviewed {camera} day(s), all held out; review more days of {camera}",
                days.len()
            ),
            Some(seen) if left.iter().all(|g| seen.contains(**g)) => {
                let other = if held > 0 { "other " } else { "" };
                format!("every {other}reviewed {camera} day was trained on; review a newer day")
            }
            Some(_) => format!(
                "its one training day of this export stays in training; review another day of \
                 {camera}"
            ),
        };
        short.push(Short {
            camera: camera.clone(),
            held,
            wanted: MIN_HELDOUT_DAYS_PER_CAMERA,
            why,
        });
    }
    let added = splits
        .iter()
        .filter(|(g, s)| **s == Split::Val && !listed.contains(*g))
        .map(|(g, _)| {
            let reason = if topped.contains(g) {
                Reason::CameraMinimum
            } else {
                Reason::Hash
            };
            (g.clone(), reason)
        })
        .collect();
    Decision {
        splits,
        added,
        short,
    }
}

/// The list in the review folder `dir`: an empty one when there is no file. A file of a later
/// version is refused as newer, one that does not parse as damaged, one that cannot be read as
/// such; none is touched.
pub fn load(dir: &Path) -> Result<HeldoutFile> {
    let path = dir.join(FILE);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HeldoutFile::default()),
        Err(e) => bail!(
            "{} does not read ({e}): it lists the camera days held out for good, so nothing is \
             exported without it; check the folder's permissions and the drive, then export again",
            path.display()
        ),
    };
    let damaged = |why: String| {
        anyhow::anyhow!(
            "{} is damaged ({why}): it lists the camera days held out for good, and a shorter \
             list would move held-out days to training, so it is never replaced or moved aside; \
             restore it from a backup or fix it by hand ({{\"version\":1,\"days\":[...]}}), then \
             export again",
            path.display()
        )
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| damaged(e.to_string()))?;
    let version = value.get("version").and_then(serde_json::Value::as_u64);
    match version {
        Some(v) if v > u64::from(VERSION) => bail!(
            "{} is newer (version {v}) than this build reads ({VERSION}): a newer Blue Onyx Prism \
             wrote it; update this one to export, the file stays as it is",
            path.display()
        ),
        Some(v) if v == u64::from(VERSION) => {}
        _ => return Err(damaged("no version 1".into())),
    }
    let mut file: HeldoutFile =
        serde_json::from_value(value).map_err(|e| damaged(e.to_string()))?;
    // A day listed twice counts once (its first listing).
    let mut once = BTreeSet::new();
    file.days.retain(|d| once.insert(d.group()));
    Ok(file)
}

/// Serializes this process's updates. Taken before [`LOCK_FILE`].
static HELDOUT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// What [`folder_lock`] got.
enum FolderLock {
    Held(std::fs::File),
    /// Held by another holder for the whole wait.
    Busy,
    /// It cannot work here (will not open or lock): why.
    Broken(String),
}

impl Drop for FolderLock {
    fn drop(&mut self) {
        if let FolderLock::Held(f) = self {
            let _ = fs4::FileExt::unlock(f);
        }
    }
}

fn folder_lock(dir: &Path, wait: Duration) -> FolderLock {
    let path = dir.join(LOCK_FILE);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path);
    let file = match file {
        Ok(f) => f,
        Err(e) => return FolderLock::Broken(format!("{} cannot be opened ({e})", path.display())),
    };
    let waited = crate::fsx::lock_wait_with(
        &file,
        crate::fsx::LockKind::Exclusive,
        crate::fsx::deadline_after(wait),
        crate::fsx::Pauses::fixed(Duration::from_millis(2)),
        || true,
    );
    match waited {
        Ok(crate::fsx::Waited::Held) => FolderLock::Held(file),
        Ok(crate::fsx::Waited::TimedOut | crate::fsx::Waited::Stopped) => FolderLock::Busy,
        Err(e) => FolderLock::Broken(format!("{} cannot be locked ({e})", path.display())),
    }
}

/// Read-modify-write of the list of the review folder `dir` under its lock: `f` gets the list and
/// returns the days to add (any already listed are skipped) and its answer. Writes only when a day
/// is new (temporary file + rename). Returns the list as written (or read) and `f`'s answer.
pub fn update<T>(
    dir: &Path,
    f: impl FnOnce(&HeldoutFile) -> Result<(Vec<HeldoutDay>, T)>,
) -> Result<(HeldoutFile, T)> {
    update_with(dir, LOCK_WAIT, f)
}

fn update_with<T>(
    dir: &Path,
    wait: Duration,
    f: impl FnOnce(&HeldoutFile) -> Result<(Vec<HeldoutDay>, T)>,
) -> Result<(HeldoutFile, T)> {
    let _process = HELDOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let lock = folder_lock(dir, wait);
    if matches!(lock, FolderLock::Busy) {
        bail!(
            "{} stayed locked for {}s: another export (or a stopped process) is updating the \
             camera days held out for good; export again in a moment",
            dir.join(LOCK_FILE).display(),
            wait.as_secs()
        );
    }
    let mut file = load(dir)?;
    let (days, answer) = f(&file)?;
    let listed = file.groups();
    let mut new: Vec<HeldoutDay> = Vec::new();
    for d in days {
        if !listed.contains(&d.group()) && !new.iter().any(|n| n.group() == d.group()) {
            new.push(d);
        }
    }
    if new.is_empty() {
        return Ok((file, answer));
    }
    if let FolderLock::Broken(why) = &lock {
        bail!(
            "{why}: the camera days held out for good cannot be added to safely (a read-only or \
             network folder?); nothing was exported. Export with other split settings, or make \
             the review folder writable"
        );
    }
    file.version = VERSION;
    file.days.extend(new);
    let mut text = serde_json::to_vec_pretty(&file)?;
    text.push(b'\n');
    let path = dir.join(FILE);
    crate::collect::layout::write_atomic(&path, &text)
        .with_context(|| format!("writing {}", path.display()))?;
    drop(lock);
    Ok((file, answer))
}

/// What an export did with the list (`ExportReport::heldout`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HeldoutReport {
    /// Camera days in the list after this export.
    pub listed: usize,
    /// Days this export added.
    pub added: Vec<HeldoutDay>,
    /// Cameras of the export still under [`MIN_HELDOUT_DAYS_PER_CAMERA`].
    pub short: Vec<Short>,
    /// The export's split settings are not the default: the listed days were held out, nothing
    /// was added.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub custom_split: bool,
}

/// [`decide`] on the list of `dir` under its lock, adding what it chose ([`update`]): the split of
/// every group and the report.
pub fn apply(
    dir: &Path,
    groups: &BTreeSet<Group>,
    cfg: &SplitConfig,
    policy: &HeldoutPolicy,
    now: SystemTime,
) -> Result<(BTreeMap<Group, Split>, HeldoutReport)> {
    apply_with(dir, groups, cfg, policy, now, LOCK_WAIT)
}

fn apply_with(
    dir: &Path,
    groups: &BTreeSet<Group>,
    cfg: &SplitConfig,
    policy: &HeldoutPolicy,
    now: SystemTime,
    wait: Duration,
) -> Result<(BTreeMap<Group, Split>, HeldoutReport)> {
    let stamp = crate::collect::utc_rfc3339(now);
    let (file, (decision, added)) = update_with(dir, wait, |file| {
        let d = decide(groups, cfg, &file.groups(), policy.seen.as_ref());
        let added: Vec<HeldoutDay> = d
            .added
            .iter()
            .map(|((camera, date), reason)| HeldoutDay {
                camera: camera.clone(),
                date: date.clone(),
                added: stamp.clone(),
                reason: *reason,
            })
            .collect();
        Ok((added.clone(), (d, added)))
    })?;
    let report = HeldoutReport {
        listed: file.days.len(),
        added,
        short: decision.short,
        custom_split: !is_default_split(cfg),
    };
    Ok((decision.splits, report))
}

#[cfg(test)]
mod tests;
