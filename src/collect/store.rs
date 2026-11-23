//! The collector thread's state: bucket selection with per-camera rate limits and sampling,
//! near-duplicate dedup per camera, the atomic writes of image + sidecar + index line, the size /
//! age caps (through [`crate::retention::prune_dataset`]) and the disk guard that stops collection
//! on low free space or a full disk and re-arms it once space is back. Single-threaded: only the
//! collector thread owns a [`Store`], and it runs both the collector's writes and retention, so
//! those two never meet. It is not the only writer of the dataset folder: an import
//! ([`crate::training::import::write`], on a job thread or in a `train import` process) adds frames
//! and index lines to the same day folders, so index lines go through [`layout::append_index`]
//! (the dataset's cross-process index lock, which retention's rewrite also takes).
//!
//! Dedup (`collect.dedup_mode`, [`DedupMode`]): each saved frame that was not itself a near
//! duplicate is remembered as a reference: its time, dHash, scene and, in `both` mode, its block
//! thumbnail ([`blockdiff::Thumb`], from the luma decoded for the dHash: one decode). A new frame
//! is a near duplicate when a reference of the same camera within [`DEDUP_WINDOW`] is within
//! `dedup_hamming` dHash bits and, in `both` mode (default), that same reference is of the same
//! scene, still has its thumbnail and the block test ([`blockdiff::Params::COLLECTOR`]) finds no
//! change against it. The offline evaluation (`tests/it/dedup_eval.rs`, docs/DATASET.md) found the
//! dHash test alone missed 56% of labelled object changes on 1,009 reviewed pairs (76% in imported
//! footage); requiring both missed none, and confirmed no drop on those frames either (rain,
//! insects, IR flicker and foliage differ by more than a small animal), so volume is held by the
//! rate limits and the storm cap below, not by the dedup. Thumbnails are capped across cameras at
//! [`DEDUP_THUMB_BYTES`]; the oldest lose theirs first.
//!
//! "Never drops a frame the old rule kept" holds per decision, against the same references: `both`
//! drops a frame only when `dhash` would too, and a lost thumbnail only takes a drop away. It does
//! not hold over a sequence: a frame `both` keeps restarts its buckets' interval clocks and becomes
//! a reference, so a later frame can be dropped (by a rate limit, or as a near duplicate of it)
//! that `dhash` mode, which had dropped the earlier one, would have kept.
//!
//! Storm cap: every dHash look-alike counts toward [`DEDUP_SKIPS_PER_CAMERA`], as the duplicates of
//! `dhash` mode do. A look-alike the block test calls different is kept for a `positive` or
//! `empty` bucket; one whose passing buckets all skip the dedup ([`Bucket::skips_dedup`]) is held
//! to the cap, so that cap is never looser than in `dhash` mode. The rate limits and the night
//! interval come first. A kept frame's sidecar records what the dedup saw ([`DedupInfo`]).

use super::blockdiff::{self, Thumb};
use super::config::DedupMode;
use super::health::LastError;
use super::layout::{self, Summary};
use super::metrics::{Dropped, METRICS};
use super::record::{Bucket, DedupInfo, IndexLine, RECORD_VERSION};
use super::{CollectSettings, Frame, Status, dhash};
use crate::image_stats::Scene;
use crate::retention::DatasetUsage;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn};

/// Frames of one camera compared for near duplicates: those saved within this window.
pub const DEDUP_WINDOW: Duration = Duration::from_secs(10 * 60);
/// At most this many references (recent saved frames) per camera.
pub const DEDUP_KEEP: usize = 32;
/// Most bytes of reference thumbnails held across all cameras: 8 MiB, 910 thumbnails of
/// [`blockdiff::THUMB_CELLS`] (9,216) bytes, a full [`DEDUP_KEEP`] for 28 busy cameras. Past it the
/// oldest thumbnails (any camera) are dropped; their references keep their dHash, so `dhash` mode
/// is unchanged and in `both` mode they can no longer confirm a drop.
pub const DEDUP_THUMB_BYTES: usize = 8 * 1024 * 1024;
/// At most this many near duplicates of one camera are saved in [`DEDUP_SKIP_WINDOW`] because
/// their bucket skips the dedup ([`Bucket::skips_dedup`]); past it they are dropped as duplicates.
/// Those buckets have no minimum interval by default, so in a storm of look-alike frames near the
/// threshold this is what stops them: at most 14,400 frames a day per camera, around 4 GB at
/// 300 kB a frame, and `max_gb` still applies. Frames that do not look like a recent one are not
/// limited by it, as before; set a bucket's interval to limit those.
pub const DEDUP_SKIPS_PER_CAMERA: usize = 10;
/// See [`DEDUP_SKIPS_PER_CAMERA`].
pub const DEDUP_SKIP_WINDOW: Duration = Duration::from_secs(60);
/// Free space is checked at most this often.
pub const FREE_CHECK_INTERVAL: Duration = Duration::from_secs(30);
/// A disk stop re-arms on its own once this much more than the stop point
/// ([`crate::retention::MIN_FREE_MB`]) is free, so collection does not flap around the stop point.
pub const REARM_MARGIN_MB: u64 = 256;
/// A size-cap prune after a write runs at most this often.
pub const PRUNE_MIN_INTERVAL: Duration = Duration::from_secs(30);
/// An I/O failure is warned about at most this often.
const WARN_INTERVAL: Duration = Duration::from_secs(10 * 60);
/// A camera that sent no frame for this long is forgotten (its rate limits and recent hashes),
/// unless a bucket's `per_camera_min_interval_s` or night / IR interval
/// ([`BucketConfig::night_interval`]) is longer: then that interval, so forgetting never lets a
/// frame in early.
pub const CAMERA_IDLE: Duration = Duration::from_secs(24 * 60 * 60);
/// Most cameras remembered; a new one past it makes the one silent the longest forgotten. The
/// camera names come from upload file names, so a client sending random names must not grow the
/// collector's memory. Blue Iris installs have tens of cameras.
pub const MAX_CAMERAS: usize = 1024;
/// Idle cameras are looked for at most this often (frame time).
const SWEEP_INTERVAL: Duration = Duration::from_secs(10 * 60);

/// What happened to one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handled {
    /// Saved under these buckets (relative image path).
    Saved {
        rel: String,
        buckets: Vec<Bucket>,
    },
    /// It matched no enabled bucket (not counted as a drop).
    NoBucket,
    /// Collection is off in its settings.
    Off,
    Dropped(Dropped),
}

/// Free-space probe: the bytes available on the folder's file system, None when unknown (no stop
/// and no re-arm then).
pub type FreeProbe = Box<dyn Fn(&Path) -> Option<u64> + Send>;

const BYTES_PER_MB: u64 = 1024 * 1024;

/// The free space at which a disk stop re-arms.
fn rearm_bytes() -> u64 {
    (crate::retention::MIN_FREE_MB + REARM_MARGIN_MB) * BYTES_PER_MB
}

/// Why collection is off on its own.
#[derive(Debug, Clone)]
struct Disabled {
    reason: String,
    /// The settings when it happened: a changed `collect` section re-arms, like Resume and free
    /// space coming back ([`Store::poll_rearm`]).
    settings: CollectSettings,
}

pub struct Store {
    shared: Arc<Mutex<Status>>,
    /// Folder the summary describes.
    dir: Option<PathBuf>,
    summary: Summary,
    /// How the folder divides up; `prunable_bytes` is what the size cap is compared with. A review
    /// meanwhile makes it too high until the next prune (10 minutes at most), never too low.
    usage: DatasetUsage,
    /// (camera, bucket) -> when its last frame was saved.
    last: HashMap<(String, Bucket), SystemTime>,
    /// camera -> its references, oldest first.
    recent: HashMap<String, VecDeque<Reference>>,
    /// Cap of the reference thumbnails' bytes ([`DEDUP_THUMB_BYTES`]; tests lower it).
    thumb_cap: usize,
    /// camera -> when near duplicates were saved for a bucket that skips the dedup.
    skips: HashMap<String, VecDeque<SystemTime>>,
    /// camera -> its latest frame that got as far as the per-camera limits. Every camera in
    /// `last`, `recent` and `skips` is here, so this bounds them ([`CAMERA_IDLE`],
    /// [`MAX_CAMERAS`]).
    seen: HashMap<String, SystemTime>,
    /// Frame time of the last sweep for idle cameras.
    swept: Option<SystemTime>,
    /// When this process last saved a frame.
    last_saved: Option<SystemTime>,
    last_error: Option<LastError>,
    disabled: Option<Disabled>,
    free_checked: Option<Instant>,
    last_prune: Option<Instant>,
    last_warn: Option<Instant>,
    free_probe: FreeProbe,
    /// Tests: make the next writes fail with this error kind.
    #[cfg(test)]
    pub(super) fail_writes: Option<std::io::ErrorKind>,
    /// Tests: the zone of day folders and stems instead of the process-wide one.
    #[cfg(test)]
    pub(super) zone: Option<crate::clock::Zone>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("dir", &self.dir)
            .field("frames", &self.summary.frames)
            .finish_non_exhaustive()
    }
}

/// Uniform in `[0, 1)` from the request id and bucket (FNV-1a), so sampling is deterministic.
pub fn sample(request_id: &str, bucket: Bucket) -> f64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in request_id
        .bytes()
        .chain(std::iter::once(0xff))
        .chain(bucket.as_str().bytes())
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Finalizer (murmur3 fmix64) so similar ids spread over the whole range.
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// How long a camera stays remembered with no frame: [`CAMERA_IDLE`], or the longest bucket
/// interval when that is longer.
fn idle_after(settings: &CollectSettings) -> Duration {
    Bucket::ALL
        .iter()
        .map(|b| {
            let c = settings.buckets.get(*b);
            Duration::from_secs(u64::from(
                c.per_camera_min_interval_s.max(c.night_interval(*b)),
            ))
        })
        .fold(CAMERA_IDLE, Duration::max)
}

/// A saved frame the dedup compares later frames of its camera with.
#[derive(Debug)]
struct Reference {
    at: SystemTime,
    hash: u64,
    scene: Scene,
    /// Its block thumbnail: None in `dhash` mode or once [`DEDUP_THUMB_BYTES`] evicted it.
    thumb: Option<Thumb>,
}

/// The scene of a sidecar's `scene` (`day`, `night`, `ir`; anything else counts as day).
fn scene_of(s: &str) -> Scene {
    match s {
        "night" => Scene::Night,
        "ir" => Scene::Ir,
        _ => Scene::Day,
    }
}

/// What the dedup decided for a frame: a near duplicate, or why not ([`DedupInfo`]).
#[derive(Debug, Clone, PartialEq)]
struct DedupCheck {
    /// Dropped as a near duplicate unless a bucket skips the dedup.
    duplicate: bool,
    /// Some reference in the window is within `dedup_hamming` dHash bits (a dHash look-alike):
    /// counted toward [`DEDUP_SKIPS_PER_CAMERA`] like the duplicates of `dhash` mode.
    look_alike: bool,
    info: DedupInfo,
}

/// The dedup decision for a frame of `scene` with dHash `hash` and thumbnail `thumb` against the
/// camera's references (already limited to the window). Pure. See the module docs for the rule:
/// one reference must pass both tests. References are tried newest first and the first that
/// confirms stops the search; `nearest_hamming` is still the minimum over all of them. The new
/// frame's side of the block test is prepared once ([`blockdiff::Probe`]), on the first
/// look-alike that needs it.
fn dedup_check(
    refs: &VecDeque<Reference>,
    hash: u64,
    thumb: Option<&Thumb>,
    scene: Scene,
    max_hamming: u32,
    mode: DedupMode,
) -> DedupCheck {
    let mut info = DedupInfo {
        mode: mode.as_str().to_string(),
        nearest_hamming: refs.iter().map(|r| dhash::hamming(r.hash, hash)).min(),
        decided_by: if refs.is_empty() {
            DedupInfo::NO_REFERENCE
        } else {
            DedupInfo::DHASH
        }
        .to_string(),
        ..DedupInfo::default()
    };
    let record = |info: &mut DedupInfo, v: &blockdiff::Verdict| {
        info.block_max_z = Some((v.max_z * 10.0).round() / 10.0);
        info.block_largest_blob = Some(u32::try_from(v.largest_blob).unwrap_or(u32::MAX));
    };
    let mut probe: Option<blockdiff::Probe> = None;
    // The block test's closest call: the look-alike with the smallest changed group.
    let mut closest: Option<blockdiff::Verdict> = None;
    let mut look_alike = false;
    for r in refs.iter().rev() {
        if dhash::hamming(r.hash, hash) > max_hamming {
            continue;
        }
        look_alike = true;
        if mode == DedupMode::Dhash {
            return DedupCheck {
                duplicate: true,
                look_alike,
                info,
            };
        }
        let (Some(rt), Some(t)) = (r.thumb.as_ref(), thumb) else {
            continue;
        };
        if r.scene != scene {
            continue;
        }
        let v = probe
            .get_or_insert_with(|| blockdiff::Probe::new(t))
            .verdict(rt, scene, None, &blockdiff::Params::COLLECTOR);
        if !v.different {
            record(&mut info, &v);
            return DedupCheck {
                duplicate: true,
                look_alike,
                info,
            };
        }
        if closest.is_none_or(|c| (v.largest_blob, v.max_z) < (c.largest_blob, c.max_z)) {
            closest = Some(v);
        }
    }
    if look_alike {
        info.decided_by = if closest.is_some() {
            DedupInfo::BLOCKDIFF
        } else {
            DedupInfo::UNCHECKED
        }
        .to_string();
    }
    if let Some(v) = closest {
        record(&mut info, &v);
    }
    DedupCheck {
        duplicate: false,
        look_alike,
        info,
    }
}

/// True for the errors of a full disk or an exhausted quota.
pub fn is_disk_full(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
    )
}

impl Store {
    pub fn new(shared: Arc<Mutex<Status>>) -> Self {
        // Tests that reach the process-wide collector must not depend on this machine's disk.
        #[cfg(test)]
        return Self::with_probe(shared, Box::new(|_| None));
        #[cfg(not(test))]
        Self::with_probe(shared, Box::new(|d: &Path| crate::retention::free_bytes(d)))
    }

    pub fn with_probe(shared: Arc<Mutex<Status>>, free_probe: FreeProbe) -> Self {
        Self {
            shared,
            dir: None,
            summary: Summary::default(),
            usage: DatasetUsage::default(),
            last: HashMap::new(),
            recent: HashMap::new(),
            thumb_cap: DEDUP_THUMB_BYTES,
            skips: HashMap::new(),
            seen: HashMap::new(),
            swept: None,
            last_saved: None,
            last_error: None,
            disabled: None,
            free_checked: None,
            last_prune: None,
            last_warn: None,
            free_probe,
            #[cfg(test)]
            fail_writes: None,
            #[cfg(test)]
            zone: None,
        }
    }

    /// Point the summary at `dir` (scanned from disk on a change).
    fn use_dir(&mut self, dir: &Path) {
        if self.dir.as_deref() == Some(dir) {
            return;
        }
        self.adopt(dir, crate::retention::dataset_usage(dir));
    }

    /// `dir` is the folder in use, of `usage`: scan its summary and publish.
    fn adopt(&mut self, dir: &Path, usage: DatasetUsage) {
        self.summary = Summary::scan(dir);
        self.usage = usage;
        self.dir = Some(dir.to_path_buf());
        self.publish();
    }

    /// Rebuild the summary of `dir` when it is the folder in use (another writer added frames).
    pub fn rescan(&mut self, dir: &Path) {
        if self.dir.as_deref() == Some(dir) {
            self.dir = None;
            self.use_dir(dir);
        }
    }

    fn publish(&self) {
        METRICS.set_bytes(self.summary.bytes);
        METRICS.set_disabled(self.disabled.is_some());
        let mut s = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        s.dir = self.dir.clone();
        s.summary = self.summary.clone();
        s.usage = self.usage;
        s.disabled = self.disabled.as_ref().map(|d| d.reason.clone());
        s.last_saved_at = self.last_saved.map(super::utc_rfc3339);
        s.last_error = self.last_error.clone();
    }

    /// Remember `message` as the last error, for the Collect tab, and warn (throttled).
    pub fn note_error(&mut self, message: &str) {
        self.warn_throttled(message);
        self.last_error = Some(LastError {
            message: message.to_string(),
            at: super::utc_rfc3339(SystemTime::now()),
        });
        self.publish();
    }

    fn disable(&mut self, reason: String, settings: &CollectSettings) {
        warn!(
            dir = %settings.dir.display(),
            "dataset collection stopped: {reason}; it starts again on its own once {} MB is free \
             (free some space or lower collect.max_gb)",
            rearm_bytes() / BYTES_PER_MB
        );
        self.last_error = Some(LastError {
            message: format!("collection stopped: {reason}"),
            at: super::utc_rfc3339(SystemTime::now()),
        });
        self.disabled = Some(Disabled {
            reason,
            settings: settings.clone(),
        });
        self.publish();
    }

    /// Re-arm after a disk stop (the Collect tab's Resume).
    pub fn resume(&mut self) {
        if self.disabled.take().is_some() {
            info!("dataset collection resumed");
            self.free_checked = None;
            self.publish();
        }
    }

    /// Re-arm a disk stop when free space is back: above the stop point by [`REARM_MARGIN_MB`],
    /// checked at most every [`FREE_CHECK_INTERVAL`]. The collector thread calls it between
    /// messages too, so it happens with no frame arriving. A disk whose free space is unknown stays
    /// stopped until Resume or a settings change.
    pub fn poll_rearm(&mut self) {
        let Some(d) = &self.disabled else { return };
        let now = Instant::now();
        if self
            .free_checked
            .is_some_and(|t| now.saturating_duration_since(t) < FREE_CHECK_INTERVAL)
        {
            return;
        }
        let free = (self.free_probe)(&d.settings.dir);
        self.free_checked = Some(now);
        if free.is_some_and(|f| f >= rearm_bytes()) {
            info!("dataset collection resumed: free space is back");
            self.disabled = None;
            self.publish();
        }
    }

    fn warn_throttled(&mut self, msg: &str) {
        let now = Instant::now();
        if self
            .last_warn
            .is_none_or(|t| now.saturating_duration_since(t) >= WARN_INTERVAL)
        {
            warn!("dataset collector: {msg}");
            self.last_warn = Some(now);
        } else {
            debug!("dataset collector: {msg}");
        }
    }

    /// Decide, dedup and save one frame.
    pub fn handle(&mut self, frame: Frame) -> Handled {
        let Frame {
            settings,
            image,
            mut record,
        } = frame;
        if !settings.enabled {
            return Handled::Off;
        }
        if let Some(d) = &self.disabled
            && d.settings != *settings
        {
            info!("dataset collection re-armed: the collect settings changed");
            self.disabled = None;
            self.free_checked = None;
        }
        self.use_dir(&settings.dir);
        self.poll_rearm();
        let at = record.at;
        let matched: Vec<Bucket> =
            super::record::decide(&record.record, settings.buckets.near_margin())
                .into_iter()
                .filter(|b| settings.buckets.get(*b).enabled)
                .collect();
        if matched.is_empty() {
            METRICS.record_no_bucket();
            return Handled::NoBucket;
        }
        if self.disabled.is_some() {
            METRICS.record_drop(Dropped::Disk);
            return Handled::Dropped(Dropped::Disk);
        }
        let camera = record.record.camera.clone();
        self.note_camera(&camera, at, &settings);
        let passing: Vec<Bucket> = matched
            .iter()
            .copied()
            .filter(|b| {
                let c = settings.buckets.get(*b);
                let sampled = f64::from(c.rate) >= 1.0
                    || sample(&record.record.request_id, *b) < f64::from(c.rate);
                let interval =
                    Duration::from_secs(u64::from(c.interval_for(*b, &record.record.scene)));
                let due = self.last.get(&(camera.clone(), *b)).is_none_or(|t| {
                    at.duration_since(*t)
                        .map_or(true, |since| since >= interval)
                });
                sampled && (interval.is_zero() || due)
            })
            .collect();
        if passing.is_empty() {
            METRICS.record_drop(Dropped::Rate);
            return Handled::Dropped(Dropped::Rate);
        }
        let mode = settings.dedup_mode;
        let scene = scene_of(&record.record.scene);
        let (hash, thumb) = match dhash::luma_of_encoded(&image) {
            Ok((luma, w, h)) => (
                dhash::of_luma(&luma, w, h),
                (mode == DedupMode::Both)
                    .then(|| Thumb::of_luma(&luma, w, h))
                    .flatten(),
            ),
            Err(e) => {
                self.note_error(&format!(
                    "could not read a frame Blue Iris sent, so it was not saved: {e:#}; the image \
                     may be damaged or in a format the service cannot decode"
                ));
                METRICS.record_drop(Dropped::Error);
                return Handled::Dropped(Dropped::Error);
            }
        };
        let recent = self.recent.entry(camera.clone()).or_default();
        recent.retain(|r| {
            at.duration_since(r.at)
                .map_or(true, |since| since <= DEDUP_WINDOW)
        });
        let mut check = dedup_check(
            recent,
            hash,
            thumb.as_ref(),
            scene,
            settings.dedup_hamming,
            mode,
        );
        let near_duplicate = check.duplicate;
        // Storm cap: every dHash look-alike counts toward DEDUP_SKIPS_PER_CAMERA, as the
        // duplicates of `dhash` mode do. A look-alike the block test calls different is kept for
        // a `positive` / `empty` bucket (the point of `both`); one whose only passing buckets
        // skip the dedup is held to the cap, so on a rainy IR night the cap is never looser than
        // in `dhash` mode.
        let plain_bucket = passing.iter().any(|b| !b.skips_dedup());
        let counted = check.look_alike && (near_duplicate || !plain_bucket);
        if check.look_alike {
            let skips = self.skips.entry(camera.clone()).or_default();
            skips.retain(|t| {
                at.duration_since(*t)
                    .map_or(true, |since| since < DEDUP_SKIP_WINDOW)
            });
            let full = skips.len() >= DEDUP_SKIPS_PER_CAMERA;
            let drop = if near_duplicate {
                !passing.iter().any(|b| b.skips_dedup()) || full
            } else {
                !plain_bucket && full
            };
            if drop {
                METRICS.record_drop(Dropped::Dup);
                return Handled::Dropped(Dropped::Dup);
            }
            if near_duplicate {
                check.info.decided_by = DedupInfo::SKIPS_DEDUP.to_string();
            }
        }
        if let Some(reason) = self.check_free(&settings) {
            self.disable(reason, &settings);
            METRICS.record_drop(Dropped::Disk);
            return Handled::Dropped(Dropped::Disk);
        }
        record.record.version = RECORD_VERSION;
        record.record.matched = matched;
        record.record.buckets = passing.clone();
        record.record.dhash = dhash::hex(hash);
        record.record.dedup = Some(check.info);
        let rel = match self.write(&settings.dir, &mut record, &image) {
            Ok(rel) => rel,
            Err(e) if is_disk_full(&e) => {
                self.disable(format!("the disk is full ({e})"), &settings);
                METRICS.record_drop(Dropped::Disk);
                return Handled::Dropped(Dropped::Disk);
            }
            Err(e) => {
                self.note_error(&format!(
                    "could not save a frame under {}: {e}; check that the folder exists and that \
                     the service's account can write to it",
                    settings.dir.display()
                ));
                METRICS.record_drop(Dropped::Error);
                return Handled::Dropped(Dropped::Error);
            }
        };
        for b in &passing {
            self.last.insert((camera.clone(), *b), at);
        }
        if counted {
            self.skips.entry(camera.clone()).or_default().push_back(at);
        }
        // A near duplicate saved for a skip bucket is no reference: its look-alike stays the one
        // remembered, so the skip does not extend the window.
        if !near_duplicate {
            let recent = self.recent.entry(camera).or_default();
            recent.push_back(Reference {
                at,
                hash,
                scene,
                thumb,
            });
            while recent.len() > DEDUP_KEEP {
                recent.pop_front();
            }
            self.cap_thumbs();
        }
        self.last_saved = Some(at);
        METRICS.record_frame(&passing);
        if self.usage.prunable_bytes > settings.max_bytes {
            let due = self
                .last_prune
                .is_none_or(|t| t.elapsed() >= PRUNE_MIN_INTERVAL);
            if due {
                self.prune(&settings);
            }
        }
        self.publish();
        Handled::Saved {
            rel,
            buckets: passing,
        }
    }

    /// Cameras remembered (tests).
    #[cfg(test)]
    fn cameras(&self) -> usize {
        debug_assert!(
            self.last.keys().all(|(c, _)| self.seen.contains_key(c))
                && self.recent.keys().all(|c| self.seen.contains_key(c))
                && self.skips.keys().all(|c| self.seen.contains_key(c))
        );
        self.seen.len()
    }

    /// `camera` sent a frame at `at`: remember it, forget cameras idle past [`CAMERA_IDLE`]
    /// (checked every [`SWEEP_INTERVAL`]) and, for a new camera past [`MAX_CAMERAS`], the one
    /// silent the longest.
    fn note_camera(&mut self, camera: &str, at: SystemTime, settings: &CollectSettings) {
        // A clock set back sweeps too (and restarts the interval from there).
        let sweep_due = self.swept.is_none_or(|t| {
            at.duration_since(t)
                .map_or(true, |since| since >= SWEEP_INTERVAL)
        });
        if sweep_due {
            self.swept = Some(at);
            let idle = idle_after(settings);
            let stale: Vec<String> = self
                .seen
                .iter()
                .filter(|(c, t)| {
                    c.as_str() != camera && at.duration_since(**t).is_ok_and(|since| since >= idle)
                })
                .map(|(c, _)| c.clone())
                .collect();
            if !stale.is_empty() {
                debug!(
                    cameras = stale.len(),
                    "dataset collector: forgetting idle cameras"
                );
                self.forget(&stale);
            }
        }
        if !self.seen.contains_key(camera)
            && self.seen.len() >= MAX_CAMERAS
            && let Some(oldest) = self
                .seen
                .iter()
                .min_by_key(|(_, t)| **t)
                .map(|(c, _)| c.clone())
        {
            self.forget(std::slice::from_ref(&oldest));
        }
        self.seen.insert(camera.to_string(), at);
    }

    /// Bytes of the reference thumbnails held, all cameras.
    fn thumb_bytes(&self) -> usize {
        self.recent
            .values()
            .flatten()
            .filter_map(|r| r.thumb.as_ref())
            .map(Thumb::heap_bytes)
            .sum()
    }

    /// Drop the oldest thumbnails (any camera) until they fit in `thumb_cap`.
    fn cap_thumbs(&mut self) {
        let mut total = self.thumb_bytes();
        while total > self.thumb_cap {
            let oldest = self
                .recent
                .values_mut()
                .flat_map(|q| q.iter_mut())
                .filter(|r| r.thumb.is_some())
                .min_by_key(|r| r.at);
            let Some(r) = oldest else {
                break;
            };
            total = total.saturating_sub(r.thumb.take().map_or(0, |t| t.heap_bytes()));
        }
    }

    /// Drop everything remembered of `cameras`.
    fn forget(&mut self, cameras: &[String]) {
        let gone: std::collections::HashSet<&str> = cameras.iter().map(String::as_str).collect();
        self.last.retain(|(c, _), _| !gone.contains(c.as_str()));
        for c in cameras {
            self.recent.remove(c);
            self.skips.remove(c);
            self.seen.remove(c);
        }
    }

    /// The low-space warning when free space should be checked now and is low.
    fn check_free(&mut self, settings: &CollectSettings) -> Option<String> {
        let now = Instant::now();
        if self
            .free_checked
            .is_some_and(|t| now.saturating_duration_since(t) < FREE_CHECK_INTERVAL)
        {
            return None;
        }
        self.free_checked = Some(now);
        let free = (self.free_probe)(&settings.dir)?;
        crate::retention::low_space_message(&settings.dir, free, crate::retention::MIN_FREE_MB)
    }

    /// Image, sidecar, index line: each atomic; the sidecar after the image, so a sidecar
    /// always has its image. Returns the image path relative to `dir`.
    fn write(
        &mut self,
        dir: &Path,
        frame: &mut super::Draft,
        image: &[u8],
    ) -> std::io::Result<String> {
        #[cfg(test)]
        if let Some(kind) = self.fail_writes {
            return Err(std::io::Error::from(kind));
        }
        // Day folder and stem in the configured zone as of this write (a zone change takes effect
        // with the next frame); the sidecar's `timestamp` stays UTC.
        let zone = crate::clock::zone();
        #[cfg(test)]
        let zone = self.zone.unwrap_or(zone);
        let r = &mut frame.record;
        let date = layout::day_folder(zone, frame.at);
        let stem = format!(
            "{}_{}",
            zone.frame_stamp(frame.at),
            layout::safe_id(&r.request_id)
        );
        let rel = layout::Rel {
            scene: r.scene.clone(),
            camera: r.camera.clone(),
            date: date.clone(),
            stem: stem.clone(),
        };
        let day = dir.join(&rel.scene).join(&rel.camera).join(&date);
        // Decoded before the day folder is made, so an image that does not decode leaves no
        // empty folder (retention removes one a failed write leaves anyway).
        let (bytes, original) = if image.starts_with(&[0xFF, 0xD8]) {
            (std::borrow::Cow::Borrowed(image), true)
        } else {
            let img = crate::image::decode(image).map_err(std::io::Error::other)?;
            let jpeg = crate::image::encode_jpeg(&img, 92).map_err(std::io::Error::other)?;
            (std::borrow::Cow::Owned(jpeg), false)
        };
        std::fs::create_dir_all(&day)?;
        r.id = stem.clone();
        r.image = super::record::ImageInfo {
            file: format!("{stem}.jpg"),
            bytes: bytes.len() as u64,
            original,
        };
        let jpg = day.join(format!("{stem}.jpg"));
        layout::write_in_day(&day, &jpg, &bytes)?;
        let json = serde_json::to_vec_pretty(&*r).map_err(std::io::Error::other)?;
        if let Err(e) = layout::write_atomic(&day.join(format!("{stem}.json")), &json) {
            let _ = std::fs::remove_file(&jpg);
            return Err(e);
        }
        let total = (bytes.len() + json.len()) as u64;
        let line = IndexLine::of(r, total);
        layout::append_index(&day, &line)?;
        self.summary.add(&line, &date);
        self.usage.prunable_bytes = self.usage.prunable_bytes.saturating_add(total);
        Ok(rel.jpg())
    }

    /// Apply the age and size caps to `settings.dir` and rescan when something went.
    pub fn prune(&mut self, settings: &CollectSettings) {
        self.last_prune = Some(Instant::now());
        match crate::retention::prune_dataset(
            &settings.dir,
            settings.max_age_days,
            settings.max_bytes,
        ) {
            Ok(r)
                if r.deleted > 0
                    || r.failed > 0
                    || r.repaired_days > 0
                    || self.dir.as_deref() != Some(&settings.dir) =>
            {
                self.adopt(&settings.dir, r.usage);
            }
            Ok(r) => {
                self.usage = r.usage;
                self.publish();
            }
            Err(e) => self.note_error(&format!(
                "could not apply the size and age caps to {}: {e:#}",
                settings.dir.display()
            )),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::record::{Det, Record};
    use crate::collect::{CollectConfig, Draft};
    use crate::image::RgbImage;
    use std::sync::atomic::{AtomicU64, Ordering};

    const MB: u64 = 1024 * 1024;
    /// Free space under the stop point, between it and the re-arm point, and above that.
    const LOW: u64 = 100 * MB;
    const BETWEEN: u64 = (crate::retention::MIN_FREE_MB + 100) * MB;
    const PLENTY: u64 = (crate::retention::MIN_FREE_MB + REARM_MARGIN_MB) * MB;

    /// A store whose free-space probe answers `free`.
    fn probed_store(free: &Arc<AtomicU64>, shared: &Arc<Mutex<Status>>) -> Store {
        let free = free.clone();
        Store::with_probe(
            shared.clone(),
            Box::new(move |_| Some(free.load(Ordering::Relaxed))),
        )
    }

    pub(crate) struct Tmp(pub PathBuf);
    impl Tmp {
        pub(crate) fn new() -> Self {
            let p = std::env::temp_dir().join(format!("bop-collect-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn settings(dir: &Path, edit: impl FnOnce(&mut CollectConfig)) -> Arc<CollectSettings> {
        let mut c = CollectConfig {
            enabled: true,
            dir: Some(dir.to_path_buf()),
            ..CollectConfig::default()
        };
        edit(&mut c);
        Arc::new(CollectSettings::from_config(&c, dir))
    }

    /// A JPEG of a scene: `seed` picks the pattern (different seeds hash far apart).
    fn jpeg(seed: u32) -> bytes::Bytes {
        let (w, h) = (96u32, 64u32);
        let img = RgbImage {
            width: w,
            height: h,
            rgb: (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let v = ((x * (seed + 1) * 37 + y * 11 * (seed + 3)) % 251) as u8;
                    [v, v / 2, 255 - v]
                })
                .collect(),
        };
        crate::image::encode_jpeg(&img, 90).unwrap().into()
    }

    fn base(camera: &str, rid: &str, at: SystemTime) -> Draft {
        Draft {
            at,
            record: Record {
                camera: camera.into(),
                camera_key: format!("m|cam:{camera}"),
                request_id: rid.into(),
                ..crate::collect::record::empty_record()
            },
        }
    }

    fn positive(camera: &str, rid: &str, at: SystemTime) -> Draft {
        let mut d = base(camera, rid, at);
        let det = Det {
            label: "person".into(),
            score: 0.9,
            bbox: [1.0, 1.0, 10.0, 20.0],
        };
        d.record.raw = vec![det.clone()];
        d.record.predictions = vec![det];
        d
    }

    fn frame(s: &Arc<CollectSettings>, d: Draft, seed: u32) -> Frame {
        Frame {
            settings: s.clone(),
            image: jpeg(seed),
            record: d,
        }
    }

    fn store() -> Store {
        Store::with_probe(Arc::new(Mutex::new(Status::default())), Box::new(|_| None))
    }

    fn t(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_790_000_000 + secs)
    }

    #[test]
    fn saves_image_sidecar_and_index_atomically() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let mut st = store();
        let out = st.handle(frame(&s, positive("drive", "abcd1234", t(0)), 1));
        let Handled::Saved { rel, buckets } = out else {
            panic!("{out:?}")
        };
        assert_eq!(buckets, vec![Bucket::Positive]);
        assert!(rel.starts_with("day/drive/"), "{rel}");
        assert!(rel.ends_with("_abcd1234.jpg"), "{rel}");
        let jpg = layout::resolve(&tmp.0, &rel, "jpg").unwrap();
        assert_eq!(
            std::fs::read(&jpg).unwrap(),
            jpeg(1).to_vec(),
            "original bytes"
        );
        let side: Record =
            serde_json::from_slice(&std::fs::read(jpg.with_extension("json")).unwrap()).unwrap();
        assert_eq!(side.version, RECORD_VERSION);
        assert_eq!(side.buckets, vec![Bucket::Positive]);
        assert_eq!(side.matched, vec![Bucket::Positive]);
        assert_eq!(side.dhash.len(), 16);
        assert!(side.image.original);
        let day = jpg.parent().unwrap();
        let idx = layout::read_index(&day.join(layout::INDEX_FILE));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0].id, side.id);
        // No temporary files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(day)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        // The summary rebuilt from disk equals the live one.
        assert_eq!(Summary::scan(&tmp.0), st.summary);
        assert_eq!(st.summary.count("day", Bucket::Positive), 1);
    }

    #[test]
    fn day_folders_and_stems_follow_the_zone() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let mut st = store();
        // 2026-09-22T00:13:20Z: still the 21st in Los Angeles (PDT), the 22nd in UTC and India.
        let at = t(10 * 3600);
        let saved = |st: &mut Store, camera: &str, rid: &str, seed: u32| {
            let out = st.handle(frame(&s, positive(camera, rid, at), seed));
            let Handled::Saved { rel, .. } = out else {
                panic!("{out:?}")
            };
            rel
        };
        st.zone = Some("America/Los_Angeles".parse().unwrap());
        let la = saved(&mut st, "a", "r1", 1);
        assert_eq!(la, "day/a/2026-09-21/20260921T171320000-0700_r1.jpg");
        st.zone = Some(crate::clock::Zone::Utc);
        let utc = saved(&mut st, "b", "r2", 2);
        assert_eq!(utc, "day/b/2026-09-22/20260922T001320000Z_r2.jpg");
        st.zone = Some("+05:30".parse().unwrap());
        let ist = saved(&mut st, "c", "r3", 3);
        assert_eq!(ist, "day/c/2026-09-22/20260922T054320000p0530_r3.jpg");
        for rel in [&la, &utc, &ist] {
            let r = layout::parse_rel(rel).unwrap();
            assert_eq!(
                crate::clock::parse_frame_stamp(&r.stem),
                Some(at.into()),
                "{rel}"
            );
            let jpg = layout::resolve(&tmp.0, rel, "jpg").unwrap();
            let side: Record =
                serde_json::from_slice(&std::fs::read(jpg.with_extension("json")).unwrap())
                    .unwrap();
            assert_eq!(side.id, r.stem);
        }
        assert_eq!(Summary::scan(&tmp.0), st.summary);
    }

    #[test]
    fn per_camera_interval_and_sampling() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 30
        });
        let mut st = store();
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Saved { .. }
        ));
        // Same camera within 30 s: rate limited (even though the scene differs).
        assert_eq!(
            st.handle(frame(&s, positive("a", "r2", t(10)), 2)),
            Handled::Dropped(Dropped::Rate)
        );
        // Another camera is independent.
        assert!(matches!(
            st.handle(frame(&s, positive("b", "r3", t(10)), 3)),
            Handled::Saved { .. }
        ));
        // After the interval: saved again.
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r4", t(31)), 4)),
            Handled::Saved { .. }
        ));
        // rate 0: never; a disabled bucket: not matched at all.
        let s0 = settings(&tmp.0, |c| c.buckets.positive.rate = 0.0);
        assert_eq!(
            st.handle(frame(&s0, positive("c", "r5", t(100)), 5)),
            Handled::Dropped(Dropped::Rate)
        );
        let off = settings(&tmp.0, |c| c.buckets.positive.enabled = false);
        assert_eq!(
            st.handle(frame(&off, positive("c", "r6", t(200)), 6)),
            Handled::NoBucket
        );
        let disabled = settings(&tmp.0, |c| c.enabled = false);
        assert_eq!(
            st.handle(frame(&disabled, positive("c", "r7", t(300)), 7)),
            Handled::Off
        );
    }

    #[test]
    fn sampling_rate_is_close_to_the_setting() {
        let n = 20_000;
        let kept = (0..n)
            .filter(|i| sample(&format!("{i:08x}"), Bucket::Empty) < 0.25)
            .count();
        let share = kept as f64 / f64::from(n);
        assert!((share - 0.25).abs() < 0.02, "{share}");
        assert_eq!(sample("abc", Bucket::Empty), sample("abc", Bucket::Empty));
    }

    /// An empty frame at night or IR is kept after the shorter `night_interval` (120 s); a day one
    /// waits the full 600 s.
    #[test]
    fn night_and_ir_empty_frames_use_the_night_interval() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let mut st = store();
        let scene = |cam: &str, rid: &str, at: u64, scene: &str| {
            let mut d = base(cam, rid, t(at));
            d.record.scene = scene.into();
            d
        };
        for (cam, scene_name) in [("n", "night"), ("i", "ir"), ("d", "day")] {
            assert!(matches!(
                st.handle(frame(&s, scene(cam, "r1", 0, scene_name), 1)),
                Handled::Saved { .. }
            ));
            // 60 s on: limited in every scene.
            assert_eq!(
                st.handle(frame(&s, scene(cam, "r2", 60, scene_name), 2)),
                Handled::Dropped(Dropped::Rate),
                "{scene_name}"
            );
            // 130 s on: night / IR are due again, day is not.
            let out = st.handle(frame(&s, scene(cam, "r3", 130, scene_name), 3));
            if scene_name == "day" {
                assert_eq!(out, Handled::Dropped(Dropped::Rate));
                assert!(matches!(
                    st.handle(frame(&s, scene(cam, "r4", 601, scene_name), 4)),
                    Handled::Saved { .. }
                ));
            } else {
                assert!(
                    matches!(out, Handled::Saved { .. }),
                    "{scene_name}: {out:?}"
                );
            }
        }
    }

    /// One camera, mixed scenes: the empty bucket has a single clock, the interval depends on the
    /// scene of the frame that asks.
    #[test]
    fn mixed_scenes_share_one_clock_with_the_asking_frames_interval() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let mut st = store();
        let mut f = |rid: &str, at: u64, scene: &str, seed: u32| {
            let mut d = base("a", rid, t(at));
            d.record.scene = scene.into();
            st.handle(frame(&s, d, seed))
        };
        assert!(matches!(f("r1", 0, "day", 1), Handled::Saved { .. }));
        assert!(matches!(f("r2", 130, "night", 2), Handled::Saved { .. }));
        assert_eq!(
            f("r3", 601, "day", 3),
            Handled::Dropped(Dropped::Rate),
            "600 s from the night save at 130 s"
        );
        assert!(matches!(f("r4", 731, "day", 4), Handled::Saved { .. }));
    }

    /// Unset night interval: never stricter than the day interval.
    #[test]
    fn an_unset_night_interval_is_never_stricter_than_the_day_one() {
        for day in [0u32, 30] {
            let tmp = Tmp::new();
            let s = settings(&tmp.0, |c| c.buckets.empty.per_camera_min_interval_s = day);
            let mut st = store();
            let mut n = 0u32;
            let mut go = |at: u64, scene: &str| {
                n += 1;
                let mut d = base("a", &format!("r{n}"), t(at));
                d.record.scene = scene.into();
                st.handle(frame(&s, d, n))
            };
            assert!(matches!(go(0, "night"), Handled::Saved { .. }));
            assert!(
                matches!(go(31, "night"), Handled::Saved { .. }),
                "day {day}"
            );
            if day == 0 {
                assert!(matches!(go(32, "ir"), Handled::Saved { .. }));
            } else {
                assert_eq!(go(40, "ir"), Handled::Dropped(Dropped::Rate));
            }
        }
    }

    #[test]
    fn a_night_interval_of_zero_is_no_limit() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.empty.night_interval_s = Some(0));
        let mut st = store();
        for (i, at) in [0u64, 1, 2].into_iter().enumerate() {
            let mut d = base("a", &format!("r{i}"), t(at));
            d.record.scene = "ir".into();
            assert!(
                matches!(st.handle(frame(&s, d, i as u32 + 1)), Handled::Saved { .. }),
                "{i}"
            );
        }
        // A day frame is still held to 600 s.
        let mut d = base("a", "rd", t(3));
        d.record.scene = "day".into();
        assert_eq!(st.handle(frame(&s, d, 9)), Handled::Dropped(Dropped::Rate));
    }

    /// Forgetting never lets a night frame in before the night interval is over.
    #[test]
    fn a_camera_inside_a_longer_night_interval_is_kept() {
        let tmp = Tmp::new();
        let two_days = 2 * CAMERA_IDLE.as_secs();
        let s = settings(&tmp.0, |c| {
            c.buckets.empty.night_interval_s = Some(u32::try_from(two_days).unwrap())
        });
        let mut st = store();
        let mut night = |cam: &str, rid: &str, at: u64, seed: u32| {
            let mut d = base(cam, rid, t(at));
            d.record.scene = "night".into();
            st.handle(frame(&s, d, seed))
        };
        assert!(matches!(night("a", "a1", 0, 1), Handled::Saved { .. }));
        let later = two_days * 3 / 4;
        assert!(matches!(night("b", "b1", later, 2), Handled::Saved { .. }));
        assert_eq!(
            night("a", "a2", later, 3),
            Handled::Dropped(Dropped::Rate),
            "past a day, inside the night interval: a is still limited"
        );
    }

    /// Both modes: `dhash` is the rule from before `both`, and on these pictures (the same one, or
    /// one far apart) `both` decides alike.
    #[test]
    fn near_duplicates_of_a_camera_are_dropped_within_the_window() {
        for mode in [DedupMode::Dhash, DedupMode::Both] {
            let tmp = Tmp::new();
            let s = settings(&tmp.0, |c| {
                c.buckets.positive.per_camera_min_interval_s = 0;
                c.dedup_mode = mode;
            });
            let mut st = store();
            assert!(matches!(
                st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
                Handled::Saved { .. }
            ));
            assert_eq!(
                st.handle(frame(&s, positive("a", "r2", t(5)), 1)),
                Handled::Dropped(Dropped::Dup),
                "{mode:?}"
            );
            // Another scene of the same camera is kept; the same picture on another camera too.
            assert!(matches!(
                st.handle(frame(&s, positive("a", "r3", t(6)), 7)),
                Handled::Saved { .. }
            ));
            assert!(matches!(
                st.handle(frame(&s, positive("b", "r4", t(6)), 1)),
                Handled::Saved { .. }
            ));
            // Past the window the same picture is saved again.
            let later = t(DEDUP_WINDOW.as_secs() + 10);
            assert!(matches!(
                st.handle(frame(&s, positive("a", "r5", later), 1)),
                Handled::Saved { .. }
            ));
            assert_eq!(st.summary.frames, 4, "{mode:?}");
            // `dhash` mode keeps no thumbnails.
            assert_eq!(st.thumb_bytes() == 0, mode == DedupMode::Dhash);
        }
    }

    /// A 640 x 360 grey picture: a left-to-right ramp of `slope` grey levels, an optional `size` x `size` blob of
    /// `delta` at `(x, y)` and `step` grey levels added to every other 71 px column band.
    fn ramp_jpeg(slope: u32, blob: Option<(u32, u32, u32, i32)>, step: i32) -> bytes::Bytes {
        let (w, h) = (640u32, 360u32);
        let rgb = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let mut v = 40 + (x * slope / w) as i32 + if (x / 71) % 2 == 1 { step } else { 0 };
                if let Some((bx, by, size, delta)) = blob
                    && (bx..bx + size).contains(&x)
                    && (by..by + size).contains(&y)
                {
                    v += delta;
                }
                let v = v.clamp(0, 255) as u8;
                [v, v, v]
            })
            .collect();
        let img = RgbImage {
            width: w,
            height: h,
            rgb,
        };
        crate::image::encode_jpeg(&img, 92).unwrap().into()
    }

    fn with_image(s: &Arc<CollectSettings>, d: Draft, image: bytes::Bytes) -> Frame {
        Frame {
            settings: s.clone(),
            image,
            record: d,
        }
    }

    /// The sidecar of a saved frame.
    fn sidecar(dir: &Path, out: &Handled) -> Record {
        let Handled::Saved { rel, .. } = out else {
            panic!("{out:?}")
        };
        let path = dir.join(rel.replace(".jpg", ".json"));
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn scene_frame(camera: &str, rid: &str, at: SystemTime, scene: &str) -> Draft {
        let mut d = positive(camera, rid, at);
        d.record.scene = scene.into();
        d
    }

    #[test]
    fn a_small_object_dhash_cannot_see_keeps_the_frame_in_both_mode_only() {
        let plain = ramp_jpeg(150, None, 0);
        let blob = ramp_jpeg(150, Some((300, 150, 24, 70)), 0);
        let (ha, _, _) = dhash::of_encoded(&plain).unwrap();
        let (hb, _, _) = dhash::of_encoded(&blob).unwrap();
        assert!(dhash::hamming(ha, hb) <= 6, "{}", dhash::hamming(ha, hb));
        for mode in [DedupMode::Both, DedupMode::Dhash] {
            let tmp = Tmp::new();
            let s = settings(&tmp.0, |c| {
                c.buckets.positive.per_camera_min_interval_s = 0;
                c.dedup_mode = mode;
            });
            let mut st = store();
            let first = st.handle(with_image(&s, positive("a", "r1", t(0)), plain.clone()));
            let info = sidecar(&tmp.0, &first).dedup.unwrap();
            assert_eq!(info.decided_by, DedupInfo::NO_REFERENCE);
            assert_eq!(info.mode, mode.as_str());
            let out = st.handle(with_image(&s, positive("a", "r2", t(5)), blob.clone()));
            if mode == DedupMode::Dhash {
                assert_eq!(out, Handled::Dropped(Dropped::Dup));
                continue;
            }
            let info = sidecar(&tmp.0, &out).dedup.unwrap();
            assert_eq!(info.decided_by, DedupInfo::BLOCKDIFF, "{info:?}");
            assert!(info.nearest_hamming.unwrap() <= 6);
            assert!(info.block_largest_blob.unwrap() >= 3, "{info:?}");
            assert!(info.block_max_z.unwrap() > 5.0, "{info:?}");
            // Both agree on the same picture again: dropped.
            assert_eq!(
                st.handle(with_image(&s, positive("a", "r3", t(6)), blob.clone())),
                Handled::Dropped(Dropped::Dup)
            );
        }
    }

    #[test]
    fn a_dhash_change_the_block_test_calls_noise_is_kept() {
        // Faint column bands flip dHash bits; 4 grey levels are under the block test's d_min.
        let plain = ramp_jpeg(0, None, 0);
        let banded = ramp_jpeg(0, None, 4);
        let (ha, _, _) = dhash::of_encoded(&plain).unwrap();
        let (hb, _, _) = dhash::of_encoded(&banded).unwrap();
        assert!(dhash::hamming(ha, hb) > 0);
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.dedup_hamming = 0;
        });
        let mut st = store();
        st.handle(with_image(&s, positive("a", "r1", t(0)), plain));
        let out = st.handle(with_image(&s, positive("a", "r2", t(5)), banded.clone()));
        let info = sidecar(&tmp.0, &out).dedup.unwrap();
        assert_eq!(info.decided_by, DedupInfo::DHASH, "{info:?}");
        // The block test alone would call it a duplicate.
        let refs = &st.recent["a"];
        let t0 = refs[0].thumb.as_ref().unwrap();
        let t1 = refs[1].thumb.as_ref().unwrap();
        let v = blockdiff::verdict_with(t1, t0, Scene::Day, None, &blockdiff::Params::COLLECTOR);
        assert!(!v.different, "{v:?}");
    }

    #[test]
    fn a_reference_of_another_scene_never_matches_in_both_mode() {
        for mode in [DedupMode::Both, DedupMode::Dhash] {
            let tmp = Tmp::new();
            let s = settings(&tmp.0, |c| {
                c.buckets.positive.per_camera_min_interval_s = 0;
                c.dedup_mode = mode;
            });
            let mut st = store();
            let img = ramp_jpeg(150, None, 0);
            st.handle(with_image(
                &s,
                scene_frame("a", "r1", t(0), "day"),
                img.clone(),
            ));
            let out = st.handle(with_image(
                &s,
                scene_frame("a", "r2", t(5), "ir"),
                img.clone(),
            ));
            if mode == DedupMode::Dhash {
                // The old rule compares across scenes.
                assert_eq!(out, Handled::Dropped(Dropped::Dup));
            } else {
                let info = sidecar(&tmp.0, &out).dedup.unwrap();
                assert_eq!(info.decided_by, DedupInfo::UNCHECKED, "{info:?}");
                // Its own scene's reference now matches.
                assert_eq!(
                    st.handle(with_image(&s, scene_frame("a", "r3", t(6), "ir"), img)),
                    Handled::Dropped(Dropped::Dup)
                );
            }
        }
    }

    #[test]
    fn the_thumbnail_cap_evicts_the_oldest_and_only_ever_keeps_more() {
        assert_eq!(DEDUP_THUMB_BYTES / blockdiff::THUMB_CELLS, 910);
        // One reference: its struct plus one thumbnail on the heap.
        assert!(
            std::mem::size_of::<Reference>() <= 96,
            "{}",
            std::mem::size_of::<Reference>()
        );
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.positive.per_camera_min_interval_s = 0);
        let mut st = store();
        st.thumb_cap = 2 * blockdiff::THUMB_CELLS;
        let img = ramp_jpeg(150, None, 0);
        for (i, cam) in ["a", "b", "c"].into_iter().enumerate() {
            let out = st.handle(with_image(&s, positive(cam, cam, t(i as u64)), img.clone()));
            assert!(matches!(out, Handled::Saved { .. }));
            assert!(st.thumb_bytes() <= st.thumb_cap);
        }
        let thumb = |cam: &str| st.recent[cam][0].thumb.as_ref().map(Thumb::heap_bytes);
        assert_eq!(thumb("a"), None, "the oldest lost its thumbnail");
        assert_eq!(thumb("b"), Some(blockdiff::THUMB_CELLS));
        assert_eq!(thumb("c"), Some(blockdiff::THUMB_CELLS));
        // Camera a's reference keeps its hash but cannot confirm a drop any more: kept.
        let out = st.handle(with_image(&s, positive("a", "a2", t(10)), img.clone()));
        let info = sidecar(&tmp.0, &out).dedup.unwrap();
        assert_eq!(info.decided_by, DedupInfo::UNCHECKED);
        // Camera c's still can.
        assert_eq!(
            st.handle(with_image(&s, positive("c", "c2", t(11)), img)),
            Handled::Dropped(Dropped::Dup)
        );
        // Forgetting a camera frees its thumbnails.
        let before = st.thumb_bytes();
        st.forget(&["c".to_string()]);
        assert!(st.thumb_bytes() < before);
    }

    /// A minute of rain-like look-alikes at 1 fps (dHash alike, each with a new blob the block test
    /// sees) on a bucket that skips the dedup: at most the cap is saved, in both modes; a
    /// `positive` look-alike the block test calls different is still kept.
    #[test]
    fn a_storm_of_block_different_look_alikes_is_capped_like_dhash_mode() {
        for mode in [DedupMode::Both, DedupMode::Dhash] {
            let tmp = Tmp::new();
            let s = settings(&tmp.0, |c| {
                c.buckets.positive.per_camera_min_interval_s = 0;
                c.dedup_mode = mode;
            });
            let mut st = store();
            let mut saved = 0;
            for k in 0..60u32 {
                let img = ramp_jpeg(
                    150,
                    Some((20 + (k % 12) * 50, 30 + (k / 12) * 60, 24, 70)),
                    0,
                );
                let rid = format!("n{k}");
                if matches!(
                    st.handle(with_image(&s, near("a", &rid, t(u64::from(k))), img)),
                    Handled::Saved { .. }
                ) {
                    saved += 1;
                }
            }
            assert!(
                saved <= 1 + DEDUP_SKIPS_PER_CAMERA,
                "{mode:?}: {saved} saved in a minute"
            );
            assert!(saved > DEDUP_SKIPS_PER_CAMERA / 2, "{mode:?}: {saved}");
            let img = ramp_jpeg(150, Some((600, 300, 24, 70)), 0);
            let out = st.handle(with_image(&s, positive("a", "p", t(59)), img));
            assert_eq!(
                matches!(out, Handled::Saved { .. }),
                mode == DedupMode::Both,
                "{mode:?}: {out:?}"
            );
        }
    }

    /// One reference must pass both tests: dHash matching reference A and the block test matching
    /// reference B is no duplicate.
    #[test]
    fn both_tests_must_pass_on_the_same_reference() {
        let plain = Thumb::of_encoded(&ramp_jpeg(150, None, 0)).unwrap().0;
        let blob = Thumb::of_encoded(&ramp_jpeg(150, Some((300, 150, 24, 70)), 0))
            .unwrap()
            .0;
        let hash = 0x0123_4567_89ab_cdef_u64;
        let refs: VecDeque<Reference> = [
            // A: the same dHash, but the block test sees the blob.
            (hash, blob),
            // B: the block test finds nothing, but dHash is far.
            (!hash, plain.clone()),
        ]
        .into_iter()
        .enumerate()
        .map(|(k, (h, th))| Reference {
            at: t(k as u64),
            hash: h,
            scene: Scene::Day,
            thumb: Some(th),
        })
        .collect();
        let c = dedup_check(&refs, hash, Some(&plain), Scene::Day, 6, DedupMode::Both);
        assert!(!c.duplicate && c.look_alike, "{c:?}");
        assert_eq!(c.info.decided_by, DedupInfo::BLOCKDIFF);
        assert_eq!(c.info.nearest_hamming, Some(0));
        // `nearest_hamming` is the minimum over every reference, also when one confirms.
        let mut refs = refs;
        refs.push_back(Reference {
            at: t(5),
            hash: hash ^ 0b111,
            scene: Scene::Day,
            thumb: Some(plain.clone()),
        });
        let c = dedup_check(&refs, hash, Some(&plain), Scene::Day, 6, DedupMode::Both);
        assert!(c.duplicate);
        assert_eq!(c.info.nearest_hamming, Some(0));
        assert!(
            c.info.block_largest_blob.is_some(),
            "the confirming verdict is recorded"
        );
    }

    #[test]
    fn the_night_interval_applies_before_the_dedup() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let mut st = store();
        let empty_ir = |rid: &str, at: u64| {
            let mut d = base("n", rid, t(at));
            d.record.scene = "ir".into();
            d
        };
        let img = ramp_jpeg(150, None, 0);
        assert!(matches!(
            st.handle(with_image(&s, empty_ir("r1", 0), img.clone())),
            Handled::Saved { .. }
        ));
        // Inside the 120 s night interval: the rate limit, before any dedup.
        assert_eq!(
            st.handle(with_image(&s, empty_ir("r2", 60), img.clone())),
            Handled::Dropped(Dropped::Rate)
        );
        // Past it: now the dedup decides (the same picture).
        assert_eq!(
            st.handle(with_image(&s, empty_ir("r3", 130), img)),
            Handled::Dropped(Dropped::Dup)
        );
    }

    /// The dedup's cost per frame on a 2688 x 1520 frame against 32 references that all pass dHash
    /// and all differ by the block test (the worst case: every one is compared). A measurement:
    /// prints, runs only with `BOP_DEDUP_BENCH` set (debug numbers are slow; use `--release` for real
    /// ones).
    #[test]
    #[ignore = "a measurement: set BOP_DEDUP_BENCH"]
    fn dedup_cost_per_frame() {
        if std::env::var_os("BOP_DEDUP_BENCH").is_none() {
            return;
        }
        let (w, h) = (2688usize, 1520usize);
        let luma = |blob_x: usize| -> Vec<u8> {
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let v = 40 + x * 150 / w + ((x / 8 + y / 8) % 3);
                    let blob = (blob_x..blob_x + 60).contains(&x) && (700..760).contains(&y);
                    (v + if blob { 60 } else { 0 }) as u8
                })
                .collect()
        };
        let img = |blob_x: usize| {
            let l = luma(blob_x);
            let rgb = l.iter().flat_map(|&v| [v, v, v]).collect();
            let img = RgbImage {
                width: w as u32,
                height: h as u32,
                rgb,
            };
            crate::image::encode_jpeg(&img, 90).unwrap()
        };
        let mut refs = VecDeque::new();
        for k in 0..DEDUP_KEEP {
            let l = luma(100 + k * 70);
            refs.push_back(Reference {
                at: t(k as u64),
                hash: dhash::of_luma(&l, w, h),
                scene: Scene::Ir,
                thumb: Thumb::of_luma(&l, w, h),
            });
        }
        let new = img(2500);
        let runs = 10;
        let (mut prep, mut check) = (Vec::new(), Vec::new());
        for _ in 0..runs {
            let t0 = Instant::now();
            let (l, lw, lh) = dhash::luma_of_encoded(&new).unwrap();
            let hash = dhash::of_luma(&l, lw, lh);
            let thumb = Thumb::of_luma(&l, lw, lh);
            let t1 = Instant::now();
            let c = dedup_check(&refs, hash, thumb.as_ref(), Scene::Ir, 6, DedupMode::Both);
            let t2 = Instant::now();
            assert!(!c.duplicate);
            assert_eq!(c.info.decided_by, DedupInfo::BLOCKDIFF);
            prep.push((t1 - t0).as_secs_f64() * 1e3);
            check.push((t2 - t1).as_secs_f64() * 1e3);
        }
        prep.sort_by(f64::total_cmp);
        check.sort_by(f64::total_cmp);
        eprintln!(
            "dedup cost, median of {runs}: decode + dHash + thumbnail {:.1} ms, check against {} \
             look-alike references {:.1} ms ({:.2} ms each)",
            prep[runs / 2],
            DEDUP_KEEP,
            check[runs / 2],
            check[runs / 2] / DEDUP_KEEP as f64
        );
        eprintln!(
            "a reference: {} bytes + a {}-byte thumbnail",
            std::mem::size_of::<Reference>(),
            blockdiff::THUMB_CELLS
        );
    }

    #[test]
    fn low_disk_disables_until_resumed_or_reconfigured() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.positive.per_camera_min_interval_s = 0);
        let free = Arc::new(AtomicU64::new(LOW));
        let shared = Arc::new(Mutex::new(Status::default()));
        let mut st = probed_store(&free, &shared);
        assert_eq!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Dropped(Dropped::Disk)
        );
        assert!(
            shared
                .lock()
                .unwrap()
                .disabled
                .as_deref()
                .unwrap()
                .contains("low disk space")
        );
        assert!(METRICS.dropped(Dropped::Disk) >= 1);
        // Stays off while nothing changes, without probing again.
        free.store(PLENTY, Ordering::Relaxed);
        assert_eq!(
            st.handle(frame(&s, positive("a", "r2", t(1)), 2)),
            Handled::Dropped(Dropped::Disk)
        );
        st.resume();
        assert!(shared.lock().unwrap().disabled.is_none());
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r3", t(2)), 3)),
            Handled::Saved { .. }
        ));
        // A full disk while writing disables it too; a settings change re-arms.
        st.fail_writes = Some(std::io::ErrorKind::StorageFull);
        assert_eq!(
            st.handle(frame(&s, positive("a", "r4", t(3)), 4)),
            Handled::Dropped(Dropped::Disk)
        );
        assert!(
            shared
                .lock()
                .unwrap()
                .disabled
                .as_deref()
                .unwrap()
                .contains("full")
        );
        st.fail_writes = None;
        let changed = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.max_gb = 4.0;
        });
        assert!(matches!(
            st.handle(frame(&changed, positive("a", "r5", t(4)), 5)),
            Handled::Saved { .. }
        ));
        // Another write error is an error, not a disk stop.
        st.fail_writes = Some(std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            st.handle(frame(&changed, positive("a", "r6", t(5)), 6)),
            Handled::Dropped(Dropped::Error)
        );
        assert!(shared.lock().unwrap().disabled.is_none());
    }

    #[test]
    fn a_disk_stop_rearms_on_its_own_once_space_is_back() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.positive.per_camera_min_interval_s = 0);
        let free = Arc::new(AtomicU64::new(LOW));
        let shared = Arc::new(Mutex::new(Status::default()));
        let mut st = probed_store(&free, &shared);
        st.handle(frame(&s, positive("a", "r1", t(0)), 1));
        let reason = shared.lock().unwrap().disabled.clone().unwrap();
        assert!(reason.contains("low disk space"), "{reason}");
        let err = shared.lock().unwrap().last_error.clone().unwrap();
        assert!(err.message.contains(&reason), "{}", err.message);
        // Within the check interval nothing is probed, even with space back.
        free.store(PLENTY, Ordering::Relaxed);
        st.poll_rearm();
        assert!(shared.lock().unwrap().disabled.is_some());
        // Above the stop point but inside the margin: still stopped.
        free.store(BETWEEN, Ordering::Relaxed);
        st.free_checked = None;
        st.poll_rearm();
        assert!(shared.lock().unwrap().disabled.is_some());
        // Past the margin, with no frame arriving: back on, and the next frame is saved.
        free.store(PLENTY, Ordering::Relaxed);
        st.free_checked = None;
        st.poll_rearm();
        assert!(shared.lock().unwrap().disabled.is_none());
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r2", t(1)), 2)),
            Handled::Saved { .. }
        ));
        // A frame triggers the check too.
        free.store(LOW, Ordering::Relaxed);
        st.free_checked = None;
        assert_eq!(
            st.handle(frame(&s, positive("a", "r3", t(2)), 3)),
            Handled::Dropped(Dropped::Disk)
        );
        free.store(PLENTY, Ordering::Relaxed);
        st.free_checked = None;
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r4", t(3)), 4)),
            Handled::Saved { .. }
        ));
    }

    #[test]
    fn free_space_inside_the_margin_does_not_stop_a_running_collector() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let free = Arc::new(AtomicU64::new(BETWEEN));
        let mut st = probed_store(&free, &Arc::new(Mutex::new(Status::default())));
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Saved { .. }
        ));
    }

    #[test]
    fn an_unknown_free_space_never_rearms() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let shared = Arc::new(Mutex::new(Status::default()));
        let mut st = Store::with_probe(shared.clone(), Box::new(|_| None));
        st.fail_writes = Some(std::io::ErrorKind::StorageFull);
        st.handle(frame(&s, positive("a", "r1", t(0)), 1));
        assert!(shared.lock().unwrap().disabled.is_some());
        st.free_checked = None;
        st.poll_rearm();
        assert!(shared.lock().unwrap().disabled.is_some());
    }

    /// A frame near the threshold (a box scoring 0.45 under the default 0.5, margin 0.1).
    fn near(camera: &str, rid: &str, at: SystemTime) -> Draft {
        let mut d = base(camera, rid, at);
        d.record.threshold = 0.5;
        d.record.below_threshold = vec![Det {
            label: "person".into(),
            score: 0.45,
            bbox: [1.0, 1.0, 10.0, 20.0],
        }];
        d.record.raw = d.record.below_threshold.clone();
        d
    }

    #[test]
    fn frames_a_reviewer_wants_skip_the_dedup_and_positive_and_empty_keep_it() {
        let tmp = Tmp::new();
        // Near-threshold only, so the row of look-alikes below tests that bucket alone.
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.buckets.empty.per_camera_min_interval_s = 0;
        });
        let mut st = store();
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Saved { .. }
        ));
        // The same picture: dropped as a plain positive, or as empty ...
        assert_eq!(
            st.handle(frame(&s, positive("a", "r2", t(1)), 1)),
            Handled::Dropped(Dropped::Dup)
        );
        assert_eq!(
            st.handle(frame(&s, base("a", "r3", t(2)), 1)),
            Handled::Dropped(Dropped::Dup)
        );
        // ... saved near the threshold, and with a removed box, a verifier verdict or a
        // disagreement.
        let out = st.handle(frame(&s, near("a", "r4", t(3)), 1));
        let Handled::Saved { buckets, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(buckets, vec![Bucket::NearThreshold]);
        let mut d = positive("a", "r5", t(4));
        d.record.suppressed = vec![crate::collect::record::Suppressed {
            label: "person".into(),
            score: 0.9,
            bbox: [1.0, 1.0, 10.0, 20.0],
            reason: "aspect".into(),
        }];
        let out = st.handle(frame(&s, d, 1));
        let Handled::Saved { buckets, .. } = out else {
            panic!("{out:?}")
        };
        // It is also a positive frame: it keeps every bucket it was kept for.
        assert_eq!(buckets, vec![Bucket::Positive, Bucket::Suppressed]);
        for (rid, outcome, want) in [
            ("r6", "rejected", Bucket::VerifyRejected),
            ("r7", "kept", Bucket::VerifyKept),
        ] {
            let mut d = base("a", rid, t(5));
            d.record.raw = vec![Det {
                label: "person".into(),
                score: 0.9,
                bbox: [1.0, 1.0, 10.0, 20.0],
            }];
            d.record.verify = Some(crate::collect::record::VerifyInfo {
                candidates: vec![crate::collect::record::VerifyCandidate {
                    label: "person".into(),
                    score: 0.9,
                    bbox: [1.0, 1.0, 10.0, 20.0],
                    outcome: outcome.into(),
                }],
                ..Default::default()
            });
            let out = st.handle(frame(&s, d, 1));
            let Handled::Saved { buckets, .. } = out else {
                panic!("{out:?}")
            };
            assert!(buckets.contains(&want), "{buckets:?}");
        }
    }

    /// M2: the per-camera maps forget a camera silent for a day, and hold at most
    /// [`MAX_CAMERAS`] however many names clients make up.
    #[test]
    fn idle_cameras_are_forgotten_and_the_count_is_capped() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 30
        });
        let mut st = store();
        let day = CAMERA_IDLE.as_secs();
        for (cam, seed) in [("a", 1), ("b", 2)] {
            assert!(matches!(
                st.handle(frame(&s, positive(cam, cam, t(0)), seed)),
                Handled::Saved { .. }
            ));
        }
        // `b` sends again before the day is out; `a` does not.
        assert!(matches!(
            st.handle(frame(&s, positive("b", "b2", t(day - 3600)), 3)),
            Handled::Saved { .. }
        ));
        assert_eq!(st.cameras(), 2);
        assert!(matches!(
            st.handle(frame(&s, positive("c", "c1", t(day + 60)), 4)),
            Handled::Saved { .. }
        ));
        assert_eq!(st.cameras(), 2, "a is forgotten, b and c stay");
        assert!(!st.seen.contains_key("a") && !st.recent.contains_key("a"));
        assert!(!st.last.keys().any(|(c, _)| c == "a"));
        assert!(st.last.contains_key(&("b".to_string(), Bucket::Positive)));
        // A flood of made-up names (rate 0: none is written, each still counts as seen).
        let none = settings(&tmp.0, |c| c.buckets.positive.rate = 0.0);
        for i in 0..MAX_CAMERAS + 10 {
            let at = t(day + 100 + i as u64);
            assert_eq!(
                st.handle(frame(&none, positive(&format!("x{i}"), "r", at), 5)),
                Handled::Dropped(Dropped::Rate)
            );
        }
        assert_eq!(st.cameras(), MAX_CAMERAS);
        assert!(
            st.last.is_empty() && st.recent.is_empty(),
            "b and c went first"
        );
        assert!(st.seen.contains_key(&format!("x{}", MAX_CAMERAS + 9)));
        assert!(!st.seen.contains_key("x0"));
    }

    /// Forgetting never lets a frame in before its bucket's interval is over.
    #[test]
    fn a_camera_inside_a_longer_interval_is_kept() {
        let tmp = Tmp::new();
        let two_days = 2 * CAMERA_IDLE.as_secs();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = u32::try_from(two_days).unwrap()
        });
        let mut st = store();
        assert!(matches!(
            st.handle(frame(&s, positive("a", "a1", t(0)), 1)),
            Handled::Saved { .. }
        ));
        let later = two_days * 3 / 4;
        assert!(matches!(
            st.handle(frame(&s, positive("b", "b1", t(later)), 2)),
            Handled::Saved { .. }
        ));
        assert_eq!(st.cameras(), 2, "past a day, inside the interval: kept");
        assert_eq!(
            st.handle(frame(&s, positive("a", "a2", t(later + 10)), 3)),
            Handled::Dropped(Dropped::Rate)
        );
        assert!(matches!(
            st.handle(frame(&s, positive("a", "a3", t(two_days + 1)), 4)),
            Handled::Saved { .. }
        ));
    }

    #[test]
    fn a_storm_of_look_alikes_is_capped_per_camera_even_at_interval_zero() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        assert_eq!(s.buckets.near_threshold.per_camera_min_interval_s, 0);
        let mut st = store();
        // The look-alike of the storm, saved as an ordinary positive first.
        assert!(matches!(
            st.handle(frame(&s, positive("a", "first", t(0)), 1)),
            Handled::Saved { .. }
        ));
        let storm = |st: &mut Store, camera: &str, from: u64| {
            (0..100u64)
                .filter(|i| {
                    let rid = format!("r{from}x{i}");
                    // Five frames a second.
                    let at = t(from) + Duration::from_millis(200 * i + 1);
                    matches!(
                        st.handle(frame(&s, near(camera, &rid, at), 1)),
                        Handled::Saved { .. }
                    )
                })
                .count()
        };
        assert_eq!(storm(&mut st, "a", 1), DEDUP_SKIPS_PER_CAMERA);
        // Another camera has its own allowance (its first frame is no duplicate of anything).
        assert_eq!(storm(&mut st, "b", 1), DEDUP_SKIPS_PER_CAMERA + 1);
        // A frame that looks different is no skip and is still saved.
        assert!(matches!(
            st.handle(frame(&s, near("a", "other", t(30)), 7)),
            Handled::Saved { .. }
        ));
        // A new window, a new allowance.
        assert_eq!(storm(&mut st, "a", 90), DEDUP_SKIPS_PER_CAMERA);
    }

    #[test]
    fn write_failures_reach_the_status_with_their_time() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let shared = Arc::new(Mutex::new(Status::default()));
        let mut st = Store::with_probe(shared.clone(), Box::new(|_| None));
        st.fail_writes = Some(std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Dropped(Dropped::Error)
        );
        let e = shared.lock().unwrap().last_error.clone().unwrap();
        assert!(
            e.message.contains("could not save a frame under"),
            "{}",
            e.message
        );
        assert!(e.at.ends_with('Z'), "{}", e.at);
        assert_eq!(shared.lock().unwrap().last_saved_at, None);
        // A later success leaves the error on show and adds the frame's time.
        st.fail_writes = None;
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r2", t(1)), 2)),
            Handled::Saved { .. }
        ));
        let now = shared.lock().unwrap().clone();
        assert!(now.last_error.is_some());
        assert_eq!(now.last_saved_at, Some(crate::collect::utc_rfc3339(t(1))));
    }

    #[test]
    fn a_frame_no_bucket_wants_is_counted_and_an_undecodable_one_is_an_error() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.positive.enabled = false);
        let shared = Arc::new(Mutex::new(Status::default()));
        let mut st = Store::with_probe(shared.clone(), Box::new(|_| None));
        let before = METRICS.no_bucket();
        assert_eq!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::NoBucket
        );
        assert!(METRICS.no_bucket() > before);
        let on = settings(&tmp.0, |_| {});
        assert_eq!(
            st.handle(Frame {
                settings: on,
                image: bytes::Bytes::from_static(b"not an image"),
                record: positive("a", "r2", t(1)),
            }),
            Handled::Dropped(Dropped::Error)
        );
        let e = shared.lock().unwrap().last_error.clone().unwrap();
        assert!(
            e.message.contains("could not read a frame"),
            "{}",
            e.message
        );
    }

    #[test]
    fn the_size_check_ignores_protected_frames_and_prunes_only_when_over() {
        use crate::collect::labels::{self, LabelFile, Status as Reviewed};
        let tmp = Tmp::new();
        let mut st = store();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.max_gb = 1.0;
        });
        let out = st.handle(frame(&s, positive("a", "r1", t(0)), 1));
        let Handled::Saved { rel, .. } = out else {
            panic!("{out:?}")
        };
        let bytes = st.summary.bytes;
        assert_eq!(st.usage.prunable_bytes, bytes);
        let jpg = layout::resolve(&tmp.0, &rel, "jpg").unwrap();
        labels::write(
            &labels::path_of_frame(&jpg),
            &LabelFile {
                status: Reviewed::Reviewed,
                ..LabelFile::default()
            },
        )
        .unwrap();
        // A review moves the frame out of the figure at the next prune.
        st.prune(&s);
        assert_eq!(st.usage.prunable_bytes, 0);
        assert_eq!(st.usage.protected_bytes, bytes);
        assert_eq!(st.usage.protected_frames, 1);
        // A cap below the protected data: with nothing prunable, no prune is due after a write,
        // although the whole folder is over the cap.
        let tiny = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.buckets.positive.rate = 1.0;
            c.max_gb = 1.0;
        });
        st.last_prune = None;
        st.handle(frame(&tiny, positive("b", "r2", t(1)), 2));
        assert!(st.last_prune.is_none(), "no prune below the cap");
        assert!(st.summary.bytes > st.usage.prunable_bytes);
        // A store that starts on the folder knows the split at once.
        let mut fresh = store();
        fresh.use_dir(&tmp.0);
        assert_eq!(fresh.usage.protected_bytes, bytes);
        assert_eq!(fresh.usage.prunable_bytes + bytes, fresh.summary.bytes);
    }

    #[test]
    fn size_cap_prunes_the_oldest_frames_and_the_index() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| {
            c.buckets.positive.per_camera_min_interval_s = 0;
            c.max_gb = 1e-9; // ~1 byte: everything but the newest goes.
        });
        let mut st = store();
        for i in 0..3u32 {
            let out = st.handle(frame(
                &s,
                positive(&format!("c{i}"), &format!("r{i}"), t(u64::from(i))),
                i,
            ));
            assert!(matches!(out, Handled::Saved { .. }), "{out:?}");
            st.last_prune = None;
        }
        let scanned = Summary::scan(&tmp.0);
        assert_eq!(scanned, st.summary);
        assert!(scanned.frames <= 1, "{}", scanned.frames);
        for day in layout::day_dirs(&tmp.0) {
            for l in layout::read_index(&day.join(layout::INDEX_FILE)) {
                assert!(day.join(format!("{}.jpg", l.id)).is_file());
            }
        }
    }

    #[test]
    fn non_jpeg_uploads_are_reencoded() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |_| {});
        let img = RgbImage {
            width: 32,
            height: 24,
            rgb: vec![90; 32 * 24 * 3],
        };
        let mut png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &img.rgb,
            32,
            24,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .unwrap();
        let mut st = store();
        let out = st.handle(Frame {
            settings: s,
            image: png.into(),
            record: positive("p", "r1", t(0)),
        });
        let Handled::Saved { rel, .. } = out else {
            panic!("{out:?}")
        };
        let jpg = layout::resolve(&tmp.0, &rel, "jpg").unwrap();
        assert!(std::fs::read(&jpg).unwrap().starts_with(&[0xFF, 0xD8]));
        let side: Record =
            serde_json::from_slice(&std::fs::read(jpg.with_extension("json")).unwrap()).unwrap();
        assert!(!side.image.original);
    }
}
