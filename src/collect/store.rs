//! The collector thread's state: bucket selection with per-camera rate limits and sampling,
//! dHash dedup per camera, the atomic writes of image + sidecar + index line, the size / age
//! caps (through [`crate::retention::prune_dataset`]) and the disk guard that stops collection
//! on low free space or a full disk and re-arms it once space is back. Single-threaded: only the
//! collector thread owns a [`Store`], so it is the only writer of the frame files and index files.

use super::health::LastError;
use super::layout::{self, Summary};
use super::metrics::{Dropped, METRICS};
use super::record::{Bucket, IndexLine, RECORD_VERSION};
use super::{CollectSettings, Frame, Status, dhash};
use crate::retention::DatasetUsage;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn};

/// Frames of one camera compared for near duplicates: those saved within this window.
pub const DEDUP_WINDOW: Duration = Duration::from_secs(10 * 60);
/// At most this many recent hashes per camera.
pub const DEDUP_KEEP: usize = 32;
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
    /// camera -> recent (time, dHash) of saved frames.
    recent: HashMap<String, VecDeque<(SystemTime, u64)>>,
    /// camera -> when near duplicates were saved for a bucket that skips the dedup.
    skips: HashMap<String, VecDeque<SystemTime>>,
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
            skips: HashMap::new(),
            last_saved: None,
            last_error: None,
            disabled: None,
            free_checked: None,
            last_prune: None,
            last_warn: None,
            free_probe,
            #[cfg(test)]
            fail_writes: None,
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
        let passing: Vec<Bucket> = matched
            .iter()
            .copied()
            .filter(|b| {
                let c = settings.buckets.get(*b);
                let sampled = f64::from(c.rate) >= 1.0
                    || sample(&record.record.request_id, *b) < f64::from(c.rate);
                let interval = Duration::from_secs(u64::from(c.per_camera_min_interval_s));
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
        let hash = match dhash::of_encoded(&image) {
            Ok((h, _, _)) => h,
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
        recent.retain(|(t, _)| {
            at.duration_since(*t)
                .map_or(true, |since| since <= DEDUP_WINDOW)
        });
        let near_duplicate = recent
            .iter()
            .any(|(_, h)| dhash::hamming(*h, hash) <= settings.dedup_hamming);
        if near_duplicate {
            let skips = self.skips.entry(camera.clone()).or_default();
            skips.retain(|t| {
                at.duration_since(*t)
                    .map_or(true, |since| since < DEDUP_SKIP_WINDOW)
            });
            if !passing.iter().any(|b| b.skips_dedup()) || skips.len() >= DEDUP_SKIPS_PER_CAMERA {
                METRICS.record_drop(Dropped::Dup);
                return Handled::Dropped(Dropped::Dup);
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
        if near_duplicate {
            // Its look-alike stays the one remembered, so the skip does not extend the window.
            self.skips.entry(camera).or_default().push_back(at);
        } else {
            let recent = self.recent.entry(camera).or_default();
            recent.push_back((at, hash));
            while recent.len() > DEDUP_KEEP {
                recent.pop_front();
            }
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
        let r = &mut frame.record;
        let date = super::utc_date(frame.at);
        let stem = format!(
            "{}_{}",
            super::utc_file_ts(frame.at),
            layout::safe_id(&r.request_id)
        );
        let rel = layout::Rel {
            scene: r.scene.clone(),
            camera: r.camera.clone(),
            date: date.clone(),
            stem: stem.clone(),
        };
        let day = dir.join(&rel.scene).join(&rel.camera).join(&date);
        std::fs::create_dir_all(&day)?;
        let (bytes, original) = if image.starts_with(&[0xFF, 0xD8]) {
            (std::borrow::Cow::Borrowed(image), true)
        } else {
            let img = crate::image::decode(image).map_err(std::io::Error::other)?;
            let jpeg = crate::image::encode_jpeg(&img, 92).map_err(std::io::Error::other)?;
            (std::borrow::Cow::Owned(jpeg), false)
        };
        r.id = stem.clone();
        r.image = super::record::ImageInfo {
            file: format!("{stem}.jpg"),
            bytes: bytes.len() as u64,
            original,
        };
        let jpg = day.join(format!("{stem}.jpg"));
        layout::write_atomic(&jpg, &bytes)?;
        let json = serde_json::to_vec_pretty(&*r).map_err(std::io::Error::other)?;
        if let Err(e) = layout::write_atomic(&day.join(format!("{stem}.json")), &json) {
            let _ = std::fs::remove_file(&jpg);
            return Err(e);
        }
        let total = (bytes.len() + json.len()) as u64;
        let line = IndexLine::of(r, total);
        let text = serde_json::to_string(&line).map_err(std::io::Error::other)?;
        layout::append_line(&day.join(layout::INDEX_FILE), &text)?;
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
                if r.deleted > 0 || r.failed > 0 || self.dir.as_deref() != Some(&settings.dir) =>
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

    #[test]
    fn near_duplicates_of_a_camera_are_dropped_within_the_window() {
        let tmp = Tmp::new();
        let s = settings(&tmp.0, |c| c.buckets.positive.per_camera_min_interval_s = 0);
        let mut st = store();
        assert!(matches!(
            st.handle(frame(&s, positive("a", "r1", t(0)), 1)),
            Handled::Saved { .. }
        ));
        assert_eq!(
            st.handle(frame(&s, positive("a", "r2", t(5)), 1)),
            Handled::Dropped(Dropped::Dup)
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
        assert_eq!(st.summary.frames, 4);
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
