//! Dataset collector (`collect` config section, default off): saves frames Blue Iris sends,
//! with a sidecar JSON of what the model saw and did, for the later review, auto-label and
//! training stages. Layout and schema: `docs/DATASET.md`.
//!
//! Flow: the worker's `finish` (or, for verified requests, the verify lane) builds a [`Draft`]
//! from the request (model, device, request id, camera key, scene, raw detections before the
//! rules, suppressed boxes with reasons, the reply's predictions, verifier outcomes), replies,
//! and then hands the draft with the request's own encoded bytes (a cheap `Bytes` clone, no
//! re-encode) to [`submit`]: a `try_send` into a bounded channel. A full channel drops the
//! frame and counts `dataset_dropped_total{reason="queue_full"}`; the inference thread never
//! waits and the reply is never delayed. One collector thread ([`store`]) decides the buckets
//! ([`record::decide`]), applies sampling, per-camera rate limits and dHash dedup, writes the
//! files atomically, maintains the per-day `index.jsonl` and the in-memory [`layout::Summary`]
//! the Dataset page shows, applies the caps through [`crate::retention::prune_dataset`] and
//! disables itself on low free space or a full disk.
//!
//! The thread runs at the normal priority: the per-thread below-normal priority helper would
//! live in `backend/priority.rs` (the only place `unsafe` is allowed) and does not exist yet.
//! Its work is bounded by the rate limits: the dHash decode (grayscale JPEG) and the writes
//! only happen for frames that passed them.

pub mod config;
pub mod dhash;
pub mod labels;
pub mod layout;
pub mod metrics;
pub mod record;
pub mod store;

pub use config::{BucketConfig, Buckets, CollectConfig, CollectSettings};
pub use layout::{Summary, camera_dir};
pub use record::{Bucket, Record, decide};

use crate::model::Detection;
use crate::model::filters::SuppressReason;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError};
use metrics::{Dropped, METRICS};
use record::{BoxPx, Det};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};
use tracing::warn;

/// Frames waiting for the collector thread at most (each holds its encoded image).
pub const QUEUE: usize = 16;
/// The collector applies the caps at least this often while it runs.
pub const PRUNE_INTERVAL: Duration = crate::retention::SWEEP_INTERVAL;
/// Most below-threshold detections recorded per frame (highest first).
pub const MAX_BELOW: usize = 20;

/// UTC RFC 3339 with milliseconds (`2026-10-22T12:34:56.789Z`).
pub fn utc_rfc3339(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

/// Day folder name (UTC `YYYY-MM-DD`).
pub fn utc_date(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%d")
        .to_string()
}

/// File-name timestamp (`20261022T123456789Z`, UTC, milliseconds).
pub fn utc_file_ts(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y%m%dT%H%M%S%3fZ")
        .to_string()
}

fn round1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

/// A detection's box clamped to the `w` x `h` frame, 0.1 px.
fn clamp_box(d: &Detection, w: u32, h: u32) -> BoxPx {
    let (w, h) = (w as f32, h as f32);
    [
        round1(d.x1.clamp(0.0, w)),
        round1(d.y1.clamp(0.0, h)),
        round1(d.x2.clamp(0.0, w)),
        round1(d.y2.clamp(0.0, h)),
    ]
}

fn det_of(d: &Detection, classes: &[String], w: u32, h: u32) -> Det {
    Det {
        label: crate::model::filters::label_of(classes, d.class_id).into_owned(),
        score: (d.score * 10_000.0).round() / 10_000.0,
        bbox: clamp_box(d, w, h),
    }
}

/// A prediction of the reply as a [`Det`].
pub fn det_of_prediction(p: &crate::api::Prediction) -> Det {
    Det {
        label: p.label.clone(),
        score: p.confidence,
        bbox: crate::verify::pred_box(p),
    }
}

/// What the worker knows of a request when it starts the draft.
#[derive(Debug, Clone)]
pub struct DraftInput<'a> {
    pub model: &'a str,
    pub device: &'a str,
    pub request_id: &'a str,
    pub routed_from: Option<&'a str>,
    pub image_name: &'a str,
    pub stats: crate::image_stats::ImageStats,
    pub width: u32,
    pub height: u32,
    pub threshold: f32,
    pub near_margin: Option<f32>,
}

/// A record in the making (the worker fills it; the collector completes it).
#[derive(Debug, Clone)]
pub struct Draft {
    pub at: SystemTime,
    pub record: Record,
}

impl Draft {
    pub fn new(i: &DraftInput<'_>) -> Self {
        let at = SystemTime::now();
        let camera_key = crate::static_memory::camera_key(i.model, i.image_name, i.width, i.height);
        let record = Record {
            request_id: i.request_id.to_string(),
            timestamp: utc_rfc3339(at),
            model: i.model.to_string(),
            device: i.device.to_string(),
            routed_from: i.routed_from.map(str::to_string),
            image_name: i.image_name.to_string(),
            camera: camera_dir(&camera_key, i.width, i.height),
            camera_key,
            scene: i.stats.scene().as_str().to_string(),
            stats: i.stats.into(),
            width: i.width,
            height: i.height,
            threshold: i.threshold,
            near_margin: i.near_margin,
            ..record::empty_record()
        };
        Self { at, record }
    }

    /// The model's detections (at or above the threshold, before any rule), those just below
    /// it, and what the rules removed (labels resolved with `classes`).
    pub fn set_detections(
        &mut self,
        raw: &[Detection],
        below: &[Detection],
        removed: &[(Detection, SuppressReason)],
        classes: &[String],
    ) {
        let (w, h) = (self.record.width, self.record.height);
        self.record.raw = raw.iter().map(|d| det_of(d, classes, w, h)).collect();
        let mut below: Vec<Det> = below.iter().map(|d| det_of(d, classes, w, h)).collect();
        below.sort_by(|a, b| b.score.total_cmp(&a.score));
        below.truncate(MAX_BELOW);
        self.record.below_threshold = below;
        self.record.suppressed = removed
            .iter()
            .map(|(d, r)| {
                let x = det_of(d, classes, w, h);
                record::Suppressed {
                    label: x.label,
                    score: x.score,
                    bbox: x.bbox,
                    reason: r.as_str().to_string(),
                }
            })
            .collect();
    }

    /// Add the verifier's outcomes; its rejected candidates join `suppressed` with reason
    /// `verify`.
    pub fn set_verify(&mut self, v: record::VerifyInfo) {
        for c in v.candidates.iter().filter(|c| c.outcome == "rejected") {
            self.record.suppressed.push(record::Suppressed {
                label: c.label.clone(),
                score: c.score,
                bbox: c.bbox,
                reason: SuppressReason::Verify.as_str().to_string(),
            });
        }
        self.record.verify = Some(v);
    }

    /// The reply's predictions.
    pub fn set_predictions(&mut self, preds: &[crate::api::Prediction]) {
        self.record.predictions = preds.iter().map(det_of_prediction).collect();
    }
}

/// A finished draft with the image and the settings it was taken under.
#[derive(Debug)]
pub struct Frame {
    pub settings: Arc<CollectSettings>,
    /// The encoded image as the request carried it.
    pub image: bytes::Bytes,
    pub record: Draft,
}

/// What the Dataset page reads of the collector.
#[derive(Debug, Clone, Default)]
pub struct Status {
    /// Folder the summary describes (None: nothing collected or scanned yet).
    pub dir: Option<PathBuf>,
    pub summary: Summary,
    /// Why collection stopped on its own (low disk, full disk).
    pub disabled: Option<String>,
}

enum Msg {
    Frame(Box<Frame>),
    Prune(Arc<CollectSettings>),
    Resume,
}

/// The process's collector: a lazily started thread and what it publishes.
#[derive(Debug)]
pub struct Collector {
    tx: OnceLock<Option<Sender<Msg>>>,
    status: Arc<Mutex<Status>>,
}

impl std::fmt::Debug for Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Msg::Frame(_) => "Frame",
            Msg::Prune(_) => "Prune",
            Msg::Resume => "Resume",
        })
    }
}

static GLOBAL: OnceLock<Collector> = OnceLock::new();

/// The process's collector.
pub fn global() -> &'static Collector {
    GLOBAL.get_or_init(|| Collector {
        tx: OnceLock::new(),
        status: Arc::new(Mutex::new(Status::default())),
    })
}

/// Hand `frame` to the collector (never blocks; a full queue drops it with a count).
pub fn submit(frame: Frame) {
    global().submit(frame);
}

impl Collector {
    fn sender(&self) -> Option<&Sender<Msg>> {
        self.tx
            .get_or_init(|| {
                let (tx, rx) = crossbeam_channel::bounded::<Msg>(QUEUE);
                let status = self.status.clone();
                match std::thread::Builder::new()
                    .name("dataset-collector".into())
                    .spawn(move || run(&rx, status))
                {
                    Ok(_) => Some(tx),
                    Err(e) => {
                        warn!("could not start the dataset collector thread ({e}); frames are not saved");
                        None
                    }
                }
            })
            .as_ref()
    }

    pub fn submit(&self, frame: Frame) {
        let Some(tx) = self.sender() else {
            METRICS.record_drop(Dropped::Error);
            return;
        };
        match tx.try_send(Msg::Frame(Box::new(frame))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => METRICS.record_drop(Dropped::QueueFull),
            Err(TrySendError::Disconnected(_)) => METRICS.record_drop(Dropped::Error),
        }
    }

    /// Apply the caps to `settings.dir` on the collector thread (the retention sweeper's tick).
    /// Started only when the folder exists.
    pub fn request_prune(&self, settings: CollectSettings) {
        if !settings.dir.is_dir() {
            return;
        }
        if let Some(tx) = self.sender() {
            let _ = tx.try_send(Msg::Prune(Arc::new(settings)));
        }
    }

    /// Messages waiting for the collector thread.
    pub fn pending(&self) -> usize {
        self.tx
            .get()
            .and_then(Option::as_ref)
            .map_or(0, Sender::len)
    }

    /// Re-arm after a disk stop.
    pub fn resume(&self) {
        if let Some(tx) = self.sender() {
            let _ = tx.try_send(Msg::Resume);
        }
        let mut s = self.status.lock().unwrap_or_else(|e| e.into_inner());
        s.disabled = None;
    }

    /// The summary of `dir` (the collector's when it describes `dir`, else scanned now; call
    /// off the async runtime) and the disk stop reason.
    pub fn status_of(&self, dir: &std::path::Path) -> Status {
        let s = self
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if s.dir.as_deref() == Some(dir) {
            return s;
        }
        Status {
            dir: Some(dir.to_path_buf()),
            summary: Summary::scan(dir),
            disabled: s.disabled,
        }
    }
}

/// Tests: hold the collector thread before it takes the next message (a slow disk).
#[cfg(test)]
pub(crate) static PAUSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn run(rx: &Receiver<Msg>, status: Arc<Mutex<Status>>) {
    let mut st = store::Store::new(status);
    let mut last: Option<Arc<CollectSettings>> = None;
    let mut last_prune = std::time::Instant::now();
    loop {
        #[cfg(test)]
        while PAUSE.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let msg = match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(m) => Some(m),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        match msg {
            Some(Msg::Frame(f)) => {
                let settings = f.settings.clone();
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.handle(*f)));
                if out.is_err() {
                    METRICS.record_drop(Dropped::Error);
                    warn!("dataset collector: saving a frame panicked; the frame was dropped");
                }
                last = Some(settings);
            }
            Some(Msg::Prune(s)) => {
                st.prune(&s);
                last_prune = std::time::Instant::now();
            }
            Some(Msg::Resume) => st.resume(),
            None => {}
        }
        if last_prune.elapsed() >= PRUNE_INTERVAL
            && let Some(s) = &last
        {
            st.prune(s);
            last_prune = std::time::Instant::now();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn timestamps() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_millis(1_790_000_000_123);
        assert_eq!(utc_rfc3339(t), "2026-09-21T14:13:20.123Z");
        assert_eq!(utc_date(t), "2026-09-21");
        assert_eq!(utc_file_ts(t), "20260921T141320123Z");
    }

    #[test]
    fn draft_records_detections_with_labels_and_clamped_boxes() {
        let stats = crate::image_stats::ImageStats {
            is_ir: true,
            ..Default::default()
        };
        let mut d = Draft::new(&DraftInput {
            model: "m",
            device: "openvino:cpu",
            request_id: "abc",
            routed_from: Some("primary"),
            image_name: "Driveway.20240101_120000.jpg",
            stats,
            width: 100,
            height: 80,
            threshold: 0.5,
            near_margin: Some(0.1),
        });
        assert_eq!(d.record.scene, "ir");
        assert_eq!(d.record.camera, "driveway");
        assert_eq!(d.record.camera_key, "m|cam:driveway");
        let det = |x1: f32, score: f32, class_id: usize| Detection {
            x1,
            y1: -5.0,
            x2: 150.0,
            y2: 40.04,
            score,
            class_id,
        };
        let classes = vec!["person".to_string()];
        let below: Vec<Detection> = (0..30)
            .map(|i| det(1.0, 0.4 + i as f32 / 1000.0, 0))
            .collect();
        d.set_detections(
            &[det(10.0, 0.8, 0)],
            &below,
            &[(det(20.0, 0.55, 3), SuppressReason::Aspect)],
            &classes,
        );
        assert_eq!(d.record.raw[0].label, "person");
        assert_eq!(d.record.raw[0].bbox, [10.0, 0.0, 100.0, 40.0]);
        assert_eq!(d.record.below_threshold.len(), MAX_BELOW);
        assert!(d.record.below_threshold[0].score >= d.record.below_threshold[1].score);
        assert_eq!(d.record.suppressed[0].label, "class_3");
        assert_eq!(d.record.suppressed[0].reason, "aspect");
        d.set_verify(record::VerifyInfo {
            verifier: "v".into(),
            mode: "crop".into(),
            candidates: vec![record::VerifyCandidate {
                label: "person".into(),
                score: 0.8,
                bbox: [1.0, 2.0, 3.0, 4.0],
                outcome: "rejected".into(),
            }],
            ..Default::default()
        });
        assert_eq!(d.record.suppressed.len(), 2);
        assert_eq!(d.record.suppressed[1].reason, "verify");
    }
}
