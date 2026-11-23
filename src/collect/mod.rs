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
//! ([`record::decide`]), applies sampling, per-camera rate limits and the dHash + block-diff
//! dedup ([`blockdiff`]; both must agree), writes the files atomically, maintains the per-day `index.jsonl` and the in-memory [`layout::Summary`]
//! the Collect tab shows, applies the caps through [`crate::retention::prune_dataset`] and
//! stops on low free space or a full disk until space is back. [`health`] turns its counts into
//! what the pages say.
//!
//! The thread runs at the normal priority: the per-thread below-normal priority helper would
//! live in `backend/priority.rs` (the only place `unsafe` is allowed) and does not exist yet.
//! Its work is bounded by the rate limits: the luma decode (grayscale JPEG), the dHash, the block
//! thumbnail and test, and the writes only happen for frames that passed them.
//!
//! Panics and stops (audit 2026-11 phase 1): every step of the thread (a frame, a prune, a
//! resume, a rescan, the disk re-arm check) runs under `catch_unwind`; a panic is the Collect
//! tab's last error and the thread goes on. A thread found gone anyway is reported and replaced by
//! the next message. [`Collector::stop_for_exit`] stops it between two messages (a prune in hand
//! between two frames, through [`stop_requested`]) and joins it by a deadline; the collector is
//! process-wide, so a later message starts a fresh thread.

pub mod blockdiff;
pub mod config;
pub mod dhash;
pub mod health;
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
use crate::sync::LockExt;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError};
use metrics::{Dropped, METRICS};
use record::{BoxPx, Det};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use tracing::warn;

/// Frames waiting for the collector thread at most (each holds its encoded image).
pub const QUEUE: usize = 16;
/// The collector applies the caps at least this often while it runs.
pub const PRUNE_INTERVAL: Duration = crate::retention::SWEEP_INTERVAL;
/// Most below-threshold detections recorded per frame (highest first).
pub const MAX_BELOW: usize = 20;
/// How long [`Collector::brief_of`] keeps the summary of a folder that is not the collector's.
const BRIEF_TTL: Duration = Duration::from_secs(10);

/// UTC RFC 3339 with milliseconds (`2026-10-22T12:34:56.789Z`).
pub fn utc_rfc3339(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
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
            image_name: crate::static_memory::cap_name(i.image_name).to_string(),
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

/// What the Collect tab reads of the collector.
#[derive(Debug, Clone, Default)]
pub struct Status {
    /// Folder the summary describes (None: nothing collected or scanned yet).
    pub dir: Option<PathBuf>,
    pub summary: Summary,
    /// Why collection stopped on its own (low disk, full disk).
    pub disabled: Option<String>,
    /// How the folder divides up; the collector's when it describes the folder, up to 10 minutes
    /// behind a review. Zero from [`Collector::brief_of`].
    pub usage: crate::retention::DatasetUsage,
    /// When this process last saved a frame (UTC, RFC 3339).
    pub last_saved_at: Option<String>,
    pub last_error: Option<health::LastError>,
}

enum Msg {
    Frame(Box<Frame>),
    Prune(Arc<CollectSettings>),
    Resume,
    /// Rebuild the summary of a folder another writer added frames to (an import).
    Rescan(PathBuf),
}

/// The collector thread of this process, while it runs.
#[derive(Debug)]
struct Running {
    tx: Sender<Msg>,
    /// Set by [`Collector::stop_for_exit`]; the thread's prunes see it too ([`stop_requested`]).
    stop: Arc<AtomicBool>,
    /// None only in tests that stand in for the thread.
    join: Option<std::thread::JoinHandle<()>>,
}

/// Whether the collector thread runs.
#[derive(Debug)]
enum Thread {
    /// Not started yet, or stopped: the next message starts one.
    Idle,
    Running(Running),
    /// It could not be started: frames are dropped with an error until a restart (no retry per
    /// frame, which would warn per frame).
    Failed,
}

/// The process's collector: a lazily started thread and what it publishes.
#[derive(Debug)]
pub struct Collector {
    thread: Mutex<Thread>,
    status: Arc<Mutex<Status>>,
    /// [`Self::brief_of`]'s summaries of folders that are not the collector's, with their age.
    briefs: Mutex<HashMap<PathBuf, (Instant, Summary)>>,
}

impl std::fmt::Debug for Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Msg::Frame(_) => "Frame",
            Msg::Prune(_) => "Prune",
            Msg::Resume => "Resume",
            Msg::Rescan(_) => "Rescan",
        })
    }
}

static GLOBAL: OnceLock<Collector> = OnceLock::new();

/// The process's collector.
pub fn global() -> &'static Collector {
    GLOBAL.get_or_init(Collector::new)
}

/// Hand `frame` to the collector (never blocks; a full queue drops it with a count).
pub fn submit(frame: Frame) {
    METRICS.record_offered();
    global().submit(frame);
}

thread_local! {
    /// The stop flag of the collector thread this code runs on (None on every other thread).
    static STOP: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Whether the collector thread this is called on was asked to stop
/// ([`Collector::stop_for_exit`]); false on any other thread. [`crate::retention::prune_dataset`]
/// asks it between two frames.
pub fn stop_requested() -> bool {
    STOP.with(|s| {
        s.borrow()
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    })
}

/// Give this thread the stop flag [`stop_requested`] reads (the collector thread at its start;
/// tests). None removes it.
pub(crate) fn bind_stop(flag: Option<Arc<AtomicBool>>) {
    STOP.with(|s| *s.borrow_mut() = flag);
}

impl Collector {
    fn new() -> Self {
        Self {
            thread: Mutex::new(Thread::Idle),
            status: Arc::new(Mutex::new(Status::default())),
            briefs: Mutex::new(HashMap::new()),
        }
    }

    /// The process is stopping (audit 2026-11 phase 1): the thread finishes the message in hand
    /// (a frame's save, or a prune up to the frame it is deleting: [`stop_requested`]), drops what
    /// is still queued, and is joined by `deadline`; past it, it is left to finish and a warning
    /// says so. Idempotent.
    ///
    /// The collector is process-wide, so this is a pause, not an end: a frame (or prune, resume)
    /// that arrives afterwards starts a fresh thread, which is how an in-process restart (Windows
    /// service control 131) gets its collector back. A fresh thread prunes only when asked by the
    /// retention sweeper or after [`PRUNE_INTERVAL`], and its disk guard starts armed.
    pub fn stop_for_exit(&self, deadline: Instant) {
        let running = {
            let mut t = self.thread.lock_recover();
            match std::mem::replace(&mut *t, Thread::Idle) {
                Thread::Running(r) => r,
                other => {
                    *t = other;
                    return;
                }
            }
        };
        let Running { tx, stop, join } = running;
        stop.store(true, Ordering::SeqCst);
        // Its sender: an idle thread wakes once the clone a submit may hold for a moment is gone.
        drop(tx);
        let Some(join) = join else { return };
        if crate::retention::join_within(join, deadline, "dataset collector").is_err() {
            warn!(
                "the dataset collector did not stop in time (a slow disk or a long write); it is \
                 left to finish the frame in hand, and the files it writes are atomic"
            );
        }
    }

    /// The running thread's sender, starting the thread when there is none.
    fn sender(&self) -> Option<Sender<Msg>> {
        let mut t = self.thread.lock_recover();
        if matches!(*t, Thread::Idle) {
            *t = match self.spawn() {
                Some(r) => Thread::Running(r),
                None => Thread::Failed,
            };
        }
        match &*t {
            Thread::Running(r) => Some(r.tx.clone()),
            Thread::Idle | Thread::Failed => None,
        }
    }

    fn spawn(&self) -> Option<Running> {
        let (tx, rx) = crossbeam_channel::bounded::<Msg>(QUEUE);
        let status = self.status.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        match std::thread::Builder::new()
            .name("dataset-collector".into())
            .spawn(move || run(&rx, status, flag))
        {
            Ok(join) => Some(Running {
                tx,
                stop,
                join: Some(join),
            }),
            Err(e) => {
                warn!("could not start the dataset collector thread ({e}); frames are not saved");
                None
            }
        }
    }

    /// Record `message` as the collector's last error, for the Collect tab.
    fn note_dead(&self, message: &str) {
        let mut s = self.status.lock_recover();
        s.last_error = Some(health::LastError {
            message: message.to_string(),
            at: utc_rfc3339(SystemTime::now()),
        });
    }

    /// The thread behind `tx` is gone (it should never end on its own: every step is guarded):
    /// forget it, so the next message starts a new one, and say so on the Collect tab.
    fn lost(&self, tx: &Sender<Msg>) {
        let mut t = self.thread.lock_recover();
        if let Thread::Running(r) = &*t
            && r.tx.same_channel(tx)
        {
            *t = Thread::Idle;
        }
        drop(t);
        warn!(
            "the dataset collector thread ended unexpectedly; a new one starts with the next frame"
        );
        self.note_dead(
            "the collector thread stopped unexpectedly, so a frame was dropped; a new one starts \
             with the next frame (please report this with the log)",
        );
    }

    pub fn submit(&self, frame: Frame) {
        let Some(tx) = self.sender() else {
            METRICS.record_drop(Dropped::Error);
            self.note_dead(
                "the collector thread could not start, so no frame is saved; restart the service",
            );
            return;
        };
        match tx.try_send(Msg::Frame(Box::new(frame))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => METRICS.record_drop(Dropped::QueueFull),
            Err(TrySendError::Disconnected(_)) => {
                METRICS.record_drop(Dropped::Error);
                self.lost(&tx);
            }
        }
    }

    /// Apply the caps to `settings.dir` on the collector thread (the retention sweeper's tick).
    /// Started only when the folder exists.
    pub fn request_prune(&self, settings: CollectSettings) {
        if !settings.dir.is_dir() {
            return;
        }
        if let Some(tx) = self.sender()
            && let Err(TrySendError::Disconnected(_)) = tx.try_send(Msg::Prune(Arc::new(settings)))
        {
            self.lost(&tx);
        }
    }

    /// Rebuild the summary of `dir` after frames were written there by an import (only when
    /// the collector thread runs in this process; otherwise the page scans the folder itself).
    pub fn rescan_if_running(&self, dir: &std::path::Path) {
        if let Thread::Running(r) = &*self.thread.lock_recover() {
            let _ = r.tx.try_send(Msg::Rescan(dir.to_path_buf()));
        }
    }

    /// Messages waiting for the collector thread.
    pub fn pending(&self) -> usize {
        match &*self.thread.lock_recover() {
            Thread::Running(r) => r.tx.len(),
            Thread::Idle | Thread::Failed => 0,
        }
    }

    /// Re-arm after a disk stop. False when the request could not be queued (the collector is
    /// busy, or its thread is gone): the stop stands and still shows.
    pub fn resume(&self) -> bool {
        let sent = self
            .sender()
            .is_some_and(|tx| tx.try_send(Msg::Resume).is_ok());
        if sent {
            let mut s = self.status.lock_recover();
            s.disabled = None;
        }
        sent
    }

    /// Everything the collector knows of `dir`: its summary and usage (the collector's when it
    /// describes `dir`, else read from disk now, every label file included; call off the async
    /// runtime), the disk stop reason, the last frame and the last error.
    pub fn status_of(&self, dir: &std::path::Path) -> Status {
        let mut s = self.status_with(dir, &|| Summary::scan(dir));
        if self.describes(dir) {
            return s;
        }
        s.usage = crate::retention::dataset_usage(dir);
        s
    }

    /// [`Self::status_of`] without the walk over the label files and, for a folder that is not the
    /// collector's, with a summary up to [`BRIEF_TTL`] old: Home and the Train strip ask often.
    pub fn brief_of(&self, dir: &std::path::Path) -> Status {
        self.status_with(dir, &|| self.brief_summary(dir, BRIEF_TTL))
    }

    /// The summary of `dir` kept for `ttl`, else a new one. No lock is held during the scan: two
    /// callers at once may both scan, and neither waits for the other's slow disk.
    fn brief_summary(&self, dir: &std::path::Path, ttl: Duration) -> Summary {
        if let Some((at, s)) = self.briefs.lock_recover().get(dir)
            && at.elapsed() < ttl
        {
            return s.clone();
        }
        let s = Summary::scan(dir);
        let mut kept = self.briefs.lock_recover();
        kept.retain(|_, (at, _)| at.elapsed() < ttl);
        kept.insert(dir.to_path_buf(), (Instant::now(), s.clone()));
        s
    }

    /// The collector's status when it describes `dir`, else with `summary`'s.
    fn status_with(&self, dir: &std::path::Path, summary: &dyn Fn() -> Summary) -> Status {
        let s = self.status.lock_recover().clone();
        if s.dir.as_deref() == Some(dir) {
            return s;
        }
        Status {
            dir: Some(dir.to_path_buf()),
            summary: summary(),
            usage: crate::retention::DatasetUsage::default(),
            last_saved_at: None,
            ..s
        }
    }

    fn describes(&self, dir: &std::path::Path) -> bool {
        let s = self.status.lock_recover();
        s.dir.as_deref() == Some(dir)
    }
}

/// Tests: hold the collector thread before it takes the next message (a slow disk).
#[cfg(test)]
pub(crate) static PAUSE: AtomicBool = AtomicBool::new(false);

/// Tests: called on the collector thread before it prunes this folder (a panic, a wait).
#[cfg(test)]
type PruneHook = (PathBuf, fn(&std::path::Path));
#[cfg(test)]
static PRUNE_HOOK: Mutex<Option<PruneHook>> = Mutex::new(None);

/// Run `f` on the store; a panic in it is caught, noted as the collector's last error (`what`
/// names the step) and the thread goes on. False when it panicked.
fn guarded(st: &mut store::Store, what: &str, f: impl FnOnce(&mut store::Store)) -> bool {
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(st))).is_ok();
    if !ok {
        st.note_error(&format!(
            "{what} panicked, so it was skipped; the collector goes on. Please report this with \
             the log"
        ));
    }
    ok
}

/// One message on the collector thread.
fn dispatch(
    st: &mut store::Store,
    msg: Msg,
    last: &mut Option<Arc<CollectSettings>>,
    last_prune: &mut Instant,
) {
    match msg {
        Msg::Frame(f) => {
            let settings = f.settings.clone();
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.handle(*f)));
            if out.is_err() {
                METRICS.record_drop(Dropped::Error);
                st.note_error(
                    "saving a frame panicked, so it was dropped; please report this with the log",
                );
            }
            *last = Some(settings);
        }
        Msg::Prune(s) => {
            guarded(st, "applying the dataset's size and age caps", |st| {
                #[cfg(test)]
                {
                    // Not under the lock: the hook may panic.
                    let hook = PRUNE_HOOK.lock_recover().clone();
                    if let Some((dir, hook)) = hook
                        && dir == s.dir
                    {
                        hook(&dir);
                    }
                }
                st.prune(&s);
            });
            *last_prune = Instant::now();
        }
        Msg::Resume => {
            guarded(st, "resuming collection", store::Store::resume);
        }
        Msg::Rescan(dir) => {
            guarded(st, "reading the dataset folder again", |st| st.rescan(&dir));
        }
    }
}

fn run(rx: &Receiver<Msg>, status: Arc<Mutex<Status>>, stop: Arc<AtomicBool>) {
    bind_stop(Some(stop.clone()));
    let mut st = store::Store::new(status);
    let mut last: Option<Arc<CollectSettings>> = None;
    let mut last_prune = Instant::now();
    loop {
        guarded(
            &mut st,
            "checking the free disk space",
            store::Store::poll_rearm,
        );
        #[cfg(test)]
        while PAUSE.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let msg = match rx.recv_timeout(store::FREE_CHECK_INTERVAL) {
            Ok(m) => Some(m),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        if stop.load(Ordering::SeqCst) {
            // What is still queued is dropped: the process is stopping.
            if let Some(m) = msg {
                tracing::debug!(
                    "dataset collector stopping: {} message(s) dropped",
                    1 + rx.len()
                );
                drop(m);
            }
            return;
        }
        if let Some(m) = msg {
            dispatch(&mut st, m, &mut last, &mut last_prune);
        }
        if last_prune.elapsed() >= PRUNE_INTERVAL
            && let Some(s) = last.clone()
        {
            guarded(&mut st, "applying the dataset's size and age caps", |st| {
                st.prune(&s);
            });
            last_prune = Instant::now();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A collector whose "thread" is the test, reading `tx`'s receiver.
    fn collector(tx: Sender<Msg>, disabled: &str) -> Collector {
        let c = Collector::new();
        c.status.lock().unwrap().disabled = Some(disabled.to_string());
        *c.thread.lock().unwrap() = Thread::Running(Running {
            tx,
            stop: Arc::new(AtomicBool::new(false)),
            join: None,
        });
        c
    }

    #[test]
    fn resume_clears_the_stop_only_when_the_request_was_queued() {
        let (tx, rx) = crossbeam_channel::bounded::<Msg>(1);
        tx.try_send(Msg::Resume).unwrap();
        let c = collector(tx, "low disk");
        // The queue is full: the collector would never hear of it, so the page must not lie.
        assert!(!c.resume());
        assert_eq!(
            c.status.lock().unwrap().disabled.as_deref(),
            Some("low disk")
        );
        rx.try_recv().unwrap();
        assert!(c.resume());
        assert_eq!(c.status.lock().unwrap().disabled, None);
        assert!(matches!(rx.try_recv(), Ok(Msg::Resume)));
        // The thread is gone.
        drop(rx);
        c.status.lock().unwrap().disabled = Some("low disk".into());
        assert!(!c.resume());
        assert!(c.status.lock().unwrap().disabled.is_some());
    }

    /// A fresh temp folder, removed on drop.
    struct Tmp(PathBuf);
    impl Tmp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("bop-collector-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn settings_for(dir: &std::path::Path) -> CollectSettings {
        let c = CollectConfig {
            enabled: true,
            dir: Some(dir.to_path_buf()),
            ..CollectConfig::default()
        };
        CollectSettings::from_config(&c, dir)
    }

    /// A frame with a person in it, for the positive bucket.
    fn person_frame(dir: &std::path::Path, rid: &str) -> Frame {
        let (w, h) = (96u32, 64u32);
        let img = crate::image::RgbImage {
            width: w,
            height: h,
            rgb: (0..w * h)
                .flat_map(|i| {
                    let v = ((i % w) * 37 % 251) as u8;
                    [v, v / 2, 255 - v]
                })
                .collect(),
        };
        let det = record::Det {
            label: "person".into(),
            score: 0.9,
            bbox: [1.0, 1.0, 10.0, 20.0],
        };
        Frame {
            settings: Arc::new(settings_for(dir)),
            image: crate::image::encode_jpeg(&img, 90).unwrap().into(),
            record: Draft {
                at: SystemTime::now(),
                record: Record {
                    camera: "drive".into(),
                    camera_key: "m|cam:drive".into(),
                    request_id: rid.into(),
                    raw: vec![det.clone()],
                    predictions: vec![det],
                    ..record::empty_record()
                },
            },
        }
    }

    /// Wait (20 s at most) until the collector has saved a frame.
    fn wait_saved(c: &Collector) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while c.status.lock().unwrap().last_saved_at.is_none() {
            assert!(Instant::now() < deadline, "no frame was saved");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn panics(_: &std::path::Path) {
        panic!("test: the prune panics");
    }

    /// Audit 2026-11 phase 1 (collect-6 / collect-12): a panic while applying the caps is caught,
    /// shown as the last error, and the thread goes on to save the next frame.
    #[test]
    fn a_panic_in_a_prune_is_caught_and_the_next_frame_is_saved() {
        let (pruned, data) = (Tmp::new(), Tmp::new());
        *PRUNE_HOOK.lock_recover() = Some((pruned.0.clone(), panics));
        let c = Collector::new();
        c.request_prune(settings_for(&pruned.0));
        c.submit(person_frame(&data.0, "abcd1234"));
        wait_saved(&c);
        let error = c.status.lock().unwrap().last_error.clone().unwrap();
        assert!(
            error.message.contains("size and age caps panicked"),
            "{}",
            error.message
        );
        assert!(matches!(*c.thread.lock().unwrap(), Thread::Running(_)));
        *PRUNE_HOOK.lock_recover() = None;
        c.stop_for_exit(Instant::now() + Duration::from_secs(10));
    }

    static ENTERED: AtomicBool = AtomicBool::new(false);
    static STOPPED_PRUNE: Mutex<Option<bool>> = Mutex::new(None);

    /// Waits on the collector thread until it is asked to stop, then prunes (cap 1 byte).
    fn waits_for_the_stop(dir: &std::path::Path) {
        ENTERED.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(20);
        while !stop_requested() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        let r = crate::retention::prune_dataset(dir, 0, 1).unwrap();
        *STOPPED_PRUNE.lock().unwrap() = Some(r.stopped);
    }

    /// A stop reaches a prune in hand between two frames, the thread is joined, and the next
    /// frame starts a fresh thread (an in-process restart).
    #[test]
    fn a_stop_ends_a_prune_between_frames_and_a_later_frame_starts_again() {
        let (pruned, data) = (Tmp::new(), Tmp::new());
        // One frame the cap would delete.
        let mut st =
            store::Store::with_probe(Arc::new(Mutex::new(Status::default())), Box::new(|_| None));
        let saved = st.handle(person_frame(&pruned.0, "oldframe"));
        let store::Handled::Saved { rel, .. } = saved else {
            panic!("{saved:?}")
        };
        let jpg = layout::resolve(&pruned.0, &rel, "jpg").unwrap();
        *PRUNE_HOOK.lock_recover() = Some((pruned.0.clone(), waits_for_the_stop));
        let c = Collector::new();
        c.request_prune(settings_for(&pruned.0));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !ENTERED.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the prune never started");
            std::thread::sleep(Duration::from_millis(2));
        }
        let started = Instant::now();
        c.stop_for_exit(Instant::now() + Duration::from_secs(10));
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "joined, not timed out"
        );
        assert_eq!(*STOPPED_PRUNE.lock().unwrap(), Some(true));
        assert!(jpg.is_file(), "the stop came before the frame was deleted");
        assert!(matches!(*c.thread.lock().unwrap(), Thread::Idle));
        *PRUNE_HOOK.lock_recover() = None;
        // Again, idempotent.
        c.stop_for_exit(Instant::now());
        // A later frame: a fresh thread saves it.
        c.submit(person_frame(&data.0, "efgh5678"));
        wait_saved(&c);
        c.stop_for_exit(Instant::now() + Duration::from_secs(10));
        assert!(matches!(*c.thread.lock().unwrap(), Thread::Idle));
    }

    /// A thread that is gone is noticed by the next frame, which says so and starts a new one.
    #[test]
    fn a_lost_thread_is_reported_and_replaced() {
        let (tx, rx) = crossbeam_channel::bounded::<Msg>(1);
        let c = collector(tx, "x");
        drop(rx);
        let data = Tmp::new();
        c.submit(person_frame(&data.0, "lost0001"));
        let error = c.status.lock().unwrap().last_error.clone().unwrap();
        assert!(
            error.message.contains("stopped unexpectedly"),
            "{}",
            error.message
        );
        assert!(matches!(*c.thread.lock().unwrap(), Thread::Idle));
        c.submit(person_frame(&data.0, "lost0002"));
        wait_saved(&c);
        c.stop_for_exit(Instant::now() + Duration::from_secs(10));
    }

    #[test]
    fn timestamps() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_millis(1_790_000_000_123);
        assert_eq!(utc_rfc3339(t), "2026-09-21T14:13:20.123Z");
    }

    /// M2: a long upload name is cut before it reaches the sidecar (the camera folder has its own,
    /// shorter cap).
    #[test]
    fn a_draft_caps_the_image_name_and_camera_key() {
        let name = format!("{}.jpg", "Ä".repeat(500));
        let d = Draft::new(&DraftInput {
            model: "m",
            device: "openvino:cpu",
            request_id: "abc",
            routed_from: None,
            image_name: &name,
            stats: crate::image_stats::ImageStats::default(),
            width: 100,
            height: 80,
            threshold: 0.5,
            near_margin: None,
        });
        let cap = crate::static_memory::MAX_NAME_BYTES;
        assert!(d.record.image_name.len() <= cap && d.record.image_name.len() > cap - 4);
        assert!(name.starts_with(&d.record.image_name));
        assert!(d.record.camera_key.len() <= cap);
        assert!(d.record.camera.len() <= 64, "the camera folder's own cap");
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

    #[test]
    fn a_brief_of_another_folder_is_kept_for_a_moment_and_a_status_reads_now() {
        let (tx, _rx) = crossbeam_channel::bounded::<Msg>(1);
        let c = collector(tx, "x");
        let dir = std::env::temp_dir().join(format!("bop-brief-{}", uuid::Uuid::new_v4()));
        let day = dir.join("ir/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        assert_eq!(c.brief_of(&dir).summary.frames, 0);
        let line = record::IndexLine {
            id: "a".into(),
            timestamp: "2026-10-22T00:00:00Z".into(),
            scene: "ir".into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: vec![],
            labels: vec![],
            bytes: 1,
        };
        let text = serde_json::to_string(&line).unwrap();
        std::fs::write(day.join(layout::INDEX_FILE), text + "\n").unwrap();
        assert_eq!(c.brief_of(&dir).summary.frames, 0, "kept");
        assert_eq!(
            c.status_of(&dir).summary.frames,
            1,
            "not kept: a page asked"
        );
        // Past its time.
        assert_eq!(c.brief_summary(&dir, Duration::ZERO).frames, 1);
        assert_eq!(c.brief_of(&dir).summary.frames, 1, "replaced");
        // Per folder.
        let other = dir.join("other");
        assert_eq!(c.brief_of(&other).summary.frames, 0);
        assert_eq!(c.brief_of(&dir).summary.frames, 1);
        // The collector's own folder never goes through it.
        c.status.lock().unwrap().dir = Some(dir.clone());
        c.status.lock().unwrap().summary.frames = 7;
        assert_eq!(c.brief_of(&dir).summary.frames, 7);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
