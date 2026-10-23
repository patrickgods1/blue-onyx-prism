//! Per-model inference thread.
//!
//! Each configured model gets one OS thread that owns its
//! [`Backend`](crate::backend::Backend) (the infer requests / sessions are created on that
//! thread) and receives work over a bounded crossbeam channel. Loading walks the device
//! candidates of the model's [`DeviceSpec`] (e.g. GPU, then CPU): a candidate is accepted only
//! once it compiles, creates its requests and passes the warm-up inference. The channel is
//! created before the model is compiled so the HTTP layer can route to it immediately; the
//! effective queue depth is a "soft capacity" computed after the warm-up inference
//! (`request_timeout / warmup_ms`, clamped to 1..=64) and checked by the server through
//! [`WorkerHandle::is_full`]. Flow modeled on blue-onyx `worker.rs` (MIT).
//!
//! The worker is pipelined (docs/REVIEW_PLAN.md, phase 2): decode + preprocess run on the
//! registry's shared [`PrepPool`], and up to
//! [`Backend::slots`](crate::backend::Backend::slots) inferences run at once (the hardware
//! profile decides how many per device, see [`crate::profile`]). The loop is a small state
//! machine that waits with `crossbeam_channel::Select` on the work channel, the prep reply of
//! the oldest request and the backend's completions (plus a tick for the shutdown token), so it
//! neither spins nor sleeps while work is ready. Requests leave preprocessing and are answered
//! in arrival order (FIFO per model); a request that fails before inference (bad image,
//! timeout) is answered in its turn without disturbing the slots. Postprocessing stays
//! on this thread.
//!
//! Layout: this module holds the public surface (work items, [`WorkerConfig`], [`LiveSettings`],
//! [`WorkerHandle`], spawning); `run` is the thread's load + run loop, `pipeline` the request
//! bookkeeping it drives, and `watch` the stuck-inference and slow-compile watchdogs.

use crate::api::{VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::{DeviceInfo, DeviceSpec, LoadRequest, Runtimes};
use crate::config::{Config, ModelConfig};
use crate::metrics::{ModelGauges, ModelMetrics, ModelStateLabel};
use crate::prep::PrepPool;
use crate::profile::{Overrides, Profile};
use crate::startup::{ModelState, StateHandle};
use crate::sync::{LockExt, RwLockExt};
use arc_swap::ArcSwap;
use crossbeam_channel::Sender;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

mod pipeline;
mod run;
#[cfg(test)]
mod run_tests;
mod verify;
mod watch;

use run::WorkerCtx;

/// A message for a worker thread.
#[derive(Debug)]
pub enum WorkItem {
    /// A queued detection request: the request, where to send the reply, and when it arrived.
    Detect(
        VisionDetectionRequest,
        tokio::sync::oneshot::Sender<VisionDetectionResponse>,
        Instant,
    ),
    /// [`WorkItem::Detect`] with a request id chosen by the caller (see [`new_request_id`]),
    /// so the server can return the same id in a response header; `Detect` gets one generated
    /// by the worker.
    DetectWithId(
        VisionDetectionRequest,
        tokio::sync::oneshot::Sender<VisionDetectionResponse>,
        Instant,
        String,
    ),
    /// A request whose image another worker already decoded (see [`Decoded`]): it skips
    /// decode and is never forwarded again.
    DetectDecoded(Box<Decoded>),
    /// Compile a lazy model now (sent once by [`WorkerHandle::trigger_load`]).
    Load,
}

/// Where a worker sends the answer of a request.
#[derive(Debug)]
pub enum Reply {
    /// Straight to the HTTP handler waiting for it.
    Client(tokio::sync::oneshot::Sender<VisionDetectionResponse>),
    /// To another worker thread (e.g. a cascade that merges a verifier's answer into its own):
    /// a crossbeam channel it can `Select` on.
    Worker(Sender<VisionDetectionResponse>),
}

impl Reply {
    /// Send `resp`; false when the receiver is gone (the client went away).
    pub fn send(self, resp: VisionDetectionResponse) -> bool {
        match self {
            Reply::Client(tx) => tx.send(resp).is_ok(),
            Reply::Worker(tx) => tx.try_send(resp).is_ok(),
        }
    }
}

impl From<tokio::sync::oneshot::Sender<VisionDetectionResponse>> for Reply {
    fn from(tx: tokio::sync::oneshot::Sender<VisionDetectionResponse>) -> Self {
        Reply::Client(tx)
    }
}

/// A request handed from one worker to another with its image already decoded: the night
/// model switch (an IR / dark frame goes to the model's `night_model`) and the crops (or frame)
/// sent to a verifier model ([`Purpose`]). The receiving worker preprocesses `img` for its own
/// model, runs it with its own live settings and the scene of `stats`, and answers `reply` in
/// its own request order. Never forwarded again.
#[derive(Debug)]
pub struct Decoded {
    pub img: crate::image::RgbImage,
    /// Scene statistics of `img` (computed by the sender at decode).
    pub stats: crate::image_stats::ImageStats,
    pub image_name: String,
    pub min_confidence: f32,
    /// The request id of the original request (same id in both workers' log lines).
    pub request_id: String,
    /// When the original request arrived (round-trip and queue times count from here).
    pub enqueued: Instant,
    /// The original request's timeout budget: the receiver drops it after
    /// `min(deadline, enqueued + its own request timeout)`.
    pub deadline: Instant,
    /// Time the sender already spent decoding (part of `processMs`).
    pub decode: Duration,
    pub reply: Reply,
    /// Name of the forwarding model (for the log).
    pub from: String,
    pub purpose: Purpose,
}

/// Why a [`Decoded`] request was handed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// The night model switch: the receiver answers the client like the sender would (its own
    /// threshold, night rules, object filter, image saving and verifier).
    NightModel,
    /// A verifier-cascade check ([`crate::verify`]): the receiver reports its raw detections
    /// at `min_confidence` (no night rules, static memory, object filter, image saving or
    /// verifier of its own), in the coordinates of `img`.
    Verify,
}

/// The served workers of a registry generation by normalized name, for workers that forward
/// requests to other workers ([`Decoded`]). Holds weak references: a retired or reloaded worker
/// drops out as soon as the registry lets go of it, and a lookup never keeps one alive. The
/// registry republishes it whenever its set of workers changes.
#[derive(Debug, Clone, Default)]
pub struct Peers(Arc<RwLock<std::collections::HashMap<String, std::sync::Weak<WorkerHandle>>>>);

impl Peers {
    pub fn new() -> Self {
        Self::default()
    }

    /// The served worker named `name` (case-insensitive, `.onnx` / `.xml` ignored).
    pub fn get(&self, name: &str) -> Option<Arc<WorkerHandle>> {
        self.0
            .read_recover()
            .get(&crate::registry::normalize_name(name))
            .and_then(std::sync::Weak::upgrade)
    }

    /// Replace the directory with `handles`.
    pub fn publish<'a>(&self, handles: impl IntoIterator<Item = &'a Arc<WorkerHandle>>) {
        let map = handles
            .into_iter()
            .map(|h| (crate::registry::normalize_name(&h.name), Arc::downgrade(h)))
            .collect();
        *self.0.write_recover() = map;
    }

    /// Empty the directory (registry shutdown).
    pub fn clear(&self) {
        self.0.write_recover().clear();
    }
}

/// A short request id for the per-request log line (8 hex digits, random).
pub fn new_request_id() -> String {
    let mut id = uuid::Uuid::new_v4().simple().to_string();
    id.truncate(8);
    id
}

/// Hard capacity of the channel when the queue is auto-sized.
pub(crate) const MAX_AUTO_QUEUE: usize = 64;
/// Largest `worker_queue_size` accepted (config validation) and applied at spawn; the bounded
/// channel allocates every slot up front.
pub const MAX_QUEUE_SIZE: usize = 1024;

/// How often a blocked worker re-checks the shutdown token.
const POLL: Duration = Duration::from_millis(250);

/// Everything a worker needs to load and run one model. Paths are already resolved.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub name: String,
    pub model: ModelConfig,
    /// Where to run; expanded into candidates by `Runtimes::plan`.
    pub device: DeviceSpec,
    pub load: LoadRequest,
    pub classes: Vec<String>,
    /// Settings applied without a reload (threshold, filter, timeout, image saving); the
    /// registry swaps them when the config changes and the worker reads them per request.
    pub live: Arc<ArcSwap<LiveSettings>>,
    /// 0 = auto-size from the warm-up inference time.
    pub queue_size: usize,
    /// Compile on the first request instead of at startup.
    pub lazy: bool,
    /// Reported as `canUseGPU` (some GPU-class device option of any runtime can run).
    pub can_use_gpu: bool,
    /// Downloads to wait for before loading (phase 7.3); progress goes into the model state.
    /// `classes` is empty then and read from disk once the files are there.
    pub wait: Option<crate::resources::provision::Wait>,
    /// Hardware profile and explicit settings that decide the compile hint and the number of
    /// inferences in flight per device; None = one at a time with the latency hint.
    pub concurrency: Option<Concurrency>,
    /// The generation's decode + preprocess pool (None: the worker starts a one-thread pool).
    pub prep: Option<Arc<PrepPool>>,
    /// The generation's other workers, for the night model switch (empty: never forwards).
    pub peers: Peers,
}

/// The per-model settings a config change applies without reloading the model (per-model
/// overrides already resolved). The worker takes one snapshot when a request arrives and uses
/// it for that request's timeout, postprocessing and image saving.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveSettings {
    pub confidence_threshold: f32,
    pub nms_iou: f32,
    /// Only report these labels (case-insensitive); empty = all.
    pub object_filter: Vec<String>,
    /// Save annotated images here.
    pub save_image_path: Option<PathBuf>,
    pub save_ref_image: bool,
    /// How long a request may wait in the queue plus processing.
    pub request_timeout: Duration,
    /// Night / IR false-positive rules applied before the object filter.
    pub filters: crate::model::filters::FilterRules,
    /// Static-box memory settings (the memory itself lives in the worker).
    pub static_suppression: crate::static_memory::StaticSuppression,
    /// Image enhancement modes, passed to the prep pool with each request.
    pub preprocess: crate::model::enhance::PreprocessModes,
    /// Served model that night / IR frames are forwarded to (`night_model`); None = run here.
    pub night_model: Option<String>,
    /// Verifier cascade (`verify`, the model's own over the global one; None = off, and
    /// always None for the verifier model itself).
    pub verify: Option<crate::verify::VerifyConfig>,
}

impl LiveSettings {
    /// The settings of model `m` under `config` (the model's threshold and object filter win
    /// over the global ones).
    pub fn for_model(config: &Config, m: &ModelConfig) -> Self {
        Self {
            confidence_threshold: m
                .confidence_threshold
                .unwrap_or(config.confidence_threshold),
            nms_iou: config.nms_iou,
            object_filter: m
                .object_filter
                .clone()
                .unwrap_or_else(|| config.object_filter.clone()),
            save_image_path: config.save_image_path.as_deref().map(crate::resolve_path),
            save_ref_image: config.save_ref_image,
            request_timeout: config.request_timeout(),
            filters: filter_rules(config, m),
            static_suppression: m
                .static_suppression
                .clone()
                .unwrap_or_else(|| config.static_suppression.clone()),
            preprocess: config.preprocess_modes(m),
            night_model: m
                .night_model
                .as_deref()
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(str::to_string),
            verify: verify_settings(config, m),
        }
    }

    /// A shareable cell holding `self` (what [`WorkerConfig::live`] and the handle share).
    pub fn shared(self) -> Arc<ArcSwap<LiveSettings>> {
        Arc::new(ArcSwap::from_pointee(self))
    }
}

/// The verifier settings of model `m`: its own `verify` over the global one; None when off,
/// without a model, or naming `m` itself (the global default never makes a model verify
/// itself).
pub fn verify_settings(config: &Config, m: &ModelConfig) -> Option<crate::verify::VerifyConfig> {
    use crate::registry::normalize_name;
    let v = m.verify.as_ref().or(config.verify.as_ref())?;
    let target = normalize_name(v.model.trim());
    (v.enabled && !v.model.trim().is_empty() && target != normalize_name(&m.effective_name()))
        .then(|| v.clone())
}

/// The night / IR rules of model `m`: its own values over the global ones; the label maps are
/// merged (the model's entries win).
pub fn filter_rules(config: &Config, m: &ModelConfig) -> crate::model::filters::FilterRules {
    let merged = |global: &std::collections::BTreeMap<String, f32>,
                  own: &Option<std::collections::BTreeMap<String, f32>>| {
        let mut out = global.clone();
        for (k, v) in own.iter().flatten() {
            out.retain(|g, _| !g.eq_ignore_ascii_case(k));
            out.insert(k.clone(), *v);
        }
        out
    };
    crate::model::filters::FilterRules {
        night_confidence_threshold: m
            .night_confidence_threshold
            .or(config.night_confidence_threshold),
        label_thresholds: merged(&config.label_thresholds, &m.label_thresholds),
        night_label_thresholds: merged(&config.night_label_thresholds, &m.night_label_thresholds),
        person_aspect: m.person_aspect.or(config.person_aspect),
        min_box_area_frac: m.min_box_area_frac.or(config.min_box_area_frac),
        suppress_overlaps: m
            .suppress_overlaps
            .clone()
            .unwrap_or_else(|| config.suppress_overlaps.clone()),
    }
}

impl Default for LiveSettings {
    fn default() -> Self {
        Self::for_model(&Config::default(), &ModelConfig::default())
    }
}

/// What decides a model's concurrency on the device it loads on.
#[derive(Debug, Clone)]
pub struct Concurrency {
    pub profile: Profile,
    pub overrides: Overrides,
}

/// One-shot "this worker is done with its startup load" signal, used by the registry to make
/// workers compile strictly in config order.
#[derive(Clone, Debug, Default)]
pub struct LoadGate(Arc<(Mutex<bool>, Condvar)>);

impl LoadGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&self) {
        let (lock, cv) = &*self.0;
        let mut g = lock.lock_recover();
        *g = true;
        cv.notify_all();
    }

    pub fn is_open(&self) -> bool {
        *self.0.0.lock_recover()
    }

    /// Block until the gate opens or `shutdown` is cancelled. Returns true when opened.
    pub fn wait(&self, shutdown: &CancellationToken) -> bool {
        let (lock, cv) = &*self.0;
        let mut g = lock.lock_recover();
        while !*g {
            if shutdown.is_cancelled() {
                return false;
            }
            g = cv
                .wait_timeout(g, POLL)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

/// HTTP-side handle to a worker thread.
#[derive(Debug)]
pub struct WorkerHandle {
    pub name: String,
    pub sender: Sender<WorkItem>,
    pub state: StateHandle,
    pub metrics: Arc<ModelMetrics>,
    pub device: Arc<RwLock<Option<DeviceInfo>>>,
    pub join: Option<JoinHandle<()>>,
    /// Model compiles on first request (state stays `Initializing` until then).
    pub lazy: bool,
    /// The device spec the registry resolved for this model (`force_cpu` and `auto` applied
    /// as configured; None for handles that never got that far).
    pub planned: Option<DeviceSpec>,
    /// Display label of the device the model is headed for (the first runnable load candidate,
    /// e.g. "ONNX Runtime CoreML"); set by the registry. Shown as `"<label> (loading)"` until the
    /// model has loaded.
    pub planned_label: Option<String>,
    soft_capacity: Arc<AtomicUsize>,
    gate: LoadGate,
    load_started: Arc<AtomicBool>,
    live: Arc<ArcSwap<LiveSettings>>,
    /// Set while an inference is stuck (see [`stuck_message`]); the thread may never return.
    stuck: Arc<AtomicBool>,
    /// The model's class names (empty while unknown: a model waiting for its download), so a
    /// live object filter can be checked against them.
    classes: Vec<String>,
}

/// The labels of `filter` that name no class of `classes` (case-insensitive); empty when
/// `classes` is unknown. A filter made only of such labels answers "0 objects" to every
/// request, so the caller warns about them.
pub fn unknown_filter_labels(filter: &[String], classes: &[String]) -> Vec<String> {
    if classes.is_empty() {
        return Vec::new();
    }
    filter
        .iter()
        .filter(|f| !classes.iter().any(|c| c.eq_ignore_ascii_case(f)))
        .cloned()
        .collect()
}

/// Warn when `filter` holds labels that are not among `classes` (see
/// [`unknown_filter_labels`]).
pub(crate) fn warn_unknown_filter(name: &str, filter: &[String], classes: &[String]) {
    let unknown = unknown_filter_labels(filter, classes);
    if unknown.is_empty() {
        return;
    }
    if unknown.len() == filter.len() {
        warn!(
            model = %name,
            labels = ?unknown,
            classes = ?classes,
            "object_filter names no class of this model: every request will answer 0 objects; \
             fix the labels (the model's class names are listed) or clear the filter"
        );
    } else {
        warn!(
            model = %name,
            labels = ?unknown,
            "object_filter has labels that are not classes of this model; they never match"
        );
    }
}

impl WorkerHandle {
    /// A handle for a model that could not even be set up (missing file, bad classes file).
    /// It is `Failed` from the start and has no thread.
    pub fn failed(
        name: impl Into<String>,
        msg: impl Into<String>,
        metrics: Arc<ModelMetrics>,
    ) -> Self {
        let (sender, _rx) = crossbeam_channel::bounded(1);
        let state = StateHandle::new();
        state.set(ModelState::Failed(msg.into()));
        let gate = LoadGate::new();
        gate.open();
        Self {
            name: name.into(),
            sender,
            state,
            metrics,
            device: Arc::new(RwLock::new(None)),
            join: None,
            lazy: false,
            planned: None,
            planned_label: None,
            soft_capacity: Arc::new(AtomicUsize::new(0)),
            gate,
            load_started: Arc::new(AtomicBool::new(true)),
            live: LiveSettings::default().shared(),
            stuck: Arc::new(AtomicBool::new(false)),
            classes: Vec::new(),
        }
    }

    /// True while the model's inference is stuck (the state is `Failed` with
    /// `stuck_message`): its thread may be blocked in the runtime for good, so it must not be
    /// joined.
    pub fn is_stuck(&self) -> bool {
        self.stuck.load(Ordering::Acquire)
    }

    /// The flag behind [`Self::is_stuck`], for code that outlives the handle.
    pub fn stuck_flag(&self) -> Arc<AtomicBool> {
        self.stuck.clone()
    }

    /// How long a request may wait in the queue plus processing (`request_timeout_secs`, live);
    /// the server waits this plus a short grace for the worker's reply.
    pub fn request_timeout(&self) -> Duration {
        self.live.load().request_timeout
    }

    /// The live settings in effect now (see [`LiveSettings`]).
    pub fn live(&self) -> Arc<LiveSettings> {
        self.live.load_full()
    }

    /// Swap in new live settings; requests that arrive from now on use them. True when they
    /// differ from the previous ones.
    pub fn set_live(&self, settings: LiveSettings) -> bool {
        let new = Arc::new(settings);
        let old = self.live.swap(new.clone());
        if old.object_filter != new.object_filter {
            warn_unknown_filter(&self.name, &new.object_filter, &self.classes);
        }
        *old != *new
    }

    /// Ask a lazy model that has not started compiling to compile now. Only the first call
    /// sends [`WorkItem::Load`] (compare-and-swap on `load_started`); returns true when this call
    /// sent it. Never blocks: when the message cannot be queued the flag is reset so a later
    /// request tries again.
    pub fn trigger_load(&self) -> bool {
        if !self.lazy
            || self
                .load_started
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return false;
        }
        match self.sender.try_send(WorkItem::Load) {
            Ok(()) => true,
            Err(e) => {
                warn!(model = %self.name, "could not start the lazy load: {e}");
                if matches!(e, crossbeam_channel::TrySendError::Full(_)) {
                    self.load_started.store(false, Ordering::Release);
                }
                false
            }
        }
    }

    /// True when the queue holds at least the (soft) capacity computed after warm-up.
    pub fn is_full(&self) -> bool {
        self.sender.is_full() || self.sender.len() >= self.soft_capacity.load(Ordering::Relaxed)
    }

    /// Effective queue depth limit.
    pub fn queue_capacity(&self) -> usize {
        self.soft_capacity.load(Ordering::Relaxed)
    }

    /// Opens once this worker has finished its startup work (loaded, failed, or deferred as lazy).
    pub fn load_gate(&self) -> LoadGate {
        self.gate.clone()
    }

    /// True for a lazy model that has not started compiling (the server triggers its load with
    /// [`Self::trigger_load`] on the first request).
    pub fn accepts_while_initializing(&self) -> bool {
        self.lazy && !self.load_started.load(Ordering::Acquire)
    }

    /// Lifecycle state for metrics: a lazy model that has not started compiling is `Lazy`.
    pub fn state_label(&self) -> ModelStateLabel {
        match self.state.get() {
            ModelState::Ready => ModelStateLabel::Ready,
            ModelState::Failed(_) => ModelStateLabel::Failed,
            ModelState::Initializing if self.accepts_while_initializing() => ModelStateLabel::Lazy,
            ModelState::Initializing => ModelStateLabel::Initializing,
        }
    }

    /// Snapshot of the live per-model values exported to Prometheus.
    pub fn gauges(&self) -> ModelGauges {
        ModelGauges {
            name: self.name.clone(),
            device: self.metrics.device(),
            provider: self.metrics.execution_provider(),
            state: self.state_label(),
            queue_length: self.sender.len(),
            queue_capacity: self.queue_capacity(),
        }
    }

    /// Execution provider string of the loaded model; before that, where it is headed:
    /// `"<planned label> (loading)"` ("not loaded" for a model that failed before planning).
    pub fn execution_provider(&self) -> String {
        if let Some(p) = loaded_provider(&self.device) {
            return p;
        }
        let failed = matches!(self.state.get(), ModelState::Failed(_));
        match (&self.planned_label, &self.planned) {
            (Some(l), _) if failed => format!("{l} (not loaded)"),
            (Some(l), _) => format!("{l} (loading)"),
            (None, Some(s)) if failed => format!("{} (not loaded)", planned_spec_label(s)),
            (None, Some(s)) => format!("{} (loading)", planned_spec_label(s)),
            (None, None) => "not loaded".to_string(),
        }
    }

    /// Where the model is headed, for messages: the planned label, else the planned spec,
    /// else "its device".
    pub fn planned_device_text(&self) -> String {
        match (&self.planned_label, &self.planned) {
            (Some(l), _) => l.clone(),
            (None, Some(s)) => planned_spec_label(s),
            (None, None) => "its device".to_string(),
        }
    }

    /// Record where the model is headed (see [`Self::planned_label`]).
    pub fn set_planned(&mut self, spec: DeviceSpec, label: Option<String>) {
        self.planned = Some(spec);
        self.planned_label = label;
    }

    /// Wait for the thread to exit (drop the sender first so it can see the disconnect).
    pub fn join(mut self) {
        let join = self.join.take();
        drop(self);
        if let Some(j) = join
            && j.join().is_err()
        {
            error!("worker thread panicked");
        }
    }
}

/// Prefix of the state detail while a candidate compiles ("compiling on openvino:gpu").
pub const COMPILING_ON: &str = "compiling on";

/// "Model 'x' is initializing (compiling on openvino:gpu)" (detail omitted when unknown).
pub fn initializing_message(name: &str, detail: Option<&str>) -> String {
    match detail {
        Some(d) => format!("Model '{name}' is initializing ({d})"),
        None => format!("Model '{name}' is initializing"),
    }
}

/// Generic display text for a device spec when no option label is known: "ONNX Runtime CoreML",
/// "OpenVINO GPU.1", "auto".
pub(crate) fn planned_spec_label(spec: &DeviceSpec) -> String {
    match spec.device() {
        None => "auto".to_string(),
        Some(d) => {
            let mut s = format!("{} {}", d.runtime.display_name(), d.target.display_name());
            if let Some(i) = d.index {
                s.push_str(&format!(".{i}"));
            }
            s
        }
    }
}

/// `clamp(timeout_ms / max(warmup_ms, 1), 1, 64)`.
pub(crate) fn auto_queue_size(request_timeout: Duration, warmup_ms: u64) -> usize {
    let t = millis_u64(request_timeout);
    let n = (t / warmup_ms.max(1)).clamp(1, MAX_AUTO_QUEUE as u64);
    usize::try_from(n).unwrap_or(MAX_AUTO_QUEUE)
}

/// Request `min_confidence` overrides the configured threshold when > 0. The result is never
/// below [`crate::model::nms::MIN_CONFIDENCE_FLOOR`] (0.01): a tiny request value or a configured
/// 0.0 uses the floor, so a client cannot turn every anchor into an NMS candidate (PL-01).
/// A value above 1 is a percentage (`40` = 0.4; CodeProject.AI takes a fraction, but a client
/// that sends percent would otherwise get "0 objects" on every request without any error);
/// above 100 it is ignored (configured threshold).
pub(crate) fn effective_threshold(min_confidence: f32, configured: f32) -> f32 {
    let floor = crate::model::nms::MIN_CONFIDENCE_FLOOR;
    let t = if min_confidence > 0.0 && min_confidence.is_finite() {
        if min_confidence <= 1.0 {
            min_confidence
        } else if min_confidence <= 100.0 {
            min_confidence / 100.0
        } else {
            configured
        }
    } else {
        configured
    };
    if t.is_finite() && t >= floor {
        t
    } else {
        floor
    }
}

/// Spawn a worker that starts loading immediately.
pub fn spawn_worker(
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    shutdown: CancellationToken,
) -> WorkerHandle {
    spawn_worker_after(runtimes, cfg, metrics, shutdown, None)
}

/// Spawn a worker that waits for `after` to open before touching the runtimes, so models
/// compile in a deterministic order.
///
/// A lazy worker does not take part in the startup order: it starts serving its channel at once
/// (so its first request compiles it as soon as the runtimes mutex is free, without waiting for
/// every earlier model) and its handle's [`WorkerHandle::load_gate`] simply forwards `after`, so
/// the next model in the config still waits for the previous non-lazy one.
pub fn spawn_worker_after(
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    shutdown: CancellationToken,
    after: Option<LoadGate>,
) -> WorkerHandle {
    let hard_cap = if cfg.queue_size > 0 {
        let clamped = cfg.queue_size.clamp(1, MAX_QUEUE_SIZE);
        if clamped != cfg.queue_size {
            warn!(
                model = %cfg.name,
                requested = cfg.queue_size,
                applied = clamped,
                "worker_queue_size is above the maximum; clamped"
            );
        }
        clamped
    } else {
        MAX_AUTO_QUEUE
    };
    let (sender, receiver) = crossbeam_channel::bounded::<WorkItem>(hard_cap);
    let state = StateHandle::new();
    let device = Arc::new(RwLock::new(None));
    let soft_capacity = Arc::new(AtomicUsize::new(hard_cap));
    let gate = LoadGate::new();
    let handle_gate = if cfg.lazy {
        after.clone().unwrap_or_else(|| {
            let g = LoadGate::new();
            g.open();
            g
        })
    } else {
        gate.clone()
    };
    let after = if cfg.lazy { None } else { after };
    let load_started = Arc::new(AtomicBool::new(false));
    let stuck = Arc::new(AtomicBool::new(false));

    let ctx = WorkerCtx {
        runtimes,
        cfg: cfg.clone(),
        metrics: metrics.clone(),
        state: state.clone(),
        device: device.clone(),
        soft_capacity: soft_capacity.clone(),
        gate: gate.clone(),
        load_started: load_started.clone(),
        stuck: stuck.clone(),
        shutdown,
        consecutive_errors: std::cell::Cell::new(0),
        consecutive_degenerate: std::cell::Cell::new(0),
        static_memory: std::cell::RefCell::new(crate::static_memory::StaticMemory::new()),
        decoded_pre: std::cell::RefCell::new(None),
        verify_lane: std::cell::RefCell::new(None),
    };
    let name = cfg.name.clone();
    let join = std::thread::Builder::new()
        .name(format!("worker-{name}"))
        .spawn(move || ctx.run(receiver, after));
    let join = match join {
        Ok(j) => Some(j),
        Err(e) => {
            let msg = format!(
                "could not start the model's worker thread ({e}); the system is out of threads or \
                 memory; free resources or restart the service"
            );
            error!(model = %name, "{msg}");
            state.set(ModelState::Failed(msg));
            gate.open();
            None
        }
    };
    WorkerHandle {
        name,
        sender,
        state,
        metrics,
        device,
        join,
        lazy: cfg.lazy,
        planned: Some(cfg.device),
        planned_label: None,
        soft_capacity,
        gate: handle_gate,
        load_started,
        live: cfg.live.clone(),
        stuck,
        classes: cfg.classes,
    }
}

/// Target of the trace events that record request order (`accepted` / `replied`, field
/// `image`); `tests/pipeline.rs` checks FIFO order with them.
pub const ORDER_TARGET: &str = "blue_onyx_prism::worker::order";

/// Execution provider string of the device the model loaded on (None before it has loaded).
/// `device` is the slot shared by [`WorkerHandle`] and the worker.
fn loaded_provider(device: &RwLock<Option<DeviceInfo>>) -> Option<String> {
    device
        .read_recover()
        .as_ref()
        .map(DeviceInfo::execution_provider)
}

/// Elapsed milliseconds as the API's i32 (saturating).
fn ms(d: Duration) -> i32 {
    i32::try_from(d.as_millis()).unwrap_or(i32::MAX)
}

/// Milliseconds as u64 (saturating; a `Duration` can exceed u64 milliseconds).
fn millis_u64(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::Receiver;

    #[test]
    fn queue_sizing() {
        let t = Duration::from_secs(15);
        assert_eq!(auto_queue_size(t, 100), 64);
        assert_eq!(auto_queue_size(t, 500), 30);
        assert_eq!(auto_queue_size(t, 0), 64);
        assert_eq!(auto_queue_size(Duration::from_secs(1), 5000), 1);
        assert_eq!(auto_queue_size(Duration::from_millis(1000), 300), 3);
    }

    #[test]
    fn threshold_override() {
        assert_eq!(effective_threshold(0.0, 0.5), 0.5);
        assert_eq!(effective_threshold(0.3, 0.5), 0.3);
        assert_eq!(effective_threshold(-1.0, 0.5), 0.5);
        assert_eq!(effective_threshold(f32::NAN, 0.5), 0.5);
        // PL-01: never below the floor, whether from the request or the config.
        let floor = crate::model::nms::MIN_CONFIDENCE_FLOOR;
        assert_eq!(floor, 0.01);
        assert_eq!(effective_threshold(0.000_001, 0.5), floor);
        assert_eq!(effective_threshold(0.0, 0.0), floor);
        assert_eq!(effective_threshold(0.0, 0.005), floor);
        assert_eq!(effective_threshold(f32::INFINITY, f32::NAN), floor);
        assert_eq!(effective_threshold(0.01, 0.5), 0.01);
        assert_eq!(effective_threshold(0.0, 0.02), 0.02);
        // Percent from a client that does not send a fraction; nonsense above 100 ignored.
        assert_eq!(effective_threshold(1.0, 0.5), 1.0);
        assert_eq!(effective_threshold(40.0, 0.5), 0.4);
        assert_eq!(effective_threshold(100.0, 0.5), 1.0);
        assert_eq!(effective_threshold(101.0, 0.5), 0.5);
        assert_eq!(effective_threshold(1e9, 0.5), 0.5);
    }

    #[test]
    fn unknown_filter_labels_are_found() {
        let classes = ["person".to_string(), "Car".to_string()];
        assert!(unknown_filter_labels(&[], &classes).is_empty());
        assert!(unknown_filter_labels(&["PERSON".into(), "car".into()], &classes).is_empty());
        assert_eq!(
            unknown_filter_labels(&["person".into(), "dog".into(), "cat".into()], &classes),
            ["dog", "cat"]
        );
        // Unknown classes: nothing to check against.
        assert!(unknown_filter_labels(&["dog".into()], &[]).is_empty());
    }

    #[test]
    fn gate_and_cancel() {
        let g = LoadGate::new();
        assert!(!g.is_open());
        let token = CancellationToken::new();
        let g2 = g.clone();
        let t2 = token.clone();
        let h = std::thread::spawn(move || g2.wait(&t2));
        std::thread::sleep(Duration::from_millis(20));
        g.open();
        assert!(h.join().unwrap());

        let g = LoadGate::new();
        let t3 = token.clone();
        let g3 = g.clone();
        let h = std::thread::spawn(move || g3.wait(&t3));
        token.cancel();
        assert!(!h.join().unwrap());
    }

    #[test]
    fn failed_handle() {
        let m = Arc::new(ModelMetrics::new("x", "CPU", ""));
        let h = WorkerHandle::failed("x", "missing file", m);
        assert_eq!(h.state.get(), ModelState::Failed("missing file".into()));
        assert!(h.is_full());
        assert!(!h.accepts_while_initializing());
        assert!(h.load_gate().is_open());
        h.join();
    }

    /// A lazy handle with no thread; the test reads its channel.
    fn lazy_handle(cap: usize) -> (WorkerHandle, Receiver<WorkItem>) {
        let (sender, rx) = crossbeam_channel::bounded(cap);
        let mut h = WorkerHandle::failed(
            "lazy",
            "unused",
            Arc::new(ModelMetrics::new("lazy", "CPU", "")),
        );
        h.sender = sender;
        h.state = StateHandle::new();
        h.lazy = true;
        h.load_started = Arc::new(AtomicBool::new(false));
        (h, rx)
    }

    #[test]
    fn trigger_load_is_idempotent() {
        let (h, rx) = lazy_handle(4);
        assert!(h.accepts_while_initializing());
        assert!(h.trigger_load());
        assert!(!h.accepts_while_initializing());
        assert!(!h.trigger_load());
        assert!(!h.trigger_load());
        assert_eq!(rx.len(), 1);
        assert!(matches!(rx.try_recv(), Ok(WorkItem::Load)));

        // Concurrent triggers send exactly one Load.
        let (h, rx) = lazy_handle(64);
        let h = Arc::new(h);
        let sent: usize = (0..8)
            .map(|_| {
                let h = h.clone();
                std::thread::spawn(move || h.trigger_load())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum();
        assert_eq!(sent, 1);
        assert_eq!(rx.len(), 1);

        // A full channel does not block and leaves the trigger armed for the next request.
        let (h, rx) = lazy_handle(1);
        let (tx, _r) = tokio::sync::oneshot::channel();
        h.sender
            .try_send(WorkItem::Detect(
                VisionDetectionRequest::default(),
                tx,
                Instant::now(),
            ))
            .unwrap();
        assert!(!h.trigger_load());
        assert!(h.accepts_while_initializing());
        drop(rx.try_recv());
        assert!(h.trigger_load());

        // Eager models and failed handles never send Load.
        let m = Arc::new(ModelMetrics::new("x", "CPU", ""));
        assert!(!WorkerHandle::failed("x", "boom", m).trigger_load());
    }

    #[test]
    fn handle_reports_stuck() {
        let m = Arc::new(ModelMetrics::new("x", "CPU", ""));
        let h = WorkerHandle::failed("x", "boom", m);
        assert!(!h.is_stuck());
        h.stuck_flag().store(true, Ordering::Release);
        assert!(h.is_stuck());
        h.join();
    }

    #[test]
    fn request_ids_are_short_and_distinct() {
        let a = new_request_id();
        let b = new_request_id();
        assert_eq!(a.len(), 8);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()), "{a}");
        assert_ne!(a, b);
    }

    #[test]
    fn initializing_text() {
        assert_eq!(
            initializing_message("m", Some("compiling on openvino:gpu")),
            "Model 'm' is initializing (compiling on openvino:gpu)"
        );
        assert_eq!(initializing_message("m", None), "Model 'm' is initializing");
    }

    #[test]
    fn provider_text_before_load_names_the_planned_device() {
        use crate::backend::spec::parse;
        let m = Arc::new(ModelMetrics::new("x", "auto", ""));
        let mut h = WorkerHandle::failed("x", "boom", m);
        assert_eq!(h.execution_provider(), "not loaded");
        h.state.set(ModelState::Initializing);
        h.set_planned(parse("ort:coreml").unwrap(), None);
        assert_eq!(h.execution_provider(), "ONNX Runtime CoreML (loading)");
        h.set_planned(parse("openvino:gpu.1").unwrap(), None);
        assert_eq!(h.execution_provider(), "OpenVINO GPU.1 (loading)");
        h.set_planned(
            DeviceSpec::Auto,
            Some("ONNX Runtime CoreML (Apple M1)".into()),
        );
        assert_eq!(
            h.execution_provider(),
            "ONNX Runtime CoreML (Apple M1) (loading)"
        );
        h.state.set(ModelState::Failed("x".into()));
        assert_eq!(
            h.execution_provider(),
            "ONNX Runtime CoreML (Apple M1) (not loaded)"
        );
        h.join();
    }

    #[test]
    fn live_settings_resolve_night_rules_per_model() {
        use std::collections::BTreeMap;
        let mut c = Config {
            night_confidence_threshold: Some(0.6),
            label_thresholds: BTreeMap::from([
                ("Person".to_string(), 0.5),
                ("car".to_string(), 0.7),
            ]),
            person_aspect: Some([0.2, 0.9]),
            ..Config::default()
        };
        c.static_suppression.enabled = true;
        let plain = ModelConfig::default();
        let l = LiveSettings::for_model(&c, &plain);
        assert_eq!(l.filters.night_confidence_threshold, Some(0.6));
        assert_eq!(l.filters.person_aspect, Some([0.2, 0.9]));
        assert_eq!(l.filters.label_thresholds.len(), 2);
        assert!(l.static_suppression.enabled);
        let own = ModelConfig {
            night_confidence_threshold: Some(0.8),
            label_thresholds: Some(BTreeMap::from([("person".to_string(), 0.65)])),
            suppress_overlaps: Some(Vec::new()),
            static_suppression: Some(crate::static_memory::StaticSuppression::default()),
            ..ModelConfig::default()
        };
        let l = LiveSettings::for_model(&c, &own);
        assert_eq!(l.filters.night_confidence_threshold, Some(0.8));
        // Merged: the model's `person` replaces the global `Person`, `car` stays.
        assert_eq!(
            l.filters.label_thresholds,
            BTreeMap::from([("car".to_string(), 0.7), ("person".to_string(), 0.65)])
        );
        assert_eq!(l.filters.person_aspect, Some([0.2, 0.9]));
        assert!(!l.static_suppression.enabled);
        // Defaults: nothing to do.
        assert!(LiveSettings::default().filters.is_noop());
        assert!(!LiveSettings::default().static_suppression.enabled);
    }
}
