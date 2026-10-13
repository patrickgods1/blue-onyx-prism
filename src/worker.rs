//! Per-model inference thread.
//!
//! Each configured model gets one OS thread that owns its [`Backend`] (the infer requests /
//! sessions are created on that thread) and receives work over a bounded crossbeam channel.
//! Loading walks the device candidates of the model's [`DeviceSpec`] (e.g. GPU, then CPU): a
//! candidate is accepted only once it compiles, creates its requests and passes the warm-up
//! inference. The channel is created before the model is compiled so the HTTP layer can route
//! to it immediately; the effective queue depth is a "soft capacity" computed after the warm-up
//! inference (`request_timeout / warmup_ms`, clamped to 1..=64) and checked by the server
//! through [`WorkerHandle::is_full`]. Flow modeled on blue-onyx `worker.rs` (MIT).
//!
//! The worker is pipelined (docs/REVIEW_PLAN.md, phase 2): decode + preprocess run on the
//! registry's shared [`PrepPool`], and up to [`Backend::slots`] inferences run at once (the
//! hardware profile decides how many per device, see [`crate::profile`]). The loop is a small
//! state machine that waits with `crossbeam_channel::Select` on the work channel, the prep
//! reply of the oldest request and the backend's completions (plus a tick for the shutdown
//! token), so it neither spins nor sleeps while work is ready. Requests leave preprocessing and
//! are answered in arrival order (FIFO per model); a request that fails before inference (bad
//! image, timeout) is answered in its turn without disturbing the slots. Postprocessing stays
//! on this thread.

use crate::api::{VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::{
    Backend, Candidate, Completion, DeviceInfo, DeviceSpec, LoadRequest, Runtimes,
};
use crate::config::{Config, ModelConfig};
use crate::metrics::{DropReason, ModelGauges, ModelMetrics, ModelStateLabel};
use crate::model::{Family, PostParams, PreprocessCtx};
use crate::prep::{PrepJob, PrepPool, PrepSpec, Prepared};
use crate::profile::{DeviceTuning, Overrides, Profile};
use crate::startup::{ModelState, StateHandle};
use crate::sync::LockExt;
use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Select, Sender, TryRecvError};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

/// A message for a worker thread.
pub enum WorkItem {
    /// A queued detection request: the request, where to send the reply, and when it arrived.
    Detect(
        VisionDetectionRequest,
        tokio::sync::oneshot::Sender<VisionDetectionResponse>,
        Instant,
    ),
    /// Compile a lazy model now (sent once by [`WorkerHandle::trigger_load`]).
    Load,
}

/// Hard capacity of the channel when the queue is auto-sized.
pub const MAX_AUTO_QUEUE: usize = 64;
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
        }
    }

    /// A shareable cell holding `self` (what [`WorkerConfig::live`] and the handle share).
    pub fn shared(self) -> Arc<ArcSwap<LiveSettings>> {
        Arc::new(ArcSwap::from_pointee(self))
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
    /// e.g. "ONNX Runtime CoreML"); set by the registry. Shown as "<label> (loading)" until the
    /// model has loaded.
    pub planned_label: Option<String>,
    soft_capacity: Arc<AtomicUsize>,
    gate: LoadGate,
    load_started: Arc<AtomicBool>,
    live: Arc<ArcSwap<LiveSettings>>,
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
        }
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
    /// "<planned label> (loading)" ("not loaded" for a model that failed before planning).
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
pub fn planned_spec_label(spec: &DeviceSpec) -> String {
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
pub fn auto_queue_size(request_timeout: Duration, warmup_ms: u64) -> usize {
    let t = request_timeout.as_millis() as u64;
    (t / warmup_ms.max(1)).clamp(1, MAX_AUTO_QUEUE as u64) as usize
}

/// Request `min_confidence` overrides the configured threshold when > 0.
pub fn effective_threshold(min_confidence: f32, configured: f32) -> f32 {
    if min_confidence > 0.0 && min_confidence.is_finite() {
        min_confidence
    } else {
        configured
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
        cfg.queue_size
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

    let ctx = WorkerCtx {
        runtimes,
        cfg: cfg.clone(),
        metrics: metrics.clone(),
        state: state.clone(),
        device: device.clone(),
        soft_capacity: soft_capacity.clone(),
        gate: gate.clone(),
        load_started: load_started.clone(),
        shutdown,
    };
    let name = cfg.name.clone();
    let join = std::thread::Builder::new()
        .name(format!("worker-{name}"))
        .spawn(move || ctx.run(receiver, after));
    let join = match join {
        Ok(j) => Some(j),
        Err(e) => {
            let msg = format!("could not spawn worker thread: {e}");
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
    }
}

/// Model state owned by the worker thread after a successful load.
struct Engine {
    backend: Backend,
    family: Box<dyn Family>,
    /// Geometry the prep pool preprocesses this model's images for.
    spec: PrepSpec,
    provider: String,
}

/// Where a request is between intake and reply.
enum Stage {
    /// Waiting for the prep pool.
    Prep(Receiver<Result<Prepared>>),
    /// Decoded and preprocessed, waiting for a free slot.
    Ready(Prepared),
    /// Already answered (dropped or failed); replied in order.
    Answer(VisionDetectionResponse),
}

/// Who asked and when.
struct Meta {
    reply: tokio::sync::oneshot::Sender<VisionDetectionResponse>,
    enqueued: Instant,
    image_name: String,
    min_confidence: f32,
    /// The live settings when the request arrived (used for its whole life).
    live: Arc<LiveSettings>,
}

struct Job {
    meta: Meta,
    stage: Stage,
}

/// A request whose inference was submitted into a slot.
struct InFlight {
    meta: Meta,
    ctx: PreprocessCtx,
    img: Option<crate::image::RgbImage>,
    decode: Duration,
    preprocess: Duration,
    seq: u64,
    submitted: Instant,
    /// When the backend reported it finished.
    done: Option<Instant>,
}

/// Replies in request order: a slot to collect, or an answer decided earlier.
enum Done {
    Slot(usize),
    Answer(Box<(Meta, VisionDetectionResponse)>),
}

/// The worker's requests between intake and reply. Requests leave `pending` in FIFO order and
/// are answered in the order they entered `order`, so replies keep the request order.
#[derive(Default)]
struct Pipeline {
    pending: VecDeque<Job>,
    order: VecDeque<Done>,
    slots: Vec<Option<InFlight>>,
    /// Round-robin start for the next free slot.
    next: usize,
}

impl Pipeline {
    fn resize(&mut self, slots: usize) {
        self.slots.resize_with(slots.max(1), || None);
    }

    fn free_slot(&mut self) -> Option<usize> {
        let n = self.slots.len();
        let i = (0..n)
            .map(|k| (self.next + k) % n)
            .find(|&i| self.slots[i].is_none())?;
        self.next = (i + 1) % n;
        Some(i)
    }

    fn busy(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.order.is_empty() && !self.busy()
    }

    /// Record a backend completion (ignores stale ones, e.g. of the warm-up).
    fn complete(&mut self, c: Completion) -> bool {
        match self.slots.get_mut(c.slot) {
            Some(Some(f)) if f.seq == c.seq && f.done.is_none() => {
                f.done = Some(c.at);
                true
            }
            _ => false,
        }
    }
}

/// What woke the worker.
enum Event {
    Work(Result<WorkItem, crossbeam_channel::RecvError>),
    Prep(Result<Result<Prepared>, crossbeam_channel::RecvError>),
    Done(Result<Completion, crossbeam_channel::RecvError>),
    Tick,
}

/// Target of the trace events that record request order (`accepted` / `replied`, field
/// `image`); `tests/pipeline.rs` checks FIFO order with them.
pub const ORDER_TARGET: &str = "blue_onyx_prism::worker::order";

struct WorkerCtx {
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    state: StateHandle,
    device: Arc<RwLock<Option<DeviceInfo>>>,
    soft_capacity: Arc<AtomicUsize>,
    gate: LoadGate,
    load_started: Arc<AtomicBool>,
    shutdown: CancellationToken,
}

/// Execution provider string of the device the model loaded on (None before it has loaded, or
/// when the lock is poisoned). `device` is the slot shared by [`WorkerHandle`] and the worker.
fn loaded_provider(device: &RwLock<Option<DeviceInfo>>) -> Option<String> {
    device
        .read()
        .ok()
        .and_then(|d| d.as_ref().map(|d| d.execution_provider()))
}

/// Elapsed milliseconds as the API's i32.
fn ms(d: Duration) -> i32 {
    d.as_millis().min(i32::MAX as u128) as i32
}

impl WorkerCtx {
    fn run(mut self, rx: Receiver<WorkItem>, after: Option<LoadGate>) {
        let name = self.cfg.name.clone();
        if let Some(wait) = self.cfg.wait.take() {
            // Do not hold up the models after this one while downloading.
            self.gate.open();
            if !self.wait_for_downloads(&wait) {
                return;
            }
            if self.cfg.classes.is_empty() {
                let classes = self.cfg.model.classes.clone();
                match crate::model::classes::resolve_class_names(
                    &self.cfg.model.path,
                    classes.as_deref(),
                ) {
                    Ok(c) => self.cfg.classes = c,
                    Err(e) => {
                        let msg = format!("{e:#}");
                        error!(model = %name, "{msg}");
                        self.state.set(ModelState::Failed(msg));
                        return;
                    }
                }
            }
        } else if let Some(prev) = after
            && !prev.wait(&self.shutdown)
        {
            self.gate.open();
            return;
        }

        let mut engine: Option<Engine> = None;
        if self.cfg.lazy {
            info!(model = %name, "lazy model: compiling on first request");
            self.gate.open();
        } else {
            engine = self.try_load();
            self.gate.open();
        }

        let prep = self.cfg.prep.clone().unwrap_or_else(|| PrepPool::new(1));
        let mut p = Pipeline::default();
        if let Some(e) = &engine {
            p.resize(e.backend.slots());
        }
        let tick = crossbeam_channel::tick(POLL);
        let mut closed = false;
        loop {
            if self.shutdown.is_cancelled() {
                break;
            }
            // Everything that can happen without waiting.
            loop {
                let mut progressed = false;
                if let Some(eng) = engine.as_mut() {
                    while let Ok(c) = eng.backend.completions().try_recv() {
                        progressed |= p.complete(c);
                    }
                    progressed |= self.reply_front(eng, &mut p);
                    progressed |= self.fill_slots(eng, &mut p);
                }
                while !closed && self.has_room(engine.as_ref(), &p, &prep) {
                    match rx.try_recv() {
                        Ok(item) => {
                            self.accept(item, &mut engine, &mut p, &prep);
                            progressed = true;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => closed = true,
                    }
                }
                if !progressed || self.shutdown.is_cancelled() {
                    break;
                }
            }
            if self.shutdown.is_cancelled() || (closed && p.is_empty()) {
                break;
            }

            // Wait for whatever can move the pipeline next.
            let event = {
                let mut sel = Select::new();
                let i_work =
                    (!closed && self.has_room(engine.as_ref(), &p, &prep)).then(|| sel.recv(&rx));
                let head_prep = match p.pending.front() {
                    Some(Job {
                        stage: Stage::Prep(r),
                        ..
                    }) if p.slots.iter().any(Option::is_none) => Some(r),
                    _ => None,
                };
                let i_prep = head_prep.map(|r| sel.recv(r));
                let done_rx = engine.as_ref().map(|e| e.backend.completions());
                let i_done = done_rx.filter(|_| p.busy()).map(|r| sel.recv(r));
                let i_tick = sel.recv(&tick);
                let op = sel.select();
                let i = op.index();
                if Some(i) == i_work {
                    Event::Work(op.recv(&rx))
                } else if Some(i) == i_prep
                    && let Some(r) = head_prep
                {
                    Event::Prep(op.recv(r))
                } else if Some(i) == i_done
                    && let Some(r) = done_rx
                {
                    Event::Done(op.recv(r))
                } else {
                    debug_assert_eq!(i, i_tick);
                    let _ = op.recv(&tick);
                    Event::Tick
                }
            };
            match event {
                Event::Work(Ok(item)) => self.accept(item, &mut engine, &mut p, &prep),
                Event::Work(Err(_)) => closed = true,
                Event::Prep(r) => {
                    if let Some(job) = p.pending.front_mut() {
                        job.stage = self.prep_stage(r, &job.meta);
                    }
                }
                Event::Done(Ok(c)) => {
                    p.complete(c);
                }
                Event::Done(Err(_)) => {}
                Event::Tick => self.unstick(&mut p),
            }
        }
        if let Some(eng) = engine.as_mut()
            && p.busy()
        {
            // Never leave an inference running past the worker (shutdown).
            eng.backend.abort();
        }
        self.answer_remaining(p, &rx);
        info!(model = %name, "worker exiting");
    }

    /// Shutdown: answer every request still in the pipeline or queued in the channel at once
    /// (instead of letting its HTTP handler wait for the request timeout).
    fn answer_remaining(&self, p: Pipeline, rx: &Receiver<WorkItem>) {
        let msg = || format!("Model '{}' is shutting down", self.cfg.name);
        let mut metas: Vec<Meta> = Vec::new();
        for d in p.order {
            match d {
                Done::Slot(_) => {}
                Done::Answer(a) => metas.push(a.0),
            }
        }
        metas.extend(p.slots.into_iter().flatten().map(|f| f.meta));
        metas.extend(p.pending.into_iter().map(|j| j.meta));
        for m in metas {
            let _ = m.reply.send(self.error_response(msg(), 0, 0));
        }
        while let Ok(item) = rx.try_recv() {
            if let WorkItem::Detect(_, reply, _) = item {
                let _ = reply.send(self.error_response(msg(), 0, 0));
            }
        }
    }

    /// Another request may enter preprocessing: always while not loaded (answered at once),
    /// else while fewer than slots + pool threads (at most 4) wait.
    fn has_room(&self, engine: Option<&Engine>, p: &Pipeline, prep: &PrepPool) -> bool {
        match engine {
            None => true,
            Some(_) => p.pending.len() < p.slots.len() + prep.threads().clamp(1, 4),
        }
    }

    /// Handle one work item: a lazy load, or a detection request that enters the pipeline (or
    /// is answered at once when the model is not loaded).
    fn accept(
        &self,
        item: WorkItem,
        engine: &mut Option<Engine>,
        p: &mut Pipeline,
        prep: &PrepPool,
    ) {
        let name = &self.cfg.name;
        let (req, reply, enqueued) = match item {
            WorkItem::Load => {
                if engine.is_none() && !matches!(self.state.get(), ModelState::Failed(_)) {
                    // Lazy model: compile now; requests meanwhile are answered
                    // "initializing" by the server.
                    *engine = self.try_load();
                    if let Some(e) = engine.as_ref() {
                        p.resize(e.backend.slots());
                    }
                }
                return;
            }
            WorkItem::Detect(req, reply, enqueued) => (req, reply, enqueued),
        };
        let Some(eng) = engine.as_ref() else {
            // Only reachable for a lazy model (between the server's trigger and the state
            // update) or after a failed load: answer at once, never block.
            let msg = match self.state.get() {
                ModelState::Failed(msg) => {
                    self.metrics.record_drop(DropReason::Failed);
                    format!("Model '{name}' failed to load: {msg}")
                }
                _ => {
                    self.metrics.record_drop(DropReason::Initializing);
                    initializing_message(name, self.state.detail().as_deref())
                }
            };
            let _ = reply.send(self.error_response(msg, 0, 0));
            return;
        };
        trace!(target: ORDER_TARGET, model = %name, image = %req.image_name, "accepted");
        let meta = Meta {
            reply,
            enqueued,
            image_name: req.image_name,
            min_confidence: req.min_confidence,
            live: self.cfg.live.load_full(),
        };
        let stage = if let Some(resp) = self.timed_out(&meta) {
            Stage::Answer(resp)
        } else {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let job = PrepJob {
                image: req.image_data,
                spec: eng.spec,
                keep_image: meta.live.save_image_path.is_some(),
                reply: tx,
            };
            match prep.submit(job) {
                Ok(()) => Stage::Prep(rx),
                Err(e) => Stage::Answer(self.failure(&meta, &eng.provider, &e)),
            }
        };
        p.pending.push_back(Job { meta, stage });
    }

    /// The drop answer when `meta` waited longer than the request timeout.
    fn timed_out(&self, meta: &Meta) -> Option<VisionDetectionResponse> {
        let waited = meta.enqueued.elapsed();
        let timeout = meta.live.request_timeout;
        if waited <= timeout {
            return None;
        }
        self.metrics.record_drop(DropReason::Timeout);
        warn!(
            model = %self.cfg.name,
            waited_ms = waited.as_millis() as u64,
            timeout_ms = timeout.as_millis() as u64,
            "dropping request that waited longer than the request timeout (server overloaded)"
        );
        Some(self.error_response(
            format!(
                "Request dropped: waited {} ms in queue (timeout {} ms)",
                waited.as_millis(),
                timeout.as_millis()
            ),
            0,
            0,
        ))
    }

    /// The error answer for a request that failed before inference (bad image, ...).
    fn failure(&self, meta: &Meta, provider: &str, e: &anyhow::Error) -> VisionDetectionResponse {
        warn!(model = %self.cfg.name, image = %meta.image_name, "detection failed: {e:#}");
        let mut r = self.error_response(format!("{e:#}"), 0, 0);
        r.executionProvider = provider.to_string();
        r
    }

    /// The stage of a request whose prep reply arrived.
    fn prep_stage(
        &self,
        r: Result<Result<Prepared>, crossbeam_channel::RecvError>,
        meta: &Meta,
    ) -> Stage {
        let provider = loaded_provider(&self.device).unwrap_or_default();
        match r {
            Ok(Ok(prepared)) => Stage::Ready(prepared),
            Ok(Err(e)) => Stage::Answer(self.failure(meta, &provider, &e)),
            Err(_) => Stage::Answer(self.failure(
                meta,
                &provider,
                &anyhow::anyhow!("the preprocessing thread stopped"),
            )),
        }
    }

    /// Move requests from the head of `pending` into free slots (or into `order` when they
    /// were answered already). True when something moved.
    fn fill_slots(&self, eng: &mut Engine, p: &mut Pipeline) -> bool {
        let mut moved = false;
        while let Some(job) = p.pending.front_mut() {
            if let Stage::Prep(rx) = &job.stage {
                let r = match rx.try_recv() {
                    Ok(r) => Ok(r),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => Err(crossbeam_channel::RecvError),
                };
                job.stage = self.prep_stage(r, &job.meta);
                moved = true;
            }
            let slot = match &job.stage {
                Stage::Answer(_) => None,
                Stage::Ready(_) => match p.free_slot() {
                    Some(s) => Some(s),
                    None => break,
                },
                Stage::Prep(_) => unreachable!("resolved above"),
            };
            let Some(Job { meta, stage }) = p.pending.pop_front() else {
                break;
            };
            moved = true;
            let prepared = match (stage, slot) {
                (Stage::Answer(resp), _) => {
                    p.order.push_back(Done::Answer(Box::new((meta, resp))));
                    continue;
                }
                (Stage::Ready(prepared), Some(_)) => prepared,
                _ => unreachable!("a ready request has a slot"),
            };
            let slot = slot.unwrap_or_default();
            // The request timeout also covers the wait for decode + preprocess.
            if let Some(resp) = self.timed_out(&meta) {
                p.order.push_back(Done::Answer(Box::new((meta, resp))));
                continue;
            }
            let extra = eng.family.extra_inputs(&prepared.ctx);
            let submitted = Instant::now();
            match eng.backend.submit(slot, &prepared.chw, &extra) {
                Ok(seq) => {
                    let Prepared {
                        ctx,
                        img,
                        decode,
                        preprocess,
                        ..
                    } = prepared; // the tensor buffer goes back to the pool here
                    p.slots[slot] = Some(InFlight {
                        meta,
                        ctx,
                        img,
                        decode,
                        preprocess,
                        seq,
                        submitted,
                        done: None,
                    });
                    p.order.push_back(Done::Slot(slot));
                }
                Err(e) => {
                    let resp = self.failure(&meta, &eng.provider, &e);
                    p.order.push_back(Done::Answer(Box::new((meta, resp))));
                }
            }
        }
        moved
    }

    /// Reply to the requests at the front of `order` that are answered or whose inference
    /// finished. True when something was replied.
    fn reply_front(&self, eng: &mut Engine, p: &mut Pipeline) -> bool {
        let mut replied = false;
        loop {
            match p.order.front() {
                Some(Done::Answer(..)) => {
                    if let Some(Done::Answer(answer)) = p.order.pop_front() {
                        let (meta, resp) = *answer;
                        self.reply(meta, resp);
                    }
                }
                Some(&Done::Slot(i)) if p.slots[i].as_ref().is_some_and(|f| f.done.is_some()) => {
                    p.order.pop_front();
                    if let Some(f) = p.slots[i].take() {
                        self.finish(eng, i, f);
                    }
                }
                _ => break,
            }
            replied = true;
        }
        replied
    }

    /// A tick: if the oldest inference has run for longer than the request timeout without a
    /// completion, collect it anyway (blocking), so a lost notification cannot stall the model.
    fn unstick(&self, p: &mut Pipeline) {
        if let Some(&Done::Slot(i)) = p.order.front()
            && let Some(f) = p.slots[i].as_mut()
            && f.done.is_none()
            && f.submitted.elapsed() > f.meta.live.request_timeout
        {
            warn!(
                model = %self.cfg.name,
                slot = i,
                "no completion after the request timeout; waiting for the inference"
            );
            f.done = Some(Instant::now());
        }
    }

    fn reply(&self, meta: Meta, resp: VisionDetectionResponse) {
        trace!(
            target: ORDER_TARGET,
            model = %self.cfg.name,
            image = %meta.image_name,
            success = resp.success,
            "replied"
        );
        if meta.reply.send(resp).is_err() {
            debug!(model = %self.cfg.name, "client went away before the response was sent");
        }
    }

    /// Block until the model's downloads are installed (true: load now). Keeps the state
    /// `Initializing` with the progress text; a failed download makes it `Failed`; a runtime
    /// that got installed keeps it waiting for the new generation (false on shutdown).
    fn wait_for_downloads(&self, wait: &crate::resources::provision::Wait) -> bool {
        use crate::resources::provision::WaitStatus;
        let name = &self.cfg.name;
        let mut last = String::new();
        loop {
            if self.shutdown.is_cancelled() {
                return false;
            }
            match wait.poll() {
                WaitStatus::Ready => {
                    info!(model = %name, "downloads installed; loading");
                    return true;
                }
                WaitStatus::Failed(msg) => {
                    error!(model = %name, "{msg}");
                    self.state.set(ModelState::Failed(msg));
                    return false;
                }
                WaitStatus::Pending(msg) | WaitStatus::AwaitingRestart(msg) => {
                    if msg != last {
                        debug!(model = %name, "{msg}");
                        self.state.set_initializing(msg.clone());
                        last = msg;
                    }
                }
            }
            std::thread::sleep(POLL);
        }
    }

    /// Load + warm up, updating state/device/soft capacity. Returns None on failure.
    fn try_load(&self) -> Option<Engine> {
        self.load_started.store(true, Ordering::Release);
        self.state.set(ModelState::Initializing);
        match self.load() {
            Ok(engine) => {
                self.state.set(ModelState::Ready);
                Some(engine)
            }
            Err(e) => {
                let msg = format!("{e:#}");
                error!(model = %self.cfg.name, "failed to load model: {msg}");
                self.state.set(ModelState::Failed(msg));
                None
            }
        }
    }

    fn load(&self) -> Result<Engine> {
        let name = &self.cfg.name;
        let candidates = {
            let rt = self.runtimes.lock_recover();
            rt.plan(&self.cfg.device, &self.cfg.load.path)
        };
        info!(
            model = %name,
            path = %self.cfg.load.path.display(),
            device = %self.cfg.load.requested,
            candidates = %candidates.iter().map(|c| c.device.to_string()).collect::<Vec<_>>().join(", "),
            "loading model"
        );
        let label = format!("model {name}");
        let (engine, compile_ms, warmup_ms) =
            crate::backend::try_candidates(&label, &candidates, |cand| self.load_on(cand))
                .with_context(|| format!("loading {}", self.cfg.load.path.display()))?;

        let dev = engine.backend.info().device.clone();
        let (in_w, in_h) = engine.backend.info().input_size;
        let cap = if self.cfg.queue_size > 0 {
            self.cfg.queue_size
        } else {
            auto_queue_size(self.cfg.live.load().request_timeout, warmup_ms)
        };
        self.soft_capacity.store(cap, Ordering::Relaxed);
        if let Ok(mut d) = self.device.write() {
            *d = Some(dev.clone());
        }
        self.metrics
            .set_loaded(dev.actual.clone(), engine.provider.clone());
        let (hint, streams) = engine.backend.hint_and_streams();
        info!(
            model = %name,
            family = %engine.family.kind(),
            device = %dev.actual,
            spec = %dev.spec,
            provider = %engine.provider,
            fell_back = dev.fell_back,
            input = %format!("{in_w}x{in_h}"),
            classes = self.cfg.classes.len(),
            compile_ms,
            warmup_ms,
            queue = cap,
            inflight = engine.backend.slots(),
            hint = hint.as_str(),
            streams = ?streams,
            "model ready"
        );
        Ok(engine)
    }

    /// The compile hint and in-flight count for `cand` (serial without a profile).
    fn tuning_for(&self, rt: &Runtimes, cand: &Candidate) -> DeviceTuning {
        match &self.cfg.concurrency {
            Some(c) => c
                .profile
                .tuning_with(rt.device_class(&cand.device), c.overrides),
            None => DeviceTuning::SERIAL,
        }
    }

    /// Compile (under the runtimes lock), create the requests and warm up on one candidate.
    /// Returns the engine with its compile and warm-up times (the warm-up is the inference
    /// only; decode and preprocess run on the prep pool).
    fn load_on(&self, cand: &Candidate) -> Result<(Engine, u64, u64)> {
        self.state
            .set_initializing(format!("{COMPILING_ON} {}", cand.device));
        let compiled = {
            let mut rt = self.runtimes.lock_recover();
            let tuning = self.tuning_for(&rt, cand);
            rt.compile_tuned(cand, &self.cfg.load, &tuning)?
        };
        let info = compiled.info();
        let provider = info.device.execution_provider();
        let (in_w, in_h) = info.input_size;
        let compile_ms = info.compile_ms;
        let family = crate::model::make_family(
            self.cfg.model.family,
            &info.inputs,
            &info.outputs,
            self.cfg.classes.len(),
        )?;
        let spec = PrepSpec::for_family(in_w, in_h, family.as_ref());
        let mut backend = compiled.into_backend()?;

        // Warm-up on a mid-gray frame the size of the model input: slot 0 timed, then every
        // other slot at once (first runs allocate per request / session).
        let gray = vec![114u8; in_w as usize * in_h as usize * 3];
        let mut pre = spec.preprocessor();
        let (chw, ctx) = pre.run(&gray, in_w, in_h)?;
        let extra = family.extra_inputs(&ctx);
        let t = Instant::now();
        backend.infer(chw, &extra).context("warm-up inference")?;
        let warmup_ms = t.elapsed().as_millis() as u64;
        let others = 1..backend.slots();
        for s in others.clone() {
            backend
                .submit(s, chw, &extra)
                .context("warm-up inference")?;
        }
        for s in others {
            let outs = backend.collect(s).context("warm-up inference")?;
            backend.recycle(s, outs);
        }
        while backend.completions().try_recv().is_ok() {}
        Ok((
            Engine {
                backend,
                family,
                spec,
                provider,
            },
            compile_ms,
            warmup_ms,
        ))
    }

    fn error_response(
        &self,
        msg: String,
        inference_ms: i32,
        process_ms: i32,
    ) -> VisionDetectionResponse {
        let provider = loaded_provider(&self.device).unwrap_or_default();
        VisionDetectionResponse {
            command: "detect".into(),
            executionProvider: provider,
            canUseGPU: self.cfg.can_use_gpu,
            inferenceMs: inference_ms,
            processMs: process_ms,
            ..VisionDetectionResponse::error(msg)
        }
    }

    /// Collect `slot`'s finished inference, postprocess on this thread and reply.
    fn finish(&self, eng: &mut Engine, slot: usize, f: InFlight) {
        let name = &self.cfg.name;
        let infer_d = f
            .done
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(f.submitted);
        let t = Instant::now();
        let live = f.meta.live.clone();
        let params = PostParams {
            confidence_threshold: effective_threshold(
                f.min_confidence(),
                live.confidence_threshold,
            ),
            nms_iou: live.nms_iou,
        };
        let result = eng.backend.collect(slot).and_then(|outs| {
            let dets = eng.family.postprocess(&outs, &f.ctx, &params);
            eng.backend.recycle(slot, outs);
            let filter = (!live.object_filter.is_empty()).then_some(live.object_filter.as_slice());
            Ok(crate::model::to_predictions(
                &dets?,
                &f.ctx,
                &self.cfg.classes,
                filter,
            ))
        });
        let post_d = t.elapsed();
        let process_d = f.decode + f.preprocess + infer_d + post_d;
        let resp = match result {
            Ok(predictions) => {
                self.metrics.inference_ms.record(infer_d.as_millis() as u64);
                self.metrics.process_ms.record(process_d.as_millis() as u64);
                debug!(
                    model = %name,
                    image = %f.meta.image_name,
                    size = %format!("{}x{}", f.ctx.orig_w, f.ctx.orig_h),
                    count = predictions.len(),
                    slot,
                    queue_ms = f.submitted.saturating_duration_since(f.meta.enqueued).as_millis() as u64,
                    decode_ms = f.decode.as_secs_f64() * 1e3,
                    preprocess_ms = f.preprocess.as_secs_f64() * 1e3,
                    inference_ms = infer_d.as_secs_f64() * 1e3,
                    postprocess_ms = post_d.as_secs_f64() * 1e3,
                    process_ms = process_d.as_secs_f64() * 1e3,
                    "detection done"
                );
                if let (Some(dir), Some(img)) = (&live.save_image_path, &f.img) {
                    let file_name =
                        if f.meta.image_name.is_empty() || f.meta.image_name == "image.jpg" {
                            format!("{}.jpg", uuid::Uuid::new_v4())
                        } else {
                            f.meta.image_name.clone()
                        };
                    if let Err(e) = crate::image::save_annotated(
                        dir,
                        &file_name,
                        img,
                        &predictions,
                        live.save_ref_image,
                    ) {
                        warn!(model = %name, "could not save annotated image: {e:#}");
                    }
                }
                VisionDetectionResponse {
                    success: true,
                    message: format!("Found {} objects", predictions.len()),
                    error: None,
                    count: predictions.len() as i32,
                    predictions,
                    command: "detect".into(),
                    moduleId: crate::MODULE_ID.to_string(),
                    executionProvider: eng.provider.clone(),
                    canUseGPU: self.cfg.can_use_gpu,
                    inferenceMs: ms(infer_d),
                    processMs: ms(process_d),
                    analysisRoundTripMs: 0,
                }
            }
            Err(e) => {
                warn!(model = %name, image = %f.meta.image_name, "detection failed: {e:#}");
                let mut r = self.error_response(format!("{e:#}"), ms(infer_d), ms(process_d));
                r.executionProvider = eng.provider.clone();
                r
            }
        };
        self.reply(f.meta, resp);
    }
}

impl InFlight {
    fn min_confidence(&self) -> f32 {
        self.meta.min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn in_flight(seq: u64) -> InFlight {
        let (reply, _rx) = tokio::sync::oneshot::channel();
        InFlight {
            meta: Meta {
                reply,
                enqueued: Instant::now(),
                image_name: String::new(),
                min_confidence: 0.0,
                live: Arc::new(LiveSettings::default()),
            },
            ctx: PreprocessCtx {
                orig_w: 1,
                orig_h: 1,
                input_w: 1,
                input_h: 1,
                mode: crate::model::ResizeMode::Letterbox,
                scale: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
            },
            img: None,
            decode: Duration::ZERO,
            preprocess: Duration::ZERO,
            seq,
            submitted: Instant::now(),
            done: None,
        }
    }

    #[test]
    fn pipeline_slots_round_robin_and_completions() {
        let mut p = Pipeline::default();
        p.resize(0);
        assert_eq!(p.slots.len(), 1, "at least one slot");
        p.resize(3);
        assert!(p.is_empty() && !p.busy());
        assert_eq!(p.free_slot(), Some(0));
        p.slots[0] = Some(in_flight(1));
        assert_eq!(p.free_slot(), Some(1));
        p.slots[1] = Some(in_flight(1));
        assert_eq!(p.free_slot(), Some(2));
        p.slots[2] = Some(in_flight(4));
        assert_eq!(p.free_slot(), None);
        assert!(p.busy() && !p.is_empty());
        let at = Instant::now();
        // Stale (older sequence number) and unknown slots are ignored.
        assert!(!p.complete(Completion {
            slot: 2,
            seq: 3,
            at
        }));
        assert!(!p.complete(Completion {
            slot: 9,
            seq: 1,
            at
        }));
        assert!(p.complete(Completion {
            slot: 2,
            seq: 4,
            at
        }));
        assert!(
            !p.complete(Completion {
                slot: 2,
                seq: 4,
                at
            }),
            "once"
        );
        assert_eq!(p.slots[2].as_ref().unwrap().done, Some(at));
        p.slots[0] = None;
        assert_eq!(p.free_slot(), Some(0));
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
}
