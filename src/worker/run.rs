//! The worker thread's run loop ([`WorkerCtx::run`]): load (compile + warm-up on each device
//! candidate), then a `Select`-driven state machine over the work channel, the prep reply of the
//! oldest request and the backend's completions. Invariants: requests are answered in arrival
//! order; nothing is left in flight when the loop ends (shutdown aborts, then answers every
//! queued request); postprocessing runs on this thread. Decode and preprocessing (forwarded
//! frames included) run on the prep pool, the annotated-image save on the `image-save` thread
//! after the reply ([`super::save`]).
//!
//! Stop: the worker's token is checked by the load too ([`LoadStopped`]): between candidates,
//! before asking the serving copy for room, while waiting for the runtimes lock, after the
//! compile and in slices during the warm-up. A compile in progress cannot be interrupted; its
//! result is dropped unused. A panic anywhere on the thread is contained
//! ([`WorkerCtx::run_contained`]): the model becomes `Failed` (the supervisor reloads it) and its
//! load gate opens, so the models after it still load.

use super::pipeline::{Done, InFlight, Job, Meta, Pipeline, Stage};
use super::reap::reap;
use super::save::{self, After, SaveJob};
use super::verify::{Lane, LaneCtx, Pending};
use super::watch::{SlowCompile, StuckWatch, collect_watched, watch_slow_compile};
use super::{
    COMPILING_ON, Decoded, LoadGate, ORDER_TARGET, POLL, Purpose, Reply, WorkItem, WorkerConfig,
    WorkerHandle, auto_queue_size, effective_threshold, initializing_message, loaded_provider,
    millis_u64, ms, new_request_id,
};
use crate::api::VisionDetectionResponse;
use crate::backend::plan::{Better, LoadFailure, lesser_device};
use crate::backend::{Backend, Candidate, Completion, DeviceInfo, Runtimes};
use crate::image_stats::Scene;
use crate::metrics::{DropReason, ModelMetrics, RouteFallback};
use crate::model::filters::{SuppressCounts, SuppressReason};
use crate::model::{Family, PostParams};
use crate::prep::{PrepInput, PrepJob, PrepPool, PrepSpec, Prepared};
use crate::profile::DeviceTuning;
use crate::startup::{ModelState, StateHandle};
use crate::static_memory::StaticMemory;
use crate::sync::{LockExt, RwLockExt};
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Select, TryRecvError};
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

/// Model state owned by the worker thread after a successful load.
struct Engine {
    backend: Backend,
    family: Box<dyn Family>,
    /// Geometry the prep pool preprocesses this model's images for.
    spec: PrepSpec,
    /// This model's use of its tensor length in the shared prep pool: when the last engine of
    /// that length drops, the pool frees those buffers ([`crate::prep::TensorPool::hold_length`]).
    _tensors: Option<crate::prep::LengthHold>,
    provider: String,
    /// Canonical spec of the device it loaded on ("openvino:gpu"), for logs.
    device: String,
}

/// A backend given up on: stops what still runs on it, then frees it. Both wait for the device,
/// so this goes to a reaper thread ([`reap`]), never dropped on a worker thread.
struct Abandoned(Backend);

impl Drop for Abandoned {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// What woke the worker.
enum Event {
    Work(Result<WorkItem, crossbeam_channel::RecvError>),
    Prep(Result<Result<Prepared>, crossbeam_channel::RecvError>),
    Done(Result<Completion, crossbeam_channel::RecvError>),
    Tick,
}

pub(super) struct WorkerCtx {
    pub(super) runtimes: Arc<Mutex<Runtimes>>,
    pub(super) cfg: WorkerConfig,
    pub(super) metrics: Arc<ModelMetrics>,
    pub(super) state: StateHandle,
    pub(super) device: Arc<RwLock<Option<DeviceInfo>>>,
    pub(super) soft_capacity: Arc<AtomicUsize>,
    pub(super) gate: LoadGate,
    pub(super) load_started: Arc<AtomicBool>,
    /// Shared with [`WorkerHandle::is_stuck`].
    pub(super) stuck: Arc<AtomicBool>,
    /// Shared with [`WorkerHandle::better`].
    pub(super) better: Arc<Mutex<Option<Better>>>,
    pub(super) shutdown: CancellationToken,
    /// Inferences in a row that returned an error (see [`MAX_CONSECUTIVE_ERRORS`]).
    pub(super) consecutive_errors: Cell<u32>,
    /// Inferences in a row whose outputs were degenerate (see [`outputs_degenerate`]).
    pub(super) consecutive_degenerate: Cell<u32>,
    /// Static-box memory of this model (kept across live-settings swaps; a reload starts a new
    /// worker and so an empty memory).
    pub(super) static_memory: RefCell<StaticMemory>,
    /// The verifier-cascade continuation thread (started on the first verification; see
    /// [`super::verify`]).
    pub(super) verify_lane: RefCell<Option<Lane>>,
}

/// Inference errors in a row after which the model counts as broken (a reset GPU makes every
/// later `run` fail with the same CUDA / DirectML error; a single failure is per request).
pub(crate) const MAX_CONSECUTIVE_ERRORS: u32 = 5;
/// Inferences in a row with [`outputs_degenerate`] outputs after which the model counts as
/// broken (a reset device can hand back unwritten buffers without an error: "no objects"
/// forever). Camera frames never produce that many all-zero raw outputs in a row.
pub(crate) const MAX_CONSECUTIVE_DEGENERATE: u32 = 25;

/// Opens the worker's [`LoadGate`] whatever happens to its thread, and turns a panic into a
/// `Failed` model ([`WorkerCtx::run_contained`]). Without it a panic while loading left the gate
/// closed, so every later model in config order waited for ever, and a panic while serving left
/// the model `Ready` with no thread behind its channel.
struct Contained {
    name: String,
    gate: LoadGate,
    state: StateHandle,
    stuck: Arc<AtomicBool>,
}

impl Contained {
    /// The thread panicked with `message`: `Failed`, and marked stuck so the supervisor reloads
    /// the model on its backoff (`supervisor::Status::Stuck`); its thread has ended, so the
    /// registry's joins of a stuck worker return at once.
    fn crashed(&self, message: &str) {
        let msg = format!(
            "worker thread panicked: {message}; this is a bug (the panic and its location are \
             in the log); the model is reloaded automatically, please report the log"
        );
        error!(model = %self.name, "{msg}");
        self.stuck.store(true, Ordering::Release);
        self.state.set(ModelState::Failed(msg));
    }
}

impl Drop for Contained {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // Only reachable when the panic escaped `catch_unwind` (a panicking payload drop).
            self.crashed("a panic while unwinding");
        }
        self.gate.open();
    }
}

/// The text of a panic payload (`panic!` gives a `&str` or a `String`).
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    crate::panic::payload_message(payload).to_string()
}

/// A load given up because the worker was stopped (its token cancelled: the service stops, or
/// the model was reloaded, disabled or retired). Checked between candidates, before the retry
/// that asks for room, while waiting for the runtimes lock, after the compile and during the
/// warm-up. A compile in progress cannot be interrupted (neither OpenVINO nor ONNX Runtime has
/// a cancel for it): the stop is seen once it returns, and its result is dropped unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoadStopped;

impl std::fmt::Display for LoadStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "the model stopped loading: the service is stopping, or the model was reloaded, \
             disabled or removed",
        )
    }
}

impl std::error::Error for LoadStopped {}

impl WorkerCtx {
    /// [`Self::run`] with its panics contained: the model becomes `Failed` (and is reloaded by
    /// the supervisor) and its [`LoadGate`] opens, so the models after it still load.
    ///
    /// `AssertUnwindSafe` holds because nothing the closure owns is used after a panic: the
    /// context, the receiver and the engine drop while unwinding (aborting what is in flight,
    /// dropping the replies, so their clients get an error at once). What other threads see
    /// afterwards is only the state, the stuck flag and the gate, set here, plus metrics
    /// (atomics) and locks that every reader takes with poison recovery (`crate::sync`).
    pub(super) fn run_contained(self, rx: Receiver<WorkItem>, after: Option<LoadGate>) {
        let guard = Contained {
            name: self.cfg.name.clone(),
            gate: self.gate.clone(),
            state: self.state.clone(),
            stuck: self.stuck.clone(),
        };
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            self.run(rx, after);
        }));
        if let Err(payload) = ran {
            guard.crashed(&panic_text(payload.as_ref()));
        }
    }

    pub(super) fn run(mut self, rx: Receiver<WorkItem>, after: Option<LoadGate>) {
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
                        let msg = format!(
                            "{e:#}; the classes file is missing or not a YOLO-style `names:` \
                             list or map; fix or remove `classes` for this model and save the \
                             config"
                        );
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
                if engine.is_some() && self.stuck.load(Ordering::Acquire) {
                    // The stuck inference came back with an error (or the session died): stop
                    // using the engine; requests are answered with the Failed state from now on.
                    // It goes now, not when this thread exits: a dead device must not be held
                    // while the replacement loads (D3D12 shares one device per adapter, CUDA
                    // keeps its context), and freeing it waits for what is still in flight.
                    self.abandon(engine.take());
                    self.fail_pipeline(&mut p);
                    progressed = true;
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
                // The head's prep reply is awaited while a slot is free, or (no slot free) when
                // it may be forwarded to the night model, which needs no slot here.
                let head_prep = match p.pending.front() {
                    Some(Job {
                        stage: Stage::Prep(r),
                        meta,
                    }) if p.slots.iter().any(Option::is_none) || may_route(meta) => Some(r),
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
        if self.stuck.load(Ordering::Acquire) {
            // Stopping these inferences waits for the device that hung.
            self.abandon(engine.take());
        } else if let Some(eng) = engine.as_mut()
            && p.busy()
        {
            // Never leave an inference running past the worker (shutdown).
            eng.backend.abort();
        }
        self.answer_remaining(p, &rx);
        // Answers the verifications still waiting (unverified) and joins the lane thread.
        drop(self.verify_lane.borrow_mut().take());
        info!(model = %name, "worker exiting");
    }

    /// Hand `engine` to a reaper thread (see [`Abandoned`]).
    fn abandon(&self, engine: Option<Engine>) {
        if let Some(e) = engine {
            reap(&self.cfg.name, &e.device, Abandoned(e.backend));
        }
    }

    /// Shutdown: answer every request still in the pipeline or queued in the channel at once
    /// (instead of letting its HTTP handler wait for the request timeout).
    fn answer_remaining(&self, p: Pipeline, rx: &Receiver<WorkItem>) {
        let msg = format!("Model '{}' is shutting down", self.cfg.name);
        self.answer_all(p, &msg);
        while let Ok(item) = rx.try_recv() {
            let reply = match item {
                WorkItem::Detect(_, reply, _) | WorkItem::DetectWithId(_, reply, _, _) => {
                    Reply::Client(reply)
                }
                WorkItem::DetectDecoded(d) => d.reply,
                WorkItem::Load => continue,
            };
            let _ = reply.send(self.error_response(msg.clone(), 0, 0));
        }
    }

    /// Answer every request in the pipeline with `msg` at once.
    fn answer_all(&self, p: Pipeline, msg: &str) {
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
            let _ = m.reply.send(self.error_response(msg.to_string(), 0, 0));
        }
    }

    /// The engine was parked as stuck: answer the requests in the pipeline with the model's
    /// `Failed` text now instead of letting them time out.
    fn fail_pipeline(&self, p: &mut Pipeline) {
        let msg = self.unavailable_message();
        self.answer_all(std::mem::take(p), &msg);
    }

    /// The answer of a failed model: "failed to load" or, after a stuck inference,
    /// "is unavailable".
    fn unavailable_message(&self) -> String {
        let name = &self.cfg.name;
        match self.state.get() {
            ModelState::Failed(msg) if self.stuck.load(Ordering::Acquire) => {
                format!("Model '{name}' is unavailable: {msg}")
            }
            ModelState::Failed(msg) => format!("Model '{name}' failed to load: {msg}"),
            _ => format!("Model '{name}' is unavailable"),
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
        let (meta, input) = match item {
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
            WorkItem::Detect(req, reply, enqueued) => {
                self.intake(req, Reply::Client(reply), enqueued, new_request_id())
            }
            WorkItem::DetectWithId(req, reply, enqueued, id) => {
                self.intake(req, Reply::Client(reply), enqueued, id)
            }
            WorkItem::DetectDecoded(d) => self.intake_decoded(*d),
        };
        let Some(eng) = engine.as_ref() else {
            // Only reachable for a lazy model (between the server's trigger and the state
            // update) or after a failed load: answer at once, never block.
            let (reason, msg) = match self.state.get() {
                ModelState::Failed(_) => (DropReason::Failed, self.unavailable_message()),
                _ => (
                    DropReason::Initializing,
                    initializing_message(name, self.state.detail().as_deref()),
                ),
            };
            self.metrics.record_drop(reason);
            let _ = meta.reply.send(VisionDetectionResponse {
                dropped: Some(reason),
                ..self.error_response(msg, 0, 0)
            });
            return;
        };
        trace!(
            target: ORDER_TARGET,
            model = %name,
            image = %meta.image_name,
            request_id = %meta.request_id,
            "accepted"
        );
        let (input, keep_image) = match input {
            // The decoded frame is forwarded when the night model takes it, and cropped for
            // the verifier.
            Input::Bytes(b) => (
                PrepInput::Encoded(b),
                meta.live.save_image_path.is_some() || may_route(&meta) || may_verify(&meta),
            ),
            // A forwarded frame: prepared on the pool like an upload, with this model's own
            // modes for the sender's scene statistics (its original frame, never the sender's
            // enhanced copy).
            Input::Decoded { img, stats, decode } => (
                PrepInput::Decoded { img, stats, decode },
                (meta.live.save_image_path.is_some() && !meta.verify_crop) || may_verify(&meta),
            ),
        };
        let stage = if let Some(resp) = self.timed_out(&meta) {
            Stage::Answer(resp)
        } else {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let job = PrepJob {
                input,
                spec: eng.spec,
                keep_image,
                preprocess: meta.live.preprocess,
                reply: tx,
            };
            match prep.submit(job) {
                Ok(()) => Stage::Prep(rx),
                Err(e) => Stage::Answer(self.failure(&meta, &eng.provider, &e)),
            }
        };
        p.pending.push_back(Job { meta, stage });
    }

    /// The bookkeeping of a request from the HTTP layer (live settings as of now).
    fn intake(
        &self,
        req: crate::api::VisionDetectionRequest,
        reply: Reply,
        enqueued: Instant,
        request_id: String,
    ) -> (Meta, Input) {
        let live = self.cfg.live.load_full();
        let original = (live.collect.enabled && !req.no_collect).then(|| req.image_data.clone());
        // The reference copy of `save_ref_image` is the upload itself (see `save`).
        let upload =
            (live.save_ref_image && live.save_image_path.is_some()).then(|| req.image_data.clone());
        let meta = Meta {
            reply,
            enqueued,
            image_name: req.image_name,
            min_confidence: req.min_confidence,
            live,
            request_id,
            budget: None,
            forwardable: true,
            routed_from: None,
            verify_crop: false,
            original,
            upload,
        };
        (meta, Input::Bytes(req.image_data))
    }

    /// The bookkeeping of a request another worker forwarded with its decoded image: this
    /// model's live settings, the sender's deadline, never forwarded again.
    fn intake_decoded(&self, d: Decoded) -> (Meta, Input) {
        let Decoded {
            img,
            stats,
            image_name,
            min_confidence,
            request_id,
            enqueued,
            deadline,
            decode,
            reply,
            from,
            purpose,
            original,
            upload,
        } = d;
        let meta = Meta {
            reply,
            enqueued,
            image_name,
            min_confidence,
            live: self.cfg.live.load_full(),
            request_id,
            budget: Some(deadline.saturating_duration_since(enqueued)),
            forwardable: false,
            routed_from: Some(from),
            verify_crop: purpose == Purpose::Verify,
            original: original.filter(|_| purpose == Purpose::NightModel),
            upload: upload.filter(|_| purpose == Purpose::NightModel),
        };
        (meta, Input::Decoded { img, stats, decode })
    }

    /// Night model switch for the request at the head of `pending` (decoded, a night / IR
    /// frame, `night_model` set, not tried before): hand its decoded image to the night model's
    /// worker when that one is ready and has room. None: forwarded (the night worker answers
    /// it); `Some(job)`: run it here (a fallback is counted and warned about, throttled). Never
    /// blocks: the send is a `try_send`, so two workers can never wait on each other.
    fn route(&self, mut job: Job) -> Option<Job> {
        job.meta.forwardable = false;
        let Some(night) = job.meta.live.night_model.clone() else {
            return Some(job);
        };
        let Job { meta, stage } = job;
        let mut prepared = match stage {
            Stage::Ready(p) => p,
            stage => return Some(Job { meta, stage }),
        };
        let scene = prepared.stats.scene();
        let late = meta.enqueued.elapsed() > meta.timeout();
        let img = match prepared.img.take() {
            // Day frame, or already too late (answered as a drop in its turn).
            Some(img) if scene != Scene::Day && !late => img,
            img => {
                // Kept for routing only: drop it unless this model saves or verifies.
                prepared.img =
                    img.filter(|_| meta.live.save_image_path.is_some() || may_verify(&meta));
                return Some(Job {
                    meta,
                    stage: Stage::Ready(prepared),
                });
            }
        };
        let target = match self.night_target(&night) {
            Ok(h) => h,
            Err(reason) => {
                return Some(self.fall_back(meta, prepared, img, &night, reason));
            }
        };
        let log_id = meta.request_id.clone();
        let log_image = meta.image_name.clone();
        let deadline = meta.enqueued + meta.timeout();
        let Meta {
            reply,
            enqueued,
            image_name,
            min_confidence,
            live,
            request_id,
            budget,
            original,
            upload,
            ..
        } = meta;
        let decoded = Decoded {
            img,
            stats: prepared.stats,
            image_name,
            min_confidence,
            request_id,
            enqueued,
            deadline,
            decode: prepared.decode,
            reply,
            from: self.cfg.name.clone(),
            purpose: Purpose::NightModel,
            original,
            upload,
        };
        let (back, reason) = match target
            .sender
            .try_send(WorkItem::DetectDecoded(Box::new(decoded)))
        {
            Ok(()) => {
                self.metrics.record_night_routed();
                // The primary's line of this request; the night model logs its own
                // "detection done" with the same request id.
                debug!(
                    model = %self.cfg.name,
                    routed_to = %target.name,
                    request_id = %log_id,
                    image = %log_image,
                    scene = %scene,
                    luma = f64::from(prepared.stats.mean_luma),
                    chroma = f64::from(prepared.stats.chroma_spread.min(prepared.stats.tint_spread)),
                    elapsed_ms = millis_u64(enqueued.elapsed()),
                    decode_ms = prepared.decode.as_secs_f64() * 1e3,
                    "detection routed to the night model"
                );
                return None;
            }
            Err(crossbeam_channel::TrySendError::Full(i)) => (i, RouteFallback::QueueFull),
            Err(crossbeam_channel::TrySendError::Disconnected(i)) => (i, RouteFallback::Stopped),
        };
        let WorkItem::DetectDecoded(d) = back else {
            // Not reachable (the item sent is the item returned); nothing to answer then.
            return None;
        };
        let d = *d;
        let meta = Meta {
            reply: d.reply,
            enqueued: d.enqueued,
            image_name: d.image_name,
            min_confidence: d.min_confidence,
            live,
            request_id: d.request_id,
            budget,
            forwardable: false,
            routed_from: None,
            verify_crop: false,
            original: d.original,
            upload: d.upload,
        };
        Some(self.fall_back(meta, prepared, d.img, &night, reason))
    }

    /// The served, ready worker of `night` with room for one more request, or why not (a lazy
    /// night model is asked to compile).
    fn night_target(&self, night: &str) -> std::result::Result<Arc<WorkerHandle>, RouteFallback> {
        Ok(self.cfg.peers.ready(night, &self.cfg.name)?)
    }

    /// Run a night / IR frame here after all: count and warn (throttled), and put the decoded
    /// image back when this model saves images.
    fn fall_back(
        &self,
        meta: Meta,
        mut prepared: Prepared,
        img: crate::image::RgbImage,
        night: &str,
        reason: RouteFallback,
    ) -> Job {
        self.metrics.record_night_fallback(reason);
        if let Some(held_back) = self.metrics.night_fallback_log.allow(self.metrics.now_ms()) {
            warn!(
                model = %self.cfg.name,
                night_model = %night,
                reason = reason.as_str(),
                request_id = %meta.request_id,
                similar_since_last_warning = held_back,
                "night / IR frame runs on this model: its night_model is {}; check that model \
                 on the Models card (enabled, loaded) or clear night_model",
                fallback_text(reason)
            );
        }
        if meta.live.save_image_path.is_some() || may_verify(&meta) {
            prepared.img = Some(img);
        }
        Job {
            meta,
            stage: Stage::Ready(prepared),
        }
    }

    /// The drop answer when `meta` waited longer than the request timeout.
    fn timed_out(&self, meta: &Meta) -> Option<VisionDetectionResponse> {
        let waited = meta.enqueued.elapsed();
        let timeout = meta.timeout();
        if waited <= timeout {
            return None;
        }
        self.metrics.record_drop(DropReason::Timeout);
        warn_timed_out(
            &self.metrics,
            self.metrics.now_ms(),
            &self.cfg.name,
            &meta.request_id,
            waited,
            timeout,
        );
        Some(VisionDetectionResponse {
            dropped: Some(DropReason::Timeout),
            ..self.error_response(
                format!(
                    "Request dropped: waited {} ms in queue (timeout {} ms)",
                    waited.as_millis(),
                    timeout.as_millis()
                ),
                0,
                0,
            )
        })
    }

    /// The error answer for a request that failed before inference (bad image, ...).
    fn failure(&self, meta: &Meta, provider: &str, e: &anyhow::Error) -> VisionDetectionResponse {
        warn!(
            model = %self.cfg.name,
            request_id = %meta.request_id,
            image = %meta.image_name,
            "detection failed: {e:#}"
        );
        self.failure_answer(provider, e)
    }

    /// [`Self::failure`] for an image that could not be decoded or was refused by its size:
    /// counted, and warned at most once per [`crate::metrics::LOG_THROTTLE_INTERVAL`] (bad input
    /// never marks the model broken, so nothing else stops a camera that sends broken frames
    /// from logging one line per frame). The line that gets through keeps the request id and the
    /// image name, so the camera can still be found.
    fn decode_failure(
        &self,
        meta: &Meta,
        provider: &str,
        e: &anyhow::Error,
    ) -> VisionDetectionResponse {
        self.metrics.record_decode_error();
        warn_decode_failure(
            &self.metrics,
            self.metrics.now_ms(),
            &self.cfg.name,
            (&meta.request_id, &meta.image_name),
            e,
        );
        self.failure_answer(provider, e)
    }

    /// The answer of [`Self::failure`] and [`Self::decode_failure`].
    fn failure_answer(&self, provider: &str, e: &anyhow::Error) -> VisionDetectionResponse {
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
            Ok(Err(e)) => Stage::Answer(self.decode_failure(meta, &provider, &e)),
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
            if matches!(job.stage, Stage::Ready(_)) && may_route(&job.meta) {
                // Night model switch: a forwarded request leaves this pipeline without a
                // slot or a place in the reply order.
                let Some(job) = p.pending.pop_front() else {
                    break;
                };
                match self.route(job) {
                    None => {
                        moved = true;
                        continue;
                    }
                    Some(job) => p.pending.push_front(job),
                }
            }
            let Some(job) = p.pending.front() else {
                break;
            };
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
            let Prepared {
                chw,
                ctx,
                img,
                stats,
                decode,
                preprocess,
            } = prepared;
            // The tensor goes to the backend, which returns it to the pool (see
            // `Backend::submit_pooled`).
            match self.submit(eng, slot, chw, &extra) {
                Ok(seq) => {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "slot was returned by p.free_slot() earlier in this loop iteration, \
                                  which only yields indices < slots.len(); nothing resizes \
                                  slots in between"
                    )]
                    let cell = &mut p.slots[slot];
                    *cell = Some(InFlight {
                        meta,
                        ctx,
                        img,
                        stats,
                        decode,
                        preprocess,
                        seq,
                        submitted,
                        done: None,
                        overdue: false,
                    });
                    p.order.push_back(Done::Slot(slot));
                }
                Err(e) => {
                    // A reset GPU fails here, before any inference runs, so nothing would
                    // reach `finish` and the model would answer errors while it stays Ready.
                    self.mark_if_dead(eng, submitted, &e);
                    self.note_health(eng, Some(&e), false);
                    let resp = self.failure(&meta, &eng.provider, &e);
                    p.order.push_back(Done::Answer(Box::new((meta, resp))));
                    if self.stuck.load(Ordering::Acquire) {
                        // The run loop parks the engine next and answers what is left.
                        break;
                    }
                }
            }
        }
        moved
    }

    /// Start one inference ([`Backend::submit_pooled`]).
    fn submit(
        &self,
        eng: &mut Engine,
        slot: usize,
        chw: crate::prep::PooledBuf,
        extra: &[crate::model::ExtraInput],
    ) -> Result<u64> {
        #[cfg(test)]
        if let Some(fault) = &self.cfg.faults.submit {
            fault()?;
        }
        eng.backend.submit_pooled(slot, chw, extra)
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
                Some(&Done::Slot(i))
                    if p.slots
                        .get(i)
                        .and_then(Option::as_ref)
                        .is_some_and(|f| f.done.is_some()) =>
                {
                    p.order.pop_front();
                    if let Some(f) = p.slots.get_mut(i).and_then(Option::take) {
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
    /// completion, collect it anyway, so a lost notification cannot stall the model.
    ///
    /// The collect is a deliberate blocking wait on this thread: the runtimes offer no
    /// non-blocking "is it done" check, and a lost completion returns at once. A hung inference
    /// (driver hang, GPU reset) does not: [`collect_watched`] marks the model `Failed` with
    /// [`stuck_message`] once the wait outlasts a second request timeout, so the server answers
    /// at once instead of queuing behind it and the operator can reload the model.
    fn unstick(&self, p: &mut Pipeline) {
        if let Some(&Done::Slot(i)) = p.order.front()
            && let Some(f) = p.slots.get_mut(i).and_then(Option::as_mut)
            && f.done.is_none()
            && f.submitted.elapsed() > f.meta.timeout()
        {
            warn!(
                model = %self.cfg.name,
                request_id = %f.meta.request_id,
                slot = i,
                "no completion after the request timeout; waiting for the inference (the model \
                 is marked failed if it takes another request timeout)"
            );
            f.done = Some(Instant::now());
            f.overdue = true;
        }
    }

    /// What the stuck watchdog of this worker needs (see [`collect_watched`]).
    fn stuck_watch(&self, eng: &Engine) -> StuckWatch {
        StuckWatch {
            name: self.cfg.name.clone(),
            device: eng.device.clone(),
            state: self.state.clone(),
            stuck: self.stuck.clone(),
        }
    }

    fn reply(&self, meta: Meta, resp: VisionDetectionResponse) {
        send_reply(&self.cfg.name, &meta.image_name, meta.reply, resp);
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
                        self.state.set_waiting_for_download(msg.clone());
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
                // The compile's transients (the read model, rewrites, throwaway sessions, the
                // copies of failed candidates) are free now (RA-077).
                crate::backend::memory::trim_soon("model loaded");
                Some(engine)
            }
            Err(e) if self.shutdown.is_cancelled() => {
                // A normal stop, not a broken model: no error line.
                info!(model = %self.cfg.name, "{LoadStopped} ({e:#})");
                self.state.set(ModelState::Failed(LoadStopped.to_string()));
                None
            }
            Err(e) => {
                let msg = load_failure_message(&e);
                error!(model = %self.cfg.name, "{msg}");
                self.state.set(ModelState::Failed(msg));
                None
            }
        }
    }

    /// `Err(LoadStopped)` once the worker's token is cancelled.
    fn check_stopped(&self) -> Result<()> {
        if self.shutdown.is_cancelled() {
            return Err(LoadStopped.into());
        }
        Ok(())
    }

    /// The runtimes lock, or `Err(LoadStopped)` when the worker is stopped while it waits for
    /// it (another model may hold it for a whole compile) or before it starts work under it.
    fn lock_runtimes(&self) -> Result<std::sync::MutexGuard<'_, Runtimes>> {
        let rt = super::lock_unless_cancelled(&self.runtimes, &self.shutdown).ok_or(LoadStopped)?;
        self.check_stopped()?;
        Ok(rt)
    }

    fn load(&self) -> Result<Engine> {
        let name = &self.cfg.name;
        let candidates = self.candidates()?;
        info!(
            model = %name,
            path = %self.cfg.load.path.display(),
            device = %self.cfg.load.requested,
            candidates = %candidates.iter().map(|c| c.device.to_string()).collect::<Vec<_>>().join(", "),
            "loading model"
        );
        let label = format!("model {name}");
        let first = candidates.first().map(|c| c.device);
        let mut better: Option<Better> = None;
        let loaded = crate::backend::try_candidates(&label, &candidates, |cand| {
            // A stopped worker tries no further candidate, and never makes the previous
            // generation give up its serving copy for a load that is not wanted any more.
            self.check_stopped()?;
            // A device a reaper still waits on is hung: another compile there only adds a
            // backend that cannot be freed (RA-082). Skipped as a lost device, so it stays the
            // better one to try again once the reaper is done.
            if let Some(hung) = self.hung(cand) {
                let e = anyhow::Error::new(hung);
                if better.is_none() {
                    better = Better::of(cand, &e);
                }
                return Err(e);
            }
            let mut attempt = self.load_on(cand);
            if Some(cand.device) == first
                && let Err(e) = &attempt
                && !self.shutdown.is_cancelled()
                && self.room_made(cand, e)
            {
                attempt = self.load_on(cand);
            }
            if better.is_none()
                && !self.shutdown.is_cancelled()
                && let Err(e) = &attempt
            {
                better = Better::of(cand, e);
            }
            attempt
        });
        *self.better.lock_recover() = better;
        let (engine, compile_ms, warmup_ms) =
            loaded.with_context(|| format!("loading {}", self.cfg.load.path.display()))?;

        let dev = engine.backend.info().device.clone();
        let (in_w, in_h) = engine.backend.info().input_size;
        let cap = if self.cfg.queue_size > 0 {
            self.cfg.queue_size
        } else {
            auto_queue_size(self.cfg.live.load().request_timeout, warmup_ms)
        };
        self.soft_capacity.store(cap, Ordering::Relaxed);
        *self.device.write_recover() = Some(dev.clone());
        self.metrics
            .set_loaded(dev.actual.clone(), engine.provider.clone());
        if let Some(before) = &self.cfg.recovering_from
            && lesser_device(&candidates, before, &dev.spec)
        {
            warn!(
                model = %name,
                before = %before,
                now = %dev.spec,
                "reloaded on a lesser device than before the fault: {before} did not come back; \
                 it is tried again later, and if the model stays on {} a service restart may be \
                 needed to recover the device",
                dev.spec
            );
        }
        let (hint, streams) = engine.backend.hint_and_streams();
        super::warn_unknown_filter(name, &self.cfg.live.load().object_filter, &self.cfg.classes);
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

    /// Why `cand` must not be tried now: a reaper still waits on its device
    /// ([`super::reap::hung_device`]).
    fn hung(&self, cand: &Candidate) -> Option<super::reap::Hung> {
        #[cfg(test)]
        if let Some(r) = self.cfg.faults.reapers {
            return r.hung(&cand.device, Instant::now());
        }
        super::reap::reapers().hung(&cand.device, Instant::now())
    }

    /// The candidates to walk: the one [`WorkerConfig::only`] names, else the plan
    /// (`Err(LoadStopped)` when stopped while waiting for the runtimes lock).
    fn candidates(&self) -> Result<Vec<Candidate>> {
        if let Some(c) = &self.cfg.only {
            return Ok(vec![c.clone()]);
        }
        #[cfg(test)]
        if let Some(c) = &self.cfg.faults.candidates {
            return Ok(c.clone());
        }
        let rt = self.lock_runtimes()?;
        Ok(rt.plan(&self.cfg.device, &self.cfg.load.path))
    }

    /// The first candidate failed with `e`. When that is lack of memory and a copy of this
    /// model that serves can be freed, free it (bounded) and say so: true means try again.
    fn room_made(&self, cand: &Candidate, e: &anyhow::Error) -> bool {
        let Some(room) = &self.cfg.make_room else {
            return false;
        };
        if LoadFailure::of(e) != LoadFailure::Memory {
            return false;
        }
        warn!(
            model = %self.cfg.name,
            device = %cand.device,
            "no memory for the new copy of this model beside the one serving ({e:#}); retiring \
             the old copy and trying again, so requests are answered \"initializing\" until it \
             has loaded"
        );
        (room.0)()
    }

    /// The compile hint and in-flight count for `cand` (serial without a profile).
    fn tuning_for(&self, rt: &Runtimes, cand: &Candidate) -> DeviceTuning {
        super::device_tuning(self.cfg.concurrency.as_ref(), rt, &cand.device)
    }

    /// Compile (under the runtimes lock), create the requests and warm up on one candidate.
    /// Returns the engine with its compile and warm-up times (the warm-up is the inference
    /// only; decode and preprocess run on the prep pool).
    fn load_on(&self, cand: &Candidate) -> Result<(Engine, u64, u64)> {
        self.state
            .set_initializing(format!("{COMPILING_ON} {}", cand.device));
        #[cfg(test)]
        if let Some(fault) = &self.cfg.faults.load {
            fault(cand)?;
        }
        let compiled = {
            let mut rt = self.lock_runtimes()?;
            let tuning = self.tuning_for(&rt, cand);
            let device = cand.device.to_string();
            let compiled = watch_slow_compile(
                &self.cfg.name,
                &device,
                &self.state,
                SlowCompile::DEFAULT,
                || rt.compile_tuned(cand, &self.cfg.load, &tuning),
            )
            .0?;
            #[cfg(test)]
            if let Some(hook) = &self.cfg.faults.compiled {
                hook();
            }
            compiled
        };
        // Stopped during the compile (which cannot be interrupted): nothing runs on the
        // compiled model yet, so it is dropped here, unused, instead of warmed up.
        self.check_stopped()?;
        let info = compiled.info();
        let provider = info.device.execution_provider();
        let device = info.device.spec.clone();
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
        #[cfg(not(test))]
        let warmup_timeout = WARMUP_TIMEOUT;
        #[cfg(test)]
        let warmup_timeout = self.cfg.faults.warmup_timeout.unwrap_or(WARMUP_TIMEOUT);
        let outs = match warm_up(&mut backend, chw, &extra, warmup_timeout, &self.shutdown) {
            Ok(outs) => outs,
            Err(e) => {
                if e.is::<WarmupTimeout>() {
                    // The inference is still running on a device that does not answer.
                    // Dropping the backend here would wait for it (OpenVINO cancels and waits,
                    // ONNX Runtime waits `ABORT_TIMEOUT`): a reaper does, so this thread goes
                    // on to the next candidate (CPU last) and the model still serves.
                    error!(model = %self.cfg.name, "{e:#}; abandoning the device");
                    reap(&self.cfg.name, &device, Abandoned(backend));
                } else if e.is::<LoadStopped>() {
                    // Stopped with the warm-up in flight: a reaper stops and frees it, so the
                    // worker thread never waits for the device.
                    reap(&self.cfg.name, &device, Abandoned(backend));
                }
                return Err(e);
            }
        };
        let warmup_ms = outs.1;
        let outs = outs.0;
        // A family or class-count mismatch fails here instead of on every request.
        let params = PostParams {
            confidence_threshold: 0.5,
            nms_iou: 0.45,
        };
        family.postprocess(&outs, &ctx, &params).with_context(|| {
            format!(
                "the model's outputs do not fit family `{}` with {} classes; `family` or the \
                 classes file does not match this model; set `family` (or `auto`) and point \
                 `classes` at the model's own label file",
                family.kind(),
                self.cfg.classes.len()
            )
        })?;
        Ok((
            Engine {
                backend,
                family,
                _tensors: self
                    .cfg
                    .prep
                    .as_ref()
                    .map(|p| p.tensors().hold_length(spec.tensor_len())),
                spec,
                provider,
                device,
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

    /// A dead session marks the model at once, whether `submit` or `collect` reported it: no
    /// later inference can work.
    fn mark_if_dead(&self, eng: &Engine, started: Instant, e: &anyhow::Error) {
        if session_dead(e) {
            self.stuck_watch(eng).mark(started.elapsed(), Some(e));
        }
    }

    /// Track the outcome of one inference: a streak of [`MAX_CONSECUTIVE_ERRORS`] errors or
    /// [`MAX_CONSECUTIVE_DEGENERATE`] degenerate outputs marks the model broken (same path as a
    /// stuck inference: parked, then reloaded by the supervisor). A good result resets both.
    fn note_health(&self, eng: &Engine, error: Option<&anyhow::Error>, degenerate: bool) {
        if self.stuck.load(Ordering::Acquire) {
            return;
        }
        match error {
            Some(e) => {
                self.consecutive_degenerate.set(0);
                let n = self.consecutive_errors.get().saturating_add(1);
                self.consecutive_errors.set(n);
                if n >= MAX_CONSECUTIVE_ERRORS {
                    self.stuck_watch(eng).mark_broken(&format!(
                        "the last {n} inferences failed (last error: {e:#})"
                    ));
                }
            }
            None if degenerate => {
                self.consecutive_errors.set(0);
                let n = self.consecutive_degenerate.get().saturating_add(1);
                self.consecutive_degenerate.set(n);
                if n >= MAX_CONSECUTIVE_DEGENERATE {
                    self.stuck_watch(eng).mark_broken(&format!(
                        "the last {n} inferences returned only zeros (or NaN) in every output"
                    ));
                }
            }
            None => {
                self.consecutive_errors.set(0);
                self.consecutive_degenerate.set(0);
            }
        }
    }

    /// The night / IR rules ([`crate::model::filters::apply`]), then the static-box memory, on
    /// one request's detections; adds what each removed to `counts`, and the removed boxes
    /// with their reasons to `removed` when given (dataset collector).
    fn filter_detections(
        &self,
        dets: Vec<crate::model::Detection>,
        f: &InFlight,
        scene: Scene,
        live: &super::LiveSettings,
        counts: &mut SuppressCounts,
        mut removed: Option<&mut Vec<(crate::model::Detection, SuppressReason)>>,
    ) -> Vec<crate::model::Detection> {
        let (w, h) = (f.ctx.orig_w, f.ctx.orig_h);
        let (dets, c) = crate::model::filters::apply_recording(
            dets,
            &self.cfg.classes,
            w,
            h,
            scene,
            &live.filters,
            removed.as_deref_mut(),
        );
        *counts = c;
        let st = &live.static_suppression;
        let mut memory = self.static_memory.borrow_mut();
        if !st.enabled && memory.keys() == 0 {
            return dets;
        }
        let key = crate::static_memory::camera_key(&self.cfg.name, &f.meta.image_name, w, h);
        let before = removed.as_ref().map(|_| dets.clone());
        let (dets, n) = memory.apply(&key, dets, &self.cfg.classes, st, Instant::now());
        counts.add(SuppressReason::Static, n);
        if let (Some(v), Some(before)) = (removed, before)
            && n > 0
        {
            v.extend(super::collect::removed_by_static(&before, &dets));
        }
        dets
    }

    /// Collect `slot`'s finished inference, postprocess on this thread and reply. A request
    /// with verifier candidates is handed to the verify lane instead ([`super::verify`]),
    /// which replies once the verifier answered (or the request's deadline passed): this
    /// thread never waits for another model.
    ///
    /// Stages, in this order: threshold, collector record, collect, postprocess (+ filters),
    /// health, the done line, then the response: handed to the verify lane (which replies),
    /// or replied here, then the collector and the image saver get the frame.
    fn finish(&self, eng: &mut Engine, slot: usize, f: InFlight) {
        let name = &self.cfg.name;
        let infer_d = f
            .done
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(f.submitted);
        let t = Instant::now();
        let live = f.meta.live.clone();
        let (threshold, threshold_source) = request_threshold(&eng.device, &f);
        // Dataset collector: the record of this request (None: collection off). With the
        // `near_threshold` bucket on, postprocessing runs `margin` below the threshold and the
        // boxes under it are only recorded: greedy NMS and its caps keep the highest scores,
        // so the boxes at or above the threshold are exactly those of a run at the threshold.
        let near_margin = live
            .collect
            .wants_below_threshold()
            .then(|| live.collect.buckets.near_margin());
        let mut collect_job = super::collect::Job::start(
            &f.meta,
            &crate::collect::DraftInput {
                model: name,
                device: &eng.device,
                request_id: &f.meta.request_id,
                routed_from: f.meta.routed_from.as_deref(),
                image_name: &f.meta.image_name,
                stats: f.stats,
                width: f.ctx.orig_w,
                height: f.ctx.orig_h,
                threshold,
                near_margin,
            },
        );
        let params = PostParams {
            confidence_threshold: post_threshold(
                threshold,
                near_margin.filter(|_| collect_job.is_some()),
            ),
            nms_iou: live.nms_iou,
        };
        let collected = self.collect_outputs(eng, slot, &f);
        let Post {
            result,
            degenerate,
            suppressed,
        } = self.postprocess(
            eng,
            slot,
            &f,
            collected,
            &params,
            threshold,
            collect_job.as_mut(),
        );
        self.metrics.record_suppressed(&suppressed);
        self.note_health(eng, result.as_ref().err(), degenerate);
        let post_d = t.elapsed();
        let line = DoneLine::new(
            name,
            &eng.device,
            &f,
            slot,
            threshold,
            threshold_source,
            &suppressed,
            infer_d,
            post_d,
        );
        let process_d = line.process;
        let (scene, crop) = (line.scene, line.verify_crop);
        let InFlight {
            meta,
            img,
            ctx,
            stats,
            ..
        } = f;
        // The annotated image, drawn, encoded and written after the reply ([`save`]).
        let mut save_job = None;
        let resp = match result {
            Ok(predictions) => {
                self.metrics.inference_ms.record(millis_u64(infer_d));
                let resp = self.success_response(eng, predictions, infer_d, process_d);
                let plan = match (&live.verify, may_verify(&meta)) {
                    (Some(cfg), true) => {
                        crate::verify::plan(&resp.predictions, ctx.orig_w, ctx.orig_h, scene, cfg)
                    }
                    _ => crate::verify::Plan::default(),
                };
                if !plan.is_empty() {
                    let deadline = meta.enqueued + meta.timeout();
                    let pending = Box::new(Pending {
                        reply: meta.reply,
                        resp,
                        img,
                        stats,
                        plan,
                        live,
                        request_id: meta.request_id,
                        image_name: meta.image_name,
                        enqueued: meta.enqueued,
                        deadline,
                        handed_off: Instant::now(),
                        line,
                        collect: collect_job.take(),
                        upload: meta.upload,
                    });
                    self.hand_off(pending);
                    return;
                }
                self.metrics.process_ms.record(millis_u64(process_d));
                line.emit(&resp.predictions, None);
                if !crop {
                    save_job = SaveJob::for_answer(
                        name,
                        &live,
                        &meta.image_name,
                        img,
                        &resp.predictions,
                        meta.upload.as_ref(),
                    );
                }
                resp
            }
            Err(e) => {
                collect_job = None;
                warn!(
                    model = %name,
                    device = %eng.device,
                    request_id = %meta.request_id,
                    image = %meta.image_name,
                    inference_ms = infer_d.as_secs_f64() * 1e3,
                    process_ms = process_d.as_secs_f64() * 1e3,
                    round_trip_ms = meta.enqueued.elapsed().as_secs_f64() * 1e3,
                    "detection failed: {e:#}"
                );
                let mut r = self.error_response(format!("{e:#}"), ms(infer_d), ms(process_d));
                r.executionProvider = eng.provider.clone();
                r
            }
        };
        // The collector and the image saver get the frame after the reply went out (each a
        // `try_send`, never waits).
        let after = After {
            collect: collect_job.map(|job| (job, resp.predictions.clone())),
            save: save_job,
        };
        self.reply(meta, resp);
        after.run(self.saver().as_deref());
    }

    /// `slot`'s finished outputs: under the stuck watchdog (the request's timeout) when no
    /// completion arrived in time ([`InFlight::overdue`]). A dead session marks the model.
    fn collect_outputs(
        &self,
        eng: &mut Engine,
        slot: usize,
        f: &InFlight,
    ) -> Result<Vec<crate::model::NamedOutput>> {
        let collected = if f.overdue {
            let watch = self.stuck_watch(eng);
            let limit = f.meta.timeout();
            collect_watched(&watch, limit, || eng.backend.collect(slot))
        } else {
            eng.backend.collect(slot)
        };
        if let Err(e) = &collected {
            self.mark_if_dead(eng, f.submitted, e);
        }
        collected
    }

    /// The family's postprocess of `collected` at `params` (the slot's buffers go back to the
    /// backend), then the request's predictions: a verifier check's raw detections in the crop
    /// (see `Purpose::Verify`), else the night / IR rules, the static memory and the object
    /// filter. With `collect_job`, the boxes under `threshold` and the ones the filters removed
    /// go to its record.
    #[allow(
        clippy::too_many_arguments,
        reason = "one stage of `finish`: each argument is one of its inputs, named"
    )]
    fn postprocess(
        &self,
        eng: &mut Engine,
        slot: usize,
        f: &InFlight,
        collected: Result<Vec<crate::model::NamedOutput>>,
        params: &PostParams,
        threshold: f32,
        collect_job: Option<&mut super::collect::Job>,
    ) -> Post {
        let live = &f.meta.live;
        let scene = f.stats.scene();
        let mut degenerate = false;
        let mut suppressed = SuppressCounts::default();
        let result = collected.and_then(|outs| {
            degenerate = outputs_degenerate(&outs);
            let dets = eng.family.postprocess(&outs, &f.ctx, params);
            eng.backend.recycle(slot, outs);
            let dets = dets?;
            if f.meta.verify_crop {
                // A verifier check: the raw detections in the crop (see `Purpose::Verify`).
                return Ok(crate::model::to_predictions(
                    &dets,
                    &f.ctx,
                    &self.cfg.classes,
                    None,
                ));
            }
            let dets = match collect_job {
                Some(job) => {
                    let (raw, below): (Vec<_>, Vec<_>) =
                        dets.into_iter().partition(|d| d.score >= threshold);
                    let mut removed = Vec::new();
                    let kept = self.filter_detections(
                        raw.clone(),
                        f,
                        scene,
                        live,
                        &mut suppressed,
                        Some(&mut removed),
                    );
                    job.draft
                        .set_detections(&raw, &below, &removed, &self.cfg.classes);
                    kept
                }
                None => self.filter_detections(dets, f, scene, live, &mut suppressed, None),
            };
            let filter = (!live.object_filter.is_empty()).then_some(live.object_filter.as_slice());
            Ok(crate::model::to_predictions(
                &dets,
                &f.ctx,
                &self.cfg.classes,
                filter,
            ))
        });
        Post {
            result,
            degenerate,
            suppressed,
        }
    }

    /// The answer to a request that found `predictions`.
    fn success_response(
        &self,
        eng: &Engine,
        predictions: Vec<crate::api::Prediction>,
        infer_d: Duration,
        process_d: Duration,
    ) -> VisionDetectionResponse {
        VisionDetectionResponse {
            success: true,
            message: format!("Found {} objects", predictions.len()),
            error: None,
            count: i32::try_from(predictions.len()).unwrap_or(i32::MAX),
            predictions,
            command: "detect".into(),
            moduleId: crate::MODULE_ID.to_string(),
            executionProvider: eng.provider.clone(),
            canUseGPU: self.cfg.can_use_gpu,
            inferenceMs: ms(infer_d),
            processMs: ms(process_d),
            analysisRoundTripMs: 0,
            dropped: None,
        }
    }

    /// The image saver of this worker: a test's ([`super::TestFaults::saver`]), else None (the
    /// process-wide one).
    #[cfg(test)]
    fn saver(&self) -> Option<Arc<save::Saver>> {
        self.cfg.faults.saver.clone()
    }

    /// The image saver of this worker: None, the process-wide one.
    #[cfg(not(test))]
    #[allow(
        clippy::unused_self,
        reason = "the test build reads the worker's faults"
    )]
    fn saver(&self) -> Option<Arc<save::Saver>> {
        None
    }

    /// Give a request with verifier candidates to the verify lane (started on first use). If
    /// the lane cannot run, the request is answered here with its candidates unverified.
    fn hand_off(&self, pending: Box<Pending>) {
        let mut lane = self.verify_lane.borrow_mut();
        let now = Instant::now();
        if lane.is_none() && VERIFY_SPAWN.with_borrow_mut(|b| b.may_try(now)) {
            match Lane::spawn(self.lane_ctx()) {
                Ok(l) => {
                    *lane = Some(l);
                    VERIFY_SPAWN.with_borrow_mut(SpawnBackoff::started);
                }
                Err(e) => {
                    let skipped = VERIFY_SPAWN.with_borrow_mut(|b| b.failed(now));
                    warn!(
                        model = %self.cfg.name,
                        similar_since_last_warning = skipped,
                        "could not start the verifier thread ({e}); answering unverified and \
                         trying again in {} s; the system is out of threads or memory",
                        VERIFY_SPAWN_BACKOFF.as_secs()
                    );
                }
            }
        }
        let pending = match lane.as_ref() {
            Some(l) => match l.submit(pending) {
                Ok(()) => return,
                Err(p) => p,
            },
            None => pending,
        };
        super::verify::complete_unverified(
            &self.lane_ctx(),
            pending,
            super::verify::Failure::Stopped,
        );
    }

    /// What the verify lane needs of this worker.
    fn lane_ctx(&self) -> LaneCtx {
        LaneCtx {
            name: self.cfg.name.clone(),
            metrics: self.metrics.clone(),
            peers: self.cfg.peers.clone(),
            saver: self.saver(),
        }
    }
}

/// The queue-timeout drop warning of [`WorkerCtx::timed_out`] at `now_ms` (on
/// [`ModelMetrics::now_ms`]), throttled by `metrics.timeout_log`: a slowed device drops at the
/// camera rate, and each line costs the worker thread (the Event Log write in the service) while
/// it is already behind. `Some(held back)` when it logged.
fn warn_timed_out(
    metrics: &ModelMetrics,
    now_ms: u64,
    model: &str,
    request_id: &str,
    waited: Duration,
    timeout: Duration,
) -> Option<u64> {
    let held_back = metrics.timeout_log.allow(now_ms)?;
    warn!(
        model = %model,
        request_id = %request_id,
        waited_ms = millis_u64(waited),
        timeout_ms = millis_u64(timeout),
        similar_since_last_warning = held_back,
        "dropping request that waited longer than the request timeout (server overloaded): the \
         model serves slower than images arrive; lower the cameras' trigger rate or use a faster \
         device or model"
    );
    Some(held_back)
}

/// The bad-image warning of [`WorkerCtx::decode_failure`] at `now_ms`, throttled by
/// `metrics.decode_error_log`, with the request id and image name of the request that gets
/// through (`(request id, image name)`). `Some(held back)` when it logged.
fn warn_decode_failure(
    metrics: &ModelMetrics,
    now_ms: u64,
    model: &str,
    (request_id, image): (&str, &str),
    e: &anyhow::Error,
) -> Option<u64> {
    let held_back = metrics.decode_error_log.allow(now_ms)?;
    warn!(
        model = %model,
        request_id = %request_id,
        image = %image,
        similar_since_last_warning = held_back,
        "detection failed: {e:#}"
    );
    Some(held_back)
}

/// Least time between two tries to start a worker's verify lane after one failed: a failed
/// start means the system is out of threads or memory, and trying (and warning) on every
/// verified request would only add load on the worker thread.
const VERIFY_SPAWN_BACKOFF: Duration = Duration::from_secs(10);

/// When a worker may try to start its verify lane again after a failed start, and how many
/// requests went unverified without a try meanwhile (reported by the next failure's warning).
#[derive(Debug, Default)]
struct SpawnBackoff {
    /// No try before this (None: try now).
    next: Option<Instant>,
    skipped: u64,
}

impl SpawnBackoff {
    const fn new() -> Self {
        Self {
            next: None,
            skipped: 0,
        }
    }

    /// Whether a start may be tried at `now` (a refusal is counted).
    fn may_try(&mut self, now: Instant) -> bool {
        if self.next.is_some_and(|n| now < n) {
            self.skipped = self.skipped.saturating_add(1);
            return false;
        }
        true
    }

    /// The try at `now` failed: none again for [`VERIFY_SPAWN_BACKOFF`]. The requests skipped
    /// since the last failure.
    fn failed(&mut self, now: Instant) -> u64 {
        self.next = now.checked_add(VERIFY_SPAWN_BACKOFF);
        std::mem::take(&mut self.skipped)
    }

    /// A start succeeded: forget earlier failures.
    fn started(&mut self) {
        *self = Self::new();
    }
}

thread_local! {
    /// The verify lane's start backoff of the worker on this thread. Per thread because the lane
    /// slot ([`WorkerCtx::verify_lane`]) is: one worker thread per model, and a reload starts a
    /// new thread with a fresh backoff.
    static VERIFY_SPAWN: RefCell<SpawnBackoff> = const { RefCell::new(SpawnBackoff::new()) };
}

/// Request `f`'s confidence threshold on `device` (the one the model loaded on) and where it
/// came from, for the done line: the profile for that device (`auto` and a fallback included),
/// else the model's / global value, or on a night / IR frame the model's `night_threshold`; a
/// request's `min_confidence` wins over all of them.
fn request_threshold(device: &str, f: &InFlight) -> (f32, String) {
    let (configured, source) = f.meta.live.threshold_for_scene(device, f.stats.scene());
    threshold_with_source(f.min_confidence(), configured, &source)
}

/// [`effective_threshold`] of `min_confidence` over `configured`, labelled `source` unless the
/// request's value is what changed it ("request").
fn threshold_with_source(
    min_confidence: f32,
    configured: f32,
    source: &crate::config::ThresholdSource,
) -> (f32, String) {
    let threshold = effective_threshold(min_confidence, configured);
    let label = if threshold == effective_threshold(0.0, configured) {
        source.label()
    } else {
        "request".to_string()
    };
    (threshold, label)
}

/// The confidence threshold postprocessing runs at: `threshold`, or `below` under it (never
/// under [`crate::model::nms::MIN_CONFIDENCE_FLOOR`]) while the collector records the boxes
/// just below the threshold (`near_threshold` bucket).
fn post_threshold(threshold: f32, below: Option<f32>) -> f32 {
    match below {
        Some(m) => (threshold - m).max(crate::model::nms::MIN_CONFIDENCE_FLOOR),
        None => threshold,
    }
}

/// What postprocessing one request gave ([`WorkerCtx::postprocess`]).
#[derive(Debug)]
struct Post {
    /// The predictions to answer with, or why there are none.
    result: Result<Vec<crate::api::Prediction>>,
    /// No output held a finite non-zero value ([`outputs_degenerate`]; health).
    degenerate: bool,
    /// What the night / IR rules and the static memory removed.
    suppressed: SuppressCounts,
}

/// The one debug line per detection ("detection done"), built in `finish` and written there or,
/// for a verified request, by the verify lane with the verifier's outcome.
#[derive(Debug)]
pub(super) struct DoneLine {
    pub(super) model: String,
    pub(super) device: String,
    pub(super) request_id: String,
    pub(super) routed_from: Option<String>,
    pub(super) verify_crop: bool,
    pub(super) image: String,
    pub(super) size: String,
    pub(super) scene: Scene,
    pub(super) luma: f64,
    pub(super) chroma: f64,
    pub(super) suppressed: String,
    /// The confidence threshold this request ran with.
    pub(super) threshold: f32,
    /// Where it came from: "profile for <device>", "model", "global" or "request"
    /// (`min_confidence`).
    pub(super) threshold_source: String,
    pub(super) slot: usize,
    pub(super) queue_ms: u64,
    pub(super) decode_ms: f64,
    pub(super) preprocess_ms: f64,
    pub(super) inference_ms: f64,
    pub(super) postprocess_ms: f64,
    /// Decode + preprocess + inference + postprocess (the verifier's time is added by the lane).
    pub(super) process: Duration,
    pub(super) enqueued: Instant,
}

/// The verifier part of a [`DoneLine`].
#[derive(Debug)]
pub(super) struct VerifyLog {
    pub(super) verifier: String,
    pub(super) ms: f64,
    /// "kept 1, rejected 2" (and the failure, e.g. "unverified 1 (timeout)").
    pub(super) outcome: String,
}

impl DoneLine {
    /// The line of request `f`, answered by `model` on `device` from `slot`, with the threshold
    /// it ran with and the times of its stages.
    #[allow(
        clippy::too_many_arguments,
        reason = "a constructor: each argument is one of the line's sources, named"
    )]
    fn new(
        model: &str,
        device: &str,
        f: &InFlight,
        slot: usize,
        threshold: f32,
        threshold_source: String,
        suppressed: &SuppressCounts,
        infer_d: Duration,
        post_d: Duration,
    ) -> Self {
        Self {
            model: model.to_string(),
            device: device.to_string(),
            request_id: f.meta.request_id.clone(),
            routed_from: f.meta.routed_from.clone(),
            verify_crop: f.meta.verify_crop,
            image: f.meta.image_name.clone(),
            size: format!("{}x{}", f.ctx.orig_w, f.ctx.orig_h),
            scene: f.stats.scene(),
            luma: f64::from(f.stats.mean_luma),
            chroma: f64::from(f.stats.chroma_spread.min(f.stats.tint_spread)),
            suppressed: suppressed_summary(suppressed),
            threshold,
            threshold_source,
            slot,
            queue_ms: millis_u64(f.submitted.saturating_duration_since(f.meta.enqueued)),
            decode_ms: f.decode.as_secs_f64() * 1e3,
            preprocess_ms: f.preprocess.as_secs_f64() * 1e3,
            inference_ms: infer_d.as_secs_f64() * 1e3,
            postprocess_ms: post_d.as_secs_f64() * 1e3,
            process: f.decode + f.preprocess + infer_d + post_d,
            enqueued: f.meta.enqueued,
        }
    }

    pub(super) fn emit(&self, predictions: &[crate::api::Prediction], verify: Option<&VerifyLog>) {
        let process = self.process.as_secs_f64() * 1e3 + verify.map_or(0.0, |v| v.ms);
        debug!(
            model = %self.model,
            device = %self.device,
            request_id = %self.request_id,
            routed_from = self.routed_from.as_deref(),
            verify_crop = self.verify_crop.then_some(true),
            image = %self.image,
            size = %self.size,
            scene = %self.scene,
            luma = self.luma,
            chroma = self.chroma,
            count = predictions.len(),
            suppressed = %self.suppressed,
            threshold = crate::config::round_threshold_f64(self.threshold),
            threshold_source = %self.threshold_source,
            detections = %detections_summary(predictions),
            verifier = verify.map(|v| v.verifier.as_str()),
            verify = verify.map(|v| v.outcome.as_str()),
            verify_ms = verify.map(|v| v.ms),
            slot = self.slot,
            queue_ms = self.queue_ms,
            decode_ms = self.decode_ms,
            preprocess_ms = self.preprocess_ms,
            inference_ms = self.inference_ms,
            postprocess_ms = self.postprocess_ms,
            process_ms = process,
            round_trip_ms = self.enqueued.elapsed().as_secs_f64() * 1e3,
            "detection done"
        );
    }
}

/// What a request brings: the encoded image (HTTP), or a frame another worker decoded.
enum Input {
    Bytes(bytes::Bytes),
    Decoded {
        img: crate::image::RgbImage,
        stats: crate::image_stats::ImageStats,
        decode: Duration,
    },
}

/// True while `meta` may still go to the model's night model (it has one, the request came
/// from the HTTP layer and was not tried yet).
fn may_route(meta: &Meta) -> bool {
    meta.forwardable && meta.live.night_model.is_some()
}

/// True when `meta`'s answer may go through the verifier cascade (`verify` is set and it is
/// not itself a verifier check); whether it does depends on the scene and the candidates.
pub(super) fn may_verify(meta: &Meta) -> bool {
    !meta.verify_crop && meta.live.verify.is_some()
}

/// Answer `reply` with `resp` for the request of `image_name` (model `model`): the
/// [`ORDER_TARGET`] "replied" trace (`tests/pipeline.rs` checks FIFO order with it), then the
/// send; a client that went away is only logged at debug level. The worker and its verify lane
/// both answer through here.
pub(super) fn send_reply(
    model: &str,
    image_name: &str,
    reply: Reply,
    resp: VisionDetectionResponse,
) {
    trace!(
        target: ORDER_TARGET,
        model = %model,
        image = %image_name,
        success = resp.success,
        "replied"
    );
    if !reply.send(resp) {
        debug!(model = %model, "client went away before the response was sent");
    }
}

impl From<super::PeerMiss> for RouteFallback {
    fn from(m: super::PeerMiss) -> Self {
        match m {
            super::PeerMiss::Unknown => RouteFallback::Unknown,
            super::PeerMiss::NotReady => RouteFallback::NotReady,
            super::PeerMiss::QueueFull => RouteFallback::QueueFull,
        }
    }
}

/// "not served", "not ready", ... for the fallback warning.
fn fallback_text(reason: RouteFallback) -> &'static str {
    match reason {
        RouteFallback::Unknown => "not served (unknown name, disabled or removed)",
        RouteFallback::NotReady => "not ready (loading, lazy or failed)",
        RouteFallback::QueueFull => "busy (its queue is full)",
        RouteFallback::Stopped => "stopping (being reloaded)",
    }
}

/// Longest the warm-up inferences of one device candidate may take. A device that does not
/// answer by then (a hung driver, a GPU that is not up yet after boot) is abandoned and the
/// next candidate tried, instead of the worker thread waiting forever with the model
/// "initializing". Generous: the first inference on a cold GPU can take many seconds.
pub(crate) const WARMUP_TIMEOUT: Duration = Duration::from_secs(180);

/// A warm-up inference that did not finish within [`WARMUP_TIMEOUT`]; the backend it ran on
/// must not be dropped on the worker thread (see `load_on`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WarmupTimeout {
    pub device: String,
    pub waited: Duration,
}

impl std::fmt::Display for WarmupTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "warm-up inference on {} did not finish within {} s; the device driver may have \
             hung or the GPU is not available yet",
            self.device,
            self.waited.as_secs()
        )
    }
}

impl std::error::Error for WarmupTimeout {}

/// Warm every slot of `backend` up on one frame, within `timeout` in total: slot 0 first and
/// timed (its outputs are returned, with the inference time in ms), then every other slot at
/// once (first runs allocate per request / session). Waits on the backend's completion channel
/// so a hung device yields [`WarmupTimeout`] instead of blocking in `collect`, in slices of
/// [`POLL`] so a cancelled `stop` yields [`LoadStopped`] within one slice. Either way an
/// inference may still be in flight: the caller hands the backend to a reaper.
pub(crate) fn warm_up(
    backend: &mut Backend,
    chw: &[f32],
    extra: &[crate::model::ExtraInput],
    timeout: Duration,
    stop: &CancellationToken,
) -> Result<(Vec<crate::model::NamedOutput>, u64)> {
    let started = Instant::now();
    let deadline = started + timeout;
    let device = backend.info().device.spec.clone();
    let wait = |backend: &Backend| -> Result<()> {
        loop {
            if stop.is_cancelled() {
                return Err(LoadStopped.into());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match backend.completions().recv_timeout(left.min(POLL)) {
                Ok(_) => return Ok(()),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) if left > POLL => {}
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    return Err(WarmupTimeout {
                        device: device.clone(),
                        waited: started.elapsed(),
                    }
                    .into());
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("the inference runtime stopped during the warm-up")
                }
            }
        }
    };
    // Completions of earlier synchronous calls (none expected) must not count.
    while backend.completions().try_recv().is_ok() {}
    let t = Instant::now();
    backend.submit(0, chw, extra).context("warm-up inference")?;
    wait(backend).context("warm-up inference")?;
    let outs = backend.collect(0).context("warm-up inference")?;
    let warmup_ms = millis_u64(t.elapsed());
    let others = 1..backend.slots();
    for s in others.clone() {
        backend.submit(s, chw, extra).context("warm-up inference")?;
    }
    for _ in others.clone() {
        wait(backend).context("warm-up inference")?;
    }
    for s in others {
        let o = backend.collect(s).context("warm-up inference")?;
        backend.recycle(s, o);
    }
    while backend.completions().try_recv().is_ok() {}
    Ok((outs, warmup_ms))
}

/// True when `outs` cannot come from a working inference: no output holds a single finite,
/// non-zero value (every value is zero, NaN or infinite). A reset GPU hands back unwritten
/// buffers (zeros) or garbage; a real frame always yields some finite non-zero coordinate or
/// score. Empty outputs (a `[1,0,6]` end-to-end result) are fine, and so is an occasional
/// overflowed (f16) logit among normal values.
pub(crate) fn outputs_degenerate(outs: &[crate::model::NamedOutput]) -> bool {
    use crate::model::OutputBuf;
    let mut any_values = false;
    for o in outs {
        let normal = match &o.data {
            OutputBuf::F32(v) => {
                any_values |= !v.is_empty();
                v.iter().any(|x| x.is_finite() && *x != 0.0)
            }
            OutputBuf::I64(v) => {
                any_values |= !v.is_empty();
                v.iter().any(|&x| x != 0)
            }
            OutputBuf::I32(v) => {
                any_values |= !v.is_empty();
                v.iter().any(|&x| x != 0)
            }
        };
        if normal {
            return false;
        }
    }
    any_values
}

/// What a detection found, for the log: `person 66%, car 91%` (response order), empty for none.
fn detections_summary(predictions: &[crate::api::Prediction]) -> String {
    predictions
        .iter()
        .map(|p| format!("{} {:.0}%", p.label, p.confidence * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

/// "night_threshold 2, static 1" (empty when nothing was suppressed), for the request log.
fn suppressed_summary(c: &SuppressCounts) -> String {
    c.nonzero()
        .map(|(r, n)| format!("{} {n}", r.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `Failed` text of a model that could not be loaded on any of its devices.
fn load_failure_message(e: &anyhow::Error) -> String {
    format!(
        "{e:#}; the model file may be damaged, of another family than configured, or need a \
         runtime that is not installed; check the log for each device's error, then fix \
         `family` / `device` or download the model again and save the config to reload it"
    )
}

/// True when a backend error means the runtime session is gone for good (no later inference
/// can succeed), so the model takes the same `Failed` path as a stuck inference.
pub(super) fn session_dead(e: &anyhow::Error) -> bool {
    crate::backend::is_session_dead(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ThresholdSource;

    /// QA-142: the threshold stage of `finish`. The configured value keeps its source; a
    /// request's `min_confidence` that changes it is "request" (a fraction, or a percentage).
    #[test]
    fn threshold_with_source_names_the_request_only_when_it_changed_the_value() {
        let t = |min, configured, source: ThresholdSource| {
            threshold_with_source(min, configured, &source)
        };
        assert_eq!(t(0.0, 0.5, ThresholdSource::Model), (0.5, "model".into()));
        assert_eq!(
            t(0.7, 0.5, ThresholdSource::Global),
            (0.7, "request".into())
        );
        assert_eq!(
            t(40.0, 0.5, ThresholdSource::Global),
            (0.4, "request".into())
        );
        // The same value as configured, or one ignored (over 100), keeps the source.
        assert_eq!(t(0.5, 0.5, ThresholdSource::Model), (0.5, "model".into()));
        assert_eq!(
            t(250.0, 0.5, ThresholdSource::Device("openvino:gpu".into())),
            (0.5, "profile for openvino:gpu".into())
        );
        let floor = crate::model::nms::MIN_CONFIDENCE_FLOOR;
        assert_eq!(
            t(0.0, 0.0, ThresholdSource::Global),
            (floor, "global".into())
        );
    }

    /// A night / IR frame runs at the model's `night_threshold`; a day frame at the fallback.
    #[test]
    fn request_threshold_follows_the_scene() {
        let mut f = InFlight::for_test(1);
        f.meta.live = Arc::new(super::super::LiveSettings {
            confidence_threshold: 0.5,
            fallback_source: ThresholdSource::Model,
            night_threshold: Some(0.3),
            ..super::super::LiveSettings::default()
        });
        assert_eq!(request_threshold("openvino:cpu", &f), (0.5, "model".into()));
        f.stats.is_ir = true;
        assert_eq!(
            request_threshold("openvino:cpu", &f),
            (0.3, "night threshold".into())
        );
        f.meta.min_confidence = 0.8;
        assert_eq!(
            request_threshold("openvino:cpu", &f),
            (0.8, "request".into())
        );
    }

    /// Postprocessing runs below the threshold only while the collector records the boxes
    /// there, and never under the floor.
    #[test]
    fn post_threshold_goes_below_only_for_the_collector() {
        assert!((post_threshold(0.5, None) - 0.5).abs() < 1e-6);
        assert!((post_threshold(0.5, Some(0.1)) - 0.4).abs() < 1e-6);
        let floor = crate::model::nms::MIN_CONFIDENCE_FLOOR;
        assert!((post_threshold(0.05, Some(0.1)) - floor).abs() < 1e-6);
    }

    /// The done line carries the request's fields and the sum of its stage times.
    #[test]
    fn done_line_from_a_request() {
        let mut f = InFlight::for_test(1);
        f.meta.request_id = "abcd1234".into();
        f.meta.image_name = "cam1.jpg".into();
        f.meta.routed_from = Some("day".into());
        f.ctx.orig_w = 640;
        f.ctx.orig_h = 480;
        f.stats.is_dark = true;
        f.stats.mean_luma = 20.0;
        f.stats.chroma_spread = 3.0;
        f.stats.tint_spread = 2.0;
        f.decode = Duration::from_millis(4);
        f.preprocess = Duration::from_millis(3);
        f.submitted = f.meta.enqueued + Duration::from_millis(7);
        let mut suppressed = SuppressCounts::default();
        suppressed.add(SuppressReason::Static, 1);
        let line = DoneLine::new(
            "m",
            "openvino:gpu",
            &f,
            2,
            0.45,
            "request".into(),
            &suppressed,
            Duration::from_millis(20),
            Duration::from_millis(1),
        );
        assert_eq!(
            (line.model.as_str(), line.device.as_str()),
            ("m", "openvino:gpu")
        );
        assert_eq!(line.request_id, "abcd1234");
        assert_eq!(line.routed_from.as_deref(), Some("day"));
        assert!(!line.verify_crop);
        assert_eq!(
            (line.image.as_str(), line.size.as_str()),
            ("cam1.jpg", "640x480")
        );
        assert_eq!(line.scene, Scene::Night);
        assert!((line.luma - 20.0).abs() < 1e-9 && (line.chroma - 2.0).abs() < 1e-9);
        assert_eq!(line.suppressed, suppressed_summary(&suppressed));
        assert!((line.threshold - 0.45).abs() < 1e-6);
        assert_eq!((line.threshold_source.as_str(), line.slot), ("request", 2));
        assert_eq!(line.queue_ms, 7);
        assert!((line.decode_ms - 4.0).abs() < 1e-9 && (line.preprocess_ms - 3.0).abs() < 1e-9);
        assert!((line.inference_ms - 20.0).abs() < 1e-9);
        assert!((line.postprocess_ms - 1.0).abs() < 1e-9);
        assert_eq!(line.process, Duration::from_millis(28));
        assert_eq!(line.enqueued, f.meta.enqueued);
    }

    #[test]
    fn detections_summary_lists_labels_with_percent() {
        let p = |label: &str, confidence: f32| crate::api::Prediction {
            x_max: 0,
            x_min: 0,
            y_max: 0,
            y_min: 0,
            confidence,
            label: label.into(),
        };
        assert_eq!(
            detections_summary(&[p("person", 0.664), p("car", 0.91)]),
            "person 66%, car 91%"
        );
        assert_eq!(detections_summary(&[]), "");
    }

    fn fake_backend(
        slots: usize,
        warmup_hangs: bool,
    ) -> (Backend, crate::backend::fake::FakeScript) {
        use crate::backend::LoadRequest;
        let mut script = crate::backend::fake::FakeScript::new(slots, (64, 64));
        script.warmup_hangs = warmup_hangs;
        let req = LoadRequest {
            path: std::path::PathBuf::from("models/fake.onnx"),
            requested: "auto".into(),
            gpu_precision: None,
            no_cache: false,
        };
        let b = script
            .compile(
                &req,
                "openvino:cpu",
                false,
                &crate::profile::DeviceTuning::SERIAL,
            )
            .into_backend();
        (Backend::Fake(b), script)
    }

    #[test]
    fn warm_up_covers_every_slot_and_leaves_no_completion_behind() {
        use crate::backend::fake::Call;
        let (mut backend, script) = fake_backend(3, false);
        let chw = vec![0.0f32; 3 * 64 * 64];
        let (outs, ms) = warm_up(
            &mut backend,
            &chw,
            &[],
            Duration::from_secs(5),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(outs.len(), 1);
        assert!(ms < 1000);
        let calls = script.calls();
        assert_eq!(calls[0], Call::WarmupSubmit { slot: 0 });
        assert_eq!(calls[1], Call::Collect { slot: 0 });
        assert!(calls.contains(&Call::WarmupSubmit { slot: 1 }));
        assert!(calls.contains(&Call::WarmupSubmit { slot: 2 }));
        assert!(calls.contains(&Call::Collect { slot: 2 }));
        assert!(backend.completions().try_recv().is_err());
        assert!(!backend.is_busy(0) && !backend.is_busy(1) && !backend.is_busy(2));
    }

    #[test]
    fn warm_up_gives_up_on_a_device_that_never_answers() {
        let (mut backend, _script) = fake_backend(2, true);
        let chw = vec![0.0f32; 3 * 64 * 64];
        let t = Instant::now();
        let err = warm_up(
            &mut backend,
            &chw,
            &[],
            Duration::from_millis(100),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(err.is::<WarmupTimeout>(), "{err:#}");
        let text = format!("{err:#}");
        assert!(text.contains("did not finish"), "{text}");
        assert!(text.contains("openvino:cpu"), "{text}");
        // Still in flight: the caller leaks the backend instead of waiting for it.
        assert!(backend.is_busy(0));
        std::mem::forget(backend);
    }

    /// A stop during a warm-up that hangs ends it within one [`POLL`] slice, long before its
    /// timeout, with [`LoadStopped`] (not [`WarmupTimeout`]).
    #[test]
    fn warm_up_ends_soon_after_a_stop() {
        let (mut backend, _script) = fake_backend(1, true);
        let chw = vec![0.0f32; 3 * 64 * 64];
        let stop = CancellationToken::new();
        let canceller = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                stop.cancel();
            })
        };
        let t = Instant::now();
        let err = warm_up(&mut backend, &chw, &[], Duration::from_secs(60), &stop).unwrap_err();
        let waited = t.elapsed();
        canceller.join().unwrap();
        assert!(err.is::<LoadStopped>(), "{err:#}");
        assert!(!err.is::<WarmupTimeout>(), "{err:#}");
        assert!(waited < Duration::from_secs(10), "{waited:?}");
        assert!(backend.is_busy(0));
        std::mem::forget(backend);
    }

    #[test]
    fn panic_text_reads_str_and_string_payloads() {
        let s = std::panic::catch_unwind(|| panic!("plain")).unwrap_err();
        assert_eq!(panic_text(s.as_ref()), "plain");
        let n = 3;
        let f = std::panic::catch_unwind(move || panic!("formatted {n}")).unwrap_err();
        assert_eq!(panic_text(f.as_ref()), "formatted 3");
        let other = std::panic::catch_unwind(|| std::panic::panic_any(7u8)).unwrap_err();
        assert_eq!(panic_text(other.as_ref()), "non-string panic payload");
    }

    #[test]
    fn degenerate_outputs() {
        use crate::model::{NamedOutput, OutputBuf};
        let out = |data: OutputBuf| NamedOutput {
            name: "o".into(),
            shape: vec![],
            data,
        };
        assert!(!outputs_degenerate(&[]));
        assert!(!outputs_degenerate(&[out(OutputBuf::F32(vec![]))]));
        assert!(outputs_degenerate(&[out(OutputBuf::F32(vec![0.0; 8]))]));
        assert!(!outputs_degenerate(&[out(OutputBuf::F32(vec![
            0.0, 0.0, 0.5
        ]))]));
        // One overflowed logit among normal values is a working model; only zeros and
        // non-finite values everywhere are not.
        assert!(!outputs_degenerate(&[out(OutputBuf::F32(vec![
            1.0,
            f32::NAN
        ]))]));
        assert!(outputs_degenerate(&[out(OutputBuf::F32(vec![
            f32::INFINITY
        ]))]));
        assert!(outputs_degenerate(&[out(OutputBuf::F32(vec![
            0.0,
            f32::NAN,
            0.0
        ]))]));
        // Zero labels with non-zero scores are a legitimate RT-DETR result.
        assert!(!outputs_degenerate(&[
            out(OutputBuf::I64(vec![0, 0, 0])),
            out(OutputBuf::F32(vec![0.9, 0.2, 0.1]))
        ]));
        assert!(outputs_degenerate(&[
            out(OutputBuf::I64(vec![0, 0])),
            out(OutputBuf::I32(vec![0])),
            out(OutputBuf::F32(vec![0.0]))
        ]));
    }

    #[test]
    fn load_failure_says_what_to_do() {
        let m = load_failure_message(&anyhow::anyhow!("every device failed"));
        assert!(m.starts_with("every device failed; "), "{m}");
        assert!(m.contains("check the log"), "{m}");
    }

    /// RA-109 / RA-110: queue timeouts and bad images log one line per interval each (their
    /// own throttles, so one flood cannot hide the other), the held-back count on the next.
    #[test]
    fn timeout_and_bad_image_warnings_are_throttled_apart() {
        let m = ModelMetrics::new("m", "CPU", "");
        let every = u64::try_from(crate::metrics::LOG_THROTTLE_INTERVAL.as_millis()).unwrap();
        let (w, t) = (Duration::from_secs(20), Duration::from_secs(15));
        let e = anyhow::anyhow!("not an image");
        let timeouts: Vec<_> = (0..20)
            .map(|i| warn_timed_out(&m, 100 + i, "m", "r", w, t))
            .collect();
        let bad: Vec<_> = (0..20)
            .map(|i| warn_decode_failure(&m, 100 + i, "m", ("r", "cam.jpg"), &e))
            .collect();
        for got in [&timeouts, &bad] {
            assert_eq!(got.first(), Some(&Some(0)), "the first one logs");
            assert!(got.iter().skip(1).all(Option::is_none), "{got:?}");
        }
        assert_eq!(warn_timed_out(&m, 100 + every, "m", "r", w, t), Some(19));
        assert_eq!(
            warn_decode_failure(&m, 100 + every, "m", ("r", "cam.jpg"), &e),
            Some(19)
        );
    }

    /// The verify lane's start is tried again only after the backoff, and the warning of the
    /// next failure counts the requests that went unverified without a try.
    #[test]
    fn a_failed_verifier_start_backs_off() {
        let t0 = Instant::now();
        let mut b = SpawnBackoff::new();
        assert!(b.may_try(t0));
        assert_eq!(b.failed(t0), 0);
        for s in 0..9 {
            assert!(!b.may_try(t0 + Duration::from_secs(s)), "{s}");
        }
        assert!(b.may_try(t0 + VERIFY_SPAWN_BACKOFF));
        assert_eq!(b.failed(t0 + VERIFY_SPAWN_BACKOFF), 9);
        assert!(!b.may_try(t0 + VERIFY_SPAWN_BACKOFF));
        b.started();
        assert!(b.may_try(t0 + VERIFY_SPAWN_BACKOFF));
        assert_eq!(b.skipped, 0);
    }
}
