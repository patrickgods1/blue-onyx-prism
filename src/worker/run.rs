//! The worker thread's run loop ([`WorkerCtx::run`]): load (compile + warm-up on each device
//! candidate), then a `Select`-driven state machine over the work channel, the prep reply of the
//! oldest request and the backend's completions. Invariants: requests are answered in arrival
//! order; nothing is left in flight when the loop ends (shutdown aborts, then answers every
//! queued request); postprocessing runs on this thread.

use super::pipeline::{Done, InFlight, Job, Meta, Pipeline, Stage};
use super::verify::{Lane, LaneCtx, Pending};
use super::watch::{SlowCompile, StuckWatch, collect_watched, watch_slow_compile};
use super::{
    COMPILING_ON, Decoded, LoadGate, ORDER_TARGET, POLL, Purpose, Reply, WorkItem, WorkerConfig,
    WorkerHandle, auto_queue_size, effective_threshold, initializing_message, loaded_provider,
    millis_u64, ms, new_request_id,
};
use crate::api::VisionDetectionResponse;
use crate::backend::{Backend, Candidate, Completion, DeviceInfo, Runtimes};
use crate::image_stats::Scene;
use crate::metrics::{DropReason, ModelMetrics, RouteFallback};
use crate::model::filters::{SuppressCounts, SuppressReason};
use crate::model::preprocess::Preprocessor;
use crate::model::{Family, PostParams};
use crate::prep::{PrepJob, PrepPool, PrepSpec, Prepared};
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
    provider: String,
    /// Canonical spec of the device it loaded on ("openvino:gpu"), for logs.
    device: String,
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
    pub(super) shutdown: CancellationToken,
    /// Inferences in a row that returned an error (see [`MAX_CONSECUTIVE_ERRORS`]).
    pub(super) consecutive_errors: Cell<u32>,
    /// Inferences in a row whose outputs were degenerate (see [`outputs_degenerate`]).
    pub(super) consecutive_degenerate: Cell<u32>,
    /// Static-box memory of this model (kept across live-settings swaps; a reload starts a new
    /// worker and so an empty memory).
    pub(super) static_memory: RefCell<StaticMemory>,
    /// Preprocessor for requests forwarded with a decoded image ([`WorkItem::DetectDecoded`]),
    /// made on first use for the engine's geometry.
    pub(super) decoded_pre: RefCell<Option<(PrepSpec, Preprocessor)>>,
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

impl WorkerCtx {
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
        // The engine of a model whose inference got stuck and then failed (kept, not dropped,
        // until the thread exits: dropping waits for whatever else is in flight).
        let mut parked: Option<Engine> = None;
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
                    parked = engine.take();
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
        if let Some(eng) = engine.as_mut()
            && p.busy()
        {
            // Never leave an inference running past the worker (shutdown).
            eng.backend.abort();
        }
        self.answer_remaining(p, &rx);
        // Answers the verifications still waiting (unverified) and joins the lane thread.
        drop(self.verify_lane.borrow_mut().take());
        drop(parked);
        info!(model = %name, "worker exiting");
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
            let msg = match self.state.get() {
                ModelState::Failed(_) => {
                    self.metrics.record_drop(DropReason::Failed);
                    self.unavailable_message()
                }
                _ => {
                    self.metrics.record_drop(DropReason::Initializing);
                    initializing_message(name, self.state.detail().as_deref())
                }
            };
            let _ = meta.reply.send(self.error_response(msg, 0, 0));
            return;
        };
        trace!(
            target: ORDER_TARGET,
            model = %name,
            image = %meta.image_name,
            request_id = %meta.request_id,
            "accepted"
        );
        let image_data = match input {
            Input::Bytes(b) => b,
            Input::Decoded { img, stats, decode } => {
                let stage = if let Some(resp) = self.timed_out(&meta) {
                    Stage::Answer(resp)
                } else {
                    match self.prepare_decoded(eng, prep, img, stats, &meta) {
                        Ok(mut prepared) => {
                            prepared.decode = decode;
                            Stage::Ready(prepared)
                        }
                        Err(e) => Stage::Answer(self.failure(&meta, &eng.provider, &e)),
                    }
                };
                p.pending.push_back(Job { meta, stage });
                return;
            }
        };
        let stage = if let Some(resp) = self.timed_out(&meta) {
            Stage::Answer(resp)
        } else {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let job = PrepJob {
                image: image_data,
                spec: eng.spec,
                // The decoded frame is forwarded when the night model takes it, and cropped
                // for the verifier.
                keep_image: meta.live.save_image_path.is_some()
                    || may_route(&meta)
                    || may_verify(&meta),
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
        let original = live.collect.enabled.then(|| req.image_data.clone());
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
        };
        (meta, Input::Decoded { img, stats, decode })
    }

    /// Preprocess a forwarded, already decoded image for this model on this thread (the
    /// letterbox / stretch of one frame, plus this model's enhancement for the frame's scene).
    fn prepare_decoded(
        &self,
        eng: &Engine,
        prep: &PrepPool,
        img: crate::image::RgbImage,
        stats: crate::image_stats::ImageStats,
        meta: &Meta,
    ) -> Result<Prepared> {
        let mut cell = self.decoded_pre.borrow_mut();
        let (spec, pre) = match cell.take() {
            Some((spec, pre)) if spec == eng.spec => (spec, pre),
            _ => (eng.spec, eng.spec.preprocessor()),
        };
        let mut pre = pre;
        let keep = (meta.live.save_image_path.is_some() && !meta.verify_crop) || may_verify(meta);
        // This model's own modes (the sender's scene statistics, its original frame).
        let modes = meta.live.preprocess;
        let tensors = prep.tensors();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PrepPool::prepare_image(&mut pre, tensors, img, stats, &spec, keep, modes)
        }));
        match result {
            Ok(r) => {
                *cell = Some((spec, pre));
                r
            }
            // The preprocessor may be half-updated: the next request makes a new one.
            Err(_) => Err(anyhow::anyhow!("preprocessing panicked")),
        }
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
        };
        Some(self.fall_back(meta, prepared, d.img, &night, reason))
    }

    /// The served, ready worker of `night` with room for one more request, or why not (a lazy
    /// night model is asked to compile).
    fn night_target(&self, night: &str) -> std::result::Result<Arc<WorkerHandle>, RouteFallback> {
        use crate::registry::normalize_name;
        let h = self
            .cfg
            .peers
            .get(night)
            .filter(|h| normalize_name(&h.name) != normalize_name(&self.cfg.name))
            .ok_or(RouteFallback::Unknown)?;
        if h.state.get() != ModelState::Ready {
            h.trigger_load();
            return Err(RouteFallback::NotReady);
        }
        if h.is_full() {
            return Err(RouteFallback::QueueFull);
        }
        Ok(h)
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
        warn!(
            model = %self.cfg.name,
            request_id = %meta.request_id,
            waited_ms = millis_u64(waited),
            timeout_ms = millis_u64(timeout),
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
        warn!(
            model = %self.cfg.name,
            request_id = %meta.request_id,
            image = %meta.image_name,
            "detection failed: {e:#}"
        );
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
            Ok(Err(e)) => {
                self.metrics.record_decode_error();
                Stage::Answer(self.failure(meta, &provider, &e))
            }
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
            match eng.backend.submit(slot, &prepared.chw, &extra) {
                Ok(seq) => {
                    let Prepared {
                        ctx,
                        img,
                        stats,
                        decode,
                        preprocess,
                        ..
                    } = prepared; // the tensor buffer goes back to the pool here
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
        trace!(
            target: ORDER_TARGET,
            model = %self.cfg.name,
            image = %meta.image_name,
            success = resp.success,
            "replied"
        );
        if !meta.reply.send(resp) {
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
                let msg = load_failure_message(&e);
                error!(model = %self.cfg.name, "{msg}");
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
        *self.device.write_recover() = Some(dev.clone());
        self.metrics
            .set_loaded(dev.actual.clone(), engine.provider.clone());
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
            let device = cand.device.to_string();
            watch_slow_compile(
                &self.cfg.name,
                &device,
                &self.state,
                SlowCompile::DEFAULT,
                || rt.compile_tuned(cand, &self.cfg.load, &tuning),
            )
            .0?
        };
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
        let outs = match warm_up(&mut backend, chw, &extra, WARMUP_TIMEOUT) {
            Ok(outs) => outs,
            Err(e) => {
                if e.is::<WarmupTimeout>() {
                    // The inference is still running on a device that does not answer.
                    // Dropping the backend would wait for it (OpenVINO cancels and waits,
                    // ONNX Runtime waits `ABORT_TIMEOUT`): leak it instead, so this thread
                    // goes on to the next candidate (CPU last) and the model still serves.
                    error!(model = %self.cfg.name, "{e:#}; abandoning the device");
                    std::mem::forget(backend);
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
    fn finish(&self, eng: &mut Engine, slot: usize, f: InFlight) {
        let name = &self.cfg.name;
        let infer_d = f
            .done
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(f.submitted);
        let t = Instant::now();
        let live = f.meta.live.clone();
        // The profile for the device that actually loaded (`auto` and a fallback included),
        // else the model's / global value; a request's `min_confidence` wins over both.
        let (configured, source) = live.threshold_for(&eng.device);
        let threshold = effective_threshold(f.min_confidence(), configured);
        let threshold_source = if threshold == effective_threshold(0.0, configured) {
            source.label()
        } else {
            "request".to_string()
        };
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
        let post_threshold = match (&collect_job, near_margin) {
            (Some(_), Some(m)) => (threshold - m).max(crate::model::nms::MIN_CONFIDENCE_FLOOR),
            _ => threshold,
        };
        let params = PostParams {
            confidence_threshold: post_threshold,
            nms_iou: live.nms_iou,
        };
        let collected = if f.overdue {
            let watch = self.stuck_watch(eng);
            let limit = f.meta.timeout();
            collect_watched(&watch, limit, || eng.backend.collect(slot))
        } else {
            eng.backend.collect(slot)
        };
        if let Err(e) = &collected
            && session_dead(e)
        {
            self.stuck_watch(eng).mark(f.submitted.elapsed(), Some(e));
        }
        let mut degenerate = false;
        let scene = f.stats.scene();
        let crop = f.meta.verify_crop;
        let mut suppressed = SuppressCounts::default();
        let result = collected.and_then(|outs| {
            degenerate = outputs_degenerate(&outs);
            let dets = eng.family.postprocess(&outs, &f.ctx, &params);
            eng.backend.recycle(slot, outs);
            let dets = dets?;
            if crop {
                // A verifier check: the raw detections in the crop (see `Purpose::Verify`).
                return Ok(crate::model::to_predictions(
                    &dets,
                    &f.ctx,
                    &self.cfg.classes,
                    None,
                ));
            }
            let dets = match collect_job.as_mut() {
                Some(job) => {
                    let (raw, below): (Vec<_>, Vec<_>) =
                        dets.into_iter().partition(|d| d.score >= threshold);
                    let mut removed = Vec::new();
                    let kept = self.filter_detections(
                        raw.clone(),
                        &f,
                        scene,
                        &live,
                        &mut suppressed,
                        Some(&mut removed),
                    );
                    job.draft
                        .set_detections(&raw, &below, &removed, &self.cfg.classes);
                    kept
                }
                None => self.filter_detections(dets, &f, scene, &live, &mut suppressed, None),
            };
            let filter = (!live.object_filter.is_empty()).then_some(live.object_filter.as_slice());
            Ok(crate::model::to_predictions(
                &dets,
                &f.ctx,
                &self.cfg.classes,
                filter,
            ))
        });
        self.metrics.record_suppressed(&suppressed);
        self.note_health(eng, result.as_ref().err(), degenerate);
        let post_d = t.elapsed();
        let process_d = f.decode + f.preprocess + infer_d + post_d;
        let line = DoneLine {
            model: name.clone(),
            device: eng.device.clone(),
            request_id: f.meta.request_id.clone(),
            routed_from: f.meta.routed_from.clone(),
            verify_crop: crop,
            image: f.meta.image_name.clone(),
            size: format!("{}x{}", f.ctx.orig_w, f.ctx.orig_h),
            scene,
            luma: f64::from(f.stats.mean_luma),
            chroma: f64::from(f.stats.chroma_spread.min(f.stats.tint_spread)),
            suppressed: suppressed_summary(&suppressed),
            threshold,
            threshold_source,
            slot,
            queue_ms: millis_u64(f.submitted.saturating_duration_since(f.meta.enqueued)),
            decode_ms: f.decode.as_secs_f64() * 1e3,
            preprocess_ms: f.preprocess.as_secs_f64() * 1e3,
            inference_ms: infer_d.as_secs_f64() * 1e3,
            postprocess_ms: post_d.as_secs_f64() * 1e3,
            process: process_d,
            enqueued: f.meta.enqueued,
        };
        let InFlight {
            meta,
            img,
            ctx,
            stats,
            ..
        } = f;
        let resp = match result {
            Ok(predictions) => {
                self.metrics.inference_ms.record(millis_u64(infer_d));
                let resp = VisionDetectionResponse {
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
                };
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
                    });
                    self.hand_off(pending);
                    return;
                }
                self.metrics.process_ms.record(millis_u64(process_d));
                line.emit(&resp.predictions, None);
                if !crop && let (Some(dir), Some(img)) = (&live.save_image_path, &img) {
                    save_annotated(
                        name,
                        dir,
                        &meta.image_name,
                        img,
                        &resp.predictions,
                        live.save_ref_image,
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
        // The collector gets the frame after the reply went out (a `try_send`, never waits).
        let collect = collect_job.map(|job| (job, resp.predictions.clone()));
        self.reply(meta, resp);
        if let Some((job, predictions)) = collect {
            job.submit(&predictions);
        }
    }

    /// Give a request with verifier candidates to the verify lane (started on first use). If
    /// the lane cannot run, the request is answered here with its candidates unverified.
    fn hand_off(&self, pending: Box<Pending>) {
        let mut lane = self.verify_lane.borrow_mut();
        if lane.is_none() {
            match Lane::spawn(LaneCtx {
                name: self.cfg.name.clone(),
                metrics: self.metrics.clone(),
                peers: self.cfg.peers.clone(),
            }) {
                Ok(l) => *lane = Some(l),
                Err(e) => warn!(
                    model = %self.cfg.name,
                    "could not start the verifier thread ({e}); answering unverified"
                ),
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
            &LaneCtx {
                name: self.cfg.name.clone(),
                metrics: self.metrics.clone(),
                peers: self.cfg.peers.clone(),
            },
            pending,
            super::verify::Failure::Stopped,
        );
    }
}

/// Save `img` annotated with `predictions` under `dir` (a warning on failure).
pub(super) fn save_annotated(
    model: &str,
    dir: &std::path::Path,
    image_name: &str,
    img: &crate::image::RgbImage,
    predictions: &[crate::api::Prediction],
    save_ref_image: bool,
) {
    let file_name = if image_name.is_empty() || image_name == "image.jpg" {
        format!("{}.jpg", uuid::Uuid::new_v4())
    } else {
        image_name.to_string()
    };
    if let Err(e) = crate::image::save_annotated(dir, &file_name, img, predictions, save_ref_image)
    {
        warn!(model = %model, "could not save annotated image: {e:#}");
    }
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
/// must not be dropped (see `load_on`).
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
/// so a hung device yields [`WarmupTimeout`] instead of blocking in `collect`.
pub(crate) fn warm_up(
    backend: &mut Backend,
    chw: &[f32],
    extra: &[crate::model::ExtraInput],
    timeout: Duration,
) -> Result<(Vec<crate::model::NamedOutput>, u64)> {
    let started = Instant::now();
    let deadline = started + timeout;
    let device = backend.info().device.spec.clone();
    let wait = |backend: &Backend| -> Result<()> {
        let left = deadline.saturating_duration_since(Instant::now());
        match backend.completions().recv_timeout(left) {
            Ok(_) => Ok(()),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(WarmupTimeout {
                device: device.clone(),
                waited: started.elapsed(),
            }
            .into()),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("the inference runtime stopped during the warm-up")
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
        };
        let b = script.compile(&req, "openvino:cpu", false).into_backend();
        (Backend::Fake(b), script)
    }

    #[test]
    fn warm_up_covers_every_slot_and_leaves_no_completion_behind() {
        use crate::backend::fake::Call;
        let (mut backend, script) = fake_backend(3, false);
        let chw = vec![0.0f32; 3 * 64 * 64];
        let (outs, ms) = warm_up(&mut backend, &chw, &[], Duration::from_secs(5)).unwrap();
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
        let err = warm_up(&mut backend, &chw, &[], Duration::from_millis(100)).unwrap_err();
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(err.is::<WarmupTimeout>(), "{err:#}");
        let text = format!("{err:#}");
        assert!(text.contains("did not finish"), "{text}");
        assert!(text.contains("openvino:cpu"), "{text}");
        // Still in flight: the caller leaks the backend instead of waiting for it.
        assert!(backend.is_busy(0));
        std::mem::forget(backend);
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
}
