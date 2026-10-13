//! The worker thread's run loop ([`WorkerCtx::run`]): load (compile + warm-up on each device
//! candidate), then a `Select`-driven state machine over the work channel, the prep reply of the
//! oldest request and the backend's completions. Invariants: requests are answered in arrival
//! order; nothing is left in flight when the loop ends (shutdown aborts, then answers every
//! queued request); postprocessing runs on this thread.

use super::pipeline::{Done, InFlight, Job, Meta, Pipeline, Stage};
use super::watch::{SlowCompile, StuckWatch, collect_watched, watch_slow_compile};
use super::{
    COMPILING_ON, LoadGate, ORDER_TARGET, POLL, WorkItem, WorkerConfig, auto_queue_size,
    effective_threshold, initializing_message, loaded_provider, millis_u64, ms, new_request_id,
};
use crate::api::VisionDetectionResponse;
use crate::backend::{Backend, Candidate, Completion, DeviceInfo, Runtimes};
use crate::metrics::{DropReason, ModelMetrics};
use crate::model::{Family, PostParams};
use crate::prep::{PrepJob, PrepPool, PrepSpec, Prepared};
use crate::profile::DeviceTuning;
use crate::startup::{ModelState, StateHandle};
use crate::sync::{LockExt, RwLockExt};
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Select, TryRecvError};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
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
}

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
        drop(parked);
        info!(model = %name, "worker exiting");
    }

    /// Shutdown: answer every request still in the pipeline or queued in the channel at once
    /// (instead of letting its HTTP handler wait for the request timeout).
    fn answer_remaining(&self, p: Pipeline, rx: &Receiver<WorkItem>) {
        let msg = format!("Model '{}' is shutting down", self.cfg.name);
        self.answer_all(p, &msg);
        while let Ok(item) = rx.try_recv() {
            if let WorkItem::Detect(_, reply, _) | WorkItem::DetectWithId(_, reply, _, _) = item {
                let _ = reply.send(self.error_response(msg.clone(), 0, 0));
            }
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
        let (req, reply, enqueued, request_id) = match item {
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
            WorkItem::Detect(req, reply, enqueued) => (req, reply, enqueued, new_request_id()),
            WorkItem::DetectWithId(req, reply, enqueued, id) => (req, reply, enqueued, id),
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
            let _ = reply.send(self.error_response(msg, 0, 0));
            return;
        };
        trace!(
            target: ORDER_TARGET,
            model = %name,
            image = %req.image_name,
            request_id = %request_id,
            "accepted"
        );
        let meta = Meta {
            reply,
            enqueued,
            image_name: req.image_name,
            min_confidence: req.min_confidence,
            live: self.cfg.live.load_full(),
            request_id,
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
            && f.submitted.elapsed() > f.meta.live.request_timeout
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
        let t = Instant::now();
        let outs = backend.infer(chw, &extra).context("warm-up inference")?;
        let warmup_ms = millis_u64(t.elapsed());
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
        let collected = if f.overdue {
            let watch = self.stuck_watch(eng);
            let limit = live.request_timeout;
            collect_watched(&watch, limit, || eng.backend.collect(slot))
        } else {
            eng.backend.collect(slot)
        };
        if let Err(e) = &collected
            && session_dead(e)
        {
            self.stuck_watch(eng).mark(f.submitted.elapsed(), Some(e));
        }
        let result = collected.and_then(|outs| {
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
        let round_trip = f.meta.enqueued.elapsed();
        let resp = match result {
            Ok(predictions) => {
                self.metrics.inference_ms.record(millis_u64(infer_d));
                self.metrics.process_ms.record(millis_u64(process_d));
                // The one log line per detection (structured fields only).
                debug!(
                    model = %name,
                    device = %eng.device,
                    request_id = %f.meta.request_id,
                    image = %f.meta.image_name,
                    size = %format!("{}x{}", f.ctx.orig_w, f.ctx.orig_h),
                    count = predictions.len(),
                    slot,
                    queue_ms = millis_u64(f.submitted.saturating_duration_since(f.meta.enqueued)),
                    decode_ms = f.decode.as_secs_f64() * 1e3,
                    preprocess_ms = f.preprocess.as_secs_f64() * 1e3,
                    inference_ms = infer_d.as_secs_f64() * 1e3,
                    postprocess_ms = post_d.as_secs_f64() * 1e3,
                    process_ms = process_d.as_secs_f64() * 1e3,
                    round_trip_ms = round_trip.as_secs_f64() * 1e3,
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
                    count: i32::try_from(predictions.len()).unwrap_or(i32::MAX),
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
                warn!(
                    model = %name,
                    device = %eng.device,
                    request_id = %f.meta.request_id,
                    image = %f.meta.image_name,
                    inference_ms = infer_d.as_secs_f64() * 1e3,
                    process_ms = process_d.as_secs_f64() * 1e3,
                    round_trip_ms = round_trip.as_secs_f64() * 1e3,
                    "detection failed: {e:#}"
                );
                let mut r = self.error_response(format!("{e:#}"), ms(infer_d), ms(process_d));
                r.executionProvider = eng.provider.clone();
                r
            }
        };
        self.reply(f.meta, resp);
    }
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
    fn load_failure_says_what_to_do() {
        let m = load_failure_message(&anyhow::anyhow!("every device failed"));
        assert!(m.starts_with("every device failed; "), "{m}");
        assert!(m.contains("check the log"), "{m}");
    }
}
