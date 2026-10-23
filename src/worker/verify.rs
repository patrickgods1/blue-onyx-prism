//! The verify lane: one small thread per worker that finishes the requests whose detections
//! go through the verifier cascade ([`crate::verify`]), so the inference thread never waits
//! for another model.
//!
//! `finish` builds the response as usual, plans the candidates and hands the request over as
//! a [`Pending`] (with the kept original frame). The lane looks the verifier up in the
//! generation's [`super::Peers`], crops the frame, sends each crop (or the frame) to the
//! verifier's worker as [`WorkItem::DetectDecoded`] with [`Purpose::Verify`] and a
//! [`Reply::Worker`] channel of its own, and waits on all of them with one `Select`, bounded by
//! the request's deadline (`enqueued + request timeout`, the budget the HTTP handler waits
//! for). Then it applies [`crate::verify::outcomes`], adds its time to `processMs`, counts
//! `verify_total{outcome}` and `detections_suppressed_total{reason="verify"}`, writes the
//! request's "detection done" line (with `verify` / `verify_ms`), saves the annotated image and
//! replies.
//!
//! Fail-open: a verifier that is not served, not ready, busy, stopping, late or failing leaves
//! the candidates unverified (kept), with a throttled warning. Nothing here blocks a worker:
//! sends are `try_send`, and a verifier check is never verified, forwarded or re-checked, so
//! two models that verify each other cannot wait on each other.
//!
//! Ordering: a verified request is answered when its verification ends, so it can overtake or
//! be overtaken by the model's other requests (each has its own client; the HTTP layer pairs
//! replies by channel, not by order). Requests without verifier candidates keep the FIFO
//! order of the worker. The in-flight cap counts a request until its handler gets the reply,
//! so verification time counts against it as it should.

use super::run::{DoneLine, VerifyLog, save_annotated};
use super::{Decoded, LiveSettings, ORDER_TARGET, Peers, Purpose, Reply, WorkItem, ms};
use crate::api::{Prediction, VisionDetectionResponse};
use crate::image::RgbImage;
use crate::image_stats::ImageStats;
use crate::metrics::ModelMetrics;
use crate::model::filters::{SuppressCounts, SuppressReason};
use crate::startup::ModelState;
use crate::verify::{Outcome, Plan, VerifyMode};
use crossbeam_channel::{Receiver, Select, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{trace, warn};

/// What the lane needs of its worker.
#[derive(Debug, Clone)]
pub(super) struct LaneCtx {
    pub(super) name: String,
    pub(super) metrics: Arc<ModelMetrics>,
    pub(super) peers: Peers,
}

/// A request whose candidates wait for the verifier.
#[derive(Debug)]
pub(super) struct Pending {
    pub(super) reply: Reply,
    /// The primary's answer with every candidate still in it.
    pub(super) resp: VisionDetectionResponse,
    /// The original frame (None: not kept, every candidate stays unverified).
    pub(super) img: Option<RgbImage>,
    pub(super) stats: ImageStats,
    pub(super) plan: Plan,
    /// The request's live settings (`verify` is Some).
    pub(super) live: Arc<LiveSettings>,
    pub(super) request_id: String,
    pub(super) image_name: String,
    pub(super) enqueued: Instant,
    /// The request's deadline; the verifier's answers are not awaited past it.
    pub(super) deadline: Instant,
    pub(super) handed_off: Instant,
    pub(super) line: DoneLine,
}

/// Why candidates stayed unverified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Failure {
    /// The verifier is not served (unknown name, disabled, removed).
    Unknown,
    /// It is compiling, lazy and not loaded yet, or failed.
    NotReady,
    /// Its queue is full.
    QueueFull,
    /// Its worker or this lane stopped (reload, shutdown).
    Stopped,
    /// The request's deadline passed before every answer arrived.
    Timeout,
    /// It answered with an error (bad crop, its own timeout, inference error).
    Error,
    /// The frame was not kept.
    NoFrame,
}

impl Failure {
    fn as_str(self) -> &'static str {
        match self {
            Failure::Unknown => "not served",
            Failure::NotReady => "not ready",
            Failure::QueueFull => "busy",
            Failure::Stopped => "stopping",
            Failure::Timeout => "timeout",
            Failure::Error => "error",
            Failure::NoFrame => "no frame",
        }
    }

    fn advice(self) -> &'static str {
        match self {
            Failure::Unknown | Failure::NotReady => {
                "check the verifier on the Models card (enabled, loaded) or clear verify"
            }
            Failure::QueueFull | Failure::Timeout => {
                "the verifier is too slow for the request timeout; use a smaller verifier, fewer \
                 max_crops, or a longer request timeout"
            }
            Failure::Stopped | Failure::Error | Failure::NoFrame => {
                "see the verifier's own log lines for this request id"
            }
        }
    }
}

/// The lane thread of one worker. Dropping it answers the verifications still waiting (their
/// candidates unverified) and joins the thread.
#[derive(Debug)]
pub(super) struct Lane {
    tx: Option<Sender<Box<Pending>>>,
    join: Option<JoinHandle<()>>,
}

impl Lane {
    pub(super) fn spawn(ctx: LaneCtx) -> std::io::Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<Box<Pending>>();
        let join = std::thread::Builder::new()
            .name(format!("verify-{}", ctx.name))
            .spawn(move || run(&ctx, &rx))?;
        Ok(Self {
            tx: Some(tx),
            join: Some(join),
        })
    }

    /// Queue `p` (never blocks); gives it back when the thread is gone.
    pub(super) fn submit(&self, p: Box<Pending>) -> Result<(), Box<Pending>> {
        match &self.tx {
            Some(tx) => tx.send(p).map_err(|e| e.0),
            None => Err(p),
        }
    }
}

impl Drop for Lane {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(j) = self.join.take()
            && j.join().is_err()
        {
            warn!("verify thread panicked");
        }
    }
}

/// A verification in progress.
struct Active {
    p: Box<Pending>,
    /// `(check index, answer channel)` of every request still unanswered.
    waits: Vec<(usize, Receiver<VisionDetectionResponse>)>,
    /// The verifier's predictions per check (mode `frame`: index 0).
    found: Vec<Option<Vec<Prediction>>>,
    failure: Option<Failure>,
    verifier: String,
}

impl Active {
    fn fail(&mut self, f: Failure) {
        self.failure.get_or_insert(f);
    }
}

/// What woke the lane.
enum Event {
    New(Result<Box<Pending>, crossbeam_channel::RecvError>),
    Answer {
        active: usize,
        check: usize,
        resp: Result<VisionDetectionResponse, crossbeam_channel::RecvError>,
    },
    Deadline,
}

fn run(ctx: &LaneCtx, rx: &Receiver<Box<Pending>>) {
    let mut active: Vec<Active> = Vec::new();
    let mut open = true;
    loop {
        if active.is_empty() {
            if !open {
                return;
            }
            match rx.recv() {
                Ok(p) => start(ctx, p, &mut active),
                Err(_) => return,
            }
        }
        while open {
            match rx.try_recv() {
                Ok(p) => start(ctx, p, &mut active),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => open = false,
            }
        }
        if !open {
            // The worker is exiting: answer now with what has arrived.
            for mut a in active.drain(..) {
                if !a.waits.is_empty() {
                    a.fail(Failure::Stopped);
                }
                finish(ctx, a);
            }
            return;
        }
        // Done or late ones.
        let now = Instant::now();
        let mut i = 0;
        while i < active.len() {
            let Some(a) = active.get_mut(i) else { break };
            if !a.waits.is_empty() && now >= a.p.deadline {
                a.fail(Failure::Timeout);
                a.waits.clear();
            }
            if a.waits.is_empty() {
                finish(ctx, active.swap_remove(i));
            } else {
                i += 1;
            }
        }
        if active.is_empty() {
            continue;
        }
        let Some(deadline) = active.iter().map(|a| a.p.deadline).min() else {
            continue;
        };
        let event = {
            let mut sel = Select::new();
            let i_new = sel.recv(rx);
            let mut refs: Vec<(usize, usize, usize, &Receiver<VisionDetectionResponse>)> =
                Vec::new();
            for (ai, a) in active.iter().enumerate() {
                for (check, r) in &a.waits {
                    refs.push((sel.recv(r), ai, *check, r));
                }
            }
            match sel.select_deadline(deadline) {
                Err(_) => Event::Deadline,
                Ok(op) => {
                    let i = op.index();
                    if i == i_new {
                        Event::New(op.recv(rx))
                    } else if let Some(&(_, ai, check, r)) = refs.iter().find(|x| x.0 == i) {
                        Event::Answer {
                            active: ai,
                            check,
                            resp: op.recv(r),
                        }
                    } else {
                        // Not reachable: every index was registered above. Complete the
                        // operation on the work channel so it is not dropped half-done.
                        Event::New(op.recv(rx))
                    }
                }
            }
        };
        match event {
            Event::New(Ok(p)) => start(ctx, p, &mut active),
            Event::New(Err(_)) => open = false,
            Event::Deadline => {}
            Event::Answer {
                active: ai,
                check,
                resp,
            } => {
                if let Some(a) = active.get_mut(ai) {
                    a.waits.retain(|(c, _)| *c != check);
                    match resp {
                        Ok(r) if r.success => {
                            if let Some(slot) = a.found.get_mut(check) {
                                *slot = Some(r.predictions);
                            }
                        }
                        Ok(r) => {
                            debug_verifier_error(ctx, &a.p.request_id, &r.message);
                            // The verifier dropped the check as late (its queue was too long).
                            a.fail(if r.message.starts_with("Request dropped") {
                                Failure::Timeout
                            } else {
                                Failure::Error
                            });
                        }
                        Err(_) => a.fail(Failure::Stopped),
                    }
                }
            }
        }
    }
}

fn debug_verifier_error(ctx: &LaneCtx, request_id: &str, message: &str) {
    tracing::debug!(
        model = %ctx.name,
        request_id = %request_id,
        "verifier answered with an error: {message}"
    );
}

/// Send `p`'s checks to its verifier; finished at once when nothing could be sent.
fn start(ctx: &LaneCtx, p: Box<Pending>, active: &mut Vec<Active>) {
    let Some(cfg) = p.live.verify.clone() else {
        // Not reachable (`finish` only hands over with `verify` set).
        complete(ctx, p, &[], None, String::new());
        return;
    };
    let checks = match cfg.mode {
        VerifyMode::Crop => p.plan.checks.len(),
        VerifyMode::Frame => 1,
    };
    let mut a = Active {
        found: vec![None; checks],
        waits: Vec::with_capacity(checks),
        failure: None,
        verifier: cfg.model.clone(),
        p,
    };
    match target(ctx, &cfg.model) {
        Err(f) => a.fail(f),
        Ok(h) => {
            a.verifier = h.name.clone();
            match a.p.img.as_ref() {
                None => a.fail(Failure::NoFrame),
                Some(img) => {
                    let save = a.p.live.save_image_path.is_some();
                    let images: Vec<(usize, RgbImage)> = match cfg.mode {
                        VerifyMode::Crop => {
                            a.p.plan
                                .checks
                                .iter()
                                .enumerate()
                                .filter_map(|(k, c)| Some((k, crate::verify::crop(img, c.crop?))))
                                .collect()
                        }
                        VerifyMode::Frame => Vec::new(),
                    };
                    let images = if cfg.mode == VerifyMode::Frame {
                        // The frame itself (moved unless it is still needed for saving).
                        let frame = if save {
                            img.clone()
                        } else {
                            match a.p.img.take() {
                                Some(f) => f,
                                None => RgbImage {
                                    width: 0,
                                    height: 0,
                                    rgb: Vec::new(),
                                },
                            }
                        };
                        vec![(0, frame)]
                    } else {
                        images
                    };
                    for (k, image) in images {
                        if h.is_full() {
                            a.fail(Failure::QueueFull);
                            break;
                        }
                        let (tx, rx) = crossbeam_channel::bounded(1);
                        let item = WorkItem::DetectDecoded(Box::new(Decoded {
                            img: image,
                            stats: a.p.stats,
                            image_name: a.p.image_name.clone(),
                            min_confidence: cfg.min_confidence,
                            request_id: a.p.request_id.clone(),
                            enqueued: a.p.enqueued,
                            deadline: a.p.deadline,
                            decode: Duration::ZERO,
                            reply: Reply::Worker(tx),
                            from: ctx.name.clone(),
                            purpose: Purpose::Verify,
                        }));
                        match h.sender.try_send(item) {
                            Ok(()) => a.waits.push((k, rx)),
                            Err(crossbeam_channel::TrySendError::Full(_)) => {
                                a.fail(Failure::QueueFull);
                                break;
                            }
                            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                                a.fail(Failure::Stopped);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    if a.waits.is_empty() {
        finish(ctx, a);
    } else {
        active.push(a);
    }
}

/// The served, ready verifier `name` (never this model), or why not (a lazy one is asked to
/// compile).
fn target(ctx: &LaneCtx, name: &str) -> Result<Arc<super::WorkerHandle>, Failure> {
    use crate::registry::normalize_name;
    let h = ctx
        .peers
        .get(name)
        .filter(|h| normalize_name(&h.name) != normalize_name(&ctx.name))
        .ok_or(Failure::Unknown)?;
    if h.state.get() != ModelState::Ready {
        h.trigger_load();
        return Err(Failure::NotReady);
    }
    if h.is_full() {
        return Err(Failure::QueueFull);
    }
    Ok(h)
}

fn finish(ctx: &LaneCtx, a: Active) {
    let Active {
        p,
        found,
        failure,
        verifier,
        ..
    } = a;
    complete(ctx, p, &found, failure, verifier);
}

/// Answer `p` with every candidate unverified (the lane could not take it).
pub(super) fn complete_unverified(ctx: &LaneCtx, p: Box<Pending>, failure: Failure) {
    let verifier = p
        .live
        .verify
        .as_ref()
        .map(|v| v.model.clone())
        .unwrap_or_default();
    complete(ctx, p, &[], Some(failure), verifier);
}

/// Apply the verifier's answers to `p` and reply.
fn complete(
    ctx: &LaneCtx,
    p: Box<Pending>,
    found: &[Option<Vec<Prediction>>],
    failure: Option<Failure>,
    verifier: String,
) {
    let Pending {
        reply,
        mut resp,
        img,
        plan,
        live,
        request_id,
        image_name,
        handed_off,
        line,
        ..
    } = *p;
    let outcomes = match &live.verify {
        Some(cfg) => crate::verify::outcomes(&plan, found, cfg),
        None => Vec::new(),
    };
    let (kept, counts) = crate::verify::apply(std::mem::take(&mut resp.predictions), &outcomes);
    ctx.metrics.record_verify(&counts);
    let mut suppressed = SuppressCounts::default();
    suppressed.add(SuppressReason::Verify, counts.rejected);
    ctx.metrics.record_suppressed(&suppressed);
    if let Some(f) = failure
        && counts.unverified > 0
        && let Some(held_back) = ctx.metrics.verify_log.allow(ctx.metrics.now_ms())
    {
        warn!(
            model = %ctx.name,
            verifier = %verifier,
            reason = f.as_str(),
            request_id = %request_id,
            unverified = counts.unverified,
            similar_since_last_warning = held_back,
            "candidates answered unverified: the verifier is {}; {}",
            f.as_str(),
            f.advice()
        );
    }
    let verify_d = handed_off.elapsed();
    resp.predictions = kept;
    resp.count = i32::try_from(resp.predictions.len()).unwrap_or(i32::MAX);
    resp.message = format!("Found {} objects", resp.predictions.len());
    resp.processMs = ms(line.process + verify_d);
    ctx.metrics
        .process_ms
        .record(super::millis_u64(line.process + verify_d));
    let outcome = {
        let parts: Vec<String> = Outcome::ALL
            .into_iter()
            .filter(|o| counts.get(*o) > 0)
            .map(|o| format!("{} {}", o.as_str(), counts.get(o)))
            .collect();
        let mut text = parts.join(", ");
        if let Some(f) = failure {
            text.push_str(&format!(" ({})", f.as_str()));
        }
        text
    };
    line.emit(
        &resp.predictions,
        Some(&VerifyLog {
            verifier,
            ms: verify_d.as_secs_f64() * 1e3,
            outcome,
        }),
    );
    if let (Some(dir), Some(img)) = (&live.save_image_path, &img) {
        save_annotated(
            &ctx.name,
            dir,
            &image_name,
            img,
            &resp.predictions,
            live.save_ref_image,
        );
    }
    trace!(
        target: ORDER_TARGET,
        model = %ctx.name,
        image = %image_name,
        success = resp.success,
        "replied"
    );
    if !reply.send(resp) {
        tracing::debug!(model = %ctx.name, "client went away before the response was sent");
    }
}
