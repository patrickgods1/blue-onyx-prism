//! The burst test (`benchmark.burst`, CLI `--burst [C,K,I]`, the Run page's Options): the latency
//! of a model x device under the load a Blue Iris produces when several cameras trigger together,
//! measured on the serving code. Off by default: it adds 10 s to a minute per combo (more when the
//! queue backs up) and loads the device like a busy Blue Iris.
//!
//! **When.** After a combo's timed run ([`Bench::run`] calls [`Burst::run`] once the run's own
//! compiled copy is dropped: the benchmark never holds a second copy beside the burst's, so at most
//! two copies exist at once, the serving worker's and the burst's).
//!
//! **The worker.** A temporary worker of the serving code ([`crate::worker::spawn_worker`]): the
//! same load (exactly the device the timed run loaded on, no fallback: [`WorkerConfig::only`]),
//! the same compile tuning and in-flight slots ([`crate::worker::device_tuning`], from the job's
//! concurrency, the one rule the timed run compiled with), its own decode + preprocess pool sized
//! by the registry's rule ([`crate::registry::preprocess_threads`] for that device), the config's
//! `worker_queue_size` (0 = auto-sized from its warm-up) and `request_timeout_secs`, and the
//! model's live settings: as served for an As served run ([`LiveSettings::for_model`]), the bare
//! model otherwise (no enhancement, filter rules, night threshold or static-box memory). Either
//! way the threshold is the one the timed run used on that device, nothing is saved or collected,
//! and the night model and the verifier stay out (they are other workers; the result notes it).
//! It is never registered: Blue Iris never reaches it, its metrics count in no live total and the
//! registry's workers keep serving beside it.
//!
//! **The load (open loop).** [`BURSTS`] bursts; in each, every one of C cameras triggers at the
//! burst's start t0 and sends K images I ms apart: camera c's k-th request is due at
//! `t0 + k * I` ([`schedule`]), whatever happened to the earlier ones. A burst starts K * I +
//! [`SETTLE`] after the one before, and at least [`SETTLE`] after its last answer (each burst
//! meets an idle worker, as triggers minutes apart do). Images go round-robin over at most
//! [`MAX_FRAMES`] frames of the combo's sets, Your cameras first ([`frame_pool`]), sent as their
//! encoded bytes through the path a detection request takes once its body is read
//! ([`crate::server::dispatch`]: the queue-full and timeout rules included).
//!
//! **What is recorded** ([`BurstResult`], [`summarize`]): a request's latency is its answer time
//! minus its *intended* send time, so a request the driver sent late still counts from when Blue
//! Iris would have sent it (no coordinated omission); p50 / p95 / p99 over the answered requests.
//! A trigger (camera x burst) lasts from its first intended send to its last answer (p50 / p95
//! over the triggers whose every request was answered; the others are counted). Each request is
//! classified from its own answer: answered, refused by the reason its answer carries
//! ([`crate::api::VisionDetectionResponse::dropped`]: `queue_full`, `timeout`, `worker_gone`,
//! `failed`, `initializing`) or [`ERROR_REASON`] (a failed answer without one), so answered +
//! refused == requests (the worker's shared drop counters would count a request the server timed
//! out and the worker dropped later twice). Also the most requests waiting or in progress at once
//! and, as for the timed phase, the live requests the service answered meanwhile and the GPU
//! video-decode load (Windows).
//!
//! **Teardown.** Always, also on cancel, on a load timeout and on errors: the worker's token is
//! cancelled (in-flight inferences are aborted, queued requests answered), then its thread is
//! waited for at most [`Burst::stop_wait`] ([`stop_wait`]: the request timeout plus
//! [`STOP_GRACE`]; [`WorkerHandle::join_within`]): its backend drops on that thread, freeing the
//! device and its VRAM holders. A thread that has not ended by then is left behind, never joined
//! (a compile cannot be interrupted, and an inference blocked in the runtime after a GPU reset, or
//! the shutdown's abort waiting on it, may never return; a stuck inference was handed to a reaper
//! by the stuck watchdog): a worker still loading finishes in the background and drops its copy
//! when it ends, and the result's notes say which. Then the prep pool's threads go. A cancel is
//! checked every [`TICK`], during the load too; a worker that cannot load (or not within
//! [`Burst::load_timeout`]) and a timer that cannot start are the result's `error`, and the run
//! goes on.

use super::images::ImageSet;
use super::run::{Bench, Cancelled, Phase, ms};
use super::{Job, RunResult, Stats};
use crate::api::VisionDetectionRequest;
use crate::backend::spec::{self, DeviceSpec};
use crate::backend::{Candidate, LoadRequest, Runtimes};
use crate::config::{BurstConfig, Config, ModelConfig};
use crate::metrics::{DropReason, ModelMetrics};
use crate::startup::ModelState;
use crate::worker::{Joined, LiveSettings, WorkerConfig, WorkerHandle};
use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Bursts per combo.
pub const BURSTS: usize = 3;
/// Added to K x I between burst starts, and the least time between a burst's last answer and
/// the next burst.
pub const SETTLE: Duration = Duration::from_secs(2);
/// Distinct frames a burst sends at most (round-robin over them).
pub const MAX_FRAMES: usize = 64;
/// Encoded bytes of those frames at most (4K JPEGs are 1-3 MB).
pub const MAX_FRAME_BYTES: usize = 96 * 1024 * 1024;
/// Longest the burst waits for its worker to compile and warm up.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(600);
/// How often a waiting burst looks at the cancel flag (and, during the load, at the worker's
/// state).
pub const TICK: Duration = Duration::from_millis(25);
/// Added to the request timeout for the worker's thread to end after its token is cancelled.
pub const STOP_GRACE: Duration = Duration::from_secs(5);

/// How long teardown waits for the worker's thread: the request timeout (an inference in
/// progress when the token is cancelled is aborted, or ends within it) plus [`STOP_GRACE`].
pub fn stop_wait(config: &Config) -> Duration {
    config.request_timeout() + STOP_GRACE
}

/// The refusal reason of a failed answer no drop counter took.
pub const ERROR_REASON: &str = "error";

/// What a burst test measured on one combo ([`RunResult::burst`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BurstResult {
    /// Cameras, images per camera and their interval.
    pub cameras: usize,
    pub images: usize,
    pub interval_ms: u64,
    pub bursts: usize,
    /// The worker ran the model's serving settings (an As served run); false: the bare model.
    #[serde(default)]
    pub served: bool,
    /// Requests sent (cameras x images x bursts, fewer when a load error ended it).
    pub requests: usize,
    /// Requests answered with detections.
    pub answered: usize,
    /// Per answered request: answer time minus intended send time (ms).
    pub latency_ms: Stats,
    /// Per trigger (camera x burst) whose every request was answered: from its first intended
    /// send to its last answer (ms).
    pub trigger_ms: Stats,
    pub triggers: usize,
    /// Triggers with a refused request (not in `trigger_ms`).
    #[serde(default)]
    pub triggers_incomplete: usize,
    /// Refused requests by the reason their answer carries (`queue_full`, `timeout`,
    /// `worker_gone`, `failed`, `initializing`), and `error` for failed answers without one:
    /// `answered` + their sum == `requests`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub refused: BTreeMap<String, u64>,
    /// The first failed answer's text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_refusal: Option<String>,
    /// The most requests sent and not answered at once (queued or being processed).
    pub max_queue_depth: usize,
    /// The worker's queue capacity (soft, after its warm-up) and inferences in flight.
    #[serde(default)]
    pub queue_capacity: usize,
    #[serde(default)]
    pub inflight: usize,
    /// Its decode + preprocess threads.
    #[serde(default)]
    pub prep_threads: usize,
    /// Distinct frames sent (round-robin) and the sets they came from.
    #[serde(default)]
    pub frames: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frame_sets: Vec<String>,
    /// The latest a request was sent after its intended time (ms): the driver's own lag, which
    /// the latencies include.
    #[serde(default)]
    pub max_send_lag_ms: f64,
    /// The worker's compile + warm-up (ms).
    #[serde(default)]
    pub load_ms: f64,
    /// From the first burst's start to the last answer (ms).
    #[serde(default)]
    pub wall_ms: f64,
    /// Detection requests the service answered meanwhile (None outside the service).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_requests_per_min: Option<f64>,
    /// The GPU video-decode engine's load meanwhile (Windows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_decode_pct: Option<crate::backend::video_decode::VideoDecode>,
    /// Why no numbers: the worker did not load, no frame could be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// What ran otherwise than Blue Iris would get it (night model and verifier left out, a
    /// variant's rules not applied, a stuck worker left behind).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl BurstResult {
    fn empty(s: &BurstConfig, bursts: usize, served: bool) -> Self {
        Self {
            cameras: s.cameras,
            images: s.images,
            interval_ms: s.interval_ms,
            bursts,
            served,
            ..Self::default()
        }
    }

    /// Refused requests in total ([`Self::errors`] included).
    pub fn refused_total(&self) -> u64 {
        self.refused.values().sum()
    }

    /// Failed answers that carried no refusal reason ([`ERROR_REASON`]).
    pub fn errors(&self) -> u64 {
        self.refused.get(ERROR_REASON).copied().unwrap_or(0)
    }

    /// "8 x 4 x 500 ms".
    pub fn load_text(&self) -> String {
        format!(
            "{} x {} x {} ms",
            self.cameras, self.images, self.interval_ms
        )
    }

    /// One line: "burst 8 x 4 x 500 ms: p50 120 ms, p95 480 ms, p99 610 ms; last answer p50 0.9 s,
    /// p95 1.6 s; 96 requests, 2 refused (queue_full 2); queue up to 12" (or the error).
    pub fn describe(&self) -> String {
        if let Some(e) = &self.error {
            return format!("burst {}: not measured ({e})", self.load_text());
        }
        let mut s = format!(
            "burst {}: request p50 {}, p95 {}, p99 {}; last answer per trigger p50 {}, p95 {}; \
             {} requests",
            self.load_text(),
            ms_text(self.latency_ms.p50),
            ms_text(self.latency_ms.p95),
            ms_text(self.latency_ms.p99),
            ms_text(self.trigger_ms.p50),
            ms_text(self.trigger_ms.p95),
            self.requests
        );
        let refused = self.refused_total();
        if refused > 0 {
            let by: Vec<String> = self
                .refused
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect();
            s.push_str(&format!(", {refused} refused ({})", by.join(", ")));
        }
        s.push_str(&format!(
            "; queue up to {} (capacity {}, {} in flight)",
            self.max_queue_depth, self.queue_capacity, self.inflight
        ));
        s
    }
}

/// "420 ms", "1.6 s".
pub fn ms_text(v: f64) -> String {
    if v >= 1000.0 {
        format!("{:.1} s", v / 1000.0)
    } else {
        format!("{v:.0} ms")
    }
}

/// One request of a burst: camera `camera`'s `image`-th, due `at` after the burst's start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Due {
    pub camera: usize,
    pub image: usize,
    pub at: Duration,
}

/// The requests of one burst in send order: every camera's k-th image at `k * interval`.
pub fn schedule(s: &BurstConfig) -> Vec<Due> {
    let step = Duration::from_millis(s.interval_ms);
    let mut out = Vec::with_capacity(s.cameras * s.images);
    for image in 0..s.images {
        for camera in 0..s.cameras {
            out.push(Due {
                camera,
                image,
                at: step * u32::try_from(image).unwrap_or(u32::MAX),
            });
        }
    }
    out
}

/// What happened to one request: its burst, its due slot, and when (from its burst's start) it
/// was sent and answered.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub burst: usize,
    pub due: Due,
    pub sent: Duration,
    pub answered: Duration,
    pub ok: bool,
    /// Why a failed answer was refused, as the answer says
    /// ([`crate::api::VisionDetectionResponse::dropped`]); None: answered, or an error.
    pub dropped: Option<DropReason>,
    /// The failed answer's text.
    pub error: Option<String>,
}

/// The numbers of [`BurstResult`] from the requests' outcomes: latency from the intended send
/// time, trigger times, refusals by each failed answer's own reason ([`Outcome::dropped`];
/// [`ERROR_REASON`] without one).
pub fn summarize(out: &mut BurstResult, outcomes: &[Outcome]) {
    let ms_of = |d: Duration| ms(d);
    out.requests = outcomes.len();
    let ok: Vec<&Outcome> = outcomes.iter().filter(|o| o.ok).collect();
    out.answered = ok.len();
    let latencies: Vec<f64> = ok
        .iter()
        .map(|o| ms_of(o.answered.saturating_sub(o.due.at)))
        .collect();
    out.latency_ms = Stats::from_samples(&latencies);
    out.max_send_lag_ms = outcomes
        .iter()
        .map(|o| ms_of(o.sent.saturating_sub(o.due.at)))
        .fold(0.0, f64::max);
    // Triggers: (burst, camera).
    let mut triggers: BTreeMap<(usize, usize), (Duration, Duration, bool)> = BTreeMap::new();
    for o in outcomes {
        let e = triggers
            .entry((o.burst, o.due.camera))
            .or_insert((o.due.at, o.answered, true));
        e.0 = e.0.min(o.due.at);
        e.1 = e.1.max(o.answered);
        e.2 &= o.ok;
    }
    out.triggers = triggers.len();
    let complete: Vec<f64> = triggers
        .values()
        .filter(|t| t.2)
        .map(|t| ms_of(t.1.saturating_sub(t.0)))
        .collect();
    out.triggers_incomplete = out.triggers - complete.len();
    out.trigger_ms = Stats::from_samples(&complete);
    let mut refused: BTreeMap<String, u64> = BTreeMap::new();
    for o in outcomes.iter().filter(|o| !o.ok) {
        let reason = o.dropped.map_or(ERROR_REASON, DropReason::as_str);
        *refused.entry(reason.to_string()).or_default() += 1;
    }
    out.refused = refused;
    out.first_refusal = outcomes.iter().find_map(|o| o.error.clone());
}

/// A frame a burst sends: a name and its encoded bytes.
#[derive(Debug, Clone)]
pub struct Frame {
    pub name: String,
    pub bytes: bytes::Bytes,
    pub set: String,
}

/// The frames of `sets` a burst sends: Your cameras' first (the uploads Blue Iris sent), then the
/// other sets', each group spread evenly over its images, at most [`MAX_FRAMES`] frames and
/// [`MAX_FRAME_BYTES`]. A set with a transform (a NIR proxy) sends its frames as transformed
/// JPEGs (what a camera in that mode sends). Frames that cannot be read are skipped (counted in
/// the returned skip count).
pub fn frame_pool(sets: &[ImageSet]) -> (Vec<Frame>, usize) {
    let is_cams = |s: &&ImageSet| s.id == super::cameras::ID;
    let groups: [Vec<&ImageSet>; 2] = [
        sets.iter().filter(is_cams).collect(),
        sets.iter().filter(|s| !is_cams(s)).collect(),
    ];
    let mut out: Vec<Frame> = Vec::new();
    let mut bytes = 0usize;
    let mut skipped = 0usize;
    for group in groups {
        let all: Vec<(&ImageSet, &super::images::SetImage)> = group
            .iter()
            .flat_map(|s| s.images.iter().map(move |i| (*s, i)))
            .collect();
        let room = MAX_FRAMES.saturating_sub(out.len());
        for (set, img) in spread(&all, room) {
            if out.len() >= MAX_FRAMES || bytes >= MAX_FRAME_BYTES {
                break;
            }
            match encoded(set, img) {
                Ok(b) if bytes + b.len() <= MAX_FRAME_BYTES => {
                    bytes += b.len();
                    out.push(Frame {
                        name: img.file.clone(),
                        bytes: b,
                        set: set.id.clone(),
                    });
                }
                Ok(_) => break,
                Err(e) => {
                    skipped += 1;
                    tracing::debug!("burst: frame {} skipped: {e:#}", img.file);
                }
            }
        }
    }
    (out, skipped)
}

/// `n` items of `v` evenly spread (all of them when it has no more).
fn spread<T: Copy>(v: &[T], n: usize) -> Vec<T> {
    if v.len() <= n {
        return v.to_vec();
    }
    (0..n)
        .filter_map(|i| v.get(i * v.len() / n).copied())
        .collect()
}

/// The bytes a camera would upload for `img` of `set`.
fn encoded(set: &ImageSet, img: &super::images::SetImage) -> Result<bytes::Bytes> {
    let raw = img.bytes()?;
    match set.transform {
        None => Ok(bytes::Bytes::from(raw.into_owned())),
        Some(t) => {
            let mut decoded = crate::image::decode(&raw)?;
            t.apply(&mut decoded);
            Ok(bytes::Bytes::from(crate::image::encode_jpeg(&decoded, 90)?))
        }
    }
}

/// The burst test of a run: its load and what its temporary workers need.
pub struct Burst<'a> {
    pub settings: BurstConfig,
    /// The runtimes the run compiles with (the registry's in the service: one OpenVINO `Core`,
    /// the ONNX Runtime library the process loaded).
    pub runtimes: &'a Arc<Mutex<Runtimes>>,
    /// The config the models' entries, live settings, queue size and timeout come from.
    pub config: &'a Config,
    /// [`BURSTS`]; tests run fewer.
    pub bursts: usize,
    /// [`SETTLE`]; tests wait less.
    pub settle: Duration,
    /// [`LOAD_TIMEOUT`].
    pub load_timeout: Duration,
    /// How long teardown waits for the worker's thread to end ([`stop_wait`]).
    pub stop_wait: Duration,
    /// What a test makes go wrong in the burst's worker.
    #[cfg(test)]
    pub(crate) faults: crate::worker::TestFaults,
    /// Test only: the burst's timer runtime fails to start.
    #[cfg(test)]
    pub(crate) runtime_fault: bool,
}

impl std::fmt::Debug for Burst<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Burst")
            .field("settings", &self.settings)
            .field("bursts", &self.bursts)
            .field("settle", &self.settle)
            .finish_non_exhaustive()
    }
}

impl<'a> Burst<'a> {
    /// The burst test with `settings` over `runtimes` under `config`.
    pub fn new(
        settings: BurstConfig,
        runtimes: &'a Arc<Mutex<Runtimes>>,
        config: &'a Config,
    ) -> Self {
        Self {
            settings,
            runtimes,
            config,
            bursts: BURSTS,
            settle: SETTLE,
            load_timeout: LOAD_TIMEOUT,
            stop_wait: stop_wait(config),
            #[cfg(test)]
            faults: Default::default(),
            #[cfg(test)]
            runtime_fault: false,
        }
    }

    /// Run the burst test of `job` on the device its timed run `run` loaded on. Errors only with
    /// [`Cancelled`]; anything else is the result's `error`.
    pub fn run(&self, bench: &Bench, job: &Job, run: &RunResult) -> Result<BurstResult> {
        let served = job.variant.as_ref().is_some_and(|v| v.served.is_some());
        let mut out = BurstResult::empty(&self.settings, self.bursts, served);
        let device_text = if run.spec.is_empty() {
            run.requested_device.clone()
        } else {
            run.spec.clone()
        };
        let device = match spec::parse(&device_text) {
            Ok(DeviceSpec::Device(d)) => d,
            _ => {
                out.error = Some(format!("{device_text} is not one device"));
                return Ok(out);
            }
        };
        let total = self.bursts * self.settings.cameras * self.settings.images;
        bench.report(job, &device_text, Phase::Burst, 0, total, "");
        let (frames, skipped) = frame_pool(bench.sets);
        if frames.is_empty() {
            out.error = Some(format!(
                "no frame of {} could be read ({skipped} skipped)",
                bench.describe_sets()
            ));
            return Ok(out);
        }
        if skipped > 0 {
            out.notes
                .push(format!("{skipped} frame(s) could not be read"));
        }
        // The same compile tuning as the timed run: its in-flight requests.
        out.inflight = run.tuning.as_ref().map_or(1, |t| t.inflight);
        out.frames = frames.len();
        for f in &frames {
            if !out.frame_sets.contains(&f.set) {
                out.frame_sets.push(f.set.clone());
            }
        }
        let (cfg, prep_threads, notes) = self.worker_config(job, run, device, served);
        out.notes.extend(notes);
        out.prep_threads = prep_threads;
        let metrics = Arc::new(ModelMetrics::new(cfg.name.clone(), &device_text, ""));
        let token = tokio_util::sync::CancellationToken::new();
        let prep = cfg.prep.clone();
        let t_load = Instant::now();
        let handle =
            crate::worker::spawn_worker(self.runtimes.clone(), cfg, metrics, token.clone());
        let worker = Temporary {
            handle: Some(handle),
            token,
            prep,
            stop_wait: self.stop_wait,
        };
        let outcome =
            self.load_and_drive(bench, job, &device_text, &worker, &frames, &mut out, t_load);
        let loading = worker
            .handle
            .as_ref()
            .is_some_and(|h| h.state.get() == ModelState::Initializing);
        match worker.teardown() {
            Joined::Ended => {}
            Joined::Stuck => out.notes.push(
                "the burst's worker had a stuck inference and was left behind to end on its own \
                 (its backend was handed to a reaper)"
                    .into(),
            ),
            Joined::Running if loading => {
                tracing::warn!(
                    "burst of {} on {device_text}: its worker was still loading; left to finish \
                     in the background",
                    job.name
                );
                out.notes.push(
                    "the burst's worker was still loading when the burst ended; it was left to \
                     finish in the background and drops its copy of the model when it ends (the \
                     next combo may run beside that compile)"
                        .into(),
                );
            }
            Joined::Running => {
                tracing::warn!(
                    "burst of {} on {device_text}: its worker did not end within {:.0} s of its \
                     stop; its thread was left behind",
                    job.name,
                    self.stop_wait.as_secs_f64()
                );
                out.notes.push(format!(
                    "the burst's worker did not end within {:.0} s of its stop (an inference \
                     blocked in the runtime, as after a GPU reset?); its thread was left behind \
                     and frees the device when it returns",
                    self.stop_wait.as_secs_f64()
                ));
            }
        }
        if let Some(outcomes) = outcome? {
            summarize(&mut out, &outcomes);
        }
        Ok(out)
    }

    /// Wait for the worker to load, then drive the bursts; Ok(None) when it did not load (the
    /// reason in `out.error`).
    #[allow(
        clippy::too_many_arguments,
        reason = "the burst's pieces, passed once from `run`"
    )]
    fn load_and_drive(
        &self,
        bench: &Bench,
        job: &Job,
        device: &str,
        worker: &Temporary,
        frames: &[Frame],
        out: &mut BurstResult,
        t_load: Instant,
    ) -> Result<Option<Vec<Outcome>>> {
        let Some(handle) = worker.handle.as_ref() else {
            return Ok(None);
        };
        loop {
            match handle.state.get() {
                ModelState::Ready => break,
                ModelState::Failed(m) => {
                    out.error = Some(format!("its worker did not load on {device}: {m}"));
                    return Ok(None);
                }
                ModelState::Initializing => {
                    if is_set(bench.cancel) {
                        return Err(Cancelled.into());
                    }
                    if t_load.elapsed() > self.load_timeout {
                        out.error = Some(format!(
                            "its worker did not load on {device} within {} s",
                            self.load_timeout.as_secs()
                        ));
                        return Ok(None);
                    }
                    std::thread::sleep(TICK);
                }
            }
        }
        out.load_ms = ms(t_load.elapsed());
        out.queue_capacity = handle.queue_capacity();
        let live_before = bench.live_requests.map(|f| f());
        let decode_load = if bench.video_decode {
            crate::backend::video_decode::Sampler::start()
        } else {
            None
        };
        let total = self.bursts * self.settings.cameras * self.settings.images;
        let done = AtomicUsize::new(0);
        let progress = || {
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            bench.report(job, device, Phase::Burst, n, total, "");
        };
        let rt = match self.timer_runtime() {
            Ok(rt) => rt,
            Err(e) => {
                out.error = Some(format!("the burst's timer runtime did not start: {e}"));
                return Ok(None);
            }
        };
        let started = Instant::now();
        let driven = rt.block_on(drive(
            handle,
            frames,
            &self.settings,
            self.bursts,
            self.settle,
            bench.cancel,
            &progress,
        ));
        let wall = started.elapsed();
        out.video_decode_pct = decode_load.and_then(|s| s.finish());
        out.live_requests = live_before
            .zip(bench.live_requests.map(|f| f()))
            .map(|(a, b)| b.saturating_sub(a));
        out.live_requests_per_min = out.live_requests.map(|n| super::run::per_minute(n, wall));
        let (outcomes, max_depth, wall_ms) = driven?;
        out.max_queue_depth = max_depth;
        out.wall_ms = wall_ms;
        Ok(Some(outcomes))
    }

    /// The current-thread runtime that times the burst's sends.
    fn timer_runtime(&self) -> std::io::Result<tokio::runtime::Runtime> {
        #[cfg(test)]
        if self.runtime_fault {
            return Err(std::io::Error::other("no timer (test)"));
        }
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
    }

    /// The temporary worker's config for `job` on `device`, its prep thread count and notes.
    fn worker_config(
        &self,
        job: &Job,
        run: &RunResult,
        device: spec::Device,
        served: bool,
    ) -> (WorkerConfig, usize, Vec<String>) {
        let config = self.config;
        let mut notes = Vec::new();
        let base = super::variant::base_name(&job.name);
        let mut model = super::find_model(config, base)
            .cloned()
            .unwrap_or_else(|| ModelConfig {
                path: job.path.clone(),
                ..ModelConfig::default()
            });
        model.path = job.path.clone();
        model.family = job.family;
        model.lazy = false;
        let mut live = LiveSettings::for_model(config, &model);
        if live.night_model.is_some() || live.verify.is_some() {
            notes.push(
                "the night model and the verifier are other workers: the burst runs the model's \
                 own worker without them"
                    .into(),
            );
        }
        if !served {
            if job.variant.is_some() {
                notes.push(format!(
                    "{}: the burst runs the bare model, without the variant's rules",
                    job.name
                ));
            }
            live.filters = Default::default();
            live.static_suppression = crate::static_memory::StaticSuppression::default();
            live.preprocess = Default::default();
            live.night_threshold = None;
        }
        live.night_model = None;
        live.verify = None;
        live.save_image_path = None;
        live.save_ref_image = false;
        live.collect = Arc::new(crate::collect::CollectSettings {
            enabled: false,
            ..crate::collect::CollectSettings::default()
        });
        live.object_filter = bench_filter(config, job);
        if let Some(t) = run.served_threshold {
            live.confidence_threshold = t;
            live.device_thresholds = Arc::new(BTreeMap::new());
        }
        let profile = job
            .concurrency
            .as_ref()
            .map(|c| c.profile.clone())
            .unwrap_or_else(|| config.profile());
        let prep_threads = crate::registry::preprocess_threads(config, &profile, &[Some(device)]);
        let cfg = WorkerConfig {
            name: format!("burst:{}", job.name),
            model,
            device: DeviceSpec::Device(device),
            load: LoadRequest {
                path: job.path.clone(),
                requested: device.to_string(),
                gpu_precision: job.gpu_precision.clone(),
                no_cache: false,
            },
            classes: job.classes.clone(),
            live: live.shared(),
            queue_size: config.worker_queue_size,
            lazy: false,
            can_use_gpu: false,
            wait: None,
            concurrency: job.concurrency.clone(),
            prep: Some(crate::prep::PrepPool::new(prep_threads)),
            peers: crate::worker::Peers::new(),
            only: Some(Candidate {
                device,
                fell_back: false,
                note: None,
            }),
            make_room: None,
            recovering_from: None,
            #[cfg(test)]
            faults: self.faults.clone(),
        };
        (cfg, prep_threads, notes)
    }
}

/// The object filter the timed run's production predictions used (`Bench::object_filter_for`
/// with the config's global filter).
fn bench_filter(config: &Config, job: &Job) -> Vec<String> {
    match &job.object_filter {
        Some(f) => f.clone(),
        None => config.object_filter.clone(),
    }
}

fn is_set(flag: Option<&AtomicBool>) -> bool {
    flag.is_some_and(|c| c.load(Ordering::Relaxed))
}

/// The temporary worker and what goes with it; [`Temporary::teardown`] (or drop) ends it.
struct Temporary {
    handle: Option<WorkerHandle>,
    token: tokio_util::sync::CancellationToken,
    prep: Option<Arc<crate::prep::PrepPool>>,
    /// [`Burst::stop_wait`].
    stop_wait: Duration,
}

impl Temporary {
    /// Cancel the worker, wait at most `stop_wait` for its thread (its backend drops there;
    /// a stuck or still running thread is left behind, see the module docs), then drop the prep
    /// pool (its threads are joined once the worker's thread no longer holds it).
    fn teardown(mut self) -> Joined {
        self.end()
    }

    fn end(&mut self) -> Joined {
        self.token.cancel();
        let joined = match self.handle.take() {
            Some(h) => h.join_within(Instant::now() + self.stop_wait),
            None => Joined::Ended,
        };
        drop(self.prep.take());
        joined
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        self.end();
    }
}

/// Drive `bursts` bursts of `s` on `handle` (open loop, see the module docs): the outcomes, the
/// most requests outstanding at once and the wall time (ms). Err([`Cancelled`]) when `cancel` is
/// set (checked every [`TICK`]); the requests still waiting are dropped with it.
async fn drive(
    handle: &WorkerHandle,
    frames: &[Frame],
    s: &BurstConfig,
    bursts: usize,
    settle: Duration,
    cancel: Option<&AtomicBool>,
    on_answer: &(dyn Fn() + Sync),
) -> Result<(Vec<Outcome>, usize, f64)> {
    let plan = schedule(s);
    let span = Duration::from_millis(s.interval_ms) * u32::try_from(s.images).unwrap_or(u32::MAX);
    let outstanding = AtomicUsize::new(0);
    let max_depth = AtomicUsize::new(0);
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(plan.len() * bursts);
    let first = Instant::now();
    let mut start = first;
    let mut last_answer = first;
    let mut sent_total = 0usize;
    for burst in 0..bursts {
        if burst > 0 {
            let next = (start + span + settle).max(last_answer + settle);
            wait_until(next, cancel).await?;
            start = next;
        }
        let t0 = start;
        let mut pending = futures_util::stream::FuturesUnordered::new();
        for due in &plan {
            let k = sent_total;
            sent_total += 1;
            let Some(frame) = frames.get(k % frames.len().max(1)) else {
                continue;
            };
            let (outstanding, max_depth) = (&outstanding, &max_depth);
            let due = *due;
            pending.push(async move {
                tokio::time::sleep_until(tokio::time::Instant::from_std(t0 + due.at)).await;
                let sent = Instant::now();
                let now = outstanding.fetch_add(1, Ordering::SeqCst) + 1;
                max_depth.fetch_max(now, Ordering::SeqCst);
                let req = VisionDetectionRequest {
                    min_confidence: 0.0,
                    image_data: frame.bytes.clone(),
                    image_name: format!("burst-cam{}-{}", due.camera, frame.name),
                    no_collect: true,
                };
                let id = crate::worker::new_request_id();
                let resp = crate::server::dispatch(handle, Ok(req), sent, &id).await;
                let answered = Instant::now();
                outstanding.fetch_sub(1, Ordering::SeqCst);
                Outcome {
                    burst,
                    due,
                    sent: sent.saturating_duration_since(t0),
                    answered: answered.saturating_duration_since(t0),
                    ok: resp.success,
                    dropped: if resp.success { None } else { resp.dropped },
                    error: (!resp.success).then(|| resp.message.clone()),
                }
            });
        }
        loop {
            tokio::select! {
                next = pending.next() => match next {
                    Some(o) => {
                        last_answer = last_answer.max(t0 + o.answered);
                        on_answer();
                        outcomes.push(o);
                    }
                    None => break,
                },
                () = tokio::time::sleep(TICK) => {
                    if is_set(cancel) {
                        return Err(Cancelled.into());
                    }
                }
            }
        }
    }
    let wall = ms(last_answer.saturating_duration_since(first));
    Ok((outcomes, max_depth.load(Ordering::SeqCst), wall))
}

/// Sleep until `at`, looking at `cancel` every [`TICK`].
async fn wait_until(at: Instant, cancel: Option<&AtomicBool>) -> Result<()> {
    loop {
        if is_set(cancel) {
            return Err(Cancelled.into());
        }
        let now = Instant::now();
        if now >= at {
            return Ok(());
        }
        tokio::time::sleep((at - now).min(TICK)).await;
    }
}

#[cfg(test)]
#[path = "burst_tests.rs"]
mod tests;
