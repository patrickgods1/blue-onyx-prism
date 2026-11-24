//! The benchmark run itself: `Bench` compiles a model on a device, warms it up and times the
//! per-image pipeline (decode -> preprocess -> infer -> postprocess) at the threshold the service
//! uses on that device, and post-processes the same outputs again (untimed) at the evaluation
//! confidence. Progress and cancellation are reported through `Progress`/`Cancelled`. One
//! inference at a time; the runtimes mutex is held only while compiling.
//!
//! What serving does, measured the same way:
//! - Compile: the worker's tuning ([`crate::worker::device_tuning`]: hint, in-flight requests,
//!   ONNX Runtime sessions; recorded as [`RunResult::tuning`]), so the run times and caches the
//!   compiled model that serves (B4). Timing stays one request at a time.
//! - Threshold: [`crate::worker::served_threshold`] for the device the model loaded on (B5).
//! - A variant (`--variants`, As served; [`super::variant`] has the order of operations): its
//!   enhancement in preprocess, the threshold of the frame's scene, filter rules, object filter
//!   and verifier in postprocess ([`super::served::returned_at`]), a night / IR frame of an As
//!   served model with a night model answered by the night model ([`Pipeline::serve`]). The
//!   untimed evaluation stores what is returned at every threshold of the sweep
//!   ([`super::served::windows`], B2), asking the verifier about each crop once
//!   ([`SWEEP_VERIFY_CAP`] more per image at most).
//! - Decode: the service's decode stage, the frame's statistics and its [`Scene`] included
//!   (`prep.rs`); the set's transform (a NIR proxy's grayscale) is applied between them, untimed,
//!   so the scene is the one of the frame the set sends.
//! - Throughput ([`RunResult::throughput_fps`]): the loop timer covers the frame's decode (the
//!   wait for it with read ahead) and the served stages only; the evaluation post-process and
//!   the verifier's copy of the frame, which the service never does per request, are outside it
//!   (B6), and so are the file read and the set's transform of the serial loop (the service gets
//!   the bytes with the request). With read ahead an iteration lasts at least its frame's decode
//!   ([`iteration_time`]), so the time taken out does not hand the reader free decode time.
//! - Warm-up: [`WarmupRule`] (several images, a minimum time), recorded as
//!   [`RunResult::warmup_runs`] / [`RunResult::warmup_ms`].
//! - What else ran: the live detection requests the service answered during the timed phase
//!   ([`Bench::live_requests`]) and, on Windows, the GPU video-decode load
//!   ([`crate::backend::video_decode`]).
//!
//! Read ahead: during the timed runs a helper thread (`bench-read`) reads and decodes the next
//! frame while the current one infers, as the service's prep pool does. The hand-over is a
//! rendezvous channel ([`READ_AHEAD`] = 1 frame decoded ahead), so at most two decoded frames
//! are in flight (a 2688x1520 RGB frame is about 12 MB). The stage times stay per image and mean
//! what they meant serially (`decode` is measured on the helper for that frame; preprocess,
//! infer and postprocess on the inference thread); only the wall-clock throughput
//! ([`RunResult::throughput_fps`], marked [`RunResult::pipelined`]) gains from the overlap.
//! Results, their order and the `on_image` order are those of the serial loop
//! ([`ReadAhead::Off`], kept for the parity test and as the fallback when the thread cannot
//! start). The helper is a scoped thread: it ends on cancel, on an error of either side and with
//! the run, never outliving it. Warm-up stays serial.
//!
//! A benchmark does not read ahead on a CPU device ([`ReadAhead::ExceptCpu`], the default): the
//! decode would take cores from the CPU inference and inflate the CPU's latency figures (on the
//! dev box, IPcam-general on `openvino:cpu`, 2688x1520 frames: infer p50 +27%, decode p50 +7%;
//! `docs/PRODUCTION_REVIEW.md`, "Benchmark read ahead and CPU-only priority"), which grade the
//! device. Detections over frames, where the time per frame counts and the stage times are not
//! kept (the quick check, refine fold prediction, relabelling, pseudo ground truth), read ahead
//! on every device ([`ReadAhead::Always`]).

use super::scoring::score;
use super::*;
use crate::config::ThresholdSource;
use crate::image_stats::{ImageStats, Scene};

/// What a benchmark is doing right now (for progress reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Compiling the model and creating the inference request.
    Loading,
    /// Untimed warm-up iterations.
    Warmup,
    /// Timed iterations over the images.
    Timing,
    /// The burst test after the timed runs ([`super::burst`]; done / total = requests answered
    /// of those it sends).
    Burst,
}

/// Progress of one device run, reported through [`Bench::progress`].
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    pub model: &'a str,
    pub device: &'a str,
    pub phase: Phase,
    /// Iterations finished in this phase.
    pub done: usize,
    /// Iterations of this phase.
    pub total: usize,
    /// Image being processed (timing phase).
    pub image: &'a str,
    /// The process priority of this device's run (None = no [`Bench::priority`]).
    pub priority: Option<crate::backend::priority::Applied>,
}

/// Error returned when the cancel flag is set during a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("benchmark cancelled")]
pub struct Cancelled;

/// Whether `e` (or its cause chain) is a [`Cancelled`].
pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Cancelled>().is_some())
}

/// Timing settings shared by every run of a benchmark.
pub struct Bench<'a> {
    /// Images to run (at least one image in total).
    pub sets: &'a [ImageSet],
    pub warmup: usize,
    /// Timed runs per image.
    pub repeat: usize,
    pub params: PostParams,
    pub object_filter: &'a [String],
    pub cache_dir: Option<&'a Path>,
    /// Checked before every iteration; a set flag ends the run with [`Cancelled`].
    pub cancel: Option<&'a AtomicBool>,
    /// Called at every phase change and after every iteration.
    pub progress: Option<&'a (dyn Fn(&Progress) + Sync)>,
    /// Called with each image's result as soon as its timed runs end (a cancelled run's
    /// finished images stay usable: the auto-labeler keeps them).
    pub on_image: Option<&'a (dyn Fn(&ImageRun) + Sync)>,
    pub weights: Weights,
    /// What the best confidence threshold optimizes (exact picks are computed per run).
    pub threshold_objective: threshold::Objective,
    /// What the accuracy grade is computed from.
    pub accuracy_metric: grade::AccuracyMetric,
    /// Stored predictions per image (most confident first; `MAX_STORED_PREDS`). The pseudo
    /// ground truth run keeps all of them (BM-10).
    pub max_stored_preds: usize,
    /// Compile the model without the OpenVINO model cache ([`LoadRequest::no_cache`]: a
    /// throwaway fold model of the dataset check); false for everything else.
    pub no_cache: bool,
    /// The run's process priority, switched per device (`benchmark.low_priority`: below normal
    /// for CPU devices only); None leaves the priority alone.
    pub priority: Option<&'a crate::backend::priority::DevicePriority>,
    /// When to read and decode the next frame on a helper thread while the current one infers
    /// (see the module docs).
    pub read_ahead: ReadAhead,
    /// When the warm-up ends ([`WarmupRule::DEFAULT`]).
    pub warmup_rule: WarmupRule,
    /// The detection requests the process has answered so far, over every model (the
    /// service's metrics), read around each device's timed phase ([`RunResult::live_requests`]);
    /// None outside the service.
    pub live_requests: Option<&'a (dyn Fn() -> u64 + Sync)>,
    /// Sample the GPU video-decode load during each timed phase
    /// ([`RunResult::video_decode_pct`]; Windows only, a no-op elsewhere). Off for detections
    /// over frames.
    pub video_decode: bool,
    /// The burst test after each device's timed run ([`super::burst`]); None = off.
    pub burst: Option<&'a super::burst::Burst<'a>>,
    /// Test only: a pause in every evaluation post-process (the untimed work of B6).
    #[cfg(test)]
    pub(super) slow_eval: Option<Duration>,
    /// Test only: a pause in every file read of the serial frame source (a slow disk).
    #[cfg(test)]
    pub(super) slow_read: Option<Duration>,
}

/// Warm-up images: the first this many of the run, in turn (frames of several sizes and formats
/// warm their decode and resize paths, not one frame repeated).
pub const WARMUP_IMAGES: usize = 5;
/// The warm-up lasts at least this long once it has made `warmup` runs: a count alone can be
/// over in milliseconds (a 3-run warm-up of a 5 ms model is 15 ms), before a GPU has left its
/// idle clocks.
pub const WARMUP_MIN_TIME: Duration = Duration::from_secs(2);
/// ...and at most this long, whatever the count (a slow model on a CPU)...
pub const WARMUP_MAX_TIME: Duration = Duration::from_secs(20);
/// ...or this many runs (or `warmup`, when larger), whichever comes first.
pub const WARMUP_MAX_RUNS: usize = 200;

/// When the warm-up ends: with `warmup` > 0, runs cycle over the first [`Self::images`] images
/// until at least `warmup` runs and [`Self::min_time`] have passed, capped at
/// [`Self::max_time`] or max(`warmup`, [`Self::max_runs`]) runs, whichever comes first;
/// `warmup` = 0 means no warm-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmupRule {
    pub images: usize,
    pub min_time: Duration,
    pub max_time: Duration,
    pub max_runs: usize,
}

impl WarmupRule {
    pub const DEFAULT: WarmupRule = WarmupRule {
        images: WARMUP_IMAGES,
        min_time: WARMUP_MIN_TIME,
        max_time: WARMUP_MAX_TIME,
        max_runs: WARMUP_MAX_RUNS,
    };

    /// The count only: `warmup` runs, no minimum time (tests and quick runs).
    pub const COUNT_ONLY: WarmupRule = WarmupRule {
        min_time: Duration::ZERO,
        ..Self::DEFAULT
    };

    /// Whether a warm-up asked for `warmup` runs is over after `runs` runs and `elapsed`.
    pub fn done(&self, warmup: usize, runs: usize, elapsed: Duration) -> bool {
        if warmup == 0 {
            return true;
        }
        if runs >= warmup.max(self.max_runs) || elapsed >= self.max_time {
            return true;
        }
        runs >= warmup && elapsed >= self.min_time
    }
}

/// When the timed loop reads ahead (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadAhead {
    /// Never: read and decode on the inference thread.
    Off,
    /// Unless the device the model loaded on is a CPU (the benchmark's default: the CPU's stage
    /// times stay free of the decode's competition for cores).
    ExceptCpu,
    /// On every device (detections over frames: only the time per frame counts).
    Always,
}

impl ReadAhead {
    /// Whether a run on the device spec `device` (the one the model loaded on) reads ahead.
    pub fn applies(self, device: &str) -> bool {
        match self {
            ReadAhead::Off => false,
            ReadAhead::ExceptCpu => !spec::parse(device).is_ok_and(|s| s.is_cpu()),
            ReadAhead::Always => true,
        }
    }
}

/// Frames decoded ahead of the one inferring: the hand-over channel's capacity is one less
/// (a rendezvous for 1: the helper holds the decoded frame until the inference thread takes it).
pub(super) const READ_AHEAD: usize = 1;

impl std::fmt::Debug for Bench<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bench")
            .field("sets", &self.sets.len())
            .field("warmup", &self.warmup)
            .field("repeat", &self.repeat)
            .field("params", &self.params)
            .field("object_filter", &self.object_filter)
            .field("cache_dir", &self.cache_dir)
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .field("on_image", &self.on_image.is_some())
            .field("weights", &self.weights)
            .field("threshold_objective", &self.threshold_objective)
            .field("accuracy_metric", &self.accuracy_metric)
            .field("max_stored_preds", &self.max_stored_preds)
            .field("no_cache", &self.no_cache)
            .field("priority", &self.priority)
            .field("read_ahead", &self.read_ahead)
            .field("warmup_rule", &self.warmup_rule)
            .field("live_requests", &self.live_requests.is_some())
            .field("video_decode", &self.video_decode)
            .field("burst", &self.burst)
            .finish()
    }
}

/// One request's timings (decode, preprocess, infer, postprocess; ms).
pub(super) struct Sample {
    pub(super) stages: [f64; 4],
    pub(super) predictions: Vec<Prediction>,
    /// The evaluation boxes (with their windows for a variant, [`super::served`]).
    pub(super) eval: Vec<PredBox>,
    /// The frame's computed scene (the service's, after decode).
    pub(super) scene: Scene,
    /// The night model answered it (As served).
    pub(super) routed: bool,
}

/// Folder levels below the cache root [`cache_files`] lists: OpenVINO blobs and TensorRT engines
/// are two levels down, in the model's own subfolder
/// (`openvino-<device>-<precision>/<stem>-<key16>/<hash>.blob`, `tensorrt/<stem>-<key16>/...`,
/// [`crate::storage::caches::model_cache_dir`]), ORT's CoreML model cache a few more
/// (`coreml/<COREML_CACHE_KEY>/...`).
const CACHE_DEPTH: usize = 4;

/// Whether `name` is a cache's own bookkeeping rather than a compiled file: the last-use
/// marker, the `model.txt` naming a subfolder's model (written before the compile, so a new
/// subfolder alone is no miss) and a temporary file of either.
fn is_cache_bookkeeping(name: &str) -> bool {
    name == crate::storage::caches::LAST_USED_FILE
        || name == crate::storage::caches::MODEL_FILE
        || crate::fsx::is_temp_name(name)
}

/// Every compiled file under `dir` (down to [`CACHE_DEPTH`] folders; bookkeeping files left
/// out, [`is_cache_bookkeeping`]), as paths relative to it.
pub(super) fn cache_files(dir: &Path) -> std::collections::HashSet<PathBuf> {
    fn walk(root: &Path, dir: &Path, depth: usize, out: &mut std::collections::HashSet<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let path = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => {
                    if depth < CACHE_DEPTH {
                        walk(root, &path, depth + 1, out);
                    }
                }
                Ok(_) => {
                    let bookkeeping = e.file_name().to_str().is_some_and(is_cache_bookkeeping);
                    if !bookkeeping && let Ok(rel) = path.strip_prefix(root) {
                        out.insert(rel.to_path_buf());
                    }
                }
                Err(_) => {}
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    walk(dir, dir, 0, &mut out);
    out
}

/// The folders under the cache root a compile on one device writes to ([`cache_scope`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CacheScope {
    /// Every folder whose name starts with this (`openvino-cpu-`).
    Prefix(String),
    /// This folder (`tensorrt`, `coreml`).
    Folder(&'static str),
}

impl CacheScope {
    /// Whether `rel` (a file path relative to the cache root) is inside the scope's folders.
    pub(super) fn holds(&self, rel: &Path) -> bool {
        let mut parts = rel.components();
        let Some(first) = parts.next().and_then(|c| c.as_os_str().to_str()) else {
            return false;
        };
        // A file directly in the root is in no folder.
        if parts.next().is_none() {
            return false;
        }
        match self {
            CacheScope::Prefix(p) => first.starts_with(p.as_str()),
            CacheScope::Folder(f) => first == *f,
        }
    }
}

/// The cache folders a compile on `device` writes to; None for a runtime that keeps no compile
/// cache (ORT CPU, CUDA, DirectML).
///
/// - OpenVINO: `openvino-<class>-<precision>` ([`crate::backend::device::ov_cache_dir`], set
///   per compile from the precision the plugin reports at that moment). Every precision folder
///   of the device's class counts: the precision is not known out here, and the CPU blob hash
///   ignores it, so the folder is all that tells a CPU f32 blob from a bf16 one.
/// - ORT TensorRT: `tensorrt/` (engines, timing cache). ORT CoreML: `coreml/` (the graph
///   rewrite's copies at its top, the CoreML model cache below).
pub(super) fn cache_scope(device: &crate::backend::device::DeviceInfo) -> Option<CacheScope> {
    use crate::backend::spec::{self, Target};
    match device.runtime {
        Runtime::OpenVino => {
            // The folder name with an empty precision: `openvino-<class>-` (the same cleaning).
            let probe = crate::backend::device::ov_cache_dir(Path::new(""), &device.actual, "x");
            let name = probe.to_string_lossy().into_owned();
            let prefix = name.strip_suffix('x').unwrap_or(&name).to_string();
            Some(CacheScope::Prefix(prefix))
        }
        Runtime::Ort => match spec::parse(&device.spec).ok()?.device()?.target {
            Target::TensorRt => Some(CacheScope::Folder("tensorrt")),
            Target::CoreMl => Some(CacheScope::Folder("coreml")),
            _ => None,
        },
    }
}

/// The cache result of a compile on `device` from the files under the cache root before
/// ([`cache_files`]) and after it: "miss" (a file appeared in the device's cache folder,
/// [`cache_scope`]), "hit" (that folder holds files and none is new), "unknown" (it holds none:
/// caching off or failing) or "none" (the runtime keeps no compile cache).
pub(super) fn cache_status(
    before: &std::collections::HashSet<PathBuf>,
    after: &std::collections::HashSet<PathBuf>,
    device: &crate::backend::device::DeviceInfo,
) -> String {
    let Some(mine) = cache_scope(device) else {
        return "none".into();
    };
    let mut any = false;
    for f in after.iter().filter(|f| mine.holds(f)) {
        if !before.contains(f) {
            return "miss".into();
        }
        any = true;
    }
    if any { "hit" } else { "unknown" }.into()
}

pub(super) fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Lock the shared runtimes, ignoring poisoning (a panicked compile leaves them usable).
pub(super) fn lock(rt: &Mutex<Runtimes>) -> std::sync::MutexGuard<'_, Runtimes> {
    rt.lock_recover()
}

pub(super) fn to_pred(p: &Prediction) -> PredBox {
    PredBox {
        label: p.label.clone(),
        confidence: p.confidence,
        bbox: [
            p.x_min as f32,
            p.y_min as f32,
            p.x_max as f32,
            p.y_max as f32,
        ],
        window: None,
    }
}

impl<'a> Bench<'a> {
    /// Settings over `sets` with no cache dir, filter, cancel flag or progress callback.
    pub fn new(sets: &'a [ImageSet], warmup: usize, repeat: usize, params: PostParams) -> Self {
        Self {
            sets,
            warmup,
            repeat,
            params,
            object_filter: &[],
            cache_dir: None,
            cancel: None,
            progress: None,
            on_image: None,
            weights: Weights::default(),
            threshold_objective: threshold::Objective::default(),
            accuracy_metric: grade::AccuracyMetric::default(),
            max_stored_preds: MAX_STORED_PREDS,
            no_cache: false,
            priority: None,
            read_ahead: ReadAhead::ExceptCpu,
            warmup_rule: WarmupRule::DEFAULT,
            live_requests: None,
            video_decode: false,
            burst: None,
            #[cfg(test)]
            slow_eval: None,
            #[cfg(test)]
            slow_read: None,
        }
    }

    pub fn image_count(&self) -> usize {
        self.sets.iter().map(|s| s.images.len()).sum()
    }

    /// Dataset ids, comma separated.
    pub fn describe_sets(&self) -> String {
        self.sets
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(super) fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    /// The test pause of the serial frame source's file reads (none outside tests).
    fn read_pause(&self) -> Duration {
        #[cfg(test)]
        {
            self.slow_read.unwrap_or_default()
        }
        #[cfg(not(test))]
        {
            Duration::ZERO
        }
    }

    pub(super) fn report(
        &self,
        job: &Job,
        device: &str,
        phase: Phase,
        done: usize,
        total: usize,
        image: &str,
    ) {
        if let Some(cb) = self.progress {
            cb(&Progress {
                model: &job.name,
                device,
                phase,
                done,
                total,
                image,
                priority: self.priority.and_then(|p| p.current()),
            });
        }
    }

    /// Set the run's priority for `device` (a device spec), when the run has one.
    fn enter_priority(&self, device: &str) {
        if let Some(p) = self.priority {
            p.enter(device);
        }
    }

    /// Compile `job` on the device spec `device` and time it. The spec's candidates mirror the
    /// server: a GPU request may fall back to CPU (see [`RunResult::fell_back`]).
    pub fn run(&self, runtimes: &Mutex<Runtimes>, job: &Job, device: &str) -> Result<RunResult> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &job.path);
        self.run_candidates(runtimes, job, device, &candidates)
    }

    /// Compile `job` on exactly `device` (no fallback) and time it.
    pub fn run_exact(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        device: Device,
    ) -> Result<RunResult> {
        let cand = Candidate {
            device,
            fell_back: false,
            note: None,
        };
        self.run_candidates(runtimes, job, &device.to_string(), &[cand])
    }

    fn run_candidates(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        requested: &str,
        candidates: &[Candidate],
    ) -> Result<RunResult> {
        if self.image_count() == 0 {
            bail!("no benchmark images");
        }
        self.check_cancel()?;
        self.report(job, requested, Phase::Loading, 0, 1, "");
        let req = LoadRequest {
            path: job.path.clone(),
            requested: requested.to_string(),
            gpu_precision: job.gpu_precision.clone(),
            no_cache: self.no_cache,
        };
        let before = self.cache_dir.filter(|_| !self.no_cache).map(cache_files);
        let t = Instant::now();
        let label = format!("model {}", job.name);
        // Compile under the runtimes lock (workers compile through the same lock), create the
        // inference request outside it. A compile can not be interrupted: a cancel takes
        // effect after the current compile, before the next fallback candidate (BM-06).
        let loaded = crate::backend::try_candidates(&label, candidates, |cand| {
            self.check_cancel()?;
            // The compile counts as the device's run (OpenVINO's GPU compile is CPU work too,
            // and it keeps the normal priority with the GPU).
            self.enter_priority(&cand.device.to_string());
            // The worker's tuning on this device (B4): what serves is what is timed and cached.
            let (compiled, tuning) = {
                let mut rt = lock(runtimes);
                let tuning =
                    crate::worker::device_tuning(job.concurrency.as_ref(), &rt, &cand.device);
                (rt.compile_tuned(cand, &req, &tuning)?, tuning)
            };
            Ok((compiled.into_backend()?, tuning))
        });
        // A cancel during the compiles is a cancel, not a load failure (the error of several
        // failed candidates no longer carries `Cancelled`).
        self.check_cancel()?;
        let (backend, tuning) =
            loaded.with_context(|| format!("loading {}", job.path.display()))?;
        let tuning = run_tuning(&backend, &tuning);
        let compile_ms = ms(t.elapsed());
        // Decided again on the device it loaded on (a fallback to CPU is a CPU run).
        match backend.info().device.spec.as_str() {
            "" => self.enter_priority(requested),
            spec => self.enter_priority(spec),
        }
        let cache = match (self.cache_dir, before) {
            (None, _) | (_, None) => "disabled".to_string(),
            (Some(dir), Some(before)) => {
                cache_status(&before, &cache_files(dir), &backend.info().device)
            }
        };
        let loaded_spec = backend.info().device.spec.clone();
        let mut record = job
            .variant
            .as_ref()
            .and_then(|v| v.served.as_ref())
            .map(|s| ServedRun {
                night_model: s.night.as_ref().map(|n| n.model().to_string()),
                notes: s.notes.clone(),
                ..ServedRun::default()
            });
        let verifier = match job.variant.as_ref().and_then(|v| v.verify.as_ref()) {
            Some(v) => match (
                self.load_on(runtimes, &v.job, &loaded_spec, "the verifier"),
                record.as_mut(),
            ) {
                (Ok(b), _) => Some(b),
                // As served: a verifier that does not load leaves the candidates unverified,
                // as the service's fail-open does.
                (Err(e), Some(r)) => {
                    self.check_cancel()?;
                    r.notes.push(format!(
                        "verifier {} did not load ({e:#}): candidates stay unverified \
                         (fail-open, as served)",
                        v.job.name
                    ));
                    None
                }
                (Err(e), None) => return Err(e),
            },
            None => None,
        };
        let night = self.load_night(runtimes, job, &loaded_spec, record.as_mut())?;
        let companions = Companions {
            verifier,
            night,
            record,
        };
        // `time` owns the backend and its companions and drops them before it returns: the
        // burst's worker never loads beside the benchmark's own copy.
        let mut result = self.time(
            job, requested, backend, companions, compile_ms, cache, tuning,
        )?;
        if let Some(burst) = self.burst {
            self.check_cancel()?;
            result.burst = Some(burst.run(self, job, &result)?);
        }
        Ok(result)
    }

    /// As served: compile the night model (and its verifier) on `device`, the spec the model
    /// loaded on, with the night model's worker tuning. A night model that cannot be used is
    /// recorded in `record` as the reason its frames run on the model itself.
    fn load_night<'j>(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &'j Job,
        device: &str,
        record: Option<&mut ServedRun>,
    ) -> Result<Option<NightLoaded<'j>>> {
        let (Some(route), Some(record)) = (
            job.variant
                .as_ref()
                .and_then(|v| v.served.as_ref())
                .and_then(|s| s.night.as_ref()),
            record,
        ) else {
            return Ok(None);
        };
        let (njob, nvariant) = match route {
            super::variant::NightRoute::Model { job, variant } => (&**job, &**variant),
            super::variant::NightRoute::Fallback { reason, .. } => {
                record.night_fallback = Some(reason.clone());
                return Ok(None);
            }
        };
        let backend = match self.load_on(runtimes, njob, device, "the night model") {
            Ok(b) => b,
            Err(e) => {
                self.check_cancel()?;
                record.night_fallback = Some(format!("it did not load on {device}: {e:#}"));
                return Ok(None);
            }
        };
        let night_spec = backend.info().device.spec.clone();
        let verifier = match nvariant.verify.as_ref() {
            Some(v) => match self.load_on(runtimes, &v.job, &night_spec, "the verifier") {
                Ok(b) => Some(b),
                Err(e) => {
                    self.check_cancel()?;
                    record.notes.push(format!(
                        "the night model's verifier {} did not load ({e:#}): its candidates \
                         stay unverified (fail-open, as served)",
                        v.job.name
                    ));
                    None
                }
            },
            None => None,
        };
        Ok(Some(NightLoaded {
            job: njob,
            variant: nvariant,
            backend,
            verifier,
        }))
    }

    /// Compile `vjob` (the verifier of a `verify:<model>` variant, or a served model's night
    /// model or verifier: `what`) on `device` (the spec the benchmarked model loaded on; falls
    /// back like the service would), with its worker's tuning.
    fn load_on(
        &self,
        runtimes: &Mutex<Runtimes>,
        vjob: &Job,
        device: &str,
        what: &str,
    ) -> Result<Backend> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &vjob.path);
        let req = LoadRequest {
            path: vjob.path.clone(),
            requested: device.to_string(),
            gpu_precision: vjob.gpu_precision.clone(),
            no_cache: false,
        };
        let label = format!("{what} {}", vjob.name);
        let loaded = crate::backend::try_candidates(&label, &candidates, |cand| {
            self.check_cancel()?;
            let compiled = {
                let mut rt = lock(runtimes);
                let tuning =
                    crate::worker::device_tuning(vjob.concurrency.as_ref(), &rt, &cand.device);
                rt.compile_tuned(cand, &req, &tuning)?
            };
            compiled.into_backend()
        });
        self.check_cancel()?;
        loaded.with_context(|| format!("loading {what} {}", vjob.path.display()))
    }

    /// Warm up and time `backend` over every image, then score its detections.
    #[allow(
        clippy::too_many_arguments,
        reason = "the loaded backend and what its load measured, passed once"
    )]
    fn time(
        &self,
        job: &Job,
        requested: &str,
        backend: Backend,
        companions: Companions,
        compile_ms: f64,
        cache: String,
        tuning: RunTuning,
    ) -> Result<RunResult> {
        let Companions {
            verifier,
            night,
            mut record,
        } = companions;
        let info = backend.info();
        let dev = info.device.clone();
        let (in_w, in_h) = info.input_size;
        let family =
            crate::model::make_family(job.family, &info.inputs, &info.outputs, job.classes.len())?;
        let loaded_on = if dev.spec.is_empty() {
            dev.actual.clone()
        } else {
            dev.spec.clone()
        };
        let (threshold, threshold_source) = self.served_threshold(job, &loaded_on);
        let night_threshold = job
            .variant
            .as_ref()
            .and_then(|v| v.served.as_ref())
            .and_then(|s| s.night_threshold);
        let night = match night {
            Some(n) => Some(Box::new(self.night_pipeline(n)?)),
            None => None,
        };
        let mut pipe = Pipeline {
            pre: Preprocessor::for_family(in_w, in_h, family.as_ref()),
            family,
            backend,
            params: PostParams {
                confidence_threshold: threshold,
                nms_iou: self.params.nms_iou,
            },
            eval_params: PostParams {
                confidence_threshold: EVAL_CONFIDENCE
                    .min(threshold)
                    .min(night_threshold.unwrap_or(EVAL_CONFIDENCE)),
                nms_iou: self.params.nms_iou,
            },
            classes: &job.classes,
            filter: self.object_filter_for(job),
            variant: job.variant.as_ref(),
            verifier: match (
                verifier,
                job.variant.as_ref().and_then(|v| v.verify.as_ref()),
            ) {
                (Some(b), Some(v)) => Some(Verifier::new(b, v, self.params)?),
                _ => None,
            },
            night_threshold,
            night,
            fixed: false,
            verify_capped: 0,
            #[cfg(test)]
            slow_eval: self.slow_eval,
        };

        let all: Vec<(&ImageSet, &images::SetImage)> = self
            .sets
            .iter()
            .flat_map(|s| s.images.iter().map(move |i| (s, i)))
            .collect();
        let (warmup_runs, warmup_ms) = self.warm_up(job, requested, &all, &mut pipe)?;
        let repeat = self.repeat.max(1);
        // What else the machine served during the timed phase (the service's live requests,
        // Windows' GPU video decode).
        let live_before = self.live_requests.map(|f| f());
        let decode_load = if self.video_decode {
            crate::backend::video_decode::Sampler::start()
        } else {
            None
        };
        let timed_from = Instant::now();
        let timed = if self.read_ahead.applies(&loaded_on) {
            self.timed_runs_ahead(job, requested, &all, &mut pipe, repeat)
        } else {
            self.timed_runs(
                job,
                requested,
                &all,
                &mut pipe,
                repeat,
                Source::Serial,
                &mut serial_frames(&all, self.read_pause()),
            )
        };
        let timed_wall = timed_from.elapsed();
        let video_decode_pct = decode_load.and_then(|s| s.finish());
        let mut timed = timed?;
        let live_requests = live_before
            .zip(self.live_requests.map(|f| f()))
            .map(|(a, b)| b.saturating_sub(a));
        let live_requests_per_min = live_requests.map(|n| per_minute(n, timed_wall));
        let loop_s = timed.loop_time.as_secs_f64();
        if let Some(r) = record.as_mut() {
            r.night_routed = timed.routed;
            r.verify_capped_images =
                pipe.verify_capped + pipe.night.as_ref().map_or(0, |n| n.verify_capped);
        }

        let stages_ms = timed.stage_stats();
        let (accuracy, accuracy_note) = score(
            job,
            &all,
            &mut timed.per_image,
            &timed.eval_preds,
            threshold,
            self.threshold_objective,
        );
        // No accuracy (no CCTV class): the held-out tags' per-class recall all the same.
        let tag_recall = if accuracy.is_none() {
            super::scoring::tag_recall_only(job, &all, &timed.eval_preds, threshold)
        } else {
            Vec::new()
        };
        let accuracy_score = accuracy
            .as_ref()
            .and_then(|a| a.score(self.accuracy_metric));
        let relative = accuracy.as_ref().is_some_and(|a| a.relative);
        let grades = Grades::with_metric(
            self.accuracy_metric,
            accuracy_score,
            stages_ms.total.p50,
            self.weights,
            relative,
        );
        let first_dets = timed
            .per_image
            .first()
            .map(|r| r.preds.iter().map(Detection::from).collect())
            .unwrap_or_default();
        let eval_preds: Vec<Vec<PredBox>> = {
            let map = ClassMap::for_model(&job.classes);
            timed.eval_preds.iter().map(|v| map.map_preds(v)).collect()
        };
        Ok(RunResult {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: pipe.family.kind().to_string(),
            requested_device: requested.to_string(),
            device: dev.actual.clone(),
            device_name: dev.full_name.clone(),
            spec: dev.spec.clone(),
            fell_back: dev.fell_back,
            input: format!("{in_w}x{in_h}"),
            image: self.describe_sets(),
            image_size: match all.as_slice() {
                [(_, one)] => format!("{}x{}", one.width, one.height),
                _ => format!("{} images", all.len()),
            },
            images: all.len(),
            compile_ms,
            cache,
            warmup_iterations: self.warmup,
            warmup_ms,
            warmup_runs,
            tuning: Some(tuning),
            served_threshold: Some(threshold),
            served_threshold_source: Some(threshold_source.label()),
            live_requests,
            live_requests_per_min,
            video_decode_pct,
            repeat,
            stages_ms,
            throughput_fps: if loop_s > 0.0 {
                timed.total_iters as f64 / loop_s
            } else {
                0.0
            },
            pipelined: timed.pipelined,
            detections: first_dets,
            by_resolution: timed.by_resolution(),
            accuracy,
            accuracy_note,
            reference_for: super::scoring::reference_sets(&job.name, &all),
            preds_capped_images: super::search::capped_images(&eval_preds),
            tag_recall,
            grades: Some(grades),
            agreement: None,
            served: record,
            burst: None,
            eval_preds,
            per_image: timed.per_image,
        })
    }

    /// The night model's pipeline of an As served run: its own stages, threshold
    /// (`threshold_for_scene` on the device it loaded on: its `night_threshold`, else its served
    /// threshold there; every frame it gets is a night / IR one), filters, object filter and
    /// verifier; never routed again.
    fn night_pipeline<'s>(&'s self, n: NightLoaded<'s>) -> Result<Pipeline<'s>> {
        let NightLoaded {
            job,
            variant,
            backend,
            verifier,
        } = n;
        let info = backend.info();
        let (in_w, in_h) = info.input_size;
        let family =
            crate::model::make_family(job.family, &info.inputs, &info.outputs, job.classes.len())?;
        let loaded_on = if info.device.spec.is_empty() {
            info.device.actual.clone()
        } else {
            info.device.spec.clone()
        };
        let threshold = match variant.served.as_ref().and_then(|s| s.night_threshold) {
            Some(t) => t,
            None => self.served_threshold(job, &loaded_on).0,
        };
        Ok(Pipeline {
            pre: Preprocessor::for_family(in_w, in_h, family.as_ref()),
            family,
            backend,
            params: PostParams {
                confidence_threshold: threshold,
                nms_iou: self.params.nms_iou,
            },
            eval_params: PostParams {
                confidence_threshold: EVAL_CONFIDENCE.min(threshold),
                nms_iou: self.params.nms_iou,
            },
            classes: &job.classes,
            filter: self.object_filter_for(job),
            variant: Some(variant),
            verifier: match (verifier, variant.verify.as_ref()) {
                (Some(b), Some(v)) => Some(Verifier::new(b, v, self.params)?),
                _ => None,
            },
            night_threshold: None,
            night: None,
            fixed: true,
            verify_capped: 0,
            #[cfg(test)]
            slow_eval: None,
        })
    }

    /// The threshold the service uses for `job` on `device` (the spec it loaded on) and where it
    /// comes from ([`crate::worker::served_threshold`]: the device's profile, else the model's
    /// own threshold, else the run's global one).
    fn served_threshold(&self, job: &Job, device: &str) -> (f32, ThresholdSource) {
        let fallback = match job.confidence_threshold {
            Some(t) => (t, ThresholdSource::Model),
            None => (self.params.confidence_threshold, ThresholdSource::Global),
        };
        crate::worker::served_threshold(&job.device_thresholds, fallback, device)
    }

    /// The object filter of `job`'s production predictions: its own (an empty list = none),
    /// else the global one (empty = none).
    fn object_filter_for<'s>(&'s self, job: &'s Job) -> Option<&'s [String]> {
        match &job.object_filter {
            Some(f) if !f.is_empty() => Some(f),
            Some(_) => None,
            None => (!self.object_filter.is_empty()).then_some(self.object_filter),
        }
    }

    /// The untimed warm-up ([`WarmupRule`]: the first [`WARMUP_IMAGES`] images in turn, each
    /// decoded and run through the served stages); returns its runs and wall time (ms).
    fn warm_up(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
    ) -> Result<(usize, f64)> {
        let t = Instant::now();
        self.report(job, requested, Phase::Warmup, 0, self.warmup, "");
        if self.warmup == 0 {
            return Ok((0, ms(t.elapsed())));
        }
        // An image that cannot be read or decoded is left out here: the timed runs report it
        // in its own context, after the images before it.
        let mut frames: Vec<_> = all
            .iter()
            .take(self.warmup_rule.images.max(1))
            .filter_map(|(set, img)| Some((set.transform, img.bytes().ok()?)))
            .collect();
        let mut runs = 0;
        let mut next = 0;
        while !self.warmup_rule.done(self.warmup, runs, t.elapsed()) {
            self.check_cancel()?;
            if frames.is_empty() {
                break;
            }
            next %= frames.len();
            let Some((transform, bytes)) = frames.get(next) else {
                break;
            };
            let Ok(frame) = decode(bytes, *transform) else {
                frames.remove(next);
                continue;
            };
            pipe.serve(frame, None).context("warm-up inference")?;
            next += 1;
            runs += 1;
            self.report(
                job,
                requested,
                Phase::Warmup,
                runs,
                self.warmup.max(runs),
                "",
            );
        }
        Ok((runs, ms(t.elapsed())))
    }

    /// [`Self::timed_runs`] with the frames read and decoded ahead on a scoped helper thread
    /// (see the module docs); the serial loop when the thread cannot start.
    fn timed_runs_ahead(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
        repeat: usize,
    ) -> Result<Timings> {
        let cancel = self.cancel;
        std::thread::scope(|scope| {
            let (tx, rx) = crossbeam_channel::bounded::<Result<Frame>>(READ_AHEAD - 1);
            let reader = std::thread::Builder::new()
                .name("bench-read".into())
                .spawn_scoped(scope, move || read_frames(all, repeat, cancel, &tx));
            let reader = match reader {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        "benchmark: could not start the image reader thread ({e}); reading \
                         and decoding on the inference thread"
                    );
                    return self.timed_runs(
                        job,
                        requested,
                        all,
                        pipe,
                        repeat,
                        Source::Serial,
                        &mut serial_frames(all, self.read_pause()),
                    );
                }
            };
            // `rx` moves into the source and drops with it when the loop ends (done, error or
            // cancel), which stops a reader waiting to hand a frame over.
            let mut source = move |_: usize| -> Result<Frame> {
                match rx.recv() {
                    Ok(f) => f,
                    // The reader stops early only on a cancel (or a panic, reported below).
                    Err(_) => {
                        self.check_cancel()?;
                        bail!("the benchmark's image reader thread stopped")
                    }
                }
            };
            let timed = self.timed_runs(
                job,
                requested,
                all,
                pipe,
                repeat,
                Source::Ahead,
                &mut source,
            );
            drop(source);
            let joined = reader.join();
            match (timed, joined) {
                (Ok(mut t), Ok(())) => {
                    t.pipelined = true;
                    Ok(t)
                }
                (Err(e), _) => Err(e),
                (Ok(_), Err(_)) => bail!("the benchmark's image reader thread panicked"),
            }
        })
    }

    /// `repeat` timed runs of every image; the last run of each image keeps its predictions
    /// (production and evaluation threshold). `frames` gives each run's decoded frame, in
    /// order (image by image, `repeat` per image; its argument is the image's index in `all`);
    /// `source` says where it reads them (what of its wait the throughput counts).
    #[allow(
        clippy::too_many_arguments,
        reason = "the timed loop's pieces, passed from its two callers"
    )]
    fn timed_runs(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
        repeat: usize,
        source: Source,
        frames: &mut dyn FnMut(usize) -> Result<Frame>,
    ) -> Result<Timings> {
        let total_iters = all.len().saturating_mul(repeat);
        let mut timed = Timings {
            samples: Default::default(),
            by_res: BTreeMap::new(),
            per_image: Vec::with_capacity(all.len()),
            eval_preds: Vec::with_capacity(all.len()),
            total_iters,
            loop_time: Duration::ZERO,
            pipelined: false,
            routed: 0,
        };
        let mut done = 0;
        self.report(job, requested, Phase::Timing, 0, total_iters, "");
        for (i, (set, img)) in all.iter().enumerate() {
            // Not pre-sized by `repeat`: the CLI and service bound it, but stay safe regardless.
            let mut totals = Vec::new();
            let mut infers = Vec::new();
            let mut last: Option<Sample> = None;
            let bucket = img.resolution();
            for r in 0..repeat {
                self.check_cancel()?;
                let t = Instant::now();
                // A read failure as it is, a decode failure in the image's context (as before
                // read ahead).
                let frame = frames(i)?;
                let waited = t.elapsed();
                let decoded = frame.with_context(|| format!("image {}", img.file))?;
                let decode_ms = decoded.decode_ms;
                // Serially the wait also holds the file read and the set's transform, which the
                // service never does (the bytes come with the request): only the decode counts.
                let waited = match source {
                    Source::Ahead => waited,
                    Source::Serial => ms_duration(decode_ms),
                };
                // Not timed: the service keeps its decoded frame for the verifier without a
                // copy (B6).
                let original = pipe
                    .needs_original(decoded.scene)
                    .then(|| decoded.img.clone());
                let t = Instant::now();
                let served = pipe
                    .serve(decoded, original)
                    .with_context(|| format!("image {}", img.file))?;
                timed.loop_time += iteration_time(waited, t.elapsed(), decode_ms);
                // Not timed either: the evaluation post-process is the benchmark's own (B6).
                let sample = pipe
                    .evaluate(served, r + 1 == repeat)
                    .with_context(|| format!("image {}", img.file))?;
                let total = timed.record(bucket, &sample);
                totals.push(total);
                infers.push(sample.stages[2]);
                last = Some(sample);
                done += 1;
                self.report(job, requested, Phase::Timing, done, total_iters, &img.file);
            }
            timed.by_res.entry(bucket).or_default().1 += 1;
            let last = last.context("no timed run (repeat must be >= 1)")?;
            timed.routed += usize::from(last.routed);
            let (run, eval) = image_run(set, img, &last, &totals, &infers, self.max_stored_preds);
            if let Some(f) = self.on_image {
                f(&run);
            }
            timed.eval_preds.push(eval);
            timed.per_image.push(run);
        }
        Ok(timed)
    }
}

/// Where the timed loop's frames come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// Read and decoded on this thread: the throughput counts the decode, not the wait.
    Serial,
    /// Read and decoded ahead on the reader thread: the throughput counts the wait for it.
    Ahead,
}

/// `ms` milliseconds (0 for a negative or not finite value).
fn ms_duration(ms: f64) -> Duration {
    Duration::try_from_secs_f64(ms.max(0.0) / 1e3).unwrap_or(Duration::ZERO)
}

/// One timed run's decoded frame, or why it did not decode (a frame source's `Err` = the image
/// could not be read).
pub(super) type Frame = Result<Decoded>;

/// A decoded frame (the set's transform applied), its decode time and its scene.
pub(super) struct Decoded {
    img: crate::image::RgbImage,
    /// Decode and the frame's statistics, as the service's decode stage counts them, without
    /// reading the file or the transform (ms).
    decode_ms: f64,
    /// The scene the service computes after decode ([`ImageStats::scene`]).
    scene: Scene,
}

/// Decode `bytes` and take the frame's statistics, both timed as the service's decode stage
/// (`prep.rs`); the set's `transform` (not part of any stage time) is applied between them, so
/// the scene is that of the frame the set stands for (a NIR proxy's grayscale frame is `ir`).
fn decode(bytes: &[u8], transform: Option<images::ImageTransform>) -> Result<Decoded> {
    let t = Instant::now();
    let mut img = crate::image::decode(bytes)?;
    let mut decode = t.elapsed();
    if let Some(tr) = transform {
        tr.apply(&mut img);
    }
    let t = Instant::now();
    let scene = ImageStats::compute(&img).scene();
    decode += t.elapsed();
    Ok(Decoded {
        img,
        decode_ms: ms(decode),
        scene,
    })
}

/// The scene [`decode`] computes for `bytes` (tests).
#[cfg(test)]
pub(super) fn decode_scene(bytes: &[u8], transform: Option<images::ImageTransform>) -> Scene {
    decode(bytes, transform)
        .map(|d| d.scene)
        .unwrap_or_else(|e| panic!("{e:#}"))
}

/// The time one timed iteration counts toward the throughput: the wait for its frame plus the
/// served stages (B6: nothing the service does not do per request), and at least the frame's
/// decode. With read ahead the reader decodes the next frame during the untimed evaluation
/// too; without the floor a decode-bound run would look faster than the 1-frame pipeline is
/// (the service's steady state is max(decode, serve) per frame). Serially the loop passes the
/// decode as the wait ([`Source::Serial`]), so the floor never applies.
pub(super) fn iteration_time(waited: Duration, served: Duration, decode_ms: f64) -> Duration {
    (waited + served).max(ms_duration(decode_ms))
}

/// `n` events over `over` as a rate per minute (0 over no time).
pub(super) fn per_minute(n: u64, over: Duration) -> f64 {
    let min = over.as_secs_f64() / 60.0;
    if min > 0.0 { n as f64 / min } else { 0.0 }
}

/// What a compiled backend records of its tuning ([`RunResult::tuning`]): OpenVINO's hint as it
/// compiled, the in-flight requests it created (after its clamps) and the ONNX Runtime sessions.
fn run_tuning(backend: &Backend, tuning: &crate::profile::DeviceTuning) -> RunTuning {
    let ov = backend.info().device.runtime == Runtime::OpenVino;
    RunTuning {
        hint: ov.then(|| backend.hint_and_streams().0.as_str().to_string()),
        inflight: backend.slots(),
        ort_sessions: if ov { 1 } else { tuning.ort_sessions },
    }
}

/// The serial frame source: each image read once (cached across its runs), decoded per run
/// on the calling thread.
fn serial_frames<'s>(
    all: &'s [(&ImageSet, &images::SetImage)],
    read_pause: Duration,
) -> impl FnMut(usize) -> Result<Frame> + 's {
    let mut cached: Option<(usize, std::borrow::Cow<'static, [u8]>)> = None;
    move |i| {
        let (set, img) = all.get(i).context("frame index out of range")?;
        let bytes = match cached.take() {
            Some((at, b)) if at == i => b,
            _ => {
                if !read_pause.is_zero() {
                    std::thread::sleep(read_pause);
                }
                img.bytes()?
            }
        };
        let frame = decode(&bytes, set.transform);
        cached = Some((i, bytes));
        Ok(frame)
    }
}

/// The read-ahead thread: every image read once, decoded `repeat` times (one frame per timed
/// run, in the timed loop's order) and handed over through `tx`. Stops at a cancel, after a
/// failed read or decode (sent first), and when the inference thread is gone (`send` fails).
fn read_frames(
    all: &[(&ImageSet, &images::SetImage)],
    repeat: usize,
    cancel: Option<&AtomicBool>,
    tx: &crossbeam_channel::Sender<Result<Frame>>,
) {
    let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
    for (set, img) in all {
        if cancelled() {
            return;
        }
        let bytes = match img.bytes() {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };
        for _ in 0..repeat {
            if cancelled() {
                return;
            }
            let frame = decode(&bytes, set.transform);
            let failed = frame.is_err();
            if tx.send(Ok(frame)).is_err() || failed {
                return;
            }
        }
    }
}

/// What a run compiles beside the model: the verifier, the night model (As served) and the
/// record of what the served pipeline did.
struct Companions<'j> {
    verifier: Option<Backend>,
    night: Option<NightLoaded<'j>>,
    record: Option<ServedRun>,
}

/// The compiled night model of an As served run.
struct NightLoaded<'j> {
    job: &'j Job,
    variant: &'j PipelineVariant,
    backend: Backend,
    verifier: Option<Backend>,
}

/// Verifier runs an image's evaluation may make beyond those of its timed request (the sweep
/// asks about the candidates of every threshold, [`super::served`]): crowded frames with many
/// verified candidates are bounded. Past it a candidate stays unverified (kept) at the
/// thresholds that would check it, as a crop the service cannot check does; the run counts
/// such images ([`ServedRun::verify_capped_images`]).
pub const SWEEP_VERIFY_CAP: usize = 32;

/// One compiled model with the production per-request stages (the code `worker.rs` runs).
pub(super) struct Pipeline<'j> {
    backend: Backend,
    family: Box<dyn crate::model::Family>,
    pre: Preprocessor,
    /// The served threshold (a day frame's; the night model's own on the night pipeline).
    params: PostParams,
    eval_params: PostParams,
    classes: &'j [String],
    filter: Option<&'j [String]>,
    /// Enhancement, filter rules and verifier on top of the plain stages ([`super::variant`]).
    variant: Option<&'j PipelineVariant>,
    /// The verifier of the variant.
    verifier: Option<Verifier<'j>>,
    /// As served: the threshold of night / IR frames run here (`night_threshold`).
    night_threshold: Option<f32>,
    /// As served: the night model's pipeline (night / IR frames go there).
    night: Option<Box<Pipeline<'j>>>,
    /// The night model's pipeline: its threshold stays whatever threshold the sweep scores.
    fixed: bool,
    /// Images whose evaluation hit [`SWEEP_VERIFY_CAP`].
    verify_capped: usize,
    /// Test only: a pause in [`Pipeline::evaluate`] (untimed work).
    #[cfg(test)]
    slow_eval: Option<Duration>,
}

/// The second model of a `verify:<model>` variant (and of an As served model with `verify`):
/// the stages the verifier's worker runs on a verifier check (`worker::Purpose::Verify`): its
/// own enhancement for the frame's scene, the resize, inference, and its raw detections at
/// `min_confidence` (no night rules, no object filter).
pub(super) struct Verifier<'j> {
    backend: Backend,
    family: Box<dyn crate::model::Family>,
    pre: Preprocessor,
    params: PostParams,
    classes: &'j [String],
    variant: &'j super::variant::VerifyVariant,
}

impl<'j> Verifier<'j> {
    fn new(
        backend: Backend,
        v: &'j super::variant::VerifyVariant,
        bench: PostParams,
    ) -> Result<Self> {
        let info = backend.info();
        let (in_w, in_h) = info.input_size;
        let family = crate::model::make_family(
            v.job.family,
            &info.inputs,
            &info.outputs,
            v.job.classes.len(),
        )?;
        let configured = v
            .job
            .confidence_threshold
            .unwrap_or(bench.confidence_threshold);
        Ok(Self {
            pre: Preprocessor::for_family(in_w, in_h, family.as_ref()),
            family,
            backend,
            params: PostParams {
                confidence_threshold: crate::worker::effective_threshold(
                    v.config.min_confidence,
                    configured,
                ),
                nms_iou: bench.nms_iou,
            },
            classes: &v.job.classes,
            variant: v,
        })
    }

    /// The verifier's predictions on `img` (a crop or the frame) of a frame of `scene`.
    fn detect(&mut self, mut img: crate::image::RgbImage, scene: Scene) -> Result<Vec<Prediction>> {
        crate::model::enhance::apply(&mut img, self.variant.modes.for_scene(scene));
        let (chw, ctx) = self.pre.run(&img.rgb, img.width, img.height)?;
        let extra = self.family.extra_inputs(&ctx);
        let outputs = self.backend.infer(chw, &extra)?;
        let dets = self.family.postprocess(&outputs, &ctx, &self.params)?;
        Ok(crate::model::to_predictions(
            &dets,
            &ctx,
            self.classes,
            None,
        ))
    }
}

/// The verifier's answers on one frame so far (each crop asked once per image, the timed
/// request's answers reused by the evaluation).
#[derive(Debug, Default)]
pub(super) struct AnswerCache {
    crops: Vec<(crate::verify::CropRect, Option<Vec<Prediction>>)>,
    frame: Option<Option<Vec<Prediction>>>,
}

/// [`super::served::Answers`] from the verifier model on the original frame (fail-open: no
/// verifier, no frame, an error or the evaluation's cap leaves a candidate unverified).
struct VerifierAnswers<'a, 'j> {
    verifier: Option<&'a mut Verifier<'j>>,
    frame: Option<&'a crate::image::RgbImage>,
    scene: Scene,
    cache: &'a mut AnswerCache,
    /// Verifier runs still allowed (None: no cap, the timed request).
    budget: Option<usize>,
    capped: bool,
}

impl VerifierAnswers<'_, '_> {
    /// One more verifier run allowed (the budget counted down).
    fn spend(&mut self) -> bool {
        match self.budget.as_mut() {
            Some(0) => {
                self.capped = true;
                false
            }
            Some(n) => {
                *n -= 1;
                true
            }
            None => true,
        }
    }
}

impl super::served::Answers for VerifierAnswers<'_, '_> {
    fn crop(&mut self, r: crate::verify::CropRect) -> Option<Vec<Prediction>> {
        if let Some((_, a)) = self.cache.crops.iter().find(|(c, _)| *c == r) {
            return a.clone();
        }
        if self.verifier.is_none() || self.frame.is_none() || !self.spend() {
            return None;
        }
        let (v, frame) = (self.verifier.as_deref_mut()?, self.frame?);
        let a = v.detect(crate::verify::crop(frame, r), self.scene).ok();
        self.cache.crops.push((r, a.clone()));
        a
    }

    fn frame(&mut self) -> Option<Vec<Prediction>> {
        if let Some(a) = &self.cache.frame {
            return a.clone();
        }
        if self.verifier.is_none() || self.frame.is_none() || !self.spend() {
            return None;
        }
        let (v, frame) = (self.verifier.as_deref_mut()?, self.frame?);
        let a = v.detect(frame.clone(), self.scene).ok();
        self.cache.frame = Some(a.clone());
        a
    }
}

/// One request after the served stages ([`Pipeline::serve`]): its stage times and production
/// predictions, and what the untimed evaluation ([`Pipeline::evaluate`]) post-processes again.
pub(super) struct Served {
    stages: [f64; 4],
    predictions: Vec<Prediction>,
    outputs: Vec<crate::model::NamedOutput>,
    ctx: crate::model::PreprocessCtx,
    scene: Scene,
    /// The night model answered it (As served).
    routed: bool,
    /// The frame before enhancement (for the evaluation's verifier checks).
    frame: Option<crate::image::RgbImage>,
    /// The verifier's answers of the timed request.
    answers: AnswerCache,
}

impl Pipeline<'_> {
    /// The frame before enhancement is needed (a verifier here, or on the night model a night /
    /// IR frame of `scene` goes to).
    fn needs_original(&self, scene: Scene) -> bool {
        match &self.night {
            Some(n) if scene.is_night() => n.needs_original(scene),
            _ => self.verifier.is_some(),
        }
    }

    /// The threshold of a frame of `scene` ([`crate::worker::LiveSettings::threshold_for_scene`]):
    /// `night_threshold` on night / IR frames when set, else the served one.
    fn threshold_for(&self, scene: Scene) -> f32 {
        match self.night_threshold {
            Some(t) if scene.is_night() && !self.fixed => t,
            _ => self.params.confidence_threshold,
        }
    }

    /// One request from the decoded frame on through the worker's stages, serially, with
    /// per-stage timings; its decode time is `frame.decode_ms` (measured where it was decoded,
    /// possibly the read-ahead thread). The set's `transform` (an image-set property, not a
    /// production stage) was applied after decode and is not part of any stage time.
    /// `original` is the frame before enhancement, for the verifier (None without one).
    ///
    /// A variant adds what the service does with the same options: the enhancement for the
    /// frame's scene counts as preprocess time; the threshold of the frame's scene, the filter
    /// rules, the object filter and the verifier ([`super::served::returned_at`]) count as
    /// postprocess time (the service adds the verifier to processMs). As served, a night / IR
    /// frame of a model with a night model runs this model's enhancement and resize (the
    /// service's prep pool does before routing; preprocess time), then the night model's
    /// stages on the original frame.
    fn serve(
        &mut self,
        frame: Decoded,
        original: Option<crate::image::RgbImage>,
    ) -> Result<Served> {
        let Decoded {
            mut img,
            decode_ms: decode,
            scene,
        } = frame;

        if scene.is_night()
            && let Some(night) = self.night.as_mut()
        {
            let t = Instant::now();
            let mut mine = img.clone();
            if let Some(v) = self.variant {
                crate::model::enhance::apply(&mut mine, v.modes().for_scene(scene));
            }
            let _ = self.pre.run(&mine.rgb, mine.width, mine.height)?;
            let before = ms(t.elapsed());
            let mut served = night.serve(
                Decoded {
                    img,
                    decode_ms: decode,
                    scene,
                },
                original,
            )?;
            served.stages[1] += before;
            served.routed = true;
            return Ok(served);
        }

        let t = Instant::now();
        if let Some(v) = self.variant {
            crate::model::enhance::apply(&mut img, v.modes().for_scene(scene));
        }
        let (chw, ctx) = self.pre.run(&img.rgb, img.width, img.height)?;
        let extra = self.family.extra_inputs(&ctx);
        let preprocess = ms(t.elapsed());

        let t = Instant::now();
        let outputs = self.backend.infer(chw, &extra)?;
        let infer = ms(t.elapsed());

        let t = Instant::now();
        let threshold = self.threshold_for(scene);
        let params = PostParams {
            confidence_threshold: threshold,
            nms_iou: self.params.nms_iou,
        };
        let dets = self.family.postprocess(&outputs, &ctx, &params)?;
        let mut answers = AnswerCache::default();
        let preds = match self.variant {
            None => crate::model::to_predictions(&dets, &ctx, self.classes, self.filter),
            Some(v) => {
                let (filters, verify) = variant_rules(v);
                let rules = super::served::PostRules {
                    classes: self.classes,
                    filters,
                    object_filter: self.filter,
                    verify,
                };
                let mut ask = VerifierAnswers {
                    verifier: self.verifier.as_mut(),
                    frame: original.as_ref(),
                    scene,
                    cache: &mut answers,
                    budget: None,
                    capped: false,
                };
                super::served::returned_at(&dets, &ctx, scene, threshold, &rules, &mut ask)
                    .predictions
            }
        };
        let postprocess = ms(t.elapsed());
        Ok(Served {
            stages: [decode, preprocess, infer, postprocess],
            predictions: preds,
            outputs,
            ctx,
            scene,
            routed: false,
            frame: original,
            answers,
        })
    }

    /// The benchmark's own work after a served request (untimed): with `eval`, the same outputs
    /// post-processed again at the evaluation threshold without the object filter; with a
    /// variant, every box the pipeline returns at some threshold of the sweep, with where
    /// ([`super::served::windows`], B2). A frame the night model answered is evaluated by it,
    /// its labels scored as this model's classes ([`night_labels`]).
    fn evaluate(&mut self, served: Served, eval: bool) -> Result<Sample> {
        #[cfg(test)]
        if let Some(d) = self.slow_eval.filter(|_| eval) {
            std::thread::sleep(d);
        }
        if served.routed
            && let Some(night) = self.night.as_mut()
        {
            let mut sample = night.evaluate(served, eval)?;
            night_labels(&ClassMap::for_model(self.classes), &mut sample.eval);
            sample.routed = true;
            return Ok(sample);
        }
        let Served {
            stages,
            predictions,
            outputs,
            ctx,
            scene,
            routed,
            frame,
            mut answers,
        } = served;
        let eval = if eval {
            let dets = self.family.postprocess(&outputs, &ctx, &self.eval_params)?;
            match self.variant {
                None => crate::model::to_predictions(&dets, &ctx, self.classes, None)
                    .iter()
                    .map(to_pred)
                    .collect(),
                Some(v) => {
                    let threshold = match (self.fixed, self.night_threshold) {
                        (true, _) => {
                            super::served::Threshold::Fixed(self.params.confidence_threshold)
                        }
                        (false, Some(t)) if scene.is_night() => super::served::Threshold::Fixed(t),
                        _ => super::served::Threshold::Sweep,
                    };
                    let (filters, verify) = variant_rules(v);
                    let rules = super::served::PostRules {
                        classes: self.classes,
                        filters,
                        object_filter: self.filter,
                        verify,
                    };
                    let mut ask = VerifierAnswers {
                        verifier: self.verifier.as_mut(),
                        frame: frame.as_ref(),
                        scene,
                        cache: &mut answers,
                        budget: Some(SWEEP_VERIFY_CAP),
                        capped: false,
                    };
                    let w = super::served::windows(
                        &dets,
                        &ctx,
                        scene,
                        threshold,
                        self.eval_params.confidence_threshold,
                        &rules,
                        &mut ask,
                    );
                    if ask.capped {
                        self.verify_capped += 1;
                    }
                    w
                }
            }
        } else {
            Vec::new()
        };
        Ok(Sample {
            stages,
            predictions,
            eval,
            scene,
            routed,
        })
    }
}

/// The filter rules and verifier settings of `v` ([`super::served::PostRules`]).
fn variant_rules(
    v: &PipelineVariant,
) -> (
    Option<&crate::model::filters::FilterRules>,
    Option<&crate::verify::VerifyConfig>,
) {
    (v.filters.as_ref(), v.verify.as_ref().map(|vv| &vv.config))
}

/// Labels of the night model's evaluation boxes the model's class map does not score, scored as
/// the class its ground truth maps them to (a night model's `car` is a `vehicle` of a model with
/// a `vehicle` class, as a `car` label is for its ground truth).
fn night_labels(map: &ClassMap, preds: &mut [PredBox]) {
    for p in preds {
        if map.pred_class(&p.label).is_none()
            && let Some(c) = map.gt_class(&p.label)
        {
            p.label = c.to_string();
        }
    }
}

/// What the timed runs of [`Bench::time`] collect.
pub(super) struct Timings {
    /// Per run: decode, preprocess, infer, postprocess, total (ms).
    samples: [Vec<f64>; 5],
    /// Per resolution bucket: pre (decode + preprocess), infer, post, total samples (ms) and
    /// the image count.
    by_res: BTreeMap<&'static str, ([Vec<f64>; 4], usize)>,
    per_image: Vec<ImageRun>,
    /// Per image: the last run's predictions at the evaluation threshold.
    eval_preds: Vec<Vec<PredBox>>,
    total_iters: usize,
    /// Time of the timed iterations (throughput): each run's wait for its frame plus its served
    /// stages, at least its frame's decode ([`iteration_time`]; B6).
    loop_time: Duration,
    /// The frames were read and decoded ahead ([`Bench::timed_runs_ahead`]).
    pipelined: bool,
    /// Images the night model answered (As served).
    routed: usize,
}

impl Timings {
    /// Add one run's stage timings (overall and to its resolution bucket); returns its total.
    fn record(&mut self, bucket: &'static str, sample: &Sample) -> f64 {
        let total: f64 = sample.stages.iter().sum();
        for (dst, v) in self.samples.iter_mut().zip(sample.stages) {
            dst.push(v);
        }
        self.samples[4].push(total);
        let e = self.by_res.entry(bucket).or_default();
        e.0[0].push(sample.stages[0] + sample.stages[1]);
        e.0[1].push(sample.stages[2]);
        e.0[2].push(sample.stages[3]);
        e.0[3].push(total);
        total
    }

    fn stage_stats(&self) -> StageStats {
        StageStats {
            decode: Stats::from_samples(&self.samples[0]),
            preprocess: Stats::from_samples(&self.samples[1]),
            infer: Stats::from_samples(&self.samples[2]),
            postprocess: Stats::from_samples(&self.samples[3]),
            total: Stats::from_samples(&self.samples[4]),
        }
    }

    /// Speed per resolution bucket with images, in [`images::RESOLUTION_BUCKETS`] order.
    fn by_resolution(&self) -> Vec<BucketSpeed> {
        images::RESOLUTION_BUCKETS
            .iter()
            .filter_map(|b| {
                let (s, n) = self.by_res.get(b)?;
                let t = Stats::from_samples(&s[3]);
                Some(BucketSpeed {
                    bucket: b.to_string(),
                    images: *n,
                    pre_p50: Stats::from_samples(&s[0]).p50,
                    infer_p50: Stats::from_samples(&s[1]).p50,
                    post_p50: Stats::from_samples(&s[2]).p50,
                    total_p50: t.p50,
                    total_p95: t.p95,
                })
            })
            .collect()
    }
}

/// The drill-down record of one image from its last run (stored predictions, most confident
/// first, at most `max_preds`; p50 timings) and that run's evaluation predictions.
pub(super) fn image_run(
    set: &ImageSet,
    img: &images::SetImage,
    last: &Sample,
    totals: &[f64],
    infers: &[f64],
    max_preds: usize,
) -> (ImageRun, Vec<PredBox>) {
    let mut preds: Vec<ImagePred> = last
        .predictions
        .iter()
        .map(|p| {
            let b = to_pred(p);
            ImagePred {
                label: b.label,
                confidence: b.confidence,
                bbox: b.bbox,
                tp: None,
            }
        })
        .collect();
    preds.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    preds.truncate(max_preds);
    let eval = last.eval.clone();
    let run = ImageRun {
        set: set.id.clone(),
        file: img.file.clone(),
        total_ms: Stats::from_samples(totals).p50,
        infer_ms: Stats::from_samples(infers).p50,
        preds,
        gt_matched: Vec::new(),
        counts: None,
        scene: Some(last.scene),
    };
    (run, eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(runtime: Runtime, actual: &str, spec: &str) -> crate::backend::device::DeviceInfo {
        crate::backend::device::DeviceInfo {
            actual: actual.into(),
            runtime,
            spec: spec.into(),
            ..Default::default()
        }
    }

    /// The cache status looks in the folder the compile writes to (`openvino-<class>-<prec>`,
    /// `tensorrt`, `coreml`), not the root: a new blob there is a miss, an unchanged folder a
    /// hit, one of another device or the root changes nothing, an empty one is unknown and a
    /// runtime without a cache says so.
    #[test]
    fn cache_status_watches_the_compiles_own_folder() {
        let root = std::env::temp_dir().join(format!("bop-cache-{}", uuid::Uuid::new_v4()));
        let cpu = crate::backend::device::ov_cache_dir(&root, "CPU", "f32");
        let gpu = crate::backend::device::ov_cache_dir(&root, "GPU.1", "f16");
        std::fs::create_dir_all(&cpu).unwrap();
        std::fs::create_dir_all(&gpu).unwrap();
        let ov_cpu = device(Runtime::OpenVino, "CPU", "openvino:cpu");
        let ov_gpu = device(Runtime::OpenVino, "GPU.0", "openvino:gpu.0");
        // A model's subfolder with only its bookkeeping (made before the compile, markers): no
        // compiled file, so neither a miss nor a hit.
        let sub = cpu.join("m-0123456789abcdef");
        std::fs::create_dir_all(&sub).unwrap();
        let before = cache_files(&root);
        std::fs::write(sub.join(crate::storage::caches::MODEL_FILE), b"/m.onnx\n").unwrap();
        std::fs::write(sub.join(crate::storage::caches::LAST_USED_FILE), b"1\n").unwrap();
        std::fs::write(cpu.join(crate::storage::caches::LAST_USED_FILE), b"1\n").unwrap();
        assert!(cache_files(&root).is_empty());
        assert_eq!(
            cache_status(&before, &cache_files(&root), &ov_cpu),
            "unknown"
        );
        // Its blob is the compile's: a miss, then a hit.
        std::fs::write(sub.join("77.blob"), b"x").unwrap();
        assert_eq!(cache_status(&before, &cache_files(&root), &ov_cpu), "miss");
        let before = cache_files(&root);
        assert_eq!(cache_status(&before, &cache_files(&root), &ov_cpu), "hit");
        std::fs::remove_dir_all(&sub).unwrap();
        assert_eq!(
            cache_scope(&ov_gpu),
            Some(CacheScope::Prefix("openvino-gpu-".into()))
        );
        assert_eq!(
            cache_scope(&ov_cpu),
            Some(CacheScope::Prefix("openvino-cpu-".into()))
        );
        // An empty folder: caching off or failing.
        let before = cache_files(&root);
        assert_eq!(
            cache_status(&before, &cache_files(&root), &ov_cpu),
            "unknown"
        );
        // The first compile writes a blob into the subfolder: a miss (the old flat listing of
        // the root saw only the folders and called it a hit).
        let before = cache_files(&root);
        std::fs::write(cpu.join("123.blob"), b"x").unwrap();
        let after = cache_files(&root);
        assert!(after.contains(&Path::new(cpu.file_name().unwrap()).join("123.blob")));
        assert_eq!(cache_status(&before, &after, &ov_cpu), "miss");
        // Unchanged: a hit.
        let before = cache_files(&root);
        assert_eq!(cache_status(&before, &cache_files(&root), &ov_cpu), "hit");
        // A blob of another precision of the same class is this compile's (CPU's hash ignores
        // the precision; the folder is chosen by the plugin's hint at compile time).
        let bf16 = crate::backend::device::ov_cache_dir(&root, "CPU", "bf16");
        std::fs::create_dir_all(&bf16).unwrap();
        std::fs::write(bf16.join("456.blob"), b"x").unwrap();
        assert_eq!(cache_status(&before, &cache_files(&root), &ov_cpu), "miss");
        // Another device's folder or a file at the root are not (`GPU.1`'s folder is every
        // GPU's: the class).
        let before = cache_files(&root);
        std::fs::write(gpu.join("789.blob"), b"x").unwrap();
        std::fs::write(root.join("stray.blob"), b"x").unwrap();
        let after = cache_files(&root);
        assert_eq!(cache_status(&before, &after, &ov_cpu), "hit");
        assert_eq!(cache_status(&before, &after, &ov_gpu), "miss");
        // ORT: TensorRT engines and the CoreML cache (nested) in their folders; CPU / CUDA /
        // DirectML keep none.
        let trt = device(Runtime::Ort, "TensorRT", "ort:tensorrt");
        let coreml = device(Runtime::Ort, "CoreML", "ort:coreml");
        let before = cache_files(&root);
        assert_eq!(cache_status(&before, &before, &trt), "unknown");
        std::fs::create_dir_all(root.join("tensorrt")).unwrap();
        std::fs::write(root.join("tensorrt/engine.engine"), b"x").unwrap();
        std::fs::create_dir_all(root.join("coreml/key/model.mlmodelc")).unwrap();
        std::fs::write(root.join("coreml/key/model.mlmodelc/weights.bin"), b"x").unwrap();
        let after = cache_files(&root);
        assert_eq!(cache_status(&before, &after, &trt), "miss");
        assert_eq!(cache_status(&before, &after, &coreml), "miss");
        assert_eq!(cache_status(&after, &after, &coreml), "hit");
        assert_eq!(cache_status(&after, &after, &ov_cpu), "hit");
        for spec in ["ort:cpu", "ort:cuda", "ort:directml"] {
            let d = device(Runtime::Ort, "x", spec);
            assert_eq!(cache_status(&before, &after, &d), "none", "{spec}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn timings_record_overall_and_by_bucket() {
        let mut t = Timings {
            samples: Default::default(),
            by_res: BTreeMap::new(),
            per_image: Vec::new(),
            eval_preds: Vec::new(),
            total_iters: 3,
            loop_time: Duration::ZERO,
            pipelined: false,
            routed: 0,
        };
        let sample = |stages: [f64; 4]| Sample {
            stages,
            predictions: Vec::new(),
            eval: Vec::new(),
            scene: Scene::Day,
            routed: false,
        };
        assert_eq!(t.record("sd", &sample([1.0, 2.0, 10.0, 0.5])), 13.5);
        assert_eq!(t.record("sd", &sample([1.0, 4.0, 20.0, 0.5])), 25.5);
        assert_eq!(t.record("fhd", &sample([3.0, 3.0, 30.0, 1.0])), 37.0);
        t.by_res.entry("sd").or_default().1 += 1;
        t.by_res.entry("fhd").or_default().1 += 1;
        let s = t.stage_stats();
        assert_eq!(s.infer, Stats::from_samples(&[10.0, 20.0, 30.0]));
        assert_eq!(s.total, Stats::from_samples(&[13.5, 25.5, 37.0]));
        let b = t.by_resolution();
        // Bucket order, empty buckets omitted.
        assert_eq!(
            b.iter().map(|b| b.bucket.as_str()).collect::<Vec<_>>(),
            ["sd", "fhd"]
        );
        assert_eq!(b[0].images, 1);
        assert_eq!(b[0].pre_p50, 3.0); // decode + preprocess, nearest rank of [3, 5]
        assert_eq!(b[0].infer_p50, 10.0);
        assert_eq!(b[0].total_p95, 25.5);
        assert_eq!(b[1].post_p50, 1.0);
    }

    #[test]
    fn cancelled_is_detected_through_context() {
        let e = anyhow::Error::from(Cancelled).context("loading x");
        assert!(is_cancelled(&e));
        assert!(!is_cancelled(&anyhow::anyhow!("other")));
    }
}
