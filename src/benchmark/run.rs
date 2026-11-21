//! The benchmark run itself: `Bench` compiles a model on a device, warms it up and times the
//! per-image pipeline (decode -> preprocess -> infer -> postprocess) at the configured and the
//! evaluation confidence. Progress and cancellation are reported through `Progress`/`Cancelled`.
//! One inference at a time; the runtimes mutex is held only while compiling.
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
//! the run, never outliving it. Warm-up stays serial (it repeats one frame).
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
            .finish()
    }
}

/// One request's timings (decode, preprocess, infer, postprocess; ms).
pub(super) struct Sample {
    pub(super) stages: [f64; 4],
    pub(super) predictions: Vec<Prediction>,
    pub(super) eval: Vec<Prediction>,
}

/// Folder levels below the cache root [`cache_files`] lists: OpenVINO blobs are one level down
/// (`openvino-<device>-<precision>/<hash>.blob`), ORT's CoreML model cache a few more
/// (`coreml/<COREML_CACHE_KEY>/...`).
const CACHE_DEPTH: usize = 4;

/// Every file under `dir` (down to [`CACHE_DEPTH`] folders), as paths relative to it.
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
                    if let Ok(rel) = path.strip_prefix(root) {
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

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    fn report(
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
            let compiled = lock(runtimes).compile(cand, &req)?;
            compiled.into_backend()
        });
        // A cancel during the compiles is a cancel, not a load failure (the error of several
        // failed candidates no longer carries `Cancelled`).
        self.check_cancel()?;
        let backend = loaded.with_context(|| format!("loading {}", job.path.display()))?;
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
        let verifier = match job.variant.as_ref().and_then(|v| v.verify.as_ref()) {
            Some(v) => Some(self.load_verifier(runtimes, &v.job, &backend.info().device.spec)?),
            None => None,
        };
        self.time(job, requested, backend, verifier, compile_ms, cache)
    }

    /// Compile the verifier of a `verify:<model>` variant on `device` (the spec the verified
    /// model loaded on; falls back like the service would).
    fn load_verifier(
        &self,
        runtimes: &Mutex<Runtimes>,
        vjob: &Job,
        device: &str,
    ) -> Result<Backend> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &vjob.path);
        let req = LoadRequest {
            path: vjob.path.clone(),
            requested: device.to_string(),
            gpu_precision: vjob.gpu_precision.clone(),
            no_cache: false,
        };
        let label = format!("verifier {}", vjob.name);
        let loaded = crate::backend::try_candidates(&label, &candidates, |cand| {
            self.check_cancel()?;
            let compiled = lock(runtimes).compile(cand, &req)?;
            compiled.into_backend()
        });
        self.check_cancel()?;
        loaded.with_context(|| format!("loading the verifier {}", vjob.path.display()))
    }

    /// Warm up and time `backend` over every image, then score its detections.
    fn time(
        &self,
        job: &Job,
        requested: &str,
        backend: Backend,
        verifier: Option<Backend>,
        compile_ms: f64,
        cache: String,
    ) -> Result<RunResult> {
        let info = backend.info();
        let dev = info.device.clone();
        let (in_w, in_h) = info.input_size;
        let family =
            crate::model::make_family(job.family, &info.inputs, &info.outputs, job.classes.len())?;
        let threshold = job
            .confidence_threshold
            .unwrap_or(self.params.confidence_threshold);
        let mut pipe = Pipeline {
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
            variant: job.variant.as_ref(),
            verifier: match (
                verifier,
                job.variant.as_ref().and_then(|v| v.verify.as_ref()),
            ) {
                (Some(b), Some(v)) => Some(Verifier::new(b, v, self.params)?),
                _ => None,
            },
        };

        let all: Vec<(&ImageSet, &images::SetImage)> = self
            .sets
            .iter()
            .flat_map(|s| s.images.iter().map(move |i| (s, i)))
            .collect();
        let warmup_ms = self.warm_up(job, requested, &all, &mut pipe)?;
        let repeat = self.repeat.max(1);
        let loaded_on = if dev.spec.is_empty() {
            dev.actual.as_str()
        } else {
            dev.spec.as_str()
        };
        let mut timed = if self.read_ahead.applies(loaded_on) {
            self.timed_runs_ahead(job, requested, &all, &mut pipe, repeat)?
        } else {
            self.timed_runs(
                job,
                requested,
                &all,
                &mut pipe,
                repeat,
                &mut serial_frames(&all),
            )?
        };
        let loop_s = timed.loop_time.as_secs_f64();

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
            tag_recall,
            grades: Some(grades),
            agreement: None,
            eval_preds: {
                let map = ClassMap::for_model(&job.classes);
                timed.eval_preds.iter().map(|v| map.map_preds(v)).collect()
            },
            per_image: timed.per_image,
        })
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

    /// The untimed warm-up iterations on the first image; returns their wall time (ms).
    fn warm_up(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
    ) -> Result<f64> {
        let t = Instant::now();
        self.report(job, requested, Phase::Warmup, 0, self.warmup, "");
        if self.warmup > 0 {
            let (set, img) = all.first().context("no benchmark images")?;
            let first = img.bytes()?;
            for i in 0..self.warmup {
                self.check_cancel()?;
                pipe.run(&first, false, set.transform)
                    .context("warm-up inference")?;
                self.report(job, requested, Phase::Warmup, i + 1, self.warmup, "");
            }
        }
        Ok(ms(t.elapsed()))
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
                        &mut serial_frames(all),
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
            let timed = self.timed_runs(job, requested, all, pipe, repeat, &mut source);
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
    /// order (image by image, `repeat` per image; its argument is the image's index in `all`).
    fn timed_runs(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
        repeat: usize,
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
                let sample = frame
                    .and_then(|d| pipe.run_decoded(d, r + 1 == repeat))
                    .with_context(|| format!("image {}", img.file))?;
                timed.loop_time += t.elapsed();
                let total = timed.record(bucket, &sample);
                totals.push(total);
                infers.push(sample.stages[2]);
                last = Some(sample);
                done += 1;
                self.report(job, requested, Phase::Timing, done, total_iters, &img.file);
            }
            timed.by_res.entry(bucket).or_default().1 += 1;
            let last = last.context("no timed run (repeat must be >= 1)")?;
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

/// One timed run's decoded frame, or why it did not decode (a frame source's `Err` = the image
/// could not be read).
pub(super) type Frame = Result<Decoded>;

/// A decoded frame (the set's transform applied) and its decode time.
pub(super) struct Decoded {
    img: crate::image::RgbImage,
    /// Decode only, without reading the file or the transform (ms).
    decode_ms: f64,
}

/// Decode `bytes` (timed) and apply the set's `transform` (not part of any stage time).
fn decode(bytes: &[u8], transform: Option<images::ImageTransform>) -> Result<Decoded> {
    let t = Instant::now();
    let mut img = crate::image::decode(bytes)?;
    let decode_ms = ms(t.elapsed());
    if let Some(tr) = transform {
        tr.apply(&mut img);
    }
    Ok(Decoded { img, decode_ms })
}

/// The serial frame source: each image read once (cached across its runs), decoded per run
/// on the calling thread.
fn serial_frames<'s>(
    all: &'s [(&ImageSet, &images::SetImage)],
) -> impl FnMut(usize) -> Result<Frame> + 's {
    let mut cached: Option<(usize, std::borrow::Cow<'static, [u8]>)> = None;
    move |i| {
        let (set, img) = all.get(i).context("frame index out of range")?;
        let bytes = match cached.take() {
            Some((at, b)) if at == i => b,
            _ => img.bytes()?,
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

/// One compiled model with the production per-request stages (the code `worker.rs` runs).
pub(super) struct Pipeline<'j> {
    backend: Backend,
    family: Box<dyn crate::model::Family>,
    pre: Preprocessor,
    params: PostParams,
    eval_params: PostParams,
    classes: &'j [String],
    filter: Option<&'j [String]>,
    /// Enhancement and filter rules on top of the plain stages ([`super::variant`]).
    variant: Option<&'j PipelineVariant>,
    /// The verifier of a `verify:<model>` variant.
    verifier: Option<Verifier<'j>>,
}

/// The second model of a `verify:<model>` variant: the stages the verifier's worker runs on a
/// verifier check (`worker::Purpose::Verify`): its own enhancement for the frame's scene, the
/// resize, inference, and its raw detections at `min_confidence` (no night rules, no object
/// filter).
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
    fn detect(
        &mut self,
        mut img: crate::image::RgbImage,
        scene: crate::image_stats::Scene,
    ) -> Result<Vec<Prediction>> {
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

    /// Check the candidates of `preds` ([`crate::verify::plan`]) on `frame` (the original,
    /// not enhanced) and drop the rejected ones from `preds` and from `eval` (the same boxes
    /// at the evaluation threshold; candidates are taken from `preds`, so evaluation boxes
    /// below the production threshold stay unverified).
    fn apply(
        &mut self,
        frame: crate::image::RgbImage,
        scene: crate::image_stats::Scene,
        preds: Vec<Prediction>,
        eval: &mut Vec<Prediction>,
    ) -> Result<Vec<Prediction>> {
        use crate::verify::VerifyMode;
        let cfg = &self.variant.config;
        let plan = crate::verify::plan(&preds, frame.width, frame.height, scene, cfg);
        if plan.is_empty() {
            return Ok(preds);
        }
        let found: Vec<Option<Vec<Prediction>>> = match cfg.mode {
            VerifyMode::Crop => {
                let mut out = Vec::with_capacity(plan.checks.len());
                for c in &plan.checks {
                    out.push(match c.crop {
                        Some(r) => Some(self.detect(crate::verify::crop(&frame, r), scene)?),
                        None => None,
                    });
                }
                out
            }
            VerifyMode::Frame => vec![Some(self.detect(frame, scene)?)],
        };
        let outcomes = crate::verify::outcomes(&plan, &found, &self.variant.config);
        let rejected: Vec<Prediction> = outcomes
            .iter()
            .filter(|(_, o)| *o == crate::verify::Outcome::Rejected)
            .filter_map(|(i, _)| preds.get(*i).cloned())
            .collect();
        eval.retain(|e| !rejected.contains(e));
        Ok(crate::verify::apply(preds, &outcomes).0)
    }
}

impl Pipeline<'_> {
    /// One request through the worker's stages, serially, with per-stage timings; with `eval`,
    /// the same outputs are post-processed again at the evaluation threshold (untimed, without
    /// the object filter). The set's `transform` (an image-set property, not a production
    /// stage) is applied after decode and is not part of any stage time.
    ///
    /// A variant adds what the service does with the same options: the scene statistics of the
    /// (transformed) frame and its enhancement count as preprocess time, its filter rules run
    /// on the family's detections (postprocess time) and on the evaluation detections too, so
    /// AP, false positives and the threshold sweep see the filtered boxes.
    fn run(
        &mut self,
        bytes: &[u8],
        eval: bool,
        transform: Option<images::ImageTransform>,
    ) -> Result<Sample> {
        self.run_decoded(decode(bytes, transform)?, eval)
    }

    /// [`Self::run`] from the decoded frame on: its decode time is `frame.decode_ms` (measured
    /// where it was decoded, possibly the read-ahead thread).
    fn run_decoded(&mut self, frame: Decoded, eval: bool) -> Result<Sample> {
        let Decoded {
            mut img,
            decode_ms: decode,
        } = frame;
        // The verifier crops the original frame, as the service does (not timed: the service
        // keeps the decoded frame without a copy).
        let original = self.verifier.is_some().then(|| img.clone());

        let t = Instant::now();
        let scene = match self.variant {
            Some(v) => {
                let scene = crate::image_stats::ImageStats::compute(&img).scene();
                crate::model::enhance::apply(&mut img, v.modes().for_scene(scene));
                scene
            }
            None => crate::image_stats::Scene::Day,
        };
        let (chw, ctx) = self.pre.run(&img.rgb, img.width, img.height)?;
        let extra = self.family.extra_inputs(&ctx);
        let preprocess = ms(t.elapsed());

        let t = Instant::now();
        let outputs = self.backend.infer(chw, &extra)?;
        let infer = ms(t.elapsed());

        let rules = self.variant.and_then(|v| v.filters.as_ref());
        let filtered = |dets: Vec<crate::model::Detection>| match rules {
            Some(r) => {
                crate::model::filters::apply(dets, self.classes, ctx.orig_w, ctx.orig_h, scene, r).0
            }
            None => dets,
        };
        let t = Instant::now();
        let dets = filtered(self.family.postprocess(&outputs, &ctx, &self.params)?);
        let preds = crate::model::to_predictions(&dets, &ctx, self.classes, self.filter);
        let postprocess = ms(t.elapsed());
        let mut eval = if eval {
            let dets = filtered(self.family.postprocess(&outputs, &ctx, &self.eval_params)?);
            crate::model::to_predictions(&dets, &ctx, self.classes, None)
        } else {
            Vec::new()
        };
        // The verifier cascade counts as postprocess time (the service adds it to processMs).
        let (preds, postprocess) = match (self.verifier.as_mut(), original) {
            (Some(v), Some(frame)) => {
                let t = Instant::now();
                let preds = v.apply(frame, scene, preds, &mut eval)?;
                (preds, postprocess + ms(t.elapsed()))
            }
            _ => (preds, postprocess),
        };
        Ok(Sample {
            stages: [decode, preprocess, infer, postprocess],
            predictions: preds,
            eval,
        })
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
    /// Wall time of the timed iterations (throughput): with read ahead, each run's wait for its
    /// frame plus its stages on the inference thread.
    loop_time: Duration,
    /// The frames were read and decoded ahead ([`Bench::timed_runs_ahead`]).
    pipelined: bool,
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
    let eval = last.eval.iter().map(to_pred).collect();
    let run = ImageRun {
        set: set.id.clone(),
        file: img.file.clone(),
        total_ms: Stats::from_samples(totals).p50,
        infer_ms: Stats::from_samples(infers).p50,
        preds,
        gt_matched: Vec::new(),
        counts: None,
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
        };
        let sample = |stages: [f64; 4]| Sample {
            stages,
            predictions: Vec::new(),
            eval: Vec::new(),
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
