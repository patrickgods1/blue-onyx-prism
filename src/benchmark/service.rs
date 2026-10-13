//! The server's benchmark runner: one run at a time on a background `std::thread`, with
//! progress, ETA, partial results and cancellation, polled by the web UI through
//! `GET /v1/benchmark`.
//!
//! A run: resolve the datasets (downloading missing built-in sets first when
//! `benchmark.auto_download_datasets` is on), compute pseudo ground truth for datasets without
//! ground truth (reference model on a CPU device, or on one of the chosen devices when the run
//! limits them), then sweep every model over the chosen devices, and merge the results into
//! `benchmark.json`.
//!
//! The run uses the registry's shared [`Runtimes`] when there is one (same OpenVINO `Core`,
//! same ONNX Runtime library: a process can load only one ONNX Runtime flavor), compiling under
//! its lock like the workers do and running inference on the benchmark thread. Without shared
//! runtimes (registries built from handles, in tests) it initializes its own.
//!
//! The service lives for the whole process (across registry generations); the runner cancels a
//! running benchmark when a generation stops.

use super::grade::{AccuracyMetric, Weights};
use super::images::{self, ImageSet};
use super::report::{HardwareSummary, RuntimeVersions};
use super::threshold::Objective;
use super::{
    Bench, BenchmarkResults, ModelResult, Phase, Progress, SweepOptions, configured_device,
    find_model, is_cancelled, job_for_config, no_pseudo_gt_device, pick_reference,
    pseudo_ground_truth, pseudo_gt_device, sweep,
};
use crate::backend::spec::{self, Device, DeviceSpec};
use crate::backend::{CoreOptions, Runtimes};
use crate::config::{BenchmarkConfig, Config, DatasetRef};
use crate::model::PostParams;
use crate::sync::LockExt;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub const MAX_REPEAT: usize = 1000;
pub const MAX_WARMUP: usize = 100;
/// Images one run benchmarks at most, over all its datasets (BM-11: every device run keeps
/// per-image results and evaluation predictions until its model is saved). `max_images` above
/// it is refused; `max_images = 0` ("all") and several datasets are cut to it, spread over the
/// datasets in proportion to their size (at least one image each), with a run warning.
pub const MAX_IMAGES_PER_RUN: usize = 2000;

/// What to benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    /// Config model names; empty = the enabled models whose files are present.
    pub models: Vec<String>,
    /// Device specs; empty = every runnable option of each model.
    pub devices: Vec<String>,
    pub datasets: Vec<DatasetRef>,
    /// Images per dataset (0 = all); a run takes at most [`MAX_IMAGES_PER_RUN`] in total.
    pub max_images: usize,
    /// Timed runs per image.
    pub repeat: usize,
    pub warmup: usize,
    /// Pseudo-ground-truth model (None = automatic).
    pub reference_model: Option<String>,
    pub weights: Weights,
    /// What the best confidence threshold optimizes.
    #[serde(default)]
    pub threshold_objective: Objective,
    /// What the accuracy grade is computed from.
    #[serde(default)]
    pub accuracy_metric: AccuracyMetric,
}

impl RunRequest {
    /// The configured defaults (`benchmark` in the config).
    pub fn defaults(b: &BenchmarkConfig) -> Self {
        Self {
            models: b.models.clone(),
            devices: b.devices.clone(),
            datasets: b.datasets.clone(),
            max_images: b.max_images_per_dataset,
            repeat: b.repeat_per_image.max(1),
            warmup: b.warmup,
            reference_model: b.reference_model.clone(),
            weights: b.weights,
            threshold_objective: b.threshold_objective,
            accuracy_metric: b.accuracy_metric,
        }
    }

    /// Parse form fields over the configured defaults: `model`, `device`, `dataset` (repeated:
    /// a given field replaces the default list), `max_images`, `repeat`, `warmup`,
    /// `reference_model` ("" = automatic), `accuracy_weight` (0..1), `threshold_objective`
    /// (`f1`, `f2`, `precision:<p>`, `recall:<r>`, `youden`, `fpr:<x>`), `accuracy_metric`
    /// (`ap50`, `roc_auc`).
    pub fn from_form(form: &[(String, String)], defaults: &BenchmarkConfig) -> Result<Self> {
        let has = |k: &str| form.iter().any(|(key, _)| key == k);
        let all = |k: &str| -> Vec<String> {
            form.iter()
                .filter(|(key, _)| key == k)
                .map(|(_, v)| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        };
        let one = |k: &str| form.iter().find(|(key, _)| key == k).map(|(_, v)| v.trim());
        let num = |k: &str, default: usize| -> Result<usize> {
            match one(k) {
                None | Some("") => Ok(default),
                Some(v) => v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("{k}: '{v}' is not a number")),
            }
        };
        let mut req = Self::defaults(defaults);
        if has("model") {
            req.models = all("model");
        }
        if has("device") {
            req.devices = all("device");
        }
        if has("dataset") {
            req.datasets = all("dataset").into_iter().map(DatasetRef::Id).collect();
        }
        req.max_images = num("max_images", req.max_images)?;
        req.repeat = num("repeat", req.repeat)?;
        req.warmup = num("warmup", req.warmup)?;
        if let Some(r) = one("reference_model") {
            req.reference_model = (!r.is_empty()).then(|| r.to_string());
        }
        if let Some(w) = one("accuracy_weight").filter(|w| !w.is_empty()) {
            let a: f64 = w
                .parse()
                .map_err(|_| anyhow::anyhow!("accuracy_weight: '{w}' is not a number"))?;
            if !(0.0..=1.0).contains(&a) {
                bail!("accuracy_weight: must be between 0 and 1");
            }
            req.weights = Weights {
                accuracy: a,
                speed: 1.0 - a,
            };
        }
        if let Some(o) = one("threshold_objective").filter(|o| !o.is_empty()) {
            req.threshold_objective = o.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        }
        if let Some(m) = one("accuracy_metric").filter(|m| !m.is_empty()) {
            req.accuracy_metric = m.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        }
        req.validate()?;
        Ok(req)
    }

    pub fn validate(&self) -> Result<()> {
        if !(1..=MAX_REPEAT).contains(&self.repeat) {
            bail!("repeat: must be 1-{MAX_REPEAT}");
        }
        if self.warmup > MAX_WARMUP {
            bail!("warmup: must be 0-{MAX_WARMUP}");
        }
        if self.max_images > MAX_IMAGES_PER_RUN {
            bail!(
                "max_images: must be 0-{MAX_IMAGES_PER_RUN} (0 = all images of each dataset, \
                 still at most {MAX_IMAGES_PER_RUN} per run)"
            );
        }
        if self.datasets.is_empty() {
            bail!("choose at least one dataset");
        }
        self.parsed_devices()?;
        Ok(())
    }

    /// The device filter: None = all runnable.
    pub fn parsed_devices(&self) -> Result<Option<Vec<Device>>> {
        parse_devices(&self.devices)
    }

    /// Save as the configured defaults.
    pub fn to_config(&self, b: &mut BenchmarkConfig) {
        b.models = self.models.clone();
        b.devices = self.devices.clone();
        b.datasets = self.datasets.clone();
        b.max_images_per_dataset = self.max_images;
        b.repeat_per_image = self.repeat;
        b.warmup = self.warmup;
        b.reference_model = self.reference_model.clone();
        b.weights = self.weights;
        b.threshold_objective = self.threshold_objective;
        b.accuracy_metric = self.accuracy_metric;
    }
}

/// Device specs to devices (`auto` is refused): None for an empty list.
pub fn parse_devices(devices: &[String]) -> Result<Option<Vec<Device>>> {
    if devices.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::new();
    for d in devices {
        match spec::parse(d).map_err(|e| anyhow::anyhow!("device '{d}': {e}"))? {
            DeviceSpec::Auto => bail!("device 'auto' is not a single device; pick devices"),
            DeviceSpec::Device(dev) => {
                if !out.contains(&dev) {
                    out.push(dev)
                }
            }
        }
    }
    Ok(Some(out))
}

/// State of the latest run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    Idle,
    Running,
    Done,
    Cancelled,
    Failed,
}

/// Live progress of a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressView {
    /// `download`, `pseudo-gt` or `benchmark`.
    pub stage: String,
    pub model: String,
    /// 1-based.
    pub model_index: usize,
    pub model_count: usize,
    pub device: String,
    /// `loading`, `warmup`, `timing`.
    pub phase: String,
    /// Iterations done / total in this phase.
    pub done: usize,
    pub total: usize,
    /// Image being processed.
    pub image: String,
    /// Device runs finished in the whole request (all models).
    pub runs_done: usize,
    pub runs_total: usize,
    /// Overall progress, 0..100.
    pub percent: f64,
    /// Estimated seconds left.
    pub eta_secs: Option<f64>,
    /// Free text ("downloading coco-cctv 42%").
    pub text: String,
}

/// Snapshot for `GET /v1/benchmark`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub state: RunState,
    pub running: bool,
    pub request: Option<RunRequest>,
    pub progress: Option<ProgressView>,
    /// Results of this run so far (finished models plus the one in progress).
    pub models: Vec<ModelResult>,
    pub message: Option<String>,
    pub error: Option<String>,
    /// Problems that did not stop the run (datasets skipped, pseudo ground truth failed).
    pub warnings: Vec<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
}

/// Refused start: a run is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a benchmark is already running")]
pub struct Busy;

#[derive(Debug)]
struct Inner {
    state: RunState,
    request: Option<RunRequest>,
    progress: Option<ProgressView>,
    done: Vec<ModelResult>,
    current: Option<ModelResult>,
    message: Option<String>,
    error: Option<String>,
    warnings: Vec<String>,
    started_at: Option<String>,
    finished_at: Option<String>,
    /// Bumped per run, so a finished thread of an old run cannot overwrite a newer one.
    run_id: u64,
}

/// The process-wide benchmark runner.
#[derive(Debug)]
pub struct BenchmarkService {
    inner: Mutex<Inner>,
    cancel: AtomicBool,
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl BenchmarkService {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                state: RunState::Idle,
                request: None,
                progress: None,
                done: Vec::new(),
                current: None,
                message: None,
                error: None,
                warnings: Vec::new(),
                started_at: None,
                finished_at: None,
                run_id: 0,
            }),
            cancel: AtomicBool::new(false),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock_recover()
    }

    pub fn status(&self) -> Status {
        let g = self.lock();
        let mut models = g.done.clone();
        models.extend(g.current.clone());
        Status {
            state: g.state,
            running: g.state == RunState::Running,
            request: g.request.clone(),
            progress: g.progress.clone(),
            models,
            message: g.message.clone(),
            error: g.error.clone(),
            warnings: g.warnings.clone(),
            started_at: g.started_at.clone(),
            finished_at: g.finished_at.clone(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.lock().state == RunState::Running
    }

    /// Ask a running benchmark to stop (it stops at the next iteration or device). Returns
    /// whether one was running.
    pub fn cancel(&self) -> bool {
        let running = self.is_running();
        if running {
            self.cancel.store(true, Ordering::SeqCst);
        }
        running
    }

    /// Start `job` on a new thread unless a run is in progress. `job` returns the final message
    /// (an error ending in [`super::Cancelled`] marks the run cancelled).
    pub fn start_job<F>(self: &Arc<Self>, request: RunRequest, job: F) -> Result<(), Busy>
    where
        F: FnOnce(&RunHandle) -> Result<String> + Send + 'static,
    {
        let run_id = {
            let mut g = self.lock();
            if g.state == RunState::Running {
                return Err(Busy);
            }
            g.run_id += 1;
            g.state = RunState::Running;
            g.request = Some(request);
            g.progress = None;
            g.done.clear();
            g.current = None;
            g.message = None;
            g.error = None;
            g.warnings.clear();
            g.started_at = Some(now());
            g.finished_at = None;
            g.run_id
        };
        self.cancel.store(false, Ordering::SeqCst);
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("benchmark".into())
            .spawn(move || {
                let handle = RunHandle {
                    svc: me.clone(),
                    run_id,
                };
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&handle)));
                let mut g = me.lock();
                if g.run_id != run_id {
                    return;
                }
                g.finished_at = Some(now());
                g.progress = None;
                match outcome {
                    Ok(Ok(msg)) => {
                        tracing::info!("benchmark finished: {msg}");
                        g.state = RunState::Done;
                        g.message = Some(msg);
                    }
                    Ok(Err(e)) if is_cancelled(&e) => {
                        tracing::info!("benchmark cancelled");
                        g.state = RunState::Cancelled;
                        g.message = Some(format!("{e:#}"));
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("benchmark failed: {e:#}");
                        g.state = RunState::Failed;
                        g.error = Some(format!("{e:#}"));
                    }
                    Err(_) => {
                        tracing::error!("benchmark thread panicked");
                        g.state = RunState::Failed;
                        g.error = Some("the benchmark thread panicked (see the log)".into());
                    }
                }
            });
        if let Err(e) = spawned {
            let mut g = self.lock();
            g.state = RunState::Failed;
            g.error = Some(format!("could not start the benchmark thread: {e}"));
        }
        Ok(())
    }

    /// Start a real benchmark of `request` (see [`run`]).
    pub fn start(self: &Arc<Self>, ctx: RunContext, request: RunRequest) -> Result<(), Busy> {
        let req = request.clone();
        self.start_job(request, move |h| run(&ctx, &req, h))
    }

    /// Wait until no run is in progress (tests, shutdown). Returns false on timeout.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while self.is_running() {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }
}

/// The running job's view of the service.
pub struct RunHandle {
    svc: Arc<BenchmarkService>,
    run_id: u64,
}

impl std::fmt::Debug for RunHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunHandle")
            .field("run_id", &self.run_id)
            .finish_non_exhaustive()
    }
}

impl RunHandle {
    pub fn cancel_flag(&self) -> &AtomicBool {
        &self.svc.cancel
    }

    pub fn is_cancelled(&self) -> bool {
        self.svc.cancel.load(Ordering::Relaxed)
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> Option<R> {
        let mut g = self.svc.lock();
        (g.run_id == self.run_id).then(|| f(&mut g))
    }

    pub fn set_progress(&self, p: ProgressView) {
        self.with(|g| g.progress = Some(p));
    }

    /// The model in progress (partial device list).
    pub fn set_current(&self, m: ModelResult) {
        self.with(|g| g.current = Some(m));
    }

    pub fn warn(&self, w: String) {
        tracing::warn!("benchmark: {w}");
        self.with(|g| g.warnings.push(w));
    }

    /// A finished model.
    pub fn push_done(&self, m: ModelResult) {
        self.with(|g| {
            g.current = None;
            g.done.push(m);
        });
    }
}

/// Everything a run needs from the server.
pub struct RunContext {
    /// The in-memory config (models, thresholds, cache dir, data root).
    pub config: Config,
    /// Where results are merged and saved (`results_path(config_path)`).
    pub results_path: PathBuf,
    /// The registry's runtimes; None = initialize dedicated ones.
    pub runtimes: Option<Arc<Mutex<Runtimes>>>,
    /// Normalized model name -> device spec the model is loaded on now.
    pub loaded: HashMap<String, String>,
    /// Download manager access (auto-download of built-in datasets).
    pub resources: Option<crate::resources::status::ResourcesCtx>,
}

impl std::fmt::Debug for RunContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunContext")
            .field("results_path", &self.results_path)
            .field("runtimes", &self.runtimes.is_some())
            .field("loaded", &self.loaded)
            .field("resources", &self.resources.is_some())
            .finish_non_exhaustive()
    }
}

/// Default models of a run: the enabled config models whose files are present.
pub fn default_models(config: &Config) -> Vec<String> {
    config
        .enabled_models()
        .filter(|m| config.data_path(&m.path).is_file())
        .map(|m| m.effective_name())
        .collect()
}

/// Queue the missing built-in datasets on the download manager and wait for them (progress in
/// the run's status). Returns warnings for sets that could not be downloaded.
fn download_datasets(ctx: &RunContext, ids: &[String], h: &RunHandle) -> Result<Vec<String>> {
    let mut warnings = Vec::new();
    let Some(rctx) = &ctx.resources else {
        warnings.push(format!(
            "datasets {} are not downloaded and this server has no download manager \
             (run `blue-onyx-prism fetch --resource bench:<id>`)",
            ids.join(", ")
        ));
        return Ok(warnings);
    };
    let mut keys: Vec<(String, String)> = Vec::new();
    for id in ids {
        let rid = format!("{}{id}", crate::resources::catalog::BENCH_ID_PREFIX);
        match crate::resources::actions::download(rctx, &ctx.config, &rid, false) {
            Ok(msg) => {
                tracing::info!("benchmark: {msg}");
                if let Some(r) = crate::resources::catalog::bench_set(id) {
                    let dir = crate::resources::status::target_dir(&ctx.config, r);
                    keys.push((id.clone(), crate::resources::Job::into_dir(r, dir).key()));
                }
            }
            Err(e) => warnings.push(format!("dataset {id}: {e}")),
        }
    }
    let manager = rctx.provisioner.manager_for(&ctx.config);
    loop {
        if h.is_cancelled() {
            return Err(super::Cancelled.into());
        }
        let mut pending = Vec::new();
        let (mut bytes, mut total) = (0u64, 0u64);
        for (id, key) in &keys {
            match manager.status(key).map(|s| s.state) {
                Some(crate::resources::State::Installed) => {}
                Some(crate::resources::State::Failed { error, .. }) => {
                    warnings.push(format!("dataset {id}: download failed: {error}"));
                    pending.push(None);
                }
                Some(crate::resources::State::Downloading { bytes: b, total: t }) => {
                    bytes += b;
                    total += t;
                    pending.push(Some(id.clone()));
                }
                _ => pending.push(Some(id.clone())),
            }
        }
        let waiting: Vec<String> = pending.into_iter().flatten().collect();
        if waiting.is_empty() {
            break;
        }
        let pct = if total > 0 {
            bytes as f64 * 100.0 / total as f64
        } else {
            0.0
        };
        h.set_progress(ProgressView {
            stage: "download".into(),
            percent: 0.0,
            text: format!("downloading datasets {} ({pct:.0}%)", waiting.join(", ")),
            ..Default::default()
        });
        // Failed jobs are reported once and dropped from the wait.
        keys.retain(|(id, _)| waiting.contains(id));
        std::thread::sleep(Duration::from_millis(250));
    }
    Ok(warnings)
}

/// Benchmark the requested models on the requested devices over the requested datasets,
/// publishing progress and partial results, then merge the finished models into the results
/// file (also when cancelled).
pub fn run(ctx: &RunContext, req: &RunRequest, h: &RunHandle) -> Result<String> {
    let devices = req.parsed_devices()?;
    let mut sets = resolve_sets(ctx, req, h)?;
    let runtimes = run_runtimes(ctx)?;
    let versions = RuntimeVersions::of(&runtimes.lock_recover());
    let run = ServiceRun {
        ctx,
        req,
        h,
        runtimes,
        devices,
        params: PostParams {
            confidence_threshold: ctx.config.confidence_threshold,
            nms_iou: ctx.config.nms_iou,
        },
        cache_dir: ctx.config.cache_dir_path(),
    };
    let image_count: usize = sets.iter().map(|s| s.images.len()).sum();

    // Pseudo ground truth for datasets without ground truth.
    if sets.iter().any(|s| !s.annotated()) {
        run.pseudo_ground_truth(&mut sets)?;
    }

    let names: Vec<String> = if req.models.is_empty() {
        default_models(&ctx.config)
    } else {
        req.models.clone()
    };
    if names.is_empty() {
        bail!("no model to benchmark: enable a model whose file is present, or choose models");
    }
    let entries: Vec<(String, Option<crate::config::ModelConfig>)> = names
        .iter()
        .map(|n| (n.clone(), find_model(&ctx.config, n).cloned()))
        .collect();
    let overall = OverallProgress {
        runs_done: std::sync::atomic::AtomicUsize::new(0),
        runs_total: run.runs_total(&entries),
        model_count: entries.len(),
        started: Instant::now(),
    };
    // Summaries of the finished models (the full results are saved as each model finishes and
    // dropped: BM-11).
    let mut finished: Vec<ModelResult> = Vec::new();
    let mut outcome: Result<()> = Ok(());
    let mut saver = Saver {
        hardware: HardwareSummary::current(),
        versions,
        retain_sets: None,
        finalized: false,
    };

    for (mi, (name, entry)) in entries.iter().enumerate() {
        if h.is_cancelled() {
            outcome = Err(super::Cancelled.into());
            break;
        }
        match run.bench_model(&sets, &overall, mi, name, entry.as_ref()) {
            Ok(r) => {
                let last = mi + 1 == entries.len();
                finished.push(r.summary());
                run.save(&mut saver, &sets, r, last)?;
            }
            Err(e) => {
                outcome = Err(e);
                break;
            }
        }
    }

    if !finished.is_empty() && !saver.finalized {
        run.finalize(&saver)?;
    }
    outcome?;
    Ok(run_summary(
        &finished,
        image_count,
        &sets,
        &ctx.results_path,
    ))
}

/// The datasets of a run, downloading missing built-in sets first when the config allows it;
/// resolution problems become run warnings.
fn resolve_sets(ctx: &RunContext, req: &RunRequest, h: &RunHandle) -> Result<Vec<ImageSet>> {
    let root = ctx.config.data_root();
    let mut resolved = images::resolve(&req.datasets, &root, req.max_images)?;
    if !resolved.not_downloaded.is_empty() && ctx.config.benchmark.auto_download_datasets {
        for w in download_datasets(ctx, &resolved.not_downloaded, h)? {
            h.warn(w);
        }
        resolved = images::resolve(&req.datasets, &root, req.max_images)?;
    }
    for w in &resolved.warnings {
        h.warn(w.clone());
    }
    let mut sets = resolved.sets;
    if let Some(w) = cap_images(&mut sets, MAX_IMAGES_PER_RUN) {
        h.warn(w);
    }
    Ok(sets)
}

/// Cut `sets` to about `cap` images in total (evenly spread within each set, each set keeping
/// a share proportional to its size and at least one image); the warning to show, or None when
/// they fit.
fn cap_images(sets: &mut [ImageSet], cap: usize) -> Option<String> {
    let total: usize = sets.iter().map(|s| s.images.len()).sum();
    if total <= cap || cap == 0 {
        return None;
    }
    for s in sets.iter_mut() {
        let len = s.images.len();
        // len * cap / total < len because cap < total; no overflow for realistic sizes.
        let quota = (len.saturating_mul(cap) / total).max(1);
        s.limit(quota);
    }
    let kept: usize = sets.iter().map(|s| s.images.len()).sum();
    Some(format!(
        "the datasets have {total} images; a run benchmarks at most {cap} \
         (max_images 0 = all, up to that cap): using {kept}, spread over {}",
        sets.iter()
            .map(|s| format!("{} ({})", s.id, s.images.len()))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The registry's shared runtimes, else dedicated ones from the config.
fn run_runtimes(ctx: &RunContext) -> Result<Arc<Mutex<Runtimes>>> {
    Ok(match &ctx.runtimes {
        Some(rt) => rt.clone(),
        None => {
            let c = &ctx.config;
            crate::backend::libs::prepare_environment(c.openvino_dir_effective().as_deref());
            let rt = Runtimes::new_with(
                &CoreOptions {
                    cache_dir: c.cache_dir_path(),
                    intra_threads: c.intra_threads,
                    openvino_dir: c.openvino_dir_effective(),
                },
                &c.ort_options(),
            );
            rt.require_any()?;
            Arc::new(Mutex::new(rt))
        }
    })
}

/// The completion message of a run.
fn run_summary(
    finished: &[ModelResult],
    image_count: usize,
    sets: &[ImageSet],
    results_path: &std::path::Path,
) -> String {
    let recs: Vec<String> = finished
        .iter()
        .map(|m| match &m.recommended {
            Some(d) => format!("{}: {d}", m.model),
            None => format!("{}: none", m.model),
        })
        .collect();
    format!(
        "Benchmarked {} model(s) on {image_count} image(s) from {}; recommended {}. Results saved \
         to {}.",
        finished.len(),
        sets.iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        recs.join(", "),
        results_path.display()
    )
}

/// Overall progress of a run across its models and devices (percentage and ETA).
struct OverallProgress {
    /// Device runs finished.
    runs_done: std::sync::atomic::AtomicUsize,
    /// Total device runs, for the overall percentage.
    runs_total: usize,
    model_count: usize,
    started: Instant,
}

impl OverallProgress {
    /// The status view of `p`, the progress of model `mi` (0-based).
    fn view(&self, mi: usize, p: &Progress) -> ProgressView {
        let done = self.runs_done.load(Ordering::Relaxed);
        let frac = if p.total == 0 {
            0.0
        } else {
            p.done as f64 / p.total as f64
        };
        // Loading and warm-up count as the first 10% of a device run, timing as the rest.
        let within = match p.phase {
            Phase::Loading => 0.0,
            Phase::Warmup => 0.1 * frac,
            Phase::Timing => 0.1 + 0.9 * frac,
        };
        let percent = if self.runs_total == 0 {
            0.0
        } else {
            ((done as f64 + within) / self.runs_total as f64 * 100.0).min(100.0)
        };
        let elapsed = self.started.elapsed().as_secs_f64();
        let eta_secs = (percent > 1.0).then(|| elapsed * (100.0 - percent) / percent);
        ProgressView {
            stage: "benchmark".into(),
            model: p.model.to_string(),
            model_index: mi + 1,
            model_count: self.model_count,
            device: p.device.to_string(),
            phase: phase_name(p.phase),
            done: p.done,
            total: p.total,
            image: p.image.to_string(),
            runs_done: done,
            runs_total: self.runs_total,
            percent,
            eta_secs,
            text: String::new(),
        }
    }
}

/// One benchmark run of the service once its datasets and runtimes are ready.
struct ServiceRun<'a> {
    ctx: &'a RunContext,
    req: &'a RunRequest,
    h: &'a RunHandle,
    runtimes: Arc<Mutex<Runtimes>>,
    /// The requested devices (None = every runnable option of each model).
    devices: Option<Vec<Device>>,
    params: PostParams,
    cache_dir: Option<PathBuf>,
}

impl ServiceRun<'_> {
    fn lock(&self) -> MutexGuard<'_, Runtimes> {
        self.runtimes.lock_recover()
    }

    /// Fill the ground truth of the sets without it from the reference model. Only
    /// cancellation fails; any other problem is a warning (accuracy not scored on those sets).
    fn pseudo_ground_truth(&self, sets: &mut [ImageSet]) -> Result<()> {
        let (ctx, h, devices) = (self.ctx, self.h, self.devices.as_deref());
        // The guard of `lock()` is dropped before pseudo ground truth locks the runtimes again.
        let reference = pick_reference(&ctx.config, self.req.reference_model.as_deref())
            .and_then(|m| job_for_config(&ctx.config, m, None))
            .map(|job| {
                let sel = self.lock().selection(Some(&job.path));
                (pseudo_gt_device(&sel, devices), job)
            });
        match reference {
            Ok((Some(device), job)) => {
                let pseudo_images: usize = sets
                    .iter()
                    .filter(|s| !s.annotated())
                    .map(|s| s.images.len())
                    .sum();
                let progress = |p: &Progress| {
                    h.set_progress(ProgressView {
                        stage: "pseudo-gt".into(),
                        model: p.model.to_string(),
                        device: p.device.to_string(),
                        phase: phase_name(p.phase),
                        done: p.done,
                        total: p.total,
                        image: p.image.to_string(),
                        text: format!(
                            "pseudo ground truth: {} on {} ({}/{} images)",
                            p.model, p.device, p.done, pseudo_images
                        ),
                        ..Default::default()
                    })
                };
                let mut pb = Bench::new(&[], 0, 1, self.params);
                pb.cache_dir = self.cache_dir.as_deref();
                pb.cancel = Some(h.cancel_flag());
                pb.progress = Some(&progress);
                match pseudo_ground_truth(&pb, &self.runtimes, &job, device, sets) {
                    Ok(_) => {
                        if self.in_run(&job.name) {
                            h.warn(format!(
                                "{} is also the pseudo-ground-truth reference: its accuracy on \
                                 the datasets without ground truth is 100% by construction",
                                job.name
                            ));
                        }
                    }
                    Err(e) if is_cancelled(&e) => return Err(e),
                    Err(e) => h.warn(format!(
                        "pseudo ground truth failed, accuracy is not scored for the datasets \
                         without ground truth: {e:#}"
                    )),
                }
            }
            Ok((None, _)) => h.warn(format!(
                "{}; accuracy is not scored for the datasets without ground truth",
                no_pseudo_gt_device(devices)
            )),
            Err(e) => h.warn(format!(
                "{e:#}; accuracy is not scored for the datasets without ground truth"
            )),
        }
        Ok(())
    }

    /// Whether the model `name` is benchmarked by this run.
    fn in_run(&self, name: &str) -> bool {
        let key = crate::registry::normalize_name;
        self.req.models.is_empty()
            && default_models(&self.ctx.config)
                .iter()
                .any(|m| key(m) == key(name))
            || self.req.models.iter().any(|m| key(m) == key(name))
    }

    /// Total device runs of `entries` (unknown models count none).
    fn runs_total(&self, entries: &[(String, Option<crate::config::ModelConfig>)]) -> usize {
        let per_model = |path: &std::path::Path| -> usize {
            match &self.devices {
                Some(d) => d.len(),
                None => self
                    .lock()
                    .selection(Some(path))
                    .options
                    .iter()
                    .filter(|o| o.runnable)
                    .count(),
            }
        };
        entries
            .iter()
            .map(|(_, m)| {
                m.as_ref()
                    .map(|m| per_model(&self.ctx.config.data_path(&m.path)))
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Sweep model `mi` (`name`, its config `entry`) over the devices, publishing progress and
    /// the result. A model that is not configured or cannot be loaded is a failed result; an
    /// error (cancellation, a failed sweep) ends the run.
    fn bench_model(
        &self,
        sets: &[ImageSet],
        overall: &OverallProgress,
        mi: usize,
        name: &str,
        entry: Option<&crate::config::ModelConfig>,
    ) -> Result<ModelResult> {
        let (ctx, req, h) = (self.ctx, self.req, self.h);
        let Some(m) = entry else {
            let r = ModelResult::failed(name, "", format!("'{name}' is not a configured model"));
            h.push_done(r.clone());
            return Ok(r);
        };
        let job = match job_for_config(&ctx.config, m, None) {
            Ok(j) => j,
            Err(e) => {
                let path = ctx.config.data_path(&m.path).display().to_string();
                let r = ModelResult::failed(&m.effective_name(), &path, format!("{e:#}"));
                h.push_done(r.clone());
                return Ok(r);
            }
        };
        let configured = ctx
            .loaded
            .get(&crate::registry::normalize_name(&job.name))
            .cloned()
            .or_else(|| configured_device(&ctx.config, m, &self.lock().selection(Some(&job.path))));
        let progress = |p: &Progress| h.set_progress(overall.view(mi, p));
        let mut bench = Bench::new(sets, req.warmup, req.repeat, self.params);
        bench.object_filter = &ctx.config.object_filter;
        bench.cache_dir = self.cache_dir.as_deref();
        bench.cancel = Some(h.cancel_flag());
        bench.progress = Some(&progress);
        bench.weights = req.weights;
        bench.threshold_objective = req.threshold_objective;
        bench.accuracy_metric = req.accuracy_metric;
        let opts = SweepOptions {
            devices: self.devices.clone(),
            configured,
            threshold_objective: req.threshold_objective,
        };
        let r = sweep(&bench, &self.runtimes, &job, &opts, &mut |partial| {
            overall.runs_done.fetch_add(1, Ordering::Relaxed);
            h.set_current(partial.summary());
        })?;
        h.push_done(r.summary());
        Ok(r)
    }

    /// Merge one finished model into the results file (and the stored predictions) as soon as
    /// it finishes, so a run holds one model's full results at a time (BM-11). `last`: the
    /// run's last model, whose save also drops the image sets no model refers to any more.
    fn save(
        &self,
        saver: &mut Saver,
        sets: &[ImageSet],
        model: ModelResult,
        last: bool,
    ) -> Result<()> {
        let (ctx, req) = (self.ctx, self.req);
        let key = crate::registry::normalize_name(&model.model);
        let results =
            BenchmarkResults::new(saver.hardware.clone(), saver.versions.clone(), vec![model])
                .with_sets(sets);
        let (prev, warnings) = BenchmarkResults::load_for_merge(&ctx.results_path)?;
        for w in warnings {
            self.h.with(|g| g.warnings.push(w));
        }
        let mut merged = saver.merge(prev, results, last);
        // Models kept from earlier runs are graded like this run's.
        merged.regrade(req.accuracy_metric, req.weights);
        merged.save(&ctx.results_path)?;
        // The model's entry in `merged` still has its evaluation predictions (not serialized).
        let new = merged
            .models
            .iter()
            .find(|m| crate::registry::normalize_name(&m.model) == key);
        for w in super::search::save_beside(
            &ctx.results_path,
            &merged,
            new.map_or(&[][..], std::slice::from_ref),
        ) {
            self.h.with(|g| g.warnings.push(w));
        }
        saver.finalized = last;
        Ok(())
    }

    /// After a run that stopped before its last model: drop the image sets no model refers
    /// to, like the last model's save would have.
    fn finalize(&self, saver: &Saver) -> Result<()> {
        if saver.retain_sets != Some(true) {
            return Ok(());
        }
        let path = &self.ctx.results_path;
        let Some(mut r) = BenchmarkResults::load(path)? else {
            return Ok(());
        };
        let before = r.image_sets.len();
        retain_referenced_sets(&mut r);
        if r.image_sets.len() != before {
            r.save(path)?;
        }
        Ok(())
    }
}

/// How a run merges its models into the results file one at a time (BM-11) so that the final
/// file is the one a single merge of all its models would give ([`BenchmarkResults::merge`]):
/// the image sets no model refers to are dropped only at the end, and only when the file the
/// run started from was comparable (same hardware and runtimes), as the single merge does.
struct Saver {
    /// Taken once per run: a model merged with other hardware facts would drop the run's
    /// earlier models as not comparable.
    hardware: HardwareSummary,
    versions: RuntimeVersions,
    /// Whether the file the run started from existed and was comparable; None before the
    /// first save.
    retain_sets: Option<bool>,
    /// The last save dropped the unreferenced image sets (nothing left to do at the end).
    finalized: bool,
}

impl Saver {
    fn merge(
        &mut self,
        prev: Option<BenchmarkResults>,
        new: BenchmarkResults,
        last: bool,
    ) -> BenchmarkResults {
        let comparable = prev
            .as_ref()
            .is_some_and(|p| p.hardware == new.hardware && p.runtimes == new.runtimes);
        let retain = *self.retain_sets.get_or_insert(comparable);
        // The sets before `merge` drops the unreferenced ones (kept until the end).
        let sets = match prev.as_ref().filter(|_| comparable) {
            Some(p) => upsert_sets(p.image_sets.clone(), new.image_sets.clone()),
            None => new.image_sets.clone(),
        };
        let mut merged = BenchmarkResults::merge(prev, new);
        merged.image_sets = sets;
        if last && retain {
            retain_referenced_sets(&mut merged);
        }
        merged
    }
}

/// `sets` with each of `new` replacing the set of the same id in place, else appended (the
/// order [`BenchmarkResults::merge`] gives).
fn upsert_sets(
    mut sets: Vec<super::report::ImageSetInfo>,
    new: Vec<super::report::ImageSetInfo>,
) -> Vec<super::report::ImageSetInfo> {
    for s in new {
        match sets.iter_mut().find(|p| p.id == s.id) {
            Some(slot) => *slot = s,
            None => sets.push(s),
        }
    }
    sets
}

/// Keep only the image sets some model refers to (as [`BenchmarkResults::merge`] does).
fn retain_referenced_sets(r: &mut BenchmarkResults) {
    let models = &r.models;
    r.image_sets
        .retain(|s| models.iter().any(|m| m.datasets.contains(&s.id)));
}

fn phase_name(p: Phase) -> String {
    serde_json::to_value(p)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    pub(crate) fn request(models: &[&str]) -> RunRequest {
        RunRequest {
            models: models.iter().map(|m| m.to_string()).collect(),
            devices: vec![],
            datasets: vec![DatasetRef::Id("sample".into())],
            max_images: 0,
            repeat: 1,
            warmup: 0,
            reference_model: None,
            weights: Weights::default(),
            threshold_objective: Objective::default(),
            accuracy_metric: AccuracyMetric::default(),
        }
    }

    #[test]
    fn request_from_form_over_config_defaults() {
        let defaults = BenchmarkConfig::default();
        let r = RunRequest::from_form(
            &form(&[
                ("model", "IPcam-general"),
                ("model", "dfine-s"),
                ("device", "openvino:cpu"),
                ("device", "ORT:CoreML"),
                ("device", "openvino:cpu"),
                ("dataset", "sample"),
                ("dataset", "dir:/tmp/x"),
                ("repeat", "10"),
                ("warmup", ""),
                ("max_images", "25"),
                ("reference_model", "rt-detrv2-x"),
                ("accuracy_weight", "0.8"),
                ("threshold_objective", "precision:0.9"),
                ("accuracy_metric", "roc_auc"),
            ]),
            &defaults,
        )
        .unwrap();
        assert_eq!(r.models, ["IPcam-general", "dfine-s"]);
        assert_eq!(r.repeat, 10);
        assert_eq!(r.warmup, defaults.warmup);
        assert_eq!(r.max_images, 25);
        assert_eq!(r.datasets.len(), 2);
        assert_eq!(r.reference_model.as_deref(), Some("rt-detrv2-x"));
        assert!((r.weights.accuracy_share() - 0.8).abs() < 1e-12);
        assert_eq!(r.threshold_objective, Objective::Precision(0.9));
        assert_eq!(r.accuracy_metric, AccuracyMetric::RocAuc);
        let d = r.parsed_devices().unwrap().unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d[1].to_string(), "ort:coreml");

        // Absent fields take the config defaults.
        let mut b = BenchmarkConfig {
            max_images_per_dataset: 7,
            repeat_per_image: 2,
            devices: vec!["ort:cpu".into()],
            ..Default::default()
        };
        let r = RunRequest::from_form(&form(&[]), &b).unwrap();
        assert_eq!((r.max_images, r.repeat), (7, 2));
        assert_eq!(r.devices, ["ort:cpu"]);
        assert_eq!(r.datasets, b.datasets);
        assert!(r.models.is_empty());
        // ...and save back.
        let r = RunRequest::from_form(
            &form(&[
                ("dataset", "sample"),
                ("repeat", "3"),
                ("threshold_objective", "youden"),
                ("accuracy_metric", "roc_auc"),
            ]),
            &b,
        )
        .unwrap();
        r.to_config(&mut b);
        assert_eq!(b.accuracy_metric, AccuracyMetric::RocAuc);
        assert_eq!(b.repeat_per_image, 3);
        assert_eq!(b.threshold_objective, Objective::Youden);
        assert_eq!(b.datasets, [DatasetRef::Id("sample".into())]);

        for (bad, needle) in [
            (form(&[("repeat", "0")]), "repeat"),
            (form(&[("repeat", "x")]), "repeat"),
            (form(&[("warmup", "1000")]), "warmup"),
            (form(&[("device", "vulkan")]), "vulkan"),
            (form(&[("device", "auto")]), "auto"),
            (form(&[("accuracy_weight", "2")]), "accuracy_weight"),
            (form(&[("dataset", "")]), "dataset"),
            (form(&[("threshold_objective", "f3")]), "objective"),
            (form(&[("accuracy_metric", "map")]), "accuracy metric"),
        ] {
            let e = RunRequest::from_form(&bad, &defaults).unwrap_err();
            assert!(format!("{e:#}").contains(needle), "{e:#}");
        }
    }

    #[test]
    fn one_run_at_a_time_and_cancel() {
        let svc = BenchmarkService::new();
        assert_eq!(svc.status().state, RunState::Idle);
        assert!(!svc.cancel());
        let req = request(&["m"]);
        svc.start_job(req.clone(), |h| {
            while !h.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(super::super::Cancelled.into())
        })
        .unwrap();
        assert!(svc.is_running());
        assert_eq!(svc.start_job(req.clone(), |_| Ok(String::new())), Err(Busy));
        assert!(svc.cancel());
        assert!(svc.wait_idle(Duration::from_secs(5)));
        let st = svc.status();
        assert_eq!(st.state, RunState::Cancelled);
        assert!(st.finished_at.is_some());

        // A job that publishes partial results and warnings, and finishes.
        svc.start_job(req.clone(), |h| {
            let r = ModelResult::failed("m", "m.onnx", "x".into());
            h.set_current(r.clone());
            h.push_done(r);
            h.warn("careful".into());
            Ok("done".into())
        })
        .unwrap();
        assert!(svc.wait_idle(Duration::from_secs(5)));
        let st = svc.status();
        assert_eq!(st.state, RunState::Done);
        assert_eq!(st.message.as_deref(), Some("done"));
        assert_eq!(st.models.len(), 1);
        assert_eq!(st.warnings, ["careful"]);

        // Failures and panics end in Failed.
        svc.start_job(req.clone(), |_| bail!("boom")).unwrap();
        assert!(svc.wait_idle(Duration::from_secs(5)));
        assert_eq!(svc.status().error.as_deref(), Some("boom"));
        assert!(svc.status().warnings.is_empty(), "cleared per run");
        svc.start_job(req, |_| panic!("test panic")).unwrap();
        assert!(svc.wait_idle(Duration::from_secs(5)));
        assert_eq!(svc.status().state, RunState::Failed);
    }

    #[test]
    fn run_records_unknown_and_missing_models() {
        let dir = std::env::temp_dir().join(format!("bop-bsvc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = Config {
            models: vec![crate::config::ModelConfig {
                name: Some("gone".into()),
                path: dir.join("gone.onnx"),
                ..Default::default()
            }],
            ..Default::default()
        };
        // Runtimes that load nothing: the run must not need them for these models.
        let rt = Runtimes::new_with(
            &CoreOptions {
                openvino_dir: Some(dir.join("none")),
                ..Default::default()
            },
            &crate::backend::OrtOptions {
                onnxruntime_dir: Some(dir.join("none")),
                default_dir: Some(dir.join("none")),
                cuda_libs_dir: None,
            },
        );
        let ctx = RunContext {
            config,
            results_path: dir.join("benchmark.json"),
            runtimes: Some(Arc::new(Mutex::new(rt))),
            loaded: HashMap::new(),
            resources: None,
        };
        let svc = BenchmarkService::new();
        let mut req = request(&["gone", "nope"]);
        req.devices = vec!["openvino:cpu".into()];
        svc.start(ctx, req).unwrap();
        assert!(svc.wait_idle(Duration::from_secs(30)));
        let st = svc.status();
        assert_eq!(st.state, RunState::Done, "{st:?}");
        assert_eq!(st.models.len(), 2);
        assert!(st.models[0].error.as_deref().unwrap().contains("not found"));
        assert!(
            st.models[1]
                .error
                .as_deref()
                .unwrap()
                .contains("not a configured")
        );
        // No reference model for the unannotated sample: a warning, not a failure.
        assert!(
            st.warnings.iter().any(|w| w.contains("reference model")),
            "{:?}",
            st.warnings
        );
        let saved = BenchmarkResults::load(&dir.join("benchmark.json"))
            .unwrap()
            .unwrap();
        assert_eq!(saved.models.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn max_images_is_capped_per_run() {
        let mut r = request(&[]);
        r.max_images = MAX_IMAGES_PER_RUN;
        r.validate().unwrap();
        r.max_images = 0;
        r.validate().unwrap();
        r.max_images = MAX_IMAGES_PER_RUN + 1;
        let e = r.validate().unwrap_err().to_string();
        assert!(e.contains("max_images") && e.contains("0 = all"), "{e}");
        let defaults = BenchmarkConfig::default();
        assert!(RunRequest::from_form(&form(&[("max_images", "999999")]), &defaults).is_err());
    }

    fn set_of(id: &str, n: usize) -> ImageSet {
        let mut s = ImageSet::sample();
        s.id = id.into();
        let img = s.images[0].clone();
        s.images = (0..n)
            .map(|i| {
                let mut x = img.clone();
                x.file = format!("{i}.jpg");
                x
            })
            .collect();
        s
    }

    #[test]
    fn cap_images_spreads_the_cap_over_the_sets() {
        let mut sets = vec![set_of("a", 10), set_of("b", 5)];
        assert_eq!(cap_images(&mut sets, 15), None);
        assert_eq!(sets[0].images.len() + sets[1].images.len(), 15);
        let mut sets = vec![set_of("big", 3000), set_of("small", 30), set_of("one", 1)];
        let w = cap_images(&mut sets, MAX_IMAGES_PER_RUN).unwrap();
        assert!(w.contains("at most 2000"), "{w}");
        let lens: Vec<usize> = sets.iter().map(|s| s.images.len()).collect();
        // 3000 * 2000 / 3031 = 1979, 30 * 2000 / 3031 = 19, and at least one image.
        assert_eq!(lens, [1979, 19, 1]);
        assert!(lens.iter().sum::<usize>() <= MAX_IMAGES_PER_RUN);
        // Evenly spread: the first and a late image of the big set are kept.
        assert_eq!(sets[0].images[0].file, "0.jpg");
        assert!(sets[0].images.last().unwrap().file != "1978.jpg");
    }

    fn model(name: &str, datasets: &[&str]) -> ModelResult {
        let mut m = ModelResult::failed(name, "", String::new());
        m.error = None;
        m.datasets = datasets.iter().map(|d| d.to_string()).collect();
        m
    }

    fn info(id: &str, title: &str) -> super::super::report::ImageSetInfo {
        let mut s = super::super::report::ImageSetInfo::of(&ImageSet::sample());
        s.id = id.into();
        s.title = title.into();
        s
    }

    fn results(
        hw: &str,
        models: Vec<ModelResult>,
        sets: Vec<super::super::report::ImageSetInfo>,
    ) -> BenchmarkResults {
        let mut r = BenchmarkResults::new(
            HardwareSummary {
                cpu: hw.into(),
                ..Default::default()
            },
            RuntimeVersions::default(),
            models,
        );
        r.image_sets = sets;
        r
    }

    /// Merging a run's models one at a time (each through the file) gives the single merge
    /// of all of them, also where a merge of the first model alone would drop a set (BM-11).
    #[test]
    fn per_model_saves_equal_one_merge() {
        let run_sets = || vec![info("s1", "new"), info("s3", "new")];
        let run_models = || vec![model("a", &[]), model("c", &["s1", "s3"])];
        let prevs = [
            Some(results(
                "hw",
                vec![model("a", &["s1"]), model("b", &["s2"])],
                vec![info("s1", "old"), info("s2", "old")],
            )),
            Some(results(
                "other",
                vec![model("b", &["s2"])],
                vec![info("s2", "old")],
            )),
            None,
        ];
        for prev in prevs {
            for models in [run_models(), vec![model("a", &[]), model("x", &[])]] {
                let one = BenchmarkResults::merge(
                    prev.clone(),
                    results("hw", models.clone(), run_sets()),
                );
                let mut saver = Saver {
                    hardware: HardwareSummary {
                        cpu: "hw".into(),
                        ..Default::default()
                    },
                    versions: RuntimeVersions::default(),
                    retain_sets: None,
                    finalized: false,
                };
                let mut file = prev.clone();
                let n = models.len();
                for (i, m) in models.into_iter().enumerate() {
                    let new = results("hw", vec![m], run_sets());
                    // Through JSON, like the file.
                    let merged = saver.merge(file.take(), new, i + 1 == n);
                    file = Some(
                        serde_json::from_str(&serde_json::to_string(&merged).unwrap()).unwrap(),
                    );
                }
                let file = file.unwrap();
                assert_eq!(file.models, one.models);
                assert_eq!(file.image_sets, one.image_sets, "{prev:?}");
            }
        }
        // The case a naive per-model merge gets wrong: "a" alone drops s1 (only "a" referred
        // to it), which then comes back after s2.
        let one = BenchmarkResults::merge(
            Some(results(
                "hw",
                vec![model("a", &["s1"]), model("b", &["s2"])],
                vec![info("s1", "old"), info("s2", "old")],
            )),
            results("hw", run_models(), run_sets()),
        );
        let ids: Vec<&str> = one.image_sets.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["s1", "s2", "s3"]);
    }
}
