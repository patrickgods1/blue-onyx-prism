//! One training run as a stage machine ([`run_training`]) over the model export's seams
//! ([`CommandRunner`] for processes, [`Fetcher`] for verified downloads) plus [`Hooks`] for what
//! the caller owns (adding the model to the config, the benchmark).
//!
//! Stages ([`TrainStage`]): setup (uv, Python, the training lock's packages, the Visual C++
//! runtime; shared code with the export, under its lock) -> self-check (`train_model.py
//! --self-check`: PyTorch imports and sees the GPU) -> weights (the pinned base weights, and the
//! public subset when asked) -> dataset (assembled in Rust, [`super::dataset`]) -> train (the
//! script trains and exports a static ONNX, checked against PyTorch with ONNX Runtime; progress
//! from its `@train` events) -> install (`<name>.onnx`, `.yaml` and `<name>.provenance.json`
//! in the models folder or the request's `install_dir`, [`TrainJob::install_dir`]) -> register
//! ([`Hooks::register`], the `add-model` code, with the installed path) -> benchmark
//! ([`Hooks::benchmark`] on the held-out `dir:` set; a failure there is reported but does not
//! undo the model). Cancellation kills the child's process tree (the runner's job).
//!
//! The installed model goes on the run's record at once ([`TrainReport::Installed`]): from then on
//! the run owns a file that blocks its name, however it ends. The checkpoints (`<run>/work`) go once
//! the model is registered, or when the run is cancelled before it installed. A failed run keeps
//! them: training and export share one process, so a late failure would otherwise cost the whole
//! training. [`TrainJob::export_only`] exports such a run again ([`super::retry`]): setup,
//! self-check, then only the export and the stages after it. A run out of memory restarts only
//! while that costs little ([`keeps_checkpoints`]).

use super::capability::Accelerator;
use super::dataset::{self, Prepared, SceneFilter};
use super::preflight;
use super::progress::{self, TrainEvent, TrainProgress};
use super::recipe::TrainRecipe;
use crate::model::ModelFamilyKind;
use crate::resources::catalog::{self, Platform, Resource, TrainFamily};
use crate::resources::export::train::{self as env, TrainLock, TrainPaths};
use crate::resources::export::{Cancelled, CommandRunner, Fetcher, Progress, Stage};
use crate::resources::manager;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Stages of a run, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStage {
    Setup,
    SelfCheck,
    Weights,
    Dataset,
    Train,
    Install,
    Register,
    Benchmark,
}

impl TrainStage {
    pub const ALL: [TrainStage; 8] = [
        TrainStage::Setup,
        TrainStage::SelfCheck,
        TrainStage::Weights,
        TrainStage::Dataset,
        TrainStage::Train,
        TrainStage::Install,
        TrainStage::Register,
        TrainStage::Benchmark,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            TrainStage::Setup => "setting up the training environment",
            TrainStage::SelfCheck => "checking PyTorch and the GPU",
            TrainStage::Weights => "downloading the base weights",
            TrainStage::Dataset => "assembling the dataset",
            TrainStage::Train => "training",
            TrainStage::Install => "installing the model files",
            TrainStage::Register => "adding the model to the config",
            TrainStage::Benchmark => "benchmarking on the held-out frames",
        }
    }

    /// 1-based position.
    pub fn number(self) -> usize {
        TrainStage::ALL.iter().position(|s| *s == self).unwrap_or(0) + 1
    }

    /// `[start, end)` of the overall percent.
    const fn span(self) -> (u8, u8) {
        match self {
            TrainStage::Setup => (0, 12),
            TrainStage::SelfCheck => (12, 14),
            TrainStage::Weights => (14, 18),
            TrainStage::Dataset => (18, 20),
            TrainStage::Train => (20, 92),
            TrainStage::Install => (92, 93),
            TrainStage::Register => (93, 94),
            TrainStage::Benchmark => (94, 100),
        }
    }

    /// Overall percent when this stage is `pct` percent done.
    pub fn overall(self, pct: u8) -> u8 {
        let (a, b) = self.span();
        let p = u32::from(pct.min(100));
        a + u8::try_from(u32::from(b - a) * p / 100).unwrap_or(b - a)
    }
}

/// What to train on what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainRequest {
    pub recipe: TrainRecipe,
    /// Review export folders.
    pub datasets: Vec<PathBuf>,
    /// Mix in the pinned public night subset (`train-data:lila-night`).
    #[serde(default)]
    pub include_public: bool,
    #[serde(default)]
    pub scene: SceneFilter,
    /// The model's classes in head order; None = every class of the datasets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub rename: BTreeMap<String, String>,
    /// Built-in benchmark sets to run besides the held-out frames (`nir-night`, `exdark-ir`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub benchmark_sets: Vec<String>,
    /// Devices the benchmark at the end runs on, one after the other (canonical specs); empty =
    /// the one device the model would load on (its device, `auto` resolved).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub benchmark_devices: Vec<String>,
    /// Downloads over the large-download limit are allowed (`--allow-large`, the page's
    /// confirmation, or `allow_large_downloads`; [`super::plan_job`] folds the config in). Gates
    /// the public night subset ([`super::check_public_download`]).
    #[serde(default)]
    pub allow_large: bool,
    /// Folder the finished model is installed in; None = the models folder. [`super::plan_job`]
    /// resolves a relative one against the data root (absolute in a queued job).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_dir: Option<PathBuf>,
    /// The GPU to train on (`cuda:N`; None = the automatic pick, [`super::Capability::for_gpu`]).
    /// [`super::plan_job`] stores the canonical device of the chosen GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// What a train / valid frame with a box of an untrained label becomes.
    #[serde(default)]
    pub excluded: dataset::Excluded,
    /// Balance the training frames (step 7 of [`dataset::plan`]); off by default.
    #[serde(default, skip_serializing_if = "dataset::Balance::is_off")]
    pub balance: dataset::Balance,
    /// Runs queued together to be compared (one per scene filter x model): the first run's id
    /// stamp ([`super::compare`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compare_group: Option<String>,
    /// The trainer's seed of a repeat ("Repeat with N seeds"); None = the options' seed. The
    /// dataset assembly keeps `options.seed`, so every repeat trains on the same frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_seed: Option<u64>,
}

impl TrainRequest {
    /// The seed handed to the trainer: the repeat's own, else `assembly_seed`.
    pub fn trainer_seed(&self, assembly_seed: u64) -> u64 {
        self.train_seed.unwrap_or(assembly_seed)
    }
}

/// The sources of a run: the review exports `datasets` (absolute), then the public subset
/// installed in `public` when given. The job and [`super::preview_plan`] read the same way.
pub fn read_sources(datasets: &[PathBuf], public: Option<&Path>) -> Result<Vec<dataset::Source>> {
    let mut sources = Vec::new();
    for d in datasets {
        sources.push(dataset::read_export(d).with_context(|| format!("reading {}", d.display()))?);
    }
    if let Some(dir) = public {
        let m = catalog::lila_night_manifest()
            .ok_or_else(|| anyhow!("the public subset's manifest does not parse"))?;
        sources.push(dataset::read_public_set(dir, m)?);
    }
    Ok(sources)
}

/// `base` (the config's assembly defaults, [`TrainJob::options`]) with the request's own choices:
/// scene filter, classes, renames, excluded mode and balance.
pub fn request_options(base: &dataset::Options, request: &TrainRequest) -> dataset::Options {
    dataset::Options {
        scene: request.scene,
        classes: request.classes.clone(),
        rename: request.rename.clone(),
        excluded: request.excluded,
        balance: request.balance.clone(),
        ..base.clone()
    }
}

/// Data loader workers of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workers {
    /// [`super::workers_now`] when training starts (the free commit then, not when queued).
    Auto,
    /// This many (`training.workers`).
    Fixed(u32),
}

/// One queued run.
#[derive(Debug, Clone)]
pub struct TrainJob {
    pub id: String,
    pub request: TrainRequest,
    /// The trained model's name (`<install dir>/<name>.onnx`, `/v1/vision/custom/<name>`).
    pub name: String,
    pub paths: TrainPaths,
    /// `<training dir>/<id>`: the prepared dataset, the log, the held-out set.
    pub run_dir: PathBuf,
    pub platform: Platform,
    pub accelerator: Accelerator,
    pub options: dataset::Options,
    /// Data root (the public subset installs under it).
    pub root: PathBuf,
    pub timeout: Duration,
    /// Data loader workers (`--workers`): sized when training starts, or forced by
    /// `training.workers`.
    pub workers: Workers,
    /// Keep the checkpoints (`<run>/work`) after the run.
    pub keep_work: bool,
    /// Export this failed run's checkpoint instead of training ([`super::retry::plan_retry`]).
    pub export_only: Option<ExportRetry>,
}

/// A failed run to export again: what the run itself no longer tells.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportRetry {
    /// `train_model.py --export-only` ([`super::retry::best_checkpoint`]).
    pub checkpoint: PathBuf,
    /// Time the training took, for the provenance (the script reports the export's).
    pub train_seconds: Option<f64>,
    /// What it trained on, for the provenance (the export runs on the CPU).
    pub device: String,
}

impl TrainJob {
    /// Where the model is installed: the request's `install_dir`, else the models folder.
    pub fn install_dir(&self) -> &Path {
        self.request
            .install_dir
            .as_deref()
            .unwrap_or(&self.paths.export.models)
    }
}

/// What a run reports.
#[derive(Debug)]
pub enum TrainReport<'a> {
    Stage {
        stage: TrainStage,
        percent: u8,
        detail: &'a str,
    },
    Line(&'a str),
    Progress(&'a TrainProgress),
    Prepared(&'a Prepared),
    /// The trainer ran out of GPU memory early and the run restarts at this batch
    /// ([`OOM_RETRIES`]); the run record keeps the batch it finishes with.
    Batch(u32),
    /// The model is in the install folder: the record names it before anything else can fail.
    Installed(&'a TrainOutput),
    /// [`Hooks::register`] added it to the config; its message.
    Registered(&'a str),
}

/// The installed model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainOutput {
    pub name: String,
    pub onnx: PathBuf,
    pub yaml: PathBuf,
    pub provenance: PathBuf,
    pub family: ModelFamilyKind,
    pub classes: Vec<String>,
    pub license: String,
    /// The held-out benchmark set (`dir:<path>`).
    pub heldout: Option<PathBuf>,
    pub device: String,
    pub train_seconds: f64,
}

/// A benchmark of a trained model (from `benchmark.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BenchSummary {
    pub datasets: Vec<String>,
    pub device: Option<String>,
    /// Total time per image (p50), ms.
    pub ms_p50: Option<f64>,
    /// AP50 over every class of the model when the sets annotate others (the benchmark's
    /// `all_classes`), else over the scored CCTV classes ([`super::bench::summarize`]).
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub images: usize,
    /// The confidence threshold the alerting figures below were measured at (the benchmark's
    /// configured threshold for the model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Recall of `person` at `threshold`, IoU 0.5 (`benchmark::metrics::AlertStats`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_recall: Option<f64>,
    /// Recall of vehicles at `threshold`, car/truck/bus/vehicle merged into one class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vehicle_recall: Option<f64>,
    /// Benchmarked frames without any scored object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empty_frames: Option<usize>,
    /// Of `empty_frames`, those with a detection at or above `threshold` (false alerts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empty_frame_alerts: Option<usize>,
    /// AP on the held-out frames per scene: `day`, `night`, `ir`, and `all` (every held-out
    /// frame) ([`super::bench::summarize`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_scene: BTreeMap<String, SceneAp>,
    /// AP on the held-out frames of the scenes the run trained on (its headline).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_scope: Option<SceneAp>,
    /// [`dataset::Plan::heldout_fingerprint`] of the held-out set benchmarked; None for runs of
    /// older versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heldout_fingerprint: Option<String>,
    /// [`crate::benchmark::report::ModelResult::trained_on_share`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_on_share: Option<f64>,
}

/// The current default model on the frames of a run's benchmark ([`Hooks::baseline`]): what the
/// new model's result is compared with on the Train tab.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Baseline {
    /// The default model; empty when the comparison failed before it was known.
    pub model: String,
    /// Its result; None = not compared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<BenchSummary>,
    /// Why it was not compared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// AP of a subset of the held-out frames.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneAp {
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub images: usize,
}

/// What the caller does for a run.
pub trait Hooks: Send + Sync {
    /// Add the installed model to the config (the `add-model` code); returns a message.
    fn register(&self, out: &TrainOutput) -> Result<String>;
    /// Benchmark the model on its held-out set and `extra` built-in sets, on `devices` one after
    /// the other (empty = the device the model would load on).
    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        devices: &[String],
        cancel: &AtomicBool,
    ) -> Result<Option<BenchSummary>>;
    /// Benchmark the current default model on the same sets and devices, its results kept under
    /// `run_dir` (never in `benchmark.json`). None = this caller does not compare (the CLI: no
    /// model is loaded there); a model that cannot be compared answers with a [`Baseline::note`].
    fn baseline(
        &self,
        _out: &TrainOutput,
        _extra: &[String],
        _devices: &[String],
        _run_dir: &Path,
        _cancel: &AtomicBool,
    ) -> Result<Option<Baseline>> {
        Ok(None)
    }
}

/// Hooks that do nothing (tests, `--no-register`).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoHooks;

impl Hooks for NoHooks {
    fn register(&self, out: &TrainOutput) -> Result<String> {
        Ok(format!("{} installed (not added to the config)", out.name))
    }
    fn benchmark(
        &self,
        _: &TrainOutput,
        _: &[String],
        _: &[String],
        _: &AtomicBool,
    ) -> Result<Option<BenchSummary>> {
        Ok(None)
    }
}

/// A successful run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunOutcome {
    pub output: TrainOutput,
    pub prepared: Prepared,
    pub registered: String,
    pub benchmark: Option<BenchSummary>,
    pub benchmark_error: Option<String>,
    pub baseline: Option<Baseline>,
}

/// Why a run failed, and in which stage.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TrainError {
    pub stage: Option<TrainStage>,
    pub message: String,
    pub cancelled: bool,
}

/// `<models>/<name>.provenance.json`.
pub fn provenance_path(models: &Path, name: &str) -> PathBuf {
    models.join(format!("{name}.provenance.json"))
}

/// The file whose presence asks the trainer to finish after the current epoch and export its
/// best one (`train_model.py --stop-file`; written by [`super::TrainingService::finish`]).
pub fn stop_file(run_dir: &Path) -> PathBuf {
    run_dir.join("work").join("stop-request")
}

/// Log file of a run.
pub const LOG_FILE: &str = "train.log";
/// Largest log written (bytes); later lines are dropped with a note.
pub const MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;
/// Timeout of the self-check.
const SELF_CHECK_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// Restarts of a run that ran out of GPU memory, each at half the batch (down to 1); as many
/// again for running out of system memory, each with half the data loader workers (down to 0).
pub const OOM_RETRIES: u32 = 3;
/// Only an out-of-memory failure before this many finished epochs is retried (later, the batch
/// had fitted; restarting would throw away the epochs).
pub const OOM_RETRY_EPOCHS: usize = 3;
/// MPS allocator limits on macOS (ratios of the recommended working set; see
/// `scripts/train_model.py` `main`): an oversized batch fails in seconds instead of swapping.
pub const MPS_WATERMARKS: [(&str, &str); 2] = [
    ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0"),
    ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8"),
];

/// Longest export of an export-only run: minutes on a CPU, so this only catches a hang.
const EXPORT_ONLY_TIMEOUT: Duration = Duration::from_secs(2 * 3600);

/// What a run that installed `o` but did not register it tells the user: Retry export has nothing
/// to do there, and the file refuses the name to a new run.
pub fn not_in_config(o: &TrainOutput) -> String {
    format!(
        "The model is installed as {} but is not in the config: add it on the Config page (Add a \
         model by path), or delete it to train under that name again",
        o.onnx.display()
    )
}

/// Run every stage of `job` (see the module docs). A failure names the checkpoint left behind, if
/// any, or the model that was installed but not registered; a cancelled training takes its
/// checkpoints along, a cancelled retry keeps them (the failed run it came from is still there to
/// retry), and so does a run cancelled with a model not registered.
pub fn run_training(
    job: &TrainJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
) -> Result<RunOutcome, TrainError> {
    run_training_with(job, None, runner, fetcher, hooks, report, cancel)
}

/// [`run_training`] for a task that holds the training lock already, across several training runs
/// and the GPU work between them (the dataset check's folds): the run uses `lock` instead of
/// taking the lock itself, which would fail against the task's own hold. `lock` must be the lock
/// of `job.paths` (an error otherwise).
pub fn run_training_holding(
    job: &TrainJob,
    lock: &TrainLock,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
) -> Result<RunOutcome, TrainError> {
    run_training_with(job, Some(lock), runner, fetcher, hooks, report, cancel)
}

fn run_training_with(
    job: &TrainJob,
    held: Option<&TrainLock>,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
) -> Result<RunOutcome, TrainError> {
    let mut stage = None;
    let mut unregistered = None;
    let mut watch = |p: TrainReport<'_>| {
        match &p {
            TrainReport::Installed(o) => unregistered = Some(not_in_config(o)),
            TrainReport::Registered(_) => unregistered = None,
            _ => {}
        }
        report(p);
    };
    let r = stages(
        job, held, runner, fetcher, hooks, &mut watch, cancel, &mut stage,
    );
    r.map_err(|e| {
        let cancelled = e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed);
        let work = super::retry::work_dir(&job.run_dir);
        if cancelled && !job.keep_work && job.export_only.is_none() && unregistered.is_none() {
            let _ = manager::remove_any(&work);
        }
        TrainError {
            stage,
            message: match &unregistered {
                _ if cancelled => "cancelled".into(),
                Some(note) => format!("{e:#}. {note}"),
                None => format!("{e:#}{}", kept_note(job, &work)),
            },
            cancelled,
        }
    })
}

/// The failure message's last sentence: the checkpoint that survived and what to do with it.
fn kept_note(job: &TrainJob, work: &Path) -> String {
    if super::retry::best_checkpoint(job.request.recipe.family.trainer(), work).is_none() {
        return String::new();
    }
    format!(
        ". Its best checkpoint is kept ({} in {}): Retry export on the run's row (train \
         retry-export {}) makes the model without training again",
        catalog::format_size(super::retry::disk_use(work)),
        work.display(),
        job.id
    )
}

/// Whether a run out of memory stops rather than restarts from epoch 1: its work folder holds a
/// checkpoint, and either training had ended (a restart would repeat all of it for an export that
/// can be retried alone) or at least [`OOM_RETRY_EPOCHS`] epochs finished. It then fails, keeps the
/// checkpoint, and the row offers its best epoch.
fn keeps_checkpoints(tp: &TrainProgress, trainer: catalog::Trainer, work: &Path) -> bool {
    (tp.phase.is_some() || tp.history.len() >= OOM_RETRY_EPOCHS)
        && super::retry::best_checkpoint(trainer, work).is_some()
}

/// A failure while writing files, as an actionable message: a full disk gets the numbers.
fn explain_full_disk(e: anyhow::Error, dir: &Path) -> anyhow::Error {
    if !super::import::disk::is_disk_full(&e) {
        return e;
    }
    let free = super::import::disk::space(dir)
        .map_or_else(|| "unknown".into(), |s| catalog::format_size(s.available));
    e.context(format!(
        "the disk filled up ({free} free under {}): {}, then train again",
        dir.display(),
        preflight::TRAINING_ADVICE
    ))
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(anyhow!(Cancelled))
    } else {
        Ok(())
    }
}

/// The run log, capped at [`MAX_LOG_BYTES`].
struct Log {
    file: Option<std::fs::File>,
    written: u64,
}

impl Log {
    fn open(path: &Path) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| tracing::warn!("cannot write the training log {}: {e}", path.display()))
            .ok();
        Self { file, written: 0 }
    }

    fn line(&mut self, l: &str) {
        let Some(f) = self.file.as_mut() else { return };
        if self.written > MAX_LOG_BYTES {
            return;
        }
        self.written += l.len() as u64 + 1;
        let _ = if self.written > MAX_LOG_BYTES {
            writeln!(f, "[log cut at {} MB]", MAX_LOG_BYTES / (1024 * 1024))
        } else {
            writeln!(f, "{l}")
        };
    }
}

#[allow(clippy::too_many_arguments)]
fn stages(
    job: &TrainJob,
    held: Option<&TrainLock>,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
    current: &mut Option<TrainStage>,
) -> Result<RunOutcome> {
    let recipe = &job.request.recipe;
    recipe.validate()?;
    super::recipe::valid_model_name(&job.name)?;
    let paths = &job.paths;
    let models = job.install_dir();
    let family = recipe.family;
    let base = recipe.base()?;
    let lane = job.accelerator.lane();
    macro_rules! enter {
        ($stage:expr, $pct:expr, $detail:expr) => {{
            *current = Some($stage);
            report(TrainReport::Stage {
                stage: $stage,
                percent: $pct,
                detail: $detail,
            });
        }};
    }
    if models.join(format!("{}.onnx", job.name)).exists() {
        bail!(
            "{} already exists; choose another model name",
            models.join(format!("{}.onnx", job.name)).display()
        );
    }
    if let Some(r) = &job.export_only {
        if !r.checkpoint.exists() {
            bail!(
                "the checkpoint {} is gone; train the model again",
                r.checkpoint.display()
            );
        }
        preflight::check_disk(vec![preflight::VolumeNeed {
            dir: models.to_path_buf(),
            bytes: super::retry::EXPORT_BYTES,
            advice: preflight::MODELS_ADVICE,
        }])?;
    }
    std::fs::create_dir_all(&job.run_dir)
        .with_context(|| format!("creating {}", job.run_dir.display()))?;
    let mut log = Log::open(&job.run_dir.join(LOG_FILE));
    log.line(&format!(
        "# {} {} on {} ({})",
        crate::long_version(),
        recipe.label(),
        job.accelerator.describe(),
        job.id
    ));
    // Taken here, or held by the task this run is part of (for the whole task).
    let _lock = match held {
        Some(l) if l.path() == paths.train_lock_file() => None,
        Some(l) => bail!(
            "the training lock held is {}, not this run's {}",
            l.path().display(),
            paths.train_lock_file().display()
        ),
        None => Some(TrainLock::try_acquire(&paths.train_lock_file())?),
    };

    // 1. Environment.
    enter!(TrainStage::Setup, 0, "checking the toolchain");
    let ej = env::env_job(paths, base, job.platform);
    env::prepare_train_env(
        &ej,
        lane,
        runner,
        fetcher,
        &mut |p| match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                // The export bar's share of these stages ends at its self-check (71%).
                let pct = u32::from(stage.overall(percent)) * 100 / 71;
                let d = format!("{}: {detail}", stage.label());
                report(TrainReport::Stage {
                    stage: TrainStage::Setup,
                    percent: u8::try_from(pct.min(100)).unwrap_or(100),
                    detail: &d,
                });
            }
            Progress::Line { line, .. } => {
                log.line(line);
                report(TrainReport::Line(line));
            }
        },
        cancel,
    )
    .map_err(|e| {
        if e.cancelled {
            anyhow!(Cancelled)
        } else {
            anyhow!(
                "{}{}",
                e.stage
                    .map(|s| format!("while {}: ", s.label()))
                    .unwrap_or_default(),
                e.message
            )
        }
    })?;
    check(cancel)?;

    // 2. Self-check.
    enter!(
        TrainStage::SelfCheck,
        0,
        "importing PyTorch and the trainer"
    );
    let (train_script, export_script) = env::write_scripts(paths)?;
    let trainer = family.trainer().as_str();
    let cmd = env::train_python_cmd(
        paths,
        job.platform,
        Stage::SelfCheck,
        &paths.scripts,
        SELF_CHECK_TIMEOUT,
        vec![
            train_script.clone().into_os_string(),
            "--self-check".into(),
            "--trainer".into(),
            trainer.into(),
            // The test kernel runs on the run's GPU (another GPU may have other kernels).
            "--device".into(),
            job.accelerator.device_arg().into(),
        ],
    );
    let mut seen: Option<TrainEvent> = None;
    runner
        .run(
            &cmd,
            &mut |l| {
                log.line(l);
                report(TrainReport::Line(l));
                if let Some(ev @ TrainEvent::SelfCheck { .. }) = progress::parse_line(l) {
                    seen = Some(ev);
                }
            },
            cancel,
        )
        .map_err(|e| env::explain_train_failure(e, paths, job.platform, TRAINING_TASK))
        .context("the training environment's self-check failed")?;
    match seen {
        Some(TrainEvent::SelfCheck {
            cuda, mps, cuda_ok, ..
        }) => match &job.accelerator {
            Accelerator::Cuda { name, .. } if !cuda => bail!(
                "PyTorch does not see the NVIDIA GPU {name}: the driver may be too old for the \
                 PyTorch {} build; update the NVIDIA driver and try again",
                lane.as_str()
            ),
            Accelerator::Cuda {
                name,
                compute_capability,
                ..
            } if cuda_ok == Some(false) => bail!(
                "a test kernel failed on {name} (compute capability {compute_capability}): the \
                 PyTorch {} build has no kernels for it",
                lane.as_str()
            ),
            Accelerator::Mps { .. } if !mps => bail!(
                "PyTorch reports no Apple MPS device (macOS 14 or newer on Apple silicon is needed)"
            ),
            _ => {}
        },
        _ => bail!("the self-check printed no device report; the training script is damaged"),
    }
    enter!(TrainStage::SelfCheck, 100, "PyTorch sees the GPU");
    check(cancel)?;
    let finish = Finish {
        job,
        base,
        hooks,
        cancel,
    };
    if let Some(retry) = &job.export_only {
        let data_dir = job.run_dir.join("data");
        let prepared = dataset::read_prepared(&data_dir)?;
        let staging = models.join(format!(".{}.train-staging", job.name));
        // Held until this returns, then removed: a crash leaves it to the storage sweep.
        let _staging = crate::storage::Busy::hold(&staging).remove_target_on_drop();
        manager::remove_any(&staging)?;
        std::fs::create_dir_all(&staging)
            .with_context(|| format!("creating {}", staging.display()))?;
        enter!(TrainStage::Train, 0, "exporting the best checkpoint");
        let work = super::retry::work_dir(&job.run_dir);
        let args: Vec<OsString> = vec![
            train_script.clone().into_os_string(),
            "--trainer".into(),
            trainer.into(),
            "--family".into(),
            family.as_str().into(),
            "--size".into(),
            recipe.size.clone().into(),
            "--name".into(),
            job.name.clone().into(),
            "--data".into(),
            data_dir.into_os_string(),
            "--imgsz".into(),
            recipe.imgsz.to_string().into(),
            "--out-dir".into(),
            staging.clone().into_os_string(),
            "--work".into(),
            work.clone().into_os_string(),
            "--export-script".into(),
            export_script.into_os_string(),
            "--export-only".into(),
            retry.checkpoint.clone().into_os_string(),
        ];
        let cmd = env::train_python_cmd(
            paths,
            job.platform,
            Stage::Export,
            &job.run_dir,
            EXPORT_ONLY_TIMEOUT,
            args,
        );
        let mut done: Option<TrainEvent> = None;
        let res = runner.run(
            &cmd,
            &mut |l| {
                log.line(l);
                match progress::parse_line(l) {
                    Some(ev @ TrainEvent::Done { .. }) => done = Some(ev),
                    Some(TrainEvent::Phase { detail, .. }) => report(TrainReport::Stage {
                        stage: TrainStage::Train,
                        percent: 50,
                        detail: &detail,
                    }),
                    Some(_) => {}
                    None => report(TrainReport::Line(l)),
                }
            },
            cancel,
        );
        if let Err(e) = res {
            let _ = manager::remove_any(&staging);
            let e = explain_full_disk(e, models);
            return Err(env::explain_train_failure(
                e,
                paths,
                job.platform,
                "the export retry",
            ));
        }
        check(cancel)?;
        return finish.run(recipe, prepared, done, &staging, &mut log, report, current);
    }

    // 3. Weights (and the public subset).
    let wdir = paths.weights_dir(base);
    let mut downloads: Vec<(&'static Resource, PathBuf)> = vec![(base, wdir.clone())];
    let public_dir = super::public_subset_dir(&job.root);
    if job.request.include_public {
        let Some(public) = catalog::lila_night() else {
            bail!(
                "the public night subset is not pinned in this build (its manifest lists no \
                 image); train without it"
            );
        };
        super::check_public_download(&job.root, job.request.allow_large)?;
        downloads.push((public, public_dir.clone()));
    }
    let missing: Vec<_> = downloads
        .iter()
        .filter(|(r, d)| !manager::is_installed(r, d))
        .collect();
    if missing.is_empty() {
        enter!(TrainStage::Weights, 100, "already downloaded");
    } else {
        let total: u64 = missing.iter().map(|(r, _)| r.size()).sum();
        let mut done = 0u64;
        for (r, dir) in missing {
            enter!(TrainStage::Weights, manager::percent(done, total), r.title);
            fetcher.fetch(
                r,
                dir,
                &mut |b, _| {
                    let d = format!(
                        "{} {} of {} MB",
                        r.title,
                        (done + b) / 1_000_000,
                        total / 1_000_000
                    );
                    report(TrainReport::Stage {
                        stage: TrainStage::Weights,
                        percent: manager::percent(done + b, total),
                        detail: &d,
                    });
                },
                cancel,
            )?;
            if !manager::is_installed(r, dir) {
                bail!(
                    "{} is not complete in {} after the download",
                    r.title,
                    dir.display()
                );
            }
            done += r.size();
            check(cancel)?;
        }
    }

    // 4. Dataset.
    enter!(TrainStage::Dataset, 0, "reading the exports");
    if job.request.datasets.is_empty() {
        bail!("no dataset: choose at least one review export folder");
    }
    let sources = read_sources(
        &job.request.datasets,
        job.request.include_public.then_some(public_dir.as_path()),
    )?;
    let opts = request_options(&job.options, &job.request);
    let plan = preflight::planned(&sources, &opts)?;
    preflight::check_plan(&plan, &sources, &opts)?;
    let data_dir = job.run_dir.join("data");
    manager::remove_any(&data_dir)?;
    preflight::check_disk(preflight::run_needs(
        &plan,
        &sources,
        recipe,
        &job.run_dir,
        models,
    )?)?;
    if let Some(w) = preflight::gpu_warning(recipe, &job.accelerator) {
        log.line(&format!("gpu: {w}"));
        tracing::warn!("training {}: {w}", job.name);
    }
    let prepared = dataset::write(&sources, &plan, &opts, &data_dir)
        .map_err(|e| explain_full_disk(e, &job.run_dir))?;
    for w in &prepared.warnings {
        log.line(&format!("dataset: {w}"));
    }
    if let Some(t) = dataset::valid_groups_text(&prepared.valid_groups) {
        log.line(&format!("dataset: {t}"));
    }
    if let Some(b) = &prepared.balance_summary {
        log.line(&format!("dataset: balanced: {b}"));
    }
    report(TrainReport::Prepared(&prepared));
    let detail = format!(
        "{} training, {} validation, {} held-out images; classes {}",
        prepared.train_images,
        prepared.valid_images,
        prepared.heldout_images,
        prepared.classes.join(", ")
    );
    enter!(TrainStage::Dataset, 100, &detail);
    if family.service_family() == ModelFamilyKind::Detr && prepared.classes.len() == 91 {
        bail!("91 classes look like a COCO category-id head to the service; add or drop a class");
    }
    check(cancel)?;

    // 5. Train (the script also exports and checks the ONNX).
    enter!(TrainStage::Train, 0, "starting the trainer");
    // A small validation split: a recommended patience doubles (its metrics are noisy).
    let (patience, convergence_line) = recipe.convergence_patience(prepared.valid_images);
    let patience_doubled = patience != recipe.patience;
    let recipe = &TrainRecipe {
        patience,
        ..recipe.clone()
    };
    // RF-DETR: at a stop point the learning rate is lowered over a few more epochs first.
    let anneal = recipe.anneal_epochs();
    let anneal_text = if anneal > 0 {
        format!(
            "; at a stop point the learning rate is lowered over {anneal} more epochs, then \
             training stops"
        )
    } else {
        String::new()
    };
    log.line(&format!("# {convergence_line}{anneal_text}"));
    if let Some(seed) = job.request.train_seed {
        log.line(&format!(
            "# training seed {seed} (the frames are the same for every seed)"
        ));
    }
    // RF-DETR on a small training split: a lower learning rate than rfdetr's default. Unique
    // frames: balancing repeats do not make a dataset larger.
    let lr = recipe.effective_lr(prepared.train_frames);
    log.line(&format!("# {}", recipe.lr_line(prepared.train_frames)));
    if !recipe.early_stopping() {
        let w = format!(
            "early stopping is off (patience {} of {} epochs): every epoch runs; the best one is \
             still exported",
            recipe.patience, recipe.epochs
        );
        log.line(&format!("# {w}"));
        report(TrainReport::Line(&w));
    }
    let staging = models.join(format!(".{}.train-staging", job.name));
    // Held until this returns, then removed (also on the early errors that skip their own
    // cleanup): a crash leaves it to the storage sweep ([`crate::storage::sweep`]).
    let _staging = crate::storage::Busy::hold(&staging).remove_target_on_drop();
    manager::remove_any(&staging)?;
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let work = job.run_dir.join("work");
    let weights_arg = match family.trainer() {
        catalog::Trainer::Transformers => wdir.clone(),
        _ => wdir.join(
            base.parts
                .first()
                .ok_or_else(|| anyhow!("{} has no file", base.id))?
                .file_name,
        ),
    };
    // RF-DETR's multi-scale training (`auto`: on when the estimate fits batch 4 with it).
    let multi_scale = recipe.multi_scale_on(job.accelerator.memory_mb());
    let train_args = |batch: u32, workers: u32| -> Vec<OsString> {
        let mut args: Vec<OsString> = vec![
            train_script.clone().into_os_string(),
            "--trainer".into(),
            trainer.into(),
            "--family".into(),
            family.as_str().into(),
            "--size".into(),
            recipe.size.clone().into(),
            "--name".into(),
            job.name.clone().into(),
            "--weights".into(),
            weights_arg.clone().into_os_string(),
            "--data".into(),
            data_dir.clone().into_os_string(),
            "--imgsz".into(),
            recipe.imgsz.to_string().into(),
            "--epochs".into(),
            recipe.epochs.to_string().into(),
            "--batch".into(),
            batch.to_string().into(),
            "--device".into(),
            job.accelerator.device_arg().into(),
            "--workers".into(),
            workers.to_string().into(),
            "--seed".into(),
            job.request
                .trainer_seed(job.options.seed)
                .to_string()
                .into(),
            "--aug-dark".into(),
            recipe.augment.dark.to_string().into(),
            "--aug-noise".into(),
            recipe.augment.noise.to_string().into(),
            "--aug-gray".into(),
            recipe.augment.gray.to_string().into(),
            "--out-dir".into(),
            staging.clone().into_os_string(),
            "--work".into(),
            work.clone().into_os_string(),
            "--export-script".into(),
            export_script.clone().into_os_string(),
        ];
        if let Some(lr) = lr {
            args.push("--lr".into());
            args.push(lr.lr.to_string().into());
            if let Some(e) = lr.encoder {
                args.push("--lr-encoder".into());
                args.push(e.to_string().into());
            }
        }
        // Only when set: the script resolves each trainer's own default.
        if let Some(s) = recipe.lr_schedule {
            args.push("--lr-scheduler".into());
            args.push(s.as_str().into());
        }
        if let Some(f) = recipe.lr_min_factor {
            args.push("--lr-min-factor".into());
            args.push(f.to_string().into());
        }
        if !recipe.select_metric.is_default() {
            args.push("--select-metric".into());
            args.push(recipe.select_metric.id().into());
        }
        // The convergence stop (one rule for every trainer, `Convergence` in the script).
        let rec = super::recipe::recommended(family);
        for (k, v) in [
            ("--patience", recipe.patience.to_string()),
            ("--min-epochs", rec.min_epochs.to_string()),
            ("--min-delta", rec.min_delta.to_string()),
            ("--smooth", rec.smooth.to_string()),
            ("--anneal", anneal.to_string()),
        ] {
            args.push(k.into());
            args.push(v.into());
        }
        args.push("--stop-file".into());
        args.push(stop_file(&job.run_dir).into_os_string());
        if family == TrainFamily::Rfdetr {
            args.push("--multi-scale".into());
            args.push(if multi_scale { "on" } else { "off" }.into());
        }
        args
    };
    let (mut workers, start_commit) = match job.workers {
        Workers::Auto => {
            let c = super::workers_now(family.trainer());
            log.line(&format!("# {}", c.describe()));
            (c.workers, c.commit)
        }
        Workers::Fixed(n) => {
            log.line(&format!("# data loader workers: {n} (training.workers)"));
            (n, None)
        }
    };
    let mut batch = recipe.batch;
    let mut retries = 0;
    let mut host_retries = 0;
    let mut note: Option<String> = None;
    let (res, done) = loop {
        let mut cmd = env::train_python_cmd(
            paths,
            job.platform,
            Stage::Export,
            &job.run_dir,
            job.timeout,
            train_args(batch, workers),
        );
        if matches!(job.accelerator, Accelerator::Mps { .. }) {
            for (k, v) in MPS_WATERMARKS {
                // A value in the service's own environment wins (e.g. 0.0 = no limit).
                if std::env::var_os(k).is_none() {
                    cmd.env.push((k.to_string(), v.into()));
                }
            }
        }
        // A finish request of an earlier attempt (or run) does not end this one at its first epoch.
        let stop = stop_file(&job.run_dir);
        if let Err(e) = std::fs::remove_file(&stop)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            bail!("cannot remove the old stop request {}: {e}", stop.display());
        }
        let started = Instant::now();
        let mut tp = TrainProgress {
            note: note.clone(),
            patience: recipe.patience,
            min_delta: super::recipe::recommended(family).min_delta_f64(),
            min_epochs: super::recipe::recommended(family).min_epochs,
            smooth: super::recipe::recommended(family).smooth,
            batch_size: Some(batch),
            imgsz: Some(recipe.imgsz),
            settings: Some(progress::RunSettings {
                lr: lr.map(|l| l.lr),
                lr_encoder: lr.and_then(|l| l.encoder),
                lr_schedule: recipe.effective_lr_schedule(),
                lr_min_factor: recipe.effective_lr_min_factor(),
                select_metric: recipe.select_metric,
                epochs: recipe.epochs,
                patience: recipe.patience,
                patience_doubled,
                batch: recipe.batch,
                workers: Some(workers),
                seed: job.request.train_seed,
                scene: Some(job.request.scene.as_str().to_string()),
                gpu: Some(job.accelerator.name().to_string()),
            }),
            ..TrainProgress::default()
        };
        let mut done: Option<TrainEvent> = None;
        let res = runner.run(
            &cmd,
            &mut |l| {
                log.line(l);
                match progress::parse_line(l) {
                    Some(ev @ TrainEvent::Done { .. }) => done = Some(ev),
                    Some(ev) => {
                        tp.apply(&ev, started.elapsed().as_secs_f64());
                        let pct = super::estimate::to_u64(tp.fraction() * 100.0).min(100);
                        let d = tp.describe();
                        report(TrainReport::Stage {
                            stage: TrainStage::Train,
                            percent: u8::try_from(pct).unwrap_or(100),
                            detail: &d,
                        });
                        report(TrainReport::Progress(&tp));
                    }
                    None => report(TrainReport::Line(l)),
                }
            },
            cancel,
        );
        let lines = res.as_ref().err().and_then(|e| {
            e.downcast_ref::<crate::resources::export::ChildFailed>()
                .map(|c| c.lines.as_slice())
        });
        let host_oom = lines.is_some_and(|l| host_out_of_memory_line(l).is_some());
        let gpu_oom = lines.is_some_and(|l| out_of_memory_line(l).is_some());
        let cancelled = cancel.load(Ordering::Relaxed);
        let keep = keeps_checkpoints(&tp, family.trainer(), &work);
        // Out of system memory (commit): fewer workers, same batch (the worker count does not
        // change the result).
        let retry = if host_oom && !cancelled && !keep && host_retries < OOM_RETRIES && workers > 0
        {
            let next = workers / 2;
            host_retries += 1;
            let msg = format!(
                "out of system memory with {}, retrying with {next}",
                super::workers_text(workers)
            );
            workers = next;
            Some((msg, None))
        } else if gpu_oom
            && !cancelled
            && retries < OOM_RETRIES
            && !keep
            && batch > 1
            && tp.history.len() < OOM_RETRY_EPOCHS
        {
            let next = (batch / 2).max(1);
            retries += 1;
            let msg = format!("out of GPU memory at batch {batch}, retrying with batch {next}");
            batch = next;
            Some((msg, Some(next)))
        } else {
            None
        };
        if let Some((msg, new_batch)) = retry {
            tracing::warn!("training {}: {msg}", job.id);
            log.line(&format!("# {msg}"));
            report(TrainReport::Line(&msg));
            if let Some(b) = new_batch {
                report(TrainReport::Batch(b));
            }
            report(TrainReport::Stage {
                stage: TrainStage::Train,
                percent: 0,
                detail: &msg,
            });
            let _ = manager::remove_any(&work);
            manager::remove_any(&staging)?;
            std::fs::create_dir_all(&staging)
                .with_context(|| format!("creating {}", staging.display()))?;
            note = Some(msg);
            continue;
        }
        break (res, done);
    };
    let finished_recipe = TrainRecipe {
        batch,
        ..recipe.clone()
    };
    if let Err(e) = res {
        let _ = manager::remove_any(&staging);
        if super::import::disk::is_disk_full(&e) {
            return Err(explain_full_disk(e, &job.run_dir));
        }
        if let Some(why) = e
            .downcast_ref::<crate::resources::export::ChildFailed>()
            .and_then(|c| {
                let host = HostMemory {
                    windows: job.platform.os == "windows",
                    workers,
                    commit: start_commit,
                };
                classify_train_failure(&c.lines, &finished_recipe, &host)
            })
        {
            return Err(e.context(why));
        }
        return Err(env::explain_train_failure(
            e,
            paths,
            job.platform,
            TRAINING_TASK,
        ));
    }
    check(cancel)?;
    finish.run(
        &finished_recipe,
        prepared,
        done,
        &staging,
        &mut log,
        report,
        current,
    )
}

/// What a run does once the script exported a model, trained or from a checkpoint.
struct Finish<'a> {
    job: &'a TrainJob,
    base: &'static Resource,
    hooks: &'a dyn Hooks,
    cancel: &'a AtomicBool,
}

impl Finish<'_> {
    /// The script's `done` report against the run: it must exist, name the family the recipe
    /// exports and carry the dataset's classes. Returns the service family and the export's
    /// device and seconds (from the retried run when the training ran earlier). A failure leaves
    /// the staging folder to the caller (`run` removes it; the job's guard is the backstop).
    fn validate(
        family: TrainFamily,
        classes_expected: &[String],
        done: Option<TrainEvent>,
        export_only: Option<&ExportRetry>,
    ) -> Result<(ModelFamilyKind, Option<String>, Option<f64>)> {
        let Some(TrainEvent::Done {
            family: fam,
            classes,
            device,
            seconds,
            ..
        }) = done
        else {
            bail!("the trainer finished without reporting its exported model");
        };
        let service_family: ModelFamilyKind = fam.parse()?;
        if service_family != family.service_family() {
            bail!(
                "the trainer exported a {service_family} model, expected {}",
                family.service_family()
            );
        }
        if classes != classes_expected {
            bail!("the trained model has classes {classes:?}, the dataset {classes_expected:?}");
        }
        // Trained earlier: the script reports the export's own device and seconds.
        Ok(match export_only {
            Some(r) => (service_family, Some(r.device.clone()), r.train_seconds),
            None => (service_family, device, seconds),
        })
    }

    /// Check the script's report, install (on the record at once), register (the checkpoints go
    /// then: nothing is left to redo) and benchmark.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        recipe: &TrainRecipe,
        prepared: Prepared,
        done: Option<TrainEvent>,
        staging: &Path,
        log: &mut Log,
        report: &mut dyn FnMut(TrainReport<'_>),
        current: &mut Option<TrainStage>,
    ) -> Result<RunOutcome> {
        let Self {
            job,
            base,
            hooks,
            cancel,
        } = *self;
        let family = recipe.family;
        let (service_family, device, seconds) =
            match Self::validate(family, &prepared.classes, done, job.export_only.as_ref()) {
                Ok(v) => v,
                Err(e) => {
                    let _ = manager::remove_any(staging);
                    return Err(e);
                }
            };

        // 6. Install.
        enter(
            current,
            report,
            TrainStage::Install,
            0,
            "checking the output",
        );
        let installed = install(
            job,
            recipe,
            staging,
            &prepared,
            base,
            service_family,
            device.as_deref(),
            seconds,
        );
        let _ = manager::remove_any(staging);
        let output = installed.map_err(|e| explain_full_disk(e, job.install_dir()))?;
        report(TrainReport::Installed(&output));
        enter(
            current,
            report,
            TrainStage::Install,
            100,
            &output.onnx.display().to_string(),
        );
        check(cancel)?;

        // 7. Register.
        enter(current, report, TrainStage::Register, 0, &output.name);
        let registered = hooks.register(&output)?;
        report(TrainReport::Registered(&registered));
        if !job.keep_work {
            let _ = manager::remove_any(&super::retry::work_dir(&job.run_dir));
        }
        enter(current, report, TrainStage::Register, 100, &registered);

        // 8. Benchmark (best effort).
        let (benchmark, benchmark_error) =
            if output.heldout.is_some() || !job.request.benchmark_sets.is_empty() {
                enter(
                    current,
                    report,
                    TrainStage::Benchmark,
                    0,
                    "waiting for the benchmark runner",
                );
                match hooks.benchmark(
                    &output,
                    &job.request.benchmark_sets,
                    &job.request.benchmark_devices,
                    cancel,
                ) {
                    Ok(b) => (b, None),
                    Err(e) if e.downcast_ref::<Cancelled>().is_some() => return Err(e),
                    Err(e) => {
                        let msg = format!("{e:#}");
                        log.line(&format!("benchmark failed: {msg}"));
                        (None, Some(msg))
                    }
                }
            } else {
                (None, None)
            };
        // 8b. The current default on the same frames (best effort too).
        let baseline = if benchmark.is_some() {
            enter(
                current,
                report,
                TrainStage::Benchmark,
                50,
                "comparing with the default model",
            );
            match hooks.baseline(
                &output,
                &job.request.benchmark_sets,
                &job.request.benchmark_devices,
                &job.run_dir,
                cancel,
            ) {
                Ok(b) => b,
                Err(e) if e.downcast_ref::<Cancelled>().is_some() => return Err(e),
                Err(e) => {
                    let msg = format!("{e:#}");
                    log.line(&format!("comparison with the default model failed: {msg}"));
                    Some(Baseline {
                        note: Some(msg),
                        ..Baseline::default()
                    })
                }
            }
        } else {
            None
        };
        enter(current, report, TrainStage::Benchmark, 100, "done");
        Ok(RunOutcome {
            output,
            prepared,
            registered,
            benchmark,
            benchmark_error,
            baseline,
        })
    }
}

/// Report `stage` and remember it as the one a failure happens in.
fn enter(
    current: &mut Option<TrainStage>,
    report: &mut dyn FnMut(TrainReport<'_>),
    stage: TrainStage,
    percent: u8,
    detail: &str,
) {
    *current = Some(stage);
    report(TrainReport::Stage {
        stage,
        percent,
        detail,
    });
}

/// What the failure messages call a training run ([`env::explain_train_failure`]).
const TRAINING_TASK: &str = "training";

/// A line of `lines` that reports the host out of memory (RAM + paging file, the Windows
/// commit), if any: of the most telling kind present (a paging file too small, then a failed CPU
/// allocation, ...), the last such line.
pub fn host_out_of_memory_line(lines: &[String]) -> Option<&String> {
    const KINDS: [&[&str]; 4] = [
        &["[WinError 1455]", "paging file is too small"],
        &["DefaultCPUAllocator: not enough memory", "[WinError 1450]"],
        &[
            "can't start new thread",
            "Cannot allocate memory",
            "[Errno 12]",
        ],
        &["is killed by signal: Killed"],
    ];
    KINDS
        .iter()
        .find_map(|pats| {
            lines
                .iter()
                .rev()
                .find(|l| pats.iter().any(|p| l.contains(p)))
        })
        // Python's own `MemoryError` only at the start of a line: `torch.OutOfMemoryError` (GPU)
        // contains it too.
        .or_else(|| {
            lines
                .iter()
                .rev()
                .find(|l| l.trim_start().starts_with("MemoryError"))
        })
}

/// The last line of `lines` that reports the GPU out of memory, if any. None when the host ran
/// out of memory too ([`host_out_of_memory_line`]): on Windows (WDDM) a CUDA allocation also
/// fails when the system commit is exhausted, with GPU memory free ("CUDA out of memory ...
/// 6.17 GiB is free"), and a smaller batch does not help there.
pub fn out_of_memory_line(lines: &[String]) -> Option<&String> {
    const PATTERNS: [&str; 5] = [
        "CUDA out of memory",
        "MPS backend out of memory",
        "torch.OutOfMemoryError",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cudaErrorMemoryAllocation",
    ];
    if host_out_of_memory_line(lines).is_some() {
        return None;
    }
    lines
        .iter()
        .rev()
        .find(|l| PATTERNS.iter().any(|p| l.contains(p)))
}

/// What [`classify_train_failure`] says about the host's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// The run is on Windows (the paging file advice).
    pub windows: bool,
    /// Data loader workers of the last attempt.
    pub workers: u32,
    /// The commit when training started (Windows, automatic workers).
    pub commit: Option<super::Commit>,
}

/// The likely cause and next action of a failed training from its last output lines, for the
/// failures the export's classifier does not know (system or GPU memory, an empty or broken
/// dataset).
pub fn classify_train_failure(
    lines: &[String],
    recipe: &TrainRecipe,
    host: &HostMemory,
) -> Option<String> {
    let has = |pats: &[&str]| {
        lines
            .iter()
            .rev()
            .find(|l| pats.iter().any(|p| l.contains(p)))
    };
    if let Some(l) = host_out_of_memory_line(lines) {
        let who = if host.workers == 0 {
            "the training process alone needed more than was free (no data loader workers)"
                .to_string()
        } else {
            format!(
                "each of the {} loads PyTorch (~1 GB each)",
                super::workers_text(host.workers)
            )
        };
        let commit = host
            .commit
            .map(|c| {
                format!(
                    "; commit {:.1} of {:.1} GB was in use when training started",
                    super::gib(c.in_use()),
                    super::gib(c.limit)
                )
            })
            .unwrap_or_default();
        let fix = if host.windows {
            "Let Windows manage the paging file size (System Properties > Advanced > \
             Performance > Virtual memory) or close other programs"
        } else {
            "Close other programs or add swap"
        };
        let force = if host.workers > 0 {
            " (or set training.workers to 0 in the config)"
        } else {
            ""
        };
        return Some(format!(
            "training ran out of system memory ({}): {who}{commit}. {fix}{force}, then train \
             again",
            l.trim()
        ));
    }
    if let Some(l) = out_of_memory_line(lines) {
        return Some(format!(
            "the GPU ran out of memory ({}): train again with a smaller batch (now {}) or input \
             size (now {}), or a smaller model size; close other programs that use the GPU",
            l.trim(),
            recipe.batch,
            recipe.imgsz
        ));
    }
    if let Some(l) = has(&[
        "no kernel image is available",
        "CUDA error: invalid device function",
    ]) {
        return Some(format!(
            "the PyTorch build has no kernels for this GPU ({}); see the Train tab's \
             capability notes",
            l.trim()
        ));
    }
    if let Some(l) = has(&[
        "Dataset images not found",
        "No labels found",
        "No images found",
    ]) {
        return Some(format!(
            "the trainer found no usable images ({}); check the export folder",
            l.trim()
        ));
    }
    None
}

/// `<name>.provenance.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub version: u32,
    pub name: String,
    pub created: String,
    pub run_id: String,
    pub license: String,
    pub license_note: String,
    pub family: TrainFamily,
    pub size: String,
    pub service_family: String,
    pub trainer: String,
    pub recipe: TrainRecipe,
    pub base_weights: BaseWeights,
    pub classes: Vec<String>,
    pub datasets: Vec<dataset::SourceSummary>,
    /// ExDark (non-commercial) is never used for training.
    pub exdark_used: bool,
    pub device: String,
    pub train_seconds: f64,
    pub onnx_sha256: String,
    pub blue_onyx_prism: String,
}

/// The pinned base weights of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseWeights {
    pub id: String,
    pub version: String,
    pub license: String,
    pub files: Vec<(String, String)>,
}

/// Move `a` to `b`: a rename, else (another volume) `copy` to a temporary name in `b`'s folder,
/// flushed and renamed into place. `b` is complete or absent: a partial `.onnx` would pass for the
/// model and refuse its name to the next run. A crash in the copy leaves only a hidden
/// `.<name>.<uuid>.tmp`.
fn move_file(
    a: &Path,
    b: &Path,
    copy: impl Fn(&Path, &Path) -> std::io::Result<u64>,
) -> Result<()> {
    if std::fs::rename(a, b).is_ok() {
        return Ok(());
    }
    let name = b.file_name().unwrap_or_default().to_string_lossy();
    let tmp = b.with_file_name(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()));
    let copied = copy(a, &tmp)
        .and_then(|_| {
            // Opened to write: flushing a read-only handle fails on Windows.
            std::fs::OpenOptions::new()
                .write(true)
                .open(&tmp)?
                .sync_all()
        })
        .and_then(|()| std::fs::rename(&tmp, b));
    if copied.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    copied.with_context(|| format!("moving {} to {}", a.display(), b.display()))
}

#[allow(clippy::too_many_arguments)]
fn install(
    job: &TrainJob,
    recipe: &TrainRecipe,
    staging: &Path,
    prepared: &Prepared,
    base: &Resource,
    family: ModelFamilyKind,
    device: Option<&str>,
    seconds: Option<f64>,
) -> Result<TrainOutput> {
    let models = job.install_dir();
    let name = &job.name;
    let (onnx_name, yaml_name) = (format!("{name}.onnx"), format!("{name}.yaml"));
    let src_onnx = staging.join(&onnx_name);
    let src_yaml = staging.join(&yaml_name);
    let len = std::fs::metadata(&src_onnx)
        .with_context(|| format!("the trainer wrote no {onnx_name}"))?
        .len();
    if len < 1024 {
        bail!("{onnx_name} is only {len} bytes");
    }
    let yaml_text = std::fs::read_to_string(&src_yaml)
        .with_context(|| format!("the trainer wrote no {yaml_name}"))?;
    let names = crate::model::classes::parse_class_names(&yaml_text)
        .with_context(|| format!("parsing {yaml_name}"))?;
    if names != prepared.classes {
        bail!(
            "{yaml_name} lists {names:?}, the dataset {:?}",
            prepared.classes
        );
    }
    let fam = recipe.family;
    let prov = Provenance {
        version: 1,
        name: name.clone(),
        created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        run_id: job.id.clone(),
        license: fam.license().into(),
        license_note: fam.notice().into(),
        family: fam,
        size: recipe.size.clone(),
        service_family: family.name().into(),
        trainer: fam.trainer().as_str().into(),
        recipe: recipe.clone(),
        base_weights: BaseWeights {
            id: base.id.into(),
            version: base.version.into(),
            license: base.license.into(),
            files: base
                .parts
                .iter()
                .map(|p| (p.url.to_string(), p.sha256.to_string()))
                .collect(),
        },
        classes: prepared.classes.clone(),
        datasets: dataset::read_summary_sources(&prepared.dir),
        exdark_used: false,
        device: device.unwrap_or_default().into(),
        train_seconds: seconds.unwrap_or_default(),
        onnx_sha256: manager::sha256_file(&src_onnx)?,
        blue_onyx_prism: crate::long_version(),
    };
    std::fs::create_dir_all(models).with_context(|| format!("creating {}", models.display()))?;
    let dst_onnx = models.join(&onnx_name);
    let dst_yaml = models.join(&yaml_name);
    let dst_prov = provenance_path(models, name);
    if dst_onnx.exists() {
        bail!(
            "{} appeared during the run; not replaced",
            dst_onnx.display()
        );
    }
    let mv = |a: &Path, b: &Path| move_file(a, b, |from, to| std::fs::copy(from, to));
    // `.yaml` and the provenance first, the `.onnx` (what makes the model present) last.
    crate::collect::layout::write_atomic(&dst_prov, &serde_json::to_vec_pretty(&prov)?)
        .with_context(|| format!("writing {}", dst_prov.display()))?;
    let r = mv(&src_yaml, &dst_yaml).and_then(|()| mv(&src_onnx, &dst_onnx));
    if let Err(e) = r {
        for p in [&dst_yaml, &dst_prov, &dst_onnx] {
            let _ = std::fs::remove_file(p);
        }
        return Err(e.context(format!(
            "installing {onnx_name} into {} failed and was undone; check free disk space",
            models.display()
        )));
    }
    Ok(TrainOutput {
        name: name.clone(),
        onnx: dst_onnx,
        yaml: dst_yaml,
        provenance: dst_prov,
        family,
        classes: prepared.classes.clone(),
        license: fam.license().into(),
        heldout: prepared.heldout_dir.clone(),
        device: device.unwrap_or_default().into(),
        train_seconds: seconds.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests;
