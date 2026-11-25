//! One training run as a stage machine ([`run_training`]) over the model export's seams
//! ([`CommandRunner`] for processes, [`Fetcher`] for verified downloads) plus [`Hooks`] for what
//! the caller owns (adding the model to the config, the benchmark).
//!
//! Stages ([`TrainStage`]): setup (uv, Python, the training lock's packages, the Visual C++
//! runtime; shared code with the export, under its lock) -> self-check (`train_model.py
//! --self-check`: PyTorch imports and sees the GPU; skipped when nothing it checks changed since
//! the last one passed, [`super::self_check`]) -> weights (the pinned base weights, and the
//! public subset when asked) -> dataset (assembled in Rust, [`super::dataset`]; a cancel stops it
//! between frames and removes the half-written `<run>/data`) -> train (the
//! script trains and exports a static ONNX, checked against PyTorch with ONNX Runtime; progress
//! from its `@train` events) -> install (`<name>.onnx`, `.yaml` and `<name>.provenance.json`
//! in the models folder or the request's `install_dir`, [`TrainJob::install_dir`]) -> register
//! ([`Hooks::register`], the `add-model` code, with the installed path) -> benchmark
//! ([`Hooks::benchmark`] on the held-out `dir:` set; a failure there is reported but does not
//! undo the model). Cancellation kills the child's process tree (the runner's job). A stop for the
//! process's exit ([`run_training_or_exit`]) kills it the same way but is no cancel: the run is
//! interrupted and keeps its checkpoints.
//!
//! The installed model goes on the run's record at once ([`TrainReport::Installed`]): from then on
//! the run owns a file that blocks its name, however it ends. The checkpoints (`<run>/work`) go once
//! the model is registered, or when the run is cancelled before it installed. A failed or
//! interrupted run keeps them: training and export share one process, so a late failure would
//! otherwise cost the whole training. A failure (not an interruption) deletes only the scratch no
//! Retry export reads ([`FAILED_RUN_SCRATCH`]) once the trainer is gone. [`TrainJob::export_only`] exports such a run again
//! ([`super::retry`]): setup, self-check, then only the export and the stages after it. A run out
//! of memory restarts only while that costs little ([`keeps_checkpoints`]).
//!
//! A run holds the training lock (`tools/.train.lock`) and its own busy marker (`<run dir>.busy`,
//! [`crate::storage::busy`]) from the lock to its end, so another process sees that it trains
//! ([`super::service::run_in_use`]).
//!
//! The stages are one function each in `job/stages.rs` (the driver `stages` reads as the stage
//! list); the out-of-memory restart rule is the pure `next_attempt` in `job/attempt.rs`.

use super::capability::Accelerator;
use super::dataset::{self, Prepared, SceneFilter};
use super::preflight;
use super::progress::{self, TrainEvent, TrainProgress};
use super::pyenv::{self, check};
use super::recipe::TrainRecipe;
use crate::benchmark::metrics::{ApBasis, ClassRecall};
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

mod attempt;
mod stages;
use stages::stages;

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
    /// Start even when the memory commit check would refuse ([`crate::training::commit::gate`]):
    /// "Start anyway" / `--ignore-memory-check`. Kept with the queued job, never in the config.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ignore_memory_check: bool,
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
/// scene filter, classes, renames, excluded mode, balance and the recipe's input size (the size
/// floor regions are judged at it, [`dataset::Options::imgsz`]).
pub fn request_options(base: &dataset::Options, request: &TrainRequest) -> dataset::Options {
    dataset::Options {
        scene: request.scene,
        classes: request.classes.clone(),
        rename: request.rename.clone(),
        excluded: request.excluded,
        balance: request.balance.clone(),
        imgsz: Some(request.recipe.imgsz),
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
    /// Earlier runs [`Workers::Auto`] learns from ([`super::workers::learned`]): filled by the
    /// queue when the run starts (the records of `runs.json` with resources), empty = the limit.
    pub past_runs: Vec<super::service::RunRecord>,
    /// Keep the checkpoints (`<run>/work`) after the run.
    pub keep_work: bool,
    /// Export this failed run's checkpoint instead of training ([`super::retry::plan_retry`]).
    pub export_only: Option<ExportRetry>,
    /// The memory commit a run can take now ([`super::commit::now`]; tests fake it), checked
    /// before the run's first Python process ([`super::trainer_commit_need`]). A dataset check's
    /// fold runs get [`super::commit::checked_already`]: the check judged that need before its
    /// first fold. An export retry is not checked (it trains nothing).
    pub headroom: fn() -> Option<super::commit::Headroom>,
    /// Dataset folders (`<run>/data`) of the other runs of this run's comparison group that
    /// started before it ([`super::service::group_data_dirs`], filled by the queue when it starts
    /// the run): the first whose plan is the same lends its images as hard links
    /// ([`dataset::write_reusing`]). Empty = every image is written from its source.
    pub reuse_from: Vec<PathBuf>,
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
    /// What the training stage used (sent once it ends, also when it failed).
    Resources(&'a super::workers::RunResources),
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
    /// The score on the held-out frames the Review queue picked at random
    /// ([`dataset::HELDOUT_RANDOM`]): nobody chose them, so it is the least biased score. None
    /// without such frames (and for runs of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random: Option<SubsetScore>,
    /// The classes `ap50` / `ap50_95` average (`benchmark::metrics::ApBasis`: those with at
    /// least `MIN_CLASS_GT` objects, else every class with objects); None for runs of older
    /// versions, whose AP averaged every class with objects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ap_basis: Option<ApBasis>,
    /// Benchmarked frames without a person (a parked car does not make a frame "with an
    /// object" here) and, of those, the frames with a person detection at `threshold`: false
    /// person alerts (`benchmark::metrics::ClassAlerts`). None for runs of older versions and
    /// models without a `person` class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_free_frames: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_false_alerts: Option<usize>,
    /// Recall per class on the held-out frames per scene, per camera and per camera in a scene
    /// ([`super::bench::summarize`], from the held-out manifest's own tags only). Empty for runs
    /// of older versions and without held-out frames.
    #[serde(default, skip_serializing_if = "RecallCells::is_empty")]
    pub recall_cells: RecallCells,
}

/// Recall per class of a trained model on its held-out frames, at the benchmark's threshold
/// ([`BenchSummary::threshold`], IoU 0.5), each cell with its integer counts (`gt` objects,
/// `missed` of them). Over every class of the model the held-out set scores (custom animals
/// included), classes with objects only. JSON:
///
/// ```json
/// {"by_scene": {"night": [{"class": "raccoon", "gt": 12, "missed": 5}],
///               "all": [...]},
///  "by_camera": {"back": [...]},
///  "by_camera_scene": [{"camera": "back", "scene": "night", "classes": [...]}]}
/// ```
///
/// `by_scene` is keyed `day` / `night` / `ir` / `all` (every held-out frame, `heldout:all`),
/// `by_camera` by camera over every scene; `by_camera_scene` is a list because a camera name may
/// hold any character (no joined key is unambiguous).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RecallCells {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_scene: BTreeMap<String, Vec<ClassRecall>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_camera: BTreeMap<String, Vec<ClassRecall>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub by_camera_scene: Vec<CameraSceneRecall>,
}

impl RecallCells {
    pub fn is_empty(&self) -> bool {
        self.by_scene.is_empty() && self.by_camera.is_empty() && self.by_camera_scene.is_empty()
    }
}

/// The recall cells of one camera in one scene ([`RecallCells::by_camera_scene`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CameraSceneRecall {
    pub camera: String,
    pub scene: String,
    pub classes: Vec<ClassRecall>,
}

/// The score on a tagged subset of the held-out frames ([`BenchSummary::random`]), from the
/// benchmark's per-tag breakdown: AP over the scored classes, recall at the benchmark's threshold
/// ([`BenchSummary::threshold`], IoU 0.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SubsetScore {
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub recall: Option<f64>,
    /// The classes its AP averages (None for runs of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ap_basis: Option<ApBasis>,
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
    /// [`super::bench::baseline_key`] of what was measured (model file, held-out set, devices,
    /// settings); None when it could not be told (and for runs of older versions): such a result
    /// is never reused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// The run of the same comparison group whose measurement this is, taken over because its
    /// key is equal ([`super::bench::reusable_baseline`]); None when measured for this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from: Option<String>,
}

/// AP of a subset of the held-out frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneAp {
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub images: usize,
    /// The classes its AP averages (None for runs of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ap_basis: Option<ApBasis>,
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
    /// Whether the config now has a model named `name` ([`crate::model::name::same_model_name`]).
    /// Checked when a run starts, before any GPU time (QA-101): the config can gain the name
    /// between the enqueue check and the start, and registration would then refuse the trained
    /// model. The default (tests, fold runs) knows no config.
    fn config_has_model(&self, _name: &str) -> bool {
        false
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
/// Largest log written (bytes; the whole file, a retry export's lines included); later lines are
/// dropped with a note.
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
/// The script sets the same values by `setdefault` for runs by hand (twin, checked by
/// `progress::tests::mps_watermarks_match_the_script`).
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
    run_training_with(job, None, runner, fetcher, hooks, report, cancel, None)
}

/// [`run_training`] for the queue, which can stop for the process's exit: `exit` set together
/// with `cancel` kills the child tree like a cancel, but the run is interrupted, not given up. It
/// keeps its work folder (the checkpoints a Retry export makes the model from) and the self-check
/// marker, and fails with `cancelled: false` and an "interrupted" message, as a run the process
/// stopped under reads after a restart.
pub fn run_training_or_exit(
    job: &TrainJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
    exit: &AtomicBool,
) -> Result<RunOutcome, TrainError> {
    run_training_with(
        job,
        None,
        runner,
        fetcher,
        hooks,
        report,
        cancel,
        Some(exit),
    )
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
    run_training_with(
        job,
        Some(lock),
        runner,
        fetcher,
        hooks,
        report,
        cancel,
        None,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the run's seams and its two stop flags, passed straight through"
)]
fn run_training_with(
    job: &TrainJob,
    held: Option<&TrainLock>,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
    exit: Option<&AtomicBool>,
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
        // Stopped for the process's exit: neither a failure of the run nor the user's cancel.
        let exiting = exit.is_some_and(|x| x.load(Ordering::SeqCst));
        let cancelled =
            !exiting && (e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed));
        let work = super::retry::work_dir(&job.run_dir);
        // A training process that failed: the next run runs the self-check again, whether this
        // one skipped it or not ([`super::self_check`]).
        if !cancelled && !exiting && stage == Some(TrainStage::Train) {
            super::self_check::forget(
                &super::self_check::marker_path(&job.paths),
                &super::self_check::key(
                    job.request.recipe.family.trainer().as_str(),
                    &job.accelerator.device_arg(),
                ),
            );
        }
        if cancelled && !job.keep_work && job.export_only.is_none() && unregistered.is_none() {
            let _ = manager::remove_any(&work);
        }
        // Cancelled while the dataset was written: the half-written `<run>/data` serves nothing
        // (no trainer read it, no later run links from it: its prepared-files record is written
        // last). Only the run's own folder; the exports it was read from are never touched.
        if cancelled && job.export_only.is_none() && stage == Some(TrainStage::Dataset) {
            let data = job.run_dir.join("data");
            if let Err(e) = manager::remove_any(&data) {
                tracing::warn!(
                    run = %job.id,
                    "could not remove the cancelled run's half-written dataset {} ({e:#}); \
                     another program may hold a file in it. The run's Remove on the Jobs tab \
                     deletes it, or delete it by hand",
                    data.display()
                );
            }
        }
        // A failure: the runner has reaped the trainer, so nothing writes `work` any more. What
        // a Retry export never reads goes; the checkpoints and the trainer's logs stay. Not on
        // the way out of the process (a stop must not wait on deleting GBs).
        if !cancelled && !exiting && !job.keep_work && job.export_only.is_none() {
            remove_failed_scratch(&job.id, &work);
        }
        TrainError {
            stage,
            message: match &unregistered {
                _ if exiting => {
                    format!("interrupted: the service stopped{}", kept_note(job, &work))
                }
                _ if cancelled => "cancelled".into(),
                Some(note) => format!("{e:#}. {note}"),
                None => format!("{e:#}{}", kept_note(job, &work)),
            },
            cancelled,
        }
    })
}

/// What a failed run's `work` folder holds that no Retry export reads (paths under `work`, from
/// `scripts/train_model.py`): RF-DETR's dataset layout with its augmented copies (`rfdetr-data`,
/// rebuilt from `<run>/data` by any new run), the export's scratch (`export`,
/// `rfdetr-export-weights`, both recreated by `--export-only`) and RF-DETR's per-epoch
/// `last.ckpt` (only for resuming the training, which is not offered). Never a checkpoint
/// [`super::retry::best_checkpoint`] reads: `hf-best.tmp` stays (a crash between the renames of
/// `save_hf_best` can leave the only whole checkpoint there), as do `training_config.json`,
/// Ultralytics' `run/` and the trainer's logs.
pub const FAILED_RUN_SCRATCH: [&[&str]; 4] = [
    &["rfdetr-data"],
    &["export"],
    &["rfdetr-export-weights"],
    &["rfdetr-out", "last.ckpt"],
];

/// Delete [`FAILED_RUN_SCRATCH`] under the failed run `id`'s `work`, each logged with what it
/// freed ([`crate::storage::size_of`]: hard links into `<run>/data` free nothing). One that cannot
/// be deleted stays for the run's Remove (no sweep touches a run's folder).
fn remove_failed_scratch(id: &str, work: &Path) {
    for parts in FAILED_RUN_SCRATCH {
        let p = parts.iter().fold(work.to_path_buf(), |p, n| p.join(n));
        if std::fs::symlink_metadata(&p).is_err() {
            continue;
        }
        let bytes = crate::storage::size_of(&p);
        match manager::remove_any(&p) {
            Ok(()) => tracing::info!(
                run = %id,
                "removed {} ({}): scratch of the failed run that no Retry export reads",
                p.display(),
                catalog::format_size(bytes)
            ),
            Err(e) => tracing::warn!(
                run = %id,
                "could not remove the failed run's scratch {} ({e:#}); another program may hold \
                 a file in it. The run's Remove on the Jobs tab deletes it, or delete it by hand",
                p.display()
            ),
        }
    }
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

/// The run log, capped at [`MAX_LOG_BYTES`] for the whole file: a retry export of the run appends
/// to the same `train.log` and counts from its size, so the cap holds across reopenings and the
/// cut note is written once.
struct Log {
    file: Option<std::fs::File>,
    written: u64,
    cap: u64,
}

/// The note a cut log ends with (`cap` in whole MB).
fn cut_note(cap: u64) -> String {
    format!("[log cut at {} MB]", cap / (1024 * 1024))
}

impl Log {
    fn open(path: &Path) -> Self {
        Self::open_capped(path, MAX_LOG_BYTES)
    }

    /// [`Log::open`] with another cap (tests use a small one).
    fn open_capped(path: &Path, cap: u64) -> Self {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(|e| tracing::warn!("cannot write the training log {}: {e}", path.display()))
            .ok();
        let written = file.as_mut().map_or(0, |f| Self::already_written(f, cap));
        Self { file, written, cap }
    }

    /// What an existing log already holds against `cap`: its size, or past the cap when it ends
    /// with the cut note (a cut log may stay just under the cap: the dropped line is not in it).
    fn already_written(f: &mut std::fs::File, cap: u64) -> u64 {
        use std::io::{Read, Seek, SeekFrom};
        let len = f.metadata().map_or(0, |m| m.len());
        let note = cut_note(cap);
        let tail = u64::try_from(note.len() + 2).unwrap_or(u64::MAX);
        let mut end = Vec::new();
        let cut = f.seek(SeekFrom::Start(len.saturating_sub(tail))).is_ok()
            && f.read_to_end(&mut end).is_ok()
            && String::from_utf8_lossy(&end).trim_end().ends_with(&note);
        if cut {
            cap.saturating_add(1).max(len)
        } else {
            len
        }
    }

    fn line(&mut self, l: &str) {
        let Some(f) = self.file.as_mut() else { return };
        if self.written > self.cap {
            return;
        }
        self.written += l.len() as u64 + 1;
        let _ = if self.written > self.cap {
            writeln!(f, "{}", cut_note(self.cap))
        } else {
            writeln!(f, "{l}")
        };
    }
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
                // Its recall cells are read by nothing (the Review queue steers by the trained
                // model's): kept out of `runs.json`.
                Ok(b) => b.map(|mut b| {
                    if let Some(s) = b.summary.as_mut() {
                        s.recall_cells = RecallCells::default();
                    }
                    b
                }),
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
        if let Some(b) = &baseline
            && let Some(from) = &b.reused_from
        {
            let l = format!(
                "baseline reused from {from}: {} was measured there on the same held-out frames, \
                 model file, devices and benchmark settings",
                b.model
            );
            log.line(&format!("# {l}"));
            report(TrainReport::Line(&l));
        }
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

pub use super::oom::{host_out_of_memory_line, out_of_memory_line};

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
