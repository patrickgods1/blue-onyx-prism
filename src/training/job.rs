//! One training run as a stage machine ([`run_training`]) over the model export's seams
//! ([`CommandRunner`] for processes, [`Fetcher`] for verified downloads) plus [`Hooks`] for what
//! the caller owns (adding the model to the config, the benchmark).
//!
//! Stages ([`TrainStage`]): setup (uv, Python, the training lock's packages, the Visual C++
//! runtime; shared code with the export, under its lock) -> self-check (`train_model.py
//! --self-check`: PyTorch imports and sees the GPU) -> weights (the pinned base weights, and the
//! public subset when asked) -> dataset (assembled in Rust, [`super::dataset`]) -> train (the
//! script trains and exports a static ONNX, checked against PyTorch with ONNX Runtime; progress
//! from its `@train` events) -> install (`models/<name>.onnx`, `.yaml` and
//! `<name>.provenance.json`) -> register ([`Hooks::register`], the `add-model` code) ->
//! benchmark ([`Hooks::benchmark`] on the held-out `dir:` set; a failure there is reported but
//! does not undo the model). Cancellation kills the child's process tree (the runner's job).

use super::capability::Accelerator;
use super::dataset::{self, Prepared, SceneFilter};
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
}

/// One queued run.
#[derive(Debug, Clone)]
pub struct TrainJob {
    pub id: String,
    pub request: TrainRequest,
    /// The trained model's name (`models/<name>.onnx`, `/v1/vision/custom/<name>`).
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
    pub workers: u32,
    /// Keep the checkpoints (`<run>/work`) after the run.
    pub keep_work: bool,
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
    /// AP50 over the scored classes.
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub images: usize,
}

/// What the caller does for a run.
pub trait Hooks: Send + Sync {
    /// Add the installed model to the config (the `add-model` code); returns a message.
    fn register(&self, out: &TrainOutput) -> Result<String>;
    /// Benchmark the model on its held-out set and `extra` built-in sets.
    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        cancel: &AtomicBool,
    ) -> Result<Option<BenchSummary>>;
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

/// Log file of a run.
pub const LOG_FILE: &str = "train.log";
/// Largest log written (bytes); later lines are dropped with a note.
pub const MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;
/// Timeout of the self-check.
const SELF_CHECK_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Run every stage of `job` (see the module docs).
pub fn run_training(
    job: &TrainJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
) -> Result<RunOutcome, TrainError> {
    let mut stage = None;
    let r = stages(job, runner, fetcher, hooks, report, cancel, &mut stage);
    r.map_err(|e| {
        let cancelled = e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed);
        TrainError {
            stage,
            message: if cancelled {
                "cancelled".into()
            } else {
                format!("{e:#}")
            },
            cancelled,
        }
    })
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
    let models = &paths.export.models;
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
    let _lock = TrainLock::try_acquire(&paths.train_lock_file())?;

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
        .map_err(|e| env::explain_train_failure(e, paths, job.platform))
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

    // 3. Weights (and the public subset).
    let wdir = paths.weights_dir(base);
    let mut downloads: Vec<(&'static Resource, PathBuf)> = vec![(base, wdir.clone())];
    let public_dir = job.root.join(catalog::LILA_NIGHT_DEST);
    if job.request.include_public {
        let Some(public) = catalog::lila_night() else {
            bail!(
                "the public night subset is not pinned in this build (its manifest lists no \
                 image); train without it"
            );
        };
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
    let mut sources = Vec::new();
    for d in &job.request.datasets {
        sources.push(dataset::read_export(d).with_context(|| format!("reading {}", d.display()))?);
    }
    if job.request.include_public {
        let m = catalog::lila_night_manifest()
            .ok_or_else(|| anyhow!("the public subset's manifest does not parse"))?;
        sources.push(dataset::read_public_set(&public_dir, m)?);
    }
    let mut opts = job.options.clone();
    opts.scene = job.request.scene;
    opts.classes = job.request.classes.clone();
    opts.rename = job.request.rename.clone();
    let plan = dataset::plan(&sources, &opts)?;
    let data_dir = job.run_dir.join("data");
    manager::remove_any(&data_dir)?;
    let prepared = dataset::write(&sources, &plan, &opts, &data_dir)?;
    for w in &prepared.warnings {
        log.line(&format!("dataset: {w}"));
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
    let staging = models.join(format!(".{}.train-staging", job.name));
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
    let mut args: Vec<OsString> = vec![
        train_script.into_os_string(),
        "--trainer".into(),
        trainer.into(),
        "--family".into(),
        family.as_str().into(),
        "--size".into(),
        recipe.size.clone().into(),
        "--name".into(),
        job.name.clone().into(),
        "--weights".into(),
        weights_arg.into_os_string(),
        "--data".into(),
        data_dir.clone().into_os_string(),
        "--imgsz".into(),
        recipe.imgsz.to_string().into(),
        "--epochs".into(),
        recipe.epochs.to_string().into(),
        "--batch".into(),
        recipe.batch.to_string().into(),
        "--device".into(),
        job.accelerator.device_arg().into(),
        "--workers".into(),
        job.workers.to_string().into(),
        "--seed".into(),
        job.options.seed.to_string().into(),
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
        export_script.into_os_string(),
    ];
    if let Some(lr) = recipe.lr {
        args.push("--lr".into());
        args.push(lr.to_string().into());
    }
    let cmd = env::train_python_cmd(
        paths,
        job.platform,
        Stage::Export,
        &job.run_dir,
        job.timeout,
        args,
    );
    let started = Instant::now();
    let mut tp = TrainProgress::default();
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
    if !job.keep_work {
        let _ = manager::remove_any(&work);
    }
    if let Err(e) = res {
        let _ = manager::remove_any(&staging);
        if let Some(why) = e
            .downcast_ref::<crate::resources::export::ChildFailed>()
            .and_then(|c| classify_train_failure(&c.lines, recipe))
        {
            return Err(e.context(why));
        }
        return Err(env::explain_train_failure(e, paths, job.platform));
    }
    check(cancel)?;
    let Some(TrainEvent::Done {
        family: fam,
        classes,
        device,
        seconds,
        ..
    }) = done
    else {
        let _ = manager::remove_any(&staging);
        bail!("the trainer finished without reporting its exported model");
    };
    let service_family: ModelFamilyKind = fam.parse()?;
    if service_family != family.service_family() {
        bail!(
            "the trainer exported a {service_family} model, expected {}",
            family.service_family()
        );
    }
    if classes != prepared.classes {
        bail!(
            "the trained model has classes {classes:?}, the dataset {:?}",
            prepared.classes
        );
    }

    // 6. Install.
    enter!(TrainStage::Install, 0, "checking the output");
    let installed = install(
        job,
        &staging,
        &prepared,
        base,
        service_family,
        device.as_deref(),
        seconds,
    );
    let _ = manager::remove_any(&staging);
    let output = installed?;
    enter!(TrainStage::Install, 100, &output.onnx.display().to_string());
    check(cancel)?;

    // 7. Register.
    enter!(TrainStage::Register, 0, &output.name);
    let registered = hooks.register(&output)?;
    enter!(TrainStage::Register, 100, &registered);

    // 8. Benchmark (best effort).
    let (benchmark, benchmark_error) =
        if output.heldout.is_some() || !job.request.benchmark_sets.is_empty() {
            enter!(TrainStage::Benchmark, 0, "waiting for the benchmark runner");
            match hooks.benchmark(&output, &job.request.benchmark_sets, cancel) {
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
    enter!(TrainStage::Benchmark, 100, "done");
    Ok(RunOutcome {
        output,
        prepared,
        registered,
        benchmark,
        benchmark_error,
    })
}

/// The likely cause and next action of a failed training from its last output lines, for the
/// failures the export's classifier does not know (GPU memory, an empty or broken dataset).
pub fn classify_train_failure(lines: &[String], recipe: &TrainRecipe) -> Option<String> {
    let has = |pats: &[&str]| {
        lines
            .iter()
            .rev()
            .find(|l| pats.iter().any(|p| l.contains(p)))
    };
    if let Some(l) = has(&[
        "CUDA out of memory",
        "MPS backend out of memory",
        "torch.OutOfMemoryError",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cudaErrorMemoryAllocation",
    ]) {
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
            "the PyTorch build has no kernels for this GPU ({}); see the Training page's \
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

fn install(
    job: &TrainJob,
    staging: &Path,
    prepared: &Prepared,
    base: &Resource,
    family: ModelFamilyKind,
    device: Option<&str>,
    seconds: Option<f64>,
) -> Result<TrainOutput> {
    let models = &job.paths.export.models;
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
    let recipe = &job.request.recipe;
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
    let mv = |a: &Path, b: &Path| -> Result<()> {
        if std::fs::rename(a, b).is_ok() {
            return Ok(());
        }
        std::fs::copy(a, b)
            .map(|_| ())
            .with_context(|| format!("moving {} to {}", a.display(), b.display()))
    };
    // `.yaml` and the provenance first, the `.onnx` (what makes the model present) last.
    std::fs::write(&dst_prov, serde_json::to_vec_pretty(&prov)?)
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
