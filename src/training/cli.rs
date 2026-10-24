//! `blue-onyx-prism train ...`: what this machine can train (`capability`), estimates
//! (`estimate`), a dry run of the dataset assembly (`prepare`), one or several training runs in a
//! row (`run`), past runs (`list`), removing the training environment (`remove-env`), footage
//! import (`import`) and auto-labelling (`autolabel`, also the self-training re-label).

use super::capability::{self, Capability};
use super::dataset::{self, SceneFilter};
use super::estimate;
use super::job::{BenchSummary, Hooks, TrainOutput, TrainRequest};
use super::recipe::{self, Augment, TrainRecipe};
use super::service::{self, RunState, TrainingService};
use crate::config::Config;
use crate::resources::catalog;
use crate::resources::export::train::{self as env, TrainPaths};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Debug, Clone, clap::Args)]
pub struct TrainArgs {
    #[command(subcommand)]
    pub command: TrainCommand,
}

#[derive(Debug, Clone, clap::Subcommand)]
pub enum TrainCommand {
    /// Show whether this machine can train, on which GPU, and which families and sizes fit.
    Capability {
        /// Print JSON.
        #[arg(long)]
        json: bool,
    },
    /// Estimate GPU memory and time of a run (rough constants, see docs/TRAINING.md).
    Estimate {
        /// Model label: yolo26n, yolo11s, yolov8m, rfdetr-nano, rt-detrv2-s, dfine-s, ...
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 1000)]
        images: usize,
        #[arg(long)]
        epochs: Option<u32>,
        #[arg(long)]
        batch: Option<u32>,
        #[arg(long)]
        imgsz: Option<u32>,
    },
    /// Assemble the dataset of a run without training (prints the counts and warnings).
    Prepare {
        #[command(flatten)]
        data: DataArgs,
        /// Output folder (must not exist or be empty).
        #[arg(long)]
        out: PathBuf,
        /// With --public: allow downloading the public night subset (about 926 MB, over the
        /// 500 MB limit). Also config `allow_large_downloads`.
        #[arg(long)]
        allow_large: bool,
    },
    /// Train one model or several in a row, export, install, add to the config and benchmark.
    Run {
        /// Model label, repeatable (runs one after the other): yolo26n, yolo11s, yolov8m,
        /// rfdetr-nano, rt-detrv2-s, dfine-s, ...
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        #[command(flatten)]
        data: DataArgs,
        #[arg(long)]
        epochs: Option<u32>,
        /// Images per batch (default: the largest the GPU fits by the estimate, at most 16).
        #[arg(long)]
        batch: Option<u32>,
        /// Square input size (default: the size's own, 640 for YOLO).
        #[arg(long)]
        imgsz: Option<u32>,
        /// Learning rate (default: the trainer's).
        #[arg(long)]
        lr: Option<f32>,
        /// Night augmentation preset: darkness 0.5, noise 0.3, grayscale 0.5.
        #[arg(long)]
        night_aug: bool,
        #[arg(long)]
        aug_dark: Option<f32>,
        #[arg(long)]
        aug_noise: Option<f32>,
        #[arg(long)]
        aug_gray: Option<f32>,
        /// Model name (only with one --model; default night-<model>-<stamp> or custom-...).
        #[arg(long)]
        name: Option<String>,
        /// Allow downloads over 500 MB: the training environment and, with --public, the public
        /// night subset (about 926 MB). Also config `allow_large_downloads`.
        #[arg(long)]
        allow_large: bool,
        /// Install the model but do not add it to the config (and do not benchmark it).
        #[arg(long)]
        no_register: bool,
        /// Do not benchmark the trained model.
        #[arg(long)]
        no_benchmark: bool,
        /// Built-in benchmark set to run besides the held-out frames, repeatable (nir-night,
        /// exdark-ir, ...).
        #[arg(long = "benchmark-set")]
        benchmark_sets: Vec<String>,
        /// Keep the checkpoints in <run>/work.
        #[arg(long)]
        keep_work: bool,
        /// Print the trainer's output.
        #[arg(long)]
        verbose: bool,
    },
    /// Import frames from a video, an image folder (Blue Iris clips / alerts), a video URL or
    /// pinned open footage into the dataset folder, queued for review.
    Import(ImportArgs),
    /// Propose boxes for dataset frames (MM-Grounding-DINO + RF-DETR L, checked by Qwen3-VL and
    /// SigLIP 2), or re-label them with a trained model (--with-model).
    Autolabel(AutolabelArgs),
    /// List the runs recorded in the training folder.
    List,
    /// Delete the training environment (tools/train-env, its base weights and settings).
    RemoveEnv,
}

/// `train import`.
#[derive(Debug, Clone, clap::Args)]
#[command(group(clap::ArgGroup::new("from").required(true).args(["path", "url", "dataset", "list_datasets"])))]
pub struct ImportArgs {
    /// A video (.mp4 .mkv .avi .mov), an image, or a folder of them (Blue Iris clip and alert
    /// folders; .bvr clips must be exported to MP4 first).
    #[arg(long)]
    pub path: Option<PathBuf>,
    /// A video page or file URL (downloaded with yt-dlp; needs --license and --confirm-license).
    #[arg(long)]
    pub url: Option<String>,
    /// Pinned open footage (`meva-sample`: MEVA, CC BY 4.0).
    #[arg(long)]
    pub dataset: Option<String>,
    /// List the pinned footage and the imports so far.
    #[arg(long)]
    pub list_datasets: bool,
    /// Camera folder of the frames (default: from the file names, else `import`).
    #[arg(long)]
    pub camera: Option<String>,
    /// Frames to keep.
    #[arg(long, default_value_t = super::import::DEFAULT_FRAMES)]
    pub frames: usize,
    /// FFmpeg scene-change threshold (0..1; 0.01 = motion in a fixed camera, 0.3 = cuts).
    #[arg(long, default_value_t = super::import::DEFAULT_SCENE)]
    pub scene: f32,
    /// Most frames per second kept from scene changes (0 = no limit).
    #[arg(long, default_value_t = super::import::DEFAULT_MAX_FPS)]
    pub max_fps: f32,
    /// Keep one frame every this many seconds anyway (0 = off).
    #[arg(long, default_value_t = super::import::DEFAULT_EVERY_S)]
    pub every: f32,
    /// dHash bits at or below which two frames are duplicates.
    #[arg(long, default_value_t = crate::collect::config::DEFAULT_DEDUP_HAMMING)]
    pub dedup: u32,
    /// Seconds within which two frames of a camera can be duplicates (0 = any time).
    #[arg(long, default_value_t = super::import::DEFAULT_DEDUP_WINDOW_S)]
    pub dedup_window: f32,
    /// siglip (SigLIP 2 embeddings + k-center) or dhash (k-center on the dHashes, no Python for
    /// image folders).
    #[arg(long, default_value = "siglip", value_parser = parse_selection)]
    pub select: super::import::Selection,
    /// Do not share the frames out equally by scene (day / night / IR).
    #[arg(long)]
    pub no_balance: bool,
    /// The footage's licence (default for files: "own footage").
    #[arg(long, alias = "licence")]
    pub license: Option<String>,
    /// Confirm that you may use the footage for training under that licence.
    #[arg(long, alias = "confirm-licence")]
    pub confirm_license: bool,
    /// Most frames extracted from one video before the selection.
    #[arg(long, default_value_t = super::import::DEFAULT_MAX_CANDIDATES)]
    pub max_candidates: usize,
    /// Allow downloads over 500 MB (the training environment, SigLIP 2).
    #[arg(long)]
    pub allow_large: bool,
    /// Keep the extracted candidates (<training dir>/imports/<id>).
    #[arg(long)]
    pub keep_work: bool,
    /// Print the helper's output.
    #[arg(long)]
    pub verbose: bool,
}

/// `train autolabel`.
#[derive(Debug, Clone, clap::Args)]
pub struct AutolabelArgs {
    /// Which frames: key=value pairs, comma-separated: bucket, scene, camera, label, status
    /// (unreviewed | reviewed | skipped | any), source (queue | all), import (<id> | any), limit.
    #[arg(long, default_value = "")]
    pub frames: String,
    /// MM-Grounding-DINO size: tiny, base, large, or off.
    #[arg(long, default_value = "tiny")]
    pub gdino: String,
    /// Do not run RF-DETR L.
    #[arg(long)]
    pub no_rfdetr: bool,
    /// Qwen3-VL crop check: 2b, 4b, 8b, or off.
    #[arg(long, default_value = "2b")]
    pub vlm: String,
    /// Do not rank species with SigLIP 2.
    #[arg(long)]
    pub no_siglip: bool,
    /// Self-training: re-label with this configured (trained) model instead of the labelers;
    /// frames whose reviewed labels it disagrees with go back to review first.
    #[arg(long)]
    pub with_model: Option<String>,
    /// Score threshold of --with-model.
    #[arg(long, default_value_t = 0.3)]
    pub threshold: f32,
    /// The class list (default collect.classes), comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub classes: Vec<String>,
    /// Allow downloads over 500 MB (the environment and the labelers).
    #[arg(long)]
    pub allow_large: bool,
    /// Print the labeler's output.
    #[arg(long)]
    pub verbose: bool,
}

fn parse_selection(s: &str) -> Result<super::import::Selection, String> {
    super::import::Selection::parse(s).ok_or_else(|| format!("'{s}' is not siglip or dhash"))
}

/// The import request of `a` (None for `--list-datasets`).
pub fn import_request(a: &ImportArgs) -> Result<Option<super::import::ImportRequest>> {
    use super::import::{ImportRequest, ImportSource};
    let source = match (&a.path, &a.url, &a.dataset) {
        (Some(p), None, None) => ImportSource::Path(p.clone()),
        (None, Some(u), None) => ImportSource::Url(u.clone()),
        (None, None, Some(d)) => ImportSource::Dataset(d.clone()),
        (None, None, None) => return Ok(None),
        _ => bail!("give one of --path, --url or --dataset"),
    };
    let mut r = ImportRequest::new(source);
    r.camera = a.camera.clone();
    r.frames = a.frames;
    r.scene_threshold = a.scene;
    r.max_fps = a.max_fps;
    r.every_s = a.every;
    r.dedup_hamming = a.dedup;
    r.dedup_window_s = a.dedup_window;
    r.selection = a.select;
    r.balance = !a.no_balance;
    r.license = a.license.clone();
    r.license_confirmed = a.confirm_license;
    r.max_candidates = a.max_candidates;
    r.allow_large = a.allow_large;
    r.keep_work = a.keep_work;
    r.validate()?;
    Ok(Some(r))
}

/// The auto-label request of `a`.
pub fn autolabel_request(a: &AutolabelArgs) -> Result<super::autolabel::AutolabelRequest> {
    use super::autolabel::{AutolabelRequest, FrameFilter, GdinoSize, VlmSize, default_status};
    let mut r = AutolabelRequest::new(FrameFilter::parse(
        &a.frames,
        default_status(a.with_model.is_some()),
    )?);
    r.gdino = match a.gdino.trim() {
        "off" | "none" | "" => None,
        g => Some(
            GdinoSize::parse(g)
                .with_context(|| format!("--gdino: '{g}' is not tiny, base, large or off"))?,
        ),
    };
    r.vlm = match a.vlm.trim() {
        "off" | "none" | "" => None,
        v => Some(
            VlmSize::parse(v).with_context(|| format!("--vlm: '{v}' is not 2b, 4b, 8b or off"))?,
        ),
    };
    r.rfdetr = !a.no_rfdetr;
    r.siglip = !a.no_siglip;
    r.with_model = a.with_model.clone();
    r.threshold = a.threshold;
    r.classes = (!a.classes.is_empty()).then(|| a.classes.clone());
    r.allow_large = a.allow_large;
    r.validate()?;
    Ok(r)
}

/// Run one import or auto-label task on a queue of this process, printing its progress; true
/// when it finished.
fn run_task_cli(config: &Config, task: Box<dyn super::task::Task>, verbose: bool) -> Result<bool> {
    let fetcher = Arc::new(crate::resources::export::cli_fetcher(&config.data_root()));
    let svc = TrainingService::start(
        Arc::new(crate::resources::export::SystemRunner),
        fetcher,
        Arc::new(super::job::NoHooks),
        Some(super::runs_file(config)),
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let _ctrl_c = crate::resources::export::CtrlC::cancels_named(cancel.clone(), "the job");
    let id = svc.enqueue_task(task).map_err(anyhow::Error::msg)?;
    let mut last = String::new();
    let mut last_step = usize::MAX;
    let mut last_print = std::time::Instant::now();
    let mut seen_log = 0usize;
    let mut ok = false;
    while let Some(r) = svc.wait_task(&id, Duration::from_secs(1)) {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            svc.cancel(&id);
        }
        if verbose {
            for l in r.log.iter().skip(seen_log.min(r.log.len())) {
                eprintln!("    {l}");
            }
            seen_log = r.log.len();
        }
        // One line per stage, else at most every 10 s.
        let line = format!("{:>3}% {}", r.percent, r.describe());
        if r.state == RunState::Running
            && line != last
            && (r.step != last_step || last_print.elapsed() >= Duration::from_secs(10))
        {
            eprintln!("{line}");
            last = line;
            last_step = r.step;
            last_print = std::time::Instant::now();
        }
        if !r.state.is_busy() {
            println!("{}: {}", r.title, r.describe());
            if let Some(res) = &r.result {
                if let Some(w) = res["warnings"].as_array() {
                    for x in w {
                        println!("  warning: {}", x.as_str().unwrap_or_default());
                    }
                }
                if let Some(s) = res["sample"].as_array().filter(|s| !s.is_empty()) {
                    println!("  frames, for example:");
                    for x in s.iter().take(5) {
                        println!("    {}", x.as_str().unwrap_or_default());
                    }
                }
            }
            ok = r.state == RunState::Done;
            break;
        }
    }
    svc.shutdown();
    Ok(ok)
}

/// What a run trains on.
#[derive(Debug, Clone, clap::Args)]
pub struct DataArgs {
    /// Review export folder (YOLO layout), repeatable. Default: `training.datasets`.
    #[arg(long = "dataset")]
    pub datasets: Vec<PathBuf>,
    /// all, day or night+ir.
    #[arg(long, default_value = "all", value_parser = parse_scene)]
    pub scene: SceneFilter,
    /// The model's classes in head order, comma-separated (default: every class of the datasets).
    #[arg(long, value_delimiter = ',')]
    pub classes: Vec<String>,
    /// Rename a label, `from=to` (empty `to` drops it), repeatable.
    #[arg(long, value_parser = parse_rename)]
    pub rename: Vec<(String, String)>,
    /// Mix in the pinned public night subset (train-data:lila-night).
    #[arg(long)]
    pub public: bool,
}

fn parse_scene(s: &str) -> Result<SceneFilter, String> {
    SceneFilter::parse(s).ok_or_else(|| format!("'{s}' is not all, day or night+ir"))
}

fn parse_rename(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))
        .filter(|(a, _)| !a.is_empty())
        .ok_or_else(|| format!("'{s}' is not from=to"))
}

/// Run a `train` subcommand.
pub fn main(args: TrainArgs, config_path: Option<PathBuf>) -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_writer(std::io::stderr)
        .try_init();
    match run(args, config_path) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn load_config(path: &Path) -> Result<Config> {
    if path.exists() {
        Config::load(path).context("fix the config file, or pass --config <another file>")
    } else {
        Ok(Config::default())
    }
}

fn run(args: TrainArgs, config_path: Option<PathBuf>) -> Result<bool> {
    let path = config_path.unwrap_or_else(Config::default_config_path);
    let config = load_config(&path)?;
    match args.command {
        TrainCommand::Capability { json } => {
            let cap = super::capability_now();
            if json {
                println!("{}", serde_json::to_string_pretty(&cap)?);
            } else {
                print!("{}", capability_text(&cap));
            }
            Ok(cap.available)
        }
        TrainCommand::Estimate {
            model,
            images,
            epochs,
            batch,
            imgsz,
        } => {
            let (family, size) = recipe::parse_label(&model)
                .with_context(|| format!("unknown model label '{model}'"))?;
            let cap = super::capability_now();
            let target = cap.accelerator.as_ref().map_or(
                estimate::Target::Cuda(crate::backend::detect::ComputeCapability::new(6, 1)),
                |a| a.target(),
            );
            let imgsz = imgsz.unwrap_or_else(|| family.default_imgsz(size));
            let batch = batch.unwrap_or(8);
            let epochs = epochs.unwrap_or(config.training.epochs);
            match estimate::estimate(family, size, imgsz, batch, images, epochs, target) {
                Some(e) => println!(
                    "{} {size} at {imgsz} px, batch {batch}, {images} images x {epochs} epochs: about {} MB of GPU memory, {} per epoch, {} in total ({}; rough estimate)",
                    family.title(),
                    e.vram_mb,
                    estimate::format_duration(e.epoch_s),
                    estimate::format_duration(e.total_s),
                    cap.accelerator
                        .as_ref()
                        .map_or_else(|| "a GTX 1080 class GPU".to_string(), |a| a.describe())
                ),
                None => println!("{} {size} does not train on this device", family.title()),
            }
            Ok(true)
        }
        TrainCommand::Prepare {
            data,
            out,
            allow_large,
        } => {
            if data.public {
                fetch_public(&config, allow_large || config.allow_large_downloads)?;
            }
            let (sources, opts) = read_sources(&config, &data)?;
            let plan = dataset::plan(&sources, &opts)?;
            let prepared = dataset::write(&sources, &plan, &opts, &out)?;
            println!(
                "{}: {} training, {} validation, {} held-out images; classes: {}",
                out.display(),
                prepared.train_images,
                prepared.valid_images,
                prepared.heldout_images,
                prepared.classes.join(", ")
            );
            if let Some(h) = &prepared.heldout_dir {
                println!("held-out benchmark set: dir:{}", h.display());
            }
            for w in &prepared.warnings {
                println!("warning: {w}");
            }
            Ok(true)
        }
        TrainCommand::Run {
            models,
            data,
            epochs,
            batch,
            imgsz,
            lr,
            night_aug,
            aug_dark,
            aug_noise,
            aug_gray,
            name,
            allow_large,
            no_register,
            no_benchmark,
            benchmark_sets,
            keep_work,
            verbose,
        } => {
            if name.is_some() && models.len() > 1 {
                bail!("--name needs exactly one --model");
            }
            let cap = super::capability_now();
            eprintln!("{}", cap.summary());
            let Some(accel) = cap.accelerator.clone() else {
                for r in &cap.reasons {
                    eprintln!("  {r}");
                }
                return Ok(false);
            };
            let paths = TrainPaths::for_config(&config);
            let platform = catalog::Platform::current();
            if !env::train_env_ready(&paths, platform, accel.lane())
                && !(allow_large || config.allow_large_downloads)
            {
                let (dl, disk) = accel.lane().estimate();
                bail!(
                    "the training environment (PyTorch {} and the trainers) is a large download \
                     (about {} GB, {} GB on disk under {}); rerun with --allow-large",
                    accel.lane().as_str(),
                    dl / 1_000_000_000,
                    disk / 1_000_000_000,
                    paths.export.tools.display()
                );
            }
            let mut base = Augment::default();
            if night_aug {
                base = Augment::NIGHT;
            }
            let augment = Augment {
                dark: aug_dark.unwrap_or(base.dark),
                noise: aug_noise.unwrap_or(base.noise),
                gray: aug_gray.unwrap_or(base.gray),
            };
            let mut jobs = Vec::new();
            for m in &models {
                let (family, size) = recipe::parse_label(m)
                    .with_context(|| format!("unknown model label '{m}' (e.g. yolo26n, rfdetr-nano, rt-detrv2-s, dfine-s)"))?;
                let mut r =
                    TrainRecipe::new(family, size, epochs.unwrap_or(config.training.epochs))?;
                if let Some(i) = imgsz {
                    r.imgsz = i;
                }
                r.batch = batch.unwrap_or_else(|| {
                    capability::Capability::family(&cap, family)
                        .and_then(|f| f.sizes.iter().find(|s| s.size == size))
                        .and_then(|s| s.suggested_batch)
                        .unwrap_or(r.batch)
                        .min(16)
                });
                r.lr = lr;
                r.augment = augment;
                let req = TrainRequest {
                    recipe: r,
                    datasets: data.datasets.clone(),
                    include_public: data.public,
                    scene: data.scene,
                    classes: (!data.classes.is_empty()).then(|| data.classes.clone()),
                    rename: data.rename.iter().cloned().collect::<BTreeMap<_, _>>(),
                    benchmark_sets: benchmark_sets.clone(),
                    allow_large,
                };
                let mut job = super::plan_job(&config, req, &cap, name.clone())?;
                job.keep_work = keep_work;
                jobs.push(job);
            }
            let hooks: Arc<dyn Hooks> = Arc::new(CliHooks {
                config_path: path.clone(),
                register: !no_register,
                benchmark: !no_register && !no_benchmark,
            });
            let fetcher = Arc::new(crate::resources::export::cli_fetcher(&config.data_root()));
            let svc = TrainingService::start(
                Arc::new(crate::resources::export::SystemRunner),
                fetcher,
                hooks,
                Some(super::runs_file(&config)),
            );
            let cancel = Arc::new(AtomicBool::new(false));
            let _ctrl_c =
                crate::resources::export::CtrlC::cancels_named(cancel.clone(), "the training");
            let ids: Vec<String> = jobs
                .into_iter()
                .map(|j| svc.enqueue(j).map_err(anyhow::Error::msg))
                .collect::<Result<_>>()?;
            let mut ok = true;
            for id in &ids {
                let mut last = (None, 0u8, String::new());
                let mut last_print = std::time::Instant::now();
                let mut seen_log = 0usize;
                while let Some(r) = svc.wait(id, Duration::from_secs(1)) {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        for i in &ids {
                            svc.cancel(i);
                        }
                    }
                    if verbose {
                        for l in r.log.iter().skip(seen_log.min(r.log.len())) {
                            eprintln!("    {l}");
                        }
                        seen_log = r.log.len();
                    }
                    // One line per stage or percent change, else at most every 10 s.
                    let line = format!("[{}] {:>3}% {}", r.name, r.percent, r.describe());
                    let key = (r.stage, r.percent, line.clone());
                    let changed = key.0 != last.0 || key.1 != last.1;
                    if r.state == RunState::Running
                        && key.2 != last.2
                        && (changed || last_print.elapsed() >= Duration::from_secs(10))
                    {
                        eprintln!("{line}");
                        last = key;
                        last_print = std::time::Instant::now();
                    }
                    if !r.state.is_busy() {
                        report_run(&r);
                        ok &= r.state == RunState::Done;
                        break;
                    }
                }
            }
            svc.shutdown();
            Ok(ok)
        }
        TrainCommand::Import(a) => {
            let Some(req) = import_request(&a)? else {
                println!("Pinned footage (--dataset):");
                for r in crate::resources::catalog::IMPORT_DATASETS {
                    println!(
                        "  {:<14} {} ({}, {}): {}",
                        crate::resources::catalog::install_key(r),
                        r.title,
                        r.license,
                        crate::resources::catalog::format_size(r.size()),
                        r.description
                    );
                }
                let dir = crate::collect::CollectSettings::for_config(&config).dir;
                let imports = super::import::write::read_imports(&dir);
                println!("Imports in {}: {}", dir.display(), imports.len());
                for i in imports.iter().rev().take(20) {
                    println!(
                        "  {}  {} frames  {}  ({})",
                        i.id, i.frames, i.source, i.license
                    );
                }
                return Ok(true);
            };
            let cap = super::capability_now();
            let job = super::import::plan_import(&config, req, &cap)?;
            eprintln!(
                "importing {} into {} (licence: {})",
                job.request.source.describe(),
                job.dataset_dir.display(),
                if job.license.is_empty() {
                    "the dataset's"
                } else {
                    &job.license
                }
            );
            run_task_cli(&config, Box::new(job), a.verbose)
        }
        TrainCommand::Autolabel(a) => {
            let req = autolabel_request(&a)?;
            let cap = super::capability_now();
            let job = super::autolabel::plan_autolabel(&config, req, &cap)?;
            eprintln!(
                "{} in {}",
                super::task::Task::title(&job),
                job.dataset_dir.display()
            );
            run_task_cli(&config, Box::new(job), a.verbose)
        }
        TrainCommand::List => {
            let runs = service::load_records(&super::runs_file(&config));
            if runs.is_empty() {
                println!(
                    "no training runs in {}",
                    super::training_dir(&config).display()
                );
            }
            for r in runs {
                println!(
                    "{}  {}  {}  {}",
                    r.id,
                    r.name,
                    r.request.recipe.label(),
                    r.describe()
                );
            }
            Ok(true)
        }
        TrainCommand::RemoveEnv => {
            let paths = TrainPaths::for_config(&config);
            env::remove_train_env(&paths)?;
            println!(
                "removed the training environment under {}",
                paths.export.tools.display()
            );
            Ok(true)
        }
    }
}

fn report_run(r: &service::RunRecord) {
    println!("{}: {}", r.name, r.describe());
    if let Some(d) = &r.dataset {
        println!(
            "  dataset: {} training, {} validation, {} held-out images; classes {}",
            d.train_images,
            d.valid_images,
            d.heldout_images,
            d.classes.join(", ")
        );
        for w in &d.warnings {
            println!("  warning: {w}");
        }
    }
    if let Some(o) = &r.output {
        println!(
            "  model: {} (family {}, licence {}), provenance {}",
            o.onnx.display(),
            o.family,
            o.license,
            o.provenance.display()
        );
    }
    if let Some(m) = &r.registered {
        println!("  {m}");
    }
    if let Some(b) = &r.benchmark {
        println!(
            "  benchmark on {}: {} images, {} on {}, AP50 {}",
            b.datasets.join(", "),
            b.images,
            b.ms_p50
                .map_or_else(|| "?".into(), |m| format!("{m:.1} ms")),
            b.device.as_deref().unwrap_or("?"),
            b.ap50.map_or_else(|| "n/a".into(), |a| format!("{a:.3}"))
        );
    }
    if let Some(e) = &r.benchmark_error {
        println!("  benchmark failed: {e}");
    }
}

/// The capability as text (`train capability`).
pub fn capability_text(cap: &Capability) -> String {
    let mut s = format!("{}\n", cap.summary());
    for r in &cap.reasons {
        s.push_str(&format!("  {r}\n"));
    }
    for f in &cap.families {
        let state = if f.available {
            String::new()
        } else {
            format!(" (not here: {})", f.why.as_deref().unwrap_or("unavailable"))
        };
        s.push_str(&format!("{} [{}]{state}\n", f.title, f.license));
        if !f.available {
            continue;
        }
        for z in &f.sizes {
            let est = match (z.vram_mb, z.epoch_s_per_100) {
                (Some(v), Some(t)) => format!(
                    "~{} MB at batch {}, ~{} per epoch per 100 images",
                    v,
                    z.suggested_batch.unwrap_or(1),
                    estimate::format_duration(t)
                ),
                _ => "no estimate".into(),
            };
            let fit = if z.fits {
                String::new()
            } else {
                format!(" -- does not fit: {}", z.why.as_deref().unwrap_or("?"))
            };
            s.push_str(&format!("  {:<7} {est}; {}{fit}\n", z.size, z.latency));
        }
    }
    s
}

/// `train prepare --public`: download the public night subset when it is missing, behind the
/// large-download gate ([`super::check_public_download`]).
fn fetch_public(config: &Config, allowed: bool) -> Result<()> {
    use crate::resources::export::Fetcher;
    let root = config.data_root();
    super::check_public_download(&root, allowed)?;
    if super::public_subset_installed(&root) {
        return Ok(());
    }
    let r = catalog::lila_night().context("the public night subset is not pinned in this build")?;
    eprintln!(
        "downloading {} ({})",
        r.title,
        catalog::format_size(r.size())
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let _ctrl_c = crate::resources::export::CtrlC::cancels_named(cancel.clone(), "the download");
    crate::resources::export::cli_fetcher(&root).fetch(
        r,
        &super::public_subset_dir(&root),
        &mut |_, _| {},
        &cancel,
    )
}

fn read_sources(
    config: &Config,
    data: &DataArgs,
) -> Result<(Vec<dataset::Source>, dataset::Options)> {
    let dirs: Vec<PathBuf> = if data.datasets.is_empty() {
        config.training.datasets.clone()
    } else {
        data.datasets.clone()
    };
    if dirs.is_empty() {
        bail!("no dataset: pass --dataset <review export folder> or set training.datasets");
    }
    let mut sources = Vec::new();
    for d in dirs {
        sources.push(dataset::read_export(&config.data_path(&d))?);
    }
    if data.public {
        let m =
            catalog::lila_night_manifest().context("the public subset manifest does not parse")?;
        if catalog::lila_night().is_none() {
            bail!("the public night subset is not pinned in this build");
        }
        sources.push(dataset::read_public_set(
            &super::public_subset_dir(&config.data_root()),
            m,
        )?);
    }
    let t = &config.training;
    let opts = dataset::Options {
        classes: (!data.classes.is_empty()).then(|| data.classes.clone()),
        rename: data.rename.iter().cloned().collect(),
        scene: data.scene,
        negative_share: t.negative_share,
        heldout_share: t.heldout_share,
        oversample: t.oversample,
        ..dataset::Options::default()
    };
    Ok((sources, opts))
}

/// The CLI's [`Hooks`]: the config file is edited like `add-model` does; the benchmark runs in
/// this process with its own runtimes.
struct CliHooks {
    config_path: PathBuf,
    register: bool,
    benchmark: bool,
}

impl Hooks for CliHooks {
    fn register(&self, out: &TrainOutput) -> Result<String> {
        if !self.register {
            return Ok(format!(
                "{} installed (--no-register: not added to the config)",
                out.name
            ));
        }
        let mut cfg = load_config(&self.config_path)?;
        let msg = add_trained_model(&mut cfg, out)?;
        cfg.save(&self.config_path)?;
        Ok(format!("{msg} ({})", self.config_path.display()))
    }

    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        cancel: &AtomicBool,
    ) -> Result<Option<BenchSummary>> {
        if !self.benchmark {
            return Ok(None);
        }
        let svc = crate::benchmark::service::BenchmarkService::new();
        let path = self.config_path.clone();
        let ctx = move || crate::benchmark::service::RunContext {
            config: load_config(&path).unwrap_or_default(),
            results_path: crate::benchmark::results_path(&path),
            runtimes: None,
            loaded: Default::default(),
            resources: None,
        };
        super::bench::run(
            &svc,
            &ctx,
            &out.name,
            super::bench::datasets(&out.name, out.heldout.as_deref(), extra),
            cancel,
        )
    }
}

/// Add the trained model to `cfg` the way `add-model` does (path, name, family, classes).
pub fn add_trained_model(cfg: &mut Config, out: &TrainOutput) -> Result<String> {
    let req = crate::resources::actions::CustomModel {
        path: out.onnx.to_string_lossy().into_owned(),
        name: Some(out.name.clone()),
        family: out.family,
        classes: Some(out.yaml.to_string_lossy().into_owned()),
    };
    crate::resources::actions::add_custom_model(cfg, &req)
        .map_err(|e| anyhow::anyhow!("{}", e.message()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{HardwareInfo, apple_gpu};

    #[test]
    fn capability_text_lists_families_and_fits() {
        let cap = capability::capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(8.0),
        );
        let t = capability_text(&cap);
        assert!(t.contains("YOLO26 [AGPL-3.0]"), "{t}");
        assert!(t.contains("RF-DETR [Apache-2.0]"), "{t}");
        assert!(t.contains("D-FINE [Apache-2.0] (not here:"), "{t}");
        let none = capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
        assert!(capability_text(&none).contains("train_model.py"));
    }

    #[derive(clap::Parser)]
    struct T {
        #[command(subcommand)]
        c: TrainCommand,
    }

    fn parse(args: &[&str]) -> Result<TrainCommand, clap::Error> {
        use clap::Parser;
        T::try_parse_from(std::iter::once("train").chain(args.iter().copied())).map(|t| t.c)
    }

    #[test]
    fn import_and_autolabel_flags_parse() {
        let TrainCommand::Import(a) = parse(&[
            "import",
            "--path",
            "clips",
            "--frames",
            "50",
            "--select",
            "dhash",
            "--confirm-licence",
            "--licence",
            "CC BY 4.0",
        ])
        .unwrap() else {
            panic!("not import")
        };
        let r = import_request(&a).unwrap().unwrap();
        assert_eq!(r.frames, 50);
        assert_eq!(r.selection, super::super::import::Selection::Dhash);
        assert!(r.license_confirmed);
        assert_eq!(r.license.as_deref(), Some("CC BY 4.0"));
        assert!(parse(&["import"]).is_err(), "a source is required");
        assert!(
            parse(&["import", "--path", "a", "--url", "https://x/y"]).is_err(),
            "one source only"
        );
        let TrainCommand::Import(l) = parse(&["import", "--list-datasets"]).unwrap() else {
            panic!()
        };
        assert!(import_request(&l).unwrap().is_none());
        assert!(parse(&["import", "--path", "a", "--select", "x"]).is_err());

        let TrainCommand::Autolabel(a) = parse(&[
            "autolabel",
            "--frames",
            "scene=ir,limit=20",
            "--gdino",
            "base",
            "--vlm",
            "off",
            "--no-siglip",
        ])
        .unwrap() else {
            panic!()
        };
        let r = autolabel_request(&a).unwrap();
        assert_eq!(r.frames.limit, 20);
        assert_eq!(r.frames.query["status"], "unreviewed");
        assert_eq!(r.gdino, Some(super::super::autolabel::GdinoSize::Base));
        assert_eq!(r.vlm, None);
        assert!(!r.siglip && r.rfdetr);
        let TrainCommand::Autolabel(m) =
            parse(&["autolabel", "--with-model", "night-yolo26n-1"]).unwrap()
        else {
            panic!()
        };
        let r = autolabel_request(&m).unwrap();
        assert_eq!(r.with_model.as_deref(), Some("night-yolo26n-1"));
        assert_eq!(
            r.frames.query["status"], "any",
            "re-labels reviewed frames too"
        );
        let TrainCommand::Autolabel(bad) = parse(&["autolabel", "--gdino", "huge"]).unwrap() else {
            panic!()
        };
        assert!(autolabel_request(&bad).is_err());
    }

    #[test]
    fn flags_parse() {
        assert_eq!(
            parse_rename("motorbike=motorcycle").unwrap(),
            ("motorbike".into(), "motorcycle".into())
        );
        assert_eq!(
            parse_rename("tree=").unwrap(),
            ("tree".into(), String::new())
        );
        assert!(parse_rename("=x").is_err());
        assert_eq!(parse_scene("night+ir").unwrap(), SceneFilter::NightIr);
        assert!(parse_scene("dusk").is_err());
    }
}
