//! Command line front end of the benchmark: the `blue-onyx-prism-benchmark` binary and the
//! `blue-onyx-prism benchmark` subcommand both parse [`BenchArgs`] and call [`main`].
//!
//! ```text
//! blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-prism-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-prism-benchmark --json                                         # every enabled model in the config
//! blue-onyx-prism-benchmark --all-devices --apply                          # sweep, save, set per-model devices
//! blue-onyx-prism-benchmark --apply-threshold --threshold-objective f2      # also set per-model thresholds
//! blue-onyx-prism-benchmark --threshold-search --tag night --threshold-objective recall:0.9
//!                                                       # search the stored predictions, no inference
//! blue-onyx-prism-benchmark --all-devices --accuracy-metric roc_auc --threshold-objective youden
//!                                                       # grade by frame ROC AUC, pick by Youden's J
//! blue-onyx-prism-benchmark --threshold-search --threshold-objective recall_at_fa:5
//!                                                       # most found, <= 5 false alerts / 100 empty frames
//! ```
//!
//! Ctrl-C (SIGINT; SIGTERM on Unix; Ctrl-Break on Windows) cancels a run cleanly ([`Interrupts`]):
//! the device in hand stops at its next timed iteration (a compile in progress ends first), the
//! models a sweep finished are saved as the service saves a cancelled run's, and the run's
//! breadcrumb goes, so the next start reports no interrupted run. A second Ctrl-C quits at once
//! (exit 130), leaving the breadcrumb: a run that does not stop may be a device that hung.

use crate::backend::{CoreOptions, OrtOptions, Runtimes, libs, priority, spec};
use crate::benchmark::grade::AccuracyMetric;
use crate::benchmark::images::{self, ImageSet};
use crate::benchmark::report::{HardwareSummary, RuntimeVersions};
use crate::benchmark::service::breadcrumb::{self, Trail};
use crate::benchmark::threshold::Objective;
use crate::benchmark::{
    Bench, BenchmarkResults, Job, ModelResult, Phase, SweepOptions, config_models,
    configured_device, export, job_as_served, job_for_config, job_for_file, no_pseudo_gt_device,
    pick_reference, pseudo_ground_truth, pseudo_gt_device, results_path,
};
use crate::config::{Config, DatasetRef, ModelConfig};
use crate::config_form::{
    DeviceChange, ProfileChange, apply_device_thresholds, apply_model_devices,
};
use crate::model::PostParams;
use crate::sync::LockExt;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

mod args;
mod print;

pub use args::BenchArgs;
pub(crate) use args::SINGLE_IMAGE_REPEAT;
use print::*;
pub(crate) use print::{BLUE_IRIS_NOTE, agreement_text};

/// Parse-independent entry point: set up logging, run, map the outcome to an exit code.
/// `config_fallback` is the main binary's `--config` (used when the subcommand has none).
pub fn main(mut args: BenchArgs, config_fallback: Option<PathBuf>) -> ExitCode {
    if args.config.is_none() {
        args.config = config_fallback;
    }
    let level = if args.verbose { "info" } else { "warn" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(level))
        .with_writer(std::io::stderr)
        .try_init();
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn load_config(path: &Path) -> Result<Config> {
    Config::load_or_default(path)
}

/// One job, with its config entry when it came from the config file.
struct Planned {
    job: Job,
    entry: Option<ModelConfig>,
}

/// A job's own sets (Your cameras without what its model saw; None = the run's) and what was left
/// out.
type OwnSets = (
    Option<Vec<ImageSet>>,
    Option<crate::benchmark::report::LeftOut>,
);

fn jobs(args: &BenchArgs, config: &Config, errors: &mut Vec<String>) -> Result<Vec<Planned>> {
    let served = served_mode(args, config)?;
    let device_override = if args.force_cpu {
        Some("CPU".to_string())
    } else {
        args.device.clone()
    };
    let mut out = Vec::new();
    if !args.model.is_empty() {
        for p in &args.model {
            out.push(Planned {
                job: job_for_file(
                    p,
                    args.family,
                    args.classes.as_deref(),
                    device_override.as_deref(),
                )?,
                entry: None,
            });
        }
    } else {
        // Each model compiled and thresholded as its worker would (B4, B5).
        for m in config_models(config)? {
            match job_as_served(config, m, device_override.as_deref(), None) {
                Ok(job) => out.push(Planned {
                    job,
                    entry: Some(m.clone()),
                }),
                Err(e) => errors.push(format!("{}: {e:#}", m.effective_name())),
            }
        }
    }
    for p in &mut out {
        if let Some(f) = args.family {
            p.job.family = f;
        }
        if let Some(c) = args.min_confidence {
            // Asked for on the command line: this threshold on every device.
            p.job.confidence_threshold = Some(c);
            p.job.device_thresholds.clear();
        }
    }
    if served {
        return Ok(as_served(out, args, config));
    }
    Ok(with_variants(out, args, config, errors))
}

/// The burst test of this run: `--burst C,K,I`, `--burst` alone with `benchmark.burst`'s load,
/// else `benchmark.burst` when it is enabled; None = no burst test.
pub(crate) fn burst_of(args: &BenchArgs, config: &Config) -> Option<crate::config::BurstConfig> {
    match args.burst {
        Some(Some(b)) => Some(b),
        Some(None) => Some(crate::config::BurstConfig {
            enabled: true,
            ..config.benchmark.burst
        }),
        None => config
            .benchmark
            .burst
            .enabled
            .then_some(config.benchmark.burst),
    }
}

/// What `--served` and `--variants` refused together say.
pub(crate) const SERVED_WITH_VARIANTS: &str = "--served and --variants cannot be combined: \
     --served runs each model with its own serving settings (results `<model>+served`), \
     --variants tries other settings on the plain model; run them separately";

/// Whether this run benchmarks the models as served: `--served`, else `benchmark.served` unless
/// `--variants` asks for other settings (said on stderr). `--served` with `--variants` is
/// refused.
pub(crate) fn served_mode(args: &BenchArgs, config: &Config) -> Result<bool> {
    match (args.served, args.variants.is_some()) {
        (true, true) => bail!("{SERVED_WITH_VARIANTS}"),
        (true, false) => Ok(true),
        (false, true) => {
            if config.benchmark.served && !args.json {
                eprintln!(
                    "benchmark.served is on, but --variants runs instead (the two do not combine)"
                );
            }
            Ok(false)
        }
        (false, false) => Ok(config.benchmark.served),
    }
}

/// Every job as served ([`crate::benchmark::PipelineVariant::served`], named
/// `<model>+served`); a `--model` file gets the global settings.
fn as_served(jobs: Vec<Planned>, args: &BenchArgs, config: &Config) -> Vec<Planned> {
    jobs.into_iter()
        .map(|mut p| {
            let file_entry = ModelConfig {
                path: p.job.path.clone(),
                ..ModelConfig::default()
            };
            let entry = p.entry.as_ref().unwrap_or(&file_entry);
            let v = crate::benchmark::PipelineVariant::served(config, entry, None);
            p.job.name = crate::benchmark::variant::served_name(&p.job.name);
            if !args.json {
                eprintln!("as served {}: {}", p.job.name, v.describe());
            }
            p.job.variant = Some(v);
            p
        })
        .collect()
}

/// Every job once per `--variants` entry (in that order per model; the plain pipeline keeps
/// the model's name, a variant is `<model>+<variant>`). A variant that cannot be resolved (an
/// unknown `verify:<model>`) is skipped with an error.
fn with_variants(
    jobs: Vec<Planned>,
    args: &BenchArgs,
    config: &Config,
    errors: &mut Vec<String>,
) -> Vec<Planned> {
    let Some(list) = &args.variants else {
        return jobs;
    };
    let mut out = Vec::with_capacity(jobs.len() * list.0.len());
    for p in jobs {
        for spec in &list.0 {
            let mut job = p.job.clone();
            job.variant = match spec.resolve(config, p.entry.as_ref(), &p.job.name) {
                Ok(v) => v,
                Err(e) => {
                    errors.push(format!("{}+{}: {e:#}", p.job.name, spec.name));
                    continue;
                }
            };
            if let Some(v) = &job.variant {
                job.name = crate::benchmark::variant::variant_name(&job.name, v);
                if !args.json {
                    eprintln!("variant {}: {}", job.name, v.describe());
                }
            }
            out.push(Planned {
                job,
                entry: p.entry.clone(),
            });
        }
    }
    out
}

/// The datasets of a run: `--image`, else `--dataset`/`--images`, else the config's.
fn dataset_refs(args: &BenchArgs, config: &Config) -> Vec<DatasetRef> {
    let mut refs: Vec<DatasetRef> = args.dataset.iter().cloned().map(DatasetRef::Id).collect();
    if let Some(dir) = &args.images {
        refs.push(DatasetRef::Dir {
            dir: dir.clone(),
            gt: args.gt.clone(),
            name: None,
        });
    }
    if refs.is_empty() {
        refs = config.benchmark.datasets.clone();
    }
    refs
}

/// `--rescore`: the stored predictions of Your cameras scored against the labels now.
fn rescore_cli(args: &BenchArgs, config: &Config, config_path: &Path) -> Result<bool> {
    let r = crate::benchmark::cameras::rescore::rescore(
        config,
        &results_path(config_path),
        &crate::benchmark::search::preds_path(config_path),
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&r)?);
    } else {
        println!(
            "{}: re-scored {} run(s) of {} model(s) on {} frames; {} frame(s) had changed labels, {} left the reviewed set, {} reviewed since need a run",
            crate::benchmark::cameras::TITLE,
            r.runs,
            r.models,
            r.frames,
            r.changes.changed,
            r.changes.left,
            r.need_run
        );
        for n in &r.not_rescored {
            println!("  not re-scored (no stored predictions): {n}");
        }
        if !r.unknown.is_empty() {
            println!(
                "  {} frame(s) could not be read this time and kept their labels: {}",
                r.unknown.len(),
                r.unknown.join(", ")
            );
        }
    }
    Ok(true)
}

/// The datasets of a run and what it keeps of Your cameras for its snapshot.
type LoadedSets = (Vec<ImageSet>, Vec<String>, Option<images::CamerasLoaded>);

/// Resolve the datasets, downloading missing built-in sets first when the config allows it.
fn load_sets(args: &BenchArgs, config: &Config) -> Result<LoadedSets> {
    if let Some(p) = &args.image {
        return Ok((vec![ImageSet::single_file(p)?], Vec::new(), None));
    }
    let refs = dataset_refs(args, config);
    let max = args
        .max_images
        .unwrap_or(config.benchmark.max_images_per_dataset);
    let root = config.data_root();
    let cameras = crate::benchmark::cameras::Source::of(config);
    let resolve = || images::resolve_with(&refs, &root, max, Some(&cameras));
    let mut r = resolve()?;
    if !r.not_downloaded.is_empty() && config.benchmark.auto_download_datasets {
        let req = crate::resources::commands::FetchRequest {
            for_config: false,
            resources: r
                .not_downloaded
                .iter()
                .map(|id| format!("{}{id}", crate::resources::catalog::BENCH_ID_PREFIX))
                .collect(),
            all_for_platform: false,
            allow_large: false,
        };
        match crate::resources::commands::fetch(config, &req) {
            Ok(()) => r = resolve()?,
            Err(e) => r
                .warnings
                .push(format!("downloading datasets failed: {e:#}")),
        }
    }
    cap_run(
        &mut r.sets,
        &mut r.warnings,
        crate::benchmark::service::MAX_IMAGES_PER_RUN,
        args.no_image_cap,
    );
    Ok((r.sets, r.warnings, r.cameras))
}

/// Cut the run to the service's image cap (RA-078: the CLI holds every result in memory until the
/// run ends) and add its warning, unless `--no-image-cap` was given.
fn cap_run(sets: &mut [ImageSet], warnings: &mut Vec<String>, cap: usize, no_cap: bool) {
    if no_cap {
        return;
    }
    if let Some(w) = crate::benchmark::service::cap_images(sets, cap) {
        warnings.push(w);
    }
}

/// `--list-datasets`.
fn list_datasets(config: &Config) {
    let root = config.data_root();
    let selected: Vec<String> = config
        .benchmark
        .datasets
        .iter()
        .map(|d| d.label())
        .collect();
    println!(
        "  {:<16} {:>7} {:>9} {:<12} {:<10} TITLE / LICENSE",
        "ID", "IMAGES", "SIZE", "STATE", "SELECTED"
    );
    let mark = |id: &str| {
        if selected.iter().any(|s| s.eq_ignore_ascii_case(id)) {
            "yes"
        } else {
            ""
        }
    };
    {
        use crate::benchmark::cameras::{self, Load, Source, TITLE};
        let src = Source::of(config);
        let (count, state, about) = match cameras::load(&src) {
            Load::Ready(l) => (
                l.set.images.len().to_string(),
                "reviewed",
                format!("{TITLE}: {} ({})", l.survey.line(), src.dir.display()),
            ),
            Load::Empty(_) => (
                "0".to_string(),
                "no frames",
                format!("{TITLE}: no reviewed frame yet ({})", src.dir.display()),
            ),
            Load::Missing(why) => ("-".to_string(), "missing", format!("{TITLE}: {why}")),
        };
        println!(
            "  {:<16} {:>7} {:>9} {:<12} {:<10} {}",
            cameras::ID,
            count,
            "-",
            state,
            mark(cameras::ID),
            about
        );
    }
    let sample = ImageSet::sample();
    println!(
        "  {:<16} {:>7} {:>9} {:<12} {:<10} {}",
        images::SAMPLE_ID,
        1,
        "-",
        "embedded",
        mark(images::SAMPLE_ID),
        sample.title
    );
    for id in images::KNOWN_SET_IDS {
        if images::builtin_manifest(id).is_none() {
            println!(
                "  {:<16} {:>7} {:>9} {:<12} {:<10} (not in this build)",
                id,
                "-",
                "-",
                "-",
                mark(id)
            );
        }
    }
    for m in images::builtin_manifests() {
        let (set, missing) = ImageSet::from_manifest(m, &images::builtin_dir(&root, &m.id));
        let state = if missing == 0 {
            "installed".to_string()
        } else if set.images.is_empty() {
            "available".to_string()
        } else {
            format!("{}/{}", set.images.len(), m.images.len())
        };
        println!(
            "  {:<16} {:>7} {:>9} {:<12} {:<10} {} ({})",
            m.id,
            m.images.len(),
            crate::resources::catalog::format_size(m.size()),
            state,
            mark(&m.id),
            m.title,
            m.license
        );
    }
    for d in &config.benchmark.datasets {
        if let Some((dir, _, _)) = d.dir() {
            println!(
                "  {:<16} {:>7} {:>9} {:<12} {:<10} {}",
                d.label(),
                "-",
                "-",
                if config.data_path(&dir).is_dir() {
                    "folder"
                } else {
                    "missing"
                },
                "yes",
                config.data_path(&dir).display()
            );
        }
    }
    println!(
        "Download a set with `blue-onyx-prism fetch --resource bench:<id>`; select datasets with \
         --dataset <id|dir:path> or `benchmark.datasets` in the config."
    );
}

/// The goal of a run: the config's, with `--threshold-objective` / `--accuracy-metric` mapped
/// onto it ([`crate::benchmark::eval::Goal::with_legacy`]), then `--goal`, `--tune-for`,
/// `--goal-classes` and `--speed-budget`.
pub(crate) fn goal_of(args: &BenchArgs, config: &Config) -> Result<crate::benchmark::eval::Goal> {
    let g = config
        .benchmark
        .goal
        .with_legacy(args.threshold_objective, args.accuracy_metric);
    g.try_with(
        args.goal.as_deref(),
        args.tune_for.as_deref(),
        args.goal_classes.as_deref(),
        args.speed_budget.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Run the benchmark described by `args`. Ok(false) when some model failed.
pub fn run(args: BenchArgs) -> Result<bool> {
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(Config::default_config_path);
    let config = load_config(&config_path)?;
    if args.list_datasets {
        list_datasets(&config);
        return Ok(true);
    }
    if args.rescore {
        return rescore_cli(&args, &config, &config_path);
    }
    if args.threshold_search
        && let Some(ok) = stored_threshold_search(&args, &config, &config_path)?
    {
        return Ok(ok);
    }
    check_args(&args, &config_path)?;
    let low_priority = !args.no_low_priority && config.benchmark.low_priority;
    let goal = goal_of(&args, &config)?;
    let (objective, metric) = goal.legacy();
    let all_devices = args.all_devices
        || args.apply
        || args.apply_threshold
        || args.threshold_search
        || args.report.is_some();
    let openvino_dir = config.openvino_dir_effective();
    libs::prepare_environment(openvino_dir.as_deref());

    if args.repeat == Some(0) {
        bail!("--repeat must be at least 1");
    }
    let mut errors = Vec::new();
    let jobs = jobs(&args, &config, &mut errors)?;
    let variant_names: std::collections::HashSet<String> = jobs
        .iter()
        .filter(|p| p.job.variant.is_some())
        .map(|p| p.job.name.clone())
        .collect();
    let (mut sets, warnings, cameras) = load_sets(&args, &config)?;
    let (repeat, warmup, weights) = timing_settings(&args, &config, &sets);
    let cache_dir = cache_dir_arg(&args, &config);
    // Unix: before the runtimes exist, so their thread pools inherit the nice value, and only
    // for a run whose devices are all CPUs (it cannot be raised again); held to the end of the
    // run. Windows switches per device instead (`bench.priority`).
    let cpu_only = !all_devices
        && !jobs.is_empty()
        && jobs.iter().all(|p| priority::lowers_device(&p.job.device));
    let _prio = priority::lower_up_front(low_priority, cpu_only);
    let per_device = priority::DevicePriority::new(low_priority);
    let runtimes = open_runtimes(
        &args,
        &config,
        &config_path,
        cache_dir.clone(),
        openvino_dir,
    )?;
    // `benchmark.devices` of the config limits the sweep, as on the web UI's Benchmark page,
    // and the pseudo-ground-truth device.
    let sweep_devices = crate::benchmark::service::parse_devices(&config.benchmark.devices)
        .context("benchmark.devices in the config")?;
    if all_devices && !args.confirm_gpus {
        let gpus = jobs.iter().fold(Vec::new(), |mut acc, p| {
            let sel = runtimes.selection(Some(&p.job.path));
            for g in sweep_gpus(&sel, sweep_devices.as_deref()) {
                if !acc.contains(&g) {
                    acc.push(g);
                }
            }
            acc
        });
        if let Some(refusal) = gpu_gate(&gpus, args.confirm_gpus) {
            bail!("{refusal}");
        }
    }
    let info = runtimes.info();
    let versions = RuntimeVersions::of(&runtimes);
    // Shared: the burst test's temporary workers compile through the same runtimes.
    let runtimes = Arc::new(Mutex::new(runtimes));
    let mut report = Report::new(info, errors, &sets, warnings);
    // A sweep of this run takes the breadcrumb over: report the one a run that never ended left.
    let data_root = config.data_root();
    // The header prints the warnings so far.
    if all_devices && let Some(left) = breadcrumb::left_behind(&data_root) {
        report.warnings.push(left.message());
    }
    // Declared before the trail, so it is dropped after it: a cancelled run unwinds normally.
    let cancel = Arc::new(AtomicBool::new(false));
    let _interrupts = Interrupts::cancel(cancel.clone());
    let trail = Trail::new(&data_root);
    if !args.json {
        print_header(
            &report,
            &versions,
            cache_dir.as_deref(),
            &sets,
            warmup,
            repeat,
        );
    }

    let params = PostParams {
        confidence_threshold: args.min_confidence.unwrap_or(config.confidence_threshold),
        nms_iou: config.nms_iou,
    };
    let progress = cli_progress(args.json);

    let ctx = RunCtx {
        args: &args,
        config: &config,
        config_path: &config_path,
        runtimes: &runtimes,
        sweep_devices: &sweep_devices,
        objective,
        metric,
        goal: &goal,
        weights,
        variant_names: &variant_names,
        trail: &trail,
    };

    // Pseudo ground truth for datasets without it (accuracy relative to a reference model).
    if sets.iter().any(|s| !s.annotated()) {
        let mut pb = Bench::new(&[], 0, 1, params);
        pb.cache_dir = cache_dir.as_deref();
        pb.cancel = Some(&cancel);
        pb.progress = Some(&progress);
        pb.priority = priority::PER_DEVICE.then_some(&per_device);
        ctx.pseudo_ground_truth(&pb, &jobs, &mut sets, &mut report);
    }

    // Per job: Your cameras without the frames its model trained or validated on.
    let own_sets: Vec<OwnSets> = jobs
        .iter()
        .map(|p| match &p.entry {
            Some(m) => crate::benchmark::cameras::sets_for_model(&config, m, &sets),
            None => (None, None),
        })
        .collect();
    let mut bench = Bench::new(&sets, warmup, repeat, params);
    bench.object_filter = &config.object_filter;
    bench.cache_dir = cache_dir.as_deref();
    bench.cancel = Some(&cancel);
    bench.weights = weights;
    bench.threshold_objective = objective;
    bench.accuracy_metric = metric;
    bench.priority = priority::PER_DEVICE.then_some(&per_device);
    // Windows: the GPU's video-decode load during each timed phase (another process's decode,
    // such as Blue Iris', shares the iGPU).
    bench.video_decode = true;
    let burst = burst_of(&args, &config)
        .map(|b| crate::benchmark::burst::Burst::new(b, &runtimes, &config));
    if let Some(b) = burst.as_ref().filter(|_| !args.json) {
        eprintln!(
            "burst test after each device's timed runs: {}, {} bursts",
            b.settings.describe(),
            b.bursts
        );
    }
    bench.burst = burst.as_ref();

    let run_id = crate::benchmark::history::new_run_id();
    let mut swept = ctx.run_jobs(
        &mut bench,
        &progress,
        &jobs,
        (&sets, &own_sets),
        all_devices,
        &mut report,
    )?;
    for m in &mut swept {
        m.run_id = Some(run_id.clone());
    }

    if cancel.load(Ordering::SeqCst) {
        // The finished models are kept, as the service keeps a cancelled run's.
        let saved = if all_devices && !swept.is_empty() {
            let results =
                BenchmarkResults::new(HardwareSummary::current(), versions, swept.clone())
                    .with_sets(&sets);
            ctx.save_results(results, &swept, &sets, cameras.as_ref(), &mut report)?;
            (!args.no_save).then(|| results_path(&config_path))
        } else {
            None
        };
        bail!("{}", cancelled_message(swept.len(), saved.as_deref()));
    }

    if all_devices
        && !swept.is_empty()
        && let Some(ok) =
            ctx.finish_sweep(&sets, cameras.as_ref(), versions, &swept, &mut report)?
    {
        return Ok(ok);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report.errors.is_empty())
}

/// The checks of `args` that need no runtime: `--apply` / `--apply-threshold` need an existing
/// config file, `--accuracy-weight` is within 0..1.
fn check_args(args: &BenchArgs, config_path: &Path) -> Result<()> {
    if (args.apply || args.apply_threshold) && !config_path.exists() {
        bail!(
            "{} needs a config file; {} does not exist",
            if args.apply {
                "--apply"
            } else {
                "--apply-threshold"
            },
            config_path.display()
        );
    }
    if let Some(w) = args.accuracy_weight
        && !(0.0..=1.0).contains(&w)
    {
        bail!("--accuracy-weight must be between 0 and 1");
    }
    Ok(())
}

/// (timed runs per image, warm-up iterations, grade weights) of a run over `sets`: the flags,
/// else the config (a single image defaults to [`SINGLE_IMAGE_REPEAT`] runs).
fn timing_settings(
    args: &BenchArgs,
    config: &Config,
    sets: &[ImageSet],
) -> (usize, usize, crate::benchmark::grade::Weights) {
    let image_total: usize = sets.iter().map(|s| s.images.len()).sum();
    let repeat = args.repeat.unwrap_or(if image_total == 1 {
        SINGLE_IMAGE_REPEAT
    } else {
        config.benchmark.repeat_per_image.max(1)
    });
    let warmup = args.warmup.unwrap_or(config.benchmark.warmup);
    let weights = match args.accuracy_weight {
        Some(a) => crate::benchmark::grade::Weights {
            accuracy: a,
            speed: 1.0 - a,
        },
        None => config.benchmark.weights,
    };
    (repeat, warmup, weights)
}

/// The compiled-model cache directory: `--cache-dir` ("" disables), else the config's.
fn cache_dir_arg(args: &BenchArgs, config: &Config) -> Option<PathBuf> {
    match &args.cache_dir {
        Some(d) if d.trim().is_empty() => None,
        Some(d) => Some(PathBuf::from(d)),
        None => config.cache_dir_path(),
    }
}

/// `--threshold-search` over the predictions stored beside the config: Some(outcome) when
/// there are any, None (after a note) when the benchmark must run first.
fn stored_threshold_search(
    args: &BenchArgs,
    config: &Config,
    config_path: &Path,
) -> Result<Option<bool>> {
    let results = BenchmarkResults::load(&results_path(config_path))?;
    let preds = crate::benchmark::search::StoredPreds::load(
        &crate::benchmark::search::preds_path(config_path),
    )?;
    match (results, preds) {
        (Some(r), Some(p)) if !p.runs.is_empty() => {
            threshold_search_cli(args, config, &r, &p).map(Some)
        }
        _ => {
            if !args.json {
                eprintln!(
                    "--threshold-search: no stored predictions beside {}; running the benchmark first",
                    config_path.display()
                );
            }
            Ok(None)
        }
    }
}

/// The runtimes of a run, with `auto` models resolved as the server would run them (the
/// configured device of a sweep, ordered by the stored results).
fn open_runtimes(
    args: &BenchArgs,
    config: &Config,
    config_path: &Path,
    cache_dir: Option<PathBuf>,
    openvino_dir: Option<PathBuf>,
) -> Result<Runtimes> {
    let opts = CoreOptions {
        cache_dir,
        intra_threads: args.threads.unwrap_or(config.intra_threads),
        openvino_dir,
    };
    let ort_opts: OrtOptions = config.ort_options();
    let mut runtimes = Runtimes::new_with(&opts, &ort_opts);
    runtimes.require_any()?;
    // `auto` models resolve as the server would run them (the configured device of a sweep).
    let orders = crate::benchmark::auto_orders(
        BenchmarkResults::load_or_warn(&results_path(config_path)).as_ref(),
        config,
        config.enabled_models(),
        runtimes.hardware(),
    );
    runtimes.set_measured_orders(orders);
    Ok(runtimes)
}

/// Progress lines on stderr (none when `quiet`): each model/device load and every 50 timed
/// requests of long runs.
fn cli_progress(quiet: bool) -> impl Fn(&crate::benchmark::Progress) + Sync {
    move |p: &crate::benchmark::Progress| {
        if quiet {
            return;
        }
        if p.phase == Phase::Loading {
            eprintln!("  {} on {} ...", p.model, p.device);
        } else if p.phase == Phase::Burst && p.done == 0 {
            eprintln!("    burst test, {} requests ...", p.total);
        } else if p.phase == Phase::Timing
            && p.total >= 50
            && p.done > 0
            && p.done.is_multiple_of(50)
        {
            eprintln!("    {}/{} requests", p.done, p.total);
        }
    }
}

/// The GPUs a sweep of a model with selection `sel` runs, as [`sweep_with`] plans them: the
/// runnable GPU-class options (`allowed` = `benchmark.devices` limits them). A device of each
/// runtime counts on its own, as on the service's Benchmark page: the spec does not tell which of
/// them share a card.
///
/// [`sweep_with`]: crate::benchmark::sweep::sweep_with
fn sweep_gpus(
    sel: &crate::backend::Selection,
    allowed: Option<&[spec::Device]>,
) -> Vec<(spec::Device, String)> {
    let mut out: Vec<(spec::Device, String)> = Vec::new();
    for o in &sel.options {
        let listed = allowed.is_none_or(|a| a.contains(&o.spec));
        if o.runnable && listed && o.spec.is_gpu_like() && !out.iter().any(|(d, _)| *d == o.spec) {
            out.push((o.spec, o.label.clone()));
        }
    }
    out
}

/// Whether a sweep over `gpus` may start: None = yes; else the refusal. Two or more GPU devices
/// need `--confirm-gpus` (`confirmed`), as the web UI asks before such a sweep.
fn gpu_gate(gpus: &[(spec::Device, String)], confirmed: bool) -> Option<String> {
    (gpus.len() > 1 && !confirmed).then(|| several_gpus_refusal(gpus))
}

/// The refusal of a sweep over `gpus` (two or more) without `--confirm-gpus`.
fn several_gpus_refusal(gpus: &[(spec::Device, String)]) -> String {
    let names: Vec<String> = gpus
        .iter()
        .map(|(d, label)| {
            if label.is_empty() || *label == d.to_string() {
                d.to_string()
            } else {
                format!("{d} ({label})")
            }
        })
        .collect();
    format!(
        "this sweep would benchmark {} GPU devices: {}. {} To run them anyway, add \
         --confirm-gpus; to sweep fewer, list the devices in benchmark.devices of the config \
         (--apply, --apply-threshold, --threshold-search and --report sweep every device).",
        gpus.len(),
        names.join(", "),
        crate::benchmark::service::SEVERAL_GPUS_WARNING
    )
}

/// The error of a run Ctrl-C cancelled: `finished` models swept, saved to `saved` (None = nothing
/// saved: a run of single devices, `--no-save`, or no model finished).
fn cancelled_message(finished: usize, saved: Option<&Path>) -> String {
    match saved {
        Some(p) if finished > 0 => format!(
            "the benchmark was cancelled; the {finished} model(s) it finished were saved to {}",
            p.display()
        ),
        _ => "the benchmark was cancelled; nothing was saved".to_string(),
    }
}

/// Ctrl-C (and SIGTERM on Unix, Ctrl-Break on Windows) of a CLI benchmark: the first sets the
/// cancel flag, so the run stops at its next check and unwinds normally (its breadcrumb goes, the
/// priority is restored, the results lock is released); a second one quits at once (exit 130).
/// Stops listening when dropped. Without it the default handler would end the process mid-run,
/// leaving a breadcrumb the next start reports as an interrupted run.
struct Interrupts {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Interrupts {
    fn cancel(cancel: Arc<AtomicBool>) -> Self {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("ctrl-c".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::warn!("Ctrl-C will not cancel the benchmark cleanly: {e}");
                        return;
                    }
                };
                rt.block_on(async move {
                    let mut signals = Signals::new();
                    interrupt_loop(rx, &mut signals, &cancel, || std::process::exit(130)).await;
                });
            });
        match thread {
            Ok(t) => Self {
                stop: Some(tx),
                thread: Some(t),
            },
            Err(e) => {
                tracing::warn!("Ctrl-C will not cancel the benchmark cleanly: {e}");
                Self {
                    stop: None,
                    thread: None,
                }
            }
        }
    }
}

impl Drop for Interrupts {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The signals [`Interrupts`] listens to.
struct Signals {
    #[cfg(unix)]
    term: Option<tokio::signal::unix::Signal>,
    #[cfg(windows)]
    brk: Option<tokio::signal::windows::CtrlBreak>,
}

impl Signals {
    fn new() -> Self {
        Self {
            #[cfg(unix)]
            term: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok(),
            #[cfg(windows)]
            brk: tokio::signal::windows::ctrl_break().ok(),
        }
    }
}

/// Where [`interrupt_loop`] hears of interrupts.
trait Interrupt {
    /// The next interrupt; None when they cannot be listened to.
    fn next(&mut self) -> impl std::future::Future<Output = Option<()>>;
}

impl Interrupt for Signals {
    async fn next(&mut self) -> Option<()> {
        #[cfg(unix)]
        let other = async {
            match self.term.as_mut() {
                Some(t) => t.recv().await,
                None => std::future::pending().await,
            }
        };
        #[cfg(windows)]
        let other = async {
            match self.brk.as_mut() {
                Some(b) => b.recv().await,
                None => std::future::pending().await,
            }
        };
        #[cfg(not(any(unix, windows)))]
        let other = std::future::pending::<Option<()>>();
        tokio::select! {
            r = tokio::signal::ctrl_c() => r.ok(),
            r = other => r,
        }
    }
}

/// [`Interrupts`]' loop: the first interrupt from `source` sets `cancel` (and says so on
/// stderr), the second calls `quit`; `done` firing (the run ended) ends it.
async fn interrupt_loop(
    mut done: tokio::sync::oneshot::Receiver<()>,
    source: &mut impl Interrupt,
    cancel: &AtomicBool,
    quit: impl FnOnce(),
) {
    tokio::select! {
        _ = &mut done => return,
        r = source.next() => if r.is_none() { return },
    }
    eprintln!(
        "Ctrl-C: cancelling the benchmark; the device in hand stops at its next step (a compile \
         in progress ends first). Press Ctrl-C again to quit at once."
    );
    cancel.store(true, Ordering::SeqCst);
    tokio::select! {
        _ = &mut done => {}
        r = source.next() => if r.is_some() { quit() },
    }
}

/// What the steps of [`run`] share once the runtimes are open.
struct RunCtx<'a> {
    args: &'a BenchArgs,
    config: &'a Config,
    config_path: &'a Path,
    runtimes: &'a Mutex<Runtimes>,
    /// `benchmark.devices` of the config (None = every runnable option).
    sweep_devices: &'a Option<Vec<spec::Device>>,
    objective: Objective,
    metric: AccuracyMetric,
    /// The goal of this run (the config's, with the flags).
    goal: &'a crate::benchmark::eval::Goal,
    weights: crate::benchmark::grade::Weights,
    /// Names of the variant jobs (`<model>+<variant>`): never applied to the config.
    variant_names: &'a std::collections::HashSet<String>,
    /// The run's breadcrumb (see [`breadcrumb`]): noted before each device of a sweep and the
    /// pseudo-ground-truth reference starts, so a run killed there is reported like the
    /// service's.
    trail: &'a Trail,
}

impl RunCtx<'_> {
    /// The device selection for the model at `path`.
    fn selection(&self, path: &Path) -> crate::backend::Selection {
        self.runtimes.lock_recover().selection(Some(path))
    }

    /// Fill the ground truth of the sets without it from the reference model; a failure only
    /// warns (accuracy is then not scored on those sets).
    fn pseudo_ground_truth(
        &self,
        pb: &Bench,
        jobs: &[Planned],
        sets: &mut [ImageSet],
        report: &mut Report,
    ) {
        let (args, config) = (self.args, self.config);
        let reference = pick_reference(
            config,
            args.reference_model
                .as_deref()
                .or(config.benchmark.reference_model.as_deref()),
        )
        .and_then(|m| job_for_config(config, m, None));
        let outcome = reference.and_then(|job| {
            let sel = self.selection(&job.path);
            let allowed = self.sweep_devices.as_deref();
            let device =
                pseudo_gt_device(&sel, allowed).with_context(|| no_pseudo_gt_device(allowed))?;
            if !args.json {
                eprintln!("pseudo ground truth: {} on {device} ...", job.name);
            }
            // Its As served / variant jobs too, by the scoring's own rule (B10, QA-027).
            if jobs
                .iter()
                .any(|p| crate::benchmark::variant::owned_by(&p.job.name, &job.name))
            {
                report.warn(
                    args.json,
                    format!(
                        "{} is also the pseudo-ground-truth reference: it is not scored on the \
                         datasets without ground truth (their labels are its own detections)",
                        job.name
                    ),
                );
            }
            self.trail.note(&device.to_string(), &job.name);
            pseudo_ground_truth(pb, self.runtimes, &job, device, sets)
        });
        if let Err(e) = outcome {
            report.warn(
                args.json,
                format!("accuracy is not scored for datasets without ground truth: {e:#}"),
            );
        }
    }

    /// Run every job: a device sweep each with `all_devices` (returned), else one run on its
    /// device (printed and recorded in `report`).
    fn run_jobs<'b>(
        &self,
        bench: &mut Bench<'b>,
        progress: &'b (dyn Fn(&crate::benchmark::Progress) + Sync),
        jobs: &[Planned],
        (sets, own_sets): (&'b [ImageSet], &'b [OwnSets]),
        all_devices: bool,
        report: &mut Report,
    ) -> Result<Vec<ModelResult>> {
        let mut swept: Vec<ModelResult> = Vec::new();
        for (k, Planned { job, entry }) in jobs.iter().enumerate() {
            if bench.cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                break;
            }
            let (own, left_out) = own_sets
                .get(k)
                .map_or((None, None), |(s, l)| (s.as_deref(), l.clone()));
            bench.sets = own.unwrap_or(sets);
            if bench.sets.is_empty() {
                report.error(
                    self.args.json,
                    format!(
                        "{}: {}",
                        job.name,
                        crate::benchmark::cameras::nothing_left(left_out.as_ref())
                    ),
                );
                continue;
            }
            if let Some(w) = left_out.as_ref().and_then(|l| l.unknown.as_deref()) {
                report.warn(
                    self.args.json,
                    format!(
                        "{w}; it is scored on every frame of {}",
                        crate::benchmark::cameras::TITLE
                    ),
                );
            }
            if all_devices {
                let mut result = match self.sweep_job(bench, progress, job, entry.as_ref(), report)
                {
                    Ok(r) => r,
                    // Ctrl-C: the models finished so far are the run's.
                    Err(e) if crate::benchmark::is_cancelled(&e) => break,
                    Err(e) => return Err(e),
                };
                if let Some(l) = left_out.as_ref().filter(|_| !self.args.json) {
                    println!(
                        "  {}: {} ({} of the set's frames run)",
                        crate::benchmark::cameras::TITLE,
                        l.why(),
                        bench
                            .sets
                            .iter()
                            .find(|s| s.id == crate::benchmark::cameras::ID)
                            .map_or(0, |s| s.images.len())
                    );
                }
                result.left_out = left_out;
                swept.push(result);
                continue;
            }
            if let Some(model_report) = self.run_job(bench, job, report) {
                if !self.args.json {
                    print_model(&model_report);
                }
                report.models.push(model_report);
            }
        }
        Ok(swept)
    }

    /// `--all-devices` for one job: sweep its devices, print and record the model report.
    fn sweep_job<'b>(
        &self,
        bench: &mut Bench<'b>,
        progress: &'b (dyn Fn(&crate::benchmark::Progress) + Sync),
        job: &Job,
        entry: Option<&ModelConfig>,
        report: &mut Report,
    ) -> Result<ModelResult> {
        let configured = entry.and_then(|m| {
            let sel = self.selection(&job.path);
            configured_device(self.config, m, &sel)
        });
        let opts = SweepOptions {
            devices: self.sweep_devices.clone(),
            configured,
            threshold_objective: self.objective,
        };
        bench.progress = Some(progress);
        let mut result = crate::benchmark::sweep::sweep_with(
            bench,
            self.runtimes,
            job,
            &opts,
            &mut |device| self.trail.note(&device.to_string(), &job.name),
            &mut |_| {},
        )?;
        bench.progress = None;
        if let Some(m) = entry {
            let files = bench
                .sets
                .iter()
                .flat_map(|s| s.images.iter())
                .map(|i| i.file.as_str());
            let (share, warning) = crate::benchmark::trained_on::of_model(self.config, m, files);
            result.trained_on_share = share;
            if let Some(w) = warning {
                tracing::warn!("{w}");
            }
        }
        match model_report(&result) {
            Ok(m) => {
                if !self.args.json {
                    print_all_devices(&result, &m);
                }
                report.models.push(m);
            }
            Err(e) => report.error(self.args.json, format!("{}: {e:#}", job.name)),
        }
        Ok(result)
    }

    /// One job on its device (and on CPU with `--compare-cpu`); None (error recorded) when it
    /// failed.
    fn run_job(&self, bench: &Bench, job: &Job, report: &mut Report) -> Option<ModelReport> {
        let primary = match bench.run(self.runtimes, job, &job.device) {
            Ok(r) => r,
            Err(e) if crate::benchmark::is_cancelled(&e) => return None,
            Err(e) => {
                report.error(self.args.json, format!("{}: {e:#}", job.name));
                return None;
            }
        };
        let mut model_report = ModelReport {
            primary,
            cpu: None,
            comparison: None,
            note: None,
            all_devices: Vec::new(),
            skipped_devices: Vec::new(),
            recommended: None,
            recommendation: None,
            reference: None,
            threshold: None,
        };
        if self.args.compare_cpu {
            if model_report.primary.device == "CPU" {
                model_report.note =
                    Some("--compare-cpu skipped: the model already runs on CPU".into());
            } else {
                match bench.run(self.runtimes, job, "CPU") {
                    Ok(cpu) => {
                        // QA-049: both runs at one threshold, paired by image name, like the
                        // sweep's agreement (saved device profiles are no disagreement).
                        let c = crate::benchmark::report::paired_comparison(
                            &model_report.primary,
                            &cpu,
                        )
                        .map(|(c, _, _)| c)
                        .unwrap_or_default();
                        model_report.comparison = Some(c);
                        model_report.cpu = Some(cpu);
                    }
                    Err(e) => report.error(self.args.json, format!("{} on CPU: {e:#}", job.name)),
                }
            }
        }
        Some(model_report)
    }

    /// After an `--all-devices` sweep: rank, print the summaries, save the results, apply
    /// devices / thresholds, write the report. Some(outcome) when `--threshold-search` ends the
    /// run.
    fn finish_sweep(
        &self,
        sets: &[ImageSet],
        cameras: Option<&images::CamerasLoaded>,
        versions: RuntimeVersions,
        swept: &[ModelResult],
        report: &mut Report,
    ) -> Result<Option<bool>> {
        let args = self.args;
        let results = BenchmarkResults::new(HardwareSummary::current(), versions, swept.to_vec())
            .with_sets(sets);
        report.ranking = results.ranking();
        if !args.json && report.ranking.len() > 1 {
            print_ranking(&report.ranking);
        }
        if !args.json {
            print_thresholds(swept, self.objective);
            print_frame_roc(swept);
        }
        let merged = self.save_results(results, swept, sets, cameras, report)?;
        let view = self.compare(&merged, swept);
        if !args.json {
            print_compare(&view);
        }
        report.compare = Some(view);
        // Variants are measurements only: the config gets the plain pipeline's picks.
        let plain: Vec<ModelResult> = swept
            .iter()
            .filter(|m| !self.variant_names.contains(&m.model))
            .cloned()
            .collect();
        if args.apply {
            report.applied = self.apply_devices(&plain)?;
        }
        if args.apply_threshold {
            let preds = crate::benchmark::search::StoredPreds::merge(None, &merged, swept);
            let names: Vec<String> = plain.iter().map(|m| m.model.clone()).collect();
            let suggested = crate::benchmark::eval::tune::suggestions(
                &merged,
                &preds,
                self.goal,
                self.weights,
                Some(&names),
            );
            report.applied_thresholds = apply_thresholds(&suggested, self.config_path, args.json)?;
        }
        if let Some(out) = &args.report {
            self.write_report(&merged, swept, out)?;
        }
        if args.threshold_search {
            let preds = crate::benchmark::search::StoredPreds::merge(None, &merged, swept);
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            return Ok(Some(
                threshold_search_cli(args, self.config, &merged, &preds)?
                    && report.errors.is_empty(),
            ));
        }
        Ok(None)
    }

    /// Merge `results` into `benchmark.json` (and the stored predictions) beside the config
    /// and record the models in the run history unless `--no-save`; returns what was saved
    /// (`results` itself with `--no-save`).
    fn save_results(
        &self,
        results: BenchmarkResults,
        swept: &[ModelResult],
        sets: &[ImageSet],
        cameras: Option<&images::CamerasLoaded>,
        report: &mut Report,
    ) -> Result<BenchmarkResults> {
        if self.args.no_save {
            return Ok(results);
        }
        let path = results_path(self.config_path);
        // One load-modify-save at a time, in this process and against the service.
        let _held = crate::benchmark::results_io::lock(
            &path,
            crate::benchmark::results_io::Holder::Writer,
        )?;
        let (prev, warnings) = BenchmarkResults::load_for_merge(&path)?;
        // The labels Your cameras was scored against, before the results that leave them out;
        // with the frames the kept runs of `prev` still score.
        let snapshot =
            crate::benchmark::cameras::write_snapshot(&path, sets, cameras, prev.as_ref())?;
        for w in warnings {
            report.warn(self.args.json, w);
        }
        let crate::benchmark::save::Committed {
            merged,
            seeds,
            merge_warning,
            ..
        } = crate::benchmark::save::commit(
            &path,
            prev,
            results,
            BenchmarkResults::merge,
            snapshot.as_ref(),
            (self.metric, self.weights),
        )?;
        if let Some(w) = merge_warning {
            report.warn(self.args.json, w);
        }
        for w in crate::benchmark::search::save_beside(&path, &merged, swept) {
            report.warn(self.args.json, w);
        }
        let names: Vec<String> = swept.iter().map(|m| m.model.clone()).collect();
        for w in crate::benchmark::history::record(
            &path,
            &seeds,
            &merged,
            &names,
            Some((self.goal, self.weights)),
        ) {
            report.warn(self.args.json, w);
        }
        if !self.args.json {
            println!("\nresults saved to {}", path.display());
        }
        report.results_file = Some(path.display().to_string());
        Ok(merged)
    }

    /// The Compare view of `merged` on its default set for the run's goal (the stored
    /// predictions of the file beside the config, with this run's).
    fn compare(
        &self,
        merged: &BenchmarkResults,
        swept: &[ModelResult],
    ) -> crate::benchmark::eval::compare::CompareView {
        let preds = crate::benchmark::search::StoredPreds::merge(
            crate::benchmark::search::StoredPreds::load(&crate::benchmark::search::preds_path(
                self.config_path,
            ))
            .ok()
            .flatten(),
            merged,
            swept,
        );
        let device = |m: &crate::config::ModelConfig| {
            let sel = self.selection(&self.config.data_path(&m.path));
            configured_device(self.config, m, &sel)
        };
        let now = crate::benchmark::current::from_config(self.config, &device, &|_| None);
        compare_view(self.config, merged, &preds, &now, self.goal, self.weights)
    }

    /// `--apply`: write the recommended devices of `swept` into the config file.
    fn apply_devices(&self, swept: &[ModelResult]) -> Result<Vec<DeviceChange>> {
        let config_path = self.config_path;
        let picks: Vec<(String, Option<String>)> = swept
            .iter()
            .filter_map(|m| m.recommended.clone().map(|d| (m.model.clone(), Some(d))))
            .collect();
        // Under the config lock (QA-016): a page save of a running service is never undone.
        let edited = crate::config_file::edit_file(
            config_path,
            crate::config_file::Missing::Error,
            |cfg| apply_model_devices(cfg, &picks),
        )?;
        let (cfg, changes) = (edited.config, edited.value);
        if !self.args.json {
            println!();
            if changes.is_empty() {
                println!(
                    "--apply: {} already uses the recommended devices; nothing changed",
                    config_path.display()
                );
            } else {
                println!("--apply: updated {}:", config_path.display());
                for c in &changes {
                    println!("  {}", c.describe());
                }
                println!("Restart the server to load the models on their new devices.");
            }
            for m in swept.iter().filter(|m| m.recommended.is_none()) {
                println!("  {}: unchanged ({})", m.model, m.recommendation);
            }
            if cfg.force_cpu {
                println!(
                    "note: force_cpu is on in the config and overrides every per-model device"
                );
            }
        }
        Ok(changes)
    }

    /// `--report`: write the standalone report of `merged`, compared with the config as it is
    /// now (after `--apply` / `--apply-threshold`).
    fn write_report(
        &self,
        merged: &BenchmarkResults,
        swept: &[ModelResult],
        out: &Path,
    ) -> Result<()> {
        let config_path = self.config_path;
        // After --apply / --apply-threshold: the report compares with the config as it is
        // when written (the config file, else the config this run used).
        let now_cfg = Config::load(config_path)
            .ok()
            .filter(|_| config_path.exists())
            .unwrap_or_else(|| self.config.clone());
        let preds = crate::benchmark::search::StoredPreds::merge(
            crate::benchmark::search::StoredPreds::load(&crate::benchmark::search::preds_path(
                config_path,
            ))
            .ok()
            .flatten(),
            merged,
            swept,
        );
        let device = |m: &crate::config::ModelConfig| {
            let sel = self.selection(&now_cfg.data_path(&m.path));
            configured_device(&now_cfg, m, &sel)
        };
        let now = crate::benchmark::current::from_config(&now_cfg, &device, &|_| None);
        let mut view = merged.clone();
        crate::benchmark::current::rebase(&mut view, Some(&preds), &now);
        export::write(&view, out)?;
        if !self.args.json {
            println!("report written to {}", out.display());
        }
        Ok(())
    }
}

/// The Compare view of `results` on its default set (pure but for the config's training
/// records and model files, [`crate::benchmark::combos::Parts`]).
pub(crate) fn compare_view(
    config: &Config,
    results: &BenchmarkResults,
    preds: &crate::benchmark::search::StoredPreds,
    now: &std::collections::HashMap<String, crate::benchmark::current::ModelNow>,
    goal: &crate::benchmark::eval::Goal,
    weights: crate::benchmark::grade::Weights,
) -> crate::benchmark::eval::compare::CompareView {
    use crate::benchmark::combos::{Inputs, Parts};
    let parts = Parts::of(results, config);
    let default_model = crate::training::switch::current_default(config);
    let inputs = Inputs {
        results,
        preds: Some(preds),
        now,
        default_model: default_model.as_deref(),
        configured: &parts.configured,
        trained: &parts.trained,
        seen: &parts.seen,
        reachable: &|s| s.dir.as_deref().is_some_and(Path::is_dir),
    };
    crate::benchmark::eval::compare::build(
        &inputs,
        crate::benchmark::eval::compare::Request {
            goal,
            weights,
            set: None,
            scene: None,
        },
    )
}

/// `--threshold-search`: search the stored predictions and print the results.
fn threshold_search_cli(
    args: &BenchArgs,
    config: &Config,
    results: &BenchmarkResults,
    preds: &crate::benchmark::search::StoredPreds,
) -> Result<bool> {
    let req = search_request(args, config);
    let current =
        |m: &str, device: &str| crate::benchmark::search::configured_threshold(config, m, device);
    let out = crate::benchmark::search::search(results, preds, &req, &current)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(out.errors.is_empty());
    }
    print_search_summary(&req, &out);
    for r in &out.results {
        if let Some(roc) = &r.roc {
            println!(
                "  {} frame ROC AUC: macro {} micro {} ({} positive / {} negative frame samples); {}",
                r.model,
                pct1(roc.auc),
                pct1(roc.micro_auc),
                roc.positives,
                roc.negatives,
                class_aucs(roc)
            );
        }
    }
    for r in &out.results {
        print_search_detail(r);
    }
    // The cross-validated pick of each searched combo for the objective (on its default set).
    let goal = goal_of(args, config)?;
    let goal = crate::benchmark::eval::Goal {
        optimize_for: crate::benchmark::eval::Optimize::Threshold(req.objective),
        ..goal
    };
    for r in &out.results {
        let saved = None;
        match crate::benchmark::eval::tune::build(
            results,
            preds,
            crate::benchmark::eval::tune::Request {
                model: &r.model,
                device: Some(&r.device),
                set: None,
                goal: &goal,
                weights: config.benchmark.weights,
                saved,
            },
        ) {
            Ok(t) => print_cv_pick(&t),
            Err(e) => println!(
                "  {} on {}: no cross-validated pick ({e:#})",
                r.model, r.device
            ),
        }
    }
    for e in &out.errors {
        eprintln!("error: {e}");
    }
    println!("{BLUE_IRIS_NOTE}");
    Ok(out.errors.is_empty())
}

/// The `--threshold-search` request of `args` (defaults from `config`).
fn search_request(args: &BenchArgs, config: &Config) -> crate::benchmark::search::SearchRequest {
    crate::benchmark::search::SearchRequest {
        models: args.search_model.clone(),
        device: args.search_device.clone(),
        datasets: args.dataset.clone(),
        tags: args.tag.clone(),
        classes: args.class.clone(),
        objective: goal_of(args, config)
            .map(|g| g.tuning())
            .unwrap_or(config.benchmark.threshold_objective),
        iou: args.iou.unwrap_or(0.5),
    }
}

/// `--apply-threshold`: write the suggested confidence of every device each config model ran on
/// (`suggested`, [`crate::benchmark::eval::tune::suggestions`]: the Tune view's cross-validated
/// pick for the run's goal, the same rule as the web UI's Save all) into its confidence profiles
/// (`device_thresholds`, [`apply_device_thresholds`]); the model's `confidence_threshold` is left
/// alone. A pick that must not be saved (pseudo labels, a target not met on its bound, a flat
/// curve) is left alone and named; a low-confidence one is written (the command asked for it) and
/// marked. Prints one line per model and device.
fn apply_thresholds(
    suggested: &[crate::benchmark::eval::tune::Suggested],
    config_path: &Path,
    quiet: bool,
) -> Result<Vec<ProfileChange>> {
    // Under the config lock (QA-016): a page save of a running service is never undone.
    let (configured, changes) =
        crate::config_file::edit_file(config_path, crate::config_file::Missing::Error, |cfg| {
            // Results may name models that are not configured (--model files): skip those.
            let configured: Vec<&crate::benchmark::eval::tune::Suggested> = suggested
                .iter()
                .filter(|s| crate::benchmark::find_model(cfg, &s.model).is_some())
                .collect();
            let entries: Vec<(String, String, Option<f32>)> = configured
                .iter()
                .filter(|s| s.refusal.is_none())
                .filter_map(|s| Some((s.model.clone(), s.device.clone(), Some(s.threshold?))))
                .collect();
            let changes = apply_device_thresholds(cfg, &entries)?;
            Ok((configured, changes))
        })?
        .value;
    if !quiet {
        println!();
        if changes.is_empty() {
            println!(
                "--apply-threshold: {} already has these confidence profiles; nothing changed",
                config_path.display()
            );
        } else {
            println!(
                "--apply-threshold: updated the confidence profiles in {} (the threshold each \
                 model uses on that device):",
                config_path.display()
            );
        }
        for s in &configured {
            let changed = changes.iter().find(|c| {
                crate::registry::normalize_name(&c.model)
                    == crate::registry::normalize_name(&s.model)
                    && crate::config::same_device(&c.device, &s.device)
            });
            let low = s
                .low_confidence
                .as_ref()
                .map(|l| format!(" ({l})"))
                .unwrap_or_default();
            match (changed, s.threshold, &s.refusal) {
                (_, _, Some(r)) => println!("  {} on {}: unchanged ({r})", s.model, s.device),
                (Some(c), _, None) => println!("  {}{low}", c.describe()),
                (None, Some(t), None) => {
                    println!("  {} on {}: {t:.2} (unchanged){low}", s.model, s.device)
                }
                (None, None, None) => {
                    println!("  {} on {}: unchanged (no suggestion)", s.model, s.device)
                }
            }
        }
        if !changes.is_empty() {
            println!(
                "A running server applies them from the file within seconds (no restart \
                 needed); a device without a profile keeps the model's threshold."
            );
        }
        println!("{BLUE_IRIS_NOTE}");
    }
    Ok(changes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct Cli {
        #[command(flatten)]
        args: BenchArgs,
    }

    fn parse(argv: &[&str]) -> Result<BenchArgs, clap::Error> {
        Cli::try_parse_from(argv).map(|c| c.args)
    }

    /// S2: `--goal`, `--tune-for` and `--threshold-objective` refuse a 100% target.
    #[test]
    fn a_target_no_finite_set_can_show_is_refused_on_the_command_line() {
        for flag in ["--goal", "--tune-for", "--threshold-objective"] {
            let e = parse(&["x", flag, "precision:1"]).unwrap_err().to_string();
            assert!(e.contains("100%"), "{flag}: {e}");
        }
    }

    /// `--goal` and its companions set the run's goal; the old flags map onto it.
    #[test]
    fn goal_flags_and_the_old_flags_map_onto_the_goal() {
        use crate::benchmark::eval::Optimize;
        let config = Config::default();
        let a = parse(&[
            "x",
            "--goal",
            "ap50_95",
            "--tune-for",
            "f2",
            "--goal-classes",
            "person,car",
            "--speed-budget",
            "1.5s",
        ])
        .unwrap();
        let g = goal_of(&a, &config).unwrap();
        assert_eq!(g.optimize_for, Optimize::Ap5095);
        assert_eq!(g.tune_for, Objective::F2);
        assert_eq!(g.classes, ["person", "car"]);
        assert_eq!(g.speed_budget_ms, Some(1500.0));
        assert_eq!(g.legacy(), (Objective::F2, AccuracyMetric::Ap50));
        // The old flags.
        let a = parse(&[
            "x",
            "--accuracy-metric",
            "roc_auc",
            "--threshold-objective",
            "youden",
        ])
        .unwrap();
        let g = goal_of(&a, &config).unwrap();
        assert_eq!(
            (g.optimize_for, g.tune_for),
            (Optimize::RocAuc, Objective::Youden)
        );
        let a = parse(&["x", "--threshold-objective", "precision:0.9"]).unwrap();
        assert_eq!(
            goal_of(&a, &config).unwrap().optimize_for,
            Optimize::Threshold(Objective::Precision(0.9))
        );
        // The config's goal is the default; bad values are refused by clap.
        let mut c = Config::default();
        c.benchmark.set_goal(crate::benchmark::eval::Goal {
            optimize_for: Optimize::Grade,
            ..Default::default()
        });
        assert_eq!(
            goal_of(&parse(&["x"]).unwrap(), &c).unwrap().optimize_for,
            Optimize::Grade
        );
        assert!(parse(&["x", "--goal", "f3"]).is_err());
        assert!(parse(&["x", "--speed-budget", "-1"]).is_err());
        assert!(parse(&["x", "--tune-for", "ap50"]).is_err());
    }

    /// `--burst`: bare = the config's load turned on, `C,K,I` sets it, out-of-range or malformed
    /// values are refused by clap; without the flag `benchmark.burst.enabled` decides.
    #[test]
    fn the_burst_flag() {
        use crate::config::BurstConfig;
        let mut config = Config::default();
        assert_eq!(burst_of(&parse(&["x"]).unwrap(), &config), None);
        config.benchmark.burst.cameras = 6;
        let bare = parse(&["x", "--burst", "--all-devices"]).unwrap();
        assert!(bare.all_devices, "a bare --burst takes no value");
        assert_eq!(
            burst_of(&bare, &config),
            Some(BurstConfig {
                enabled: true,
                cameras: 6,
                ..BurstConfig::default()
            })
        );
        assert_eq!(
            burst_of(&parse(&["x", "--burst", "5,3,250"]).unwrap(), &config),
            Some(BurstConfig {
                enabled: true,
                cameras: 5,
                images: 3,
                interval_ms: 250
            })
        );
        for bad in [
            "0,3,250",
            "33,3,250",
            "5,21,250",
            "5,3,49",
            "5,3",
            "a,b,c",
            "5,3,250,1",
        ] {
            assert!(parse(&["x", "--burst", bad]).is_err(), "{bad}");
        }
        config.benchmark.burst.enabled = true;
        assert_eq!(
            burst_of(&parse(&["x"]).unwrap(), &config).map(|b| b.cameras),
            Some(6)
        );
    }

    /// `--served` with `--variants` is refused with a message saying why; `benchmark.served`
    /// is the default, and `--variants` runs instead of it.
    #[test]
    fn served_mode_and_its_refusal() {
        let mut config = Config::default();
        let both = parse(&["x", "--served", "--variants", "filters"]).unwrap();
        let e = served_mode(&both, &config).unwrap_err().to_string();
        assert!(
            e.contains("--served and --variants cannot be combined"),
            "{e}"
        );
        let mut errors = Vec::new();
        let e = jobs(&both, &config, &mut errors).err().unwrap().to_string();
        assert!(e.contains("cannot be combined"), "{e}");
        assert!(served_mode(&parse(&["x", "--served"]).unwrap(), &config).unwrap());
        assert!(!served_mode(&parse(&["x"]).unwrap(), &config).unwrap());
        config.benchmark.served = true;
        assert!(served_mode(&parse(&["x"]).unwrap(), &config).unwrap());
        let variants = parse(&["x", "--json", "--variants", "filters"]).unwrap();
        assert!(!served_mode(&variants, &config).unwrap());
    }

    #[test]
    fn timing_cache_and_arg_checks() {
        let config = Config::default();
        let one = vec![ImageSet::sample()];
        let two = vec![ImageSet::sample(), ImageSet::sample()];
        let args = parse(&["x"]).unwrap();
        let b = &config.benchmark;
        assert_eq!(
            timing_settings(&args, &config, &one),
            (SINGLE_IMAGE_REPEAT, b.warmup, b.weights)
        );
        assert_eq!(
            timing_settings(&args, &config, &two),
            (b.repeat_per_image.max(1), b.warmup, b.weights)
        );
        let args = parse(&[
            "x",
            "--repeat",
            "7",
            "--warmup",
            "2",
            "--accuracy-weight",
            "0.25",
        ])
        .unwrap();
        let (repeat, warmup, w) = timing_settings(&args, &config, &one);
        assert_eq!((repeat, warmup, w.accuracy, w.speed), (7, 2, 0.25, 0.75));

        assert_eq!(
            cache_dir_arg(&parse(&["x", "--cache-dir", " "]).unwrap(), &config),
            None
        );
        assert_eq!(
            cache_dir_arg(&parse(&["x", "--cache-dir", "c2"]).unwrap(), &config),
            Some(PathBuf::from("c2"))
        );
        assert_eq!(
            cache_dir_arg(&parse(&["x"]).unwrap(), &config),
            config.cache_dir_path()
        );

        let missing = Path::new("/nonexistent/blue-onyx-prism/cfg.json");
        assert!(check_args(&parse(&["x"]).unwrap(), missing).is_ok());
        let err = check_args(&parse(&["x", "--apply-threshold"]).unwrap(), missing).unwrap_err();
        assert!(
            err.to_string()
                .starts_with("--apply-threshold needs a config file")
        );
        // Out-of-range weights are rejected by the argument parser.
        assert!(parse(&["x", "--accuracy-weight", "1.5"]).is_err());
    }

    #[test]
    fn args_parse() {
        let a = parse(&[
            "x",
            "--model",
            "a.onnx",
            "--model",
            "b.xml",
            "--family",
            "yolo5",
            "--repeat",
            "20",
            "--compare-cpu",
            "--json",
        ])
        .unwrap();
        assert_eq!(a.model.len(), 2);
        assert_eq!(a.repeat, Some(20));
        assert_eq!(a.warmup, None);
        assert!(a.compare_cpu && a.json);
        let a = parse(&["x"]).unwrap();
        assert_eq!(a.repeat, None);
        assert!(!a.all_devices && !a.apply && !a.no_save && !a.no_low_priority);
        assert!(parse(&["x", "--no-low-priority"]).unwrap().no_low_priority);
        let a = parse(&[
            "x",
            "--dataset",
            "coco-cctv",
            "--dataset",
            "dir:/srv/a",
            "--max-images",
            "10",
            "--report",
            "r.html",
            "--accuracy-weight",
            "0.5",
        ])
        .unwrap();
        assert_eq!(a.dataset, ["coco-cctv", "dir:/srv/a"]);
        assert_eq!(a.max_images, Some(10));
        assert!(parse(&["x", "--image", "a.jpg", "--dataset", "sample"]).is_err());
        assert!(
            parse(&["x", "--gt", "g.json"]).is_err(),
            "--gt needs --images"
        );
        assert!(parse(&["x", "--report", "r.md", "--device", "cpu"]).is_err());
    }

    #[test]
    fn device_all_devices_and_apply_flags() {
        let a = parse(&["x", "--device", "ORT:CUDA:1"]).unwrap();
        assert_eq!(a.device.as_deref(), Some("ORT:CUDA:1"));
        assert!(parse(&["x", "--device", "vulkan"]).is_err());
        let a = parse(&["x", "--all-devices"]).unwrap();
        assert!(a.all_devices);
        assert!(parse(&["x", "--all-devices", "--device", "cpu"]).is_err());
        assert!(parse(&["x", "--all-devices", "--force-cpu"]).is_err());
        let a = parse(&["x", "--all-devices", "--apply", "--config", "c.json"]).unwrap();
        assert!(a.apply);
        assert!(parse(&["x", "--apply", "--model", "a.onnx"]).is_err());
        assert!(parse(&["x", "--apply", "--device", "cpu"]).is_err());
        // Thresholds: separate from --apply, both allowed together.
        let a = parse(&[
            "x",
            "--apply",
            "--apply-threshold",
            "--threshold-objective",
            "precision:0.9",
        ])
        .unwrap();
        assert!(a.apply && a.apply_threshold);
        assert_eq!(a.threshold_objective, Some(Objective::Precision(0.9)));
        let a = parse(&["x", "--apply-threshold"]).unwrap();
        assert!(a.apply_threshold && !a.apply && a.threshold_objective.is_none());
        assert!(parse(&["x", "--threshold-objective", "f3"]).is_err());
        let a = parse(&[
            "x",
            "--threshold-search",
            "--threshold-objective",
            "recall_at_fa:5",
        ])
        .unwrap();
        assert_eq!(a.threshold_objective, Some(Objective::RecallAtFa(5.0)));
        let a = parse(&["x", "--threshold-objective", "f0.5"]).unwrap();
        assert_eq!(a.threshold_objective, Some(Objective::F05));
        assert!(parse(&["x", "--threshold-objective", "recall_at_fa:0"]).is_err());
        assert!(parse(&["x", "--apply-threshold", "--model", "a.onnx"]).is_err());
        assert!(parse(&["x", "--apply-threshold", "--min-confidence", "0.3"]).is_err());
    }

    #[test]
    fn apply_needs_an_existing_config() {
        let dir = std::env::temp_dir().join(format!("bop-bcli-{}", uuid::Uuid::new_v4()));
        let a = parse(&[
            "x",
            "--apply",
            "--config",
            dir.join("missing.json").to_str().unwrap(),
        ])
        .unwrap();
        let err = run(a).unwrap_err();
        assert!(
            format!("{err:#}").contains("needs a config file"),
            "{err:#}"
        );
        let a = parse(&[
            "x",
            "--apply-threshold",
            "--config",
            dir.join("missing.json").to_str().unwrap(),
        ])
        .unwrap();
        let err = run(a).unwrap_err();
        assert!(
            format!("{err:#}").contains("--apply-threshold needs a config file"),
            "{err:#}"
        );
    }

    #[test]
    fn threshold_search_uses_stored_predictions() {
        assert!(
            parse(&["x", "--tag", "night"]).is_err(),
            "--tag needs --threshold-search"
        );
        assert!(parse(&["x", "--threshold-search", "--device", "cpu"]).is_err());
        let a = parse(&[
            "x",
            "--threshold-search",
            "--tag",
            "night",
            "--class",
            "person",
            "--iou",
            "0.6",
            "--threshold-objective",
            "recall:0.9",
        ])
        .unwrap();
        assert_eq!((a.tag.len(), a.class.len(), a.iou), (1, 1, Some(0.6)));
        assert_eq!(a.threshold_objective, Some(Objective::Recall(0.9)));

        let dir = std::env::temp_dir().join(format!("bop-bsearch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("cfg.json");
        Config::default().save(&cfg).unwrap();
        let (results, preds) = crate::benchmark::search::tests::fixture();
        results.save(&results_path(&cfg)).unwrap();
        preds
            .save(&crate::benchmark::search::preds_path(&cfg))
            .unwrap();
        let c = cfg.to_str().unwrap();
        assert!(
            run(parse(&["x", "--threshold-search", "--json", "--config", c]).unwrap()).unwrap()
        );
        assert!(
            run(parse(&[
                "x",
                "--threshold-search",
                "--tag",
                "night",
                "--search-device",
                "openvino:cpu",
                "--config",
                c
            ])
            .unwrap())
            .unwrap()
        );
        // A model without stored predictions: reported, not fatal; exit status false.
        assert!(
            !run(parse(&[
                "x",
                "--threshold-search",
                "--search-model",
                "m",
                "--search-model",
                "zzz",
                "--json",
                "--config",
                c
            ])
            .unwrap())
            .unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--apply-threshold` writes the suggestions (the Tune view's picks, the web UI's rule) of
    /// the configured models into their profiles; an unsavable one stays as it was.
    #[test]
    fn apply_thresholds_writes_the_suggestions() {
        use crate::benchmark::eval::tune::{Suggested, suggestions};
        let dir = std::env::temp_dir().join(format!("bop-bthr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cfg.json");
        let cfg = Config {
            models: ["ipcam-general", "dfine-s"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        cfg.save(&path).unwrap();
        let (results, preds) = crate::benchmark::eval::tests::fixture(10, 10);
        let goal = crate::benchmark::eval::Goal::default();
        let mut s = suggestions(&results, &preds, &goal, Default::default(), None);
        assert_eq!(s.len(), 3, "every stored combo");
        assert!(
            s.iter()
                .all(|x| x.threshold.is_some() && x.refusal.is_none()),
            "{s:?}"
        );
        // One the page would refuse, and one of a model that is not configured.
        s[2].refusal = Some("not savable: a flat curve".into());
        s.push(Suggested {
            model: "removed".into(),
            device: "ort:cpu".into(),
            threshold: Some(0.4),
            refusal: None,
            low_confidence: None,
            recommended: true,
            figures: None,
        });
        let changes = apply_thresholds(&s, &path, true).unwrap();
        let want: Vec<(String, String, Option<f32>, Option<f32>)> = s[..2]
            .iter()
            .map(|x| (x.model.clone(), x.device.clone(), None, x.threshold))
            .collect();
        assert_eq!(
            changes
                .iter()
                .map(|c| (c.model.clone(), c.device.clone(), c.from, c.to))
                .collect::<Vec<_>>(),
            want
        );
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.models[0].device_thresholds.len(), 2);
        assert!(
            saved.models[1].device_thresholds.is_empty(),
            "refused: unchanged"
        );
        assert_eq!(saved.models[0].confidence_threshold, None);
        // Again: nothing to change (and printing works).
        assert!(apply_thresholds(&s, &path, false).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dataset_precedence() {
        let mut config = Config::default();
        config.benchmark.datasets = vec![DatasetRef::Id("sample".into())];
        let a = parse(&["x"]).unwrap();
        assert_eq!(dataset_refs(&a, &config), config.benchmark.datasets);
        let a = parse(&[
            "x",
            "--dataset",
            "coco-cctv",
            "--images",
            "/srv/a",
            "--gt",
            "/srv/g.json",
        ])
        .unwrap();
        let refs = dataset_refs(&a, &config);
        assert_eq!(refs[0], DatasetRef::Id("coco-cctv".into()));
        assert_eq!(refs[1].dir().unwrap().1, Some(PathBuf::from("/srv/g.json")));
        // An unknown dataset is a clear error before any runtime is loaded.
        let a = parse(&["x", "--dataset", "nope", "--model", "m.onnx"]).unwrap();
        let err = load_sets(&a, &config).unwrap_err();
        assert!(format!("{err:#}").contains("valid ids"), "{err:#}");
    }

    fn selection_of(opts: &[(&str, bool)]) -> crate::backend::Selection {
        let options: Vec<_> = opts
            .iter()
            .map(|&(s, runnable)| crate::backend::select::DeviceOption {
                spec: *spec::parse(s).unwrap().device().unwrap(),
                label: format!("label of {s}"),
                runnable,
                reason: (!runnable).then(|| "not runnable".to_string()),
                download: None,
            })
            .collect();
        crate::backend::Selection {
            auto: options
                .iter()
                .filter(|o| o.runnable)
                .map(|o| o.spec)
                .collect(),
            options,
            auto_measured: false,
        }
    }

    fn names(gpus: &[(spec::Device, String)]) -> Vec<String> {
        gpus.iter().map(|(d, _)| d.to_string()).collect()
    }

    /// The GTX 1070 Ti + UHD 630 machine: what a sweep would touch, and the refusal.
    #[test]
    fn a_sweep_over_several_gpus_needs_a_confirmation() {
        let sel = selection_of(&[
            ("ort:cuda:0", true),
            ("ort:tensorrt:0", false),
            ("openvino:gpu.0", true),
            ("ort:directml:1", true),
            ("openvino:cpu", true),
            ("ort:cpu", true),
        ]);
        let all = sweep_gpus(&sel, None);
        assert_eq!(
            names(&all),
            ["ort:cuda:0", "openvino:gpu.0", "ort:directml:1"],
            "runnable GPU-class options only"
        );
        let dev = |s: &str| *spec::parse(s).unwrap().device().unwrap();
        let one = [
            dev("openvino:cpu"),
            dev("ort:cuda:0"),
            dev("ort:tensorrt:0"),
        ];
        assert_eq!(names(&sweep_gpus(&sel, Some(&one))), ["ort:cuda:0"]);
        assert!(sweep_gpus(&sel, Some(&[])).is_empty());
        let msg = several_gpus_refusal(&all);
        assert!(msg.contains("3 GPU devices"), "{msg}");
        // The gate: one GPU device goes, two need the confirmation, which lets them through.
        assert!(gpu_gate(&all[..1], false).is_none(), "one GPU device");
        assert!(gpu_gate(&[], false).is_none());
        assert_eq!(
            gpu_gate(&all[..2], false),
            Some(several_gpus_refusal(&all[..2]))
        );
        assert!(gpu_gate(&all, true).is_none(), "confirmed");
        assert!(
            msg.contains("ort:cuda:0 (label of ort:cuda:0), openvino:gpu.0 (label of"),
            "{msg}"
        );
        assert!(msg.contains("has hung a machine"), "{msg}");
        assert!(msg.contains("--confirm-gpus") && msg.contains("benchmark.devices"));
        assert!(
            parse(&["x", "--apply", "--confirm-gpus"])
                .unwrap()
                .confirm_gpus
        );
        assert!(!parse(&["x", "--apply"]).unwrap().confirm_gpus);
    }

    /// A sweep's breadcrumb is the service's: noted before a device starts, held while the CLI
    /// lives (so the service never reports it), left behind when the process dies there.
    #[test]
    fn the_cli_trail_is_the_service_breadcrumb() {
        let dir = std::env::temp_dir().join(format!("bop-bcrumb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(breadcrumb::BREADCRUMB_FILE);
        let trail = Trail::new(&dir);
        trail.note("ort:cuda:0", "IPcam-general");
        assert!(file.exists());
        assert_eq!(
            breadcrumb::left_behind(&dir),
            None,
            "a live run is not interrupted"
        );
        drop(trail);
        assert!(!file.exists(), "a run that ends leaves nothing");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Interrupts from a channel (None: it closed), for [`interrupt_loop`].
    struct Fake(tokio::sync::mpsc::UnboundedReceiver<()>);

    impl Interrupt for Fake {
        async fn next(&mut self) -> Option<()> {
            self.0.recv().await
        }
    }

    /// Never an interrupt.
    struct Quiet;

    impl Interrupt for Quiet {
        async fn next(&mut self) -> Option<()> {
            std::future::pending().await
        }
    }

    /// Audit 2026-11 phase 1: the first interrupt cancels the run, a second quits; a run that
    /// ends stops the listening.
    #[tokio::test]
    async fn the_first_interrupt_cancels_and_the_second_quits() {
        // Two interrupts.
        let (sig_tx, sig_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        let (_done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let (cancel, quit) = (AtomicBool::new(false), AtomicBool::new(false));
        sig_tx.send(()).unwrap();
        sig_tx.send(()).unwrap();
        let mut fake = Fake(sig_rx);
        interrupt_loop(done_rx, &mut fake, &cancel, || {
            quit.store(true, Ordering::SeqCst);
        })
        .await;
        assert!(cancel.load(Ordering::SeqCst) && quit.load(Ordering::SeqCst));

        // One interrupt, then the run ends: cancelled, never quit.
        let (sig_tx, sig_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let (cancel, quit) = (AtomicBool::new(false), AtomicBool::new(false));
        sig_tx.send(()).unwrap();
        let ended = async {
            while !cancel.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
            done_tx.send(()).unwrap();
        };
        let mut fake = Fake(sig_rx);
        let listen = interrupt_loop(done_rx, &mut fake, &cancel, || {
            quit.store(true, Ordering::SeqCst);
        });
        tokio::join!(listen, ended);
        assert!(cancel.load(Ordering::SeqCst) && !quit.load(Ordering::SeqCst));
        drop(sig_tx);

        // No interrupt: the run's end stops it, nothing cancelled.
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let cancel = AtomicBool::new(false);
        done_tx.send(()).unwrap();
        interrupt_loop(done_rx, &mut Quiet, &cancel, || panic!("quit")).await;
        assert!(!cancel.load(Ordering::SeqCst));
        // Signals that cannot be listened to: nothing happens either.
        let (sig_tx, sig_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        drop(sig_tx);
        let (_keep, done_rx) = tokio::sync::oneshot::channel::<()>();
        interrupt_loop(done_rx, &mut Fake(sig_rx), &cancel, || panic!("quit")).await;
        assert!(!cancel.load(Ordering::SeqCst));
    }

    #[test]
    fn a_cancelled_run_says_what_was_saved() {
        let p = Path::new("/cfg/benchmark.json");
        let m = cancelled_message(2, Some(p));
        assert!(
            m.contains("2 model(s)") && m.contains("benchmark.json"),
            "{m}"
        );
        assert!(cancelled_message(0, Some(p)).contains("nothing was saved"));
        assert!(cancelled_message(3, None).contains("nothing was saved"));
    }

    #[test]
    fn float_and_count_flags_are_range_checked() {
        for bad in ["nan", "inf", "-0.1", "1.5", "x"] {
            let e = parse(&["x", &format!("--min-confidence={bad}")])
                .unwrap_err()
                .to_string();
            assert!(e.contains("0 to 1"), "{bad}: {e}");
            assert!(
                parse(&["x", &format!("--accuracy-weight={bad}")]).is_err(),
                "{bad}"
            );
            assert!(
                parse(&["x", "--threshold-search", &format!("--iou={bad}")]).is_err(),
                "{bad}"
            );
        }
        let a = parse(&["x", "--min-confidence", "0.4", "--accuracy-weight", "1"]).unwrap();
        assert_eq!(
            (a.min_confidence, a.accuracy_weight),
            (Some(0.4), Some(1.0))
        );
        assert!(parse(&["x", "--repeat", "0"]).is_err());
        assert!(parse(&["x", "--repeat", "1001"]).is_err());
        assert!(parse(&["x", "--repeat", "18446744073709551615"]).is_err());
        assert!(parse(&["x", "--warmup", "101"]).is_err());
        let a = parse(&["x", "--repeat", "1000", "--warmup", "0"]).unwrap();
        assert_eq!((a.repeat, a.warmup), (Some(1000), Some(0)));
    }

    /// RA-078: a CLI run over the service's cap is cut with a warning; `--no-image-cap` keeps all.
    #[test]
    fn cli_runs_are_capped_unless_told_not_to() {
        let set = |n: usize| {
            let mut s = ImageSet::sample();
            let img = s.images[0].clone();
            s.images = (0..n)
                .map(|i| {
                    let mut x = img.clone();
                    x.file = format!("{i}.jpg");
                    x
                })
                .collect();
            s
        };
        let cap = crate::benchmark::service::MAX_IMAGES_PER_RUN;
        let (mut sets, mut warnings) = (vec![set(cap + 500)], Vec::new());
        cap_run(&mut sets, &mut warnings, cap, false);
        assert!(sets[0].images.len() <= cap);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let (mut sets, mut warnings) = (vec![set(cap + 500)], Vec::new());
        cap_run(&mut sets, &mut warnings, cap, true);
        assert_eq!(sets[0].images.len(), cap + 500);
        assert!(warnings.is_empty());
        let a = parse(&["x", "--no-image-cap"]).unwrap();
        assert!(a.no_image_cap && !parse(&["x"]).unwrap().no_image_cap);
    }
}
