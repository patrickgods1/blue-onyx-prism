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
//! ```

use crate::backend::{CoreOptions, OrtOptions, Runtimes, libs, spec};
use crate::benchmark::grade::AccuracyMetric;
use crate::benchmark::images::{self, ImageSet};
use crate::benchmark::report::{HardwareSummary, RuntimeVersions};
use crate::benchmark::threshold::Objective;
use crate::benchmark::{
    Bench, BenchmarkResults, Comparison, Job, MATCH_IOU, ModelResult, Phase, SweepOptions, compare,
    config_models, configured_device, export, job_for_config, job_for_file, no_pseudo_gt_device,
    pick_reference, pseudo_ground_truth, pseudo_gt_device, results_path, sweep,
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
use std::sync::Mutex;

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
    if path.exists() {
        Config::load(path)
    } else {
        Ok(Config::default())
    }
}

/// One job, with its config entry when it came from the config file.
struct Planned {
    job: Job,
    entry: Option<ModelConfig>,
}

fn jobs(args: &BenchArgs, config: &Config, errors: &mut Vec<String>) -> Result<Vec<Planned>> {
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
        for m in config_models(config)? {
            match job_for_config(config, m, device_override.as_deref()) {
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
            p.job.confidence_threshold = Some(c);
        }
    }
    Ok(with_variants(out, args, config, errors))
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

/// Resolve the datasets, downloading missing built-in sets first when the config allows it.
fn load_sets(args: &BenchArgs, config: &Config) -> Result<(Vec<ImageSet>, Vec<String>)> {
    if let Some(p) = &args.image {
        return Ok((vec![ImageSet::single_file(p)?], Vec::new()));
    }
    let refs = dataset_refs(args, config);
    let max = args
        .max_images
        .unwrap_or(config.benchmark.max_images_per_dataset);
    let root = config.data_root();
    let mut r = images::resolve(&refs, &root, max)?;
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
            Ok(()) => r = images::resolve(&refs, &root, max)?,
            Err(e) => r
                .warnings
                .push(format!("downloading datasets failed: {e:#}")),
        }
    }
    Ok((r.sets, r.warnings))
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
    if args.threshold_search
        && let Some(ok) = stored_threshold_search(&args, &config, &config_path)?
    {
        return Ok(ok);
    }
    check_args(&args, &config_path)?;
    // Before the runtimes exist, so their thread pools inherit the nice value on Unix; held to
    // the end of the run.
    let _prio = crate::backend::priority::lower_for_benchmark(
        !args.no_low_priority && config.benchmark.low_priority,
        false,
    );
    let objective = args
        .threshold_objective
        .unwrap_or(config.benchmark.threshold_objective);
    let metric = args
        .accuracy_metric
        .unwrap_or(config.benchmark.accuracy_metric);
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
    let (mut sets, warnings) = load_sets(&args, &config)?;
    let (repeat, warmup, weights) = timing_settings(&args, &config, &sets);
    let cache_dir = cache_dir_arg(&args, &config);
    let runtimes = open_runtimes(
        &args,
        &config,
        &config_path,
        cache_dir.clone(),
        openvino_dir,
    )?;
    let info = runtimes.info();
    let versions = RuntimeVersions::of(&runtimes);
    let runtimes = Mutex::new(runtimes);
    let mut report = Report::new(info, errors, &sets, warnings);
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

    // `benchmark.devices` of the config limits the sweep, as on the web UI's Benchmark page,
    // and the pseudo-ground-truth device.
    let sweep_devices = crate::benchmark::service::parse_devices(&config.benchmark.devices)
        .context("benchmark.devices in the config")?;
    let ctx = RunCtx {
        args: &args,
        config: &config,
        config_path: &config_path,
        runtimes: &runtimes,
        sweep_devices: &sweep_devices,
        objective,
        metric,
        weights,
        variant_names: &variant_names,
    };

    // Pseudo ground truth for datasets without it (accuracy relative to a reference model).
    if sets.iter().any(|s| !s.annotated()) {
        let mut pb = Bench::new(&[], 0, 1, params);
        pb.cache_dir = cache_dir.as_deref();
        pb.progress = Some(&progress);
        ctx.pseudo_ground_truth(&pb, &jobs, &mut sets, &mut report);
    }

    let mut bench = Bench::new(&sets, warmup, repeat, params);
    bench.object_filter = &config.object_filter;
    bench.cache_dir = cache_dir.as_deref();
    bench.weights = weights;
    bench.threshold_objective = objective;
    bench.accuracy_metric = metric;

    let run_id = crate::benchmark::history::new_run_id();
    let mut swept = ctx.run_jobs(&mut bench, &progress, &jobs, all_devices, &mut report)?;
    for m in &mut swept {
        m.run_id = Some(run_id.clone());
    }

    if all_devices
        && !swept.is_empty()
        && let Some(ok) = ctx.finish_sweep(&sets, versions, &swept, &mut report)?
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
        } else if p.phase == Phase::Timing
            && p.total >= 50
            && p.done > 0
            && p.done.is_multiple_of(50)
        {
            eprintln!("    {}/{} requests", p.done, p.total);
        }
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
    weights: crate::benchmark::grade::Weights,
    /// Names of the variant jobs (`<model>+<variant>`): never applied to the config.
    variant_names: &'a std::collections::HashSet<String>,
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
            if jobs
                .iter()
                .any(|p| p.job.name.eq_ignore_ascii_case(&job.name))
            {
                report.warn(
                    args.json,
                    format!(
                        "{} is also the pseudo-ground-truth reference: its accuracy on the datasets \
                         without ground truth is 100% by construction",
                        job.name
                    ),
                );
            }
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
        all_devices: bool,
        report: &mut Report,
    ) -> Result<Vec<ModelResult>> {
        let mut swept: Vec<ModelResult> = Vec::new();
        for Planned { job, entry } in jobs {
            if all_devices {
                let result = self.sweep_job(bench, progress, job, entry.as_ref(), report)?;
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
        let result = sweep(bench, self.runtimes, job, &opts, &mut |_| {})?;
        bench.progress = None;
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
                        let mut c = Comparison::default();
                        for (a, b) in model_report.primary.per_image.iter().zip(&cpu.per_image) {
                            let a: Vec<_> = a
                                .preds
                                .iter()
                                .map(crate::benchmark::Detection::from)
                                .collect();
                            let b: Vec<_> = b
                                .preds
                                .iter()
                                .map(crate::benchmark::Detection::from)
                                .collect();
                            c.merge(&compare(&a, &b, MATCH_IOU));
                        }
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
        let merged = self.save_results(results, swept, report)?;
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
            report.applied_thresholds = apply_thresholds(&plain, self.config_path, args.json)?;
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
        report: &mut Report,
    ) -> Result<BenchmarkResults> {
        if self.args.no_save {
            return Ok(results);
        }
        let path = results_path(self.config_path);
        let (prev, warnings) = BenchmarkResults::load_for_merge(&path)?;
        for w in warnings {
            report.warn(self.args.json, w);
        }
        if let Some(w) = BenchmarkResults::merge_warning(prev.as_ref(), &results) {
            report.warn(self.args.json, w);
        }
        // The results before this save seed the history of models that have none yet.
        let seeds = prev
            .as_ref()
            .map(crate::benchmark::history::seed_entries)
            .unwrap_or_default();
        let mut merged = BenchmarkResults::merge(prev, results);
        // Models kept from earlier runs are graded like this run's.
        merged.regrade(self.metric, self.weights);
        merged.save(&path)?;
        for w in crate::benchmark::search::save_beside(&path, &merged, swept) {
            report.warn(self.args.json, w);
        }
        let names: Vec<String> = swept.iter().map(|m| m.model.clone()).collect();
        for w in crate::benchmark::history::record(&path, &seeds, &merged, &names) {
            report.warn(self.args.json, w);
        }
        if !self.args.json {
            println!("\nresults saved to {}", path.display());
        }
        report.results_file = Some(path.display().to_string());
        Ok(merged)
    }

    /// `--apply`: write the recommended devices of `swept` into the config file.
    fn apply_devices(&self, swept: &[ModelResult]) -> Result<Vec<DeviceChange>> {
        let config_path = self.config_path;
        let picks: Vec<(String, Option<String>)> = swept
            .iter()
            .filter_map(|m| m.recommended.clone().map(|d| (m.model.clone(), Some(d))))
            .collect();
        let mut cfg = Config::load(config_path)?;
        let changes = apply_model_devices(&mut cfg, &picks)?;
        if !changes.is_empty() {
            cfg.save(config_path)?;
        }
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

/// `--threshold-search`: search the stored predictions and print the results.
fn threshold_search_cli(
    args: &BenchArgs,
    config: &Config,
    results: &BenchmarkResults,
    preds: &crate::benchmark::search::StoredPreds,
) -> Result<bool> {
    let req = search_request(args, config);
    let current = |m: &str| {
        crate::benchmark::find_model(config, m).map(|c| {
            c.confidence_threshold
                .unwrap_or(config.confidence_threshold)
        })
    };
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
        objective: args
            .threshold_objective
            .unwrap_or(config.benchmark.threshold_objective),
        iou: args.iou.unwrap_or(0.5),
    }
}

/// `--apply-threshold`: write the best threshold of every device each config model of `swept`
/// ran on into its confidence profiles (`device_thresholds`, [`apply_device_thresholds`]); the
/// model's `confidence_threshold` is left alone. Prints one line per model and device.
fn apply_thresholds(
    swept: &[ModelResult],
    config_path: &Path,
    quiet: bool,
) -> Result<Vec<ProfileChange>> {
    let mut cfg = Config::load(config_path)?;
    // Results may name models that are not configured (--model files): skip those.
    let configured: Vec<&ModelResult> = swept
        .iter()
        .filter(|m| crate::benchmark::find_model(&cfg, &m.model).is_some())
        .collect();
    let entries: Vec<(String, String, Option<f32>)> = configured
        .iter()
        .filter_map(|m| Some((m, m.threshold.as_ref()?.device_apply_values())))
        .flat_map(|(m, v)| {
            v.into_iter()
                .map(move |(d, t)| (m.model.clone(), d, Some(t)))
        })
        .collect();
    let changes = apply_device_thresholds(&mut cfg, &entries)?;
    if !changes.is_empty() {
        cfg.save(config_path)?;
    }
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
        for m in &configured {
            let Some(a) = &m.threshold else {
                println!("  {}: unchanged (no threshold curve)", m.model);
                continue;
            };
            for d in &m.devices {
                let changed = changes.iter().find(|c| {
                    crate::registry::normalize_name(&c.model)
                        == crate::registry::normalize_name(&m.model)
                        && crate::config::same_device(&c.device, &d.device)
                });
                match (changed, a.apply_value_on(&d.device)) {
                    (Some(c), _) => println!("  {}", c.describe()),
                    (None, Some(t)) => {
                        println!("  {} on {}: {t:.2} (unchanged)", m.model, d.device)
                    }
                    (None, None) => println!(
                        "  {} on {}: unchanged ({})",
                        m.model,
                        d.device,
                        a.by_device
                            .iter()
                            .find(|p| crate::config::same_device(&p.device, &d.device))
                            .and_then(|p| p.best.as_ref())
                            .and_then(|b| b.note.clone())
                            .unwrap_or_else(|| "no usable threshold".into())
                    ),
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

    #[test]
    fn apply_thresholds_writes_the_config() {
        let dir = std::env::temp_dir().join(format!("bop-bthr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cfg.json");
        let cfg = Config {
            models: ["a", "b"]
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
        // Advice with one best per device: CPU 0.31, CoreML 0.27, GPU not applicable.
        let mk = |name: &str, t: Option<f32>| {
            let mut m = ModelResult::failed(name, "", String::new());
            m.threshold = t.map(|t| {
                let mut a = crate::benchmark::threshold::test_advice(t);
                let pick = |t: f32, met: bool| {
                    let mut p = a.best.clone().unwrap();
                    p.threshold = t;
                    p.met = met;
                    Some(p)
                };
                a.by_device = vec![
                    crate::benchmark::threshold::DevicePick {
                        device: "openvino:cpu".into(),
                        best: pick(t, true),
                    },
                    crate::benchmark::threshold::DevicePick {
                        device: "ort:coreml".into(),
                        best: pick(0.27, true),
                    },
                    crate::benchmark::threshold::DevicePick {
                        device: "openvino:gpu".into(),
                        best: pick(0.6, false),
                    },
                ];
                a
            });
            m
        };
        let swept = [mk("a", Some(0.31)), mk("b", None), mk("c.onnx", Some(0.2))];
        let changes = apply_thresholds(&swept, &path, true).unwrap();
        assert_eq!(
            changes
                .iter()
                .map(|c| (c.model.as_str(), c.device.as_str(), c.from, c.to))
                .collect::<Vec<_>>(),
            [
                ("a", "openvino:cpu", None, Some(0.31)),
                ("a", "ort:coreml", None, Some(0.27))
            ]
        );
        let saved = Config::load(&path).unwrap();
        assert_eq!(
            saved.models[0].device_thresholds,
            [
                ("openvino:cpu".to_string(), 0.31),
                ("ort:coreml".to_string(), 0.27)
            ]
            .into()
        );
        // The model thresholds (devices without a profile) are unchanged.
        assert_eq!(saved.models[0].confidence_threshold, None);
        assert_eq!(saved.models[1].confidence_threshold, None);
        assert!(saved.models[1].device_thresholds.is_empty());
        // Again: nothing to change (and printing works).
        assert!(apply_thresholds(&swept, &path, false).unwrap().is_empty());
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
}
