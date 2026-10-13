//! Command line arguments of the benchmark (`BenchArgs`) and their value parsers.
//! Every range check lives in the `parse_*` helpers so clap rejects bad values before any
//! model is loaded; `check_args` in the parent module handles combinations of flags.

use crate::backend::spec;
use crate::benchmark::grade::AccuracyMetric;
use crate::benchmark::threshold::Objective;
use crate::model::ModelFamilyKind;
use anyhow::Result;
use std::path::PathBuf;

// The field docs are clap help text: placeholders like `<model>` and `[:N]` stay verbatim.
#[allow(rustdoc::invalid_html_tags, rustdoc::broken_intra_doc_links)]
#[derive(Debug, Clone, clap::Args)]
pub struct BenchArgs {
    /// Model file (.xml or .onnx), repeatable. Default: every enabled model in the config file.
    #[arg(long)]
    pub model: Vec<PathBuf>,
    /// Model family for --model (default: auto-detect).
    #[arg(long, value_parser = crate::cli::family_parser())]
    pub family: Option<ModelFamilyKind>,
    /// Class names YAML for --model (default: <model>.yaml beside the model, else COCO-80).
    #[arg(long)]
    pub classes: Option<PathBuf>,
    /// Device spec, one of: auto, openvino:gpu[.N], openvino:cpu, openvino:npu, ort:cuda[:N],
    /// ort:tensorrt[:N], ort:directml[:N], ort:coreml, ort:cpu (legacy GPU, GPU.N, CPU, NPU mean
    /// OpenVINO). A GPU request falls back to CPU like the server does (without the warm-up check).
    /// Default: auto for --model, the configured device for config models.
    #[arg(long, value_parser = parse_device_arg)]
    pub device: Option<String>,
    /// Run every runnable device option (see `list-devices`; `benchmark.devices` of the config
    /// limits them) for each model, grade accuracy and
    /// speed, compare detections with the CPU reference, print the recommended device, and save
    /// the results to `benchmark.json` next to the config file (web UI Benchmark page). Cannot
    /// be combined with --device, --force-cpu or --compare-cpu.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub all_devices: bool,
    /// Implies --all-devices: write each benchmarked config model's recommended device into its
    /// `device` in the config file and print what changed (restart the server to apply).
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu", "model"])]
    pub apply: bool,
    /// Implies --all-devices: write each benchmarked config model's best confidence threshold
    /// (see --threshold-objective) into its `confidence_threshold` in the config file and print
    /// what changed. Independent of --apply (devices); both may be given.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu", "model", "min_confidence"])]
    pub apply_threshold: bool,
    /// What the best confidence threshold optimizes: f1 (balance), f2 (favor recall: fewer
    /// missed objects) or precision:<p> (highest recall with precision >= p, e.g.
    /// precision:0.9 for fewer false alerts) or recall:<r> (highest precision with recall >= r),
    /// youden (frame-level Youden's J: alert rate on frames with objects minus false-alert rate
    /// on frames without) or fpr:<x> (highest frame alert rate with at most x false alerts per
    /// frame without objects, e.g. fpr:0.05). Default: benchmark.threshold_objective (f1).
    #[arg(long, value_parser = parse_objective_arg)]
    pub threshold_objective: Option<Objective>,
    /// What the accuracy grade, the ranking and the device recommendation use: ap50 (box-level
    /// AP@0.5 with small-object recall) or roc_auc (macro frame-level ROC AUC: how well a
    /// frame's top confidence separates frames with objects from frames without). Default:
    /// benchmark.accuracy_metric (ap50).
    #[arg(long, value_parser = parse_metric_arg)]
    pub accuracy_metric: Option<AccuracyMetric>,
    /// Search the best confidence threshold over the predictions stored by the last benchmark
    /// (`benchmark-preds.json` beside the config; no inference), filtered by --dataset, --tag,
    /// --class, --search-model, --search-device and --iou, for --threshold-objective (also
    /// `recall:<r>`). Runs the benchmark first when nothing is stored.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub threshold_search: bool,
    /// --threshold-search: only images with this tag (repeatable: all must match), e.g. night.
    #[arg(long, requires = "threshold_search")]
    pub tag: Vec<String>,
    /// --threshold-search: only these scored classes (repeatable), e.g. person.
    #[arg(long = "class", requires = "threshold_search")]
    pub class: Vec<String>,
    /// --threshold-search: only these models (repeatable; default every stored model).
    #[arg(long, requires = "threshold_search")]
    pub search_model: Vec<String>,
    /// --threshold-search: the device whose predictions to use (default each model's
    /// recommended device).
    #[arg(long, requires = "threshold_search", value_parser = parse_device_arg)]
    pub search_device: Option<String>,
    /// --threshold-search: matching IoU (default 0.5).
    #[arg(long, requires = "threshold_search", value_parser = parse_unit_f32)]
    pub iou: Option<f32>,
    /// Implies --all-devices: also write a standalone report of the saved results (`.html` or
    /// `.md`).
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub report: Option<PathBuf>,
    /// Do not write `benchmark.json` (--all-devices).
    #[arg(long)]
    pub no_save: bool,
    /// Force CPU inference.
    #[arg(long)]
    pub force_cpu: bool,
    /// One JPEG/PNG to run on (overrides the datasets).
    #[arg(long, conflicts_with_all = ["dataset", "images"])]
    pub image: Option<PathBuf>,
    /// Dataset, repeatable: a built-in set id (see --list-datasets), `sample` (the embedded
    /// image) or `dir:<path>`. Default: `benchmark.datasets` of the config.
    #[arg(long)]
    pub dataset: Vec<String>,
    /// A directory of images (same as `--dataset dir:<path>`), with optional COCO JSON / YOLO
    /// ground truth (see --gt) and manifest.json tags.
    #[arg(long)]
    pub images: Option<PathBuf>,
    /// Ground truth for --images: a COCO instances JSON file or a YOLO labels directory
    /// (default: found in the directory).
    #[arg(long, requires = "images")]
    pub gt: Option<PathBuf>,
    /// Images per dataset, evenly spread (0 = all). Default: benchmark.max_images_per_dataset.
    #[arg(long)]
    pub max_images: Option<usize>,
    /// List the datasets (built-in sets, their download state, the configured selection) and exit.
    #[arg(long)]
    pub list_datasets: bool,
    /// Model whose detections are the pseudo ground truth of datasets without ground truth
    /// (default: benchmark.reference_model, else the most accurate installed model).
    #[arg(long)]
    pub reference_model: Option<String>,
    /// Share of accuracy in the overall grade, 0..1 (default: benchmark.weights, 0.6).
    #[arg(long, value_parser = parse_unit_f64)]
    pub accuracy_weight: Option<f64>,
    /// Timed runs per image. Default: 100 for a single image, else benchmark.repeat_per_image.
    #[arg(long, value_parser = parse_repeat_arg)]
    pub repeat: Option<usize>,
    /// Untimed warm-up iterations. Default: benchmark.warmup.
    #[arg(long, value_parser = parse_warmup_arg)]
    pub warmup: Option<usize>,
    /// Also run on CPU and compare timings and detections.
    #[arg(long)]
    pub compare_cpu: bool,
    /// Compiled-model cache directory ("" disables). Default: the configured cache dir.
    #[arg(long)]
    pub cache_dir: Option<String>,
    /// CPU inference threads (0 = all physical cores but one on machines with at least 4).
    /// Default: the configured intra_threads.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Keep the normal process priority (default: below normal so the desktop stays
    /// responsive; also off with benchmark.low_priority = false).
    #[arg(long)]
    pub no_low_priority: bool,
    /// Confidence threshold. Default: the configured confidence_threshold.
    #[arg(long, value_parser = parse_unit_f32)]
    pub min_confidence: Option<f32>,
    /// Config file for defaults (default: <exe_dir>/blue_onyx_prism_config.json if present).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Print machine-readable JSON instead of tables.
    #[arg(long)]
    pub json: bool,
    /// Show info-level logs (on stderr).
    #[arg(long, short)]
    pub verbose: bool,
}

fn parse_objective_arg(s: &str) -> Result<Objective, String> {
    s.parse()
}

fn parse_metric_arg(s: &str) -> Result<AccuracyMetric, String> {
    s.parse()
}

/// A finite number in 0..=1 (rejects `nan` and `inf`, which Rust's float parser accepts and
/// which would be saved as `null` in benchmark.json).
fn parse_unit_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a number (expected 0 to 1)"))?;
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!(
            "'{s}' is out of range: expected a number from 0 to 1"
        ))
    }
}

/// [`parse_unit_f64`] as f32.
fn parse_unit_f32(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a number (expected 0 to 1)"))?;
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!(
            "'{s}' is out of range: expected a number from 0 to 1"
        ))
    }
}

/// An integer in `lo..=hi`.
fn parse_bounded(s: &str, lo: usize, hi: usize) -> Result<usize, String> {
    let v: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a whole number (expected {lo} to {hi})"))?;
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(format!("'{s}' is out of range: expected {lo} to {hi}"))
    }
}

/// `--repeat`: 1..=[`crate::benchmark::service::MAX_REPEAT`] (the web UI's bound).
fn parse_repeat_arg(s: &str) -> Result<usize, String> {
    parse_bounded(s, 1, crate::benchmark::service::MAX_REPEAT)
}

/// `--warmup`: 0..=[`crate::benchmark::service::MAX_WARMUP`] (the web UI's bound).
fn parse_warmup_arg(s: &str) -> Result<usize, String> {
    parse_bounded(s, 0, crate::benchmark::service::MAX_WARMUP)
}

/// Default timed runs for a single image (the pre-dataset default).
pub(crate) const SINGLE_IMAGE_REPEAT: usize = 100;

/// Validate `--device` with the spec parser, keeping the string as typed.
fn parse_device_arg(s: &str) -> Result<String, String> {
    spec::parse(s)
        .map(|_| s.trim().to_string())
        .map_err(|e| e.to_string())
}
