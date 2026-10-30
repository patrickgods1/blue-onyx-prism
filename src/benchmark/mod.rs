//! Benchmark core shared by the `blue-onyx-prism-benchmark` binary, the `benchmark` subcommand
//! and the web UI's Benchmark page.
//!
//! A run compiles a model on a device, warms it up, then sends every image of the chosen
//! datasets ([`images`]) through the production per-request stages (decode -> preprocess ->
//! infer -> postprocess, the code `worker.rs` runs), one image at a time on one thread with one
//! infer request, `repeat` timed times per image. The service is faster than these per-image
//! latencies imply: it decodes and preprocesses on a pool while inferring and can keep several
//! inferences in flight (`crate::profile`), so measure its throughput with
//! `test-blue-onyx-prism --repeat 200 --parallel 6`. Timing
//! uses the model's configured confidence threshold; the same inference outputs are also
//! post-processed (untimed) at [`EVAL_CONFIDENCE`] for AP. Accuracy ([`metrics`]) is measured
//! against the dataset's ground truth, or against pseudo ground truth from a reference model
//! (`pseudo_ground_truth`) for datasets without it; [`grade`] turns accuracy and speed into
//! letter grades. [`sweep`] runs one model on every chosen device, checks each device's
//! detections against a CPU reference device and recommends a device ([`report::recommend`]).
//! [`report`] holds the persisted results (`benchmark.json` next to the config file),
//! [`history`] the run history beside it (`benchmark-history.json`: what changed between runs),
//! [`service`] the background run behind `/v1/benchmark`, [`cli`] the command line, and
//! [`export`] the standalone HTML / Markdown report. [`threshold`] sweeps the confidence
//! threshold over the same low-threshold predictions (no extra inference) and picks the best
//! threshold per model.
//!
//! Runtimes are shared through a `Mutex<Runtimes>` that is held only while compiling, so the
//! server can benchmark through the registry's own runtimes (one OpenVINO `Core`, the ONNX
//! Runtime flavor the process loaded) while the workers keep serving.

pub mod cli;
pub mod current;
pub mod export;
pub mod forget;
pub mod grade;
pub mod history;
pub mod images;
pub mod metrics;
pub mod report;
pub mod rescore;
pub mod roc;
pub mod search;
pub mod service;
pub mod threshold;
pub mod variant;

mod run;
mod scoring;
mod sweep;

pub use report::{
    Agreement, BenchmarkResults, DeviceResult, ModelResult, Recommendation, Verdict, auto_orders,
    recommend, results_path, speed_order,
};

pub use run::{Bench, Cancelled, Phase, Progress, is_cancelled};
pub(crate) use scoring::{
    REFERENCE_DEVICES, no_pseudo_gt_device, pseudo_ground_truth, pseudo_gt_device,
};
pub use sweep::{SweepOptions, sweep};
pub use variant::PipelineVariant;

use crate::api::Prediction;
use crate::backend::spec::{self, Device, Runtime, Target};
use crate::backend::{Backend, Candidate, LoadRequest, Runtimes, Selection};
use crate::config::{Config, ModelConfig};
use crate::model::classes::resolve_class_names;
use crate::model::preprocess::Preprocessor;
use crate::model::{ModelFamilyKind, PostParams};
use crate::sync::LockExt;
use anyhow::{Context, Result, bail};
use grade::{Grades, Weights};
use images::{GroundTruth, ImageSet};
use metrics::{ClassMap, Counts, EvalImage, GtBox, PredBox, Summary};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Default benchmark image, embedded so the benchmark works without any file.
pub const DEFAULT_IMAGE: &[u8] = include_bytes!("../../tests/data/dog_bike_car.jpg");
pub(crate) const DEFAULT_IMAGE_NAME: &str = "dog_bike_car.jpg (embedded)";
/// Minimum IoU for a detection and a reference detection of the same label to count as the same
/// object (device agreement).
pub(crate) const MATCH_IOU: f32 = 0.5;
/// Confidence threshold of the (untimed) post-processing used for AP.
pub const EVAL_CONFIDENCE: f32 = 0.05;
/// Pseudo ground truth: reference detections at or above this confidence are objects...
pub(crate) const PSEUDO_GT_CONFIDENCE: f32 = 0.5;
/// ...and those between this and [`PSEUDO_GT_CONFIDENCE`] are `ignore` regions (borderline
/// objects that neither count as misses nor as false positives).
pub(crate) const PSEUDO_IGNORE_CONFIDENCE: f32 = 0.3;
/// Predictions kept per image in the results (drill-down), most confident first.
pub(crate) const MAX_STORED_PREDS: usize = 100;

/// min / mean / p50 / p95 / max in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Stats {
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Stats {
    /// Nearest-rank percentiles over `samples` (ms). All zeros for an empty slice.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        Self {
            min: s.first().copied().unwrap_or(0.0),
            mean: s.iter().sum::<f64>() / s.len() as f64,
            p50: percentile(&s, 50.0),
            p95: percentile(&s, 95.0),
            max: s.last().copied().unwrap_or(0.0),
        }
    }
}

/// Nearest-rank percentile of an ascending slice: the smallest value with at least `p`% of the
/// samples at or below it. 0.0 for an empty slice (like `Stats::default`); a non-finite `p` is
/// treated as 0 (the minimum).
pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let raw = ((p / 100.0) * n as f64).ceil();
    // `as usize` saturates (NaN -> 0); the clamp keeps the rank in 1..=n.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to 0..=n before the cast"
    )]
    let rank = if raw.is_finite() {
        raw.clamp(0.0, n as f64) as usize
    } else {
        0
    };
    sorted.get(rank.clamp(1, n) - 1).copied().unwrap_or(0.0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StageStats {
    pub decode: Stats,
    pub preprocess: Stats,
    pub infer: Stats,
    pub postprocess: Stats,
    /// The whole request (sum of the four stages).
    pub total: Stats,
}

/// One detection (pixel box), as reported to Blue Iris.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub x_min: usize,
    pub y_min: usize,
    pub x_max: usize,
    pub y_max: usize,
}

impl From<&Prediction> for Detection {
    fn from(p: &Prediction) -> Self {
        Self {
            label: p.label.clone(),
            confidence: p.confidence,
            x_min: p.x_min,
            y_min: p.y_min,
            x_max: p.x_max,
            y_max: p.y_max,
        }
    }
}

impl From<&ImagePred> for Detection {
    fn from(p: &ImagePred) -> Self {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to >= 0 (NaN -> 0 via f32::max); `as` saturates large values"
        )]
        let c = |v: f32| v.max(0.0).round() as usize;
        Self {
            label: p.label.clone(),
            confidence: p.confidence,
            x_min: c(p.bbox[0]),
            y_min: c(p.bbox[1]),
            x_max: c(p.bbox[2]),
            y_max: c(p.bbox[3]),
        }
    }
}

/// Speed of the images of one resolution bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketSpeed {
    /// `sd`, `hd`, `fhd`, `4mp+`.
    pub bucket: String,
    pub images: usize,
    /// Decode + preprocess p50.
    pub pre_p50: f64,
    pub infer_p50: f64,
    pub post_p50: f64,
    pub total_p50: f64,
    pub total_p95: f64,
}

/// A stored prediction of one image (at the model's confidence threshold).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagePred {
    pub label: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    /// True / false positive against the (pseudo) ground truth; None = not scored (class not
    /// scored, ignored region, or no ground truth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<bool>,
}

/// One image of a device run (drill-down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRun {
    /// Dataset id.
    pub set: String,
    pub file: String,
    /// p50 over the image's timed runs.
    pub total_ms: f64,
    pub infer_ms: f64,
    pub preds: Vec<ImagePred>,
    /// Per ground-truth object of the image (dataset order): matched; None = not scored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gt_matched: Vec<Option<bool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counts: Option<Counts>,
}

/// Metrics of a subset of images (one dataset or one tag).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breakdown {
    /// Dataset id or tag.
    pub key: String,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    /// Against pseudo ground truth.
    pub relative: bool,
    /// Frame-level ROC AUC (macro over classes, see [`roc`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc_auc: Option<f64>,
}

impl Breakdown {
    fn of(key: &str, s: &Summary, relative: bool, roc_auc: Option<f64>) -> Self {
        Self {
            key: key.to_string(),
            images: s.images,
            gt: s.counts.tp + s.counts.fn_,
            precision: s.precision,
            recall: s.recall,
            f1: s.f1,
            ap50: s.ap50,
            ap50_95: s.ap50_95,
            relative,
            roc_auc,
        }
    }
}

/// Accuracy of one model on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccuracyReport {
    /// "ground truth" or "relative to `<reference>`, not ground truth".
    #[serde(default)]
    pub ground_truth: String,
    #[serde(default)]
    pub relative: bool,
    /// Scored classes in the model's label space.
    #[serde(default)]
    pub classes: Vec<String>,
    /// Confidence threshold of precision / recall / F1.
    pub threshold: f32,
    pub overall: Summary,
    #[serde(default)]
    pub by_dataset: Vec<Breakdown>,
    #[serde(default)]
    pub by_tag: Vec<Breakdown>,
    /// AP over every class of the model (not only the CCTV ones), same images; None when the
    /// model has no other classes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_classes: Option<AllClasses>,
    /// Precision / recall / F1 / F2 over confidence thresholds (results of older versions
    /// have none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep: Option<threshold::ThresholdSweep>,
    /// Frame-level ROC of the overall images (per class AUC, macro and micro; no curve: the
    /// sweep has it). Results of older versions have none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_roc: Option<roc::FrameRoc>,
}

impl AccuracyReport {
    /// The accuracy score `metric` grades: AP@0.5 blended with small-object recall
    /// (`grade::accuracy_score`), or the macro frame ROC AUC.
    pub fn score(&self, metric: grade::AccuracyMetric) -> Option<f64> {
        match metric {
            grade::AccuracyMetric::Ap50 => {
                let small = self
                    .overall
                    .by_size
                    .iter()
                    .find(|b| b.bucket == "small" && b.gt > 0)
                    .and_then(|b| b.recall);
                self.overall.ap50.map(|ap| grade::accuracy_score(ap, small))
            }
            grade::AccuracyMetric::RocAuc => self.roc_auc(),
        }
    }

    /// Macro frame ROC AUC.
    pub fn roc_auc(&self) -> Option<f64> {
        self.frame_roc.as_ref().and_then(|r| r.auc)
    }
}

/// AP over all of a model's classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllClasses {
    /// Classes with ground truth in the images.
    pub classes: usize,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
}

/// The scored classes a dataset annotates: every class of `map`, or those of the set's
/// `scored_labels` (`allowed`, from [`eval_input`]). Frame samples exist only for these.
pub(crate) fn frame_classes(map: &ClassMap, allowed: Option<&[String]>) -> Vec<String> {
    map.classes
        .iter()
        .filter(|c| allowed.is_none_or(|a| a.contains(c)))
        .cloned()
        .collect()
}

/// One image in a class map's label space: ground truth per original object (None = not
/// scored) and the evaluation input. Applies the set's `scored_labels`. A wildlife box the
/// model cannot score (a raccoon for a COCO model, [`metrics::WILDLIFE`]) adds ignore regions
/// for the model's animal classes after the mapped boxes, so a `cat` on a raccoon is neither a
/// hit nor a false positive (and the frame is excluded for those classes).
pub(crate) fn eval_input(
    map: &ClassMap,
    scored_labels: Option<&[String]>,
    gt: &[GtBox],
    eval_preds: &[PredBox],
) -> (Vec<Option<GtBox>>, EvalImage, Option<Vec<String>>) {
    let allowed = scored_labels.map(|l| map.restrict(l));
    let ok = |c: &str| allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == c));
    let mapped: Vec<Option<GtBox>> = gt
        .iter()
        .map(|g| {
            map.gt_class(&g.label).filter(|c| ok(c)).map(|c| GtBox {
                label: c.to_string(),
                ..g.clone()
            })
        })
        .collect();
    let mut gt_eval: Vec<GtBox> = mapped.iter().flatten().cloned().collect();
    for g in gt {
        for c in map.wildlife_ignore(&g.label) {
            if ok(c) {
                gt_eval.push(GtBox {
                    label: c.to_string(),
                    bbox: g.bbox,
                    ignore: true,
                    crowd: false,
                });
            }
        }
    }
    let eval = EvalImage {
        gt: gt_eval,
        preds: map
            .map_preds(eval_preds)
            .into_iter()
            .filter(|p| ok(&p.label))
            .collect(),
    };
    (mapped, eval, allowed)
}

/// Result of running one model on one device over the datasets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub model: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub requested_device: String,
    /// Runtime device ("CPU", "GPU.0" for OpenVINO; "CoreML", ... for ONNX Runtime).
    pub device: String,
    #[serde(default)]
    pub device_name: String,
    /// Canonical spec of the device that ran ("openvino:cpu", "ort:coreml").
    #[serde(default)]
    pub spec: String,
    #[serde(default)]
    pub fell_back: bool,
    #[serde(default)]
    pub input: String,
    /// Dataset ids (comma separated) or the image file.
    #[serde(default)]
    pub image: String,
    /// "WxH" of a single image, else "N images".
    #[serde(default)]
    pub image_size: String,
    #[serde(default)]
    pub images: usize,
    /// read + reshape + compile + request creation wall time.
    #[serde(default)]
    pub compile_ms: f64,
    /// "disabled", "hit" (no new cache file written), "miss" (a blob was written) or "unknown".
    #[serde(default)]
    pub cache: String,
    #[serde(default)]
    pub warmup_iterations: usize,
    #[serde(default)]
    pub warmup_ms: f64,
    /// Timed runs per image.
    #[serde(default)]
    pub repeat: usize,
    #[serde(default)]
    pub stages_ms: StageStats,
    /// Sequential requests per second over the timed loop (1 concurrent client).
    #[serde(default)]
    pub throughput_fps: f64,
    /// Detections of the first image.
    #[serde(default)]
    pub detections: Vec<Detection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub by_resolution: Vec<BucketSpeed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyReport>,
    /// Why there is no accuracy (no ground truth, no scored classes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy_note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grades: Option<Grades>,
    /// Detections compared with the CPU reference device (device sweeps only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<Agreement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_image: Vec<ImageRun>,
    /// Per image (same order as `per_image`): the predictions at the evaluation threshold in
    /// a scored class (labelled with it), in post-processing order. Not in `benchmark.json`: persisted in
    /// `search::PREDS_FILE` for the threshold search.
    #[serde(skip)]
    pub eval_preds: Vec<Vec<PredBox>>,
}

/// Detections of one run matched against another's (`--compare-cpu`, agreement).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Comparison {
    pub matched: usize,
    pub only_primary: usize,
    pub only_cpu: usize,
    pub max_confidence_diff: f32,
    pub mean_confidence_diff: f32,
}

impl Comparison {
    /// Sum of per-image comparisons (mean confidence difference weighted by matches).
    pub fn merge(&mut self, o: &Comparison) {
        let total = self.matched + o.matched;
        if total > 0 {
            self.mean_confidence_diff = (self.mean_confidence_diff * self.matched as f32
                + o.mean_confidence_diff * o.matched as f32)
                / total as f32;
        }
        self.matched = total;
        self.only_primary += o.only_primary;
        self.only_cpu += o.only_cpu;
        self.max_confidence_diff = self.max_confidence_diff.max(o.max_confidence_diff);
    }
}

/// Intersection over union of two pixel boxes.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the f64 ratio is in 0..=1; the result is an f32 score"
)]
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    // f64 from the differences: boxes parsed from a hand-edited benchmark.json can sit near
    // `usize::MAX`, where a usize or f32 area product would overflow or lose everything.
    let ix = (a.x_max.min(b.x_max) as f64 - a.x_min.max(b.x_min) as f64).max(0.0);
    let iy = (a.y_max.min(b.y_max) as f64 - a.y_min.max(b.y_min) as f64).max(0.0);
    let inter = ix * iy;
    let area = |d: &Detection| {
        d.x_max.saturating_sub(d.x_min) as f64 * d.y_max.saturating_sub(d.y_min) as f64
    };
    let union = area(a) + area(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union) as f32
    }
}

/// Greedy one-to-one matching of same-label detections by descending IoU (>= `min_iou`);
/// reports confidence differences of matched pairs.
pub fn compare(primary: &[Detection], cpu: &[Detection], min_iou: f32) -> Comparison {
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (i, a) in primary.iter().enumerate() {
        for (j, b) in cpu.iter().enumerate() {
            if a.label == b.label {
                let v = iou(a, b);
                if v >= min_iou {
                    pairs.push((v, i, j));
                }
            }
        }
    }
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut used_a = vec![false; primary.len()];
    let mut used_b = vec![false; cpu.len()];
    let mut diffs = Vec::new();
    #[allow(
        clippy::indexing_slicing,
        reason = "i < primary.len() and j < cpu.len(): both come from enumerate() above"
    )]
    for (_, i, j) in pairs {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            diffs.push((primary[i].confidence - cpu[j].confidence).abs());
        }
    }
    let matched = diffs.len();
    Comparison {
        matched,
        only_primary: primary.len() - matched,
        only_cpu: cpu.len() - matched,
        max_confidence_diff: diffs.iter().copied().fold(0.0, f32::max),
        mean_confidence_diff: if matched == 0 {
            0.0
        } else {
            diffs.iter().sum::<f32>() / matched as f32
        },
    }
}

/// One model to benchmark with its resolved settings.
#[derive(Debug, Clone)]
pub struct Job {
    pub name: String,
    pub path: PathBuf,
    pub family: ModelFamilyKind,
    pub classes: Vec<String>,
    /// Device spec for single-device runs ([`Bench::run`]); sweeps ignore it.
    pub device: String,
    pub gpu_precision: Option<String>,
    /// Per-model confidence threshold (else the bench's).
    pub confidence_threshold: Option<f32>,
    /// Per-model label filter (else the bench's).
    pub object_filter: Option<Vec<String>>,
    /// Enhancement and filter rules on top of the plain pipeline ([`variant`]); `name` is then
    /// `<model>+<variant>`.
    pub variant: Option<PipelineVariant>,
}

/// Models to benchmark from a config: every enabled entry, in order.
pub fn config_models(config: &Config) -> Result<Vec<&ModelConfig>> {
    if config.models.is_empty() {
        bail!("no --model given and no models in the config file");
    }
    let models: Vec<&ModelConfig> = config.enabled_models().collect();
    if models.is_empty() {
        bail!("no --model given and every model in the config file is disabled");
    }
    Ok(models)
}

/// The config entry named `name` (matched like `/v1/vision/custom/{model}`).
pub fn find_model<'a>(config: &'a Config, name: &str) -> Option<&'a ModelConfig> {
    let key = crate::registry::normalize_name(name);
    config
        .models
        .iter()
        .find(|m| crate::registry::normalize_name(&m.effective_name()) == key)
}

/// A job for a model file given on the command line.
pub(crate) fn job_for_file(
    path: &Path,
    family: Option<ModelFamilyKind>,
    classes: Option<&Path>,
    device: Option<&str>,
) -> Result<Job> {
    Ok(Job {
        name: ModelConfig {
            path: path.to_path_buf(),
            ..Default::default()
        }
        .effective_name(),
        path: path.to_path_buf(),
        family: family.unwrap_or(ModelFamilyKind::Auto),
        classes: resolve_class_names(path, classes)?,
        device: device.unwrap_or("auto").to_string(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
        variant: None,
    })
}

/// A job for a config entry: paths resolved like the registry does, device from the config
/// unless `device_override` is given.
pub(crate) fn job_for_config(
    config: &Config,
    m: &ModelConfig,
    device_override: Option<&str>,
) -> Result<Job> {
    let path = config.data_path(&m.path);
    if !path.is_file() {
        bail!("model file not found: {}", path.display());
    }
    let classes_path = m.classes.as_deref().map(|c| config.data_path(c));
    let classes = resolve_class_names(&path, classes_path.as_deref())?;
    let device = match device_override {
        Some(d) => d.to_string(),
        None => config.device_spec_for(m)?.to_string(),
    };
    Ok(Job {
        name: m.effective_name(),
        family: m.family,
        classes,
        device,
        gpu_precision: m.gpu_precision.clone(),
        confidence_threshold: m.confidence_threshold,
        object_filter: m.object_filter.clone(),
        path,
        variant: None,
    })
}

/// The device a config model runs on (or would, after a restart), as a canonical spec:
/// `force_cpu` -> the best CPU option, `auto` -> the first option of the ranking, else the
/// configured device. None when the device does not parse or nothing is runnable for `auto`.
pub fn configured_device(config: &Config, m: &ModelConfig, sel: &Selection) -> Option<String> {
    if config.force_cpu {
        return Some(crate::backend::best_cpu(sel).to_string());
    }
    match config.device_spec_for(m).ok()? {
        spec::DeviceSpec::Auto => sel.auto.first().map(|d| d.to_string()),
        spec::DeviceSpec::Device(d) => Some(d.to_string()),
    }
}

/// Reference models for pseudo ground truth, most accurate first (matched as name prefixes,
/// case-insensitive).
pub(crate) const REFERENCE_PREFERENCE: [&str; 16] = [
    "rt-detrv2-x",
    "dfine-x",
    "rfdetr-large",
    "rt-detrv2-l",
    "dfine-l",
    "rfdetr-base",
    "rt-detrv2-m",
    "dfine-m",
    "yolo26x",
    "yolo26l",
    "rt-detrv2-ms",
    "rt-detrv2-s",
    "dfine-s",
    "yolo26m",
    "yolo26s",
    "yolo26n",
];

/// The pseudo-ground-truth reference: `explicit` when given (must be a configured model whose
/// file exists), else the most accurate present config model with COCO-style labels
/// ([`REFERENCE_PREFERENCE`]), else the largest present one with CCTV classes.
pub(crate) fn pick_reference<'a>(
    config: &'a Config,
    explicit: Option<&str>,
) -> Result<&'a ModelConfig> {
    let present = |m: &&ModelConfig| config.data_path(&m.path).is_file();
    if let Some(name) = explicit.map(str::trim).filter(|n| !n.is_empty()) {
        let m = find_model(config, name)
            .with_context(|| format!("reference model '{name}' is not a configured model"))?;
        if !present(&m) {
            bail!(
                "reference model '{name}': file {} not found",
                config.data_path(&m.path).display()
            );
        }
        return Ok(m);
    }
    let candidates: Vec<&ModelConfig> = config.models.iter().filter(present).collect();
    let rank = |m: &ModelConfig| {
        let n = m.effective_name().to_ascii_lowercase();
        REFERENCE_PREFERENCE
            .iter()
            .position(|p| n.starts_with(p))
            .unwrap_or(usize::MAX)
    };
    if let Some(m) = candidates
        .iter()
        .filter(|m| rank(m) != usize::MAX)
        .min_by_key(|m| rank(m))
    {
        return Ok(m);
    }
    // Unknown names: the largest file among models with many CCTV classes.
    candidates
        .into_iter()
        .filter(|m| {
            let path = config.data_path(&m.path);
            let classes = m.classes.as_deref().map(|c| config.data_path(c));
            resolve_class_names(&path, classes.as_deref())
                .map(|c| ClassMap::for_model(&c).classes.len() >= 6)
                .unwrap_or(false)
        })
        .max_by_key(|m| {
            std::fs::metadata(config.data_path(&m.path))
                .map(|md| md.len())
                .unwrap_or(0)
        })
        .context(
            "no reference model for pseudo ground truth: install a COCO model such as \
             rt-detrv2-x or dfine-s, or set benchmark.reference_model",
        )
}

#[cfg(test)]
mod tests {
    use super::scoring::reference_device;
    use super::*;

    fn det(label: &str, conf: f32, b: [usize; 4]) -> Detection {
        Detection {
            label: label.into(),
            confidence: conf,
            x_min: b[0],
            y_min: b[1],
            x_max: b[2],
            y_max: b[3],
        }
    }

    #[test]
    fn stats_helpers_handle_empty_and_nan() {
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[1.0, 2.0], f64::NAN), 1.0);
        assert_eq!(percentile(&[1.0, 2.0], 1e9), 2.0);
        assert_eq!(percentile(&[1.0, 2.0], -5.0), 1.0);
        assert_eq!(Stats::from_samples(&[]), Stats::default());
        let s = Stats::from_samples(&[3.0, 1.0, 2.0]);
        assert_eq!((s.min, s.p50, s.max), (1.0, 2.0, 3.0));
    }

    #[test]
    fn config_models_are_the_enabled_entries() {
        let m = |n: &str, enabled: bool| ModelConfig {
            path: format!("models/{n}.onnx").into(),
            enabled,
            ..Default::default()
        };
        let mut config = Config {
            models: vec![m("a", false), m("b", true), m("c", false), m("d", true)],
            ..Default::default()
        };
        let names: Vec<String> = config_models(&config)
            .unwrap()
            .iter()
            .map(|m| m.effective_name())
            .collect();
        assert_eq!(names, ["b", "d"]);
        assert_eq!(find_model(&config, "C.onnx").unwrap().effective_name(), "c");
        assert!(find_model(&config, "zzz").is_none());

        for model in &mut config.models {
            model.enabled = false;
        }
        let err = config_models(&config).unwrap_err();
        assert!(format!("{err:#}").contains("disabled"), "{err:#}");
        let err = config_models(&Config::default()).unwrap_err();
        assert!(format!("{err:#}").contains("no models"), "{err:#}");
    }

    #[test]
    fn stats_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = Stats::from_samples(&v);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p95, 95.0);
        assert!((s.mean - 50.5).abs() < 1e-12);

        // Unsorted input, small n: p95 of 3 samples is the max; p50 the middle.
        let s = Stats::from_samples(&[3.0, 1.0, 2.0]);
        assert_eq!((s.p50, s.p95), (2.0, 3.0));
        let s = Stats::from_samples(&[7.0]);
        assert_eq!((s.min, s.p50, s.p95, s.max), (7.0, 7.0, 7.0, 7.0));
        assert_eq!(Stats::from_samples(&[]).max, 0.0);
    }

    #[test]
    fn iou_math() {
        let a = det("a", 1.0, [0, 0, 10, 10]);
        assert_eq!(iou(&a, &a), 1.0);
        // Half-overlap: inter 50, union 150.
        let b = det("a", 1.0, [5, 0, 15, 10]);
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        let c = det("a", 1.0, [20, 20, 30, 30]);
        assert_eq!(iou(&a, &c), 0.0);
    }

    /// BM-07: boxes near `usize::MAX` (a hand-edited benchmark.json) do not overflow.
    #[test]
    fn iou_near_usize_max_does_not_overflow() {
        let big = det("a", 1.0, [0, 0, usize::MAX, usize::MAX]);
        assert_eq!(iou(&big, &big), 1.0);
        let half = det("a", 1.0, [0, 0, usize::MAX, usize::MAX / 2]);
        let v = iou(&big, &half);
        assert!(v.is_finite() && (v - 0.5).abs() < 1e-3, "{v}");
        let small = det("a", 1.0, [0, 0, 10, 10]);
        assert!(iou(&big, &small) < 1e-9);
    }

    #[test]
    fn compare_matches_same_label_one_to_one() {
        let gpu = vec![
            det("dog", 0.90, [0, 0, 100, 100]),
            det("car", 0.80, [200, 200, 300, 300]),
            det("person", 0.6, [400, 0, 450, 100]),
        ];
        let cpu = vec![
            det("car", 0.78, [201, 200, 300, 301]),
            det("dog", 0.93, [1, 1, 100, 100]),
            // Same box as the GPU dog but another label: not a match.
            det("cat", 0.5, [0, 0, 100, 100]),
        ];
        let c = compare(&gpu, &cpu, 0.5);
        assert_eq!(c.matched, 2);
        assert_eq!(c.only_primary, 1);
        assert_eq!(c.only_cpu, 1);
        assert!((c.max_confidence_diff - 0.03).abs() < 1e-5);
        assert!((c.mean_confidence_diff - 0.025).abs() < 1e-5);

        // Two CPU boxes compete for one GPU box: only the higher-IoU one is matched.
        let gpu = vec![det("dog", 0.9, [0, 0, 100, 100])];
        let cpu = vec![
            det("dog", 0.5, [30, 0, 130, 100]),
            det("dog", 0.8, [0, 0, 100, 100]),
        ];
        let c = compare(&gpu, &cpu, 0.5);
        assert_eq!((c.matched, c.only_primary, c.only_cpu), (1, 0, 1));
        assert!((c.max_confidence_diff - 0.1).abs() < 1e-6);

        let c = compare(&[], &[], 0.5);
        assert_eq!(c.matched, 0);
        assert_eq!(c.max_confidence_diff, 0.0);

        // Merging per-image comparisons weights the mean by matches.
        let mut a = Comparison {
            matched: 1,
            mean_confidence_diff: 0.1,
            max_confidence_diff: 0.1,
            ..Default::default()
        };
        a.merge(&Comparison {
            matched: 3,
            only_cpu: 2,
            mean_confidence_diff: 0.02,
            max_confidence_diff: 0.05,
            ..Default::default()
        });
        assert_eq!((a.matched, a.only_cpu), (4, 2));
        assert!((a.mean_confidence_diff - 0.04).abs() < 1e-6);
        assert!((a.max_confidence_diff - 0.1).abs() < 1e-6);
    }

    #[test]
    fn embedded_image_decodes() {
        let img = crate::image::decode(DEFAULT_IMAGE).unwrap();
        assert!(img.width > 0 && img.height > 0);
        assert_eq!(img.rgb.len(), (img.width * img.height * 3) as usize);
    }

    #[test]
    fn configured_device_resolves_auto_and_force_cpu() {
        use crate::backend::select::{RuntimeProbe, select};
        let hw = crate::backend::HardwareInfo::this_platform_without_gpus();
        let sel = select(&hw, &RuntimeProbe::openvino_only(&["CPU".into()]), true);
        let m = ModelConfig::default();
        let mut c = Config::default();
        assert_eq!(
            configured_device(&c, &m, &sel).as_deref(),
            Some("openvino:cpu")
        );
        let over = ModelConfig {
            device: Some("ort:coreml".into()),
            ..Default::default()
        };
        assert_eq!(
            configured_device(&c, &over, &sel).as_deref(),
            Some("ort:coreml")
        );
        c.force_cpu = true;
        assert_eq!(
            configured_device(&c, &over, &sel).as_deref(),
            Some("openvino:cpu")
        );
        assert_eq!(reference_device(&sel), Some(Device::OPENVINO_CPU));
    }

    #[test]
    fn reference_model_choice() {
        let dir = std::env::temp_dir().join(format!("bop-ref-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["IPcam-general.onnx", "dfine-s.onnx", "rt-detrv2-l.onnx"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::write(
            dir.join("IPcam-general.yaml"),
            "NAMES:\n  - person\n  - vehicle\n  - unknown\n",
        )
        .unwrap();
        let m = |n: &str| ModelConfig {
            name: Some(n.into()),
            path: dir.join(format!("{n}.onnx")),
            ..Default::default()
        };
        let c = Config {
            models: vec![
                m("IPcam-general"),
                m("dfine-s"),
                m("rt-detrv2-x"),
                m("rt-detrv2-l"),
            ],
            ..Default::default()
        };
        // rt-detrv2-x is configured but its file is missing: the next best present one.
        assert_eq!(
            pick_reference(&c, None).unwrap().effective_name(),
            "rt-detrv2-l"
        );
        assert_eq!(
            pick_reference(&c, Some("DFINE-S"))
                .unwrap()
                .effective_name(),
            "dfine-s"
        );
        let e = pick_reference(&c, Some("rt-detrv2-x")).unwrap_err();
        assert!(format!("{e:#}").contains("not found"), "{e:#}");
        assert!(pick_reference(&c, Some("zzz")).is_err());
        // Only a non-COCO model: no reference.
        let only = Config {
            models: vec![m("IPcam-general")],
            ..Default::default()
        };
        assert!(pick_reference(&only, None).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Camera-trap wildlife a model cannot score (raccoon, opossum for a COCO model) is not a
    /// miss, and an animal-class box on it is neither TP nor FP; a person box on it is still a
    /// false positive. A model that has the class is scored on it.
    #[test]
    fn unscorable_wildlife_is_an_animal_ignore_region() {
        let g = |l: &str, b: [f32; 4], ignore: bool| GtBox {
            label: l.into(),
            bbox: b,
            ignore,
            crowd: false,
        };
        let p = |l: &str, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: 0.9,
            bbox: b,
        };
        let raccoon = [0.0, 0.0, 100.0, 100.0];
        let opossum = [200.0, 0.0, 300.0, 100.0];
        let gt = vec![
            g("raccoon", raccoon, false),
            g("Virginia Opossum", opossum, true),
            g("dog", [400.0, 0.0, 500.0, 100.0], false),
        ];
        let preds = vec![
            p("cat", raccoon),
            p("dog", opossum),
            p("person", raccoon),
            p("dog", [400.0, 0.0, 500.0, 100.0]),
        ];
        let scored: Vec<String> = ["person", "car", "dog", "cat", "raccoon"]
            .map(String::from)
            .to_vec();
        let coco = ClassMap::for_model(&crate::model::classes::coco80());
        let (mapped, eval, allowed) = eval_input(&coco, Some(&scored), &gt, &preds);
        assert_eq!(
            allowed.as_deref(),
            Some(&["car", "cat", "dog", "person"].map(String::from)[..])
        );
        assert!(
            mapped[0].is_none() && mapped[1].is_none(),
            "raccoon/opossum not scored"
        );
        // dog + 2 x (dog, cat) ignore regions (bird/horse/... are not scored on this set).
        assert_eq!(eval.gt.len(), 5, "{:?}", eval.gt);
        let (c, _, m) = metrics::image_counts(&eval, 0.5);
        assert_eq!((c.tp, c.fp, c.fn_), (1, 1, 0), "{m:?}");
        assert_eq!(m.pred, [None, None, Some(false), Some(true)]);
        // Frame level: cat and dog are excluded / positive, person is a false alert.
        let frames = roc::frame_samples(&eval, &frame_classes(&coco, allowed.as_deref()));
        let classes: Vec<&str> = frames.iter().map(|f| f.class.as_str()).collect();
        assert_eq!(
            classes,
            ["person", "car", "dog"],
            "cat excluded: only an ignore region"
        );

        // A model with a raccoon class (all classes) is scored on it: no proxies for raccoon.
        let mut labels = crate::model::classes::coco80();
        labels.push("raccoon".into());
        let all = ClassMap::all_classes(&labels);
        let (mapped, eval, _) = eval_input(&all, Some(&scored), &gt, &preds);
        assert_eq!(
            mapped[0].as_ref().map(|g| g.label.as_str()),
            Some("raccoon")
        );
        assert_eq!(
            eval.gt.len(),
            4,
            "raccoon + dog + opossum's dog/cat regions"
        );
        // Models without animal classes get no regions; existing labels are not wildlife.
        let ipcam = ClassMap::for_model(&["person".into(), "vehicle".into()]);
        assert!(ipcam.wildlife_ignore("raccoon").is_empty());
        for l in [
            "dog",
            "teddy bear",
            "baseball bat",
            "person",
            "mouse",
            "car",
        ] {
            assert!(!metrics::is_wildlife(l), "{l}");
        }
        for l in [
            "mountain_lion",
            "Striped Skunk",
            "Eastern Gray Squirrel",
            "animal",
            "black bear",
        ] {
            assert!(metrics::is_wildlife(l), "{l}");
        }
    }
}
