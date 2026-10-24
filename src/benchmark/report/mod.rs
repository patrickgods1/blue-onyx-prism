//! Benchmark results: per-device agreement with the CPU reference, the recommended device per
//! model, and the persisted `benchmark.json` (the latest run of every model, next to the config
//! file) that the CLI and the web UI both read and write.

use super::grade::{AccuracyMetric, Grade, Grades, Weights};
use super::images::{GroundTruth, ImageSet, SetKind};
use super::metrics::GtBox;
use super::threshold::{Objective, ThresholdAdvice};
use super::{Bench, Comparison, Detection, Job, MATCH_IOU, REFERENCE_DEVICES, RunResult, compare};
use crate::backend::detect::HardwareInfo;
use crate::backend::spec::Device;
use crate::config::{Config, ModelConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod recommend;
#[cfg(test)]
pub(crate) mod test_support;

pub use recommend::{ACCURACY_TIE, Recommendation, auto_orders, recommend, speed_order};

/// File name of the persisted results, in the config file's directory.
pub(crate) const RESULTS_FILE: &str = "benchmark.json";
/// Devices within this fraction of the fastest total p50 count as equally fast (timing noise);
/// among them the configured device is kept.
pub(crate) const NOISE_MARGIN: f64 = 0.05;

/// `benchmark.json` beside `config_path`.
pub fn results_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(|d| d.join(RESULTS_FILE))
        .unwrap_or_else(|| PathBuf::from(RESULTS_FILE))
}

/// How a device's detections compare with the CPU reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// This device is the reference.
    Reference,
    /// Same objects, confidences within a few hundredths.
    Agrees,
    /// Mostly the same objects (e.g. one borderline detection more or less).
    Minor,
    /// Different objects or confidences: the device computes something else (precision
    /// problem, broken kernel). Never recommended.
    Disagrees,
}

impl Verdict {
    /// Can be recommended.
    pub fn acceptable(self) -> bool {
        self != Verdict::Disagrees
    }
}

/// Agreement of one device's detections with the reference device's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agreement {
    /// Reference device spec ("openvino:cpu" or "ort:cpu").
    pub reference: String,
    pub matched: usize,
    /// Detections only this device found.
    pub only_device: usize,
    /// Reference detections this device missed.
    pub only_reference: usize,
    pub max_confidence_diff: f32,
    pub mean_confidence_diff: f32,
    /// `matched / max(device, reference)` detections (1 when both found nothing).
    pub score: f32,
    pub verdict: Verdict,
}

/// Score at or above which (with small confidence differences) a device agrees.
const AGREE_SCORE: f32 = 0.8;
const AGREE_CONF: f32 = 0.05;
/// Below this score (or above this mean confidence difference) a device disagrees.
const MINOR_SCORE: f32 = 0.5;
const MINOR_CONF: f32 = 0.15;

/// Compare `device` detections with `reference` detections (IoU >= [`MATCH_IOU`], same label).
pub(crate) fn agreement(
    device: &[Detection],
    reference: &[Detection],
    reference_spec: &str,
) -> Agreement {
    let c: Comparison = compare(device, reference, MATCH_IOU);
    verdict_of(&c, device.len().max(reference.len()), reference_spec)
}

/// Agreement of `device`'s detections with `reference`'s over every image both ran.
pub(crate) fn run_agreement(
    device: &RunResult,
    reference: &RunResult,
    reference_spec: &str,
) -> Agreement {
    let dets = |r: &RunResult, k: usize| -> Vec<Detection> {
        r.per_image
            .get(k)
            .map(|i| i.preds.iter().map(Detection::from).collect())
            .unwrap_or_default()
    };
    if device.per_image.is_empty() || device.per_image.len() != reference.per_image.len() {
        return agreement(&device.detections, &reference.detections, reference_spec);
    }
    let mut total = Comparison::default();
    let (mut n_dev, mut n_ref) = (0, 0);
    for k in 0..device.per_image.len() {
        let (a, b) = (dets(device, k), dets(reference, k));
        n_dev += a.len();
        n_ref += b.len();
        total.merge(&compare(&a, &b, MATCH_IOU));
    }
    verdict_of(&total, n_dev.max(n_ref), reference_spec)
}

fn verdict_of(c: &Comparison, n: usize, reference_spec: &str) -> Agreement {
    let score = if n == 0 {
        1.0
    } else {
        c.matched as f32 / n as f32
    };
    let verdict = if score >= AGREE_SCORE && c.mean_confidence_diff <= AGREE_CONF {
        Verdict::Agrees
    } else if score >= MINOR_SCORE && c.mean_confidence_diff <= MINOR_CONF {
        Verdict::Minor
    } else {
        Verdict::Disagrees
    };
    Agreement {
        reference: reference_spec.to_string(),
        matched: c.matched,
        only_device: c.only_primary,
        only_reference: c.only_cpu,
        max_confidence_diff: c.max_confidence_diff,
        mean_confidence_diff: c.mean_confidence_diff,
        score,
        verdict,
    }
}

/// One device of a model's sweep: the run, or why it did not run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceResult {
    /// Device spec ("openvino:cpu", "ort:coreml").
    pub device: String,
    /// Option label ("ONNX Runtime CoreML (Apple M1)").
    pub label: String,
    /// Why it could not run (not runnable, compile or inference error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunResult>,
}

impl DeviceResult {
    pub fn ran(device: &Device, label: &str, run: RunResult) -> Self {
        Self {
            device: device.to_string(),
            label: label.to_string(),
            error: None,
            run: Some(run),
        }
    }

    pub fn failed(device: &Device, label: &str, error: String) -> Self {
        Self {
            device: device.to_string(),
            label: label.to_string(),
            error: Some(error),
            run: None,
        }
    }

    /// Ran on the requested device itself (a fallback run does not count).
    pub fn ok(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.fell_back)
    }

    /// Full per-request p50 (ms) of a successful run.
    pub fn total_p50(&self) -> Option<f64> {
        self.run
            .as_ref()
            .filter(|_| self.ok())
            .map(|r| r.stages_ms.total.p50)
    }

    pub fn agreement(&self) -> Option<&Agreement> {
        self.run.as_ref().and_then(|r| r.agreement.as_ref())
    }

    pub fn grades(&self) -> Option<&Grades> {
        self.run
            .as_ref()
            .filter(|_| self.ok())
            .and_then(|r| r.grades.as_ref())
    }

    /// Ran and its detections are acceptable (or could not be checked: no reference).
    pub fn eligible(&self) -> bool {
        self.ok() && self.agreement().is_none_or(|a| a.verdict.acceptable())
    }
}

/// One model's sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResult {
    /// Model name (config `name` / file stem).
    pub model: String,
    #[serde(default)]
    pub path: String,
    /// Configured family ("auto" when detected at load time; each run reports the actual one).
    #[serde(default)]
    pub family: String,
    /// RFC 3339 time the sweep started.
    #[serde(default)]
    pub timestamp: String,
    /// The run this result belongs to: the run's start time (RFC 3339, the same for every model
    /// of one run; [`super::history::new_run_id`]). None in files of older versions.
    #[serde(default)]
    pub run_id: Option<String>,
    /// Dataset ids, comma separated.
    #[serde(default)]
    pub image: String,
    /// Dataset ids.
    #[serde(default)]
    pub datasets: Vec<String>,
    #[serde(default)]
    pub images: usize,
    #[serde(default)]
    pub warmup: usize,
    #[serde(default)]
    pub repeat: usize,
    /// Device the model was configured to run on at benchmark time.
    #[serde(default)]
    pub configured: Option<String>,
    /// Device whose detections the others were compared with.
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub devices: Vec<DeviceResult>,
    /// Options not tried because they cannot run here (sweeps over every runnable option).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<String>,
    pub recommended: Option<String>,
    #[serde(default)]
    pub recommendation: String,
    /// Why the model could not be benchmarked at all (e.g. the file is missing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// What the best confidence threshold optimizes (config `benchmark.threshold_objective`).
    #[serde(default)]
    pub threshold_objective: Objective,
    /// Best confidence threshold (from the recommended device's curves).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<ThresholdAdvice>,
}

impl ModelResult {
    /// An empty result for `job` (devices are added by the sweep).
    pub fn new(job: &Job, bench: &Bench, configured: Option<String>) -> Self {
        Self {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: job.family.to_string(),
            timestamp: now(),
            run_id: None,
            image: bench.describe_sets(),
            datasets: bench.sets.iter().map(|s| s.id.clone()).collect(),
            images: bench.image_count(),
            warmup: bench.warmup,
            repeat: bench.repeat,
            configured,
            reference: None,
            devices: Vec::new(),
            skipped: Vec::new(),
            recommended: None,
            recommendation: String::new(),
            error: None,
            threshold_objective: Objective::default(),
            threshold: None,
        }
    }

    /// A model that could not be benchmarked.
    pub fn failed(model: &str, path: &str, error: String) -> Self {
        Self {
            model: model.to_string(),
            path: path.to_string(),
            family: String::new(),
            timestamp: now(),
            run_id: None,
            image: String::new(),
            datasets: Vec::new(),
            images: 0,
            warmup: 0,
            repeat: 0,
            configured: None,
            reference: None,
            devices: Vec::new(),
            skipped: Vec::new(),
            recommended: None,
            recommendation: String::new(),
            error: Some(error),
            threshold_objective: Objective::default(),
            threshold: None,
        }
    }

    /// Recompute the reference, each device's agreement and the recommendation.
    pub fn finish(&mut self) {
        let reference = REFERENCE_DEVICES.iter().find_map(|r| {
            let spec = r.to_string();
            self.devices
                .iter()
                .find(|d| d.device == spec && d.ok())
                .and_then(|d| d.run.clone())
                .map(|run| (spec, run))
        });
        self.reference = reference.as_ref().map(|(s, _)| s.clone());
        for d in &mut self.devices {
            let ok = d.ok();
            let Some(run) = d.run.as_mut() else { continue };
            run.agreement = match &reference {
                Some((spec, _)) if ok && *spec == d.device => Some(Agreement {
                    reference: spec.clone(),
                    matched: run.detections.len(),
                    only_device: 0,
                    only_reference: 0,
                    max_confidence_diff: 0.0,
                    mean_confidence_diff: 0.0,
                    score: 1.0,
                    verdict: Verdict::Reference,
                }),
                Some((spec, reference)) if ok => Some(run_agreement(run, reference, spec)),
                _ => None,
            };
        }
        let rec = recommend(&self.devices, self.configured.as_deref(), NOISE_MARGIN);
        self.recommended = rec.device;
        self.recommendation = match &self.reference {
            Some(_) => rec.reason,
            None if self.devices.iter().any(DeviceResult::ok) => format!(
                "{} (no CPU reference ran, detections not checked)",
                rec.reason
            ),
            None => rec.reason,
        };
        self.threshold = super::threshold::advise(
            &self.devices,
            self.recommended.as_deref(),
            self.threshold_objective,
        );
        if self.recommended.is_some()
            && let Some(t) = self.threshold.as_ref().and_then(ThresholdAdvice::summary)
        {
            self.recommendation.push_str("; ");
            self.recommendation.push_str(&t);
        }
    }

    /// Grades of the recommended device.
    pub fn recommended_grades(&self) -> Option<&Grades> {
        let rec = self.recommended.as_deref()?;
        self.devices.iter().find(|d| d.device == rec)?.grades()
    }

    /// A copy without per-image details and per-device threshold curves (for polling; the
    /// threshold advice keeps the chosen device's curve).
    pub fn summary(&self) -> Self {
        let mut m = self.clone();
        for d in &mut m.devices {
            if let Some(r) = d.run.as_mut() {
                r.per_image.clear();
                r.eval_preds.clear();
                if let Some(a) = r.accuracy.as_mut() {
                    a.sweep = None;
                }
            }
        }
        m
    }

    /// The device row with the lowest total p50 among successful runs.
    pub fn fastest(&self) -> Option<&DeviceResult> {
        self.devices
            .iter()
            .filter_map(|d| d.total_p50().map(|p| (d, p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(d, _)| d)
    }
}

/// Host facts recorded with the results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HardwareSummary {
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub cpu: String,
    #[serde(default)]
    pub gpus: Vec<String>,
}

impl HardwareSummary {
    /// The results were measured on `hw`'s GPUs: same OS/arch and the same GPUs (index and
    /// name) in the same order, so `ort:directml:N` / `openvino:gpu.N` still name the same
    /// devices. Other GPU facts (VRAM, CUDA details) may differ.
    pub fn same_gpus(&self, hw: &HardwareInfo) -> bool {
        self.os == hw.os
            && self.arch == hw.arch
            && self.gpus.len() == hw.gpus.len()
            && self
                .gpus
                .iter()
                .zip(&hw.gpus)
                .all(|(s, g)| s.starts_with(&format!("#{} {} (", g.index, g.name)))
    }

    /// `#0 Name` of a recorded GPU string, without the trailing `(vendor, kind, VRAM, CUDA)`
    /// facts ([`GpuAdapter`]'s Display; the facts themselves contain no parenthesis).
    pub(crate) fn gpu_key(s: &str) -> &str {
        s.rfind(" (").map_or(s, |i| &s[..i]).trim()
    }

    /// Why results recorded on `self` are not comparable with ones measured on `other`, or
    /// None when they are: same OS/arch, the same GPUs (index and name, in order; VRAM and
    /// CUDA facts may differ, like [`Self::same_gpus`]) and the same CPU when both name one.
    pub fn difference(&self, other: &Self) -> Option<String> {
        if self.os != other.os || self.arch != other.arch {
            return Some(format!(
                "the platform changed ({} {} -> {} {})",
                self.os, self.arch, other.os, other.arch
            ));
        }
        let (a, b) = (self.cpu.trim(), other.cpu.trim());
        if !a.is_empty() && !b.is_empty() && a != b {
            return Some(format!("the CPU changed ({a} -> {b})"));
        }
        let a: Vec<&str> = self.gpus.iter().map(|g| Self::gpu_key(g)).collect();
        let b: Vec<&str> = other.gpus.iter().map(|g| Self::gpu_key(g)).collect();
        if a != b {
            let list = |v: &[&str]| {
                if v.is_empty() {
                    "none".to_string()
                } else {
                    v.join(", ")
                }
            };
            return Some(format!("the GPUs changed ({} -> {})", list(&a), list(&b)));
        }
        None
    }

    pub fn current() -> Self {
        let hw = crate::backend::detect::hardware();
        Self {
            os: hw.os.clone(),
            arch: hw.arch.clone(),
            cpu: crate::system_info::cpu_name(),
            gpus: hw.gpus.iter().map(|g| g.to_string()).collect(),
        }
    }
}

/// Runtime versions recorded with the results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RuntimeVersions {
    pub openvino: Option<String>,
    pub onnxruntime: Option<String>,
    pub onnxruntime_flavor: Option<String>,
}

impl RuntimeVersions {
    pub fn of(rt: &crate::backend::Runtimes) -> Self {
        let info = rt.info();
        Self {
            openvino: (!info.openvino_version.is_empty()).then_some(info.openvino_version),
            onnxruntime: rt.onnxruntime_version(),
            onnxruntime_flavor: rt
                .has_ort()
                .then(|| rt.probe().ort.flavor.clone())
                .flatten(),
        }
    }
}

/// One image of a dataset as recorded in the results (for the drill-down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInfo {
    pub file: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ground truth (or pseudo ground truth) used for scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<GtBox>>,
}

/// A dataset as recorded in the results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageSetInfo {
    pub id: String,
    #[serde(default)]
    pub title: String,
    pub kind: SetKind,
    /// Directory of the images (None for the embedded sample).
    #[serde(default)]
    pub dir: Option<std::path::PathBuf>,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
    pub ground_truth: GroundTruth,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scored_labels: Option<Vec<String>>,
    pub images: Vec<ImageInfo>,
}

impl ImageSetInfo {
    pub fn of(s: &ImageSet) -> Self {
        Self {
            id: s.id.clone(),
            title: s.title.clone(),
            kind: s.kind,
            dir: s.dir.clone(),
            license: s.license.clone(),
            attribution: s.attribution.clone(),
            source: s.source.clone(),
            ground_truth: s.ground_truth.clone(),
            scored_labels: s.scored_labels.clone(),
            images: s
                .images
                .iter()
                .map(|i| ImageInfo {
                    file: i.file.clone(),
                    width: i.width,
                    height: i.height,
                    tags: i.tags.clone(),
                    objects: i.gt.clone(),
                })
                .collect(),
        }
    }

    /// The file of image `file`, when it is one of this set's images (no other path is ever
    /// served). None for the embedded sample.
    pub fn image_path(&self, file: &str) -> Option<std::path::PathBuf> {
        if !super::images::safe_file_name(file) || !self.images.iter().any(|i| i.file == file) {
            return None;
        }
        self.dir.as_ref().map(|d| d.join(file))
    }
}

/// One row of the cross-model ranking: a model on its recommended device (or, in
/// [`BenchmarkResults::points`], on any device that ran).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankRow {
    pub rank: usize,
    pub model: String,
    pub device: String,
    pub overall: Grade,
    pub overall_points: f64,
    pub accuracy: Option<Grade>,
    pub speed: Grade,
    pub p50_ms: f64,
    pub ap50: Option<f64>,
    /// Macro frame ROC AUC.
    pub roc_auc: Option<f64>,
    /// What the accuracy grade (and the ranking) was computed from.
    pub metric: AccuracyMetric,
    /// The accuracy score of `metric`.
    pub accuracy_score: Option<f64>,
    pub relative: bool,
    pub datasets: Vec<String>,
    /// `openvino` or `onnxruntime`.
    pub runtime: String,
    /// No other row of the same list is both at least as fast and at least as accurate
    /// (and strictly better in one): [`super::history::pareto`].
    pub best_tradeoff: bool,
    /// The run the model's result is from.
    pub run_id: Option<String>,
    /// From the newest run of the results.
    pub this_run: bool,
    pub trust: super::history::Trust,
    /// Recall / precision at the model's best confidence threshold on this device (the
    /// advice), else at the run's configured threshold.
    pub recall: Option<f64>,
    pub precision: Option<f64>,
    /// The threshold `recall` / `precision` are at.
    pub threshold: Option<f32>,
}

impl RankRow {
    /// The row of device `d` of `m`; None when it has no grades (did not run, fell back).
    fn of(m: &ModelResult, d: &DeviceResult, latest: Option<&str>) -> Option<(Self, Grades)> {
        let g = *d.grades()?;
        let run = d.run.as_ref()?;
        let acc = run.accuracy.as_ref();
        // At the best threshold of this device when the advice has one, else the configured.
        let best = m.threshold.as_ref().and_then(|a| {
            if a.device == d.device {
                a.best.clone()
            } else {
                a.by_device
                    .iter()
                    .find(|p| p.device == d.device)
                    .and_then(|p| p.best.clone())
            }
        });
        let (recall, precision, threshold) = match &best {
            Some(b) => (b.point.recall, b.point.precision, Some(b.threshold)),
            None => (
                acc.and_then(|a| a.overall.recall),
                acc.and_then(|a| a.overall.precision),
                acc.map(|a| a.threshold),
            ),
        };
        Some((
            Self {
                rank: 0,
                model: m.model.clone(),
                device: d.device.clone(),
                overall: g.overall,
                overall_points: g.overall_points,
                accuracy: g.accuracy,
                speed: g.speed,
                p50_ms: g.p50_ms,
                ap50: acc.and_then(|a| a.overall.ap50),
                roc_auc: acc.and_then(|a| a.roc_auc()),
                metric: g.metric,
                accuracy_score: g.accuracy_score,
                relative: g.relative,
                datasets: m.datasets.clone(),
                runtime: super::history::runtime_of(&d.device).to_string(),
                best_tradeoff: false,
                run_id: m.run_id.clone(),
                this_run: latest.is_some() && m.run_id.as_deref() == latest,
                trust: super::history::trust::device_trust(m, d),
                recall,
                precision,
                threshold,
            },
            g,
        ))
    }
}

/// Rank `rows` (best first, [`Grades::rank_cmp`]) and mark the best trade-offs among them.
fn ranked(mut rows: Vec<(RankRow, Grades)>) -> Vec<RankRow> {
    rows.sort_by(|a, b| Grades::rank_cmp(&a.1, &b.1));
    let marks = super::history::pareto(
        &rows
            .iter()
            .map(|(r, _)| (r.p50_ms, r.accuracy_score))
            .collect::<Vec<_>>(),
    );
    rows.into_iter()
        .zip(marks)
        .enumerate()
        .map(|(i, ((mut r, _), best))| {
            r.rank = i + 1;
            r.best_tradeoff = best;
            r
        })
        .collect()
}

/// The persisted results file: the latest sweep of every benchmarked model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Blue Onyx Prism version that wrote the file.
    #[serde(default)]
    pub version: String,
    /// RFC 3339 time of the last update.
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub hardware: HardwareSummary,
    #[serde(default)]
    pub runtimes: RuntimeVersions,
    #[serde(default)]
    pub models: Vec<ModelResult>,
    /// Datasets the models were run on (images, tags, ground truth).
    #[serde(default)]
    pub image_sets: Vec<ImageSetInfo>,
}

/// `name` is in `only` (normalized names), or `only` is None.
fn wanted(only: Option<&[String]>, name: &str) -> bool {
    only.is_none_or(|o| {
        o.iter()
            .any(|n| crate::registry::normalize_name(n) == crate::registry::normalize_name(name))
    })
}

/// The text of `path`; None when it does not exist.
pub(crate) fn read_if_exists(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Parse the JSON object `text` (the file `path`) as `T` with its array field `list` taken
/// out and deserialized entry by entry as `E`. Entries that fail are dropped with a warning
/// naming the file and the entry (its `model` and `device` when present, else its position).
pub(crate) fn parse_lenient<T, E>(
    text: &str,
    list: &str,
    path: &Path,
) -> Result<(T, Vec<E>, Vec<String>)>
where
    T: serde::de::DeserializeOwned,
    E: serde::de::DeserializeOwned,
{
    let ctx = || format!("parsing {}", path.display());
    let mut v: serde_json::Value = serde_json::from_str(text).with_context(ctx)?;
    let obj = v
        .as_object_mut()
        .with_context(|| format!("parsing {}: not a JSON object", path.display()))?;
    let entries = match obj.remove(list) {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::Array(a)) => a,
        Some(_) => anyhow::bail!("parsing {}: `{list}` is not a list", path.display()),
    };
    let head: T = serde_json::from_value(v).with_context(ctx)?;
    let mut items = Vec::with_capacity(entries.len());
    let mut warnings = Vec::new();
    for (i, e) in entries.into_iter().enumerate() {
        let field = |k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let name = match (field("model"), field("device")) {
            (Some(m), Some(d)) => format!("#{} ('{m}' on {d})", i + 1),
            (Some(m), None) => format!("#{} ('{m}')", i + 1),
            _ => format!("#{}", i + 1),
        };
        match serde_json::from_value::<E>(e) {
            Ok(x) => items.push(x),
            Err(err) => warnings.push(format!(
                "{}: dropped `{list}` entry {name}, which does not load: {err}",
                path.display()
            )),
        }
    }
    Ok((head, items, warnings))
}

/// Rename `path` aside to `<name>.bad-<YYYYmmddTHHMMSS>` (local time) in the same directory;
/// returns the new path.
pub(crate) fn move_aside(path: &Path) -> Result<PathBuf> {
    let ts = chrono::Local::now().format("%Y%m%dT%H%M%S");
    let mut name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?
        .to_os_string();
    name.push(format!(".bad-{ts}"));
    let dest = path.with_file_name(name);
    std::fs::rename(path, &dest)
        .with_context(|| format!("renaming {} to {}", path.display(), dest.display()))?;
    Ok(dest)
}

/// `load(path)`, and when the file exists but does not load: rename it aside with a warning
/// (logged and returned) and continue without it. Errors when the rename fails, so the caller
/// does not overwrite the file.
pub(crate) fn load_or_move_aside<T>(
    path: &Path,
    load: impl Fn(&Path) -> Result<(Option<T>, Vec<String>)>,
) -> Result<(Option<T>, Vec<String>)> {
    match load(path) {
        Ok((v, warnings)) => {
            for w in &warnings {
                tracing::warn!("{w}");
            }
            Ok((v, warnings))
        }
        Err(e) => {
            let dest = move_aside(path).with_context(|| {
                format!(
                    "{} does not load ({e:#}) and could not be moved aside; not overwriting it",
                    path.display()
                )
            })?;
            let w = format!(
                "{} does not load ({e:#}); moved it aside to {} and started a new file",
                path.display(),
                dest.display()
            );
            tracing::warn!("{w}");
            Ok((None, vec![w]))
        }
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl BenchmarkResults {
    pub fn new(
        hardware: HardwareSummary,
        runtimes: RuntimeVersions,
        models: Vec<ModelResult>,
    ) -> Self {
        Self {
            version: crate::VERSION.to_string(),
            timestamp: now(),
            hardware,
            runtimes,
            models,
            image_sets: Vec::new(),
        }
    }

    /// With the datasets the models ran on.
    pub fn with_sets(mut self, sets: &[ImageSet]) -> Self {
        self.image_sets = sets.iter().map(ImageSetInfo::of).collect();
        self
    }

    pub fn image_set(&self, id: &str) -> Option<&ImageSetInfo> {
        self.image_sets.iter().find(|s| s.id == id)
    }

    /// Models ranked by the grades of their recommended device (best first): "the best model
    /// for this machine" is the first row.
    pub fn ranking(&self) -> Vec<RankRow> {
        let latest = super::history::view::latest_run_id(self);
        ranked(
            self.models
                .iter()
                .filter_map(|m| {
                    let rec = m.recommended.as_deref()?;
                    let d = m.devices.iter().find(|d| d.device == rec)?;
                    RankRow::of(m, d, latest.as_deref())
                })
                .collect(),
        )
    }

    /// Every model on every device that ran (with grades), ranked like [`Self::ranking`], the
    /// best trade-offs marked over all of them (the Compare scatter).
    pub fn points(&self) -> Vec<RankRow> {
        let latest = super::history::view::latest_run_id(self);
        ranked(
            self.models
                .iter()
                .flat_map(|m| {
                    m.devices
                        .iter()
                        .filter_map(|d| RankRow::of(m, d, latest.as_deref()))
                        .collect::<Vec<_>>()
                })
                .collect(),
        )
    }

    /// The metric the grades were computed from (the first graded run's; AP50 without any).
    pub fn accuracy_metric(&self) -> AccuracyMetric {
        self.models
            .iter()
            .flat_map(|m| m.devices.iter())
            .find_map(|d| d.run.as_ref()?.grades.filter(|g| g.accuracy.is_some()))
            .map_or(AccuracyMetric::Ap50, |g| g.metric)
    }

    /// Re-grade every run on `metric` with `weights` from its stored accuracy and speed, and
    /// recompute each model's recommendation: models kept from earlier runs (merged results)
    /// are then graded like the new ones, so the ranking compares like with like.
    pub fn regrade(&mut self, metric: AccuracyMetric, weights: Weights) {
        for m in &mut self.models {
            let mut changed = false;
            for d in &mut m.devices {
                let Some(run) = d.run.as_mut() else { continue };
                let Some(old) = run.grades else { continue };
                let score = run.accuracy.as_ref().and_then(|a| a.score(metric));
                let g = Grades::with_metric(metric, score, old.p50_ms, weights, old.relative);
                if g != old {
                    run.grades = Some(g);
                    changed = true;
                }
            }
            if changed {
                m.finish();
            }
        }
    }

    /// A copy without per-image details (for polling).
    pub fn summary(&self) -> Self {
        let mut r = self.clone();
        r.models = r.models.iter().map(ModelResult::summary).collect();
        for s in &mut r.image_sets {
            for i in &mut s.images {
                i.objects = None;
            }
        }
        r
    }

    /// The file at `path`; None when it does not exist. Model entries that do not load are
    /// dropped (logged); see [`Self::load_with_warnings`].
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let (r, warnings) = Self::load_with_warnings(path)?;
        for w in warnings {
            tracing::warn!("{w}");
        }
        Ok(r)
    }

    /// [`Self::load`] returning the warnings instead of logging them: the `models` list is
    /// deserialized entry by entry, so one bad entry drops only that entry (with a warning
    /// naming it). An unreadable or unparsable file is an error.
    pub fn load_with_warnings(path: &Path) -> Result<(Option<Self>, Vec<String>)> {
        let Some(text) = read_if_exists(path)? else {
            return Ok((None, Vec::new()));
        };
        let (mut r, models, warnings) = parse_lenient::<Self, ModelResult>(&text, "models", path)?;
        r.models = models;
        Ok((Some(r), warnings))
    }

    /// The previous results to merge a new run into before saving: like
    /// [`Self::load_with_warnings`], but a file that exists and does not load is renamed aside
    /// (`move_aside`) with a warning instead of being overwritten by the next save. Errors
    /// only when that rename fails (the caller must then not save).
    pub fn load_for_merge(path: &Path) -> Result<(Option<Self>, Vec<String>)> {
        load_or_move_aside(path, Self::load_with_warnings)
    }

    /// [`Self::load`], logging (not returning) errors.
    pub fn load_or_warn(path: &Path) -> Option<Self> {
        Self::load(path)
            .map_err(|e| tracing::warn!("{e:#}"))
            .ok()
            .flatten()
    }

    /// Write pretty JSON through a temporary file and a rename (readers never see half a file).
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }

    /// Why `self`'s results (an earlier run) are not comparable with `new`'s, or None: the
    /// hardware differs ([`HardwareSummary::difference`]) or a runtime version changed.
    pub fn incomparable(&self, new: &Self) -> Option<String> {
        if let Some(d) = self.hardware.difference(&new.hardware) {
            return Some(d);
        }
        if self.runtimes != new.runtimes {
            let v = |r: &RuntimeVersions| {
                format!(
                    "OpenVINO {}, ONNX Runtime {}{}",
                    r.openvino.as_deref().unwrap_or("none"),
                    r.onnxruntime.as_deref().unwrap_or("none"),
                    r.onnxruntime_flavor
                        .as_deref()
                        .map(|f| format!(" ({f})"))
                        .unwrap_or_default()
                )
            };
            return Some(format!(
                "the runtime versions changed ({} -> {})",
                v(&self.runtimes),
                v(&new.runtimes)
            ));
        }
        None
    }

    /// The warning for a merge of `new` into `previous` that drops the previous results
    /// ([`Self::merge`]), naming what changed and how many models are lost; None when they are
    /// kept (or there are none).
    pub fn merge_warning(previous: Option<&Self>, new: &Self) -> Option<String> {
        let prev = previous?;
        let reason = prev.incomparable(new)?;
        let new_keys: Vec<String> = new
            .models
            .iter()
            .map(|m| crate::registry::normalize_name(&m.model))
            .collect();
        let lost: Vec<&str> = prev
            .models
            .iter()
            .filter(|m| !new_keys.contains(&crate::registry::normalize_name(&m.model)))
            .map(|m| m.model.as_str())
            .collect();
        if lost.is_empty() {
            return None;
        }
        Some(format!(
            "The earlier benchmark results of {} were dropped. They were measured with other \
             hardware or runtimes: {reason}, so their timings are not comparable. Benchmark \
             those models again to rank them with this run.",
            lost.join(", ")
        ))
    }

    /// Fold `new` into the previous results: models in `new` replace their previous entries
    /// (matched by normalized name), other models are kept, unless the hardware or runtime
    /// versions changed ([`Self::incomparable`]; then the previous results are dropped as not
    /// comparable, see [`Self::merge_warning`]).
    pub fn merge(previous: Option<Self>, new: Self) -> Self {
        let Some(prev) = previous else { return new };
        if prev.incomparable(&new).is_some() {
            return new;
        }
        let key = |m: &ModelResult| crate::registry::normalize_name(&m.model);
        let mut models: Vec<ModelResult> = prev.models;
        for m in new.models {
            match models.iter_mut().find(|p| key(p) == key(&m)) {
                Some(slot) => *slot = m,
                None => models.push(m),
            }
        }
        let mut image_sets = prev.image_sets;
        for s in new.image_sets {
            match image_sets.iter_mut().find(|p| p.id == s.id) {
                Some(slot) => *slot = s,
                None => image_sets.push(s),
            }
        }
        // Keep only the sets some model still refers to.
        image_sets.retain(|s| models.iter().any(|m| m.datasets.contains(&s.id)));
        Self {
            models,
            image_sets,
            ..new
        }
    }

    /// The latest result for model `name` (normalized match).
    pub fn find(&self, name: &str) -> Option<&ModelResult> {
        let key = crate::registry::normalize_name(name);
        self.models
            .iter()
            .find(|m| crate::registry::normalize_name(&m.model) == key)
    }

    /// The measured `auto` order of model `name` on `hw` ([`speed_order`] of its latest sweep):
    /// None when it has no results, they were measured on other GPUs
    /// ([`HardwareSummary::same_gpus`]) or no device qualified. Specs that no longer parse are
    /// dropped.
    pub fn auto_order(&self, name: &str, hw: &HardwareInfo) -> Option<Vec<Device>> {
        if !self.hardware.same_gpus(hw) {
            return None;
        }
        let order: Vec<Device> = speed_order(&self.find(name)?.devices)
            .into_iter()
            .filter_map(|s| crate::backend::spec::parse(s).ok()?.device().copied())
            .collect();
        (!order.is_empty()).then_some(order)
    }

    /// `(model, best confidence threshold)` of every model with threshold advice, optionally
    /// limited to `only` (normalized names).
    pub fn thresholds(&self, only: Option<&[String]>) -> Vec<(String, f32)> {
        self.models
            .iter()
            .filter(|m| wanted(only, &m.model))
            .filter_map(|m| {
                let t = m.threshold.as_ref()?.apply_value()?;
                Some((m.model.clone(), t))
            })
            .collect()
    }

    /// `(model, device, best confidence threshold)` for every device that ran of every model
    /// with threshold advice ([`ThresholdAdvice::device_apply_values`]: devices whose pick does
    /// not apply are left out), optionally limited to `only` (normalized names). Models in
    /// result order, devices in each model's [`ThresholdAdvice::by_device`] order. These are
    /// the confidence profiles an "apply all" writes
    /// ([`crate::config_form::apply_device_thresholds`]).
    pub fn device_thresholds(&self, only: Option<&[String]>) -> Vec<(String, String, f32)> {
        self.models
            .iter()
            .filter(|m| wanted(only, &m.model))
            .filter_map(|m| Some((m, m.threshold.as_ref()?.device_apply_values())))
            .flat_map(|(m, v)| v.into_iter().map(|(d, t)| (m.model.clone(), d, t)))
            .collect()
    }

    /// `(model, recommended device)` of every model with a recommendation, optionally limited
    /// to `only` (normalized names).
    pub fn recommendations(&self, only: Option<&[String]>) -> Vec<(String, String)> {
        let wanted = |name: &str| {
            only.is_none_or(|o| {
                o.iter().any(|n| {
                    crate::registry::normalize_name(n) == crate::registry::normalize_name(name)
                })
            })
        };
        self.models
            .iter()
            .filter(|m| wanted(&m.model))
            .filter_map(|m| m.recommended.as_ref().map(|d| (m.model.clone(), d.clone())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn agreement_verdicts() {
        let r = scene();
        // Identical: agrees.
        let a = agreement(&r, &r, "openvino:cpu");
        assert_eq!((a.matched, a.score, a.verdict), (3, 1.0, Verdict::Agrees));
        // Slightly shifted boxes and confidences: still agrees.
        let near: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                confidence: d.confidence - 0.01,
                x_min: d.x_min + 2,
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&near, &r, "x").verdict, Verdict::Agrees);
        // One of three missing: 2/3 matched -> minor.
        let a = agreement(&r[..2], &r, "x");
        assert_eq!(a.only_reference, 1);
        assert_eq!(a.verdict, Verdict::Minor);
        // Nothing found where the reference found three: disagrees.
        assert_eq!(agreement(&[], &r, "x").verdict, Verdict::Disagrees);
        // Wrong labels: disagrees.
        let wrong: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                label: "person".into(),
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&wrong, &r, "x").verdict, Verdict::Disagrees);
        // Same boxes, confidences far off (precision problem): disagrees.
        let off: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                confidence: d.confidence - 0.3,
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&off, &r, "x").verdict, Verdict::Disagrees);
        // Both empty: agrees.
        assert_eq!(agreement(&[], &[], "x").verdict, Verdict::Agrees);
    }

    #[test]
    fn results_round_trip_and_merge() {
        let dir = std::env::temp_dir().join(format!("bop-bench-{}", uuid::Uuid::new_v4()));
        let cfg = dir.join("blue_onyx_prism_config.json");
        let path = results_path(&cfg);
        assert_eq!(path, dir.join(RESULTS_FILE));
        assert!(BenchmarkResults::load(&path).unwrap().is_none());

        let hw = HardwareSummary {
            os: "macos".into(),
            arch: "aarch64".into(),
            cpu: "Apple M1".into(),
            gpus: vec!["#0 Apple M1".into()],
        };
        let rtv = RuntimeVersions {
            openvino: Some("2026.0.0".into()),
            onnxruntime: Some("1.24.4".into()),
            onnxruntime_flavor: Some("coreml".into()),
        };
        let mut a = model(vec![dev("openvino:cpu", 37.0, scene())], None);
        a.model = "IPcam-general".into();
        let mut b = model(
            vec![
                dev("ort:coreml", 44.0, scene()),
                dev("openvino:cpu", 89.0, scene()),
            ],
            None,
        );
        b.model = "dfine-s".into();
        let first = BenchmarkResults::new(hw.clone(), rtv.clone(), vec![a.clone(), b.clone()]);
        first.save(&path).unwrap();
        let back = BenchmarkResults::load(&path).unwrap().unwrap();
        assert_eq!(back, first);
        assert_eq!(
            back.find("ipcam-general.onnx").unwrap().model,
            "IPcam-general"
        );
        assert_eq!(
            back.recommendations(None),
            [
                ("IPcam-general".to_string(), "openvino:cpu".to_string()),
                ("dfine-s".to_string(), "ort:coreml".to_string())
            ]
        );
        assert_eq!(
            back.recommendations(Some(&["DFINE-S".to_string()])).len(),
            1
        );

        // A new run of one model replaces only that model.
        let mut a2 = model(vec![dev("ort:cpu", 50.0, scene())], None);
        a2.model = "ipcam-general".into();
        let second = BenchmarkResults::new(hw.clone(), rtv.clone(), vec![a2.clone()]);
        let merged = BenchmarkResults::merge(Some(back.clone()), second);
        assert_eq!(merged.models.len(), 2);
        assert_eq!(merged.models[0], a2);
        assert_eq!(merged.models[1], b);
        // Different runtimes: the old results are dropped.
        let other = RuntimeVersions {
            onnxruntime: Some("1.25.0".into()),
            ..rtv
        };
        let third = BenchmarkResults::new(hw, other, vec![a2]);
        assert_eq!(BenchmarkResults::merge(Some(back), third).models.len(), 1);

        // Garbage is an error, not a panic.
        std::fs::write(&path, "{not json").unwrap();
        assert!(BenchmarkResults::load(&path).is_err());
        assert!(BenchmarkResults::load_or_warn(&path).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Results stay comparable when only GPU facts (VRAM, CUDA suffix) changed; another GPU,
    /// CPU, platform or runtime drops them, with a warning naming the lost models (BM-03).
    #[test]
    fn merge_compares_gpus_by_index_and_name() {
        let hw = |gpu: &str| HardwareSummary {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz".into(),
            gpus: vec![
                gpu.into(),
                "#1 Intel(R) UHD Graphics 630 (Intel, integrated, 128 MB VRAM)".into(),
            ],
        };
        let old_gpu = "#0 NVIDIA GeForce GTX 1070 Ti (NVIDIA, discrete, 8060 MB VRAM)";
        let new_gpu = "#0 NVIDIA GeForce GTX 1070 Ti (NVIDIA, discrete, 8192 MB VRAM, \
                       CUDA device 0 compute capability 6.1)";
        let named = |name: &str| {
            let mut m = model(vec![dev("openvino:cpu", 37.0, scene())], None);
            m.model = name.into();
            m
        };
        let prev = BenchmarkResults::new(
            hw(old_gpu),
            RuntimeVersions::default(),
            vec![named("a"), named("b")],
        );
        let new = BenchmarkResults::new(hw(new_gpu), RuntimeVersions::default(), vec![named("a")]);
        assert_eq!(prev.incomparable(&new), None);
        assert_eq!(BenchmarkResults::merge_warning(Some(&prev), &new), None);
        assert_eq!(
            BenchmarkResults::merge(Some(prev.clone()), new)
                .models
                .len(),
            2
        );

        // Another GPU: dropped, and the warning names the lost model and the change.
        let other = BenchmarkResults::new(
            hw("#0 NVIDIA GeForce RTX 3060 (NVIDIA, discrete, 12288 MB VRAM)"),
            RuntimeVersions::default(),
            vec![named("a")],
        );
        let w = BenchmarkResults::merge_warning(Some(&prev), &other).unwrap();
        assert!(w.contains(" b ") && !w.contains(" a,"), "{w}");
        assert!(w.contains("RTX 3060"), "{w}");
        assert_eq!(
            BenchmarkResults::merge(Some(prev.clone()), other)
                .models
                .len(),
            1
        );

        // A GPU without the facts suffix (older files) matches by index and name.
        let mut bare = prev.clone();
        bare.hardware.gpus[0] = "#0 NVIDIA GeForce GTX 1070 Ti".into();
        let new = BenchmarkResults::new(hw(new_gpu), RuntimeVersions::default(), vec![]);
        assert_eq!(bare.incomparable(&new), None);

        // Another CPU (when both name one), platform or runtime is not comparable.
        let mut cpu = new.clone();
        cpu.hardware.cpu = "AMD Ryzen 5 5600X".into();
        assert!(prev.incomparable(&cpu).unwrap().contains("CPU"));
        cpu.hardware.cpu.clear();
        assert_eq!(prev.incomparable(&cpu), None);
        let mut os = new.clone();
        os.hardware.os = "linux".into();
        assert!(prev.incomparable(&os).unwrap().contains("platform"));
        let mut rt = new.clone();
        rt.runtimes.onnxruntime = Some("1.24.4".into());
        assert!(
            prev.incomparable(&rt)
                .unwrap()
                .contains("ONNX Runtime 1.24.4")
        );
        // Nothing lost (the new run re-measured every earlier model): no warning.
        let mut all = rt.clone();
        all.models = vec![named("A"), named("b")];
        assert_eq!(BenchmarkResults::merge_warning(Some(&prev), &all), None);
    }

    #[test]
    fn unloadable_results_are_moved_aside_and_bad_entries_dropped() {
        let dir = std::env::temp_dir().join(format!("bop-bench-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(RESULTS_FILE);
        let mut a = model(vec![dev("openvino:cpu", 37.0, scene())], None);
        a.model = "IPcam-general".into();
        let mut b = model(vec![dev("ort:cpu", 44.0, scene())], None);
        b.model = "dfine-s".into();
        let r = BenchmarkResults::new(
            HardwareSummary::default(),
            RuntimeVersions::default(),
            vec![a.clone(), b],
        );
        r.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        // One bad entry (devices is not a list): only that entry is dropped, with a warning.
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["models"][1]["devices"] = serde_json::json!("oops");
        std::fs::write(&path, v.to_string()).unwrap();
        let (back, warnings) = BenchmarkResults::load_for_merge(&path).unwrap();
        let back = back.unwrap();
        assert_eq!(back.models, vec![a.clone()]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("dfine-s"), "{}", warnings[0]);
        assert!(path.is_file(), "a partly bad file is not moved");

        // Missing optional fields load (older/newer writers).
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["models"][0]["devices"][0]["run"]
            .as_object_mut()
            .unwrap()
            .remove("cache");
        v.as_object_mut().unwrap().remove("timestamp");
        std::fs::write(&path, v.to_string()).unwrap();
        let (back, warnings) = BenchmarkResults::load_with_warnings(&path).unwrap();
        assert_eq!(back.unwrap().models.len(), 2);
        assert!(warnings.is_empty(), "{warnings:?}");

        // Truncated: moved aside to `benchmark.json.bad-<ts>`, with a warning naming it.
        std::fs::write(&path, &text[..text.len() / 2]).unwrap();
        let (back, warnings) = BenchmarkResults::load_for_merge(&path).unwrap();
        assert!(back.is_none());
        assert!(!path.exists());
        let aside: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(&format!("{RESULTS_FILE}.bad-")))
            .collect();
        assert_eq!(aside.len(), 1, "{aside:?}");
        assert_eq!(aside[0].len(), RESULTS_FILE.len() + ".bad-".len() + 15);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains(&aside[0]), "{}", warnings[0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn regrade_switches_metric_from_stored_accuracy() {
        use super::super::AccuracyReport;
        use super::super::metrics::{ClassMetrics, Summary};
        use super::super::roc::FrameRoc;
        // Model a: AP50 0.80, AUC 0.85; model b: AP50 0.55, AUC 0.97. AP50 ranks a first,
        // ROC AUC ranks b first.
        let with_acc = |name: &str, ap: f64, auc: f64| {
            let mut d = graded("openvino:cpu", 40.0, Some(ap));
            d.run.as_mut().unwrap().accuracy = Some(AccuracyReport {
                ground_truth: "ground truth".into(),
                relative: false,
                classes: vec!["person".into()],
                threshold: 0.5,
                overall: Summary {
                    ap50: Some(ap),
                    per_class: vec![ClassMetrics {
                        class: "person".into(),
                        gt: 3,
                        ap50: Some(ap),
                        ap50_95: None,
                        precision: None,
                        recall: None,
                    }],
                    ..Default::default()
                },
                by_dataset: vec![],
                by_tag: vec![],
                all_classes: None,
                sweep: None,
                frame_roc: Some(FrameRoc {
                    auc: Some(auc),
                    ..Default::default()
                }),
            });
            let mut m = model(vec![d], None);
            m.model = name.into();
            m
        };
        let mut r = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![with_acc("a", 0.80, 0.85), with_acc("b", 0.55, 0.97)],
        );
        let names = |r: &BenchmarkResults| -> Vec<String> {
            r.ranking().into_iter().map(|x| x.model).collect()
        };
        assert_eq!(names(&r), ["a", "b"]);
        assert_eq!(r.accuracy_metric(), AccuracyMetric::Ap50);
        r.regrade(AccuracyMetric::RocAuc, Default::default());
        assert_eq!(r.accuracy_metric(), AccuracyMetric::RocAuc);
        assert_eq!(names(&r), ["b", "a"]);
        let g = r.models[1].recommended_grades().unwrap();
        assert_eq!((g.accuracy, g.accuracy_score), (Some(Grade::A), Some(0.97)));
        assert!(r.models[1].recommendation.contains("by ROC AUC"));
        // And back.
        r.regrade(AccuracyMetric::Ap50, Default::default());
        assert_eq!(names(&r), ["a", "b"]);
        // A run without frame ROC (older results) is unscored under ROC AUC.
        let mut old = with_acc("old", 0.7, 0.9);
        old.devices[0]
            .run
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .frame_roc = None;
        let mut r = BenchmarkResults::new(Default::default(), Default::default(), vec![old]);
        r.regrade(AccuracyMetric::RocAuc, Default::default());
        assert_eq!(r.models[0].recommended_grades().unwrap().accuracy, None);
    }

    #[test]
    fn agreement_over_many_images() {
        let img = |dets: Vec<Detection>| super::super::ImageRun {
            set: "s".into(),
            file: "f".into(),
            total_ms: 1.0,
            infer_ms: 1.0,
            preds: dets
                .iter()
                .map(|d| super::super::ImagePred {
                    label: d.label.clone(),
                    confidence: d.confidence,
                    bbox: [
                        d.x_min as f32,
                        d.y_min as f32,
                        d.x_max as f32,
                        d.y_max as f32,
                    ],
                    tp: None,
                })
                .collect(),
            gt_matched: vec![],
            counts: None,
        };
        let mut reference = run("openvino:cpu", 10.0, vec![]);
        reference.per_image = vec![img(scene()), img(scene())];
        let mut same = run("ort:coreml", 5.0, vec![]);
        same.per_image = vec![img(scene()), img(scene())];
        let a = run_agreement(&same, &reference, "openvino:cpu");
        assert_eq!((a.matched, a.verdict), (6, Verdict::Agrees));
        // Second image empty: 3 of 6 -> minor.
        let mut half = run("ort:coreml", 5.0, vec![]);
        half.per_image = vec![img(scene()), img(vec![])];
        let a = run_agreement(&half, &reference, "openvino:cpu");
        assert_eq!(
            (a.matched, a.only_reference, a.verdict),
            (3, 3, Verdict::Minor)
        );
        let mut none = run("ort:coreml", 5.0, vec![]);
        none.per_image = vec![img(vec![]), img(vec![])];
        assert_eq!(
            run_agreement(&none, &reference, "x").verdict,
            Verdict::Disagrees
        );
    }

    #[test]
    fn image_paths_are_restricted_to_the_set() {
        let info = ImageSetInfo {
            id: "s".into(),
            title: "s".into(),
            kind: SetKind::Dir,
            dir: Some("/data/s".into()),
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            images: vec![ImageInfo {
                file: "a.jpg".into(),
                width: 1,
                height: 1,
                tags: vec![],
                objects: None,
            }],
        };
        assert_eq!(info.image_path("a.jpg"), Some("/data/s/a.jpg".into()));
        assert_eq!(info.image_path("b.jpg"), None);
        assert_eq!(info.image_path("../a.jpg"), None);
    }

    #[test]
    fn threshold_advice_from_the_recommended_device() {
        // CPU (reference) and a faster CoreML that agrees: CoreML is recommended and its curve
        // is used. Kept sets: CPU best keeps down to 0.45, CoreML down to 0.35.
        let cpu = with_sweep(
            dev("openvino:cpu", 50.0, scene()),
            3,
            &[(0.9, true), (0.6, true), (0.45, true), (0.2, false)],
        );
        let coreml = with_sweep(
            dev("ort:coreml", 10.0, scene()),
            3,
            &[
                (0.9, true),
                (0.6, true),
                (0.55, false),
                (0.35, true),
                (0.2, false),
            ],
        );
        let m = model(vec![cpu, coreml], None);
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        let a = m.threshold.as_ref().unwrap();
        assert_eq!(a.device, "ort:coreml");
        assert_eq!(a.objective, "f1");
        let best = a.best.as_ref().unwrap();
        // F1: {.9,.6} .8, {+.55 FP} .667, {+.35} .857, {+.2 FP} .75 -> (0.2, 0.35] -> 0.27.
        assert_eq!(best.exact_threshold, Some(0.35));
        assert_eq!(best.threshold, 0.27);
        assert_eq!(a.configured.threshold, 0.5);
        assert_eq!(a.by_device.len(), 2);
        let cpu_best = a.by_device[0].best.as_ref().unwrap();
        assert_eq!(cpu_best.exact_threshold, Some(0.45));
        assert_eq!(a.alternatives.len(), 3);
        assert_eq!(a.by_dataset[0].key, "exdark-night");
        assert_eq!(a.per_class[0].key, "person");
        assert!(!a.exact.is_empty() && a.curve.len() == 19);
        assert!(
            m.recommendation
                .contains("P/R at the best confidence threshold 0.27 (F1)"),
            "{}",
            m.recommendation
        );
        assert!(m.recommendation.contains("at the configured 0.50"));
        // Polling summary: no per-device curves, the advice stays.
        let s = m.summary();
        assert!(s.devices.iter().all(|d| {
            d.run
                .as_ref()
                .unwrap()
                .accuracy
                .as_ref()
                .unwrap()
                .sweep
                .is_none()
        }));
        assert_eq!(s.threshold, m.threshold);
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m.clone()]);
        assert_eq!(r.thresholds(None), [("m".to_string(), 0.27)]);
        assert!(r.thresholds(Some(&["other".into()])).is_empty());
        // Per device: every scored device row whose pick applies.
        let per_device = r.device_thresholds(None);
        let by_device = &m.threshold.as_ref().unwrap().by_device;
        assert!(!per_device.is_empty());
        assert_eq!(
            per_device.len(),
            by_device
                .iter()
                .filter(|p| p
                    .best
                    .as_ref()
                    .is_some_and(|b| b.met && b.point.counts.tp > 0))
                .count()
        );
        for (model, device, t) in &per_device {
            assert_eq!(model, "m");
            assert_eq!(
                Some(*t),
                m.threshold.as_ref().unwrap().apply_value_on(device),
                "{device}"
            );
        }
        assert_eq!(r.device_thresholds(Some(&["M".into()])), per_device);
        assert!(r.device_thresholds(Some(&["other".into()])).is_empty());

        // Another objective: F2 favors the lower threshold that finds everything.
        let mut m2 = m.clone();
        m2.threshold_objective = Objective::F2;
        m2.finish();
        let b2 = m2.threshold.as_ref().unwrap().best.as_ref().unwrap();
        assert_eq!(b2.objective, "f2");
        assert!(b2.point.recall.unwrap() >= best.point.recall.unwrap());

        // Results of older versions (no sweep, no advice fields) still load.
        let mut v = serde_json::to_value(&m).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("threshold");
        o.remove("threshold_objective");
        for d in o["devices"].as_array_mut().unwrap() {
            d["run"]["accuracy"]
                .as_object_mut()
                .unwrap()
                .remove("sweep");
        }
        let mut old: ModelResult = serde_json::from_value(v).unwrap();
        assert!(old.threshold.is_none());
        assert_eq!(old.threshold_objective, Objective::F1);
        old.finish();
        assert!(old.threshold.is_none());
        // A round trip keeps everything.
        let back: ModelResult = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }
}
