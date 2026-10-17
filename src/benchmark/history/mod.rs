//! Run history of the benchmark: `benchmark-history.json` beside `benchmark.json`.
//!
//! `benchmark.json` keeps one result per model (a re-run replaces it). The history keeps a
//! small snapshot of every finished model run ([`HistoryEntry`]: settings, hardware, runtimes,
//! per-device speed and accuracy, the threshold advice), the last [`MAX_ENTRIES_PER_MODEL`] per
//! model plus one pinned baseline, so the page can say what changed since the previous run and
//! whether that change is real ([`compare`]). A model with no history is seeded from its
//! `benchmark.json` result ([`seed_entries`]), so the first run after an upgrade already has a
//! baseline. Failed and cancelled models are never recorded.
//!
//! [`trust`] grades how much a result can be trusted, [`view`] adds the history-derived fields
//! to `GET /v1/benchmark`. All of it is pure except the file functions here.

pub mod compare;
pub mod trust;
pub mod view;

pub use compare::{Changes, MetricChange, Verdict, comparable, compare, compare_with_history};
pub use trust::{Trust, TrustLevel, pareto};

use super::images::GroundTruth;
use super::report::{
    BenchmarkResults, DeviceResult, HardwareSummary, ModelResult, RuntimeVersions,
};
use crate::sync::LockExt;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// File name of the history, beside `benchmark.json`.
pub(crate) const HISTORY_FILE: &str = "benchmark-history.json";
/// Unpinned entries kept per model (the oldest go first); a pinned baseline does not count.
pub const MAX_ENTRIES_PER_MODEL: usize = 20;
/// Format version written into the file.
const FORMAT_VERSION: u32 = 1;

/// Serializes every read-modify-write of a history file in this process (a benchmark run
/// appending while the page pins or deletes).
static HISTORY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `benchmark-history.json` beside the results file `results_file`.
pub fn history_path(results_file: &Path) -> PathBuf {
    results_file
        .parent()
        .map(|d| d.join(HISTORY_FILE))
        .unwrap_or_else(|| PathBuf::from(HISTORY_FILE))
}

/// A new run id: the current time, RFC 3339 UTC with milliseconds (two runs started in the
/// same second still differ).
pub fn new_run_id() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Where a run's accuracy labels came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Labels {
    /// Human ground truth.
    Human,
    /// A reference model's detections (pseudo ground truth).
    Reference,
    /// Some datasets of each.
    Mixed,
    /// No labels: no accuracy.
    #[default]
    None,
}

/// The settings a run used (what makes two runs comparable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RunSettings {
    #[serde(default)]
    pub datasets: Vec<String>,
    #[serde(default)]
    pub images: usize,
    #[serde(default)]
    pub repeat: usize,
    #[serde(default)]
    pub warmup: usize,
    /// Threshold objective (`f1`, `precision:0.9`, ...).
    #[serde(default)]
    pub objective: String,
    /// Accuracy metric of the grades (`ap50`, `roc_auc`).
    #[serde(default)]
    pub metric: String,
    /// Pseudo-ground-truth model name, when a dataset used one.
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub labels: Labels,
}

/// One device of a history entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DeviceSnap {
    pub device: String,
    #[serde(default)]
    pub label: String,
    /// `openvino` or `onnxruntime`.
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub total_p50_ms: Option<f64>,
    #[serde(default)]
    pub total_p95_ms: Option<f64>,
    #[serde(default)]
    pub infer_p50_ms: Option<f64>,
    #[serde(default)]
    pub fps: Option<f64>,
    #[serde(default)]
    pub ap50: Option<f64>,
    #[serde(default)]
    pub ap50_95: Option<f64>,
    #[serde(default)]
    pub roc_auc: Option<f64>,
    /// At the run's configured confidence threshold.
    #[serde(default)]
    pub precision: Option<f64>,
    #[serde(default)]
    pub recall: Option<f64>,
    #[serde(default)]
    pub f1: Option<f64>,
    #[serde(default)]
    pub tp: Option<usize>,
    #[serde(default)]
    pub fp: Option<usize>,
    #[serde(default, rename = "fn")]
    pub fn_: Option<usize>,
    /// Scored ground-truth objects (tp + fn).
    #[serde(default)]
    pub ground_truth: Option<usize>,
    /// Frame samples of the ROC AUC (positives + negatives).
    #[serde(default)]
    pub frames: Option<usize>,
    /// Labels were a reference model's.
    #[serde(default)]
    pub relative: bool,
    /// Overall grade letter.
    #[serde(default)]
    pub grade: Option<String>,
    #[serde(default)]
    pub fell_back: bool,
    #[serde(default)]
    pub recommended: bool,
    #[serde(default)]
    pub error: Option<String>,
}

impl DeviceSnap {
    /// The device ran (on itself) and has timings.
    pub fn ran(&self) -> bool {
        self.error.is_none() && !self.fell_back && self.total_p50_ms.is_some()
    }
}

/// The threshold advice of a history entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdSnap {
    pub objective: String,
    /// The run's configured confidence threshold.
    pub configured: f32,
    /// The best confidence threshold (None when the objective has no pick).
    #[serde(default)]
    pub best: Option<f32>,
    /// The near-optimal range the best threshold was taken from.
    #[serde(default)]
    pub plateau: Option<[f32; 2]>,
    /// Objective score of the exact optimum.
    #[serde(default)]
    pub exact_score: Option<f64>,
    /// Device whose curves the advice used.
    #[serde(default)]
    pub device: String,
}

/// One finished model run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub run_id: String,
    /// RFC 3339 time the model's sweep started.
    #[serde(default)]
    pub timestamp: String,
    pub model: String,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub settings: RunSettings,
    /// `os/arch/cpu/gpus` (GPUs by index and name).
    #[serde(default)]
    pub hardware_key: String,
    /// The facts behind `hardware_key` (comparability uses them like `benchmark.json` does).
    #[serde(default)]
    pub hardware: HardwareSummary,
    #[serde(default)]
    pub runtimes: RuntimeVersions,
    #[serde(default)]
    pub devices: Vec<DeviceSnap>,
    #[serde(default)]
    pub threshold: Option<ThresholdSnap>,
}

/// `openvino` / `onnxruntime` of a device spec (`openvino:gpu`, `ort:cuda:0`, legacy `GPU`).
pub fn runtime_of(spec: &str) -> &'static str {
    let s = spec.trim().to_ascii_lowercase();
    if s.starts_with("ort:") || s.starts_with("onnxruntime") {
        "onnxruntime"
    } else {
        "openvino"
    }
}

/// The run key of a result: its `run_id`, else (older files) its start time, else `unknown`.
pub fn run_key(m: &ModelResult) -> String {
    m.run_id
        .clone()
        .filter(|r| !r.trim().is_empty())
        .or_else(|| (!m.timestamp.trim().is_empty()).then(|| m.timestamp.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// `os/arch/cpu/gpu, gpu` with the GPUs by index and name (VRAM / CUDA facts dropped).
pub fn hardware_key(h: &HardwareSummary) -> String {
    let gpus: Vec<&str> = h.gpus.iter().map(|g| HardwareSummary::gpu_key(g)).collect();
    format!("{}/{}/{}/{}", h.os, h.arch, h.cpu.trim(), gpus.join(", "))
}

/// A model finished and has something worth recording: no model-level error and at least one
/// device that ran.
pub fn recordable(m: &ModelResult) -> bool {
    m.error.is_none() && m.devices.iter().any(|d| d.run.is_some())
}

fn device_snap(m: &ModelResult, d: &DeviceResult) -> DeviceSnap {
    let mut s = DeviceSnap {
        device: d.device.clone(),
        label: d.label.clone(),
        runtime: runtime_of(&d.device).to_string(),
        recommended: m.recommended.as_deref() == Some(d.device.as_str()),
        error: d.error.clone(),
        ..Default::default()
    };
    let Some(run) = &d.run else { return s };
    s.fell_back = run.fell_back;
    if run.fell_back {
        s.error = Some(format!(
            "ran on {} instead of {}",
            if run.spec.is_empty() {
                &run.device
            } else {
                &run.spec
            },
            d.device
        ));
    }
    s.total_p50_ms = Some(run.stages_ms.total.p50);
    s.total_p95_ms = Some(run.stages_ms.total.p95);
    s.infer_p50_ms = Some(run.stages_ms.infer.p50);
    s.fps = Some(run.throughput_fps);
    s.grade = run.grades.map(|g| g.overall.letter().to_string());
    if let Some(a) = &run.accuracy {
        let o = &a.overall;
        s.ap50 = o.ap50;
        s.ap50_95 = o.ap50_95;
        s.roc_auc = a.roc_auc();
        s.precision = o.precision;
        s.recall = o.recall;
        s.f1 = o.f1;
        s.tp = Some(o.counts.tp);
        s.fp = Some(o.counts.fp);
        s.fn_ = Some(o.counts.fn_);
        s.ground_truth = Some(o.counts.tp + o.counts.fn_);
        s.frames = a
            .frame_roc
            .as_ref()
            .map(|r| r.positives + r.negatives)
            .filter(|n| *n > 0);
        s.relative = a.relative;
    }
    s
}

/// The reference model of "relative to <model> on <device>[, ...], not ground truth".
fn reference_of_desc(desc: &str) -> Option<String> {
    let rest = desc.strip_prefix("relative to ")?;
    let rest = rest.strip_suffix(", not ground truth").unwrap_or(rest);
    let first = rest.split(", ").next()?;
    let name = first.split_once(" on ").map_or(first, |(m, _)| m).trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Labels and reference model of `m`: from the datasets' ground truth in `sets` when they are
/// recorded, else from the accuracy reports.
fn labels_of(m: &ModelResult, sets: &[super::report::ImageSetInfo]) -> (Labels, Option<String>) {
    let (mut human, mut pseudo, mut reference) = (false, false, None);
    let mut known = false;
    for id in &m.datasets {
        let Some(s) = sets.iter().find(|s| s.id == *id) else {
            continue;
        };
        known = true;
        match &s.ground_truth {
            g if g.is_real() => human = true,
            GroundTruth::Pseudo(r) => {
                pseudo = true;
                if reference.is_none() {
                    reference = reference_of_desc(&format!("relative to {r}, not ground truth"));
                }
            }
            _ => {}
        }
    }
    if !known {
        for a in m
            .devices
            .iter()
            .filter_map(|d| d.run.as_ref()?.accuracy.as_ref())
        {
            if a.relative {
                pseudo = true;
                if reference.is_none() {
                    reference = reference_of_desc(&a.ground_truth);
                }
            } else {
                human = true;
            }
        }
    }
    let labels = match (human, pseudo) {
        (true, true) => Labels::Mixed,
        (true, false) => Labels::Human,
        (false, true) => Labels::Reference,
        (false, false) => Labels::None,
    };
    (labels, reference)
}

impl From<&ModelResult> for HistoryEntry {
    /// The snapshot of `m` alone: no hardware or runtime facts, labels from the accuracy
    /// reports (see [`HistoryEntry::of`] for the full one).
    fn from(m: &ModelResult) -> Self {
        let (labels, reference) = labels_of(m, &[]);
        let metric = m
            .devices
            .iter()
            .find_map(|d| d.run.as_ref()?.grades)
            .map_or("ap50", |g| g.metric.as_str());
        Self {
            run_id: run_key(m),
            timestamp: if m.timestamp.is_empty() {
                run_key(m)
            } else {
                m.timestamp.clone()
            },
            model: m.model.clone(),
            pinned: false,
            settings: RunSettings {
                datasets: m.datasets.clone(),
                images: m.images,
                repeat: m.repeat,
                warmup: m.warmup,
                objective: m.threshold_objective.to_string(),
                metric: metric.to_string(),
                reference,
                labels,
            },
            hardware_key: String::new(),
            hardware: HardwareSummary::default(),
            runtimes: RuntimeVersions::default(),
            devices: m.devices.iter().map(|d| device_snap(m, d)).collect(),
            threshold: m.threshold.as_ref().map(|a| ThresholdSnap {
                objective: a.objective.clone(),
                configured: a.configured.threshold,
                best: a.best.as_ref().map(|b| b.threshold),
                plateau: a.best.as_ref().and_then(|b| b.plateau),
                exact_score: a.best.as_ref().and_then(|b| b.exact_score),
                device: a.device.clone(),
            }),
        }
    }
}

impl HistoryEntry {
    /// The snapshot of `m` with the hardware, runtimes and datasets of `results` (the file `m`
    /// is in).
    pub fn of(m: &ModelResult, results: &BenchmarkResults) -> Self {
        let mut e = Self::from(m);
        let (labels, reference) = labels_of(m, &results.image_sets);
        e.settings.labels = labels;
        e.settings.reference = reference;
        e.hardware_key = hardware_key(&results.hardware);
        e.hardware = results.hardware.clone();
        e.runtimes = results.runtimes.clone();
        e
    }

    /// The device row `spec`.
    pub fn device(&self, spec: &str) -> Option<&DeviceSnap> {
        self.devices.iter().find(|d| d.device == spec)
    }

    /// The recommended device row, else the first that ran.
    pub fn main_device(&self) -> Option<&DeviceSnap> {
        self.devices
            .iter()
            .find(|d| d.recommended && d.ran())
            .or_else(|| self.devices.iter().find(|d| d.ran()))
    }
}

/// The entries `benchmark.json` gives a history that has none: one per recordable model.
pub fn seed_entries(results: &BenchmarkResults) -> Vec<HistoryEntry> {
    results
        .models
        .iter()
        .filter(|m| recordable(m))
        .map(|m| HistoryEntry::of(m, results))
        .collect()
}

/// The history file: per model (display name), its entries oldest first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct History {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub models: BTreeMap<String, Vec<HistoryEntry>>,
}

fn same_model(a: &str, b: &str) -> bool {
    crate::registry::normalize_name(a) == crate::registry::normalize_name(b)
}

impl History {
    /// A history seeded from `results` (one entry per recordable model).
    pub fn seeded(results: Option<&BenchmarkResults>) -> Self {
        let mut h = Self::default();
        for e in results.map(seed_entries).unwrap_or_default() {
            h.append(e);
        }
        h
    }

    /// The key of model `name` in [`Self::models`] (normalized match).
    fn key_of(&self, name: &str) -> Option<String> {
        self.models.keys().find(|k| same_model(k, name)).cloned()
    }

    /// The entries of model `name`, oldest first (empty when it has none).
    pub fn entries(&self, name: &str) -> &[HistoryEntry] {
        self.key_of(name)
            .and_then(|k| self.models.get(&k))
            .map_or(&[], Vec::as_slice)
    }

    /// The entry `run_id` of model `name`.
    pub fn find(&self, name: &str, run_id: &str) -> Option<&HistoryEntry> {
        self.entries(name).iter().find(|e| e.run_id == run_id)
    }

    /// Add `entry` as the newest of its model (an entry with the same run id is replaced, its
    /// pin kept), then drop the oldest unpinned entries past [`MAX_ENTRIES_PER_MODEL`].
    pub fn append(&mut self, mut entry: HistoryEntry) {
        let key = self
            .key_of(&entry.model)
            .unwrap_or_else(|| entry.model.clone());
        let list = self.models.entry(key).or_default();
        if let Some(i) = list.iter().position(|e| e.run_id == entry.run_id) {
            entry.pinned |= list.get(i).is_some_and(|e| e.pinned);
            list.remove(i);
        }
        list.push(entry);
        cap(list, MAX_ENTRIES_PER_MODEL);
    }

    /// Pin (or unpin) entry `run_id` of model `name` as its baseline; pinning unpins the
    /// model's other entries. Errors when there is no such entry.
    pub fn pin(&mut self, name: &str, run_id: &str, pinned: bool) -> Result<()> {
        let list = self
            .key_of(name)
            .and_then(|k| self.models.get_mut(&k))
            .with_context(|| format!("no benchmark history for '{name}'"))?;
        if !list.iter().any(|e| e.run_id == run_id) {
            bail!("'{name}' has no run {run_id} in the benchmark history");
        }
        for e in list.iter_mut() {
            if e.run_id == run_id {
                e.pinned = pinned;
            } else if pinned {
                e.pinned = false;
            }
        }
        // Unpinning may put the model over the cap.
        cap(list, MAX_ENTRIES_PER_MODEL);
        Ok(())
    }

    /// Delete entry `run_id` of model `name`. Errors when there is no such entry.
    pub fn delete(&mut self, name: &str, run_id: &str) -> Result<()> {
        let key = self
            .key_of(name)
            .with_context(|| format!("no benchmark history for '{name}'"))?;
        let list = self
            .models
            .get_mut(&key)
            .context("history entry vanished")?;
        let before = list.len();
        list.retain(|e| e.run_id != run_id);
        if list.len() == before {
            bail!("'{name}' has no run {run_id} in the benchmark history");
        }
        if list.is_empty() {
            self.models.remove(&key);
        }
        Ok(())
    }

    /// Delete every entry of model `name`; returns how many.
    pub fn delete_model(&mut self, name: &str) -> usize {
        self.key_of(name)
            .and_then(|k| self.models.remove(&k))
            .map_or(0, |l| l.len())
    }

    /// Delete everything; returns how many entries.
    pub fn clear(&mut self) -> usize {
        let n = self.models.values().map(Vec::len).sum();
        self.models.clear();
        n
    }

    /// The file at `path` (None when it does not exist) and warnings for entries that do not
    /// load (dropped). An unreadable or unparsable file is an error.
    pub fn load_with_warnings(path: &Path) -> Result<(Option<Self>, Vec<String>)> {
        let Some(text) = super::report::read_if_exists(path)? else {
            return Ok((None, Vec::new()));
        };
        let ctx = || format!("parsing {}", path.display());
        let v: serde_json::Value = serde_json::from_str(&text).with_context(ctx)?;
        let obj = v
            .as_object()
            .with_context(|| format!("parsing {}: not a JSON object", path.display()))?;
        let mut h = Self {
            version: obj
                .get("version")
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(FORMAT_VERSION),
            models: BTreeMap::new(),
        };
        let mut warnings = Vec::new();
        if let Some(models) = obj.get("models").and_then(serde_json::Value::as_object) {
            for (name, list) in models {
                let Some(list) = list.as_array() else {
                    warnings.push(format!(
                        "{}: dropped the history of '{name}', which is not a list",
                        path.display()
                    ));
                    continue;
                };
                let mut entries = Vec::with_capacity(list.len());
                for (i, e) in list.iter().enumerate() {
                    match serde_json::from_value::<HistoryEntry>(e.clone()) {
                        Ok(x) => entries.push(x),
                        Err(err) => warnings.push(format!(
                            "{}: dropped history entry #{} of '{name}', which does not load: {err}",
                            path.display(),
                            i + 1
                        )),
                    }
                }
                if !entries.is_empty() {
                    h.models.insert(name.clone(), entries);
                }
            }
        }
        Ok((Some(h), warnings))
    }

    /// [`Self::load_with_warnings`], logging the warnings.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let (h, warnings) = Self::load_with_warnings(path)?;
        for w in warnings {
            tracing::warn!("{w}");
        }
        Ok(h)
    }

    /// The history at `path`, else (no file, or one that does not load: logged) one seeded
    /// in memory from `results`. Nothing is written.
    pub fn load_or_seed(path: &Path, results: Option<&BenchmarkResults>) -> Self {
        match Self::load(path) {
            Ok(Some(h)) => h,
            Ok(None) => Self::seeded(results),
            Err(e) => {
                tracing::warn!("{e:#}; showing the history seeded from the saved results");
                Self::seeded(results)
            }
        }
    }

    /// Write pretty JSON through a temporary file and a rename (readers never see half a file).
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out = self.clone();
        out.version = FORMAT_VERSION;
        let text = serde_json::to_string_pretty(&out)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }

    /// Load the file at `path` (a file that does not load is moved aside with a warning;
    /// missing: seeded from `seed`), apply `edit`, save. Serialized in this process. Returns
    /// `edit`'s value and the warnings.
    pub fn update<T>(
        path: &Path,
        seed: Option<&BenchmarkResults>,
        edit: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<(T, Vec<String>)> {
        let _held = HISTORY_LOCK.lock_recover();
        let (h, warnings) = super::report::load_or_move_aside(path, Self::load_with_warnings)?;
        let mut h = h.unwrap_or_else(|| Self::seeded(seed));
        let out = edit(&mut h)?;
        h.save(path)?;
        Ok((out, warnings))
    }
}

/// Drop the oldest unpinned entries of `list` until at most `max` unpinned remain.
fn cap(list: &mut Vec<HistoryEntry>, max: usize) {
    let mut unpinned = list.iter().filter(|e| !e.pinned).count();
    while unpinned > max {
        let Some(i) = list.iter().position(|e| !e.pinned) else {
            break;
        };
        list.remove(i);
        unpinned -= 1;
    }
}

/// Record the finished models `models` of `merged` (just saved to `results_file`) in the
/// history beside it. `seeds` are the entries of the results file before this save
/// ([`seed_entries`] of it): a missing history file starts from all of them, and a model with
/// no history gets its seed as the baseline first. Failed models are skipped. Never fails the
/// run: returns warnings (also logged).
pub fn record(
    results_file: &Path,
    seeds: &[HistoryEntry],
    merged: &BenchmarkResults,
    models: &[String],
) -> Vec<String> {
    let entries: Vec<HistoryEntry> = models
        .iter()
        .filter_map(|n| merged.find(n))
        .filter(|m| recordable(m))
        .map(|m| HistoryEntry::of(m, merged))
        .collect();
    if entries.is_empty() {
        return Vec::new();
    }
    let path = history_path(results_file);
    let fresh = !path.exists();
    let outcome = History::update(&path, None, |h| {
        for s in seeds {
            let wanted = fresh || entries.iter().any(|e| same_model(&e.model, &s.model));
            if wanted && h.entries(&s.model).is_empty() {
                h.append(s.clone());
            }
        }
        for e in &entries {
            h.append(e.clone());
        }
        Ok(())
    });
    match outcome {
        Ok(((), warnings)) => warnings,
        Err(e) => {
            let w = format!(
                "The benchmark history ({}) was not updated: {e:#}. The results themselves were \
                 saved; check that the folder is writable, the next run records again.",
                path.display()
            );
            tracing::warn!("{w}");
            vec![w]
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
