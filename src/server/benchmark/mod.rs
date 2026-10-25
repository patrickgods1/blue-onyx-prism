//! Benchmark page (`/benchmark`) and API:
//! - `GET /v1/benchmark`: the current/last run (state, progress, ETA, partial results), the
//!   saved results (`benchmark.json`, without per-image details), the cross-model ranking, the
//!   datasets and the config's per-model devices;
//! - `POST /v1/benchmark`: start a run; `POST /v1/benchmark/cancel`;
//! - `POST /v1/benchmark/apply`: write per-model devices into the config;
//! - `POST /v1/benchmark/apply-threshold`: write per-model confidence thresholds (or one device's
//!   confidence profile) into the config;
//! - `POST /v1/benchmark/apply-profiles`: write per-device confidence profiles and per-model
//!   devices into the config in one save (one revision, one Undo);
//! - `GET /v1/benchmark/threshold-search`: what can be searched (models, devices, datasets,
//!   tags, classes); `POST /v1/benchmark/threshold-search`: the exact confidence-threshold
//!   search over the stored predictions (no inference; [`crate::benchmark::search`]);
//! - `POST /v1/benchmark/settings`: save the form as the `benchmark` config defaults;
//! - `GET /v1/benchmark/images?model=&device=`: per-image drill-down (ground truth and
//!   predictions); `GET /v1/benchmark/image?set=&file=`: the image file itself (only files
//!   listed in the saved results);
//! - `GET /v1/benchmark/history[?model=]`, `GET /v1/benchmark/history/compare?model=&a=&b=`,
//!   `POST /v1/benchmark/history/pin`, `POST /v1/benchmark/history/delete`: the run history
//!   (`benchmark-history.json`, [`crate::benchmark::history`]).
//!
//! The run itself is [`crate::benchmark::service`]. This module keeps the page, the status
//! (`GET /v1/benchmark`) and the shared form/error helpers; `run` holds the start, settings,
//! cancel and drill-down endpoints, `apply` the device/threshold apply and threshold search,
//! `history` the run history endpoints.

mod apply;
mod history;
mod profiles;
mod run;

use apply::BLUE_IRIS_THRESHOLD_NOTE;

pub(super) use apply::{apply, apply_profiles, apply_threshold, search_options, threshold_search};
pub(super) use history::{history_compare, history_delete, history_get, history_pin};
pub(super) use run::{cancel, image_file, images_detail, settings, start};

use super::{AppState, DevicesSnapshot, devices_snapshot, global_device_label, render};
use crate::benchmark::images::{self, ImageSet};
use crate::benchmark::service::default_models;
use crate::benchmark::{BenchmarkResults, configured_device, results_path};
use crate::config::Config;
use crate::sync::LockExt;
use askama::Template;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::warn;

/// Shown on the page and in the API: benchmarking competes with live requests.
pub(super) const LIVE_WARNING: &str = "Benchmarking runs on the same CPU/GPU as live detections: \
     Blue Iris requests are slower while it runs, and its numbers are noisier when the server is \
     busy. Run it when the cameras are quiet.";

struct ModelOption {
    name: String,
    family: String,
    enabled: bool,
    exists: bool,
    checked: bool,
}

struct DeviceOptionRow {
    spec: String,
    label: String,
    runnable: bool,
    checked: bool,
    reason: Option<String>,
}

/// One dataset row of the form.
struct DatasetRow {
    /// Value posted as `dataset` (`coco-cctv`, `sample`, `dir:/path`).
    value: String,
    title: String,
    /// "120 images · 45 MB · CC BY 4.0".
    detail: String,
    /// `installed`, `partial`, `available`, `embedded`, `folder`, `missing`, `unavailable`.
    state: &'static str,
    state_text: String,
    checked: bool,
    /// Resource id to download (built-in sets not fully installed).
    download_id: Option<String>,
    /// A user folder (removable from the defaults).
    folder: bool,
    selectable: bool,
}

#[derive(Template)]
#[template(path = "benchmark.html")]
struct BenchmarkTemplate {
    nav: &'static str,
    version: &'static str,
    models: Vec<ModelOption>,
    devices: Vec<DeviceOptionRow>,
    datasets: Vec<DatasetRow>,
    reference_models: Vec<(String, bool)>,
    repeat: usize,
    warmup: usize,
    /// `benchmark.low_priority`: the "Run at low priority" checkbox.
    low_priority: bool,
    max_images: usize,
    accuracy_weight: String,
    warning: &'static str,
    force_cpu: bool,
    results_file: String,
    speed_thresholds: String,
    accuracy_thresholds: String,
    auc_thresholds: String,
    /// (value, label, selected) of the threshold objective select.
    objectives: Vec<(String, String, bool)>,
    /// (value, label, selected) of the accuracy metric select.
    accuracy_metrics: Vec<(String, String, bool)>,
    blue_iris_note: &'static str,
    /// The benchmark goal's plain name ("Balanced", "Miss fewer objects", "Fewer false
    /// alerts", "Fastest / lightest", "Custom"; [`profiles::goal_label`]).
    goal_label: String,
}

/// Threshold objective choices; the configured one is added when it is not among them.
fn objective_options(
    current: crate::benchmark::threshold::Objective,
) -> Vec<(String, String, bool)> {
    use crate::benchmark::threshold::Objective;
    let mut list = vec![
        (Objective::F1, "F1: balance precision and recall"),
        (Objective::F2, "F2: favor recall (fewer missed objects)"),
        (
            Objective::Precision(0.8),
            "Precision \u{2265} 80%, highest recall",
        ),
        (
            Objective::Precision(0.9),
            "Precision \u{2265} 90%, highest recall (fewer false alerts)",
        ),
        (
            Objective::Precision(0.95),
            "Precision \u{2265} 95%, highest recall",
        ),
        (
            Objective::Youden,
            "Youden's J: frame alert rate minus false-alert rate",
        ),
        (
            Objective::Fpr(0.05),
            "Frame false alerts \u{2264} 5%, highest alert rate",
        ),
        (
            Objective::Fpr(0.1),
            "Frame false alerts \u{2264} 10%, highest alert rate",
        ),
    ]
    .into_iter()
    .map(|(o, l)| (o.to_string(), l.to_string(), o == current))
    .collect::<Vec<_>>();
    if !list.iter().any(|o| o.2) {
        list.push((current.to_string(), current.describe(), true));
    }
    list
}

/// Accuracy metric choices.
fn metric_options(current: crate::benchmark::grade::AccuracyMetric) -> Vec<(String, String, bool)> {
    use crate::benchmark::grade::AccuracyMetric;
    AccuracyMetric::ALL
        .iter()
        .map(|m| {
            let label = match m {
                AccuracyMetric::Ap50 => "AP50: box accuracy (AP@0.5, small-object recall)",
                AccuracyMetric::RocAuc => {
                    "ROC AUC: frame-level alerting (does a frame with an object score higher?)"
                }
            };
            (m.as_str().to_string(), label.to_string(), *m == current)
        })
        .collect()
}

fn model_options(c: &Config) -> Vec<ModelOption> {
    let defaults = if c.benchmark.models.is_empty() {
        default_models(c)
    } else {
        c.benchmark.models.clone()
    };
    let key = crate::registry::normalize_name;
    c.models
        .iter()
        .map(|m| {
            let exists = c.data_path(&m.path).is_file();
            let name = m.effective_name();
            ModelOption {
                checked: exists && defaults.iter().any(|d| key(d) == key(&name)),
                name,
                family: m.family.to_string(),
                enabled: m.enabled,
                exists,
            }
        })
        .collect()
}

/// Every device option, runnable first (downloadable ones count as not runnable here: the
/// benchmark only measures what is installed).
fn device_rows(snap: &DevicesSnapshot, c: &Config) -> Vec<DeviceOptionRow> {
    let chosen: Vec<String> = c
        .benchmark
        .devices
        .iter()
        .filter_map(|d| crate::backend::spec::parse(d).ok().map(|s| s.to_string()))
        .collect();
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    ok.into_iter()
        .chain(bad)
        .map(|o| {
            let spec = o.spec.to_string();
            DeviceOptionRow {
                checked: o.runnable && (chosen.is_empty() || chosen.contains(&spec)),
                spec,
                label: o.label.clone(),
                runnable: o.runnable,
                reason: (!o.runnable).then(|| match &o.download {
                    Some(d) => format!("not installed ({})", d.summary),
                    None => o.reason.clone().unwrap_or_else(|| "not runnable".into()),
                }),
            }
        })
        .collect()
}

/// Download state of a built-in set from the manager (busy) or the disk.
fn dataset_rows(state: &AppState, c: &Config) -> Vec<DatasetRow> {
    let selected: Vec<String> = c.benchmark.datasets.iter().map(|d| d.label()).collect();
    let is_sel = |id: &str| selected.iter().any(|s| s.eq_ignore_ascii_case(id));
    let root = c.data_root();
    let rows_by_id: HashMap<String, crate::resources::status::ResourceRow> =
        crate::resources::status::rows(state.resources.as_ref(), c)
            .into_iter()
            .filter(|r| r.kind == "bench-images")
            .map(|r| (r.id.to_string(), r))
            .collect();
    let mut out = Vec::new();
    for m in images::builtin_manifests() {
        let (set, missing) = ImageSet::from_manifest(m, &images::builtin_dir(&root, &m.id));
        let rid = format!("{}{}", crate::resources::catalog::BENCH_ID_PREFIX, m.id);
        let busy = rows_by_id.get(&rid).filter(|r| r.busy);
        let (st, text) = match busy {
            Some(r) => ("downloading", r.state_text.clone()),
            None if missing == 0 => ("installed", "installed".to_string()),
            None if !set.images.is_empty() => (
                "partial",
                format!("{} of {} images", set.images.len(), m.images.len()),
            ),
            None => ("available", "not downloaded".to_string()),
        };
        out.push(DatasetRow {
            value: m.id.clone(),
            title: m.title.clone(),
            detail: format!(
                "{} images \u{b7} {} \u{b7} {} \u{b7} {}",
                m.images.len(),
                crate::resources::catalog::format_size(m.size()),
                if m.has_ground_truth() {
                    "ground truth"
                } else {
                    "no ground truth"
                },
                m.license
            ),
            state: st,
            state_text: text,
            checked: is_sel(&m.id),
            download_id: (missing > 0 && busy.is_none()).then_some(rid),
            folder: false,
            selectable: true,
        });
    }
    for id in images::KNOWN_SET_IDS {
        if images::builtin_manifest(id).is_none() {
            out.push(DatasetRow {
                value: id.to_string(),
                title: id.to_string(),
                detail: "curated set; not included in this build yet".into(),
                state: "unavailable",
                state_text: "not in this build".into(),
                checked: is_sel(id),
                download_id: None,
                folder: false,
                selectable: true,
            });
        }
    }
    out.push(DatasetRow {
        value: images::SAMPLE_ID.into(),
        title: "Embedded sample".into(),
        detail: "1 image (dog, bicycle, car) \u{b7} works offline \u{b7} no ground truth".into(),
        state: "embedded",
        state_text: "always available".into(),
        checked: is_sel(images::SAMPLE_ID),
        download_id: None,
        folder: false,
        selectable: true,
    });
    for d in &c.benchmark.datasets {
        if let Some((dir, _, _)) = d.dir() {
            let abs = c.data_path(&dir);
            let ok = abs.is_dir();
            out.push(DatasetRow {
                value: d.label(),
                title: d.label(),
                detail: abs.display().to_string(),
                state: if ok { "folder" } else { "missing" },
                state_text: if ok {
                    "folder on the server".into()
                } else {
                    "directory not found".into()
                },
                checked: true,
                download_id: None,
                folder: true,
                selectable: ok,
            });
        }
    }
    out
}

/// `GET /benchmark`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    super::blocking(move || {
        let snap = devices_snapshot(&state);
        let cfg = state.config_read().clone();
        let b = &cfg.benchmark;
        use crate::benchmark::grade::{
            ACCURACY_THRESHOLDS as A, ROC_AUC_THRESHOLDS as U, SPEED_THRESHOLDS_MS as S,
        };
        render(&BenchmarkTemplate {
            nav: "benchmark",
            version: crate::VERSION,
            models: model_options(&cfg),
            devices: device_rows(&snap, &cfg),
            datasets: dataset_rows(&state, &cfg),
            reference_models: cfg
                .models
                .iter()
                .filter(|m| cfg.data_path(&m.path).is_file())
                .map(|m| {
                    let n = m.effective_name();
                    let sel = b.reference_model.as_deref() == Some(n.as_str());
                    (n, sel)
                })
                .collect(),
            repeat: b.repeat_per_image.max(1),
            warmup: b.warmup,
            low_priority: b.low_priority,
            max_images: b.max_images_per_dataset,
            accuracy_weight: format!("{:.2}", b.weights.accuracy_share()),
            warning: LIVE_WARNING,
            force_cpu: cfg.force_cpu,
            results_file: results_path(&state.config_path).display().to_string(),
            speed_thresholds: format!("A < {} ms, B < {}, C < {}, D < {}", S[0], S[1], S[2], S[3]),
            accuracy_thresholds: format!(
                "A \u{2265} {:.2}, B \u{2265} {:.2}, C \u{2265} {:.2}, D \u{2265} {:.2}",
                A[0], A[1], A[2], A[3]
            ),
            auc_thresholds: format!(
                "A \u{2265} {:.2}, B \u{2265} {:.2}, C \u{2265} {:.2}, D \u{2265} {:.2}",
                U[0], U[1], U[2], U[3]
            ),
            objectives: objective_options(b.threshold_objective),
            accuracy_metrics: metric_options(b.accuracy_metric),
            blue_iris_note: BLUE_IRIS_THRESHOLD_NOTE,
            goal_label: profiles::goal_label(b).to_string(),
        })
    })
    .await
}

/// Normalized model name -> spec of the device it is loaded on (this generation).
pub(super) fn loaded_devices(state: &AppState) -> HashMap<String, String> {
    state
        .registry
        .workers()
        .iter()
        .filter_map(|w| {
            let d = w.device.read().ok()?.clone()?;
            Some((crate::registry::normalize_name(&w.name), d.spec))
        })
        .collect()
}

/// The config's view for the page: global device, and per model its `device`, the device it
/// runs on (loaded, else what the config resolves to) and its execution provider.
fn config_view(state: &AppState, snap: &DevicesSnapshot) -> serde_json::Value {
    super::sync_config(state);
    let (revision, cfg) = state.config.snapshot();
    let pending = crate::config_merge::pending(&state.running(), &cfg);
    let thr = crate::config::round_threshold_f64;
    let loaded = loaded_devices(state);
    let models: Vec<serde_json::Value> = cfg
        .models
        .iter()
        .map(|m| {
            let name = m.effective_name();
            let key = crate::registry::normalize_name(&name);
            let sel = snap.model_selection(state, &cfg, m);
            let configured = configured_device(&cfg, m, &sel);
            let loaded_spec = loaded.get(&key).cloned().filter(|_| m.enabled);
            // Where the running threshold comes from: the worker's (its loaded device's
            // profile, else the fallback), else the config's view for the loaded device.
            let source = m
                .enabled
                .then(|| state.registry.by_name(&name))
                .flatten()
                .map(|w| w.effective_threshold().1)
                .unwrap_or_else(|| {
                    m.threshold_on(loaded_spec.as_deref(), cfg.confidence_threshold)
                        .1
                });
            let effective = loaded_spec.or_else(|| configured.clone());
            let provider = m
                .enabled
                .then(|| state.registry.by_name(&name))
                .flatten()
                .map(|w| w.execution_provider());
            let reasons = pending.for_model(&name);
            let running = super::live::running_view(state, &name);
            serde_json::json!({
                "name": name,
                "enabled": m.enabled,
                "configuredDevice": configured,
                "confidenceThreshold": thr(m.confidence_threshold.unwrap_or(cfg.confidence_threshold)),
                "ownThreshold": m.confidence_threshold.is_some(),
                "deviceThresholds": m
                    .device_thresholds
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::json!(thr(*v))))
                    .collect::<serde_json::Map<_, _>>(),
                "thresholdSource": source,
                "device": m.device,
                "effective": effective,
                "provider": provider,
                "runningThreshold": running.threshold,
                "runningDevice": running.device,
                "restartNeeded": !reasons.is_empty(),
                "pending": reasons,
            })
        })
        .collect();
    serde_json::json!({
        "revision": revision,
        "restartNeeded": pending.restart_needed(),
        "forceCpu": cfg.force_cpu,
        "globalDevice": cfg.device,
        "globalLabel": global_device_label(&cfg, snap),
        "confidenceThreshold": thr(cfg.confidence_threshold),
        "models": models,
        "benchmark": cfg.benchmark,
    })
}

/// What the config says now about each model (threshold, device, running threshold): the saved
/// results are shown against it ([`crate::benchmark::current`]).
fn models_now(
    state: &AppState,
    cfg: &Config,
    snap: &DevicesSnapshot,
) -> HashMap<String, crate::benchmark::current::ModelNow> {
    // The threshold the served model's worker uses now (live settings: the profile of the
    // device it loaded on, else the fallback); a model without a worker (disabled) runs with
    // nothing.
    let running = |name: &str| {
        state
            .registry
            .by_name(name)
            .map(|w| w.effective_threshold().0)
    };
    crate::benchmark::current::from_config(
        cfg,
        &|m| configured_device(cfg, m, &snap.model_selection(state, cfg, m)),
        &running,
    )
}

/// (modified, length) of a file; None when it does not exist.
pub(super) fn file_stamp(p: &std::path::Path) -> Option<(Option<std::time::SystemTime>, u64)> {
    std::fs::metadata(p)
        .ok()
        .map(|m| (m.modified().ok(), m.len()))
}

/// `results`, `ranking`, `points` and `latest_run_id` of `GET /v1/benchmark`: the saved results
/// rebased on the config now, with the history-derived fields
/// ([`crate::benchmark::history::view`]).
#[derive(Debug, Clone, Default)]
struct SavedView {
    results: serde_json::Value,
    ranking: serde_json::Value,
    points: serde_json::Value,
    latest_run_id: Option<String>,
    /// The measured part of `profiles` ([`profiles::facts`]).
    facts: Vec<profiles::ModelFacts>,
    /// The hardware the results were measured on (`otherHardware`).
    hardware: Option<crate::benchmark::report::HardwareSummary>,
}

/// The last [`SavedView`] and what it was computed from: the results, stored-predictions and
/// history files' stamps and the config revision with what it says about each model. Polling
/// reads none of the files (tens of MB) again until one of them or the config changes.
type SavedCache = Option<(String, Arc<SavedView>)>;
static SAVED_CACHE: std::sync::Mutex<SavedCache> = std::sync::Mutex::new(None);

/// How many times [`saved_view`] computed (tests: polling does not recompute).
static SAVED_COMPUTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn saved_view(
    state: &AppState,
    revision: u64,
    now: &HashMap<String, crate::benchmark::current::ModelNow>,
) -> Arc<SavedView> {
    use crate::benchmark::history::{History, history_path, view};
    use crate::benchmark::search::{StoredPreds, preds_path};
    let path = results_path(&state.config_path);
    let preds = preds_path(&state.config_path);
    let hist = history_path(&path);
    let mut sorted: Vec<_> = now.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let key = format!(
        "{}|{:?}|{:?}|{:?}|{revision}|{sorted:?}",
        path.display(),
        file_stamp(&path),
        file_stamp(&preds),
        file_stamp(&hist)
    );
    let mut cache = SAVED_CACHE.lock_recover();
    if let Some((k, v)) = cache.as_ref()
        && *k == key
    {
        return v.clone();
    }
    SAVED_COMPUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let view = match BenchmarkResults::load_or_warn(&path) {
        Some(mut r) => {
            let ranking = serde_json::to_value(r.ranking()).unwrap_or_default();
            let points = serde_json::to_value(r.points()).unwrap_or_default();
            let stored = StoredPreds::load_cached(&preds)
                .map_err(|e| warn!("{e:#}"))
                .ok()
                .flatten();
            crate::benchmark::current::rebase(&mut r, stored.as_deref(), now);
            let facts = profiles::facts(&r);
            // No history file yet: seeded in memory (written by the next run, pin or delete).
            let history = History::load_or_seed(&hist, Some(&r));
            let mut results = serde_json::to_value(r.summary()).unwrap_or_default();
            view::annotate(&mut results, &r, &history);
            SavedView {
                results,
                ranking,
                points,
                latest_run_id: view::latest_run_id(&r),
                facts,
                hardware: Some(r.hardware.clone()),
            }
        }
        None => SavedView::default(),
    };
    let view = Arc::new(view);
    *cache = Some((key, view.clone()));
    view
}

/// `GET /v1/benchmark`: the current/last run (state, progress, partial results), the saved
/// results (without per-image details), the ranking and the config's per-model devices. Results
/// are shown against the config now: each threshold advice carries `now` (the effective
/// threshold in the config, the counts there, exact from the stored predictions) and each
/// model's `configured` is the device the config resolves to now. `profiles` lists per model and
/// device the measured best threshold and the saved confidence profile ([`profiles::profiles`]),
/// `otherHardware` whether the results were measured on other GPUs, `goal` the benchmark goal.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    super::blocking(move || {
        let snap = devices_snapshot(&state);
        let path = results_path(&state.config_path);
        let config = config_view(&state, &snap);
        let (revision, cfg) = state.config.snapshot();
        let now = models_now(&state, &cfg, &snap);
        let saved = saved_view(&state, revision, &now);
        let loaded = loaded_devices(&state);
        let profile_rows = profiles::profiles(
            &saved.facts,
            &cfg,
            &|name| loaded.get(&crate::registry::normalize_name(name)).cloned(),
            &|m| configured_device(&cfg, m, &snap.model_selection(&state, &cfg, m)),
        );
        let other_hardware = saved
            .hardware
            .as_ref()
            .is_some_and(|h| !h.same_gpus(&snap.hardware));
        let mut run = state.benchmark.status();
        for m in &mut run.models {
            if let Some(n) = now.get(&crate::registry::normalize_name(&m.model)) {
                crate::benchmark::current::rebase_model(m, &[], None, n);
            }
        }
        Json(serde_json::json!({
            "success": true,
            "state": run.state,
            "running": run.running,
            "run": run,
            "results": saved.results,
            "ranking": saved.ranking,
            "points": saved.points,
            "latest_run_id": saved.latest_run_id,
            "noise": crate::benchmark::history::view::noise_json(),
            "resultsFile": path.display().to_string(),
            "config": config,
            "profiles": profile_rows,
            "otherHardware": other_hardware,
            "goal": {
                "label": profiles::goal_label(&cfg.benchmark),
                "objective": cfg.benchmark.threshold_objective.to_string(),
            },
            "warning": LIVE_WARNING,
        }))
    })
    .await
}

pub(super) fn json_error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message })),
    )
        .into_response()
}

pub(super) fn truthy(v: &str) -> bool {
    !matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

/// An API error answered as `{"success": false, "message": ...}` with its status (see
/// [`json_error`]); small, unlike a [`Response`], so helpers can return it in a `Result`.
#[derive(Debug)]
pub(super) struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub(super) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        json_error(self.status, self.message)
    }
}

/// The value of the first `key` field of a form (untrimmed).
pub(super) fn form_field<'f>(form: &'f [(String, String)], key: &str) -> Option<&'f str> {
    form.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The non-empty `model` fields of a form, trimmed.
pub(super) fn form_models(form: &[(String, String)]) -> Vec<String> {
    form.iter()
        .filter(|(k, _)| k == "model")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::router;
    use super::*;
    use crate::config::ModelConfig;
    use crate::metrics::{Metrics, ModelMetrics};
    use crate::registry::{ModelRegistry, RuntimeInfo};
    use crate::worker::WorkerHandle;
    use axum::body::Body;
    use axum::http::{Request, header};
    use tower::ServiceExt;

    #[test]
    fn form_helpers_and_api_errors() {
        let form: Vec<(String, String)> = [
            ("model", " a "),
            ("all", "1"),
            ("model", ""),
            ("model", "b"),
            ("all", "0"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        assert_eq!(form_models(&form), ["a", "b"]);
        assert_eq!(form_field(&form, "all"), Some("1"));
        assert_eq!(form_field(&form, "device"), None);
        // Small enough to return in a `Result` without boxing.
        assert!(std::mem::size_of::<ApiError>() <= 32);
        let r = ApiError::new(StatusCode::NOT_FOUND, "nope").into_response();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    pub(super) fn state() -> Arc<AppState> {
        let dir = std::env::temp_dir().join(format!("bo_bench_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let metrics = Metrics::new("0.0.0");
        let mm = Arc::new(ModelMetrics::new("ipcam-general", "CPU", ""));
        let workers = vec![WorkerHandle::failed(
            "ipcam-general",
            "model file not found",
            mm,
        )];
        let info = RuntimeInfo {
            openvino_version: "2026.0.0-test".into(),
            available_devices: vec!["CPU".into()],
            has_gpu: false,
        };
        let registry = ModelRegistry::from_handles(workers, Some(0), info);
        let config = Config {
            models: ["ipcam-general", "dfine-s"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: dir.join(format!("{n}.onnx")),
                    enabled: *n == "ipcam-general",
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        Arc::new(AppState::new(
            Arc::new(registry),
            Arc::new(metrics),
            config,
            dir.join("cfg.json"),
        ))
    }

    /// Saved results for the profile tests: `ipcam-general` (configured, enabled) ran on
    /// openvino:cpu (best 0.40, recommended), ort:cpu (best 0.33) and openvino:gpu (failed);
    /// `dfine-s` (configured, disabled) on ort:coreml (best 0.30); `removed` (not configured)
    /// on ort:cpu (0.25). `hardware` is where they were measured.
    pub(super) fn profile_results(
        hardware: crate::benchmark::report::HardwareSummary,
    ) -> BenchmarkResults {
        use crate::benchmark::report::test_support::{dev, failed, model, scene};
        use crate::benchmark::threshold::{DevicePick, test_advice};
        let mk = |name: &str, devs: Vec<crate::benchmark::DeviceResult>, picks: &[(&str, f32)]| {
            let mut m = model(devs, None);
            m.model = name.into();
            let mut a = test_advice(picks[0].1);
            a.device = picks[0].0.into();
            a.by_device = picks
                .iter()
                .map(|(d, t)| DevicePick {
                    device: d.to_string(),
                    best: a.best.clone().map(|mut b| {
                        b.threshold = *t;
                        b
                    }),
                })
                .collect();
            m.threshold = Some(a);
            m
        };
        BenchmarkResults::new(
            hardware,
            Default::default(),
            vec![
                mk(
                    "ipcam-general",
                    vec![
                        dev("openvino:cpu", 20.0, scene()),
                        dev("ort:cpu", 30.0, scene()),
                        failed("openvino:gpu"),
                    ],
                    &[("openvino:cpu", 0.4), ("ort:cpu", 0.33)],
                ),
                mk(
                    "dfine-s",
                    vec![dev("ort:coreml", 15.0, scene())],
                    &[("ort:coreml", 0.3)],
                ),
                mk(
                    "removed",
                    vec![dev("ort:cpu", 25.0, scene())],
                    &[("ort:cpu", 0.25)],
                ),
            ],
        )
    }

    /// `pairs` urlencoded (the `undo` list of an apply answer posted back).
    pub(super) fn form_body(pairs: &serde_json::Value) -> String {
        let enc = |s: &str| {
            s.bytes()
                .map(|b| match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                        (b as char).to_string()
                    }
                    _ => format!("%{b:02X}"),
                })
                .collect::<String>()
        };
        pairs
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                format!(
                    "{}={}",
                    enc(p[0].as_str().unwrap()),
                    enc(p[1].as_str().unwrap())
                )
            })
            .collect::<Vec<_>>()
            .join("&")
    }

    pub(super) async fn call(
        state: &Arc<AppState>,
        req: Request<Body>,
    ) -> (StatusCode, serde_json::Value) {
        let resp = router(state.clone()).oneshot(local(req)).await.unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v = serde_json::from_slice(&body)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&body).into()));
        (status, v)
    }

    /// Requests come from this machine (the admin routes refuse remote clients).
    pub(super) fn local(req: Request<Body>) -> Request<Body> {
        crate::server::test_util::from_peer(req, "127.0.0.1:50000")
    }

    pub(super) fn post(uri: &str, body: &str) -> Request<Body> {
        local(
            Request::post(uri)
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
    }

    pub(super) fn get(uri: &str) -> Request<Body> {
        local(Request::get(uri).body(Body::empty()).unwrap())
    }

    #[tokio::test]
    async fn page_lists_models_and_devices() {
        let st = state();
        let resp = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains(
            "<a href=\"/benchmark\" class=\"active\" aria-current=\"page\">Benchmark</a>"
        ));
        assert!(
            html.contains("name=\"model\" value=\"ipcam-general\""),
            "{html}"
        );
        // Files are missing: listed but not checked.
        assert!(!html.contains("value=\"ipcam-general\" checked"));
        assert!(html.contains("name=\"device\" value=\"openvino:cpu\" checked"));
        // ONNX Runtime is not in this probe: listed, disabled.
        assert!(html.contains("name=\"device\" value=\"ort:cpu\" disabled"));
        // The configured datasets are checked; built-in sets offer a download.
        assert!(
            html.contains("name=\"dataset\" value=\"coco-cctv\" checked"),
            "{html}"
        );
        assert!(html.contains("data-download=\"bench:coco-cctv\""));
        assert!(html.contains("name=\"dataset\" value=\"sample\">"));
        assert!(html.contains("competes") || html.contains("same CPU/GPU"));
        // The page script lives in `/static/benchmark.js` (CSP: no inline scripts).
        assert!(
            html.contains("<script src=\"/static/benchmark.js?v="),
            "{html}"
        );
        let js = super::super::pages::static_file("benchmark.js").expect("benchmark.js");
        assert!(js.contains("fetch(\"/v1/benchmark\""));
        // The steps: Run, Profiles, Apply, Advanced.
        for id in ["tab-run", "tab-profiles", "tab-apply", "tab-advanced"] {
            assert!(html.contains(&format!("id=\"{id}\"")), "{id}: {html}");
            assert!(html.contains(&format!("href=\"#{id}\"")), "{id}: {html}");
        }
        // The goal's plain name (the default weights are the Balanced preset).
        assert!(
            html.contains("<strong id=\"goal-label\">Balanced</strong>"),
            "{html}"
        );
        // The profiles script.
        assert!(
            html.contains("<script src=\"/static/bench-profiles.js?v="),
            "{html}"
        );
        assert!(super::super::pages::static_file("bench-profiles.js").is_some());
        // Threshold objective (config default f1, in the Profiles step, part of the run form)
        // and the threshold apply actions.
        assert!(
            html.contains("<select name=\"threshold_objective\" form=\"bench-form\""),
            "{html}"
        );
        assert!(html.contains("<option value=\"f1\" selected>"), "{html}");
        assert!(html.contains("<option value=\"precision:0.9\">"), "{html}");
        assert!(html.contains("id=\"apply-all-thr\""));
        assert!(
            html.contains("/v1/benchmark/apply-threshold")
                || js.contains("/v1/benchmark/apply-threshold")
        );
        assert!(html.contains("min_confidence") || js.contains("min_confidence"));
        // Accuracy metric (config default ap50), the frame objectives and the ROC chart code.
        assert!(
            html.contains("<select name=\"accuracy_metric\" form=\"bench-form\""),
            "{html}"
        );
        assert!(html.contains("<option value=\"ap50\" selected>"), "{html}");
        assert!(html.contains("<option value=\"roc_auc\">"), "{html}");
        assert!(html.contains("<option value=\"youden\">"), "{html}");
        assert!(html.contains("<option value=\"fpr:0.05\">"), "{html}");
        assert!(
            super::super::pages::static_file("bench-charts.js")
                .expect("bench-charts.js")
                .contains("function rocSvg")
        );
    }

    #[tokio::test]
    async fn models_card_saves_per_model_devices() {
        let st = state();
        let body = "enabled=ipcam-general&default_model=ipcam-general&device.ipcam-general=ort%3Acpu&device.dfine-s=&action=save";
        let resp = router(st.clone())
            .oneshot(post("/config/models", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let saved = Config::load(&st.config_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("ort:cpu"));
        assert_eq!(saved.models[1].device, None);
        // The select shows the saved value; "" = global.
        let resp = router(st.clone()).oneshot(get("/config")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(
            html.contains("<option value=\"ort:cpu\" selected"),
            "{html}"
        );
        // A bad device is rejected and nothing changes.
        let body = "enabled=ipcam-general&device.ipcam-general=vulkan&action=save";
        let resp = router(st.clone())
            .oneshot(post("/config/models", body))
            .await
            .unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("Models not saved"), "{html}");
        assert_eq!(Config::load(&st.config_path).unwrap(), saved);
    }

    #[tokio::test]
    async fn status_and_search_follow_the_configured_threshold_now() {
        use std::sync::atomic::Ordering;
        let st = state();
        let cfg_path = st.config_path.clone();
        let (results, preds) = crate::benchmark::current::tests::advised_fixture();
        results.save(&results_path(&cfg_path)).unwrap();
        preds
            .save(&crate::benchmark::search::preds_path(&cfg_path))
            .unwrap();
        st.config
            .update("test", |c| {
                c.models.push(ModelConfig {
                    name: Some("m".into()),
                    path: cfg_path.with_file_name("m.onnx"),
                    enabled: false,
                    ..Default::default()
                });
                Ok(())
            })
            .unwrap();
        let advice = |v: &serde_json::Value| v["results"]["models"][0]["threshold"].clone();

        // Before: the configured 0.50 (the run's), gain vs it.
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let a = advice(&v);
        assert_eq!(a["now"]["threshold"].as_f64().unwrap() as f32, 0.5);
        assert_eq!(a["now"]["source"], "run");
        assert_eq!(a["now"]["point"]["fp"], a["configured"]["fp"]);
        let rec = v["results"]["models"][0]["recommendation"]
            .as_str()
            .unwrap();
        assert!(rec.contains("at the configured 0.50"), "{rec}");
        // Polling with nothing changed does not recompute.
        let computed = SAVED_COMPUTED.load(Ordering::Relaxed);
        let (_, v2) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(advice(&v2), a);

        // Apply all thresholds: Now is the new value, P/R/F1 evaluated there exactly.
        let (s, j) = call(&st, post("/v1/benchmark/apply-threshold", "all=1")).await;
        assert_eq!(s, StatusCode::OK, "{j}");
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert!(SAVED_COMPUTED.load(Ordering::Relaxed) > computed);
        let a = advice(&v);
        let best = a["best"]["threshold"].as_f64().unwrap() as f32;
        assert_eq!(best, 0.35);
        assert_eq!(a["now"]["threshold"].as_f64().unwrap() as f32, 0.35);
        assert_eq!(a["now"]["source"], "preds");
        assert_eq!(
            (
                &a["now"]["point"]["tp"],
                &a["now"]["point"]["fp"],
                &a["now"]["point"]["fn"]
            ),
            (
                &serde_json::json!(3),
                &serde_json::json!(0),
                &serde_json::json!(0)
            )
        );
        assert_eq!(a["now"]["point"]["f1"], a["best"]["point"]["f1"]);
        // The run's own threshold is kept as history.
        assert_eq!(a["configured"]["threshold"].as_f64().unwrap() as f32, 0.5);
        assert!(a["by_dataset"][0]["now"]["f1"].is_number());
        let rec = v["results"]["models"][0]["recommendation"]
            .as_str()
            .unwrap();
        assert!(rec.contains("(the configured threshold)"), "{rec}");
        // The per-model config view shows it too.
        let cm = v["config"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == "m")
            .unwrap()
            .clone();
        assert_eq!(cm["confidenceThreshold"].as_f64().unwrap() as f32, 0.35);

        // Another config change (a different threshold): recomputed at that value.
        st.config
            .update("test", |c| {
                c.models
                    .iter_mut()
                    .find(|m| m.name.as_deref() == Some("m"))
                    .unwrap()
                    .confidence_threshold = Some(0.42);
                Ok(())
            })
            .unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let a = advice(&v);
        assert_eq!(a["now"]["threshold"].as_f64().unwrap() as f32, 0.42);
        assert_eq!(a["now"]["point"]["tp"], 2);

        // The search's "configured" columns are at the current value too.
        let (s, v) = call(&st, post("/v1/benchmark/threshold-search", "")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["now"]["threshold"].as_f64().unwrap() as f32, 0.42);
        assert_eq!(r["now"]["tp"], 2);
        assert_eq!(r["configured"]["threshold"].as_f64().unwrap() as f32, 0.5);
        assert_eq!(
            r["by_dataset"][0]["now"]["threshold"].as_f64().unwrap() as f32,
            0.42
        );
        assert!(
            v["csv"]
                .as_str()
                .unwrap()
                .contains("\nm,ort:coreml,f1,0.42,"),
            "{}",
            v["csv"]
        );

        // The page renders the live view.
        let (s, page) = call(&st, get("/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        let page = page.as_str().unwrap();
        assert!(
            page.contains("<script src=\"/static/benchmark.js?v="),
            "{page}"
        );
        let js = crate::server::pages::static_file("benchmark.js").expect("benchmark.js");
        assert!(js.contains("function nowOf(a)"));
        assert!(js.contains("L.keep($(\"thresholds\")"));
    }
    #[tokio::test]
    async fn status_lists_the_confidence_profiles() {
        let st = state();
        let cfg_path = st.config_path.clone();
        // No results: no profiles; the goal is the config's.
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["profiles"], serde_json::json!([]));
        assert_eq!(v["otherHardware"], false);
        assert_eq!(v["goal"]["label"], "Balanced");
        assert_eq!(v["goal"]["objective"], "f1");
        assert_eq!(
            v["config"]["models"][0]["deviceThresholds"],
            serde_json::json!({})
        );
        assert_eq!(v["config"]["models"][0]["thresholdSource"], "global");

        // Measured on other hardware (an empty summary never matches this machine).
        profile_results(Default::default())
            .save(&results_path(&cfg_path))
            .unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["otherHardware"], true);
        let p = v["profiles"].as_array().unwrap();
        assert_eq!(p.len(), 3);
        let ip = &p[0];
        assert_eq!(ip["model"], "ipcam-general");
        assert_eq!(ip["configured"], true);
        assert_eq!(ip["device"], "global");
        assert_eq!(ip["running"], serde_json::Value::Null);
        assert_eq!(ip["fallback"], 0.5);
        assert_eq!(ip["fallbackSource"], "global");
        assert_eq!(ip["status"], "unsaved");
        let d = &ip["devices"];
        assert_eq!(d.as_array().unwrap().len(), 3);
        let cpu = &d[0];
        assert_eq!(cpu["device"], "openvino:cpu");
        assert_eq!(cpu["label"], "openvino:cpu");
        assert_eq!(cpu["recommended"], true);
        assert_eq!(cpu["ok"], true);
        assert_eq!(cpu["error"], serde_json::Value::Null);
        assert_eq!(cpu["fellBack"], false);
        assert_eq!(cpu["ms"], 20.0);
        // test_advice: TP 3, FP 1, FN 1 at the best threshold.
        assert_eq!(cpu["finds"], 0.75);
        assert_eq!(cpu["correct"], 0.75);
        assert_eq!(cpu["best"], 0.4);
        assert_eq!(cpu["saved"], serde_json::Value::Null);
        assert_eq!(cpu["differs"], true);
        assert!(cpu["now"].is_boolean());
        assert_eq!(d[1]["best"], 0.33);
        let gpu = &d[2];
        assert_eq!(
            (&gpu["ok"], &gpu["error"]),
            (
                &serde_json::json!(false),
                &serde_json::json!("compile failed")
            )
        );
        assert_eq!(
            (&gpu["best"], &gpu["differs"], &gpu["ms"]),
            (
                &serde_json::Value::Null,
                &serde_json::json!(false),
                &serde_json::Value::Null
            )
        );
        // Every field of the contract, nothing else.
        let keys = |o: &serde_json::Value| {
            let mut k: Vec<String> = o.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(
            keys(ip),
            [
                "configured",
                "device",
                "devices",
                "fallback",
                "fallbackSource",
                "model",
                "running",
                "status"
            ]
        );
        assert_eq!(
            keys(cpu),
            [
                "best",
                "correct",
                "device",
                "differs",
                "error",
                "fellBack",
                "finds",
                "label",
                "ms",
                "now",
                "ok",
                "recommended",
                "saved"
            ]
        );
        assert_eq!(p[2]["model"], "removed");
        assert_eq!(p[2]["configured"], false);

        // Measured here; every profile applied: saved, and shown on the config models.
        profile_results(crate::benchmark::report::HardwareSummary::current())
            .save(&results_path(&cfg_path))
            .unwrap();
        let (s, j) = call(&st, post("/v1/benchmark/apply-profiles", "all=1")).await;
        assert_eq!(s, StatusCode::OK, "{j}");
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["otherHardware"], false);
        assert_eq!(v["profiles"][0]["status"], "saved");
        assert_eq!(v["profiles"][0]["devices"][0]["saved"], 0.4);
        assert_eq!(v["profiles"][0]["devices"][0]["differs"], false);
        assert_eq!(v["profiles"][1]["status"], "saved");
        assert_eq!(
            v["config"]["models"][0]["deviceThresholds"],
            serde_json::json!({"openvino:cpu": 0.4, "ort:cpu": 0.33})
        );
        // One profile changed by hand: differs.
        st.config
            .update("test", |c| {
                c.models[0].device_thresholds.insert("ort:cpu".into(), 0.5);
                c.models[0].confidence_threshold = Some(0.45);
                Ok(())
            })
            .unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let ip = &v["profiles"][0];
        assert_eq!(ip["status"], "differs");
        assert_eq!(
            (&ip["fallback"], &ip["fallbackSource"]),
            (&serde_json::json!(0.45), &serde_json::json!("model"))
        );
        assert_eq!(ip["devices"][1]["differs"], true);
        assert_eq!(v["config"]["models"][0]["thresholdSource"], "model");
    }
}
