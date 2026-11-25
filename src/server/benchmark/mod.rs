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
//! - `POST /v1/benchmark/datasets/remove`: drop one saved folder dataset from those defaults;
//! - `POST /v1/benchmark/report`: save the HTML report of the saved results into a chosen
//!   folder (`report`);
//! - `GET /v1/benchmark/images?model=&device=`: per-image drill-down (ground truth and
//!   predictions); `GET /v1/benchmark/image?set=&file=`: the image file itself (only files
//!   listed in the saved results);
//! - `GET /v1/benchmark/compare`, `GET /v1/benchmark/tune`, `GET /v1/benchmark/tune/images`,
//!   `POST /v1/benchmark/goal`: the evaluation core's views and the goal (`eval`,
//!   [`crate::benchmark::eval`]);
//! - `GET /v1/benchmark/history[?model=]`, `GET /v1/benchmark/history/compare?model=&a=&b=`,
//!   `POST /v1/benchmark/history/pin`, `POST /v1/benchmark/history/delete`: the run history
//!   (`benchmark-history.json`, [`crate::benchmark::history`]).
//!
//! The run itself is [`crate::benchmark::service`]. This module keeps the page, the status
//! (`GET /v1/benchmark`) and the shared form/error helpers; `run` holds the start, settings,
//! cancel and drill-down endpoints, `apply` the device/threshold apply and threshold search,
//! `history` the run history endpoints.

mod apply;
mod cameras;
mod eval;
mod history;
mod pages;
mod poll;
mod report;
mod run;

use apply::BLUE_IRIS_THRESHOLD_NOTE;
pub(super) use pages::{compare_page, history_page, images_page, index, run_page, tune_page};

pub(super) use apply::{apply, apply_profiles, apply_threshold, search_options, threshold_search};
pub(super) use cameras::rescore;
pub(super) use eval::{compare_route, goal_route, tune_images_route, tune_route};
pub(super) use history::{history_compare, history_delete, history_get, history_pin};
pub(super) use report::save_report;
pub(super) use run::{cancel, image_file, images_detail, remove_dataset, settings, start};

use super::answer::json_fail;
use super::{AppState, DevicesSnapshot, devices_snapshot, global_device_label};
use crate::benchmark::search::file_stamp;
use crate::benchmark::{BenchmarkResults, configured_device, results_path};
use crate::config::Config;
use axum::extract::{RawQuery, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::warn;

/// Shown on the page and in the API: benchmarking competes with live requests.
pub(super) const LIVE_WARNING: &str = "Benchmarking runs on the same CPU/GPU as live detections: \
     Blue Iris requests are slower while it runs, and its numbers are noisier when the server is \
     busy. Run it when the cameras are quiet.";

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
            let key = m.key();
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

/// Each benchmarked model's place in the cross-model ranking ([`BenchmarkResults::ranking`],
/// [`crate::benchmark::grade::Grades::rank_cmp`]), keyed by normalized model name: the `best`
/// ordinal of the Home and Stats tables (`data-best`, `/stats.json` `best`). A model without a
/// graded result on its recommended device has none.
fn ranks_of(ranking: &[crate::benchmark::report::RankRow]) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for r in ranking {
        out.entry(crate::registry::normalize_name(&r.model))
            .or_insert(r.rank);
    }
    out
}

/// The last [`ranks_of`] of the saved results, keyed by [`ranks_key`]: the Stats poll (every 5 s)
/// parses the file again only after a benchmark rewrote it, the config changed or a model moved
/// to another device or threshold.
static RANKS_CACHE: crate::benchmark::search::IdleCache<String, Arc<HashMap<String, usize>>> =
    crate::benchmark::search::IdleCache::new(crate::benchmark::search::CACHE_IDLE);

/// What the served models run with now, sorted by name: each worker's loaded device and running
/// threshold. Part of [`model_ranks`]' key (QA-095: a fallback to another device or a new
/// running threshold changes no file and no config revision, yet [`models_now`] rebases the
/// ranks on it). Read from the registry's memory only: the Stats poll never probes devices.
fn running_stamp(state: &AppState) -> Vec<(String, Option<String>, u32)> {
    let mut out: Vec<_> = state
        .registry
        .workers()
        .iter()
        .map(|w| {
            let device = w
                .device
                .read()
                .ok()
                .and_then(|d| d.as_ref().map(|d| d.spec.clone()));
            (w.name.clone(), device, w.effective_threshold().0.to_bits())
        })
        .collect();
    out.sort();
    out
}

/// The [`RANKS_CACHE`] key of the results at `path` (stamped `stamp`): the files the ranks are
/// computed from, the config revision and the [`running_stamp`].
fn ranks_key(
    state: &AppState,
    path: &std::path::Path,
    stamp: (Option<std::time::SystemTime>, u64),
) -> String {
    let preds = crate::benchmark::search::preds_path(&state.config_path);
    let gt = crate::benchmark::cameras::snapshot::path_beside(path);
    format!(
        "{}|{stamp:?}|{:?}|{:?}|{}|{:?}",
        path.display(),
        file_stamp(&preds),
        file_stamp(&gt),
        state.config.revision(),
        running_stamp(state)
    )
}

/// [`ranks_of`] of the results beside the config (empty without results), in the goal's order
/// on the default Compare set ([`goal_order`]). Blocking: it may parse the files.
pub(in crate::server) fn model_ranks(state: &AppState) -> Arc<HashMap<String, usize>> {
    let path = results_path(&state.config_path);
    let Some(stamp) = file_stamp(&path) else {
        return Arc::default();
    };
    let preds = crate::benchmark::search::preds_path(&state.config_path);
    let key = ranks_key(state, &path, stamp);
    RANKS_CACHE.get_or_insert_with(key, || {
        Arc::new(
            BenchmarkResults::load_or_warn(&path)
                .map(|r| {
                    let stored = crate::benchmark::search::StoredPreds::load_cached(&preds)
                        .ok()
                        .flatten();
                    let snap = devices_snapshot(state);
                    let cfg = state.config_read().clone();
                    let now = models_now(state, &cfg, &snap);
                    let order = goal_order(state, &r, stored.as_deref(), &now);
                    ranks_of(&crate::benchmark::report::follow_order(r.ranking(), &order))
                })
                .unwrap_or_default(),
        )
    })
}

/// `results`, `ranking`, `points` and `latest_run_id` of `GET /v1/benchmark`: the saved results
/// rebased on the config now, with the history-derived fields
/// ([`crate::benchmark::history::view`]). The trees are kept serialized: a poll copies their
/// bytes into the answer, never the trees.
#[derive(Debug, Clone)]
struct SavedView {
    /// The [`SAVED_CACHE`] key this view was computed for (part of the answer's tag,
    /// [`poll`]).
    key: String,
    results: Box<RawValue>,
    ranking: Box<RawValue>,
    points: Box<RawValue>,
    latest_run_id: Option<String>,
    /// The hardware the results were measured on (`otherHardware`).
    hardware: Option<crate::benchmark::report::HardwareSummary>,
}

/// `value` serialized as its `serde_json::Value` serializes (sorted keys, f32 widened to f64:
/// the bytes `json!` gave), `null` if it cannot be (as `to_value(..).unwrap_or_default()` gave).
fn raw<T: serde::Serialize + ?Sized>(value: &T) -> Box<RawValue> {
    serde_json::to_value(value)
        .and_then(|v| serde_json::value::to_raw_value(&v))
        .unwrap_or_else(|_| null())
}

fn null() -> Box<RawValue> {
    RawValue::NULL.to_owned()
}

/// The last [`SavedView`] and what it was computed from: the results, stored-predictions and
/// history files' stamps and the config revision with what it says about each model. Polling
/// reads none of the files (tens of MB) again until one of them or the config changes. The view
/// goes after [`crate::benchmark::search::CACHE_IDLE`] with no poll (M10): an idle
/// service does not keep it.
static SAVED_CACHE: crate::benchmark::search::IdleCache<String, Arc<SavedView>> =
    crate::benchmark::search::IdleCache::new(crate::benchmark::search::CACHE_IDLE);

/// How many times [`saved_view`] computed (tests: polling does not recompute).
static SAVED_COMPUTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn saved_view(
    state: &AppState,
    revision: u64,
    now: &HashMap<String, crate::benchmark::current::ModelNow>,
) -> Arc<SavedView> {
    use crate::benchmark::history::history_path;
    use crate::benchmark::search::preds_path;
    let path = results_path(&state.config_path);
    let preds = preds_path(&state.config_path);
    let hist = history_path(&path);
    let gt = crate::benchmark::cameras::snapshot::path_beside(&path);
    let mut sorted: Vec<_> = now.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    // The snapshot too: the results are scored against it.
    let key = format!(
        "{}|{:?}|{:?}|{:?}|{:?}|{revision}|{sorted:?}",
        path.display(),
        file_stamp(&path),
        file_stamp(&preds),
        file_stamp(&hist),
        file_stamp(&gt)
    );
    // Computed under the cache's lock: polls arriving together compute once.
    SAVED_CACHE.get_or_insert_with(key.clone(), || {
        Arc::new(compute_saved_view(state, key, &path, &preds, &hist, now))
    })
}

/// What [`saved_view`] caches.
fn compute_saved_view(
    state: &AppState,
    key: String,
    path: &std::path::Path,
    preds: &std::path::Path,
    hist: &std::path::Path,
    now: &HashMap<String, crate::benchmark::current::ModelNow>,
) -> SavedView {
    use crate::benchmark::history::{History, view};
    use crate::benchmark::search::StoredPreds;
    SAVED_COMPUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match BenchmarkResults::load_or_warn(path) {
        Some(mut r) => {
            let stored = StoredPreds::load_cached(preds)
                .map_err(|e| warn!("{e:#}"))
                .ok()
                .flatten();
            // `ranking` and `points` follow the goal's rank on the default Compare set.
            let order = goal_order(state, &r, stored.as_deref(), now);
            let ranking = raw(&crate::benchmark::report::follow_order(r.ranking(), &order));
            let points = raw(&crate::benchmark::report::follow_order(r.points(), &order));
            crate::benchmark::current::rebase(&mut r, stored.as_deref(), now);
            crate::benchmark::trained_on::backfill(&mut r, &state.config_read());
            // No history file yet: seeded in memory (written by the next run, pin or delete).
            let history = History::load_or_seed(hist, Some(&r));
            let mut results = serde_json::to_value(r.summary()).unwrap_or_default();
            view::annotate(&mut results, &r, &history);
            SavedView {
                key,
                results: raw(&results),
                ranking,
                points,
                latest_run_id: view::latest_run_id(&r),
                hardware: Some(r.hardware.clone()),
            }
        }
        None => SavedView {
            key,
            results: null(),
            ranking: null(),
            points: null(),
            latest_run_id: None,
            hardware: None,
        },
    }
}

/// The (model, device) order of the goal's Compare view on the default set of `results`
/// ([`crate::benchmark::eval::compare`], the saved goal); empty without stored predictions.
fn goal_order(
    state: &AppState,
    results: &BenchmarkResults,
    preds: Option<&crate::benchmark::search::StoredPreds>,
    now: &HashMap<String, crate::benchmark::current::ModelNow>,
) -> Vec<(String, String)> {
    use crate::benchmark::combos::{Inputs, Parts};
    let Some(preds) = preds else {
        return Vec::new();
    };
    let cfg = state.config_read();
    let mut view = results.clone();
    crate::benchmark::current::rebase(&mut view, Some(preds), now);
    crate::benchmark::trained_on::backfill(&mut view, &cfg);
    let parts = Parts::of(&view, &cfg);
    let default_model = crate::training::switch::current_default(&cfg);
    let inputs = Inputs {
        results: &view,
        preds: Some(preds),
        now,
        default_model: default_model.as_deref(),
        configured: &parts.configured,
        trained: &parts.trained,
        seen: &parts.seen,
        reachable: &|s| s.dir.as_deref().is_some_and(std::path::Path::is_dir),
    };
    crate::benchmark::eval::compare::build(
        &inputs,
        crate::benchmark::eval::compare::Request {
            goal: &cfg.benchmark.goal,
            weights: cfg.benchmark.weights,
            set: None,
            scene: None,
        },
    )
    .rows
    .into_iter()
    .map(|r| (r.model, r.device))
    .collect()
}

/// `GET /v1/benchmark`: the current/last run (state, progress, partial results), the saved
/// results (without per-image details), the ranking and the config's per-model devices. Results
/// are shown against the config now: each threshold advice carries `now` (the effective
/// threshold in the config, the counts there, exact from the stored predictions) and each
/// model's `configured` is the device the config resolves to now. `otherHardware` says whether
/// the results were measured on other GPUs, `goal` is the saved goal as the views name it
/// ([`crate::benchmark::eval::compare::GoalView`]). The pages read the evaluation views instead
/// (`/v1/benchmark/compare`, `/tune`) and poll `?only=run`.
///
/// `?only=run[&models=<modelsTag>]` answers the run alone with the tag of the rest (`viewTag`),
/// for polling during a run; the full answer has an `ETag` (304 on `If-None-Match`) and
/// `X-Benchmark-View` ([`poll`]).
pub(super) async fn status(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
    headers: axum::http::HeaderMap,
) -> Response {
    let if_none_match = headers
        .get(axum::http::header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    super::blocking(move || poll::answer(&state, query.as_deref(), if_none_match.as_deref())).await
}

/// An API error answered as `{"success": false, "message": ...}` with its status
/// ([`json_fail`]); small, unlike a [`Response`], so helpers can return it in a `Result`.
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
        json_fail(self.status, self.message)
    }
}

/// The value of the first `key` field of a form (untrimmed).
pub(super) fn form_field<'f>(form: &'f [(String, String)], key: &str) -> Option<&'f str> {
    form.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The non-empty `model` fields of a form, trimmed ([`crate::server::form::all`]).
pub(super) fn form_models(form: &[(String, String)]) -> Vec<String> {
    crate::server::form::all(form, "model")
        .into_iter()
        .map(str::to_string)
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

    /// QA-095: a model moving to another device or running threshold, with no file or config
    /// revision change, changes the Home / Stats ranks' cache key (they stayed stale before).
    #[test]
    fn model_ranks_key_follows_the_running_device_and_threshold() {
        let st = state();
        let path = results_path(&st.config_path);
        let stamp = (None, 0);
        let first = ranks_key(&st, &path, stamp);
        assert_eq!(ranks_key(&st, &path, stamp), first, "nothing changed");
        let w = st.registry.workers()[0].clone();
        *w.device.write().unwrap() = Some(crate::backend::DeviceInfo {
            spec: "openvino:cpu".into(),
            ..crate::backend::DeviceInfo::default()
        });
        let on_cpu = ranks_key(&st, &path, stamp);
        assert_ne!(on_cpu, first, "a device fallback");
        let mut m = ModelConfig::default();
        m.device_thresholds.insert("openvino:cpu".into(), 0.3);
        w.set_live(crate::worker::LiveSettings::for_model(
            &Config::default(),
            &m,
        ));
        assert_ne!(
            ranks_key(&st, &path, stamp),
            on_cpu,
            "a new running threshold"
        );
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
            // The data root in the temp folder: the signed-in test user lives there.
            download_dir: Some(dir.clone()),
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
    async fn benchmark_goes_to_the_step_with_something_to_show() {
        use axum::http::header::LOCATION;
        let st = state();
        let to = |r: &Response| r.headers()[LOCATION].to_str().unwrap().to_string();
        let r = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        assert_eq!(r.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(to(&r), "/benchmark/run", "no results yet");
        profile_results(Default::default())
            .save(&results_path(&st.config_path))
            .unwrap();
        let r = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        assert_eq!(to(&r), "/benchmark/compare");
        let r = router(st.clone())
            .oneshot(get("/benchmark?goal=f2&set=cameras"))
            .await
            .unwrap();
        assert_eq!(
            to(&r),
            "/benchmark/compare?goal=f2&set=cameras",
            "the query is kept"
        );
    }

    /// Every step: the shell (title, goal chip, stepper with the step current and Run ticked
    /// once results exist, History / Images), the Goal dialog, and its scripts (CSP: no inline
    /// scripts; bench-core.js first).
    #[tokio::test]
    async fn every_step_has_the_shell_and_its_scripts() {
        let st = state();
        let steps = [
            ("run", "bench-run.js"),
            ("compare", "bench-compare.js"),
            ("tune", "bench-tune.js"),
            ("history", "bench-history.js"),
            ("images", "bench-images.js"),
        ];
        for with_results in [false, true] {
            if with_results {
                profile_results(Default::default())
                    .save(&results_path(&st.config_path))
                    .unwrap();
            }
            for (step, script) in steps {
                let (s, html) =
                    crate::server::test_util::get_text(&st, &format!("/benchmark/{step}")).await;
                assert_eq!(s, StatusCode::OK, "{step}");
                assert!(
                    html.contains(
                        "<a href=\"/benchmark\" class=\"active\" aria-current=\"page\">Benchmark</a>"
                    ),
                    "{step}"
                );
                assert!(html.contains("<h1>Benchmark</h1>"), "{step}");
                assert!(html.contains("id=\"goal-chip\""), "{step}");
                assert!(
                    html.contains("<b id=\"goal-chip-v\">F1</b>"),
                    "{step}: the saved goal"
                );
                assert!(
                    html.contains("<dialog class=\"bn-dialog\" id=\"goal-dlg\""),
                    "{step}"
                );
                assert_eq!(
                    html.matches("aria-current=\"step\"").count()
                        + html
                            .matches("data-step=\"history\" aria-current=\"page\"")
                            .count()
                        + html
                            .matches("data-step=\"images\" aria-current=\"page\"")
                            .count(),
                    1,
                    "{step}: one current step"
                );
                assert!(
                    html.contains(&format!("data-step=\"{step}\" aria-current=")),
                    "{step}: current"
                );
                let ticked = html.contains("class=\"bn-step done\"");
                assert_eq!(
                    ticked,
                    with_results && step != "run",
                    "{step}: the Run tick"
                );
                let core = html.find("/static/bench-core.js?v=").expect(step);
                let shell = html.find("/static/bench-shell.js?v=").expect(step);
                let page = html.find(&format!("/static/{script}?v=")).expect(script);
                assert!(core < shell && shell < page, "{step}: script order");
                assert!(
                    super::super::pages::static_file(script).is_some(),
                    "{script}"
                );
            }
        }
        // The empty states point at Run.
        let (_, html) = crate::server::test_util::get_text(&state(), "/benchmark/compare").await;
        assert!(
            html.contains("id=\"cmp-empty\" class=\"bn-empty\">"),
            "{html}"
        );
        assert!(html.contains("id=\"cmp-body\" hidden"), "{html}");
        // The pages ask the evaluation endpoints, the shell polls the run.
        let js = |n: &str| super::super::pages::static_file(n).unwrap();
        assert!(js("bench-compare.js").contains("\"/v1/benchmark/compare\""));
        assert!(js("bench-tune.js").contains("\"/v1/benchmark/tune\""));
        assert!(js("bench-tune.js").contains("\"/v1/benchmark/tune/images\""));
        assert!(js("bench-tune.js").contains("\"/v1/benchmark/apply-profiles\""));
        assert!(js("bench-shell.js").contains("\"/v1/benchmark/goal\""));
        assert!(js("bench-shell.js").contains("\"/v1/benchmark/rescore\""));
        assert!(js("bench-history.js").contains("\"/v1/benchmark/history\""));
        assert!(js("bench-images.js").contains("/v1/benchmark/tune/images?"));
    }

    #[tokio::test]
    async fn the_run_step_lists_images_models_and_devices() {
        let st = state();
        let (s, html) = crate::server::test_util::get_text(&st, "/benchmark/run").await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            html.contains("name=\"model\" value=\"ipcam-general\""),
            "{html}"
        );
        // Files are missing: listed but not checked.
        assert!(!html.contains("value=\"ipcam-general\" data-timing=\"{}\" checked"));
        assert!(
            html.contains("name=\"device\" value=\"openvino:cpu\" checked"),
            "{html}"
        );
        assert!(
            html.contains("data-cpu=\"1\""),
            "the CPU reference is marked"
        );
        // ONNX Runtime is not in this probe: listed, disabled.
        assert!(
            html.contains("name=\"device\" value=\"ort:cpu\" disabled"),
            "{html}"
        );
        // Your cameras first; the configured datasets are checked; built-in sets offer a download.
        let cams = html.find("value=\"cameras\"").expect("Your cameras");
        let coco = html
            .find("name=\"dataset\" value=\"coco-cctv\"")
            .expect("coco");
        assert!(cams < coco, "Your cameras first");
        assert!(
            html.contains("name=\"dataset\" value=\"coco-cctv\" data-images=\""),
            "{html}"
        );
        assert!(html.contains("data-download=\"bench:coco-cctv\""));
        assert!(html.contains("name=\"dataset\" value=\"sample\" data-images=\"1\">"));
        assert!(html.contains("same CPU/GPU"));
        assert!(html.contains("id=\"o-served\""), "As served");
        // The burst test: off by default, its load from `benchmark.burst` in the ranges.
        assert!(
            html.contains("<input type=\"checkbox\" id=\"o-burst\"> "),
            "unticked by default: {html}"
        );
        assert!(html.contains("id=\"o-burst-fields\" disabled"), "{html}");
        assert!(
            html.contains(
                "id=\"o-burst-cameras\" min=\"1\" max=\"32\" inputmode=\"numeric\" value=\"8\""
            ),
            "{html}"
        );
        assert!(html.contains("id=\"o-burst-images\" min=\"1\" max=\"20\""));
        assert!(html.contains("id=\"o-burst-interval\" min=\"50\" max=\"5000\""));
        assert!(
            html.contains("id=\"gpu-confirm\""),
            "the in-page GPU confirmation"
        );
        assert_eq!(html.matches("<dialog id=\"browse-dialog\">").count(), 1);
        let js = super::super::pages::static_file("bench-run.js").unwrap();
        for needle in [
            "\"/v1/benchmark\"",
            "\"/v1/benchmark/settings\"",
            "BC.cpuReference",
            "BC.estimate",
            "BC.comboStates",
            "BC.burstSeconds",
            "\"burst_cameras\"",
            "\"burst_images\"",
            "\"burst_interval_ms\"",
        ] {
            assert!(js.contains(needle), "{needle}");
        }
    }

    #[tokio::test]
    async fn saved_devices_that_cannot_run_here_are_not_ticked() {
        let st = state();
        let ticked = |html: &str, spec: &str| html.contains(&format!("value=\"{spec}\" checked"));
        // (the device boxes carry no data-timing: `value` is followed by `checked` directly)
        // Only a vanished iGPU saved: the device the model runs on is ticked instead of none.
        st.config
            .update("test", |c| {
                c.benchmark.devices = vec!["openvino:gpu.0".into()];
                Ok(())
            })
            .unwrap();
        let (_, html) = crate::server::test_util::get_text(&st, "/benchmark/run").await;
        assert!(ticked(&html, "openvino:cpu"), "{html}");
        assert!(!html.contains("value=\"openvino:gpu.0\""), "{html}");
        // Every saved one unavailable (ORT is not in this probe): the same, and a disabled box
        // is never ticked.
        st.config
            .update("test", |c| {
                c.benchmark.devices = vec!["openvino:gpu.0".into(), "ort:cpu".into()];
                Ok(())
            })
            .unwrap();
        let (_, html) = crate::server::test_util::get_text(&st, "/benchmark/run").await;
        assert!(ticked(&html, "openvino:cpu"), "{html}");
        assert!(!ticked(&html, "ort:cpu"), "{html}");
        let js = super::super::pages::static_file("bench-run.js").expect("bench-run.js");
        assert!(js.contains("Tick at least one device"));
    }

    fn gpu_box() -> DevicesSnapshot {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{RuntimeProbe, select};
        let adapter = |index, vendor, name: &str, discrete| GpuAdapter {
            vendor,
            name: name.into(),
            vram_mb: if discrete { 8192 } else { 0 },
            index,
            discrete,
            cuda: None,
        };
        let hardware = crate::backend::HardwareInfo::new(
            "linux",
            "x86_64",
            vec![
                adapter(0, GpuVendor::Intel, "UHD 630", false),
                adapter(1, GpuVendor::Intel, "Arc A770", true),
            ],
        );
        let names: Vec<String> = ["CPU", "GPU.0", "GPU.1"].map(String::from).to_vec();
        let probe = RuntimeProbe::openvino_only(&names);
        let selection = select(&hardware, &probe, true);
        DevicesSnapshot {
            hardware,
            selection,
            probe,
            openvino_version: "test".into(),
            openvino_devices: names,
            onnxruntime: None,
        }
    }

    /// G1: with nothing saved the form ticks the devices the models run on, not the machine's
    /// every device; a saved choice wins; with neither, one device.
    #[test]
    fn the_form_starts_on_the_devices_in_use() {
        let snap = gpu_box();
        let ticked = |c: &Config, in_use: &[&str]| -> Vec<String> {
            let in_use: Vec<String> = in_use.iter().map(|s| s.to_string()).collect();
            super::pages::device_rows(&snap, c, &in_use)
                .into_iter()
                .filter(|r| r.checked)
                .map(|r| r.spec)
                .collect()
        };
        let mut c = Config::default();
        let runnable = super::pages::device_rows(&snap, &c, &[])
            .iter()
            .filter(|r| r.runnable)
            .count();
        assert!(runnable >= 3, "the fixture has several devices");

        assert_eq!(ticked(&c, &["openvino:gpu.1"]), ["openvino:gpu.1"]);
        // Spelled as the config may spell it.
        assert_eq!(ticked(&c, &["GPU.1", "openvino:gpu.1"]).len(), 1);
        assert_eq!(
            ticked(&c, &["openvino:gpu.0", "openvino:cpu"]).len(),
            2,
            "two models on two devices"
        );
        assert_eq!(ticked(&c, &["ort:coreml"]).len(), 1, "gone: one, not all");
        assert_eq!(ticked(&c, &[]).len(), 1);

        c.benchmark.devices = vec!["openvino:cpu".into()];
        assert_eq!(ticked(&c, &["openvino:gpu.1"]), ["openvino:cpu"]);

        let rows = super::pages::device_rows(&snap, &Config::default(), &[]);
        let gpu = |spec: &str| rows.iter().find(|r| r.spec == spec).unwrap().gpu;
        assert!(gpu("openvino:gpu.0") && gpu("openvino:gpu.1"));
        assert!(!gpu("openvino:cpu"));
    }

    /// The form's side of the confirmation: the boxes are marked, the warning sits under them,
    /// Start asks in the page (no `confirm()`) and then sends the field the server wants.
    #[tokio::test]
    async fn the_form_warns_about_several_gpus() {
        use crate::benchmark::service::SEVERAL_GPUS_WARNING;
        let (_, html) = crate::server::test_util::get_text(&state(), "/benchmark/run").await;
        assert!(html.contains("id=\"gpu-warn\" hidden"), "{html}");
        assert!(html.contains("never found"), "{html}");
        assert!(SEVERAL_GPUS_WARNING.contains("has hung a machine"));
        let js = super::super::pages::static_file("bench-run.js").expect("bench-run.js");
        assert!(js.contains("dataset.gpu") && js.contains("\"confirm_gpus\""));
        assert!(!js.contains("confirm("), "asked in the page");
        let template = include_str!("../../../templates/benchmark/run.html");
        assert!(template.contains("data-gpu=\"1\""));
    }

    /// G1: a breadcrumb in the data root is shown on the page once, and the file goes with it.
    #[tokio::test]
    async fn a_run_that_never_ended_is_shown_once() {
        use crate::benchmark::service::breadcrumb::{BREADCRUMB_FILE, Breadcrumb};
        let st = crate::server::test_util::test_state();
        let file = st.config_read().data_root().join(BREADCRUMB_FILE);
        let (_, html) = crate::server::test_util::get_text(&st, "/benchmark/run").await;
        assert!(!html.contains("id=\"bench-interrupted\""), "nothing left");
        let crumb = Breadcrumb {
            device: "ort:directml:1".into(),
            model: "<b>m</b>".into(),
            time: "2026-11-06T10:00:00Z".into(),
        };
        std::fs::write(&file, serde_json::to_vec(&crumb).unwrap()).unwrap();
        let (_, html) = crate::server::test_util::get_text(&st, "/benchmark/compare").await;
        assert!(html.contains("id=\"bench-interrupted\""), "{html}");
        assert!(html.contains("ort:directml:1"), "{html}");
        assert!(!html.contains("<b>m</b>"), "escaped");
        assert!(!file.exists(), "shown, so gone");
        let (_, html) = crate::server::test_util::get_text(&st, "/benchmark/tune").await;
        assert!(!html.contains("id=\"bench-interrupted\""), "once");
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

        // A saved threshold: Now is the new value, P/R/F1 evaluated there exactly.
        let (s, j) = call(
            &st,
            post("/v1/benchmark/apply-threshold", "model=m&threshold=0.35"),
        )
        .await;
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
    }
    /// The status keeps the config's per-model view (profiles, sources), whether the results
    /// were measured on other GPUs, and the goal as the views name it.
    #[tokio::test]
    async fn status_shows_the_config_view_and_the_goal() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["otherHardware"], false);
        assert_eq!(v["goal"]["optimizeFor"], "f1");
        assert_eq!(v["goal"]["summary"], "F1");
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
        profile_results(crate::benchmark::report::HardwareSummary::current())
            .save(&results_path(&cfg_path))
            .unwrap();
        st.config
            .update("test", |c| {
                c.models[0].device_thresholds.insert("ort:cpu".into(), 0.5);
                c.models[0].confidence_threshold = Some(0.45);
                c.benchmark.goal.optimize_for = "f2".parse().unwrap();
                Ok(())
            })
            .unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["otherHardware"], false);
        assert_eq!(
            v["config"]["models"][0]["deviceThresholds"],
            serde_json::json!({"ort:cpu": 0.5})
        );
        assert_eq!(v["config"]["models"][0]["thresholdSource"], "model");
        assert_eq!(v["goal"]["metric"], "F2");
    }
}
