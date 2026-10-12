//! Live config for the pages:
//! - `GET /v1/config`: the effective config, its revision and change log, per model what is
//!   configured vs what the running worker uses, what a restart would apply, and the main
//!   form's field values the Config page refreshes untouched fields from (ETag /
//!   If-None-Match);
//! - the Config page's forms ([`server_form_edit`], [`models_card_edit`]): applied as
//!   submitted (last write wins; the change log keeps what changed).

use super::{AppState, ConfigView};
use crate::config::{Config, validate_config};
use crate::config_merge::{self, Pending};
use crate::registry::normalize_name;
use crate::startup::ModelState;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Changes listed by `GET /v1/config`.
pub(super) const CHANGES_SHOWN: usize = 20;

// ---------------------------------------------------------------------------------------------
// Benchmark facts (cached by file stamp: the results file can be tens of MB)

/// What the last benchmark says about a model.
#[derive(Debug, Clone, Default)]
pub(super) struct BenchFacts {
    pub model: String,
    pub recommended: Option<String>,
    pub recommendation: String,
    /// Best confidence threshold that can be applied, and its objective.
    pub best_threshold: Option<f32>,
    pub objective: String,
    /// Measured `auto` order on this machine (`BenchmarkResults::auto_order`).
    pub auto_order: Option<Vec<crate::backend::spec::Device>>,
    /// The threshold advice (the recommendation text is rewritten against the config now).
    pub advice: Option<crate::benchmark::threshold::ThresholdAdvice>,
}

impl BenchFacts {
    /// The recommendation with its threshold part against the threshold configured now
    /// (`threshold`) and the one the loaded model runs with (`running`); see
    /// [`crate::benchmark::current`]. Exact counts only where the results store them.
    pub fn recommendation_now(&self, threshold: f32, running: Option<f64>) -> String {
        let Some(advice) = &self.advice else {
            return self.recommendation.clone();
        };
        let mut a = advice.clone();
        let n = crate::benchmark::current::ModelNow {
            threshold,
            running: running.map(|r| r as f32),
            device: None,
        };
        crate::benchmark::current::rebase_advice(
            &mut a,
            &self.model,
            &self.recommendation,
            &[],
            None,
            &n,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct BenchSummary {
    /// Normalized model name -> facts.
    pub models: HashMap<String, BenchFacts>,
    pub count: usize,
    pub timestamp: String,
}

type BenchCache = Option<(PathBuf, Option<SystemTime>, u64, Arc<Option<BenchSummary>>)>;
static BENCH_CACHE: Mutex<BenchCache> = Mutex::new(None);

/// The saved benchmark results' per-model facts (None when there are none), re-read only when
/// the file changes.
pub(super) fn bench_summary(path: &Path) -> Arc<Option<BenchSummary>> {
    let md = std::fs::metadata(path).ok();
    let stamp = md.as_ref().map(|m| (m.modified().ok(), m.len()));
    let mut cache = BENCH_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let (Some(c), Some((mt, len))) = (cache.as_ref(), stamp)
        && c.0 == path
        && c.1 == mt
        && c.2 == len
    {
        return c.3.clone();
    }
    let Some((mt, len)) = stamp else {
        return Arc::new(None);
    };
    let hw = crate::backend::detect::hardware();
    let summary = crate::benchmark::BenchmarkResults::load_or_warn(path).map(|r| BenchSummary {
        count: r.models.len(),
        timestamp: r.timestamp.clone(),
        models: r
            .models
            .iter()
            .map(|m| {
                (
                    normalize_name(&m.model),
                    BenchFacts {
                        model: m.model.clone(),
                        recommended: m.recommended.clone(),
                        recommendation: m.recommendation.clone(),
                        best_threshold: m.threshold.as_ref().and_then(|t| t.apply_value()),
                        objective: m
                            .threshold
                            .as_ref()
                            .map(|t| t.objective.clone())
                            .unwrap_or_else(|| m.threshold_objective.to_string()),
                        auto_order: r.auto_order(&m.model, hw),
                        advice: m.threshold.clone().map(|mut a| {
                            // Only what the recommendation text needs.
                            a.exact.clear();
                            a.roc = None;
                            a.by_device.clear();
                            a
                        }),
                    },
                )
            })
            .collect(),
    });
    let summary = Arc::new(summary);
    *cache = Some((path.to_path_buf(), mt, len, summary.clone()));
    summary
}

// ---------------------------------------------------------------------------------------------
// Running state

/// What the running worker of a model uses.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RunningView {
    /// A worker of this generation serves the model.
    pub present: bool,
    /// "Ready", "Loading", "Failed:", "Lazy", "Waiting for download", "not loaded".
    pub state: String,
    /// Badge class: ok, loading, wait, fail, lazy, off.
    pub state_class: &'static str,
    pub detail: Option<String>,
    /// Execution provider ("OpenVINO CPU", "ONNX Runtime CoreML (loading)").
    pub provider: Option<String>,
    /// Canonical spec of the loaded device.
    pub device: Option<String>,
    /// Effective confidence threshold of the worker.
    pub threshold: Option<f64>,
    /// One line for the Models card's Running column.
    pub text: String,
}

pub(super) fn running_view(state: &AppState, name: &str) -> RunningView {
    let Some(w) = state.registry.by_name(name) else {
        return RunningView {
            present: false,
            state: "not loaded".into(),
            state_class: "off",
            detail: None,
            provider: None,
            device: None,
            threshold: None,
            text: "not loaded".into(),
        };
    };
    let (st, class, detail) = super::state_view(&w);
    let provider = w.execution_provider();
    let device = w
        .device
        .read()
        .ok()
        .and_then(|d| d.as_ref().map(|d| d.spec.clone()));
    let threshold = Some(crate::config::round_threshold_f64(
        w.live().confidence_threshold,
    ));
    let text = match w.state.get() {
        ModelState::Failed(m) => format!("Failed: {m}"),
        ModelState::Ready => match threshold {
            Some(t) => format!("{provider} \u{b7} threshold {t:.2}"),
            None => provider.clone(),
        },
        // The provider label already ends in "(loading)" while the model compiles; the state
        // and its detail say the same, once.
        _ => match &detail {
            Some(d) => format!(
                "{} \u{b7} {st}: {d}",
                provider.trim_end_matches(" (loading)")
            ),
            None => provider.clone(),
        },
    };
    RunningView {
        present: true,
        state: st,
        state_class: class,
        detail,
        provider: Some(provider),
        device,
        threshold,
        text,
    }
}

/// Hardware and runtime probe of the loaded runtimes (cheap; None while they are busy
/// compiling).
fn quick_probe(
    state: &AppState,
) -> Option<(
    crate::backend::HardwareInfo,
    crate::backend::select::RuntimeProbe,
)> {
    let rt = state.registry.runtimes()?;
    let rt = rt.try_lock().ok()?;
    Some((rt.hardware().clone(), rt.probe().clone()))
}

/// The device `m` runs on after a restart, as a canonical spec (`auto` resolved with the
/// model's own selection, see [`super::model_selection`]).
fn configured_resolved(
    state: &AppState,
    cfg: &Config,
    m: &crate::config::ModelConfig,
    probe: Option<&(
        crate::backend::HardwareInfo,
        crate::backend::select::RuntimeProbe,
    )>,
) -> Option<String> {
    match probe {
        Some((hw, probe)) => {
            let sel = super::model_selection(state, hw, probe, cfg, m);
            crate::benchmark::configured_device(cfg, m, &sel)
        }
        None => match cfg.device_spec_for(m).ok()? {
            crate::backend::DeviceSpec::Auto => state
                .registry
                .by_name(&m.effective_name())
                .and_then(|w| w.planned.as_ref().map(|p| p.to_string())),
            d => Some(d.to_string()),
        },
    }
}

// ---------------------------------------------------------------------------------------------
// GET /v1/config

/// The payload of `GET /v1/config`.
pub(super) fn payload(state: &AppState) -> Value {
    super::sync_config(state);
    let (revision, cfg) = state.config.snapshot();
    let pending = config_merge::pending(&state.running(), &cfg);
    let bench = bench_summary(&crate::benchmark::results_path(&state.config_path));
    let probe = quick_probe(state);
    let default = cfg.effective_default_index();
    let thr = crate::config::round_threshold_f64;
    let models: Vec<Value> = cfg
        .models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = m.effective_name();
            let key = normalize_name(&name);
            let reasons = pending.for_model(&name);
            let resolved = configured_resolved(state, &cfg, m, probe.as_ref());
            let facts = (*bench)
                .as_ref()
                .and_then(|b| b.models.get(&key).cloned());
            let running = running_view(state, &name);
            json!({
                "name": name,
                "key": key,
                "family": m.family.to_string(),
                "path": m.path.display().to_string(),
                "enabled": m.enabled,
                "default": default == Some(i),
                "device": m.device,
                "deviceFor": cfg.device_for(m),
                "resolvedDevice": resolved,
                "threshold": m.confidence_threshold.map(thr),
                "effectiveThreshold": thr(m.confidence_threshold.unwrap_or(cfg.confidence_threshold)),
                "running": running,
                "restartNeeded": !reasons.is_empty(),
                "reasons": reasons,
                "bench": facts.map(|f| json!({
                    "recommended": f.recommended,
                    "recommendation": f.recommendation_now(
                        m.confidence_threshold.unwrap_or(cfg.confidence_threshold),
                        running.threshold,
                    ),
                    "inUse": f.recommended.is_some() && f.recommended == resolved,
                    "bestThreshold": f.best_threshold.map(thr),
                    "objective": f.objective,
                    "thresholdInUse": f.best_threshold.is_some_and(|t| {
                        thr(t) == thr(m.confidence_threshold.unwrap_or(cfg.confidence_threshold))
                    }),
                    "href": format!("/benchmark#m-{}", super::encode_uri_component(&f.model)),
                })),
            })
        })
        .collect();
    // Workers of this generation whose model left the config.
    let gone: Vec<String> = state
        .registry
        .workers()
        .iter()
        .filter(|w| crate::benchmark::find_model(&cfg, &w.name).is_none())
        .map(|w| w.name.clone())
        .collect();
    json!({
        "success": true,
        "revision": revision,
        "epoch": state.config.epoch(),
        "generation": state.config.generation(),
        "path": state.config_path.display().to_string(),
        "fileError": state.config.disk_error(),
        "config": cfg,
        "globalThreshold": thr(cfg.confidence_threshold),
        "restartNeeded": pending.restart_needed(),
        "pendingCount": pending.count(),
        "pending": pending_json(&pending, &cfg),
        "models": models,
        "loadedNotConfigured": gone,
        "changes": state.config.changes(CHANGES_SHOWN),
        "forms": {
            "server": {
                "view": ConfigView::from_config(&cfg),
            },
        },
    })
}

/// Pending reasons as a flat list ("IPcam-general: device: auto → openvino:cpu").
fn pending_json(p: &Pending, cfg: &Config) -> Value {
    let display = |key: &str| {
        cfg.models
            .iter()
            .map(|m| m.effective_name())
            .find(|n| normalize_name(n) == key)
            .unwrap_or_else(|| key.to_string())
    };
    let mut all: Vec<String> = p.global.clone();
    for (k, reasons) in &p.models {
        for r in reasons {
            all.push(format!("{}: {r}", display(k)));
        }
    }
    all.extend(p.process.iter().cloned());
    json!({
        "global": p.global,
        "process": p.process,
        "models": p.models,
        "all": all,
    })
}

/// `GET /v1/config` with a weak ETag over the payload (If-None-Match answers 304).
pub(super) async fn config_json(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    // Reads the config file when it changed on disk, `benchmark.json`, and probes devices.
    let body = super::blocking(move || payload(&state)).await;
    let text = body.to_string();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    let etag = format!("W/\"{}-{:016x}\"", body["revision"], h.finish());
    let matches = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag));
    let etag_value = HeaderValue::from_str(&etag).unwrap_or(HeaderValue::from_static("W/\"0\""));
    let rev = HeaderValue::from_str(&body["revision"].to_string())
        .unwrap_or(HeaderValue::from_static("0"));
    if matches {
        return (
            StatusCode::NOT_MODIFIED,
            [
                (header::ETAG, etag_value),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            ],
        )
            .into_response();
    }
    let mut resp = (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (header::ETAG, etag_value),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        text,
    )
        .into_response();
    resp.headers_mut().insert("x-config-revision", rev);
    resp
}

// ---------------------------------------------------------------------------------------------
// Forms (last write wins)

/// The main `/config` form: applied to the current config `c` as submitted (a field the form
/// does not carry keeps its value; see [`crate::config::apply_config_form`]).
pub(super) fn server_form_edit(
    c: &mut Config,
    form: &HashMap<String, String>,
) -> anyhow::Result<()> {
    crate::config::apply_config_form(c, form)
}

fn field<'a>(form: &'a [(String, String)], k: &'a str) -> impl Iterator<Item = &'a str> {
    form.iter()
        .filter(move |(key, _)| key == k)
        .map(|(_, v)| v.as_str())
}

/// The Models card over the current config `c`, applied as submitted: `enabled` (repeated; the
/// listed models are enabled, the others disabled), `default_model`, `device.<name>` ("" =
/// global) and `threshold.<name>` ("" = global).
pub(super) fn models_card_edit(c: &mut Config, form: &[(String, String)]) -> anyhow::Result<()> {
    let enabled: Vec<String> = field(form, "enabled").map(str::to_string).collect();
    let default = field(form, "default_model").next();
    crate::config::apply_models_selection(c, &enabled, default)?;
    let devices: Vec<(String, Option<String>)> = form
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("device.")
                .map(|name| (name.to_string(), Some(v.clone())))
        })
        .collect();
    crate::config::apply_model_devices(c, &devices)?;
    let mut thresholds = Vec::new();
    for (k, v) in form {
        if let Some(name) = k.strip_prefix("threshold.") {
            let v = v.trim();
            let t = if v.is_empty() {
                None
            } else {
                Some(
                    v.parse::<f32>()
                        .map_err(|_| anyhow::anyhow!("threshold: '{v}' is not a number"))?,
                )
            };
            thresholds.push((name.to_string(), t));
        }
    }
    for (name, t) in thresholds {
        let key = normalize_name(&name);
        let Some(m) = c
            .models
            .iter_mut()
            .find(|m| normalize_name(&m.effective_name()) == key)
        else {
            anyhow::bail!("'{name}' is not one of the configured models");
        };
        m.confidence_threshold = t;
    }
    validate_config(c)
}

pub(super) fn wants_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("application/json"))
}

/// JSON answer of a Config page form.
pub(super) fn saved_json(state: &AppState, message: String, restarting: bool) -> Response {
    Json(json!({
        "success": true,
        "message": message,
        "revision": state.config.revision(),
        "restarting": restarting,
    }))
    .into_response()
}

pub(super) fn error_json(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({ "success": false, "message": message })),
    )
        .into_response()
}
