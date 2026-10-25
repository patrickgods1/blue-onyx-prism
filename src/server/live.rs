//! Live config for the pages:
//! - `GET /v1/config`: for an admin peer ([`super::admin_allowed`]) the effective config, its
//!   revision and change log, per model what is configured vs what the running worker uses,
//!   what a restart would apply, and the main form's field values the Config page refreshes
//!   untouched fields from (ETag / If-None-Match); for anyone else only the revision,
//!   generation and epoch counters plus the pending-change count the pages' restart poller
//!   and header indicator need ([`minimal_payload`]), read without touching the config file;
//! - the Config page's forms ([`server_form_edit`], [`models_card_edit`]): applied as
//!   submitted (last write wins; the change log keeps what changed).

use super::{AppState, ConfigView};
use crate::config::{Config, validate_config};
use crate::config_merge::{self, Pending};
use crate::registry::normalize_name;
use crate::startup::ModelState;
use crate::sync::LockExt;
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
    /// Every device the benchmark measured: its speed and its best confidence threshold.
    pub devices: Vec<DeviceFact>,
}

/// What the last benchmark measured for a model on one device.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DeviceFact {
    /// Device spec of the benchmark row ("ort:coreml").
    pub device: String,
    /// Full per-request p50 (ms) of a successful run.
    pub p50_ms: Option<f64>,
    /// The device's best threshold that may be applied
    /// ([`crate::benchmark::threshold::ThresholdAdvice::apply_value_on`]), rounded.
    #[serde(serialize_with = "crate::config::thr::ser_opt")]
    pub best: Option<f32>,
}

impl BenchFacts {
    /// The facts of the benchmark row for `device` (a spec): the exact row first, else a row
    /// naming the same device ([`crate::config::same_device`]).
    pub fn device_fact(&self, device: &str) -> Option<&DeviceFact> {
        self.devices
            .iter()
            .find(|d| d.device == device)
            .or_else(|| {
                self.devices
                    .iter()
                    .find(|d| crate::config::same_device(&d.device, device))
            })
    }

    /// The best threshold for `device`: its own row's, else the overall best when the advice
    /// was computed on that device (results of older versions have no per-device rows).
    pub fn best_on(&self, device: &str) -> Option<f32> {
        if let Some(t) = self.device_fact(device).and_then(|d| d.best) {
            return Some(t);
        }
        let same = |d: &str| d == device || crate::config::same_device(d, device);
        let a = self.advice.as_ref()?;
        (same(&a.device) && !a.by_device.iter().any(|p| same(&p.device)))
            .then_some(self.best_threshold)
            .flatten()
    }

    /// The recommendation with its threshold part against the config now (model `m`, global
    /// threshold `global`, resolved device `device`) and the threshold the loaded model runs
    /// with (`running`); see [`crate::benchmark::current`]. Exact counts only where the results
    /// store them.
    pub fn recommendation_now(
        &self,
        m: &crate::config::ModelConfig,
        global: f32,
        running: Option<f64>,
        device: Option<String>,
    ) -> String {
        let Some(advice) = &self.advice else {
            return self.recommendation.clone();
        };
        let mut a = advice.clone();
        let running = running.map(|r| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a confidence threshold (0..1); f32 is the precision the models use"
            )]
            let r = r as f32;
            r
        });
        let n = crate::benchmark::current::ModelNow::new(
            m.confidence_threshold.unwrap_or(global),
            running,
            device,
        )
        .with_profiles(m.device_thresholds.clone());
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

/// The facts of every measured device of `m`, in the results' device order (rows that neither
/// ran nor have a pick are left out).
fn device_facts(m: &crate::benchmark::ModelResult) -> Vec<DeviceFact> {
    let advice = m.threshold.as_ref();
    let best = |d: &str| {
        advice
            .and_then(|a| a.apply_value_on(d))
            .map(crate::config::round_threshold)
    };
    let mut out: Vec<DeviceFact> = m
        .devices
        .iter()
        .map(|d| DeviceFact {
            device: d.device.clone(),
            p50_ms: d.total_p50(),
            best: best(&d.device),
        })
        .filter(|f| f.p50_ms.is_some() || f.best.is_some())
        .collect();
    for p in advice.map(|a| a.by_device.as_slice()).unwrap_or_default() {
        if !out.iter().any(|f| f.device == p.device)
            && let Some(t) = best(&p.device)
        {
            out.push(DeviceFact {
                device: p.device.clone(),
                p50_ms: None,
                best: Some(t),
            });
        }
    }
    out
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
    let mut cache = BENCH_CACHE.lock_recover();
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
                        devices: device_facts(m),
                        advice: m.threshold.clone().map(|mut a| {
                            // Only what the recommendation text and the per-device picks
                            // need.
                            a.exact.clear();
                            a.roc = None;
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
// Confidence per device

/// The canonical profile key ([`crate::config::device_threshold_key`]) of a resolved device
/// spec; "" when there is none (nothing resolves, `auto`, or not a spec).
pub(super) fn device_key(spec: Option<&str>) -> String {
    spec.and_then(|s| crate::config::device_threshold_key(s).ok())
        .unwrap_or_default()
}

/// What the Models card's Threshold cell holds for a model on one device.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ThresholdCell {
    /// Canonical device key ("" = no device known: the cell edits `confidence_threshold`).
    pub key: String,
    /// The model's profile for `key`.
    pub profile: Option<f32>,
    /// What a device without a profile uses: the model's own threshold, else the global one.
    pub fallback: f32,
    /// "model default" or "global".
    pub fallback_source: &'static str,
}

impl ThresholdCell {
    pub fn new(m: &crate::config::ModelConfig, global: f32, key: &str) -> Self {
        let profile = (!key.is_empty())
            .then(|| crate::config::lookup_device_threshold(&m.device_thresholds, key))
            .flatten();
        let (fallback, fallback_source) = match m.confidence_threshold {
            Some(t) => (t, "model default"),
            None => (global, "global"),
        };
        Self {
            key: key.to_string(),
            profile,
            fallback,
            fallback_source,
        }
    }

    /// The threshold the model uses on `key`.
    pub fn effective(&self) -> f32 {
        self.profile.unwrap_or(self.fallback)
    }

    /// Where [`Self::effective`] comes from: "profile", "model default 0.38", "global 0.50".
    pub fn source_text(&self) -> String {
        match self.profile {
            Some(_) => "profile".to_string(),
            None => format!("{} {}", self.fallback_source, fmt_thr(self.fallback)),
        }
    }

    /// The input's placeholder: the fallback and its source ("0.38 model default").
    pub fn placeholder(&self) -> String {
        format!("{} {}", fmt_thr(self.fallback), self.fallback_source)
    }
}

/// A threshold as the page shows it ("0.30").
pub(super) fn fmt_thr(t: f32) -> String {
    format!("{:.2}", crate::config::round_threshold_f64(t))
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
    /// Effective confidence threshold of the worker on its loaded device.
    pub threshold: Option<f64>,
    /// Where `threshold` comes from ([`crate::config::ThresholdSource::label`]: "profile for
    /// ort:coreml", "model", "global").
    pub threshold_source: Option<String>,
    /// One line for the Models card's Running column ("Failed" alone: the reason is `detail`).
    pub text: String,
}

/// What the worker of model `name` runs with. The threshold is the worker's
/// ([`crate::worker::WorkerHandle::effective_threshold`]): the profile for the device it loaded
/// on (`auto` and a fallback count as the device that actually loaded), else its fallback.
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
            threshold_source: None,
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
    let (t, source) = w.effective_threshold();
    let threshold = Some(crate::config::round_threshold_f64(t));
    let source_text = match &source {
        crate::config::ThresholdSource::Model => "model default".to_string(),
        s => s.label(),
    };
    let text = match w.state.get() {
        // The reason is `detail` (the card shows it on one truncated line under this).
        ModelState::Failed(_) => "Failed".to_string(),
        ModelState::Ready => match threshold {
            Some(t) => format!("{provider} \u{b7} threshold {t:.2} ({source_text})"),
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
        threshold_source: Some(source.label()),
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
            let facts = (*bench).as_ref().and_then(|b| b.models.get(&key).cloned());
            let running = running_view(state, &name);
            // The device the threshold follows: the loaded one, else what the config resolves to.
            let threshold_key = device_key(running.device.as_deref().or(resolved.as_deref()));
            let cell = ThresholdCell::new(m, cfg.confidence_threshold, &threshold_key);
            let effective = thr(cell.effective());
            let source = m
                .threshold_on(
                    (!threshold_key.is_empty()).then_some(threshold_key.as_str()),
                    cfg.confidence_threshold,
                )
                .1;
            let profiles: serde_json::Map<String, Value> = m
                .device_thresholds
                .iter()
                .map(|(k, t)| (k.clone(), json!(thr(*t))))
                .collect();
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
                "deviceThresholds": profiles,
                "thresholdKey": threshold_key,
                "thresholdSource": source,
                "fallback": {
                    "value": thr(cell.fallback),
                    "source": cell.fallback_source,
                    "text": cell.placeholder(),
                },
                "effectiveThreshold": effective,
                "running": running,
                "restartNeeded": !reasons.is_empty(),
                "reasons": reasons,
                "bench": facts.map(|f| {
                    let best_here = (!threshold_key.is_empty())
                        .then(|| f.best_on(&threshold_key))
                        .flatten();
                    json!({
                        "recommended": f.recommended,
                        "recommendation": f.recommendation_now(
                            m,
                            cfg.confidence_threshold,
                            running.threshold,
                            resolved.clone(),
                        ),
                        "inUse": f.recommended.is_some() && f.recommended == resolved,
                        "bestThreshold": f.best_threshold.map(thr),
                        "bestHere": best_here.map(thr),
                        "objective": f.objective,
                        "thresholdInUse": best_here.is_some_and(|t| thr(t) == effective),
                        "byDevice": f.devices,
                        "href": format!("/benchmark#m-{}", super::encode_uri_component(&f.model)),
                    })
                }),
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

/// The `GET /v1/config` answer for a client that may not administer: the counters the restart
/// poller (`live.js`, `message.html`) compares and the header's pending-change count. No file
/// read, no model reload ([`super::sync_config`] is not called), nothing of the config itself.
pub(super) fn minimal_payload(state: &AppState) -> Value {
    let (revision, cfg) = state.config.snapshot();
    let pending = config_merge::pending(&state.running(), &cfg);
    json!({
        "success": true,
        "revision": revision,
        "epoch": state.config.epoch(),
        "generation": state.config.generation(),
        "restartNeeded": pending.restart_needed(),
        "pendingCount": pending.count(),
    })
}

/// `GET /v1/config` with a weak ETag over the payload (If-None-Match answers 304). The full
/// [`payload`] for admin peers, [`minimal_payload`] for everyone else (SV-05).
pub(super) async fn config_json(
    State(state): State<Arc<AppState>>,
    peer: Result<
        axum::extract::ConnectInfo<std::net::SocketAddr>,
        axum::extract::rejection::ExtensionRejection,
    >,
    headers: HeaderMap,
) -> Response {
    let peer = peer.ok().map(|c| c.0);
    let admin = super::admin_allowed(peer, state.config_read().allow_remote_admin);
    let body = if admin {
        // Reads the config file when it changed on disk, `benchmark.json`, and probes devices.
        super::blocking(move || payload(&state)).await
    } else {
        minimal_payload(&state)
    };
    let text = body.to_string();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    // `Value::Null` (shown as `null`) when missing, as indexing gave.
    let revision = body.get("revision").cloned().unwrap_or_default();
    let etag = format!("W/\"{}-{:016x}\"", revision, h.finish());
    let matches = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag));
    let etag_value = HeaderValue::from_str(&etag).unwrap_or(HeaderValue::from_static("W/\"0\""));
    let rev = HeaderValue::from_str(&revision.to_string()).unwrap_or(HeaderValue::from_static("0"));
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
/// does not carry keeps its value; see [`crate::config_form::apply_config_form`]).
pub(super) fn server_form_edit(
    c: &mut Config,
    form: &HashMap<String, String>,
) -> anyhow::Result<()> {
    crate::config_form::apply_config_form(c, form)
}

fn field<'a>(form: &'a [(String, String)], k: &'a str) -> impl Iterator<Item = &'a str> {
    form.iter()
        .filter(move |(key, _)| key == k)
        .map(|(_, v)| v.as_str())
}

/// A threshold typed in the Models card for `name` (`what`: "threshold", "default threshold");
/// "" = None (`empty`: what an empty field means, for the error text).
fn parse_card_threshold(
    name: &str,
    what: &str,
    v: &str,
    empty: &str,
) -> anyhow::Result<Option<f32>> {
    let v = v.trim();
    if v.is_empty() {
        return Ok(None);
    }
    let t = v.parse::<f32>().map_err(|_| {
        anyhow::anyhow!(
            "{what} for '{name}': '{v}' is not a number (probably a typo); enter a value \
             between 0 and 1 (e.g. 0.5) or leave it empty {empty}"
        )
    })?;
    // The same range check as the benchmark's threshold apply and config loading.
    crate::config::check_threshold(name, t).map_err(|_| {
        anyhow::anyhow!(
            "{what} for '{name}': {v} is outside 0 to 1 (a confidence is never above 1, so it \
             would drop every detection); enter a value between 0 and 1 or leave it empty \
             {empty}"
        )
    })?;
    Ok(Some(t))
}

/// The config entry of model `name` (matched like `/v1/vision/custom/{model}`).
fn card_model<'a>(
    c: &'a mut Config,
    name: &str,
) -> anyhow::Result<&'a mut crate::config::ModelConfig> {
    let key = normalize_name(name);
    c.models
        .iter_mut()
        .find(|m| normalize_name(&m.effective_name()) == key)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "'{name}' is not one of the configured models (it was removed or renamed \
                 meanwhile); reload the Config page"
            )
        })
}

/// The Models card over the current config `c`, applied as submitted: `enabled` (repeated; the
/// listed models are enabled, the others disabled), `default_model`, `device.<name>` ("" =
/// global), the thresholds and `night.<name>` (the night model, "" = none; `@self` = this
/// model with `night_threshold.<name>` on night / IR frames).
///
/// Thresholds: `threshold.<name>` with a non-empty `threshold_key.<name>` (the device the cell
/// shows) sets the model's confidence profile for that device (`device_thresholds`; "" removes
/// it); without the key it sets `confidence_threshold` ("" = global), as older pages do.
/// `threshold_default.<name>` sets `confidence_threshold` (the default for devices without a
/// profile; "" = global), and `profile_remove.<name>` (repeatable, a device) removes a profile.
pub(super) fn models_card_edit(c: &mut Config, form: &[(String, String)]) -> anyhow::Result<()> {
    let enabled: Vec<String> = field(form, "enabled").map(str::to_string).collect();
    let default = field(form, "default_model").next();
    crate::config_form::apply_models_selection(c, &enabled, default)?;
    let devices: Vec<(String, Option<String>)> = form
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("device.")
                .map(|name| (name.to_string(), Some(v.clone())))
        })
        .collect();
    crate::config_form::apply_model_devices(c, &devices)?;
    // The default first: without a device key, `threshold.<name>` writes the same field (and
    // the page disables the default field then).
    for (k, v) in form {
        if let Some(name) = k.strip_prefix("threshold_default.") {
            let t = parse_card_threshold(name, "default threshold", v, "for the global one")?;
            card_model(c, name)?.confidence_threshold = t;
        }
    }
    let mut profiles: Vec<(String, String, Option<f32>)> = Vec::new();
    for (k, v) in form {
        let Some(name) = k.strip_prefix("threshold.") else {
            continue;
        };
        let key_field = format!("threshold_key.{name}");
        let key = field(form, &key_field)
            .map(str::trim)
            .find(|k| !k.is_empty());
        match key {
            Some(key) => {
                let device = crate::config::device_threshold_key(key).map_err(|e| {
                    anyhow::anyhow!(
                        "threshold for '{name}': the device '{key}' is not one device ({e:#}); \
                         reload the Config page and save again"
                    )
                })?;
                let t = parse_card_threshold(name, "threshold", v, "for the model's default")?;
                card_model(c, name)?;
                profiles.push((name.to_string(), device, t));
            }
            None => {
                let t = parse_card_threshold(name, "threshold", v, "for the global one")?;
                card_model(c, name)?.confidence_threshold = t;
            }
        }
    }
    for (k, v) in form {
        if let Some(name) = k.strip_prefix("profile_remove.") {
            let v = v.trim();
            if v.is_empty() {
                continue;
            }
            card_model(c, name)?;
            profiles.push((name.to_string(), v.to_string(), None));
        }
    }
    crate::config_form::apply_device_thresholds(c, &profiles)?;
    for (k, v) in form {
        let Some(name) = k.strip_prefix("night.") else {
            continue;
        };
        let v = v.trim();
        let night_threshold = if v == super::config_page::NIGHT_SELF {
            let key = format!("night_threshold.{name}");
            let raw = field(form, &key).next().unwrap_or("");
            let t = parse_card_threshold(name, "night threshold", raw, "to turn it off")?;
            Some(t.ok_or_else(|| {
                anyhow::anyhow!(
                    "night threshold for '{name}': give the threshold for night / IR frames \
                     (0 to 1), or choose none in the Night model column"
                )
            })?)
        } else {
            None
        };
        let m = card_model(c, name)?;
        m.night_model = (!v.is_empty() && night_threshold.is_none()).then(|| v.to_string());
        m.night_threshold = night_threshold;
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

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::server::test_util::*;

    fn cfg() -> Config {
        let mut c = Config {
            models: ["m", "n"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        c.models[0].confidence_threshold = Some(0.38);
        c.models[0].device_thresholds = [("ort:coreml".to_string(), 0.3)].into();
        c.normalize();
        c
    }

    fn form(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut f: Vec<(String, String)> = vec![
            ("enabled".into(), "m".into()),
            ("enabled".into(), "n".into()),
        ];
        f.extend(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        f
    }

    fn edit(c: &Config, pairs: &[(&str, &str)]) -> anyhow::Result<Config> {
        let mut c = c.clone();
        models_card_edit(&mut c, &form(pairs))?;
        Ok(c)
    }

    #[test]
    fn card_threshold_with_a_device_key_writes_that_profile() {
        let c = cfg();
        // A new profile under the canonical key; the model default stays.
        let out = edit(&c, &[("threshold.m", "0.34"), ("threshold_key.m", "CPU")]).unwrap();
        let m = &out.models[0];
        assert_eq!(m.confidence_threshold, Some(0.38));
        assert_eq!(
            m.device_thresholds,
            [
                ("openvino:cpu".to_string(), 0.34),
                ("ort:coreml".to_string(), 0.3)
            ]
            .into()
        );
        // Changing an existing one; another spelling of the same device replaces it.
        let out = edit(
            &c,
            &[("threshold.m", "0.25"), ("threshold_key.m", " ORT:CoreML ")],
        )
        .unwrap();
        assert_eq!(
            out.models[0].device_thresholds,
            [("ort:coreml".to_string(), 0.25)].into()
        );
        // Empty removes the profile (the device then uses the model default).
        let out = edit(
            &c,
            &[("threshold.m", ""), ("threshold_key.m", "ort:coreml")],
        )
        .unwrap();
        assert!(out.models[0].device_thresholds.is_empty());
        assert_eq!(out.models[0].confidence_threshold, Some(0.38));
        // Unchanged value: nothing changes.
        let out = edit(
            &c,
            &[("threshold.m", "0.30"), ("threshold_key.m", "ort:coreml")],
        )
        .unwrap();
        assert_eq!(out, c);
        // `auto` and nonsense are not one device; bad values are named.
        for key in ["auto", "vulkan"] {
            let e = format!(
                "{:#}",
                edit(&c, &[("threshold.m", "0.3"), ("threshold_key.m", key)]).unwrap_err()
            );
            assert!(e.contains("is not one device"), "{e}");
        }
        let e = format!(
            "{:#}",
            edit(
                &c,
                &[("threshold.m", "abc"), ("threshold_key.m", "ort:cpu")]
            )
            .unwrap_err()
        );
        assert!(e.contains("'abc' is not a number"), "{e}");
        let e = format!(
            "{:#}",
            edit(
                &c,
                &[("threshold.m", "1.5"), ("threshold_key.m", "ort:cpu")]
            )
            .unwrap_err()
        );
        assert!(e.contains("outside 0 to 1"), "{e}");
        let e = format!(
            "{:#}",
            edit(
                &c,
                &[("threshold.zz", "0.3"), ("threshold_key.zz", "ort:cpu")]
            )
            .unwrap_err()
        );
        assert!(e.contains("not one of the configured models"), "{e}");
    }

    #[test]
    fn card_threshold_without_a_key_writes_the_model_threshold() {
        let c = cfg();
        for pairs in [
            vec![("threshold.n", "0.4")],
            vec![("threshold.n", "0.4"), ("threshold_key.n", "  ")],
        ] {
            let out = edit(&c, &pairs).unwrap();
            assert_eq!(out.models[1].confidence_threshold, Some(0.4));
            assert!(out.models[1].device_thresholds.is_empty());
        }
        let out = edit(&c, &[("threshold.m", "")]).unwrap();
        assert_eq!(out.models[0].confidence_threshold, None, "empty = global");
        assert_eq!(
            out.models[0].device_thresholds,
            c.models[0].device_thresholds
        );
    }

    #[test]
    fn card_default_and_profile_removal() {
        let mut c = cfg();
        c.models[0]
            .device_thresholds
            .insert("openvino:gpu".into(), 0.2);
        // The default for other devices; the profile field for the shown device in the same post.
        let out = edit(
            &c,
            &[
                ("threshold_default.m", "0.45"),
                ("threshold.m", "0.30"),
                ("threshold_key.m", "ort:coreml"),
            ],
        )
        .unwrap();
        assert_eq!(out.models[0].confidence_threshold, Some(0.45));
        assert_eq!(out.models[0].device_thresholds.len(), 2);
        let out = edit(&c, &[("threshold_default.m", "")]).unwrap();
        assert_eq!(out.models[0].confidence_threshold, None);
        let e = format!(
            "{:#}",
            edit(&c, &[("threshold_default.m", "x")]).unwrap_err()
        );
        assert!(e.contains("default threshold for 'm'"), "{e}");
        // Remove (repeatable, any spelling); a removal wins over the shown device's value.
        let out = edit(
            &c,
            &[
                ("threshold.m", "0.30"),
                ("threshold_key.m", "ort:coreml"),
                ("profile_remove.m", "ort:coreml"),
                ("profile_remove.m", "GPU.0"),
                ("profile_remove.m", ""),
            ],
        )
        .unwrap();
        assert!(out.models[0].device_thresholds.is_empty());
        assert_eq!(out.models[0].confidence_threshold, Some(0.38));
        // Removing a profile that does not exist changes nothing.
        let out = edit(&c, &[("profile_remove.m", "ort:cpu")]).unwrap();
        assert_eq!(out, c);
    }

    /// Saves benchmark results for `ipcam-general` beside `state`'s config: the overall best
    /// 0.35 on ort:coreml, per device 0.30 on ort:coreml and 0.34 on openvino:cpu.
    pub(in crate::server) fn write_profiled_bench(state: &AppState) {
        let (mut results, _) = crate::benchmark::current::tests::advised_fixture();
        let m = &mut results.models[0];
        m.model = "ipcam-general".into();
        let a = m.threshold.as_mut().unwrap();
        let pick = |t: f32| {
            let mut p = a.best.clone().unwrap();
            p.threshold = t;
            Some(p)
        };
        a.by_device = vec![
            crate::benchmark::threshold::DevicePick {
                device: "ort:coreml".into(),
                best: pick(0.3),
            },
            crate::benchmark::threshold::DevicePick {
                device: "openvino:cpu".into(),
                best: pick(0.34),
            },
        ];
        results
            .save(&crate::benchmark::results_path(&state.config_path))
            .unwrap();
    }

    #[test]
    fn payload_carries_the_per_device_thresholds() {
        let state = test_state_with(|c| {
            let m = &mut c.models[0];
            m.device = Some("openvino:cpu".into());
            m.confidence_threshold = Some(0.38);
            m.device_thresholds = [("openvino:cpu".to_string(), 0.3)].into();
        });
        write_profiled_bench(&state);
        let j = payload(&state);
        let m = &j["models"][0];
        assert_eq!(m["deviceThresholds"], json!({"openvino:cpu": 0.3}), "{m}");
        assert_eq!(m["thresholdKey"], "openvino:cpu");
        assert_eq!(m["thresholdSource"], "profile for openvino:cpu");
        assert_eq!(m["effectiveThreshold"], 0.3);
        assert_eq!(
            m["fallback"],
            json!({"value": 0.38, "source": "model default", "text": "0.38 model default"})
        );
        let by = &m["bench"]["byDevice"];
        assert_eq!(
            by,
            &json!([
                {"device": "ort:coreml", "p50Ms": null, "best": 0.3},
                {"device": "openvino:cpu", "p50Ms": null, "best": 0.34}
            ]),
            "{m}"
        );
        assert_eq!(m["bench"]["bestHere"], 0.34);
        assert_eq!(m["bench"]["bestThreshold"], 0.35);
        assert_eq!(m["bench"]["thresholdInUse"], false);
        // The other model has no profile: global, no benchmark.
        let n = &j["models"][1];
        assert_eq!(n["deviceThresholds"], json!({}));
        assert_eq!(n["fallback"]["source"], "global");
        assert!(n["bench"].is_null());
    }

    #[test]
    fn threshold_cell_texts() {
        let c = cfg();
        let cell = ThresholdCell::new(&c.models[0], 0.5, "ort:coreml");
        assert_eq!((cell.profile, cell.effective()), (Some(0.3), 0.3));
        assert_eq!(cell.source_text(), "profile");
        assert_eq!(cell.placeholder(), "0.38 model default");
        let cell = ThresholdCell::new(&c.models[0], 0.5, "openvino:cpu");
        assert_eq!(cell.source_text(), "model default 0.38");
        let cell = ThresholdCell::new(&c.models[1], 0.5, "");
        assert_eq!((cell.profile, cell.effective()), (None, 0.5));
        assert_eq!(cell.source_text(), "global 0.50");
        assert_eq!(device_key(Some("GPU")), "openvino:gpu");
        assert_eq!(device_key(Some("auto")), "");
        assert_eq!(device_key(None), "");
    }
}
