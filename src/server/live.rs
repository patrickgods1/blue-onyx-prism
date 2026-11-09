//! Live config for the pages:
//! - `GET /v1/config`: for a signed-in request over HTTPS (a session or the API token, under an
//!   allowed `Host`) the effective config, its
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
use crate::config_store::ConfigRef;
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
#[derive(Debug, Clone, Default, PartialEq)]
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

#[derive(Debug, Clone, Default, PartialEq)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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

/// Hardware and runtime probe of the loaded runtimes (cheap; the startup facts while they are
/// busy compiling, see [`super::devices::runtime_facts`]; None without runtimes).
fn quick_probe(
    state: &AppState,
) -> Option<(
    crate::backend::HardwareInfo,
    crate::backend::select::RuntimeProbe,
)> {
    super::devices::runtime_facts(state)
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

/// Everything the full answer is built from, read once per poll. Two polls with equal inputs
/// ([`Inputs::same`]) get the same answer, so the second is served from [`AnswerCache`] without
/// building, serializing or hashing anything (RR-C2).
///
/// The inputs, all found by reading [`build_payload`]: the store state (config, revision,
/// generation, change log, file error: [`ConfigRef::serial`]), what this generation runs
/// ([`AppState::running`], swapped by every apply), the benchmark facts (re-read by file
/// stamp, [`bench_summary`]), the runtimes' hardware and probe (None without runtimes), per
/// configured model whether an `.onnx` sits beside it (the `auto` selection checks it), the
/// worker's view (state, detail, provider, device, threshold) and its planned device, and the
/// names of this generation's workers. The epoch and the config path never change for a state.
pub(super) struct Inputs {
    view: ConfigRef,
    running: Arc<Config>,
    bench: Arc<Option<BenchSummary>>,
    probe: Option<(
        crate::backend::HardwareInfo,
        crate::backend::select::RuntimeProbe,
    )>,
    onnx: Vec<bool>,
    views: Vec<RunningView>,
    planned: Vec<Option<String>>,
    workers: Vec<String>,
}

impl Inputs {
    /// Read the inputs of `state` now (cheap: atomic loads, a few `stat`s, the workers' state).
    fn gather(state: &AppState) -> Self {
        let view = state.config_read();
        let names: Vec<String> = view.models.iter().map(|m| m.effective_name()).collect();
        Self {
            running: state.running(),
            bench: bench_summary(&crate::benchmark::results_path(&state.config_path)),
            probe: quick_probe(state),
            onnx: view
                .models
                .iter()
                .map(|m| crate::backend::select::onnx_path_for(&view.data_path(&m.path)).is_some())
                .collect(),
            views: names.iter().map(|n| running_view(state, n)).collect(),
            planned: names
                .iter()
                .map(|n| {
                    state
                        .registry
                        .by_name(n)
                        .and_then(|w| w.planned.as_ref().map(|p| p.to_string()))
                })
                .collect(),
            workers: state
                .registry
                .workers()
                .iter()
                .map(|w| w.name.clone())
                .collect(),
            view,
        }
    }

    /// Both answers would be the same. The `Arc`s compare by pointer: a cached entry holds its
    /// own, so a pointer is never reused while it is compared.
    fn same(&self, o: &Self) -> bool {
        ConfigRef::serial(&self.view) == ConfigRef::serial(&o.view)
            && Arc::ptr_eq(&self.running, &o.running)
            // By content: `bench_summary` keeps one file's summary, so another caller's file (or a
            // missing one, a fresh `None`) gives a new pointer to the same facts.
            && (Arc::ptr_eq(&self.bench, &o.bench) || *self.bench == *o.bench)
            && self.probe == o.probe
            && self.onnx == o.onnx
            && self.views == o.views
            && self.planned == o.planned
            && self.workers == o.workers
    }
}

/// The payload of `GET /v1/config` (what [`full_answer`] serializes).
#[cfg(test)]
pub(super) fn payload(state: &AppState) -> Value {
    sync_for_poll(state);
    build_payload(state, &Inputs::gather(state))
}

/// Pick up a config file changed on disk and apply a revision not applied yet, like
/// [`super::sync_config`], without ever waiting: the file check only tries the store's writer
/// lock, and when `apply_lock` is held (a save writes and applies its edit under it, a restart
/// or the disk watch applies) the holder applies the revision itself, so the poll answers with
/// what runs now and the next poll shows the result.
fn sync_for_poll(state: &AppState) {
    state.config.check_disk();
    let applied = state
        .applied_revision
        .load(std::sync::atomic::Ordering::Acquire);
    if state.config.revision() != applied
        && let Some(held) = state.apply_lock.try_lock_recover()
    {
        super::apply::apply_config_locked(state, &held);
    }
}

/// The full payload from `inputs`.
fn build_payload(state: &AppState, inputs: &Inputs) -> Value {
    let cfg: &Config = &inputs.view;
    let revision = ConfigRef::revision(&inputs.view);
    let pending = config_merge::pending(&inputs.running, cfg);
    let bench = &inputs.bench;
    let probe = inputs.probe.as_ref();
    let default = cfg.effective_default_index();
    let thr = crate::config::round_threshold_f64;
    let models: Vec<Value> = cfg
        .models
        .iter()
        .zip(&inputs.views)
        .enumerate()
        .map(|(i, (m, running))| {
            let name = m.effective_name();
            let key = normalize_name(&name);
            let reasons = pending.for_model(&name);
            let resolved = configured_resolved(state, cfg, m, probe);
            let facts = (**bench).as_ref().and_then(|b| b.models.get(&key).cloned());
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
    let gone: Vec<&String> = inputs
        .workers
        .iter()
        .filter(|w| crate::benchmark::find_model(cfg, w).is_none())
        .collect();
    json!({
        "success": true,
        "revision": revision,
        "epoch": state.config.epoch(),
        "generation": ConfigRef::generation(&inputs.view),
        "path": state.config_path.display().to_string(),
        "fileError": ConfigRef::disk_error(&inputs.view),
        "config": cfg,
        "globalThreshold": thr(cfg.confidence_threshold),
        "restartNeeded": pending.restart_needed(),
        "pendingCount": pending.count(),
        "pending": pending_json(&pending, cfg),
        "models": models,
        "loadedNotConfigured": gone,
        "changes": ConfigRef::changes(&inputs.view, CHANGES_SHOWN),
        "forms": {
            "server": {
                "view": ConfigView::from_config(cfg),
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
fn minimal_payload(state: &AppState, view: &ConfigRef, running: &Config) -> Value {
    let pending = config_merge::pending(running, view);
    json!({
        "success": true,
        "revision": ConfigRef::revision(view),
        "epoch": state.config.epoch(),
        "generation": ConfigRef::generation(view),
        "restartNeeded": pending.restart_needed(),
        "pendingCount": pending.count(),
    })
}

// ---------------------------------------------------------------------------------------------
// Answer cache

/// A serialized `GET /v1/config` answer and its validators.
#[derive(Debug)]
pub(super) struct Answer {
    text: axum::body::Bytes,
    /// Weak ETag over the text: `W/"<revision>-<hash>"`.
    etag: HeaderValue,
    /// `X-Config-Revision`.
    revision: HeaderValue,
}

impl Answer {
    fn of(body: &Value) -> Self {
        let text = body.to_string();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut h);
        // `Value::Null` (shown as `null`) when missing, as indexing gave.
        let revision = body.get("revision").cloned().unwrap_or_default();
        let etag = format!("W/\"{}-{:016x}\"", revision, h.finish());
        Self {
            text: text.into(),
            etag: HeaderValue::from_str(&etag).unwrap_or(HeaderValue::from_static("W/\"0\"")),
            revision: HeaderValue::from_str(&revision.to_string())
                .unwrap_or(HeaderValue::from_static("0")),
        }
    }
}

/// What the minimal answer is built from: the store state ([`ConfigRef::serial`]) and what runs.
type MinimalKey = (u64, Arc<Config>);

/// The last answers of one [`AppState`] (one generation).
#[derive(Default)]
struct Slots {
    full: Option<(Inputs, Arc<Answer>)>,
    /// The minimal answer depends on the store state and what runs only.
    minimal: Option<(MinimalKey, Arc<Answer>)>,
    /// Full answers built (the tests count them).
    builds: u64,
}

/// Cached answers per [`AppState`], held weakly: a generation's entry goes once its state is
/// dropped. A static rather than an `AppState` field so the state's constructors stay as
/// they are. The lock is held only to look up or store an entry, never while building one
/// (two polls racing on a change may both build; the answers are equal).
struct AnswerCache(Mutex<Vec<(std::sync::Weak<AppState>, Slots)>>);

static ANSWERS: AnswerCache = AnswerCache(Mutex::new(Vec::new()));

impl AnswerCache {
    fn with<R>(&self, state: &Arc<AppState>, f: impl FnOnce(&mut Slots) -> R) -> R {
        let mut all = self.0.lock_recover();
        all.retain(|(w, _)| w.strong_count() > 0);
        let i = match all
            .iter()
            .position(|(w, _)| std::ptr::eq(w.as_ptr(), Arc::as_ptr(state)))
        {
            Some(i) => i,
            None => {
                all.push((Arc::downgrade(state), Slots::default()));
                all.len() - 1
            }
        };
        match all.get_mut(i) {
            Some((_, slots)) => f(slots),
            None => f(&mut Slots::default()),
        }
    }
}

/// The admin answer: from the cache when nothing it is built from changed, else built,
/// serialized and hashed here (callers run this in [`super::blocking`]: it may read the
/// config file, `benchmark.json` and probe devices).
fn full_answer(state: &Arc<AppState>) -> Arc<Answer> {
    sync_for_poll(state);
    let inputs = Inputs::gather(state);
    let hit = ANSWERS.with(state, |s| {
        s.full
            .as_ref()
            .filter(|(k, _)| k.same(&inputs))
            .map(|(_, a)| a.clone())
    });
    if let Some(a) = hit {
        return a;
    }
    let answer = Arc::new(Answer::of(&build_payload(state, &inputs)));
    ANSWERS.with(state, |s| {
        s.full = Some((inputs, answer.clone()));
        s.builds += 1;
    });
    answer
}

/// The minimal answer ([`minimal_payload`]), cached the same way.
fn minimal_answer(state: &Arc<AppState>) -> Arc<Answer> {
    let view = state.config_read();
    let running = state.running();
    let serial = ConfigRef::serial(&view);
    let hit = ANSWERS.with(state, |s| {
        s.minimal
            .as_ref()
            .filter(|((k, r), _)| *k == serial && Arc::ptr_eq(r, &running))
            .map(|(_, a)| a.clone())
    });
    if let Some(a) = hit {
        return a;
    }
    let answer = Arc::new(Answer::of(&minimal_payload(state, &view, &running)));
    ANSWERS.with(state, |s| {
        s.minimal = Some(((serial, running), answer.clone()))
    });
    answer
}

/// Full answers built for `state` so far.
#[cfg(test)]
pub(super) fn answers_built(state: &Arc<AppState>) -> u64 {
    ANSWERS.with(state, |s| s.builds)
}

/// `GET /v1/config` with a weak ETag over the payload (If-None-Match answers 304). The full
/// [`build_payload`] for a signed-in request over HTTPS (a session or the API token, the same
/// check as `admin_only`), [`minimal_payload`] for everyone else (SV-05). Nobody is refused:
/// the pages' poller reads this route, so a missing or wrong credential gets the minimal answer. A
/// wrong token counts against the peer's limit, or this would be a free way to test tokens, and
/// over the limit even the right one gets the minimal answer. Both answers are cached until
/// something they are built from changes ([`Inputs`]), so a poll with no change only compares.
pub(super) async fn config_json(
    State(state): State<Arc<AppState>>,
    peer: Result<
        axum::extract::ConnectInfo<std::net::SocketAddr>,
        axum::extract::rejection::ExtensionRejection,
    >,
    transport: Option<axum::Extension<super::Transport>>,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    let peer = peer.ok().map(|c| c.0);
    // This route is open, so `admin_only` never runs: the same checks here, or a DNS-rebound
    // page could read the admin payload (see `guard`). The plain port never gets it: no
    // credential is accepted there.
    let foreign = {
        let c = state.config_read();
        super::guard::foreign_host(&headers, &uri, &c.allowed_hosts)
    };
    let tls = transport.is_some_and(|t| t.0 == super::Transport::Tls);
    let admin = tls
        && foreign.is_none()
        && matches!(
            super::auth::authenticate(&state, peer, &headers).await,
            super::auth::Outcome::SignedIn(_)
        );
    let answer = if admin {
        // Reads the config file when it changed on disk, `benchmark.json`, and probes devices.
        super::blocking(move || full_answer(&state)).await
    } else {
        minimal_answer(&state)
    };
    let matches = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(',')
                .any(|t| t.trim().as_bytes() == answer.etag.as_bytes())
        });
    if matches {
        (
            StatusCode::NOT_MODIFIED,
            [
                (header::ETAG, answer.etag.clone()),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            ],
        )
            .into_response()
    } else {
        let mut resp = (
            [
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                ),
                (header::ETAG, answer.etag.clone()),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            ],
            answer.text.clone(),
        )
            .into_response();
        resp.headers_mut()
            .insert("x-config-revision", answer.revision.clone());
        resp
    }
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
/// `object_filter.<name>` of the Models card: [`OBJECT_FILTER_GLOBAL`] = no per-model filter
/// (`None`, the global one applies), empty = every class (`Some([])`), else a comma list.
fn parse_card_object_filter(name: &str, v: &str) -> anyhow::Result<Option<Vec<String>>> {
    let v = v.trim();
    if v == super::config_page::OBJECT_FILTER_GLOBAL {
        return Ok(None);
    }
    let names: Vec<String> = v
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if names.is_empty() && !v.is_empty() {
        anyhow::bail!(
            "classes for '{name}': '{v}' names no class; list class names separated by \
             commas, or leave it empty for every class"
        );
    }
    Ok(Some(names))
}

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
        if let Some(name) = k.strip_prefix("object_filter.") {
            let filter = parse_card_object_filter(name, v)?;
            card_model(c, name)?.object_filter = filter;
        }
    }
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
    fn card_object_filter_global_empty_and_list() {
        let c = cfg();
        let out = edit(&c, &[("object_filter.m", " Person, car ,")]).unwrap();
        assert_eq!(
            out.models[0].object_filter,
            Some(vec!["Person".to_string(), "car".to_string()])
        );
        assert_eq!(out.models[1].object_filter, None);
        // Empty = every class; `__global__` = no per-model filter.
        let all = edit(&out, &[("object_filter.m", "")]).unwrap();
        assert_eq!(all.models[0].object_filter, Some(vec![]));
        let global = edit(&all, &[("object_filter.m", "__global__")]).unwrap();
        assert_eq!(global.models[0].object_filter, None);
        // A field for another model leaves this one alone; nothing named is refused.
        assert_eq!(
            edit(&out, &[("object_filter.n", "dog")]).unwrap().models[0].object_filter,
            out.models[0].object_filter
        );
        assert!(edit(&c, &[("object_filter.m", " , ")]).is_err());
        assert!(edit(&c, &[("object_filter.zzz", "dog")]).is_err());
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

    fn config_get(etag: Option<&str>) -> axum::http::Request<axum::body::Body> {
        let mut r =
            axum::http::Request::get("/v1/config").header(header::ACCEPT, "application/json");
        if let Some(e) = etag {
            r = r.header(header::IF_NONE_MATCH, e);
        }
        r.body(axum::body::Body::empty()).unwrap()
    }

    /// RR-C2: a poll with nothing changed builds nothing (304 or the same 200 from the cache),
    /// and every input of the answer invalidates it: a save, an edit on disk, a file error,
    /// what runs, the benchmark file, a worker's state.
    #[tokio::test]
    async fn config_answer_is_cached_until_an_input_changes() {
        let state = test_state();
        let (s, h, first) = call(&state, config_get(None)).await;
        assert_eq!(s, StatusCode::OK);
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        assert_eq!(answers_built(&state), 1);
        for _ in 0..3 {
            let (s, h, b) = call(&state, config_get(Some(&etag))).await;
            assert_eq!((s, b.len()), (StatusCode::NOT_MODIFIED, 0));
            assert_eq!(h[header::ETAG].to_str().unwrap(), etag);
        }
        let (s, h, b) = call(&state, config_get(None)).await;
        assert_eq!((s, &b), (StatusCode::OK, &first));
        assert_eq!(h["x-config-revision"], "1");
        assert_eq!(answers_built(&state), 1, "an unchanged poll builds nothing");
        // The cached answer is what a fresh build gives.
        assert_eq!(
            serde_json::from_slice::<Value>(&b).unwrap(),
            payload(&state)
        );

        let mut built = 1;
        let mut etag = etag;
        let mut changed = async |what: &str, state: &Arc<AppState>| {
            let (s, h, _) = call(state, config_get(Some(&etag))).await;
            assert_eq!(s, StatusCode::OK, "{what}: a new answer");
            built += 1;
            assert_eq!(answers_built(state), built, "{what}: built once");
            let (s, _, _) = call(state, config_get(Some(h[header::ETAG].to_str().unwrap()))).await;
            assert_eq!(s, StatusCode::NOT_MODIFIED, "{what}: then cached again");
            assert_eq!(answers_built(state), built);
            etag = h[header::ETAG].to_str().unwrap().to_string();
        };
        // A save (new revision).
        super::super::save_config(&state, "test", |c| {
            c.nms_iou = 0.45;
            Ok(())
        })
        .unwrap();
        changed("a save", &state).await;
        // An edit on disk, picked up by the poll.
        let mut other = Config::load(&state.config_path).unwrap();
        other.port = 4321;
        other.save(&state.config_path).unwrap();
        changed("an edit on disk", &state).await;
        // A file that does not validate: no new revision, but `fileError`.
        std::fs::write(&state.config_path, "{ not json").unwrap();
        changed("a file error", &state).await;
        // What runs, swapped by an apply.
        state.running.store(Arc::new(Config::default()));
        changed("what runs", &state).await;
        // The benchmark results.
        write_profiled_bench(&state);
        changed("the benchmark file", &state).await;
        // A worker's state.
        state.registry.workers()[0]
            .state
            .set_initializing("waiting for x");
        changed("a worker's state", &state).await;
    }

    /// The minimal answer (a request that is not signed in) is cached as well, and follows the
    /// store.
    #[tokio::test]
    async fn minimal_answer_is_cached() {
        let state = test_state();
        let remote = |etag: Option<&str>| anonymous(from_peer(config_get(etag), REMOTE));
        let (s, h, b) = call(&state, remote(None)).await;
        assert_eq!(s, StatusCode::OK);
        let j: Value = serde_json::from_slice(&b).unwrap();
        assert!(j.get("config").is_none(), "{j}");
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        let (s, _, _) = call(&state, remote(Some(&etag))).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        assert_eq!(
            answers_built(&state),
            0,
            "no full answer when not signed in"
        );
        super::super::save_config(&state, "test", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let (s, _, b) = call(&state, remote(Some(&etag))).await;
        assert_eq!(s, StatusCode::OK);
        let j: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(
            (j["revision"].as_u64(), j["pendingCount"].as_u64()),
            (Some(2), Some(1))
        );
    }

    /// RR-C1 end to end: while a save from a page is inside its file write (holding
    /// `apply_lock` and the store's writer lock), `GET /v1/config` (full and minimal) and the
    /// request handlers' config reads answer with the previous config; after it, the new one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_poll_does_not_wait_for_a_save() {
        const STEP: std::time::Duration = std::time::Duration::from_secs(20);
        let state = test_state();
        // Write the file once, so the parked save below is a change.
        let (s, _, _) = call(&state, config_get(None)).await;
        assert_eq!(s, StatusCode::OK);
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Mutex::new(release_rx);
        state.config.set_write_hook(Some(Arc::new(move || {
            entered_tx.send(()).unwrap();
            release_rx
                .lock_recover()
                .recv_timeout(STEP)
                .expect("released");
        })));
        let saver = {
            let state = state.clone();
            std::thread::spawn(move || {
                super::super::save_config(&state, "config page", |c| {
                    c.port = 4000;
                    Ok(())
                })
                .unwrap()
            })
        };
        tokio::task::spawn_blocking(move || entered.recv_timeout(STEP))
            .await
            .unwrap()
            .expect("the save reached its write");
        assert!(
            state.apply_lock.try_lock().is_err(),
            "the save holds apply_lock"
        );
        let polls = async {
            let full = call(&state, config_get(None)).await;
            let minimal = call(&state, from_peer(config_get(None), REMOTE)).await;
            (full, minimal, state.config_read().port)
        };
        let ((full, minimal, port), _) = (
            tokio::time::timeout(STEP, polls)
                .await
                .expect("a poll waited for the save"),
            (),
        );
        assert_eq!(full.0, StatusCode::OK);
        let j: Value = serde_json::from_slice(&full.2).unwrap();
        assert_eq!(j["revision"], 1);
        assert_eq!(j["config"]["port"], crate::DEFAULT_PORT);
        let j: Value = serde_json::from_slice(&minimal.2).unwrap();
        assert_eq!(j["revision"], 1);
        assert_eq!(port, crate::DEFAULT_PORT);
        release.send(()).unwrap();
        let msg = tokio::task::spawn_blocking(move || saver.join().unwrap())
            .await
            .unwrap();
        assert!(msg.starts_with("Saved to "), "{msg}");
        let (_, _, b) = call(&state, config_get(None)).await;
        let j: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(
            (j["revision"].as_u64(), j["config"]["port"].as_u64()),
            (Some(2), Some(4000))
        );
        state.config.set_write_hook(None);
    }

    /// Work and bytes per `GET /v1/config` poll (run with `--ignored --nocapture`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "measurement"]
    async fn measure_config_poll() {
        use axum::body::Body;
        use axum::http::Request;
        let state = test_state();
        write_profiled_bench(&state);
        let get = |etag: Option<String>| {
            let mut r = Request::get("/v1/config").header(header::ACCEPT, "application/json");
            if let Some(e) = etag {
                r = r.header(header::IF_NONE_MATCH, e);
            }
            r.body(Body::empty()).unwrap()
        };
        let (s, h, b) = call(&state, get(None)).await;
        assert_eq!(s, StatusCode::OK);
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        let n = 500;
        let t = std::time::Instant::now();
        let mut bytes = 0;
        for _ in 0..n {
            let (s, _, b) = call(&state, get(Some(etag.clone()))).await;
            assert_eq!(s, StatusCode::NOT_MODIFIED);
            bytes += b.len();
        }
        let per_304 = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let (s, _, _) = call(&state, get(None)).await;
            assert_eq!(s, StatusCode::OK);
        }
        let per_200 = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(payload(&state).to_string());
        }
        let per_build = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let r = Request::get("/static/no-such-file")
                .body(Body::empty())
                .unwrap();
            std::hint::black_box(call(&state, r).await);
        }
        let per_route = t.elapsed() / n;
        println!(
            "full body {} bytes; 304 poll {per_304:?} ({} body bytes); 200 poll {per_200:?}; \
             build+stringify {per_build:?}; test router round trip alone {per_route:?}; \
             answers built over {} polls: {}",
            b.len(),
            bytes / n as usize,
            2 * n + 1,
            answers_built(&state)
        );
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
