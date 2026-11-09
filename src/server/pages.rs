//! Read-only pages and assets: welcome, stats, test, logs, static files, fallback.

use super::config_page::LevelChoice;
use super::detect::{Upload, new_request_id, parse_detection_body, run_detection};
use super::devices::{devices_snapshot, force_cpu_note, profile_view};
use super::{AppState, blocking, default_name, render, state_view};
use crate::config::LogLevel;
use crate::metrics::{DropReason, Histogram};
use crate::registry::ModelRegistry;
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use askama::Template;
use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const CSS: &str = "text/css; charset=utf-8";
const JS: &str = "text/javascript; charset=utf-8";

/// The text of `assets/<file>` in a release build. A debug build embeds nothing: `include_str!`
/// alone makes cargo recompile the crate when the file changes, however the body is used, so the
/// embedded copy has to go for a page edit to need no rebuild ([`static_body`] reads the file).
#[cfg(not(debug_assertions))]
macro_rules! embedded {
    ($file:literal) => {
        include_str!(concat!("../../assets/", $file))
    };
}

#[cfg(debug_assertions)]
macro_rules! embedded {
    ($file:literal) => {
        ""
    };
}

macro_rules! asset {
    ($file:literal, $content_type:expr) => {
        ($file, $content_type, embedded!($file))
    };
}

/// Files served under `/static/<name>`: name, Content-Type, embedded body (empty in a debug
/// build, use [`static_body`]). Every page script lives here (the CSP allows `script-src 'self'`
/// only: no inline scripts, handlers or style attributes); pages pass server values through
/// `data-*` attributes.
pub(super) static STATIC_FILES: &[(&str, &str, &str)] = &[
    asset!("style.css", CSS),
    asset!("live.js", JS),
    asset!("bench-glossary.js", JS),
    asset!("bench-table.js", JS),
    asset!("bench-charts.js", JS),
    asset!("bench-compare.js", JS),
    asset!("bench-lineup.js", JS),
    asset!("bench-profiles.js", JS),
    asset!("benchmark.js", JS),
    asset!("config.js", JS),
    asset!("logs.js", JS),
    asset!("stats.js", JS),
    asset!("test.js", JS),
    asset!("training.js", JS),
    asset!("train-area.js", JS),
    asset!("train-dataset.js", JS),
    asset!("train-metrics.js", JS),
    asset!("browse.js", JS),
    asset!("class-picker.js", JS),
    asset!("model-picker.js", JS),
    asset!("review-rules.js", JS),
    asset!("review-touch.js", JS),
    asset!("review.js", JS),
    asset!("welcome.js", JS),
    asset!("manifest.webmanifest", "application/manifest+json"),
];

/// `Cache-Control` of a static file. A debug build must show an edit on the next reload.
const STATIC_CACHE: &str = if cfg!(debug_assertions) {
    "no-cache"
} else {
    "public, max-age=3600"
};

/// The body of the static file `name`: the embedded copy.
#[cfg(not(debug_assertions))]
fn static_body(_name: &str, embedded: &'static str) -> std::io::Result<Cow<'static, str>> {
    Ok(Cow::Borrowed(embedded))
}

/// The body of the static file `name`, read from the source tree on every request, so a JS or CSS
/// edit needs a reload and no rebuild. There is no embedded copy to fall back on ([`embedded`]): a
/// debug binary moved off its source tree fails here, and says so.
#[cfg(debug_assertions)]
fn static_body(name: &str, _embedded: &'static str) -> std::io::Result<Cow<'static, str>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name);
    std::fs::read_to_string(&path)
        .map(Cow::Owned)
        .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

/// Body of the static file `name` (`"benchmark.js"`), for tests that check page code.
#[cfg(test)]
pub(super) fn static_file(name: &str) -> Option<&'static str> {
    let (_, _, embedded) = STATIC_FILES.iter().find(|(n, _, _)| *n == name)?;
    let body = static_body(name, embedded).ok()?;
    Some(Box::leak(body.into_owned().into_boxed_str()))
}

static FAVICON_ICO: &[u8] = include_bytes!("../../assets/favicon.ico");

const PNG: &str = "image/png";

/// `Cache-Control` of the icons: they change with a release at most, and a stale one only looks old.
const ICON_CACHE: &str = "public, max-age=86400";

/// Binary files served under `/static/<name>` (flat names, `{file}` matches no `/`): name,
/// Content-Type, body. Always embedded, in a debug build too: they are made by
/// `scripts/gen_icons.py` (`--check` in CI), never edited by hand. The manifest
/// (`manifest.webmanifest`) lists the `icon-*` ones; `/apple-touch-icon.png` serves the first.
pub(super) static STATIC_BINARY: &[(&str, &str, &[u8])] = &[
    (
        "apple-touch-icon.png",
        PNG,
        include_bytes!("../../assets/icons/apple-touch-icon.png"),
    ),
    (
        "icon-192.png",
        PNG,
        include_bytes!("../../assets/icons/icon-192.png"),
    ),
    (
        "icon-512.png",
        PNG,
        include_bytes!("../../assets/icons/icon-512.png"),
    ),
    (
        "icon-maskable-512.png",
        PNG,
        include_bytes!("../../assets/icons/icon-maskable-512.png"),
    ),
];

/// State text for the HTML pages and `/stats`: "Initializing (downloading ...)", "Ready", ...;
/// a lazy model that has not been requested yet says so.
fn state_label(w: &WorkerHandle) -> String {
    let s = w.state.get();
    match (&s, w.lazy) {
        (ModelState::Initializing, true) if w.accepts_while_initializing() => {
            "Lazy (loads on first request)".to_string()
        }
        _ => w.state.describe(),
    }
}

/// `1d 2h 3m 4s`, leading zero units omitted.
fn format_uptime(d: Duration) -> String {
    let s = d.as_secs();
    let (days, h, m, sec) = (s / 86_400, s / 3600 % 24, s / 60 % 60, s % 60);
    match (days, h, m) {
        (0, 0, 0) => format!("{sec}s"),
        (0, 0, _) => format!("{m}m {sec}s"),
        (0, _, _) => format!("{h}h {m}m {sec}s"),
        _ => format!("{days}d {h}h {m}m {sec}s"),
    }
}

// ---------------------------------------------------------------------------------------------
// Welcome page

struct ModelRow {
    name: String,
    is_default: bool,
    /// Short state text ("Ready", "Loading", "Waiting for download", "Failed:", "Lazy").
    state: String,
    /// CSS class of the state badge: `ok`, `loading`, `wait`, `fail`, `lazy`.
    state_class: &'static str,
    /// Download progress, failure reason, ...
    state_detail: Option<String>,
    provider: String,
    requests: u64,
    queue: String,
    /// What a restart would change for this model ("" = nothing).
    pending: String,
}

/// Distinct execution providers of the loaded models, in config order.
fn providers_in_use(reg: &ModelRegistry) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in reg.workers() {
        let p = w
            .device
            .read()
            .ok()
            .and_then(|d| d.as_ref().map(|d| d.execution_provider()));
        if let Some(p) = p
            && !out.contains(&p)
        {
            out.push(p);
        }
    }
    out
}

#[derive(Template)]
#[template(path = "welcome.html")]
struct WelcomeTemplate {
    nav: &'static str,
    version: &'static str,
    openvino_version: String,
    onnxruntime: String,
    devices: String,
    gpus: Vec<String>,
    auto_pick: String,
    /// The hardware profile line ("low-spec: 6 logical cores, ...").
    profile: String,
    /// Distinct execution providers of the loaded models.
    in_use: String,
    force_cpu_note: Option<String>,
    /// `openvino_dir` / bundled `./openvino` overridden by `OPENVINO_INSTALL_DIR` (BE-08).
    openvino_note: Option<String>,
    uptime: String,
    port: u16,
    default_model: String,
    models: Vec<ModelRow>,
    /// The last default switch, while it can be reverted.
    revert: Option<crate::training::switch::RevertView>,
    /// The dataset collector: collecting or not, and the frames so far.
    dataset_line: String,
    /// Why collection stopped on its own, while it is stopped.
    dataset_stopped: Option<String>,
    /// `collect.enabled`: off, Home suggests it in one line.
    collecting: bool,
}

pub(super) async fn welcome(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let reg = &state.registry;
        let snap = devices_snapshot(&state);
        let default = default_name(reg);
        let pending = crate::config_merge::pending(&state.running(), &state.config_read());
        let models = reg
            .workers()
            .iter()
            .map(|w| {
                let (state, state_class, state_detail) = state_view(w);
                ModelRow {
                    pending: pending.for_model(&w.name).join("; "),
                    name: w.name.clone(),
                    is_default: w.name == default,
                    state,
                    state_class,
                    state_detail,
                    provider: w.execution_provider(),
                    requests: w.metrics.requests.load(Ordering::Relaxed),
                    queue: format!("{}/{}", w.sender.len(), w.queue_capacity()),
                }
            })
            .collect();
        let in_use = providers_in_use(reg);
        let force_cpu_note = force_cpu_note(&state.config_read(), &snap);
        let (collecting, dataset_dir) = {
            let cfg = state.config_read();
            (
                cfg.collect.enabled,
                crate::collect::CollectSettings::for_config(&cfg).dir,
            )
        };
        let dataset = crate::collect::global().brief_of(&dataset_dir);
        render(&WelcomeTemplate {
            nav: "home",
            version: crate::VERSION,
            openvino_version: if reg.runtime_info.openvino_version.is_empty() {
                "not available (run setup-openvino)".to_string()
            } else {
                reg.runtime_info.openvino_version.clone()
            },
            onnxruntime: snap.onnxruntime_label(),
            openvino_note: crate::backend::libs::openvino_dir_override_warning(),
            devices: if reg.runtime_info.available_devices.is_empty() {
                "none".to_string()
            } else {
                reg.runtime_info.available_devices.join(", ")
            },
            gpus: snap.hardware.gpus.iter().map(|g| g.to_string()).collect(),
            auto_pick: snap.auto_pick_label(),
            profile: {
                let (p, n) = profile_view(&state);
                p.summary(n)
            },
            in_use: if in_use.is_empty() {
                "nothing loaded yet".to_string()
            } else {
                in_use.join("; ")
            },
            force_cpu_note,
            uptime: format_uptime(state.started.elapsed()),
            port: state.config_read().port,
            default_model: default,
            models,
            revert: crate::training::switch::revert_view(&state.config_read()),
            dataset_line: crate::collect::health::home_line(
                collecting,
                dataset.disabled.is_some(),
                dataset.summary.frames,
            ),
            dataset_stopped: dataset.disabled,
            collecting,
        })
    })
    .await
}

pub(super) async fn update_available(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if !state.config_read().check_updates {
        return Json(crate::update::check_off(crate::VERSION));
    }
    Json(crate::update::check_update(crate::VERSION).await)
}

// ---------------------------------------------------------------------------------------------
// Stats

/// One timing in one cell: the average, with min-max under it (`seen` false = no requests yet).
struct TimingCells {
    avg: String,
    min: String,
    max: String,
    seen: bool,
}

impl TimingCells {
    fn from_stat(s: &Histogram) -> Self {
        if s.count() == 0 {
            let dash = || "-".to_string();
            return Self {
                avg: dash(),
                min: dash(),
                max: dash(),
                seen: false,
            };
        }
        Self {
            avg: format!("{:.1}", s.avg()),
            min: s.min().to_string(),
            max: s.max().to_string(),
            seen: true,
        }
    }
}

struct StatsRow {
    name: String,
    is_default: bool,
    /// Short state ("Ready", "Failed:", "Loading"), as on the Home page.
    state: String,
    state_class: &'static str,
    /// The failure, download or compile detail under the short state.
    state_detail: Option<String>,
    device: String,
    provider: String,
    /// "OpenVINO 2026.0.0", "ONNX Runtime 1.24.4"; "-" before the model has loaded.
    runtime: String,
    fell_back: &'static str,
    requests: u64,
    dropped: u64,
    queue: String,
    /// Inference, process, round trip.
    timings: [TimingCells; 3],
}

#[derive(Template)]
#[template(path = "stats.html")]
struct StatsTemplate {
    nav: &'static str,
    version: &'static str,
    uptime: String,
    models: Vec<StatsRow>,
}

/// Runtime name and version a loaded model runs on ("OpenVINO 2026.0.0").
fn runtime_text(
    reg: &ModelRegistry,
    ort_version: &Option<String>,
    dev: Option<&crate::backend::DeviceInfo>,
) -> String {
    use crate::backend::spec::Runtime;
    match dev.map(|d| d.runtime) {
        None => "-".to_string(),
        Some(Runtime::OpenVino) => {
            // "2026.4.0-22959-99c81491cc3-releases/2026/4 (OpenVINO Runtime)" -> "2026.4.0".
            let v = &reg.runtime_info.openvino_version;
            let short = v.split(['-', ' ']).next().unwrap_or(v);
            format!("OpenVINO {short}").trim().to_string()
        }
        Some(Runtime::Ort) => match ort_version {
            Some(v) => format!("ONNX Runtime {v}"),
            None => "ONNX Runtime".to_string(),
        },
    }
}

/// Version of the loaded ONNX Runtime: the shared runtimes' when they are free to ask, else
/// (a worker compiles) the one recorded at startup ([`crate::registry::StartupDevices`]).
pub(super) fn ort_version(reg: &ModelRegistry) -> Option<String> {
    let live = reg.runtimes().and_then(|rt| {
        let rt = rt.try_lock().ok()?;
        Some(rt.onnxruntime_version())
    });
    match live {
        Some(v) => v,
        None => reg
            .startup_devices()
            .and_then(|s| s.onnxruntime_version.clone()),
    }
}

pub(super) async fn stats_page(State(state): State<Arc<AppState>>) -> Response {
    let reg = &state.registry;
    let default = default_name(reg);
    let ortv = ort_version(reg);
    let models = reg
        .workers()
        .iter()
        .map(|w| {
            let m = &w.metrics;
            let dev = w.device.read().ok().and_then(|d| d.clone());
            let (state, state_class, state_detail) = state_view(w);
            StatsRow {
                name: w.name.clone(),
                is_default: w.name == default,
                state,
                state_class,
                state_detail,
                runtime: runtime_text(reg, &ortv, dev.as_ref()),
                device: dev
                    .as_ref()
                    .map(|d| d.actual.clone())
                    .unwrap_or_else(|| m.requested_device()),
                provider: w.execution_provider(),
                fell_back: match dev.as_ref().map(|d| d.fell_back) {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "-",
                },
                requests: m.requests.load(Ordering::Relaxed),
                dropped: m.dropped_failures(),
                queue: format!("{}/{}", w.sender.len(), w.queue_capacity()),
                timings: [
                    TimingCells::from_stat(&m.inference_ms),
                    TimingCells::from_stat(&m.process_ms),
                    TimingCells::from_stat(&m.round_trip_ms),
                ],
            }
        })
        .collect();
    render(&StatsTemplate {
        nav: "stats",
        version: crate::VERSION,
        uptime: format_uptime(state.started.elapsed()),
        models,
    })
}

pub(super) async fn stats_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let reg = &state.registry;
    let default = reg.default_model().map(|w| w.name.clone());
    let ortv = ort_version(reg);
    let models: Vec<serde_json::Value> = reg
        .workers()
        .iter()
        .map(|w| {
            let m = &w.metrics;
            let dev = w.device.read().ok().and_then(|d| d.clone());
            let stat = |s: &Histogram| {
                serde_json::json!({ "count": s.count(), "avg": s.avg(), "min": s.min(), "max": s.max() })
            };
            let (short, class, detail) = state_view(w);
            serde_json::json!({
                "name": w.name,
                "default": default.as_deref() == Some(w.name.as_str()),
                "state": w.state.describe(),
                // Short state + detail as the pages show them (Home, Stats): the failure,
                // download or compile detail goes on its own line.
                "stateShort": short,
                "stateDetail": detail,
                "stateText": state_label(w),
                "stateClass": class,
                "runtime": runtime_text(reg, &ortv, dev.as_ref()),
                "lazy": w.lazy,
                "reloads": reg.reloads(&w.name),
                "requestedDevice": m.requested_device(),
                "device": dev.as_ref().map(|d| d.actual.clone()),
                "deviceName": dev.as_ref().map(|d| d.full_name.clone()),
                "fellBack": dev.as_ref().map(|d| d.fell_back),
                "executionProvider": w.execution_provider(),
                "queueLength": w.sender.len(),
                "queueCapacity": w.queue_capacity(),
                "requests": m.requests.load(Ordering::Relaxed),
                "dropped": m.dropped_failures(),
                "rejectedInitializing": m.dropped(DropReason::Initializing),
                "inferenceMs": stat(&m.inference_ms),
                "processMs": stat(&m.process_ms),
                "roundTripMs": stat(&m.round_trip_ms),
            })
        })
        .collect();
    let process = crate::metrics::ProcessStats::read();
    Json(serde_json::json!({
        "version": crate::VERSION,
        "uptimeSecs": state.metrics.uptime().as_secs(),
        "uptime": format_uptime(state.started.elapsed()),
        "openvino": reg.runtime_info,
        "models": models,
        "rejectedOverloaded": state.metrics.rejected_overloaded.load(Ordering::Relaxed),
        "inflightRequests": state.metrics.inflight(),
        // null where the platform has no cheap source; the footprint is macOS only.
        "residentMemoryBytes": process.resident_memory_bytes,
        "memoryFootprintBytes": process.memory_footprint_bytes,
    }))
}

pub(super) async fn prometheus(State(state): State<Arc<AppState>>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        crate::metrics::render_prometheus(&state.metrics, &state.registry.gauges()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------------------------
// Test page

struct ModelChoice {
    value: String,
    selected: bool,
    is_default: bool,
}

struct TestResult {
    summary: String,
    /// The sentence shown when nothing was found ([`no_objects_line`]); empty otherwise.
    none_found: String,
    /// Its hint (how to see weaker boxes); empty when there is none.
    none_hint: &'static str,
    /// `data:image/jpeg;base64,...` of the annotated image; empty when it could not be drawn.
    image_data_uri: String,
    json: String,
}

/// A threshold as the Test page shows it: two decimals, three when they matter (0.375).
fn fmt_threshold(t: f32) -> String {
    let s = format!("{t:.3}");
    s.strip_suffix('0').map(str::to_string).unwrap_or(s)
}

/// The Test page's sentence for a detection that found nothing: the threshold it ran at, as the
/// worker reads it ([`crate::worker::effective_threshold`]: the form's Min confidence when it
/// is set, a value over 1 being a percentage, else the model's `model` threshold, with its
/// `night` threshold named when it differs), and how to see weaker boxes ([`LOWER_HINT`],
/// unless the threshold is already the floor). `assets/test.js` `noneLine` is its twin.
fn no_objects_line(min_confidence: &str, model: f32, night: Option<f32>) -> (String, &'static str) {
    use crate::worker::effective_threshold;
    let asked = min_confidence.trim().parse::<f32>().unwrap_or(0.0);
    let t = effective_threshold(asked, model);
    let from_request = t != effective_threshold(0.0, model);
    let night = night
        .filter(|_| !from_request)
        .map(|n| effective_threshold(asked, n))
        .filter(|n| fmt_threshold(*n) != fmt_threshold(t));
    let mut line = format!("No objects at or above {}", fmt_threshold(t));
    if let Some(n) = night {
        line.push_str(&format!(" ({} on night and IR frames)", fmt_threshold(n)));
    }
    line.push('.');
    let hint = if t.min(night.unwrap_or(t)) > crate::model::nms::MIN_CONFIDENCE_FLOOR {
        LOWER_HINT
    } else {
        ""
    };
    (line, hint)
}

/// The hint under "No objects at or above ..." (also in `assets/test.js`).
const LOWER_HINT: &str = "Lower Min confidence to see weaker detections.";

#[derive(Template)]
#[template(path = "test.html")]
struct TestTemplate {
    nav: &'static str,
    version: &'static str,
    models: Vec<ModelChoice>,
    /// `{"<model>": {"t": <threshold now>, "night": <night_threshold>|null}}`: what test.js
    /// names when a detection finds nothing (the select's `data-thresholds`).
    thresholds: String,
    min_confidence: String,
    error: Option<String>,
    result: Option<TestResult>,
}

fn test_template(
    state: &AppState,
    selected: &str,
    min_confidence: String,
    error: Option<String>,
    result: Option<TestResult>,
) -> TestTemplate {
    let default = default_name(&state.registry);
    let selected = if selected.is_empty() {
        default.as_str()
    } else {
        selected
    };
    let models = state
        .registry
        .names()
        .into_iter()
        .map(|n| ModelChoice {
            selected: crate::registry::normalize_name(&n)
                == crate::registry::normalize_name(selected),
            is_default: n == default,
            value: n,
        })
        .collect::<Vec<_>>();
    let thresholds: serde_json::Map<String, serde_json::Value> = models
        .iter()
        .filter_map(|m| {
            let h = state.registry.by_name(&m.value)?;
            // Three decimals, as shown (an f32 0.21 would read 0.20999999344348907).
            let round = |t: f32| (f64::from(t) * 1000.0).round() / 1000.0;
            let t = serde_json::json!({
                "t": round(h.effective_threshold().0),
                "night": h.live().night_threshold.map(round),
            });
            Some((m.value.clone(), t))
        })
        .collect();
    TestTemplate {
        nav: "test",
        version: crate::VERSION,
        models,
        thresholds: serde_json::Value::Object(thresholds).to_string(),
        min_confidence,
        error,
        result,
    }
}

pub(super) async fn test_page(State(state): State<Arc<AppState>>) -> Response {
    render(&test_template(&state, "", String::new(), None, None))
}

/// Same dispatch path as the API (metrics and queues behave identically), except that the upload
/// is no camera's frame and stays out of the dataset; then draws the predictions on the uploaded
/// image off the async thread.
pub(super) async fn test_submit(State(state): State<Arc<AppState>>, upload: Upload) -> Response {
    let start = Instant::now();
    let id = new_request_id();
    let (mut req, fields) = match parse_detection_body(upload).await {
        Ok(v) => v,
        Err(e) => {
            e.record(&state);
            return render(&test_template(
                &state,
                "",
                String::new(),
                Some(e.message),
                None,
            ));
        }
    };

    let model = fields.get("model").map(|s| s.trim()).unwrap_or_default();
    let min_conf = fields.get("min_confidence").cloned().unwrap_or_default();
    let (handle, command) = if model.is_empty() {
        (state.registry.default_model(), "detect")
    } else {
        (state.registry.by_name(model), "custom")
    };
    let Some(handle) = handle else {
        let e = if model.is_empty() {
            "No default model: enable a model on the Config page".to_string()
        } else {
            format!("Unknown model '{model}'")
        };
        return render(&test_template(&state, model, min_conf, Some(e), None));
    };

    req.no_collect = true;
    let image = req.image_data.clone();
    let resp = run_detection(&state, &handle, Ok(req), start, command, &id).await;
    let json = serde_json::to_string_pretty(&resp).unwrap_or_default();
    let mut summary = if resp.success {
        format!(
            "{} object(s) from model '{}' on {}: inference {} ms, process {} ms, round trip {} ms.",
            resp.count,
            handle.name,
            resp.executionProvider,
            resp.inferenceMs,
            resp.processMs,
            resp.analysisRoundTripMs
        )
    } else {
        format!("Detection failed: {}", resp.message)
    };
    let (none_found, none_hint) = if resp.success && resp.predictions.is_empty() {
        no_objects_line(
            &min_conf,
            handle.effective_threshold().0,
            handle.live().night_threshold,
        )
    } else {
        (String::new(), "")
    };
    let mut image_data_uri = String::new();
    if resp.success {
        let preds = resp.predictions.clone();
        let drawn = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let img = crate::image::decode(&image)?;
            let jpeg =
                crate::image::encode_jpeg(&crate::image::draw_predictions(&img, &preds), 85)?;
            Ok(format!(
                "data:image/jpeg;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(jpeg)
            ))
        })
        .await;
        match drawn {
            Ok(Ok(uri)) => image_data_uri = uri,
            Ok(Err(e)) => summary.push_str(&format!(" (could not draw the image: {e:#})")),
            Err(e) => summary.push_str(&format!(" (could not draw the image: {e})")),
        }
    }
    let selected = handle.name.clone();
    render(&test_template(
        &state,
        &selected,
        min_conf,
        None,
        Some(TestResult {
            summary,
            none_found,
            none_hint,
            image_data_uri,
            json,
        }),
    ))
}

// ---------------------------------------------------------------------------------------------
// Logs page

#[derive(Template)]
#[template(path = "logs.html")]
struct LogsTemplate {
    nav: &'static str,
    version: &'static str,
    log_levels: Vec<LevelChoice>,
    /// Log capture is available (this process initialized logging).
    available: bool,
    capacity: usize,
}

pub(super) fn level_choices(current: &str) -> Vec<LevelChoice> {
    [
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Error,
    ]
    .iter()
    .map(|l| LevelChoice {
        value: l.as_str(),
        selected: l.as_str().eq_ignore_ascii_case(current.trim()),
    })
    .collect()
}

pub(super) async fn logs_page(State(state): State<Arc<AppState>>) -> Response {
    let level = state.config_read().log_level.as_str().to_string();
    render(&LogsTemplate {
        nav: "logs",
        version: crate::VERSION,
        log_levels: level_choices(&level),
        available: state.log_buffer().is_some(),
        capacity: crate::logbuf::DEFAULT_CAPACITY,
    })
}

/// `GET /logs.json?after=<seq>&level=<min>&limit=<n>`: captured log events newer than `after`
/// at `level` or more severe, oldest first. `last` is the cursor for the next poll; a `last`
/// below the client's cursor means the process restarted.
pub(super) async fn logs_json(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    let bad = |msg: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "message": msg })),
        )
            .into_response()
    };
    let after = match q.get("after").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(n) => n,
            Err(_) => return bad(format!("after: '{v}' is not a number")),
        },
    };
    let min = match q.get("level").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => None,
        Some(v) if v.eq_ignore_ascii_case("all") => None,
        Some(v) => match crate::logbuf::parse_level(v) {
            Some(l) => Some(l),
            None => return bad(format!("level: '{v}' is not trace|debug|info|warn|error")),
        },
    };
    let limit = q
        .get("limit")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(crate::logbuf::MAX_PAGE)
        .clamp(1, crate::logbuf::MAX_PAGE);
    let level = state.config_read().log_level.as_str();
    match state.log_buffer() {
        Some(buf) => {
            let page = buf.since(after, min, limit);
            Json(serde_json::json!({
                "success": true,
                "level": level,
                "entries": page.entries,
                "last": page.last,
                "first": page.first,
                "truncated": page.truncated,
            }))
            .into_response()
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "log capture is not available in this process",
            "level": level,
            "entries": [],
            "last": 0,
            "first": 1,
            "truncated": false,
        }))
        .into_response(),
    }
}

// ---------------------------------------------------------------------------------------------
// Static assets

/// One entry of [`STATIC_FILES`] or [`STATIC_BINARY`]; 404 (JSON, like the router's fallback) for
/// any other name.
fn serve_static(name: &str) -> Response {
    if let Some((_, content_type, body)) = STATIC_BINARY.iter().find(|(n, _, _)| *n == name) {
        return (
            [
                (header::CONTENT_TYPE, *content_type),
                (header::CACHE_CONTROL, ICON_CACHE),
            ],
            *body,
        )
            .into_response();
    }
    let Some((_, content_type, embedded)) = STATIC_FILES.iter().find(|(n, _, _)| *n == name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": format!("Not found: /static/{name}"),
            })),
        )
            .into_response();
    };
    match static_body(name, embedded) {
        Ok(body) => (
            [
                (header::CONTENT_TYPE, *content_type),
                (header::CACHE_CONTROL, STATIC_CACHE),
            ],
            body,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "error": format!(
                    "A debug build reads its pages from the source tree and could not read one \
                     ({e}). Run it where it was built, or use a release build."
                ),
            })),
        )
            .into_response(),
    }
}

/// `GET /static/{file}`: the page scripts, the stylesheet, the web app manifest and its icons.
pub(super) async fn static_asset(Path(file): Path<String>) -> Response {
    serve_static(&file)
}

pub(super) async fn style_css() -> Response {
    serve_static("style.css")
}

/// Shared page script: polls `GET /v1/config` and keeps the header's pending-restart indicator
/// (and the pages that listen) up to date.
pub(super) async fn live_js() -> Response {
    serve_static("live.js")
}

/// `GET /apple-touch-icon.png` (and `-precomposed`): iOS asks the site's root for it when a page is
/// added to the Home Screen without a `<link rel="apple-touch-icon">` it could read, so it is open.
pub(super) async fn apple_touch_icon() -> Response {
    serve_static("apple-touch-icon.png")
}

pub(super) async fn favicon() -> Response {
    (
        [
            (header::CONTENT_TYPE, "image/x-icon"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        FAVICON_ICO,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::server::test_util::*;

    use crate::cli::LogReloadHandle;
    use crate::config::{Config, LogLevel};

    use axum::http::{StatusCode, header};

    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn the_update_check_can_be_turned_off_and_then_asks_nobody() {
        let state = test_state_with(|c| c.check_updates = false);
        let r = update_available(State(state)).await.into_response();
        assert_eq!(r.status(), StatusCode::OK);
        let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(
            (v["success"].as_bool(), v["updateAvailable"].as_bool()),
            (Some(false), Some(false))
        );
        assert!(
            v["message"].as_str().unwrap().contains("check_updates"),
            "{v}"
        );
    }

    use axum::body::Body;
    use axum::http::Request;

    #[test]
    fn uptime_format() {
        assert_eq!(format_uptime(Duration::from_secs(5)), "5s");
        assert_eq!(format_uptime(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_uptime(Duration::from_secs(3 * 3600 + 5)), "3h 0m 5s");
        assert_eq!(format_uptime(Duration::from_secs(90_061)), "1d 1h 1m 1s");
    }

    #[tokio::test]
    async fn pages_render_and_escape() {
        let state = test_state();
        let (s, home) = get_text(&state, "/").await;
        assert_eq!(s, StatusCode::OK);
        assert!(home.contains("2026.0.0-test"), "{home}");
        // The footer names the build: version, git commit and build date.
        assert!(
            home.contains(&format!(
                "Blue Onyx Prism {} &middot;",
                crate::long_version()
            )),
            "{home}"
        );
        assert!(home.contains("<td>ipcam-general</td>"));
        // The HTTP integration tests poll for `<td>Ready</td>` / `Failed:` in this page.
        assert!(home.contains(
            "<td>Failed:<br><span class=\"detail\">model file not found: x.onnx</span></td>"
        ));
        assert!(home.contains("<tr class=\"st-fail\">"));
        // A restart badge's reasons open on a tap (live.js PrismTip), not only as a title.
        assert!(
            home.contains("<button type=\"button\" class=\"term tag restart pending-badge\" data-model=\"ipcam-general\" data-tip=\""),
            "{home}"
        );
        assert!(home.contains("Running on") && home.contains("nothing loaded yet"));
        assert!(home.contains("OpenVINO devices"));
        assert!(!home.contains("Force CPU is on"));
        // Prometheus moved to the footer, Logs is in the nav.
        assert!(home.contains("<a href=\"/logs\">Logs</a>"));
        assert!(home.contains("<a href=\"/prometheus\">Prometheus metrics</a>"));
        assert!(!home.contains("<b>evil</b>") && home.contains("&#60;b&#62;evil"));
        assert!(home.contains("href=\"/static/style.css?v="));

        let (s, stats) = get_text(&state, "/stats").await;
        assert_eq!(s, StatusCode::OK);
        assert!(!stats.contains("http-equiv=\"refresh\""));
        assert!(stats.contains("<script src=\"/static/stats.js?v="));
        assert!(static_file("stats.js").is_some_and(|js| js.contains("fetch(\"/stats.json\"")));
        assert!(stats.contains("class=\"stats sticky-first rcards rgrid\""));
        // The device spec shows under the provider (it was a title only, which a finger cannot
        // open).
        assert!(
            stats.contains("ipcam-general")
                && stats.contains("data-label=\"Device\" title=\"not loaded\">not loaded<span class=\"sub\">CPU</span></td>"),
            "{stats}"
        );
        assert!(!stats.contains("<b>evil</b>"));
        // Short state, the reason on its own line (as on Home).
        assert!(
            stats.contains("<td class=\"state\">Failed:<br><span class=\"detail\">"),
            "{stats}"
        );

        let (s, json) = get_text(&state, "/stats.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["models"][0]["name"], "ipcam-general");
        assert_eq!(v["models"][0]["default"], true);
        assert!(v["residentMemoryBytes"].as_u64().is_some_and(|b| b > 0));
        assert_eq!(
            v["memoryFootprintBytes"].as_u64().is_some(),
            cfg!(target_os = "macos")
        );
        assert_eq!(v["models"][0]["stateClass"], "fail");
        assert_eq!(v["models"][0]["stateShort"], "Failed:");
        assert!(
            v["models"][0]["stateDetail"]
                .as_str()
                .is_some_and(|d| !d.is_empty())
        );
        assert!(static_file("stats.js").is_some_and(|js| js.contains("m.stateShort")));
        assert_eq!(v["models"][0]["runtime"], "-");
        assert_eq!(v["models"][0]["executionProvider"], "not loaded");

        let (s, test) = get_text(&state, "/test").await;
        assert_eq!(s, StatusCode::OK);
        assert!(test.contains("<option value=\"ipcam-general\" selected>ipcam-general (default)"));
        assert!(test.contains("enctype=\"multipart/form-data\""));

        let (s, cfg) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(cfg.contains("name=\"models_json\""));
        assert!(cfg.contains("&#34;ipcam-general&#34;"), "{cfg}");
        assert!(cfg.contains("<option value=\"info\" selected>"));
        // One log level control on this page (the Server form); the JSON is under Advanced.
        assert_eq!(cfg.matches("name=\"log_level\"").count(), 1);
        assert!(!cfg.contains("action=\"/config/loglevel\""));
        assert!(cfg.contains("<summary>Advanced: Models (JSON)</summary>"));
        assert!(cfg.contains("nav class=\"toc tabs\" role=\"tablist\""));
        assert!(cfg.contains("Local models (not in config)"));
        // Tab panels and the deep-link anchors other pages and the README use.
        for id in [
            "tab-models",
            "tab-resources",
            "tab-server",
            "at-models",
            "at-server",
            "at-restart",
            "inference",
            "logging",
            "resources",
            "local-models",
            "custom-model",
            "models-json",
            "pending-box",
        ] {
            assert!(cfg.contains(&format!("id=\"{id}\"")), "missing #{id}");
        }
        // The models JSON has its own form; the Server form carries the checkbox markers.
        assert!(cfg.contains("id=\"models-json-form\""));
        assert!(cfg.contains("name=\"inference_settings\""));
        assert!(cfg.contains("name=\"image_settings\""));
    }

    #[test]
    fn a_debug_build_embeds_no_page_so_a_page_edit_needs_no_rebuild() {
        for (name, _, embedded) in STATIC_FILES {
            assert_eq!(embedded.is_empty(), cfg!(debug_assertions), "{name}");
        }
        assert_eq!(
            STATIC_CACHE == "no-cache",
            cfg!(debug_assertions),
            "a debug build must show an edit on the next reload"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_debug_build_names_the_page_it_cannot_read() {
        let e = static_body("no-such-page.js", "").unwrap_err().to_string();
        assert!(e.contains("no-such-page.js"), "{e}");
    }

    #[tokio::test]
    async fn static_assets() {
        let state = test_state();
        let (s, h, b) = call(
            &state,
            Request::get("/favicon.ico").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "image/x-icon");
        assert_eq!(&b[..4], &[0, 0, 1, 0]);
        let (s, h, b) = call(
            &state,
            Request::get("/static/style.css")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/css")
        );
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("prefers-color-scheme")
        );
        let (s, _) = get_text(&state, "/nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // Every page script and the stylesheet, with its Content-Type and cache policy.
        for (name, content_type, _) in STATIC_FILES {
            let body = static_file(name).unwrap();
            assert!(!body.is_empty(), "{name}");
            let (s, h, b) = call(
                &state,
                Request::get(format!("/static/{name}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{name}");
            assert_eq!(h[header::CONTENT_TYPE], *content_type, "{name}");
            assert_eq!(h[header::CACHE_CONTROL], STATIC_CACHE, "{name}");
            assert_eq!(b, body.as_bytes(), "{name}");
            if name.ends_with(".js") {
                assert!(content_type.starts_with("text/javascript"), "{name}");
                // Server values come from data-* attributes, never template tags.
                assert!(!body.contains("{{") && !body.contains("{%"), "{name}");
            }
        }
        for name in [
            "live.js",
            "benchmark.js",
            "class-picker.js",
            "model-picker.js",
            "config.js",
            "logs.js",
            "stats.js",
            "test.js",
            "welcome.js",
        ] {
            assert!(static_file(name).is_some(), "{name}");
        }
        for uri in ["/static/nope.js", "/static/..%2Fsrc%2Fmain.rs"] {
            let (s, _) = get_text(&state, uri).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{uri}");
        }
    }

    /// `uri` with no credential: over HTTPS from another machine, or on the plain port.
    async fn fetch_anonymous(
        state: &Arc<AppState>,
        uri: &str,
        on_plain_port: bool,
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let req = from_peer(Request::get(uri).body(Body::empty()).unwrap(), REMOTE);
        let req = if on_plain_port {
            plain(req)
        } else {
            anonymous(req)
        };
        call(state, req).await
    }

    /// The PNG at `uri`, answered to nobody signed in on both ports: its width and height and
    /// whether every pixel is opaque.
    async fn served_png(state: &Arc<AppState>, uri: &str) -> (u32, u32, bool) {
        let mut seen = None;
        for on_plain_port in [false, true] {
            let (s, h, b) = fetch_anonymous(state, uri, on_plain_port).await;
            assert_eq!(s, StatusCode::OK, "{uri} plain={on_plain_port}");
            assert_eq!(h[header::CONTENT_TYPE], "image/png", "{uri}");
            assert_eq!(h[header::CACHE_CONTROL], ICON_CACHE, "{uri}");
            // No session for nobody (the plain port only ever clears the legacy cookie).
            if !on_plain_port {
                assert!(!h.contains_key(header::SET_COOKIE), "{uri}");
            }
            if let Some(first) = &seen {
                assert_eq!(&b, first, "{uri}: the same bytes on both ports");
            }
            seen = Some(b);
        }
        let png = seen.unwrap();
        let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .unwrap_or_else(|e| panic!("{uri}: {e}"))
            .to_rgba8();
        let opaque = img.pixels().all(|p| p.0[3] == 255);
        (img.width(), img.height(), opaque)
    }

    /// A Home Screen app: the icons iOS asks the root for, the manifest Chrome reads and every
    /// icon it lists, all without a sign-in (browsers fetch them without cookies) and on the
    /// plain port too.
    #[tokio::test]
    async fn the_home_screen_icons_and_manifest_are_open() {
        let state = test_state();
        for uri in ["/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"] {
            // Opaque: iOS shows black where an icon is transparent.
            assert_eq!(served_png(&state, uri).await, (180, 180, true), "{uri}");
        }
        assert_eq!(
            served_png(&state, "/static/apple-touch-icon.png").await,
            (180, 180, true)
        );

        let (s, h, b) = fetch_anonymous(&state, "/static/manifest.webmanifest", true).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "application/manifest+json");
        let m: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(m["id"], "/");
        assert_eq!(m["name"], "Blue Onyx Prism");
        assert_eq!(m["short_name"], "Prism");
        assert_eq!(m["start_url"], "/");
        assert_eq!(m["scope"], "/");
        assert_eq!(m["display"], "standalone");
        assert_eq!(m["background_color"], "#0f2747");
        assert_eq!(m["theme_color"], "#0f2747");
        let icons = m["icons"].as_array().unwrap();
        let mut purposes = Vec::new();
        for icon in icons {
            let src = icon["src"].as_str().unwrap();
            assert_eq!(icon["type"], "image/png", "{src}");
            let (w, h, opaque) = served_png(&state, src).await;
            assert_eq!(icon["sizes"], format!("{w}x{h}"), "{src}");
            let purpose = icon["purpose"].as_str().unwrap();
            // A maskable icon is cropped by the launcher: it must have no transparent edge.
            if purpose == "maskable" {
                assert!(opaque, "{src}");
            }
            purposes.push((purpose.to_string(), w));
        }
        for want in [("any", 192), ("any", 512), ("maskable", 512)] {
            assert!(
                purposes.contains(&(want.0.to_string(), want.1)),
                "{want:?}: {purposes:?}"
            );
        }
        // Every binary file is served, and listed or the root icon.
        for (name, _, _) in STATIC_BINARY {
            let listed = icons.iter().any(|i| i["src"] == format!("/static/{name}"));
            assert!(listed || *name == "apple-touch-icon.png", "{name}");
        }
    }

    /// Every page links the manifest, the touch icon and a header colour per scheme; signed in,
    /// the nav and the JSON links are there.
    #[tokio::test]
    async fn pages_link_the_manifest_icon_and_theme_colours() {
        let state = test_state();
        let (s, home) = get_text(&state, "/").await;
        assert_eq!(s, StatusCode::OK);
        for want in [
            "<link rel=\"manifest\" href=\"/static/manifest.webmanifest?v=",
            "<link rel=\"apple-touch-icon\" href=\"/apple-touch-icon.png\">",
            "<meta name=\"apple-mobile-web-app-title\" content=\"Prism\">",
            "<meta name=\"theme-color\" content=\"#0f2747\" media=\"(prefers-color-scheme: light)\">",
            "<meta name=\"theme-color\" content=\"#0a0d12\" media=\"(prefers-color-scheme: dark)\">",
            "<nav aria-label=\"Main\"",
            "<div class=\"endpoints\">",
            "id=\"pending-restart\"",
        ] {
            assert!(home.contains(want), "{want}");
        }
        // The theme colours are the header's (`--head`), light then dark.
        let css = static_file("style.css").unwrap();
        let (light, dark) = css
            .split_once("@media (prefers-color-scheme: dark)")
            .unwrap();
        assert!(light.contains("--head: #0f2747;"));
        assert!(dark.contains("--head: #0a0d12;"));
        // A Home Screen app has no address bar to come back from raw JSON.
        assert!(css.contains(
            "@media (display-mode: standalone) {\n  footer .endpoints { display: none; }\n}"
        ));
    }

    /// A page for nobody signed in (login, setup, a refusal) has nothing that leads to a
    /// sign-in: no nav links, no JSON links, no restart indicator, no sign-out.
    #[tokio::test]
    async fn a_page_without_nav_links_nowhere_it_cannot_open() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::ACCEPT, "text/html".parse().unwrap());
        let page =
            super::super::guard::refused(StatusCode::FORBIDDEN, &headers, "Forbidden", "no".into());
        let body = axum::body::to_bytes(page.into_body(), usize::MAX)
            .await
            .unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        assert!(html.contains("<a class=\"brand\" href=\"/\">"), "{html}");
        assert!(html.contains("rel=\"manifest\""), "{html}");
        for absent in [
            "<nav",
            "href=\"/stats",
            "href=\"/config",
            "class=\"endpoints\"",
            "href=\"/v1/",
            "pending-restart",
            "action=\"/logout\"",
        ] {
            assert!(!html.contains(absent), "{absent}: {html}");
        }
    }

    /// The committed icons are what `scripts/gen_icons.py` draws (CI also runs `--check` in
    /// `scripts/ci/python_self_tests.sh`); skipped without a Python 3 unless
    /// `BOP_REQUIRE_PYTHON` is set.
    #[test]
    fn the_committed_icons_are_what_gen_icons_draws() {
        use crate::training::progress::test_python;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = root.join("scripts/gen_icons.py");
        let Some(py) = test_python::python3("gen_icons.py --check") else {
            return;
        };
        let out = test_python::run(py, &["-I", script.to_str().unwrap(), "--check"], "");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("icons ok"), "{text}");
    }

    /// Attribute names of every tag in `html` (quoted values skipped), with the tag text.
    fn tags(html: &str) -> Vec<(String, Vec<String>)> {
        let mut out = Vec::new();
        let mut rest = html;
        while let Some(start) = rest.find('<') {
            let after = &rest[start + 1..];
            let mut end = after.len();
            let mut quote: Option<char> = None;
            for (i, c) in after.char_indices() {
                match (quote, c) {
                    (Some(q), c) if c == q => quote = None,
                    (Some(_), _) => {}
                    (None, '"' | '\'') => quote = Some(c),
                    (None, '>') => {
                        end = i;
                        break;
                    }
                    _ => {}
                }
            }
            let tag = &after[..end];
            rest = after.get(end + 1..).unwrap_or("");
            // End tags, comments and the doctype carry no attributes.
            if tag.starts_with(['/', '!']) {
                continue;
            }
            let mut names = Vec::new();
            let mut quote: Option<char> = None;
            let mut word = String::new();
            for c in tag.chars() {
                match (quote, c) {
                    (Some(q), c) if c == q => quote = None,
                    (Some(_), _) => {}
                    (None, '"' | '\'') => quote = Some(c),
                    (None, '=') => {
                        names.push(std::mem::take(&mut word).to_ascii_lowercase());
                    }
                    (None, c) if c.is_whitespace() || c == '/' => {
                        if !word.is_empty() {
                            names.push(std::mem::take(&mut word).to_ascii_lowercase());
                        }
                    }
                    (None, c) => word.push(c),
                }
            }
            if !word.is_empty() {
                names.push(word.to_ascii_lowercase());
            }
            out.push((tag.to_string(), names));
        }
        out
    }

    /// The page works under `script-src 'self'; style-src 'self'`: every `<script>` loads a
    /// file or is a JSON data block, and no tag carries an `on*=` handler or a `style=`.
    fn assert_csp_clean(page: &str, html: &str) {
        for (tag, attrs) in tags(html) {
            // The first word is the element name; the rest are attributes.
            let (name, attrs) = attrs
                .split_first()
                .map_or(("", &[][..]), |(n, a)| (n.as_str(), a));
            if name == "script" {
                assert!(
                    attrs.iter().any(|a| a == "src") || tag.contains("type=\"application/json\""),
                    "{page}: inline <{tag}>"
                );
            }
            for a in attrs {
                assert!(
                    !(a.starts_with("on")
                        && a.len() > 2
                        && a[2..].bytes().all(|b| b.is_ascii_lowercase())),
                    "{page}: handler attribute in <{tag}>"
                );
                assert!(a != "style", "{page}: style attribute in <{tag}>");
            }
        }
        // Belt and braces on the raw text.
        assert!(!html.contains("<script>"), "{page}: inline script");
        assert!(!html.contains(" style=\""), "{page}: style attribute");
    }

    #[tokio::test]
    async fn pages_need_no_inline_script_or_style() {
        let state = test_state();
        for page in [
            "/",
            "/stats",
            "/test",
            "/config",
            "/logs",
            "/benchmark",
            "/train/collect",
            "/train/import",
            "/train/autolabel",
            "/train/review",
            "/train/dataset",
            "/train/train",
            "/train/models",
            "/train/jobs",
        ] {
            let (s, html) = get_text(&state, page).await;
            assert_eq!(s, StatusCode::OK, "{page}");
            assert!(html.contains("<script src=\"/static/live.js?v="), "{page}");
            assert_csp_clean(page, &html);
        }
        // A test result page (data: image, JSON in <pre>) and the logs page with capture.
        let img = crate::image::RgbImage {
            width: 8,
            height: 8,
            rgb: vec![100; 8 * 8 * 3],
        };
        let jpeg = crate::image::encode_jpeg(&img, 90).unwrap();
        let mut req = multipart(&[("image", &jpeg), ("model", b"ipcam-general")]);
        req.headers_mut()
            .insert(header::HOST, header::HeaderValue::from_static("127.0.0.1"));
        let (s, _, b) = call(&state, req).await;
        let html = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK, "{html}");
        assert!(html.contains("id=\"server-result\""), "{html}");
        assert_csp_clean("/test (POST)", &html);
        let (_, html) = get_text(&logs_state(), "/logs").await;
        assert!(html.contains("data-capacity=\"2000\""), "{html}");
        assert_csp_clean("/logs", &html);

        // The restart page passes its values as data attributes, HTML-escaped (SV-16).
        let html = super::super::config_page::MessageTemplate {
            nav: "config",
            version: crate::VERSION,
            heading: "Restarting".into(),
            message: "m".into(),
            refresh_secs: 5,
            refresh_url: "/config\"></p><script>alert(1)</script>".into(),
            restarting: true,
            generation: 7,
        }
        .render()
        .unwrap();
        assert_csp_clean("message", &html);
        assert!(html.contains("data-generation=\"7\""), "{html}");
        assert!(
            html.contains("data-target=\"/config&#34;&#62;&#60;/p&#62;&#60;script&#62;alert(1)&#60;/script&#62;\""),
            "{html}"
        );
    }

    #[tokio::test]
    async fn test_page_uses_dispatch_path() {
        let state = test_state();
        let img = crate::image::RgbImage {
            width: 8,
            height: 8,
            rgb: vec![100; 8 * 8 * 3],
        };
        let jpeg = crate::image::encode_jpeg(&img, 90).unwrap();
        let (s, _, b) = call(
            &state,
            multipart(&[
                ("image", &jpeg),
                ("model", b"IPCAM-General.onnx"),
                ("min_confidence", b"0.3"),
            ]),
        )
        .await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("Detection failed: Model &#39;ipcam-general&#39; failed to load"));
        assert!(
            page.contains("&#34;command&#34;: &#34;custom&#34;"),
            "{page}"
        );
        assert!(page.contains("value=\"0.3\""));
        // Counted like an API request.
        let w = state.registry.by_name("ipcam-general").unwrap();
        assert_eq!(w.metrics.requests.load(Ordering::Relaxed), 1);

        let (_, _, b) = call(&state, multipart(&[("model", b"ipcam-general")])).await;
        assert!(String::from_utf8(b).unwrap().contains("No image provided"));
        let (_, _, b) = call(&state, multipart(&[("image", &jpeg), ("model", b"zzz")])).await;
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("Unknown model &#39;zzz&#39;")
        );
    }

    #[test]
    fn no_objects_line_names_the_threshold_that_ran() {
        // The model's own threshold, its night one when it differs.
        assert_eq!(
            super::no_objects_line("", 0.5, None),
            ("No objects at or above 0.50.".into(), super::LOWER_HINT)
        );
        assert_eq!(
            super::no_objects_line(" ", 0.375, Some(0.3)).0,
            "No objects at or above 0.375 (0.30 on night and IR frames)."
        );
        assert_eq!(
            super::no_objects_line("", 0.4, Some(0.4)).0,
            "No objects at or above 0.40."
        );
        // The form's value wins over both, read like the worker reads it.
        assert_eq!(
            super::no_objects_line("0.3", 0.5, Some(0.2)).0,
            "No objects at or above 0.30."
        );
        assert_eq!(
            super::no_objects_line("45", 0.5, None).0,
            "No objects at or above 0.45."
        );
        assert_eq!(
            super::no_objects_line("500", 0.5, None).0,
            "No objects at or above 0.50."
        );
        assert_eq!(
            super::no_objects_line("abc", 0.6, None).0,
            "No objects at or above 0.60."
        );
        // At the floor there is nothing lower to try.
        assert_eq!(
            super::no_objects_line("0.001", 0.5, None),
            ("No objects at or above 0.01.".into(), "")
        );
    }

    #[test]
    fn test_page_says_when_nothing_was_found() {
        let state = test_state();
        let result = |none_found: &str, none_hint: &'static str| super::TestResult {
            summary: "0 object(s) from model 'ipcam-general'".into(),
            none_found: none_found.into(),
            none_hint,
            image_data_uri: String::new(),
            json: "{}".into(),
        };
        let (line, hint) = super::no_objects_line("", 0.5, None);
        let html = super::test_template(&state, "", String::new(), None, Some(result(&line, hint)))
            .render()
            .unwrap();
        assert!(
            html.contains(
                "<p class=\"zero-state\">No objects at or above 0.50. <span class=\"muted\">Lower Min confidence to see weaker detections.</span></p>"
            ),
            "{html}"
        );
        assert_csp_clean("/test (no objects)", &html);
        // test.js reads each model's threshold from the select.
        assert!(html.contains("data-thresholds=\"{"), "{html}");
        assert!(html.contains("ipcam-general"), "{html}");
        // An answer with boxes has no such line.
        let html = super::test_template(&state, "", String::new(), None, Some(result("", "")))
            .render()
            .unwrap();
        assert!(!html.contains("zero-state"), "{html}");
    }

    fn logs_state() -> Arc<AppState> {
        let base = test_state();
        let mut state = AppState::new(
            base.registry.clone(),
            base.metrics.clone(),
            base.config_read().clone(),
            base.config_path.clone(),
        );
        state.log_reload = Some(LogReloadHandle::detached(LogLevel::Info));
        Arc::new(state)
    }

    #[tokio::test]
    async fn logs_page_and_endpoint() {
        let state = logs_state();
        let buf = state.log_buffer().unwrap().clone();
        buf.push(
            tracing::Level::INFO,
            "blue_onyx_prism::registry",
            "model ready".into(),
        );
        buf.push(
            tracing::Level::WARN,
            "blue_onyx_prism::server",
            "queue <full>".into(),
        );
        buf.push(tracing::Level::DEBUG, "x", "detail".into());

        let (s, page) = get_text(&state, "/logs").await;
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("id=\"logview\""));
        assert!(page.contains("<option value=\"info\" selected>info</option>"));
        assert!(page.contains("class=\"active\" aria-current=\"page\">Logs</a>"));

        let (s, body) = get_text(&state, "/logs.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["last"], 3);
        assert_eq!(v["entries"].as_array().unwrap().len(), 3);
        assert_eq!(v["entries"][1]["message"], "queue <full>");
        assert_eq!(v["entries"][1]["level"], "WARN");
        assert_eq!(v["level"], "info");

        let (_, body) = get_text(&state, "/logs.json?after=1&level=warn").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let e = v["entries"].as_array().unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0]["seq"], 2);

        let (_, body) = get_text(&state, "/logs.json?after=3").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["entries"].as_array().unwrap().is_empty());
        assert_eq!(v["last"], 3);

        let (s, _) = get_text(&state, "/logs.json?level=loud").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = get_text(&state, "/logs.json?after=x").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // The level control applies through the handle and is saved.
        let (s, _, b) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(
            Config::load(&state.config_path).unwrap().log_level,
            LogLevel::Debug
        );

        // Without a handle the endpoint says so.
        let plain = test_state();
        let (s, body) = get_text(&plain, "/logs.json").await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn train_tabs_load_the_class_picker() {
        let state = test_state();
        for (uri, own_js) in [
            ("/train/review", "/static/review.js?v="),
            ("/train/dataset", "/static/train-dataset.js?v="),
            ("/train/import", "/static/training.js?v="),
            ("/train/autolabel", "/static/training.js?v="),
            ("/train/train", "/static/training.js?v="),
            ("/config", "/static/config.js?v="),
        ] {
            let (s, html) = get_text(&state, uri).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
            if uri == "/train/autolabel" {
                let mp = html.find("/static/model-picker.js?v=");
                assert!(mp.is_some() && mp < html.find(own_js), "{uri}");
            }
            let picker = html.find("/static/class-picker.js?v=");
            let own = html.find(own_js);
            assert!(picker.is_some() && own.is_some(), "{uri}");
            // The picker defines ClassPicker before the page script uses it.
            assert!(picker < own, "{uri}");
        }
        let (_, review) = get_text(&state, "/train/review").await;
        // The finger rules (pinch, two-tap box, one finger's drag) before the page script that
        // reads them.
        let touch = review.find("/static/review-touch.js?v=");
        assert!(
            touch.is_some() && touch < review.find("/static/review.js?v="),
            "review-touch.js before review.js"
        );
        assert!(static_file("review-touch.js").is_some_and(|js| js.contains("ReviewTouch")));
        for id in ["rv-draw", "rv-redraw", "rv-view-menu"] {
            assert!(review.contains(&format!("id=\"{id}\"")), "missing #{id}");
        }
        assert!(review.contains("id=\"rv-label-picker\""));
        assert!(review.contains("id=\"rv-show-picker\""));
        assert!(review.contains("<option value=\"both\">found by 2+ labelers</option>"));
        let (_, dataset) = get_text(&state, "/train/dataset").await;
        assert!(dataset.contains("id=\"rv-classes-picker\""));
        let (_, import) = get_text(&state, "/train/import").await;
        assert!(import.contains("id=\"import-al-picker\""));
        let (_, training) = get_text(&state, "/train/autolabel").await;
        assert!(training.contains("id=\"al-picker\""));
        assert!(training.contains("name=\"mode\" value=\"labelers\""));
        assert!(training.contains("name=\"mode\" value=\"model\""));
        assert!(training.contains("id=\"al-hide-unticked\""));
        assert!(training.contains("id=\"al-preset\""));
        assert!(training.contains("id=\"al-model-picker\""));
        assert!(training.contains("name=\"sam3_threshold\""));
        assert!(training.contains("<option value=\"both\">found by 2+ labelers</option>"));
    }

    /// Phones: the page may use the whole screen (notch insets reach the CSS only with
    /// `viewport-fit=cover`), follows the system's light / dark choice from the first paint, and
    /// can always be zoomed (WCAG 1.4.4: never `maximum-scale` or `user-scalable`).
    #[tokio::test]
    async fn the_base_page_covers_the_screen_and_stays_zoomable() {
        let state = test_state();
        for uri in ["/", "/config", "/train/review", "/benchmark", "/logs"] {
            let (s, html) = get_text(&state, uri).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
            assert!(
                html.contains("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"),
                "{uri}"
            );
            assert!(
                html.contains("<meta name=\"color-scheme\" content=\"light dark\">"),
                "{uri}"
            );
            assert!(
                !html.contains("user-scalable") && !html.contains("maximum-scale"),
                "{uri}: the page must stay zoomable"
            );
        }
        let css = static_file("style.css").unwrap();
        // The insets are 0 without `viewport-fit=cover`: every use of them depends on it.
        assert!(css.contains("env(safe-area-inset-bottom)"));
        assert!(css.contains("env(safe-area-inset-left)"));
    }

    /// The declarations of each rule of `css` as (property, value), comments left out. Nested
    /// blocks (`@media`) give their inner rules; a rule is what follows the last `{` before a `}`.
    fn css_rules(css: &str) -> Vec<Vec<(String, String)>> {
        let mut text = String::with_capacity(css.len());
        let mut rest = css;
        while let Some(start) = rest.find("/*") {
            text.push_str(&rest[..start]);
            rest = rest[start..]
                .find("*/")
                .map_or("", |end| &rest[start + end + 2..]);
        }
        text.push_str(rest);
        text.split('}')
            .filter_map(|chunk| chunk.rsplit_once('{').map(|(_, body)| body))
            .map(|body| {
                body.split(';')
                    .filter_map(|d| d.split_once(':'))
                    .map(|(p, v)| (p.trim().to_ascii_lowercase(), v.trim().to_string()))
                    .collect()
            })
            .collect()
    }

    /// Each style rule's selector in `css` with the at-rule preludes around it (outermost
    /// first), comments and quoted strings skipped.
    fn css_selectors(css: &str) -> Vec<(Vec<String>, String)> {
        let (mut out, mut stack, mut prelude) = (Vec::new(), Vec::<String>::new(), String::new());
        let mut chars = css.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let mut last = ' ';
                    for d in chars.by_ref() {
                        if last == '*' && d == '/' {
                            break;
                        }
                        last = d;
                    }
                }
                '"' | '\'' => {
                    prelude.push(c);
                    for d in chars.by_ref() {
                        prelude.push(d);
                        if d == c {
                            break;
                        }
                    }
                }
                '{' => {
                    let p = prelude.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !p.starts_with('@') {
                        out.push((stack.clone(), p.clone()));
                    }
                    stack.push(p);
                    prelude.clear();
                }
                '}' => {
                    stack.pop();
                    prelude.clear();
                }
                ';' => prelude.clear(),
                _ => prelude.push(c),
            }
        }
        out
    }

    #[test]
    fn css_selectors_know_their_media_blocks() {
        let got = css_selectors(
            "a:hover { x: \"}\" } /* b:hover { } */\n@media (hover: hover) { b:hover { y: 1; } }\nc { }",
        );
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[0], (vec![], "a:hover".to_string()));
        assert_eq!(got[1].0, vec!["@media (hover: hover)".to_string()]);
        assert_eq!(got[2], (vec![], "c".to_string()));
    }

    /// Phones and tablets: a tap leaves `:hover` set (iOS until the next tap elsewhere), so a
    /// hover fill such as `button.danger:hover` stayed red after a tap and a cancelled confirm.
    /// Every `:hover` rule is inside an `@media` whose condition has `hover: hover`; the
    /// `:focus-visible` and `[aria-expanded]` twins that shared a rule with one stay outside.
    #[test]
    fn hover_styles_apply_only_where_a_pointer_hovers() {
        let css = static_file("style.css").unwrap();
        let rules = css_selectors(css);
        let bare: Vec<&String> = rules
            .iter()
            .filter(|(within, sel)| {
                sel.contains(":hover")
                    && !within
                        .iter()
                        .any(|w| w.starts_with("@media") && w.contains("hover: hover"))
            })
            .map(|(_, sel)| sel)
            .collect();
        assert!(
            bare.is_empty(),
            ":hover outside @media (hover: hover): {bare:?}"
        );
        assert!(
            rules.iter().filter(|(_, s)| s.contains(":hover")).count() >= 20,
            "the parser finds the hover rules"
        );
        for twin in [
            "header nav a.active",
            "button.browse-item:focus-visible",
            "button.term[aria-expanded=\"true\"]",
            "button.term.chip[aria-expanded=\"true\"]",
        ] {
            assert!(
                rules
                    .iter()
                    .any(|(within, s)| s == twin && within.is_empty()),
                "{twin} applies without hover"
            );
        }
        // A line chart scrolls up and down and pinch-zooms under a finger and scrubs sideways
        // (on the svg: Chrome ignores touch-action on its shapes).
        assert!(css.contains(
            "svg.thr-chart:not(.pb-scatter), .thr-chart .thr-hit { touch-action: pan-y pinch-zoom; }"
        ));
    }

    /// Whether `value` holds a plain `vh` length (`65vh`, `100vh`; not `dvh` / `svh` / `lvh`).
    fn plain_vh(value: &str) -> bool {
        let b = value.as_bytes();
        value
            .match_indices("vh")
            .any(|(i, _)| i > 0 && b.get(i - 1).is_some_and(u8::is_ascii_digit))
    }

    #[test]
    fn css_rules_reads_declarations_inside_media_blocks() {
        let rules = css_rules(
            "a { height: 1vh; /* x: 2vh; */ }\n@media (max-width: 1px) { b { max-height: 3vh; max-height: 3dvh } }",
        );
        assert_eq!(
            rules[0],
            vec![("height".to_string(), "1vh".to_string())],
            "comments are no declarations"
        );
        assert_eq!(rules[1].len(), 2);
        assert!(plain_vh("calc(100vh - 1rem)") && !plain_vh("90dvh") && !plain_vh("80svh"));
    }

    /// Phones: `vh` is the tallest viewport (toolbars hidden), so a panel or dialog sized by it
    /// runs under the browser's toolbars. Each such height is followed by a `dvh` / `svh` one in
    /// the same rule (the `vh` stays first as the fallback). iOS inflates text in landscape
    /// unless told not to.
    #[test]
    fn heights_follow_the_visible_screen_and_text_keeps_its_size() {
        let css = static_file("style.css").unwrap();
        let mut missing = Vec::new();
        for rule in css_rules(css) {
            for (i, (prop, value)) in rule.iter().enumerate() {
                if !matches!(prop.as_str(), "height" | "max-height" | "min-height")
                    || !plain_vh(value)
                {
                    continue;
                }
                let followed = rule
                    .iter()
                    .skip(i + 1)
                    .any(|(p, v)| p == prop && (v.contains("dvh") || v.contains("svh")));
                if !followed {
                    missing.push(format!("{prop}: {value}"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "vh without a dvh/svh after it: {missing:?}"
        );
        // The ones the phone pass is about, by name (the parser above could miss a rule).
        for needle in [
            "height: calc(100vh - 1rem); height: calc(100dvh - 1rem);",
            "height: 65vh;\n  height: 65dvh;",
            "max-height: 92vh; max-height: 92dvh;",
            ".browse-list { max-height: 60vh; max-height: 60dvh;",
        ] {
            assert!(css.contains(needle), "{needle}");
        }
        let html_rule = css_rules(css)
            .into_iter()
            .find(|r| r.iter().any(|(p, _)| p == "-webkit-text-size-adjust"))
            .expect("-webkit-text-size-adjust is set");
        assert!(
            html_rule
                .iter()
                .any(|(p, v)| p == "text-size-adjust" && v == "100%")
        );
        // The on-screen keyboard lets the sticky bars scroll away (live.js sets the class).
        assert!(css.contains("html.kb-open .rv-actions-main, html.kb-open form#models > .actions, html.kb-open form#server > .actions { position: static; }"));
        assert!(
            static_file("live.js").is_some_and(|js| js.contains("PrismViewport")
                && js.contains("\"kb-open\"")
                && js.contains("\"--toc-h\""))
        );
    }

    /// The server-side file browser is one include on every page that has it: Close at the top,
    /// Use / Cancel at the bottom, and browse.js loaded once.
    #[tokio::test]
    async fn every_browse_dialog_has_a_close_button_and_one_script() {
        let state = test_state();
        for uri in [
            "/config",
            "/benchmark",
            "/train/review",
            "/train/dataset",
            "/train/collect",
        ] {
            let (s, html) = get_text(&state, uri).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
            assert_eq!(
                html.matches("<dialog id=\"browse-dialog\">").count(),
                1,
                "{uri}"
            );
            assert!(
                html.contains("id=\"browse-close\" class=\"small secondary\" aria-label=\"Close\""),
                "{uri}"
            );
            assert!(html.contains("id=\"browse-cancel\""), "{uri}");
            assert_eq!(html.matches("/static/browse.js?v=").count(), 1, "{uri}");
        }
        let js = static_file("browse.js").unwrap();
        assert!(js.contains("(pointer: fine)") && js.contains("browse-close"));
    }
}
