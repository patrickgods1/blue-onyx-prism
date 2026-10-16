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
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const CSS: &str = "text/css; charset=utf-8";
const JS: &str = "text/javascript; charset=utf-8";

/// Files served under `/static/<name>`: name, Content-Type, body. Every page script lives here
/// (the CSP allows `script-src 'self'` only: no inline scripts, handlers or style attributes);
/// pages pass server values through `data-*` attributes.
pub(super) static STATIC_FILES: &[(&str, &str, &str)] = &[
    ("style.css", CSS, include_str!("../../assets/style.css")),
    ("live.js", JS, include_str!("../../assets/live.js")),
    (
        "bench-glossary.js",
        JS,
        include_str!("../../assets/bench-glossary.js"),
    ),
    (
        "bench-charts.js",
        JS,
        include_str!("../../assets/bench-charts.js"),
    ),
    (
        "bench-compare.js",
        JS,
        include_str!("../../assets/bench-compare.js"),
    ),
    (
        "benchmark.js",
        JS,
        include_str!("../../assets/benchmark.js"),
    ),
    ("config.js", JS, include_str!("../../assets/config.js")),
    ("logs.js", JS, include_str!("../../assets/logs.js")),
    ("stats.js", JS, include_str!("../../assets/stats.js")),
    ("test.js", JS, include_str!("../../assets/test.js")),
    ("welcome.js", JS, include_str!("../../assets/welcome.js")),
];

/// Body of the static file `name` (`"benchmark.js"`), for tests that check page code.
#[cfg(test)]
pub(super) fn static_file(name: &str) -> Option<&'static str> {
    STATIC_FILES
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, _, body)| *body)
}

static FAVICON_ICO: &[u8] = include_bytes!("../../assets/favicon.ico");

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
        })
    })
    .await
}

pub(super) async fn update_available() -> impl IntoResponse {
    Json(crate::update::check_update(crate::VERSION).await)
}

// ---------------------------------------------------------------------------------------------
// Stats

struct TimingCells {
    avg: String,
    min: String,
    max: String,
}

impl TimingCells {
    fn from_stat(s: &Histogram) -> Self {
        if s.count() == 0 {
            let dash = || "-".to_string();
            return Self {
                avg: dash(),
                min: dash(),
                max: dash(),
            };
        }
        Self {
            avg: format!("{:.1}", s.avg()),
            min: s.min().to_string(),
            max: s.max().to_string(),
        }
    }
}

struct StatsRow {
    name: String,
    is_default: bool,
    state: String,
    state_class: &'static str,
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

/// Version of the loaded ONNX Runtime, when the shared runtimes are free to ask.
fn ort_version(reg: &ModelRegistry) -> Option<String> {
    let rt = reg.runtimes()?.try_lock().ok()?;
    rt.onnxruntime_version()
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
            StatsRow {
                name: w.name.clone(),
                is_default: w.name == default,
                state: state_label(w),
                state_class: state_view(w).1,
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
            serde_json::json!({
                "name": w.name,
                "default": default.as_deref() == Some(w.name.as_str()),
                "state": w.state.describe(),
                "stateDetail": w.state.detail(),
                "stateText": state_label(w),
                "stateClass": state_view(w).1,
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
    Json(serde_json::json!({
        "version": crate::VERSION,
        "uptimeSecs": state.metrics.uptime().as_secs(),
        "uptime": format_uptime(state.started.elapsed()),
        "openvino": reg.runtime_info,
        "models": models,
        "rejectedOverloaded": state.metrics.rejected_overloaded.load(Ordering::Relaxed),
        "inflightRequests": state.metrics.inflight(),
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
    /// `data:image/jpeg;base64,...` of the annotated image; empty when it could not be drawn.
    image_data_uri: String,
    json: String,
}

#[derive(Template)]
#[template(path = "test.html")]
struct TestTemplate {
    nav: &'static str,
    version: &'static str,
    models: Vec<ModelChoice>,
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
        .collect();
    TestTemplate {
        nav: "test",
        version: crate::VERSION,
        models,
        min_confidence,
        error,
        result,
    }
}

pub(super) async fn test_page(State(state): State<Arc<AppState>>) -> Response {
    render(&test_template(&state, "", String::new(), None, None))
}

/// Same dispatch path as the API (metrics and queues behave identically), then draws the
/// predictions on the uploaded image off the async thread.
pub(super) async fn test_submit(State(state): State<Arc<AppState>>, upload: Upload) -> Response {
    let start = Instant::now();
    let id = new_request_id();
    let (req, fields) = match parse_detection_body(upload).await {
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

/// One entry of [`STATIC_FILES`]; 404 (JSON, like the router's fallback) for any other name.
fn serve_static(name: &str) -> Response {
    match STATIC_FILES.iter().find(|(n, _, _)| *n == name) {
        Some((_, content_type, body)) => (
            [
                (header::CONTENT_TYPE, *content_type),
                (header::CACHE_CONTROL, "public, max-age=3600"),
            ],
            *body,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": format!("Not found: /static/{name}"),
            })),
        )
            .into_response(),
    }
}

/// `GET /static/{file}`: the page scripts and the stylesheet.
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
        assert!(stats.contains("class=\"stats sticky-first\""));
        assert!(stats.contains("ipcam-general") && stats.contains("title=\"CPU\""));
        assert!(!stats.contains("<b>evil</b>"));

        let (s, json) = get_text(&state, "/stats.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["models"][0]["name"], "ipcam-general");
        assert_eq!(v["models"][0]["default"], true);
        assert_eq!(v["models"][0]["stateClass"], "fail");
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
        for (name, content_type, body) in STATIC_FILES {
            let (s, h, b) = call(
                &state,
                Request::get(format!("/static/{name}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{name}");
            assert_eq!(h[header::CONTENT_TYPE], *content_type, "{name}");
            assert_eq!(h[header::CACHE_CONTROL], "public, max-age=3600", "{name}");
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
        for page in ["/", "/stats", "/test", "/config", "/logs", "/benchmark"] {
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
}
