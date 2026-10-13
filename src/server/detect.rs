//! Detection endpoints: multipart parsing, dispatch to the model workers, `custom/list`.

use super::AppState;
use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::metrics::DropReason;
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use axum::Json;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, Path, State};
use crossbeam_channel::TrySendError;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Maximum accepted request body (multipart image upload).
pub const BODY_LIMIT: usize = 32 * 1024 * 1024;

/// Added to the worker's request timeout while a handler waits for its reply, so the worker's
/// own "waited longer than the timeout" answer wins the race and reaches the client.
pub const RESPONSE_GRACE: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------------------------
// Detection endpoints

/// Parse the multipart body: `image` (required), `min_confidence` (parse errors become 0.0)
/// and any other text fields, returned in the map (e.g. `model` on the test page).
async fn read_multipart(
    mut multipart: Multipart,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    let mut req = VisionDetectionRequest::default();
    let mut fields = HashMap::new();
    let mut got_image = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return Err(format!("Invalid multipart body: {e}")),
        };
        match field.name().map(str::to_string) {
            Some(name) if name == "min_confidence" => {
                let text = field.text().await.unwrap_or_default();
                req.min_confidence = text.trim().parse::<f32>().unwrap_or(0.0);
                fields.insert(name, text.trim().to_string());
            }
            Some(name) if name == "image" => {
                if let Some(n) = field.file_name() {
                    req.image_name = n.to_string();
                }
                req.image_data = field
                    .bytes()
                    .await
                    .map_err(|e| format!("Reading image field: {e}"))?;
                got_image = true;
            }
            Some(name) => {
                if let Ok(text) = field.text().await {
                    fields.insert(name, text);
                }
            }
            None => {}
        }
    }
    if !got_image || req.image_data.is_empty() {
        return Err("No image provided (expected multipart field 'image')".into());
    }
    Ok((req, fields))
}

pub(super) async fn parse_detection_body(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    match multipart {
        Ok(mp) => read_multipart(mp).await,
        Err(e) => Err(format!("Expected a multipart/form-data body: {e}")),
    }
}

pub(super) async fn detection_default(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Json<VisionDetectionResponse> {
    let start = Instant::now();
    // Read the upload before any early answer: an unread body makes hyper close the
    // connection, which breaks the client's next request on it.
    let req = parse_detection_body(multipart).await.map(|(r, _)| r);
    let Some(handle) = state.registry.default_model() else {
        return Json(VisionDetectionResponse::error(
            "No default model: enable a model on the Config page",
        ));
    };
    Json(run_detection(&state, &handle, req, start, "detect").await)
}

pub(super) async fn detection_custom(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Json<VisionDetectionResponse> {
    let start = Instant::now();
    // Read the upload first, as in [`detection_default`].
    let req = parse_detection_body(multipart).await.map(|(r, _)| r);
    let Some(handle) = state.registry.by_name(&model) else {
        // Client-chosen text: shortened and debug-escaped before it reaches the log.
        warn!(model = ?short(&model), "request for unknown model");
        let mut r = VisionDetectionResponse::error(format!("Unknown model '{model}'"));
        r.command = "custom".into();
        return Json(r);
    };
    Json(run_detection(&state, &handle, req, start, "custom").await)
}

/// Longest client-chosen name written to the log.
const LOG_NAME_CHARS: usize = 64;

/// `s` cut to [`LOG_NAME_CHARS`] characters (with an ellipsis) for the log.
fn short(s: &str) -> String {
    let mut it = s.chars();
    let head: String = it.by_ref().take(LOG_NAME_CHARS).collect();
    if it.next().is_some() {
        format!("{head}\u{2026}")
    } else {
        head
    }
}

pub(super) async fn run_detection(
    state: &AppState,
    handle: &WorkerHandle,
    req: Result<VisionDetectionRequest, String>,
    start: Instant,
    command: &str,
) -> VisionDetectionResponse {
    let mut resp = dispatch(handle, req, start).await;
    resp.command = command.to_string();
    resp.analysisRoundTripMs = i32::try_from(start.elapsed().as_millis()).unwrap_or(i32::MAX);
    if resp.success {
        handle
            .metrics
            .round_trip_ms
            .record(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
    }
    resp.canUseGPU = state.registry.runtime_info.has_gpu;
    resp
}

pub(super) async fn dispatch(
    handle: &WorkerHandle,
    req: Result<VisionDetectionRequest, String>,
    start: Instant,
) -> VisionDetectionResponse {
    let name = &handle.name;
    let req = match req {
        Ok(r) => r,
        Err(e) => return VisionDetectionResponse::error(e),
    };
    handle.metrics.requests.fetch_add(1, Ordering::Relaxed);

    match handle.state.get() {
        ModelState::Ready => {}
        ModelState::Initializing => {
            // Never queued: a lazy model's first request starts its compile and every request
            // until it is ready is answered at once.
            if handle.trigger_load() {
                info!(model = %name, "first request: compiling lazy model");
            }
            // The worker names the candidate it compiles on; until it got that far, a lazy
            // model is headed for its planned device.
            let detail = handle.state.detail().or_else(|| {
                handle.lazy.then(|| {
                    format!(
                        "{} {}",
                        crate::worker::COMPILING_ON,
                        handle.planned_device_text()
                    )
                })
            });
            handle.metrics.record_drop(DropReason::Initializing);
            return VisionDetectionResponse::error(crate::worker::initializing_message(
                name,
                detail.as_deref(),
            ));
        }
        ModelState::Failed(m) => {
            handle.metrics.record_drop(DropReason::Failed);
            return VisionDetectionResponse::error(format!("Model '{name}' failed to load: {m}"));
        }
    }

    if handle.is_full() {
        warn!(model = %name, "worker queue is full, server is overloaded; rejecting request");
        handle.metrics.record_drop(DropReason::QueueFull);
        return VisionDetectionResponse::error("Worker queue is full");
    }

    let (tx, rx) = tokio::sync::oneshot::channel();
    match handle
        .sender
        .try_send(crate::worker::WorkItem::Detect(req, tx, start))
    {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            handle.metrics.record_drop(DropReason::QueueFull);
            return VisionDetectionResponse::error("Worker queue is full");
        }
        Err(TrySendError::Disconnected(_)) => {
            handle.metrics.record_drop(DropReason::WorkerGone);
            return VisionDetectionResponse::error(format!("Model '{name}' worker is not running"));
        }
    }

    match tokio::time::timeout(handle.request_timeout() + RESPONSE_GRACE, rx).await {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => {
            warn!(model = %name, "worker dropped the request without replying");
            handle.metrics.record_drop(DropReason::WorkerGone);
            VisionDetectionResponse::error("Worker dropped the request")
        }
        Err(_) => {
            warn!(model = %name, "timed out waiting for the worker");
            handle.metrics.record_drop(DropReason::Timeout);
            VisionDetectionResponse::error("Operation timed out")
        }
    }
}

pub(super) async fn custom_list(
    State(state): State<Arc<AppState>>,
) -> Json<VisionCustomListResponse> {
    let reg = &state.registry;
    Json(VisionCustomListResponse {
        success: true,
        models: reg.names(),
        moduleId: crate::MODULE_ID.to_string(),
        moduleName: crate::MODULE_NAME.to_string(),
        command: "list".to_string(),
        statusData: None,
        inferenceDevice: reg
            .default_model()
            .map(|w| w.execution_provider())
            .unwrap_or_default(),
        analysisRoundTripMs: 0,
        processedBy: crate::PROCESSED_BY.to_string(),
        timestampUTC: chrono::Utc::now().to_rfc3339(),
    })
}
