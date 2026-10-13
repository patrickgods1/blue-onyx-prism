//! Detection endpoints: multipart parsing, dispatch to the model workers, `custom/list`.

use super::AppState;
use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::metrics::{DropReason, InflightPermit};
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use axum::Json;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use crossbeam_channel::TrySendError;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Maximum accepted request body (multipart image upload).
///
/// Worst-case memory of the detection routes: every admitted request may hold one full body,
/// so uploads take at most `max_inflight × BODY_LIMIT` (see [`max_inflight`]; e.g. 64 × 32 MiB
/// = 2 GiB with the default cap, the bound the in-flight cap exists for), plus, per
/// preprocessing thread, one decoded frame of up to `max_image_pixels × 3` bytes (RGB8) and its
/// resized input tensor. Requests beyond the cap are answered 503 before their body is read.
pub const BODY_LIMIT: usize = 32 * 1024 * 1024;

/// Added to the worker's request timeout while a handler waits for its reply, so the worker's
/// own "waited longer than the timeout" answer wins the race and reaches the client.
pub const RESPONSE_GRACE: Duration = Duration::from_millis(500);

/// Longest a detection request may take to deliver its multipart body; a slower (or stalled)
/// upload is answered with an error and its connection closed, releasing its in-flight slot.
pub const MULTIPART_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Server-wide in-flight cap used when the models report no queue capacity (none loaded yet,
/// all failed): requests answered without inference still hold their upload while it is read.
pub const DEFAULT_MAX_INFLIGHT: usize = 64;

/// Added to the sum of the models' queue capacities to form the in-flight cap: requests being
/// inferred (out of the queue), waiting for an early answer, or reading their body.
pub const INFLIGHT_MARGIN: usize = 16;

/// Most of an unwanted body (a request answered before its upload is needed: unknown model,
/// failed or compiling model, full queue) read and thrown away, chunk by chunk (nothing is
/// buffered, so this costs bandwidth, not memory). It equals [`BODY_LIMIT`] on purpose: a body
/// left unread makes hyper close the connection after the answer, and closing a socket with
/// unread upload data can reset it before the client reads the answer; a pooling client (Blue
/// Iris, reqwest) then also fails its next request on that connection.
pub const DISCARD_LIMIT: usize = BODY_LIMIT;

/// Longest spent discarding an unwanted body (see [`DISCARD_LIMIT`]); after it the body is left
/// unread and the connection closes.
pub const DISCARD_TIMEOUT: Duration = MULTIPART_READ_TIMEOUT;

/// The server-wide cap on detection requests in flight: a fixed `Metrics::inflight_limit` when
/// set (tests), else the sum of the models' queue capacities plus [`INFLIGHT_MARGIN`], or
/// [`DEFAULT_MAX_INFLIGHT`] when that sum is 0. Read per request, so it follows reloads and the
/// queue sizing done after warm-up.
pub(super) fn max_inflight(state: &AppState) -> usize {
    let fixed = state.metrics.inflight_limit.load(Ordering::Relaxed);
    if fixed > 0 {
        return fixed;
    }
    let queued: usize = state
        .registry
        .workers()
        .iter()
        .map(|w| w.queue_capacity())
        .fold(0, usize::saturating_add);
    if queued == 0 {
        DEFAULT_MAX_INFLIGHT
    } else {
        queued.saturating_add(INFLIGHT_MARGIN)
    }
}

/// Admit a detection request under [`max_inflight`], or the 503 answer for an overloaded
/// server (counted in `rejected_overloaded`, warned about at most once per 10 s).
fn admit(state: &AppState, command: &str) -> Result<InflightPermit, Box<Response>> {
    let limit = max_inflight(state);
    if let Some(p) = state.metrics.try_begin_detection(limit) {
        return Ok(p);
    }
    state
        .metrics
        .rejected_overloaded
        .fetch_add(1, Ordering::Relaxed);
    if let Some(held_back) = state.metrics.overload_log.allow(state.metrics.now_ms()) {
        warn!(
            limit,
            similar_since_last_warning = held_back,
            "server is overloaded: too many detection requests in flight; answering 503"
        );
    }
    let mut r = VisionDetectionResponse::error(format!(
        "Server is overloaded ({limit} detection requests in flight); retry later"
    ));
    r.command = command.to_string();
    Err(Box::new(
        (StatusCode::SERVICE_UNAVAILABLE, Json(r)).into_response(),
    ))
}

/// Read and throw away the start of a body that will not be used: at most [`DISCARD_LIMIT`]
/// bytes within [`DISCARD_TIMEOUT`]. A body that ends within both leaves the connection
/// reusable; otherwise hyper closes it after the answer (it drains only what is buffered).
async fn discard_body(multipart: Result<Multipart, MultipartRejection>) {
    discard_body_within(multipart, DISCARD_TIMEOUT).await;
}

/// [`discard_body`] with an explicit time limit.
async fn discard_body_within(multipart: Result<Multipart, MultipartRejection>, limit: Duration) {
    let Ok(mut mp) = multipart else { return };
    let drain = async move {
        let mut seen = 0usize;
        while let Ok(Some(mut field)) = mp.next_field().await {
            while let Ok(Some(chunk)) = field.chunk().await {
                seen = seen.saturating_add(chunk.len());
                if seen > DISCARD_LIMIT {
                    return;
                }
            }
        }
    };
    let _ = tokio::time::timeout(limit, drain).await;
}

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

/// Parse a detection upload within [`MULTIPART_READ_TIMEOUT`]. The image field is read into
/// one `Bytes` (`Field::bytes`) that moves on to the worker without another copy.
pub(super) async fn parse_detection_body(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    parse_body_within(multipart, MULTIPART_READ_TIMEOUT).await
}

/// [`parse_detection_body`] with an explicit time limit.
async fn parse_body_within(
    multipart: Result<Multipart, MultipartRejection>,
    limit: Duration,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    match multipart {
        Ok(mp) => tokio::time::timeout(limit, read_multipart(mp))
            .await
            .unwrap_or_else(|_| {
                Err(format!(
                    "Request body not received within {} s",
                    limit.as_secs()
                ))
            }),
        Err(e) => Err(format!("Expected a multipart/form-data body: {e}")),
    }
}

pub(super) async fn detection_default(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let start = Instant::now();
    // The permit covers the whole request: body read, queue wait and the worker's reply.
    let _permit = match admit(&state, "detect") {
        Ok(p) => p,
        Err(resp) => return *resp,
    };
    let Some(handle) = state.registry.default_model() else {
        discard_body(multipart).await;
        return Json(VisionDetectionResponse::error(
            "No default model: enable a model on the Config page",
        ))
        .into_response();
    };
    Json(detect_on(&state, &handle, multipart, start, "detect").await).into_response()
}

pub(super) async fn detection_custom(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let start = Instant::now();
    let _permit = match admit(&state, "custom") {
        Ok(p) => p,
        Err(resp) => return *resp,
    };
    let Some(handle) = state.registry.by_name(&model) else {
        discard_body(multipart).await;
        // Client-chosen text: shortened and debug-escaped before it reaches the log.
        warn!(model = ?short(&model), "request for unknown model");
        let mut r = VisionDetectionResponse::error(format!("Unknown model '{model}'"));
        r.command = "custom".into();
        return Json(r).into_response();
    };
    Json(detect_on(&state, &handle, multipart, start, "custom").await).into_response()
}

/// A detection on `handle`. When the model cannot take it now (compiling, failed, queue full)
/// the answer is known before the upload is needed: the body is discarded (bounded, see
/// [`discard_body`]) instead of buffered, and the request answered at once.
async fn detect_on(
    state: &AppState,
    handle: &WorkerHandle,
    multipart: Result<Multipart, MultipartRejection>,
    start: Instant,
    command: &str,
) -> VisionDetectionResponse {
    if !matches!(handle.state.get(), ModelState::Ready) || handle.is_full() {
        handle.metrics.requests.fetch_add(1, Ordering::Relaxed);
        if let Some(early) = answer_without_inference(handle) {
            discard_body(multipart).await;
            return finish(state, handle, early, start, command);
        }
        // It became ready (or got room) meanwhile: serve it normally (the request is
        // already counted).
        let req = parse_detection_body(multipart).await.map(|(r, _)| r);
        let resp = match req {
            Ok(r) => send_to_worker(handle, r, start).await,
            Err(e) => VisionDetectionResponse::error(e),
        };
        return finish(state, handle, resp, start, command);
    }
    let req = parse_detection_body(multipart).await.map(|(r, _)| r);
    run_detection(state, handle, req, start, command).await
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
    let resp = dispatch(handle, req, start).await;
    finish(state, handle, resp, start, command)
}

/// Stamp the command, round trip and GPU flag on `resp` (and record the round trip of a
/// successful detection).
fn finish(
    state: &AppState,
    handle: &WorkerHandle,
    mut resp: VisionDetectionResponse,
    start: Instant,
    command: &str,
) -> VisionDetectionResponse {
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
    let req = match req {
        Ok(r) => r,
        Err(e) => return VisionDetectionResponse::error(e),
    };
    handle.metrics.requests.fetch_add(1, Ordering::Relaxed);
    if let Some(r) = answer_without_inference(handle) {
        return r;
    }
    send_to_worker(handle, req, start).await
}

/// The immediate answer (and its drop metric) when `handle` cannot take a request now: a
/// compiling model (a lazy one starts compiling), a failed one, or a full queue. `None` when
/// the request can be queued.
fn answer_without_inference(handle: &WorkerHandle) -> Option<VisionDetectionResponse> {
    let name = &handle.name;
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
            return Some(VisionDetectionResponse::error(
                crate::worker::initializing_message(name, detail.as_deref()),
            ));
        }
        ModelState::Failed(m) => {
            handle.metrics.record_drop(DropReason::Failed);
            return Some(VisionDetectionResponse::error(format!(
                "Model '{name}' failed to load: {m}"
            )));
        }
    }
    if handle.is_full() {
        warn!(model = %name, "worker queue is full, server is overloaded; rejecting request");
        handle.metrics.record_drop(DropReason::QueueFull);
        return Some(VisionDetectionResponse::error("Worker queue is full"));
    }
    None
}

/// Queue `req` on the worker and wait for its reply (request timeout plus [`RESPONSE_GRACE`]).
async fn send_to_worker(
    handle: &WorkerHandle,
    req: VisionDetectionRequest,
    start: Instant,
) -> VisionDetectionResponse {
    let name = &handle.name;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_util::*;
    use axum::body::{Body, Bytes};
    use axum::extract::FromRequest;
    use axum::http::{Request, header};
    use futures_util::StreamExt;

    const IMAGE_HEAD: &str = "--XBOUNDARYX\r\nContent-Disposition: form-data; name=\"image\"; \
                              filename=\"t.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nxx";

    /// A multipart upload to `uri` that sends the start of its image and then stalls forever.
    fn stalled(uri: &str) -> Request<Body> {
        let head = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(IMAGE_HEAD.as_bytes()))
        });
        let body = Body::from_stream(head.chain(futures_util::stream::pending()));
        Request::post(uri)
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=XBOUNDARYX",
            )
            .body(body)
            .unwrap()
    }

    async fn multipart_of(req: Request<Body>) -> Result<Multipart, MultipartRejection> {
        Multipart::from_request(req, &()).await
    }

    async fn wait_inflight(state: &AppState, n: usize) {
        for _ in 0..400 {
            if state.metrics.inflight() == n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!(
            "in-flight count stayed {} (want {n})",
            state.metrics.inflight()
        );
    }

    /// SV-04: with the cap at 2 and two requests holding their slots (their uploads stall),
    /// further detection requests are answered 503 at once and counted.
    #[tokio::test]
    async fn requests_beyond_the_inflight_cap_get_503() {
        let state = test_state();
        state.metrics.inflight_limit.store(2, Ordering::Relaxed);
        assert_eq!(max_inflight(&state), 2);
        let held: Vec<_> = ["/v1/vision/detection", "/v1/vision/custom/ipcam-general"]
            .into_iter()
            .map(|uri| {
                let st = state.clone();
                tokio::spawn(async move { call(&st, stalled(uri)).await })
            })
            .collect();
        wait_inflight(&state, 2).await;

        let started = Instant::now();
        let (a, b) = tokio::join!(
            call(&state, stalled("/v1/vision/detection")),
            call(&state, stalled("/v1/vision/custom/ipcam-general")),
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "answered at once"
        );
        for ((status, _, body), command) in [(a, "detect"), (b, "custom")] {
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            let r: VisionDetectionResponse = serde_json::from_slice(&body).unwrap();
            assert!(!r.success);
            assert!(r.error.as_deref().unwrap().contains("overloaded"), "{r:?}");
            assert_eq!(r.command, command);
        }
        assert_eq!(state.metrics.rejected_overloaded.load(Ordering::Relaxed), 2);
        assert!(
            held.iter().all(|h| !h.is_finished()),
            "the first two still wait"
        );
        let text = crate::metrics::render_prometheus(&state.metrics, &[]);
        assert!(text.contains("blue_onyx_prism_requests_rejected_total{reason=\"overloaded\"} 2"));

        // Dropping a request (client gone) releases its slot.
        for h in held {
            h.abort();
        }
        wait_inflight(&state, 0).await;
    }

    #[test]
    fn inflight_cap_defaults() {
        // Both test models failed: no queue capacity, so the fallback applies.
        let state = test_state();
        assert_eq!(max_inflight(&state), DEFAULT_MAX_INFLIGHT);
        state.metrics.inflight_limit.store(7, Ordering::Relaxed);
        assert_eq!(max_inflight(&state), 7);
        const { assert!(DISCARD_LIMIT >= BODY_LIMIT) };
        assert_eq!(MULTIPART_READ_TIMEOUT, Duration::from_secs(30));
    }

    /// A stalled upload is answered after [`MULTIPART_READ_TIMEOUT`] instead of holding its
    /// slot forever.
    #[tokio::test]
    async fn stalled_upload_times_out() {
        let mp = multipart_of(stalled("/v1/vision/detection")).await;
        let limit = Duration::from_millis(1500);
        let started = Instant::now();
        let err = parse_body_within(mp, limit).await.unwrap_err();
        assert_eq!(err, "Request body not received within 1 s");
        assert!(started.elapsed() >= limit);
    }

    /// Discarding an unwanted body is bounded in time too.
    #[tokio::test]
    async fn discarding_a_stalled_body_is_bounded() {
        let mp = multipart_of(stalled("/v1/vision/detection")).await;
        let limit = Duration::from_millis(200);
        let started = Instant::now();
        discard_body_within(mp, limit).await;
        let waited = started.elapsed();
        assert!(
            waited >= limit && waited < limit + Duration::from_secs(5),
            "{waited:?}"
        );
    }

    /// A complete small body is discarded to its end (the request stream is fully consumed).
    #[tokio::test]
    async fn discard_reads_a_complete_body() {
        let consumed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = consumed.clone();
        let chunks = futures_util::stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(IMAGE_HEAD.as_bytes())),
            Ok(Bytes::from_static(b"\r\n--XBOUNDARYX--\r\n")),
        ])
        .chain(futures_util::stream::poll_fn(move |_| {
            flag.store(true, Ordering::SeqCst);
            std::task::Poll::Ready(None)
        }));
        let req = Request::post("/")
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=XBOUNDARYX",
            )
            .body(Body::from_stream(chunks))
            .unwrap();
        discard_body(multipart_of(req).await).await;
        assert!(consumed.load(Ordering::SeqCst));
    }
}
