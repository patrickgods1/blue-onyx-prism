//! Detection endpoints: multipart parsing, dispatch to the model workers, `custom/list`.

use super::AppState;
use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::metrics::{DropReason, InflightPermit, RejectReason};
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use axum::Json;
use axum::body::Body;
use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::{FromRequest, Multipart, Path, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use crossbeam_channel::TrySendError;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Maximum accepted request body (multipart image upload).
///
/// Worst-case memory of the detection routes: every admitted request may hold one full body,
/// so uploads take at most `max_inflight × BODY_LIMIT` (see [`max_inflight`]; e.g. 64 × 32 MiB
/// = 2 GiB with the default cap, the bound the in-flight cap exists for), plus, per
/// preprocessing thread, one decoded frame of up to `max_image_pixels × 3` bytes (RGB8) and its
/// resized input tensor. Requests beyond the cap are answered 503 before their body is read.
pub(crate) const BODY_LIMIT: usize = 32 * 1024 * 1024;

/// Added to the worker's request timeout while a handler waits for its reply, so the worker's
/// own "waited longer than the timeout" answer wins the race and reaches the client.
pub(crate) const RESPONSE_GRACE: Duration = Duration::from_millis(500);

/// Longest a detection request may take to deliver its multipart body; a slower (or stalled)
/// upload is answered with an error and its connection closed, releasing its in-flight slot.
pub(crate) const MULTIPART_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Server-wide in-flight cap used when the models report no queue capacity (none loaded yet,
/// all failed): requests answered without inference still hold their upload while it is read.
pub(crate) const DEFAULT_MAX_INFLIGHT: usize = 64;

/// Added to the sum of the models' queue capacities to form the in-flight cap: requests being
/// inferred (out of the queue), waiting for an early answer, or reading their body.
pub(crate) const INFLIGHT_MARGIN: usize = 16;

/// Most of an unwanted body (a request answered before its upload is needed: unknown model,
/// failed or compiling model, full queue) read and thrown away, chunk by chunk (nothing is
/// buffered, so this costs bandwidth, not memory). It equals [`BODY_LIMIT`] on purpose: a body
/// left unread makes hyper close the connection after the answer, and closing a socket with
/// unread upload data can reset it before the client reads the answer; a pooling client (Blue
/// Iris, reqwest) then also fails its next request on that connection.
pub(crate) const DISCARD_LIMIT: usize = BODY_LIMIT;

/// Longest spent discarding an unwanted body (see [`DISCARD_LIMIT`]); after it the body is left
/// unread and the connection closes.
pub(crate) const DISCARD_TIMEOUT: Duration = MULTIPART_READ_TIMEOUT;

/// Response header carrying the id of a detection request: the id the worker puts in its
/// per-request log line ([`crate::worker::WorkItem::DetectWithId`]) and the handler in its
/// debug line.
pub(crate) const REQUEST_ID_HEADER: &str = "x-request-id";

pub(super) use crate::worker::new_request_id;

/// `resp` with the [`REQUEST_ID_HEADER`] set to `id`.
fn with_request_id(mut resp: Response, id: &str) -> Response {
    if let Ok(v) = HeaderValue::from_str(id) {
        resp.headers_mut().insert(REQUEST_ID_HEADER, v);
    }
    resp
}

/// Why a detection body could not be used: the answer text and the metric reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BodyError {
    pub(super) reason: RejectReason,
    pub(super) message: String,
}

impl BodyError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            reason: RejectReason::BadRequest,
            message: message.into(),
        }
    }

    /// From a multipart read error: too large (the body limit) or malformed.
    fn from_multipart(e: &MultipartError, what: &str) -> Self {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            Self {
                reason: RejectReason::BodyTooLarge,
                message: format!(
                    "The upload is larger than the {} MiB limit; send a smaller image \
                     (e.g. a JPEG of the camera frame).",
                    BODY_LIMIT / (1024 * 1024)
                ),
            }
        } else {
            Self::bad_request(format!(
                "{what}: {}; the upload was cut off or is not valid multipart/form-data, \
                 resend the image as the multipart field 'image'.",
                e.body_text()
            ))
        }
    }

    /// Count this rejection in the server metrics.
    pub(super) fn record(&self, state: &AppState) {
        state.metrics.record_rejected(self.reason);
    }
}

/// A detection upload: the multipart stream, or (not multipart/form-data) the rejection and the
/// unread body, kept so it can be discarded before the answer (SV-13: a body left unread makes
/// hyper close the connection, and a pooling client then fails its next request on it).
pub(super) struct Upload(Result<Multipart, NotMultipart>);

pub(super) struct NotMultipart {
    rejection: MultipartRejection,
    body: Body,
}

impl<S: Send + Sync> FromRequest<S> for Upload {
    type Rejection = Infallible;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let (parts, body) = req.into_parts();
        // `Multipart` rejects only on the content type: probe it on a body-less request so a
        // rejected body stays here to be drained.
        let mut probe = Request::new(Body::empty());
        if let Some(ct) = parts.headers.get(header::CONTENT_TYPE) {
            probe.headers_mut().insert(header::CONTENT_TYPE, ct.clone());
        }
        if let Err(rejection) = Multipart::from_request(probe, state).await {
            return Ok(Self(Err(NotMultipart { rejection, body })));
        }
        let mp = Multipart::from_request(Request::from_parts(parts, body), state).await;
        Ok(Self(mp.map_err(|rejection| NotMultipart {
            rejection,
            body: Body::empty(),
        })))
    }
}

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
    state.metrics.record_rejected(RejectReason::Overloaded);
    if let Some(held_back) = state.metrics.overload_log.allow(state.metrics.now_ms()) {
        warn!(
            limit,
            similar_since_last_warning = held_back,
            "server is overloaded: too many detection requests in flight; answering 503"
        );
    }
    let mut r = VisionDetectionResponse::error(format!(
        "Server is overloaded ({limit} detection requests in flight); more cameras send \
         images than the models can serve, so retry later or lower the trigger rate."
    ));
    r.command = command.to_string();
    Err(Box::new(
        (StatusCode::SERVICE_UNAVAILABLE, Json(r)).into_response(),
    ))
}

/// Read and throw away the start of a body that will not be used: at most [`DISCARD_LIMIT`]
/// bytes within [`DISCARD_TIMEOUT`]. A body that ends within both leaves the connection
/// reusable; otherwise hyper closes it after the answer (it drains only what is buffered).
async fn discard_body(upload: Upload) {
    discard_body_within(upload, DISCARD_TIMEOUT).await;
}

/// [`discard_body`] with an explicit time limit.
async fn discard_body_within(upload: Upload, limit: Duration) {
    match upload.0 {
        Ok(mut mp) => {
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
        Err(not) => discard_raw_within(not.body, limit).await,
    }
}

/// Read and drop a raw (not multipart) body: at most [`DISCARD_LIMIT`] bytes within `limit`.
async fn discard_raw_within(body: Body, limit: Duration) {
    let drain = async move {
        let mut stream = body.into_data_stream();
        let mut seen = 0usize;
        while let Some(Ok(chunk)) = stream.next().await {
            seen = seen.saturating_add(chunk.len());
            if seen > DISCARD_LIMIT {
                return;
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
) -> Result<(VisionDetectionRequest, HashMap<String, String>), BodyError> {
    let mut req = VisionDetectionRequest::default();
    let mut fields = HashMap::new();
    let mut got_image = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return Err(BodyError::from_multipart(&e, "Invalid multipart body")),
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
                    .map_err(|e| BodyError::from_multipart(&e, "Reading the image field failed"))?;
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
        return Err(BodyError::bad_request(
            "No image provided: the upload has no (or an empty) multipart field 'image'; \
             send the picture in a form field named 'image'.",
        ));
    }
    Ok((req, fields))
}

/// Parse a detection upload within [`MULTIPART_READ_TIMEOUT`]. The image field is read into
/// one `Bytes` (`Field::bytes`) that moves on to the worker without another copy.
/// A body that is not multipart/form-data is discarded (bounded, see [`discard_body`]) before
/// the error is returned, so the connection stays usable (SV-13).
pub(super) async fn parse_detection_body(
    upload: Upload,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), BodyError> {
    parse_body_within(upload, MULTIPART_READ_TIMEOUT).await
}

/// [`parse_detection_body`] with an explicit time limit.
async fn parse_body_within(
    upload: Upload,
    limit: Duration,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), BodyError> {
    match upload.0 {
        Ok(mp) => tokio::time::timeout(limit, read_multipart(mp))
            .await
            .unwrap_or_else(|_| {
                Err(BodyError::bad_request(format!(
                    "Request body not received within {} s; the client stalled or the network \
                     is too slow, so retry the request.",
                    limit.as_secs()
                )))
            }),
        Err(NotMultipart { rejection, body }) => {
            discard_raw_within(body, limit.min(DISCARD_TIMEOUT)).await;
            Err(BodyError::bad_request(format!(
                "Expected a multipart/form-data body ({}); the image was probably posted raw, \
                 so send it as the multipart form field 'image'.",
                rejection.body_text()
            )))
        }
    }
}

pub(super) async fn detection_default(
    State(state): State<Arc<AppState>>,
    upload: Upload,
) -> Response {
    let start = Instant::now();
    let id = new_request_id();
    // The permit covers the whole request: body read, queue wait and the worker's reply.
    let _permit = match admit(&state, "detect") {
        Ok(p) => p,
        Err(resp) => return with_request_id(*resp, &id),
    };
    let Some(handle) = state.registry.default_model() else {
        discard_body(upload).await;
        let resp = Json(VisionDetectionResponse::error(
            "No default model: every model is disabled or none is configured; enable a model \
             on the Config page.",
        ));
        return with_request_id(resp.into_response(), &id);
    };
    let resp = detect_on(&state, &handle, upload, start, "detect", &id).await;
    with_request_id(Json(resp).into_response(), &id)
}

pub(super) async fn detection_custom(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
    upload: Upload,
) -> Response {
    let start = Instant::now();
    let id = new_request_id();
    let _permit = match admit(&state, "custom") {
        Ok(p) => p,
        Err(resp) => return with_request_id(*resp, &id),
    };
    let Some(handle) = state.registry.by_name(&model) else {
        discard_body(upload).await;
        // Client-chosen text: shortened and debug-escaped before it reaches the log.
        warn!(model = ?short(&model), request_id = %id, "request for unknown model");
        let mut r = VisionDetectionResponse::error(format!(
            "Unknown model '{model}': it is not configured or not enabled; check the model \
             name in Blue Iris against /v1/vision/custom/list."
        ));
        r.command = "custom".into();
        return with_request_id(Json(r).into_response(), &id);
    };
    let resp = detect_on(&state, &handle, upload, start, "custom", &id).await;
    with_request_id(Json(resp).into_response(), &id)
}

/// Parse the upload, counting a rejection.
async fn parse_counted(state: &AppState, upload: Upload) -> Result<VisionDetectionRequest, String> {
    match parse_detection_body(upload).await {
        Ok((r, _)) => Ok(r),
        Err(e) => {
            e.record(state);
            Err(e.message)
        }
    }
}

/// A detection on `handle`. When the model cannot take it now (compiling, failed, queue full)
/// the answer is known before the upload is needed: the body is discarded (bounded, see
/// [`discard_body`]) instead of buffered, and the request answered at once.
async fn detect_on(
    state: &AppState,
    handle: &WorkerHandle,
    upload: Upload,
    start: Instant,
    command: &str,
    request_id: &str,
) -> VisionDetectionResponse {
    if !matches!(handle.state.get(), ModelState::Ready) || handle.is_full() {
        handle.metrics.requests.fetch_add(1, Ordering::Relaxed);
        if let Some(early) = answer_without_inference(handle) {
            discard_body(upload).await;
            return finish(state, handle, early, start, command, request_id);
        }
        // It became ready (or got room) meanwhile: serve it normally (the request is
        // already counted).
        let resp = match parse_counted(state, upload).await {
            Ok(r) => send_to_worker(handle, r, start, request_id).await,
            Err(e) => VisionDetectionResponse::error(e),
        };
        return finish(state, handle, resp, start, command, request_id);
    }
    let req = parse_counted(state, upload).await;
    run_detection(state, handle, req, start, command, request_id).await
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
    request_id: &str,
) -> VisionDetectionResponse {
    let resp = dispatch(handle, req, start, request_id).await;
    finish(state, handle, resp, start, command, request_id)
}

/// Stamp the command, round trip and GPU flag on `resp` (and record the round trip of a
/// successful detection); logs one debug line with the request id, model and timings.
fn finish(
    state: &AppState,
    handle: &WorkerHandle,
    mut resp: VisionDetectionResponse,
    start: Instant,
    command: &str,
    request_id: &str,
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
    debug!(
        request_id,
        model = %handle.name,
        command,
        success = resp.success,
        count = resp.count,
        inference_ms = resp.inferenceMs,
        process_ms = resp.processMs,
        round_trip_ms = resp.analysisRoundTripMs,
        "detection answered"
    );
    resp
}

pub(super) async fn dispatch(
    handle: &WorkerHandle,
    req: Result<VisionDetectionRequest, String>,
    start: Instant,
    request_id: &str,
) -> VisionDetectionResponse {
    let req = match req {
        Ok(r) => r,
        Err(e) => return VisionDetectionResponse::error(e),
    };
    handle.metrics.requests.fetch_add(1, Ordering::Relaxed);
    if let Some(r) = answer_without_inference(handle) {
        return r;
    }
    send_to_worker(handle, req, start, request_id).await
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
            // A stuck inference (worker::collect_watched) is reported with the worker's own
            // wording, not as a load failure.
            let text = if handle.is_stuck() {
                format!("Model '{name}' is unavailable: {m}")
            } else {
                format!("Model '{name}' failed to load: {m}")
            };
            return Some(VisionDetectionResponse::error(text));
        }
    }
    if handle.is_full() {
        // Throttled: under a sustained overload this would otherwise log (and, in the
        // service, write to the event log) one line per rejected request.
        if let Some(held_back) = handle.metrics.queue_full_log.allow(handle.metrics.now_ms()) {
            warn!(
                model = %name,
                queue = handle.queue_capacity(),
                similar_since_last_warning = held_back,
                "worker queue is full, server is overloaded; rejecting request"
            );
        }
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
    request_id: &str,
) -> VisionDetectionResponse {
    let name = &handle.name;
    let (tx, rx) = tokio::sync::oneshot::channel();
    match handle
        .sender
        .try_send(crate::worker::WorkItem::DetectWithId(
            req,
            tx,
            start,
            request_id.to_string(),
        )) {
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
    use axum::body::Bytes;

    const IMAGE_HEAD: &str = "--XBOUNDARYX\r\nContent-Disposition: form-data; name=\"image\"; \
                              filename=\"t.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nxx";

    /// A multipart upload to `uri` that sends the start of its image and then stalls forever.
    fn stalled(uri: &str) -> Request {
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

    async fn multipart_of(req: Request) -> Upload {
        match Upload::from_request(req, &()).await {
            Ok(u) => u,
            Err(never) => match never {},
        }
    }

    /// A body of `chunks` that sets the returned flag once it has been read to its end.
    fn tracked_body(chunks: Vec<Bytes>) -> (Body, Arc<std::sync::atomic::AtomicBool>) {
        let consumed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = consumed.clone();
        let stream = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>))
            .chain(futures_util::stream::poll_fn(move |_| {
                flag.store(true, Ordering::SeqCst);
                std::task::Poll::Ready(None)
            }));
        (Body::from_stream(stream), consumed)
    }

    /// SV-13: a raw (not multipart) image post is read to its end before the error answer, so
    /// a keep-alive connection survives; the rejection is counted as `bad_request`.
    #[tokio::test]
    async fn raw_image_post_is_drained_and_counted() {
        let (body, consumed) = tracked_body(vec![Bytes::from_static(b"\xFF\xD8 raw jpeg")]);
        let req = Request::post("/")
            .header(header::CONTENT_TYPE, "image/jpeg")
            .body(body)
            .unwrap();
        let err = parse_detection_body(multipart_of(req).await)
            .await
            .unwrap_err();
        assert!(consumed.load(Ordering::SeqCst), "body drained");
        assert_eq!(err.reason, RejectReason::BadRequest);
        assert!(
            err.message
                .starts_with("Expected a multipart/form-data body"),
            "{}",
            err.message
        );
        assert!(err.message.contains("field 'image'"), "{}", err.message);

        // Through the router: a model that cannot take the request (unknown name, failed
        // model) discards the raw body as well, and the answer carries a request id.
        let state = test_state();
        for uri in ["/v1/vision/custom/nope", "/v1/vision/detection"] {
            let (body, consumed) = tracked_body(vec![Bytes::from_static(b"raw")]);
            let req = Request::post(uri)
                .header(header::CONTENT_TYPE, "image/jpeg")
                .body(body)
                .unwrap();
            let (status, headers, _) = call(&state, req).await;
            assert_eq!(status, StatusCode::OK);
            assert!(consumed.load(Ordering::SeqCst), "{uri}: body drained");
            let id = headers[REQUEST_ID_HEADER].to_str().unwrap();
            assert!(
                !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit()),
                "{id}"
            );
        }
    }

    /// An upload over the body limit is counted as `body_too_large` (the extractor's default
    /// limit of 2 MB applies here, without the router's `DefaultBodyLimit` layer).
    #[tokio::test]
    async fn oversized_upload_is_body_too_large() {
        let mut body = IMAGE_HEAD.as_bytes().to_vec();
        body.resize(body.len() + 3 * 1024 * 1024, b'x');
        body.extend_from_slice(b"\r\n--XBOUNDARYX--\r\n");
        let req = Request::post("/")
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=XBOUNDARYX",
            )
            .body(Body::from(body))
            .unwrap();
        let err = parse_detection_body(multipart_of(req).await)
            .await
            .unwrap_err();
        assert_eq!(err.reason, RejectReason::BodyTooLarge, "{}", err.message);
        assert!(err.message.contains("MiB limit"), "{}", err.message);

        let state = test_state();
        err.record(&state);
        BodyError::bad_request("x").record(&state);
        let text = crate::metrics::render_prometheus(&state.metrics, &[]);
        assert!(
            text.contains("blue_onyx_prism_requests_rejected_total{reason=\"body_too_large\"} 1")
        );
        assert!(text.contains("blue_onyx_prism_requests_rejected_total{reason=\"bad_request\"} 1"));
    }

    /// A multipart body without an `image` field is a `bad_request`.
    #[tokio::test]
    async fn missing_image_is_bad_request() {
        let body = "--XBOUNDARYX\r\nContent-Disposition: form-data; name=\"min_confidence\"\r\n\r\n0.4\r\n--XBOUNDARYX--\r\n";
        let req = Request::post("/")
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=XBOUNDARYX",
            )
            .body(Body::from(body))
            .unwrap();
        let err = parse_detection_body(multipart_of(req).await)
            .await
            .unwrap_err();
        assert_eq!(err.reason, RejectReason::BadRequest);
        assert!(
            err.message.starts_with("No image provided"),
            "{}",
            err.message
        );
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
        assert_eq!(state.metrics.rejected(RejectReason::Overloaded), 2);
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
        assert_eq!(err.reason, RejectReason::BadRequest);
        assert!(
            err.message
                .starts_with("Request body not received within 1 s"),
            "{}",
            err.message
        );
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

    /// What the detection parser made of one body: the image size or the rejection.
    #[derive(Debug, PartialEq)]
    enum Parsed {
        Image(usize),
        Rejected(RejectReason),
    }

    /// Answers like the detection API does for a parse error (`VisionDetectionResponse::error`,
    /// 200, `success: false`), plus the reject reason; `{"success": true, "bytes": n}` otherwise.
    async fn parse_only(upload: Upload) -> Response {
        match parse_detection_body(upload).await {
            Ok((r, _)) => Json(serde_json::json!({ "success": true, "bytes": r.image_data.len() }))
                .into_response(),
            Err(e) => {
                let mut v = serde_json::to_value(VisionDetectionResponse::error(e.message))
                    .expect("response serializes");
                v["reason"] = e.reason.as_str().into();
                Json(v).into_response()
            }
        }
    }

    /// Run `body` (with content type `ct`) through the real `Upload` extractor and
    /// [`parse_detection_body`] behind the router's body limit, and check the answer is JSON:
    /// either an accepted non-empty image or `success: false` with a message.
    async fn parse_via_router(ct: &str, body: Vec<u8>) -> Parsed {
        let app = axum::Router::new()
            .route("/p", axum::routing::post(parse_only))
            .layer(axum::extract::DefaultBodyLimit::max(BODY_LIMIT));
        let req = Request::post("/p")
            .header(header::CONTENT_TYPE, ct)
            .body(Body::from(body))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("a JSON answer");
        if v["success"] == true {
            let n = v["bytes"].as_u64().unwrap() as usize;
            assert!(n > 0, "an accepted upload has an image");
            return Parsed::Image(n);
        }
        assert_eq!(v["success"], false, "{v}");
        assert!(!v["message"].as_str().unwrap().is_empty(), "{v}");
        assert_eq!(v["error"], v["message"], "{v}");
        match v["reason"].as_str().unwrap() {
            "bad_request" => Parsed::Rejected(RejectReason::BadRequest),
            "body_too_large" => Parsed::Rejected(RejectReason::BodyTooLarge),
            other => panic!("unexpected reject reason {other}"),
        }
    }

    /// The same body through the real detection route of a model that cannot take it (the
    /// body is discarded, not parsed): always a JSON `success: false` answer, never a panic.
    async fn discard_via_api(state: &Arc<AppState>, ct: &str, body: Vec<u8>) {
        let req = Request::post("/v1/vision/custom/ipcam-general")
            .header(header::CONTENT_TYPE, ct)
            .body(Body::from(body))
            .unwrap();
        let (status, _, body) = call(state, req).await;
        assert_eq!(status, StatusCode::OK);
        let r: VisionDetectionResponse = serde_json::from_slice(&body).unwrap();
        assert!(!r.success);
    }

    const MP: &str = "multipart/form-data; boundary=XBOUNDARYX";

    /// One `image` part with `data`, then the closing boundary.
    fn image_body(data: &[u8]) -> Vec<u8> {
        let mut b = IMAGE_HEAD.as_bytes()[..IMAGE_HEAD.len() - 2].to_vec();
        b.extend_from_slice(data);
        b.extend_from_slice(b"\r\n--XBOUNDARYX--\r\n");
        b
    }

    /// Crafted malformed multipart bodies: each is answered (JSON, accepted or `success:
    /// false`) without a panic, both by the parser and by the discard path of the API.
    #[tokio::test]
    async fn multipart_parser_survives_crafted_bodies() {
        use Parsed::*;
        use RejectReason::*;
        let part = |headers: &str, data: &str| {
            format!("--XBOUNDARYX\r\n{headers}\r\n\r\n{data}\r\n").into_bytes()
        };
        let img_hdr = "Content-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"";
        let mut cases: Vec<(&str, Vec<u8>, Option<Parsed>)> = vec![
            ("empty body", Vec::new(), Some(Rejected(BadRequest))),
            ("well formed", image_body(b"jpeg"), Some(Image(4))),
            (
                "missing final boundary",
                [IMAGE_HEAD.as_bytes(), b"\r\n"].concat(),
                Some(Rejected(BadRequest)),
            ),
            (
                "boundary never appears",
                b"just some bytes\r\nwithout any boundary".to_vec(),
                Some(Rejected(BadRequest)),
            ),
            (
                "closing boundary with trailing junk",
                [image_body(b"jpeg"), b"\x00\xFFjunk after the end".to_vec()].concat(),
                None,
            ),
            (
                "boundary with trailing junk on its line",
                b"--XBOUNDARYXjunk\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nx\r\n--XBOUNDARYX--\r\n".to_vec(),
                None,
            ),
            (
                "part with no headers",
                b"--XBOUNDARYX\r\n\r\ndata\r\n--XBOUNDARYX--\r\n".to_vec(),
                Some(Rejected(BadRequest)),
            ),
            (
                "header line over 8 KiB",
                [
                    part(&format!("{img_hdr}\r\nX-Long: {}", "a".repeat(9 * 1024)), "x"),
                    b"--XBOUNDARYX--\r\n".to_vec(),
                ]
                .concat(),
                None,
            ),
            (
                "nested multipart part",
                [
                    part(
                        "Content-Disposition: form-data; name=\"inner\"\r\n\
                         Content-Type: multipart/mixed; boundary=INNER",
                        "--INNER\r\nContent-Type: text/plain\r\n\r\nhi\r\n--INNER--",
                    ),
                    part(img_hdr, "jpeg"),
                    b"--XBOUNDARYX--\r\n".to_vec(),
                ]
                .concat(),
                Some(Image(4)),
            ),
            (
                "LF-only line endings",
                b"--XBOUNDARYX\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\n\njpeg\n--XBOUNDARYX--\n".to_vec(),
                None,
            ),
            (
                "non-UTF-8 field name",
                [
                    b"--XBOUNDARYX\r\nContent-Disposition: form-data; name=\"\xFF\xFE\x80\"\r\n\r\nv\r\n".to_vec(),
                    part(img_hdr, "jpeg"),
                    b"--XBOUNDARYX--\r\n".to_vec(),
                ]
                .concat(),
                None,
            ),
            (
                "duplicated image fields (the last one wins)",
                [
                    part(img_hdr, "first"),
                    part(img_hdr, "second!"),
                    b"--XBOUNDARYX--\r\n".to_vec(),
                ]
                .concat(),
                Some(Image(7)),
            ),
            (
                "empty image field",
                [part(img_hdr, ""), b"--XBOUNDARYX--\r\n".to_vec()].concat(),
                Some(Rejected(BadRequest)),
            ),
        ];
        let mut many = Vec::new();
        for _ in 0..10_000 {
            many.extend_from_slice(b"--XBOUNDARYX\r\n\r\n\r\n");
        }
        many.extend_from_slice(b"--XBOUNDARYX--\r\n");
        cases.push(("10k empty parts", many, Some(Rejected(BadRequest))));

        let state = test_state();
        for (what, body, want) in cases {
            let got = parse_via_router(MP, body.clone()).await;
            if let Some(want) = want {
                assert_eq!(got, want, "{what}");
            }
            discard_via_api(&state, MP, body).await;
        }
        // A content type without a boundary, and a multipart type that is not form-data.
        for ct in [
            "multipart/form-data",
            "multipart/mixed; boundary=XBOUNDARYX",
        ] {
            let got = parse_via_router(ct, image_body(b"jpeg")).await;
            assert_eq!(got, Rejected(BadRequest), "{ct}");
            discard_via_api(&state, ct, image_body(b"jpeg")).await;
        }
    }

    /// A body of exactly [`BODY_LIMIT`] bytes is accepted; one byte more is `body_too_large`.
    #[tokio::test]
    async fn multipart_body_at_the_limit_and_one_byte_over() {
        let overhead = image_body(b"").len();
        let at = image_body(&vec![b'x'; BODY_LIMIT - overhead]);
        assert_eq!(at.len(), BODY_LIMIT);
        assert_eq!(
            parse_via_router(MP, at).await,
            Parsed::Image(BODY_LIMIT - overhead)
        );
        let over = image_body(&vec![b'x'; BODY_LIMIT - overhead + 1]);
        assert_eq!(over.len(), BODY_LIMIT + 1);
        assert_eq!(
            parse_via_router(MP, over).await,
            Parsed::Rejected(RejectReason::BodyTooLarge)
        );
    }

    /// Deterministic byte soup (seeded xorshift, 200 bodies of 0..4 KiB, some framed by the
    /// boundary): the parser and the API's discard path answer every one without a panic.
    #[tokio::test]
    async fn multipart_parser_survives_random_bodies() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        // Fragments that make the soup look like multipart now and then.
        const PIECES: [&[u8]; 8] = [
            b"--XBOUNDARYX",
            b"--XBOUNDARYX--",
            b"\r\n",
            b"\n",
            b"Content-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"",
            b"Content-Disposition: form-data; name=\"min_confidence\"",
            b"\r\n\r\n",
            b"Content-Type: multipart/mixed; boundary=XBOUNDARYX",
        ];
        let state = test_state();
        for i in 0..200 {
            let len = (next() % 4096) as usize;
            let mut body = Vec::with_capacity(len + 64);
            while body.len() < len {
                let r = next();
                if r % 4 == 0 {
                    body.extend_from_slice(PIECES[(r >> 8) as usize % PIECES.len()]);
                } else {
                    body.push((r >> 16) as u8);
                }
            }
            body.truncate(len);
            if i % 3 == 0 {
                body.splice(0..0, b"--XBOUNDARYX\r\n".iter().copied());
            }
            let _ = parse_via_router(MP, body.clone()).await;
            discard_via_api(&state, MP, body).await;
        }
    }

    /// The in-flight cap under a burst: `limit + 8` concurrent requests whose uploads stall;
    /// exactly `limit` hold a slot and the 8 others are answered 503 quickly and counted.
    #[tokio::test]
    async fn burst_over_the_inflight_cap_is_rejected_fast() {
        const LIMIT: usize = 2;
        let state = test_state();
        state.metrics.inflight_limit.store(LIMIT, Ordering::Relaxed);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let tasks: Vec<_> = (0..LIMIT + 8)
            .map(|i| {
                let st = state.clone();
                let tx = tx.clone();
                let uri = if i % 2 == 0 {
                    "/v1/vision/detection"
                } else {
                    "/v1/vision/custom/ipcam-general"
                };
                tokio::spawn(async move {
                    let started = Instant::now();
                    let (status, _, body) = call(&st, stalled(uri)).await;
                    let _ = tx.send((status, body, started.elapsed()));
                })
            })
            .collect();
        drop(tx);
        for _ in 0..8 {
            let (status, body, took) = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                .await
                .expect("8 requests are rejected without waiting for their upload")
                .unwrap();
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            let r: VisionDetectionResponse = serde_json::from_slice(&body).unwrap();
            assert!(!r.success);
            assert!(r.error.as_deref().unwrap().contains("overloaded"), "{r:?}");
            // Generous for loaded CI runners; the rejection does no I/O (see the direct test).
            assert!(took < Duration::from_secs(2), "{took:?}");
        }
        wait_inflight(&state, LIMIT).await;
        assert_eq!(state.metrics.rejected(RejectReason::Overloaded), 8);
        let text = crate::metrics::render_prometheus(&state.metrics, &[]);
        assert!(text.contains("blue_onyx_prism_requests_rejected_total{reason=\"overloaded\"} 8"));
        assert!(rx.try_recv().is_err(), "the admitted requests still wait");
        for t in tasks {
            t.abort();
        }
        wait_inflight(&state, 0).await;
    }

    /// The admission guard itself: with every slot held, each further `admit` answers the
    /// overloaded 503 in under 50 ms and counts it.
    #[tokio::test]
    async fn admit_rejects_at_once_when_every_slot_is_held() {
        let state = test_state();
        state.metrics.inflight_limit.store(3, Ordering::Relaxed);
        let held: Vec<_> = (0..3).map(|_| admit(&state, "detect").unwrap()).collect();
        for n in 1..=8u64 {
            let started = Instant::now();
            let resp = admit(&state, "custom").unwrap_err();
            assert!(started.elapsed() < Duration::from_millis(50));
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(state.metrics.rejected(RejectReason::Overloaded), n);
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let r: VisionDetectionResponse = serde_json::from_slice(&body).unwrap();
            assert!(!r.success && r.command == "custom", "{r:?}");
        }
        drop(held);
        assert_eq!(state.metrics.inflight(), 0);
        assert!(admit(&state, "detect").is_ok());
    }
}
