//! The Review tab's memory of a camera: the reviewer's history (`GET /v1/review/history`), the
//! spot edits over every unreviewed frame (`POST /v1/review/reject-spot`, `.../accept-spot`) and
//! the spot store ([`labels::spots`]: `GET /v1/review/spots`, `POST /v1/review/spots/parked`,
//! `.../retire`, `.../moved`).

use super::{
    ACCEPT_AUTO_DEFAULT_LIMIT, Answer, AppState, Problem, blocking, bulk_limit, context, fail,
    filters, hidden, json_body, not_found, now, push_failed, push_past_limit, query_map, success,
};
use crate::collect::labels::{self, Frame, bulk};
use crate::collect::layout;
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, QueryRejection};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

/// `GET /v1/review/history?camera=<c>`: the reviewer's decisions on that camera's frames in the
/// folder under review (accepted / rejected model and auto boxes, not by a bulk rule; newest
/// first, at most [`labels::repeats::HISTORY_CAP`], `truncated` past it), for the Review tab's
/// "rejected here before" hint. Cached until a label file of the folder is written.
pub(in crate::server) async fn history(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Answer {
    let camera = spot_camera(query_map(query).get("camera"))?;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let h = labels::repeats::history(&dir, &camera);
        success(serde_json::json!({
            "camera": h.camera,
            "decisions": h.decisions,
            "truncated": h.truncated,
        }))
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotBody {
    camera: String,
    label: String,
    bbox: [f32; 4],
    width: u32,
    height: u32,
    #[serde(default)]
    min_iou: Option<f32>,
    /// reject-spot only: the reviewer's reason (`reject_reason`).
    #[serde(default)]
    reject_reason: Option<labels::RejectReason>,
    /// The queue the page shows (`queue|all`; default queue).
    #[serde(default)]
    source: Option<String>,
    /// A frame left alone (the one the page shows: it applies the rule there itself, so its
    /// open edits do not run into a save conflict).
    #[serde(default)]
    skip_frame: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-spot` with `{camera, label, bbox, width, height, min_iou?,
/// reject_reason?, source?, skip_frame?, dry_run, limit?}`: on the unreviewed frames of that camera (of the
/// same size), every pending model / auto box of the same object class (vehicle classes count as
/// one) at IoU >= `min_iou` (default 0.6, 0.3..=1) with `bbox` is rejected (`reject_reason` as
/// given; `reason: repeat_spot` when it had no provenance) and the frame marked repeat-spot
/// ([`bulk::bulk_spot`]), at most `limit` frames (default 500). `dry_run` only counts.
pub(in crate::server) async fn reject_spot(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    spot_route(state, headers, body, false).await
}

/// `POST /v1/review/accept-spot`: [`reject_spot`]'s mirror for a real object that stays put (a
/// parked car): the pending boxes of the same label at IoU >= `min_iou` (default and minimum
/// 0.8) are accepted. `reject_reason` is refused.
pub(in crate::server) async fn accept_spot(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    spot_route(state, headers, body, true).await
}

async fn spot_route(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
    accept: bool,
) -> Answer {
    use labels::repeats::{ACCEPT_SPOT_IOU, REJECT_SPOT_IOU, Spot, Verdict};
    let b: SpotBody = json_body(&headers, body)?;
    let camera = b.camera.trim().to_string();
    if !layout::camera_ok(&camera) {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            "camera: not a camera name of the dataset",
        ));
    }
    if crate::collect::config::validate_classes(std::slice::from_ref(&b.label)).is_err() {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            "label: not a valid class name",
        ));
    }
    let [x0, y0, x1, y1] = b.bbox;
    if !b.bbox.iter().all(|v| v.is_finite())
        || x1 <= x0
        || y1 <= y0
        || b.width == 0
        || b.height == 0
    {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            "bbox: give [x0, y0, x1, y1] in pixels of the width x height frame, x1 > x0, y1 > y0",
        ));
    }
    let (lo, default) = if accept {
        (ACCEPT_SPOT_IOU, ACCEPT_SPOT_IOU)
    } else {
        (0.3, REJECT_SPOT_IOU)
    };
    let min_iou = b.min_iou.unwrap_or(default);
    if !(lo..=1.0).contains(&min_iou) {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            format!("min_iou: must be between {lo} and 1, got {min_iou}"),
        ));
    }
    if accept && b.reject_reason.is_some() {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            "reject_reason: only for reject-spot",
        ));
    }
    if let Some(labels::RejectReason::Other(r)) = &b.reject_reason {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            format!(
                "reject_reason: '{}' is not a reason this build knows",
                r.escape_debug()
            ),
        ));
    }
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let mut q = BTreeMap::new();
    if let Some(s) = b.source {
        q.insert("source".to_string(), s);
    }
    let filters = filters(&q)?;
    let spot = Spot {
        camera,
        label: b.label,
        bbox: b.bbox,
        width: b.width,
        height: b.height,
        min_iou,
    };
    let verdict = if accept {
        Verdict::Accept
    } else {
        Verdict::Reject(b.reject_reason)
    };
    let (dry_run, skip) = (b.dry_run, b.skip_frame.unwrap_or_default());
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let o = bulk::bulk_spot(
            &dir,
            &filters,
            &classes,
            &hidden,
            &spot,
            &verdict,
            limit,
            dry_run,
            &|r| r != skip,
        );
        let verb = if accept { "Accepted" } else { "Rejected" };
        let mut message = format!(
            "{verb} {} {} box(es) at that spot on {} unreviewed {} frame(s) (the frames stay \
             unreviewed; frames left with nothing to decide can be saved with \"Save \
             repeat-only frames\").",
            o.boxes, spot.label, o.done_count, spot.camera
        );
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        if !dry_run {
            push_past_limit(&mut message, o.eligible, limit, "more frame(s)");
        }
        if !dry_run {
            tracing::info!(
                boxes = o.boxes,
                frames = o.done_count,
                camera = %spot.camera,
                label = %spot.label,
                accept,
                "review: spot edited on every unreviewed frame"
            );
        }
        success(serde_json::json!({
            "eligible": o.eligible,
            "boxes": o.boxes,
            "changed": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": (!dry_run).then_some(message),
        }))
    })
    .await)
}

/// The camera of a spots route, or 400.
fn spot_camera(camera: Option<&String>) -> Result<String, Problem> {
    camera
        .map(|c| c.trim().to_string())
        .filter(|c| layout::camera_ok(c))
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "give `camera` (a camera name of the dataset, as in the frame paths)".to_string(),
            )
        })
}

/// A box of a spots route, or 400: a valid class name, `[x0, y0, x1, y1]` in pixels, inside a
/// `width` x `height` frame once clamped, both sides at least [`labels::MIN_BOX_PX`].
fn spot_box(label: &str, bbox: [f32; 4], width: u32, height: u32) -> Result<[f32; 4], Problem> {
    if crate::collect::config::validate_classes(&[label.to_string()]).is_err() {
        return Err((
            StatusCode::BAD_REQUEST,
            "label: not a valid class name".to_string(),
        ));
    }
    let c = labels::clamp_box(bbox, width, height);
    let finite = bbox.iter().all(|v| v.is_finite());
    if !finite || c[2] - c[0] < labels::MIN_BOX_PX || c[3] - c[1] < labels::MIN_BOX_PX {
        return Err((
            StatusCode::BAD_REQUEST,
            "bbox: give [x0, y0, x1, y1] in pixels inside the frame".to_string(),
        ));
    }
    Ok(c)
}

/// `GET /v1/review/spots?camera=<c>`: the camera's parked spots (every frame size, retired ones
/// too), its rejected spots as of today, derived from the history
/// ([`labels::spots::CameraMemory::rejected_spots`]), `moved_at`, and how many spots
/// "Camera moved" would forget (`live`).
pub(in crate::server) async fn spots(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Answer {
    let camera = spot_camera(query_map(query).get("camera"))?;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let store = match labels::spots::read(&dir, &camera) {
            Ok(s) => s,
            Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        };
        let today = chrono::Utc::now().date_naive();
        let memory =
            labels::spots::CameraMemory::new(&labels::repeats::history(&dir, &camera), store);
        let rejected = memory.rejected_spots(today);
        let store = memory.store();
        success(serde_json::json!({
            "camera": camera,
            "moved_at": store.moved_at,
            "parked": store.parked,
            "rejected": rejected,
            "live": store.live_count() + rejected.len(),
        }))
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotParkedBody {
    frame: String,
    label: String,
    bbox: [f32; 4],
}

/// `POST /v1/review/spots/parked` with `{frame, label, bbox}` (the page's "Real, parked"): a parked
/// spot on the frame's camera and size, from the frame's date ([`labels::spots::add_parked`]). The
/// page accepts the box itself, in its unsaved edits. Answers the spot and whether it is `new`.
pub(in crate::server) async fn spots_parked(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: SpotParkedBody = json_body(&headers, body)?;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let Some(frame) = Frame::resolve(&dir, &b.frame) else {
            return not_found();
        };
        let rec = match frame.record() {
            Ok(r) => r,
            Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        };
        let bbox = match spot_box(&b.label, b.bbox, rec.width, rec.height) {
            Ok(c) => c,
            Err((s, m)) => return fail(s, m),
        };
        let at = labels::spots::Place {
            camera: &frame.rel.camera,
            width: rec.width,
            height: rec.height,
            label: &b.label,
            bbox,
        };
        match labels::spots::add_parked(&dir, &at, &frame.rel.date) {
            Ok((spot, new)) => {
                tracing::info!(
                    camera = at.camera,
                    label = at.label,
                    new,
                    "review: parked spot"
                );
                success(serde_json::json!({ "spot": spot, "new": new }))
            }
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotRetireBody {
    camera: String,
    width: u32,
    height: u32,
    label: String,
    bbox: [f32; 4],
}

/// `POST /v1/review/spots/retire` with `{camera, width, height, label, bbox}`: that parked spot
/// stops counting ([`labels::spots::retire`]); 404 when there is none.
pub(in crate::server) async fn spots_retire(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: SpotRetireBody = json_body(&headers, body)?;
    let camera = spot_camera(Some(&b.camera))?;
    let bbox = spot_box(&b.label, b.bbox, b.width, b.height)?;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let at = labels::spots::Place {
            camera: &camera,
            width: b.width,
            height: b.height,
            label: &b.label,
            bbox,
        };
        let today = crate::collect::utc_date(std::time::SystemTime::now());
        match labels::spots::retire(&dir, &at, &today) {
            Ok(true) => success(serde_json::json!({})),
            Ok(false) => fail(StatusCode::NOT_FOUND, "no parked spot there"),
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotMovedBody {
    camera: String,
}

/// `POST /v1/review/spots/moved` with `{camera}` ("Camera moved: forget spots"): every parked spot
/// retires and `moved_at` is set, so rejected spots from before it stop counting too
/// ([`labels::spots::camera_moved`]). A run already going keeps the spots it read at its start.
pub(in crate::server) async fn spots_moved(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: SpotMovedBody = json_body(&headers, body)?;
    let camera = spot_camera(Some(&b.camera))?;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let at = now();
        match labels::spots::camera_moved(&dir, &camera, &at) {
            Ok(retired) => {
                tracing::info!(camera, retired, "review: camera moved, spots forgotten");
                success(serde_json::json!({ "retired": retired, "moved_at": at }))
            }
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await)
}
