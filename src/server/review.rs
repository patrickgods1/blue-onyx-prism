//! The Review tab (`/train/review`, admin only) and its JSON API over the dataset collector's
//! frames ([`crate::collect::labels`]): `GET /v1/review/next` (the uncertainty-sorted queue
//! with filters and counts), `GET`/`POST /v1/review/labels?frame=` (a frame's label file, or
//! its pre-fill from the sidecar, with its revision; a save naming a stale revision is refused
//! with 409), `POST /v1/review/accept-auto` (bulk accept of frames the auto-labeler's triage
//! settles), `POST /v1/review/reject-low` (bulk reject of the pending auto boxes scoring
//! below a minimum, `reason: low_score`), `POST /v1/review/reject-other-class` (bulk reject of
//! the pending model and auto boxes of unwanted classes, outside `collect.classes` or in
//! `collect.hidden_classes`, `reason: other_class`; aliases renamed to the list's spelling),
//! `POST /v1/review/reject-unwanted-reviewed` (the accepted boxes of unwanted classes on
//! reviewed frames, on request), `POST /v1/review/reject-stacked` (the duplicates of stacked
//! boxes on reviewed and unreviewed frames, `reason: duplicate`; overlapping boxes are left to a
//! reviewer, `auto=overlaps`), `POST /v1/review/restore-reopened` (frames a re-label reopened, back
//! to reviewed once nothing is pending), `POST /v1/review/export` (YOLO / COCO / X-AnyLabeling /
//! benchmark set), `POST /v1/review/import` (X-AnyLabeling files), `POST /v1/review/classes`
//! (`collect.classes` and `collect.hidden_classes`; classes that become unwanted have their
//! pending boxes rejected right away), `GET /v1/review/class-catalog` (the class picker's
//! catalog, [`crate::training::autolabel::catalog`]) and the spot memory
//! ([`labels::spots`]): `GET /v1/review/spots`, `POST /v1/review/spots/parked`, `.../retire` and
//! `.../moved`.
//! Every frame reference goes through
//! [`Frame::resolve`] (404 otherwise); POST bodies are JSON (415 otherwise) with per-route size
//! caps (413); the admin guard and the cross-site check of the router apply. A folder under review
//! that is not there (an unplugged drive) is its own state, `folder_missing: true` with a message
//! ([`folder_problem`]: 200 from `/v1/review/next`, 503 where the folder is needed), not an empty
//! dataset.

use super::{AppState, Shared, blocking, encode_uri_component, render, save_config};
use crate::collect::CollectSettings;
#[cfg(test)]
use crate::collect::labels::bulk::accept_one;
use crate::collect::labels::export::{
    self, BenchSplit, ClassFilter, ExportRequest, Format, SplitConfig,
};
use crate::collect::labels::queue::{self, Filters};
use crate::collect::labels::{self, Frame, LabelFile, bulk, xany};
use crate::collect::{Bucket, layout};
use askama::Template;
use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, QueryRejection};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Largest `POST /v1/review/labels` body.
pub(super) const LABELS_BODY_LIMIT: usize = 256 * 1024;
/// Largest body of the other review POST routes.
pub(super) const SMALL_BODY_LIMIT: usize = 16 * 1024;
/// The folder of the page's default exports, `<UTC time>` under it unless `out_dir` is given: where the Train tab looks for them
/// ([`crate::training::discover_exports`]), so a custom `training.dir` moves it too.
fn exports_dir(state: &AppState) -> std::path::PathBuf {
    crate::training::training_dir(&state.config_read()).join("exports")
}

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message.into() })),
    )
        .into_response()
}

fn not_found() -> Response {
    fail(StatusCode::NOT_FOUND, "not a frame of the dataset")
}

/// An error answer not built yet (status, message): see [`fail`].
type Problem = (StatusCode, String);

/// Why the folder under review `dir` cannot be used, as the page words it; None when it can. A
/// folder that does not exist, an unmounted drive first of all, would otherwise look like an empty
/// dataset. Only the dataset folder may be absent, when its parent is there: nothing was collected
/// yet (a parent that is gone too is an unplugged drive). A chosen `review_dir` must exist.
pub(super) fn folder_problem(state: &AppState, dir: &std::path::Path) -> Option<String> {
    let e = std::fs::read_dir(dir).err()?;
    let collect = &state.config_read().collect;
    if e.kind() == std::io::ErrorKind::NotFound
        && collect.review_dir.is_none()
        && (collect.dir.is_none() || dir.parent().is_some_and(std::path::Path::is_dir))
    {
        return None;
    }
    let why = match e.kind() {
        std::io::ErrorKind::NotFound => String::new(),
        _ => format!(" ({e})"),
    };
    Some(format!(
        "The folder under review is not available: {}{why}; connect the drive or choose another \
         folder.",
        dir.display()
    ))
}

/// The answer of a route that needs the folder under review while it is unavailable.
fn folder_unavailable(message: String) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({ "success": false, "folder_missing": true, "message": message })),
    )
        .into_response()
}

/// [`not_found`], or [`folder_unavailable`] when that is why `dir` has no such frame.
fn no_frame(state: &AppState, dir: &std::path::Path) -> Response {
    folder_problem(state, dir).map_or_else(not_found, folder_unavailable)
}

/// The JSON body of a POST: 413 over the route's cap, 415 without a JSON content type, 400 when
/// it does not parse.
fn json_body<T: for<'de> Deserialize<'de>>(
    headers: &HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<T, Problem> {
    let body = body.map_err(|e| {
        let s = e.status();
        if s == StatusCode::PAYLOAD_TOO_LARGE {
            (s, "request body too large".to_string())
        } else {
            (s, format!("could not read the request body: {e}"))
        }
    })?;
    let json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"));
    if !json {
        return Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "send the body as JSON (Content-Type: application/json)".into(),
        ));
    }
    serde_json::from_slice(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")))
}

fn query_map(
    q: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> BTreeMap<String, String> {
    q.map(|Query(m)| m).unwrap_or_default()
}

/// Folder under review (`collect.review_dir`, else the dataset folder), classes and data root
/// of the current config.
fn context(state: &AppState) -> (PathBuf, Vec<String>, PathBuf) {
    let cfg = state.config_read();
    (
        cfg.collect.resolved_review_dir(&cfg.data_root()),
        cfg.collect.classes.clone(),
        cfg.data_root(),
    )
}

/// `collect.hidden_classes` of the current config: with `collect.classes`, what the reviewer
/// wants ([`bulk::Wanted`]).
fn hidden(state: &AppState) -> Vec<String> {
    state.config_read().collect.hidden_classes.clone()
}

fn now() -> String {
    crate::collect::utc_rfc3339(std::time::SystemTime::now())
}

#[derive(Template)]
#[template(path = "train/review.html")]
struct ReviewTemplate {
    nav: &'static str,
    tab: &'static str,
    tab_title: &'static str,
    version: &'static str,
    dir: String,
    classes: Vec<String>,
    classes_json: String,
    buckets: Vec<&'static str>,
    scenes: Vec<&'static str>,
    /// The dataset folder (the collector's; the default of the folder under review).
    dataset_dir: String,
    /// `dir` is another folder than the dataset folder.
    custom_dir: bool,
    /// Folders the picker suggests: the dataset folder and where finished imports wrote.
    known_dirs: Vec<String>,
}

/// The dataset folder and every folder a finished import wrote to (newest first), once each.
fn known_dirs(state: &AppState, dataset: &std::path::Path) -> Vec<String> {
    let mut v = vec![dataset.display().to_string()];
    for t in super::training::tasks(state).iter().rev() {
        if let Some(d) = t
            .result
            .as_ref()
            .and_then(|r| r.get("dataset_dir"))
            .and_then(|d| d.as_str())
            && !v.iter().any(|k| k == d)
        {
            v.push(d.to_string());
        }
    }
    v
}

/// [`super::browse::allowed_path`] for a folder of frames, also accepting a folder this server
/// already works on: one a finished import wrote to ([`known_dirs`]) or the folder under review.
/// Both passed that check when chosen, often as a native pick, which it trusts for an hour only;
/// going back to one later must not need another pick. Exact folders only, canonicalized.
pub(super) fn allowed_folder(
    state: &AppState,
    config: &crate::config::Config,
    raw: &str,
) -> Result<PathBuf, (StatusCode, String)> {
    let err = match super::browse::allowed_path(config, raw) {
        Ok(p) => return Ok(p),
        Err(e) => e.into_parts(),
    };
    if err.0 != StatusCode::FORBIDDEN {
        return Err(err);
    }
    let trimmed = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let Ok(canon) = config
        .data_path(std::path::Path::new(trimmed))
        .canonicalize()
    else {
        return Err(err);
    };
    let canon = super::browse::plain(&canon);
    let dataset = CollectSettings::for_config(config).dir;
    let mut trusted: Vec<PathBuf> = known_dirs(state, &dataset)
        .into_iter()
        .map(PathBuf::from)
        .collect();
    trusted.push(config.collect.resolved_review_dir(&config.data_root()));
    let known = trusted
        .iter()
        .filter_map(|k| k.canonicalize().ok())
        .any(|k| super::browse::plain(&k) == canon);
    if known { Ok(canon) } else { Err(err) }
}

/// `GET /train/review`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let dataset = CollectSettings::for_config(&state.config_read()).dir;
        render(&ReviewTemplate {
            nav: "train",
            tab: super::train_area::REVIEW,
            tab_title: super::train_area::tab_title(super::train_area::REVIEW),
            version: crate::VERSION,
            dir: dir.display().to_string(),
            custom_dir: dir != dataset,
            known_dirs: known_dirs(&state, &dataset),
            dataset_dir: dataset.display().to_string(),
            classes_json: serde_json::to_string(&classes).unwrap_or_else(|_| "[]".into()),
            classes,
            buckets: Bucket::ALL.iter().map(|b| b.as_str()).collect(),
            scenes: layout::SCENES.to_vec(),
        })
    })
    .await
}

/// `GET /v1/review/next?bucket=&scene=&camera=&label=&status=&source=&auto=&after=`: the most
/// uncertain frame matching the filters (after `after`, the frame shown now, when given; past
/// the last one the first again with `wrapped`), its position, the counts and the cameras and
/// classes for the filter lists.
pub(super) async fn next(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query_map(query);
    let mut filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    blocking(move || {
        let (dir, classes, _) = context(&state);
        filters.wanted = Some((classes, hidden(&state)));
        let built = queue::build(&dir, &filters);
        let after = q.get("after").map(String::as_str).filter(|a| !a.is_empty());
        let found = queue::next_after(&built.items, after);
        let damaged = layout::damaged_lines(&dir, filters.source == queue::QueueSource::All);
        let mut answer = serde_json::json!({
            "success": true,
            "item": found.map(|(_, i, _)| i),
            "position": found.map(|(p, _, _)| p + 1),
            "wrapped": found.is_some_and(|(_, _, w)| w),
            "total": built.items.len(),
            "counts": built.counts,
            "cameras": built.cameras,
            "labels": built.labels,
            "unreadable": built.unreadable + damaged,
            "upcoming": built.items.iter().skip(found.map_or(0, |(p, _, _)| p + 1)).take(5).map(|i| &i.frame).collect::<Vec<_>>(),
        });
        if let (Some(message), Some(fields)) =
            (folder_problem(&state, &dir), answer.as_object_mut())
        {
            fields.insert("folder_missing".into(), true.into());
            fields.insert("message".into(), message.into());
        }
        Json(answer).into_response()
    })
    .await
}

fn frame_param(q: &BTreeMap<String, String>) -> Result<String, Problem> {
    q.get("frame")
        .filter(|f| !f.is_empty())
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "give `frame` (<scene>/<camera>/<date>/<stem>.jpg)".into(),
            )
        })
}

/// The labels answer for `frame` (saved, else pre-filled), with the label file's revision
/// (`""` = not saved yet) to send back with a save.
fn labels_answer(
    frame: &Frame,
    classes: &[String],
    hidden: &[String],
    message: Option<String>,
) -> Response {
    let rec = match frame.record() {
        Ok(r) => r,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let (saved, revision) = match labels::read_with_revision(&frame.labels) {
        Ok(l) => l,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let is_saved = saved.is_some();
    let file = saved.unwrap_or_else(|| labels::prefill(&rec, classes, hidden));
    let version = labels::classes_version(classes);
    let wanted = bulk::Wanted { classes, hidden };
    let overlap_live: Vec<usize> = labels::overlaps::live(&file, Some(&wanted))
        .iter()
        .map(|l| l.index)
        .collect();
    let q = encode_uri_component(&frame.rel.jpg());
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "frame": frame.rel.jpg(),
        "image_url": format!("/dataset/image?path={q}"),
        "sidecar_url": format!("/dataset/sidecar?path={q}"),
        "width": rec.width,
        "height": rec.height,
        "saved": is_saved,
        "revision": revision,
        "stale_classes": is_saved && !file.classes_version.is_empty() && file.classes_version != version,
        "overlap_live": overlap_live,
        "labels": file,
        "classes": classes,
        "hidden": hidden,
        "classes_version": version,
        "record": {
            "id": rec.id,
            "timestamp": rec.timestamp,
            "scene": rec.scene,
            "camera": rec.camera,
            "model": rec.model,
            "device": rec.device,
            "image_name": rec.image_name,
            "threshold": rec.threshold,
            "buckets": rec.buckets,
            "routed_from": rec.routed_from,
            "source": rec.source,
            "provenance": rec.provenance,
        },
    }))
    .into_response()
}

/// `GET /v1/review/labels?frame=<scene>/<camera>/<date>/<stem>.jpg`.
pub(super) async fn labels_get(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Response {
    let rel = match frame_param(&query_map(query)) {
        Ok(r) => r,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        match Frame::resolve(&dir, &rel) {
            Some(f) => labels_answer(&f, &classes, &hidden, None),
            None => no_frame(&state, &dir),
        }
    })
    .await
}

/// Message of a save refused because the frame changed since the client read it.
pub(super) const CONFLICT_MESSAGE: &str = "This frame changed since you opened it (an auto-label \
     run or another tab saved it). Reload it to see the new boxes; your changes to it were not \
     saved.";

/// `POST /v1/review/labels?frame=...[&revision=<rev>]` with a label file as JSON: checked
/// ([`labels::normalize`]) and saved atomically; answers like the GET. With `revision` (the one
/// the GET answered; empty = not saved then) the save is refused with 409 and `conflict: true`
/// when the file changed meanwhile; without it, it always overwrites (X-AnyLabeling and other
/// clients).
pub(super) async fn labels_post(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let q = query_map(query);
    let rel = match frame_param(&q) {
        Ok(r) => r,
        Err((s, m)) => return fail(s, m),
    };
    let expected = q.get("revision").map(|r| r.trim().to_string());
    let file: LabelFile = match json_body(&headers, body) {
        Ok(f) => f,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let Some(frame) = Frame::resolve(&dir, &rel) else {
            return no_frame(&state, &dir);
        };
        let rec = match frame.record() {
            Ok(r) => r,
            Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        };
        let status = file.status;
        let f = match labels::normalize(file, &classes, rec.width, rec.height, &now()) {
            Ok(f) => f,
            Err(e) => return fail(StatusCode::UNPROCESSABLE_ENTITY, format!("Not saved: {e:#}")),
        };
        let written = match &expected {
            Some(rev) => labels::write_checked(&frame.labels, &f, rev),
            None => labels::write(&frame.labels, &f).map(Ok),
        };
        match written {
            Ok(Ok(())) => {}
            Ok(Err(current)) => {
                tracing::debug!(frame = %rel, current = %current, "review save refused: stale revision");
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "success": false,
                        "conflict": true,
                        "message": CONFLICT_MESSAGE,
                    })),
                )
                    .into_response();
            }
            Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("Not saved: {e:#}")),
        }
        tracing::debug!(frame = %rel, status = status.as_str(), boxes = f.boxes.len(), "review labels saved");
        labels_answer(
            &frame,
            &classes,
            &hidden,
            Some(format!("Saved as {}.", status.as_str())),
        )
    })
    .await
}

/// Default most frames of one bulk accept.
pub(super) const ACCEPT_AUTO_DEFAULT_LIMIT: usize = 500;
/// Most frames of one bulk accept.
pub(super) const ACCEPT_AUTO_MAX_LIMIT: usize = 20_000;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptAutoBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/accept-auto` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}` (empty = any): the unreviewed frames of those filters whose auto-label triage alone
/// settles them (`auto=accept`, [`labels::auto_accept_eligible`]) saved as reviewed, at most
/// `limit` (default 500; [`bulk::bulk_accept`]). `dry_run` only counts them.
pub(super) async fn accept_auto(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: AcceptAutoBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let mut q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    q.insert("status".into(), "unreviewed".into());
    q.insert("auto".into(), "accept".into());
    let mut filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
        filters.wanted = Some((classes.clone(), hidden(&state)));
        let o = bulk::bulk_accept(&dir, &filters, &classes, limit, &now(), dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return Json(serde_json::json!({
                "success": true,
                "eligible": eligible,
                "accepted": 0,
                "frames": o.done,
            }))
            .into_response();
        }
        let mut message = format!(
            "Accepted {} frame(s) the auto-labelers settled (saved as reviewed).",
            o.done_count
        );
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} not accepted (listed); review them by hand.",
                o.failed_count
            ));
        }
        if eligible > limit {
            message.push_str(&format!(
                " {} more eligible beyond the limit of {limit}; run it again.",
                eligible - limit
            ));
        }
        tracing::info!(
            accepted = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: auto-labels bulk accepted"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": eligible,
            "accepted": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
        .into_response()
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectLowBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    min_score: f32,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-low` with `{bucket?, scene?, camera?, label?, source?, min_score,
/// dry_run, limit?}`: on the unreviewed frames of those filters, the pending auto boxes scoring
/// below `min_score` (0.01..=1) rejected with `reason: low_score` (kept as hard-negative
/// evidence; [`bulk::bulk_reject_below`]), at most `limit` frames (default 500). `dry_run`
/// only counts the frames and boxes.
pub(super) async fn reject_low(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: RejectLowBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    if !(0.01..=1.0).contains(&b.min_score) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("min_score: must be between 0.01 and 1, got {}", b.min_score),
        );
    }
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let mut q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    q.insert("status".into(), "unreviewed".into());
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let (dry_run, min) = (b.dry_run, b.min_score);
    blocking(move || {
        let (dir, _, _) = context(&state);
        let o = bulk::bulk_reject_below(&dir, &filters, min, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return Json(serde_json::json!({
                "success": true,
                "eligible": eligible,
                "boxes": o.boxes,
                "rejected": 0,
                "frames": o.done,
            }))
            .into_response();
        }
        let mut message = format!(
            "Rejected {} auto box(es) scoring below {min} on {} frame(s) (kept as hard \
             negatives; the frames stay unreviewed).",
            o.boxes, o.done_count
        );
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); review them by hand.",
                o.failed_count
            ));
        }
        if eligible > limit {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                eligible - limit
            ));
        }
        tracing::info!(
            rejected = o.boxes,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            min_score = min,
            "review: low-scoring auto boxes rejected"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
        .into_response()
    })
    .await
}

/// `GET /v1/review/history?camera=<c>`: the reviewer's decisions on that camera's frames in the
/// folder under review (accepted / rejected model and auto boxes, not by a bulk rule; newest
/// first, at most [`labels::repeats::HISTORY_CAP`], `truncated` past it), for the Review tab's
/// "rejected here before" hint. Cached until a label file of the folder is written.
pub(super) async fn history(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query_map(query);
    let Some(camera) = q
        .get("camera")
        .map(|c| c.trim().to_string())
        .filter(|c| layout::camera_ok(c))
    else {
        return fail(
            StatusCode::BAD_REQUEST,
            "give `camera` (a camera name of the dataset, as in the frame paths)",
        );
    };
    blocking(move || {
        let (dir, _, _) = context(&state);
        let h = labels::repeats::history(&dir, &camera);
        Json(serde_json::json!({
            "success": true,
            "camera": h.camera,
            "decisions": h.decisions,
            "truncated": h.truncated,
        }))
        .into_response()
    })
    .await
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
pub(super) async fn reject_spot(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    spot_route(state, headers, body, false).await
}

/// `POST /v1/review/accept-spot`: [`reject_spot`]'s mirror for a real object that stays put (a
/// parked car): the pending boxes of the same label at IoU >= `min_iou` (default and minimum
/// 0.8) are accepted. `reject_reason` is refused.
pub(super) async fn accept_spot(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    spot_route(state, headers, body, true).await
}

async fn spot_route(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
    accept: bool,
) -> Response {
    use labels::repeats::{ACCEPT_SPOT_IOU, REJECT_SPOT_IOU, Spot, Verdict};
    let b: SpotBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let camera = b.camera.trim().to_string();
    if !layout::camera_ok(&camera) {
        return fail(
            StatusCode::BAD_REQUEST,
            "camera: not a camera name of the dataset",
        );
    }
    if crate::collect::config::validate_classes(std::slice::from_ref(&b.label)).is_err() {
        return fail(StatusCode::BAD_REQUEST, "label: not a valid class name");
    }
    let [x0, y0, x1, y1] = b.bbox;
    if !b.bbox.iter().all(|v| v.is_finite())
        || x1 <= x0
        || y1 <= y0
        || b.width == 0
        || b.height == 0
    {
        return fail(
            StatusCode::BAD_REQUEST,
            "bbox: give [x0, y0, x1, y1] in pixels of the width x height frame, x1 > x0, y1 > y0",
        );
    }
    let (lo, default) = if accept {
        (ACCEPT_SPOT_IOU, ACCEPT_SPOT_IOU)
    } else {
        (0.3, REJECT_SPOT_IOU)
    };
    let min_iou = b.min_iou.unwrap_or(default);
    if !(lo..=1.0).contains(&min_iou) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("min_iou: must be between {lo} and 1, got {min_iou}"),
        );
    }
    if accept && b.reject_reason.is_some() {
        return fail(
            StatusCode::BAD_REQUEST,
            "reject_reason: only for reject-spot",
        );
    }
    if let Some(labels::RejectReason::Other(r)) = &b.reject_reason {
        return fail(
            StatusCode::BAD_REQUEST,
            format!(
                "reject_reason: '{}' is not a reason this build knows",
                r.escape_debug()
            ),
        );
    }
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let mut q = BTreeMap::new();
    if let Some(s) = b.source {
        q.insert("source".to_string(), s);
    }
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
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
    blocking(move || {
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
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); review them by hand.",
                o.failed_count
            ));
        }
        if o.eligible > limit && !dry_run {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                o.eligible - limit
            ));
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
        Json(serde_json::json!({
            "success": true,
            "eligible": o.eligible,
            "boxes": o.boxes,
            "changed": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": (!dry_run).then_some(message),
        }))
        .into_response()
    })
    .await
}

/// `POST /v1/review/accept-repeats` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: the repeat-only frames of those filters (spot edits settled every box, none
/// pending; [`labels::repeats::repeats_only`]) saved as reviewed with `bulk_accepted`
/// ([`bulk::bulk_accept_repeats`]), at most `limit` (default 500). `dry_run` only counts.
pub(super) async fn accept_repeats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: AcceptAutoBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let o =
            bulk::bulk_accept_repeats(&dir, &filters, &classes, limit, &now(), dry_run, &|_| true);
        if dry_run {
            return Json(serde_json::json!({
                "success": true,
                "eligible": o.eligible,
                "accepted": 0,
                "frames": o.done,
            }))
            .into_response();
        }
        let mut message = format!("Saved {} repeat-only frame(s) as reviewed.", o.done_count);
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} not saved (listed); review them by hand.",
                o.failed_count
            ));
        }
        if o.eligible > limit {
            message.push_str(&format!(
                " {} more beyond the limit of {limit}; run it again.",
                o.eligible - limit
            ));
        }
        tracing::info!(
            saved = o.done_count,
            failed = o.failed_count,
            "review: repeat-only frames saved"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": o.eligible,
            "accepted": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
        .into_response()
    })
    .await
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
pub(super) async fn spots(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Response {
    let camera = match spot_camera(query_map(query).get("camera")) {
        Ok(c) => c,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
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
        Json(serde_json::json!({
            "success": true,
            "camera": camera,
            "moved_at": store.moved_at,
            "parked": store.parked,
            "rejected": rejected,
            "live": store.live_count() + rejected.len(),
        }))
        .into_response()
    })
    .await
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
pub(super) async fn spots_parked(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: SpotParkedBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
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
                Json(serde_json::json!({ "success": true, "spot": spot, "new": new }))
                    .into_response()
            }
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await
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
pub(super) async fn spots_retire(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: SpotRetireBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let camera = match spot_camera(Some(&b.camera)) {
        Ok(c) => c,
        Err((s, m)) => return fail(s, m),
    };
    let bbox = match spot_box(&b.label, b.bbox, b.width, b.height) {
        Ok(c) => c,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
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
            Ok(true) => Json(serde_json::json!({ "success": true })).into_response(),
            Ok(false) => fail(StatusCode::NOT_FOUND, "no parked spot there"),
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotMovedBody {
    camera: String,
}

/// `POST /v1/review/spots/moved` with `{camera}` ("Camera moved: forget spots"): every parked spot
/// retires and `moved_at` is set, so rejected spots from before it stop counting too
/// ([`labels::spots::camera_moved`]). A run already going keeps the spots it read at its start.
pub(super) async fn spots_moved(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: SpotMovedBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let camera = match spot_camera(Some(&b.camera)) {
        Ok(c) => c,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let (dir, _, _) = context(&state);
        let at = now();
        match labels::spots::camera_moved(&dir, &camera, &at) {
            Ok(retired) => {
                tracing::info!(camera, retired, "review: camera moved, spots forgotten");
                Json(serde_json::json!({ "success": true, "retired": retired, "moved_at": at }))
                    .into_response()
            }
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await
}

/// `POST /v1/review/reject-stacked` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: on the reviewed and unreviewed frames with a label file that the bucket, scene, camera
/// and class filters select, the boxes stacked on a better box of the object (IoU >= 0.7) are
/// rejected (`reject_reason: duplicate`, kept as evidence; [`bulk::bulk_reject_stacked`]), at most
/// `limit` frames (default 500). The frames keep their status; drawn boxes and frames a reviewer
/// checked (`overlaps_checked`) are left alone. `source` is checked but does not narrow the walk: a
/// reviewed frame need not be in the review queue. `dry_run` only counts. `boxes` counts the boxes
/// rejected, `rejected` the frames changed (as reject-low), `overlaps_left` the frames that still
/// need a reviewer once the duplicates are gone (nested or overlapping boxes no rule can settle:
/// `auto=overlaps`), in a dry run too.
pub(super) async fn reject_stacked(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: RejectOtherClassBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let done = bulk::bulk_reject_stacked(&dir, &filters, &wanted, limit, dry_run, &|_| true);
        let (o, left, left_reviewed) = (
            done.outcome,
            done.overlaps_left,
            done.overlaps_left_reviewed,
        );
        let mut message = if o.boxes == 0 {
            "No stacked duplicates on the frames these filters select.".to_string()
        } else if dry_run {
            format!(
                "Would reject {} stacked duplicate box(es) on {} frame(s); the frames keep their \
                 status.",
                o.boxes, o.eligible
            )
        } else {
            format!(
                "Rejected {} stacked duplicate box(es) on {} frame(s) (kept as evidence, reason \
                 duplicate; reviewed frames stay reviewed).",
                o.boxes, o.done_count
            )
        };
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); review them by hand.",
                o.failed_count
            ));
        }
        if !dry_run && o.eligible > limit {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                o.eligible - limit
            ));
        }
        if left > 0 {
            message.push_str(&format!(
                " {left} frame(s) still have overlapping boxes no rule can settle, \
                 {left_reviewed} of them reviewed (those are the ones training uses): set Auto \
                 to overlapping boxes, Status to reviewed and Frames to all collected, then keep \
                 or reject them by hand."
            ));
        }
        if !dry_run {
            tracing::info!(
                boxes = o.boxes,
                frames = o.done_count,
                failed = o.failed_count,
                eligible = o.eligible,
                overlaps_left = left,
                "review: stacked duplicates rejected"
            );
        }
        Json(serde_json::json!({
            "success": true,
            "eligible": o.eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
            "overlaps_left": left,
            "overlaps_left_reviewed": left_reviewed,
        }))
        .into_response()
    })
    .await
}

/// `POST /v1/review/restore-reopened` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: the undo of a re-label that reopened reviewed frames. On the unreviewed frames of those
/// filters, a frame with a `disagreement` note, a `reviewed_at` and nothing pending goes back to
/// reviewed and loses the note ([`bulk::bulk_restore_reopened`], re-checked under the write lock),
/// at most `limit` frames (default 500). Frames with a pending box and frames never reviewed are
/// not touched. `dry_run` only counts. `eligible` counts the frames, `restored` those changed.
pub(super) async fn restore_reopened(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: RejectOtherClassBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let mut q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    q.insert("status".into(), "unreviewed".into());
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, _, _) = context(&state);
        let o = bulk::bulk_restore_reopened(&dir, &filters, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return Json(serde_json::json!({
                "success": true,
                "eligible": eligible,
                "restored": 0,
                "frames": o.done,
            }))
            .into_response();
        }
        let mut message = format!(
            "Restored {} reopened frame(s) to reviewed (their boxes are unchanged; they are back \
             in exports).",
            o.done_count
        );
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); review them by hand.",
                o.failed_count
            ));
        }
        if eligible > limit {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                eligible - limit
            ));
        }
        tracing::info!(
            restored = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: reopened frames restored"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": eligible,
            "restored": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
        .into_response()
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SuggestOverlapsBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    gpu: Option<String>,
    #[serde(default)]
    allow_large: bool,
    /// Also pairs with an answer.
    #[serde(default)]
    again: bool,
    #[serde(default)]
    dry_run: bool,
}

/// `POST /v1/review/suggest-overlaps` with `{bucket?, scene?, camera?, label?, gpu?, allow_large?,
/// again?, dry_run?}`: queue the overlap check ([`crate::training::overlaps`]) on the label files
/// of the folder under review those filters select, reviewed frames too (no `source` or `status`
/// narrows it, a reviewed frame need not be in the review queue). It writes suggestions and nothing
/// else. `dry_run` only counts (`frames`, `pairs`, `crops`, `no_zoom`, `answered`, `passed_over`), and
/// needs no GPU. Otherwise `{success, message, id}`, the task id in `GET /v1/training` `tasks[]`.
pub(super) async fn suggest_overlaps(
    State(shared): State<Shared>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    use crate::training::overlaps::{OverlapsRequest, plan_overlaps, select};
    let b: SuggestOverlapsBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let state = shared.current();
        let mut req = match OverlapsRequest::parse(if b.again { "again=1" } else { "" }) {
            Ok(r) => r,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        };
        for (k, v) in [
            ("bucket", b.bucket),
            ("scene", b.scene),
            ("camera", b.camera),
            ("label", b.label),
        ] {
            if let Some(v) = v.filter(|v| !v.trim().is_empty()) {
                req.frames.query.insert(k.to_string(), v);
            }
        }
        req.gpu = b.gpu;
        req.allow_large = b.allow_large;
        if let Err(e) = req.frames.filters() {
            return fail(StatusCode::BAD_REQUEST, format!("{e:#}"));
        }
        if b.dry_run {
            let (dir, classes, _) = context(&state);
            let hidden = hidden(&state);
            let wanted = bulk::Wanted {
                classes: &classes,
                hidden: &hidden,
            };
            let s = match select::select(&dir, &req.frames, &wanted) {
                Ok(s) => s,
                Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
            };
            let message = (s.pairs == 0).then(|| {
                let again = if s.answered > 0 {
                    "; tick Ask again to check the answered ones again"
                } else {
                    ""
                };
                format!(
                    "No open overlap without an answer on these frames ({} pairs answered before, \
                     {} frames skipped or checked by a reviewer){again}.",
                    s.answered, s.passed_over
                )
            });
            return Json(serde_json::json!({
                "success": true,
                "frames": s.frame_count,
                "pairs": s.pairs,
                "no_zoom": s.no_zoom,
                "crops": s.crops,
                "answered": s.answered,
                "passed_over": s.passed_over,
                "message": message,
            }))
            .into_response();
        }
        let config = state.config_read().clone();
        let cap = crate::training::capability_now();
        let job = match plan_overlaps(&config, req, &cap) {
            Ok(j) => j,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("Not queued: {e:#}")),
        };
        let title = crate::training::task::Task::title(&job);
        match super::training::service(&shared).enqueue_task(Box::new(job)) {
            Ok(id) => Json(serde_json::json!({
                "success": true,
                "id": id,
                "message": format!(
                    "Queued: {title}. It writes suggestions only: no box changes until you \
                     confirm one here."
                ),
            }))
            .into_response(),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not queued: {e}")),
        }
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectOtherClassBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-other-class` with `{bucket?, scene?, camera?, label?, source?,
/// dry_run, limit?}`: on the unreviewed frames of those filters, the pending model and auto boxes
/// of unwanted classes (not in `collect.classes`, or in `collect.hidden_classes`) rejected with
/// `reason: other_class` (kept as evidence), or renamed to the list's spelling when the label is
/// a wanted class under another name (`pottedplant` -> `potted plant`;
/// [`bulk::bulk_reject_other_class`]), at most `limit` frames (default 500). `dry_run` only
/// counts. `boxes` counts the boxes changed, `relabelled` the renamed ones among them,
/// `rejected` the frames changed (as reject-low); `reviewed_with_unwanted` the reviewed frames
/// that still hold training boxes of unwanted classes
/// ([`bulk::reviewed_with_unwanted`]; `POST /v1/review/reject-unwanted-reviewed` rejects them).
pub(super) async fn reject_other_class(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: RejectOtherClassBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_DEFAULT_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let mut q: BTreeMap<String, String> = [
        ("bucket", b.bucket),
        ("scene", b.scene),
        ("camera", b.camera),
        ("label", b.label),
        ("source", b.source),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    q.insert("status".into(), "unreviewed".into());
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let o = bulk::bulk_reject_other_class(&dir, &filters, &wanted, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        let reviewed = bulk::reviewed_with_unwanted(&dir, &wanted);
        if dry_run {
            return Json(serde_json::json!({
                "success": true,
                "eligible": eligible,
                "boxes": o.boxes,
                "relabelled": o.relabelled,
                "rejected": 0,
                "frames": o.done,
                "reviewed_with_unwanted": reviewed,
            }))
            .into_response();
        }
        let mut message = format!(
            "Rejected {} pending box(es) of unwanted classes on {} frame(s) (kept as evidence, \
             reason other_class; the frames stay unreviewed).",
            o.boxes - o.relabelled,
            o.done_count
        );
        if o.relabelled > 0 {
            message.push_str(&format!(
                " Renamed {} box(es) to the class list's spelling.",
                o.relabelled
            ));
        }
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); review them by hand.",
                o.failed_count
            ));
        }
        if eligible > limit {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                eligible - limit
            ));
        }
        tracing::info!(
            boxes = o.boxes,
            relabelled = o.relabelled,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: pending boxes of unwanted classes rejected"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": eligible,
            "boxes": o.boxes,
            "relabelled": o.relabelled,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
            "reviewed_with_unwanted": reviewed,
        }))
        .into_response()
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectUnwantedReviewedBody {
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-unwanted-reviewed` with `{dry_run, limit?}`: on the reviewed frames
/// of the folder under review, the training boxes (accepted or added) of unwanted classes (not in
/// `collect.classes`, or in `collect.hidden_classes`) rejected with `reason: other_class`; the
/// frames stay reviewed ([`bulk::bulk_reject_unwanted`] with [`bulk::Scope::Reviewed`]), at most
/// `limit` frames (default and most [`ACCEPT_AUTO_MAX_LIMIT`]). `dry_run` only counts: `eligible`
/// frames, `boxes`, `classes` (their labels). The Review tab asks before it calls this.
pub(super) async fn reject_unwanted_reviewed(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: RejectUnwantedReviewedBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let limit = b.limit.unwrap_or(ACCEPT_AUTO_MAX_LIMIT);
    if !(1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        );
    }
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        if dry_run {
            let r = bulk::reviewed_with_unwanted(&dir, &wanted);
            return Json(serde_json::json!({
                "success": true,
                "eligible": r.frames,
                "boxes": r.boxes,
                "classes": r.classes,
                "rejected": 0,
            }))
            .into_response();
        }
        let o = bulk::bulk_reject_unwanted(&dir, &wanted, bulk::Scope::Reviewed, limit, false);
        let eligible = o.eligible;
        let mut message = format!(
            "Rejected {} accepted box(es) of unwanted classes on {} reviewed frame(s) (kept as \
             evidence, reason other_class; the frames stay reviewed).",
            o.boxes, o.done_count
        );
        if o.failed_count > 0 {
            message.push_str(&format!(
                " {} frame(s) not changed (listed); fix them by hand.",
                o.failed_count
            ));
        }
        if eligible > limit {
            message.push_str(&format!(
                " {} more frame(s) beyond the limit of {limit}; run it again.",
                eligible - limit
            ));
        }
        tracing::info!(
            boxes = o.boxes,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: accepted boxes of unwanted classes rejected on reviewed frames"
        );
        Json(serde_json::json!({
            "success": true,
            "eligible": eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
        .into_response()
    })
    .await
}

/// One format or a list (`"yolo"`, `["yolo", "coco"]`, `"all"`).
#[derive(Deserialize)]
#[serde(untagged)]
enum Formats {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportBody {
    #[serde(alias = "format")]
    formats: Option<Formats>,
    out_dir: Option<String>,
    #[serde(default)]
    holdout_cameras: Vec<String>,
    #[serde(default)]
    holdout_days: Vec<String>,
    holdout_fraction: Option<f32>,
    #[serde(default)]
    benchmark_split: BenchSplit,
    #[serde(default)]
    include_unreviewed: bool,
    /// Only frames with a training box of these classes; empty = every class.
    #[serde(default)]
    labels: Vec<String>,
    /// Also frames confirmed empty.
    #[serde(default = "default_true")]
    include_empty: bool,
}

fn default_true() -> bool {
    true
}

/// The export's class filter: names trimmed, blanks and repeats dropped, every one in
/// `classes` (else the message for a 400).
fn class_filter(
    labels: Vec<String>,
    include_empty: bool,
    classes: &[String],
) -> Result<ClassFilter, String> {
    let mut out: Vec<String> = Vec::new();
    for l in labels.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        if !out.iter().any(|o| o == l) {
            out.push(l.to_string());
        }
    }
    let f = ClassFilter {
        labels: out,
        include_empty,
    };
    f.check(classes)
        .map_err(|e| format!("Not exported: {e:#}"))?;
    Ok(f)
}

fn parse_formats(f: Option<Formats>) -> Result<Vec<Format>, String> {
    let list = match f {
        None => return Ok(Format::ALL.to_vec()),
        Some(Formats::One(s)) => vec![s],
        Some(Formats::Many(v)) => v,
    };
    let mut out = Vec::new();
    for s in list {
        if s.trim().eq_ignore_ascii_case("all") {
            return Ok(Format::ALL.to_vec());
        }
        let f = Format::parse(&s).ok_or_else(|| {
            format!("format: '{s}' is not one of yolo, coco, xanylabeling, benchmark, all")
        })?;
        if !out.contains(&f) {
            out.push(f);
        }
    }
    if out.is_empty() {
        return Err("format: give at least one of yolo, coco, xanylabeling, benchmark".into());
    }
    Ok(out)
}

/// A folder a client named for this server to read or write: well formed and no system folder
/// ([`crate::config::validate_dir_setting`], 400), then [`allowed_folder`]. `Ok(None)` when
/// `allowed_folder` found nothing there.
fn named_folder(
    state: &AppState,
    key: &str,
    raw: &str,
    root: &std::path::Path,
) -> Result<Option<PathBuf>, (StatusCode, String)> {
    match crate::config::validate_dir_setting(key, raw, root) {
        Ok(Some(_)) => {}
        Ok(None) => return Err((StatusCode::BAD_REQUEST, format!("{key}: empty"))),
        Err(e) => return Err((StatusCode::BAD_REQUEST, format!("{e:#}"))),
    }
    let config = state.config_read().clone();
    match allowed_folder(state, &config, raw) {
        Ok(p) => Ok(Some(p)),
        Err((StatusCode::NOT_FOUND, _)) => Ok(None),
        Err(e) => Err(folder_refusal(key, e)),
    }
}

/// `refusal` of a folder under the setting's name, with what to do about it.
fn folder_refusal(key: &str, (status, message): (StatusCode, String)) -> (StatusCode, String) {
    let message = format!("{key}: {message}; choose the folder with the Browse button instead");
    (status, message)
}

/// Where a client-named export goes ([`named_folder`]). The export creates it, so a folder that
/// does not exist yet is fine inside the browsable folders or a native pick.
fn export_folder(
    state: &AppState,
    raw: &str,
    root: &std::path::Path,
) -> Result<PathBuf, (StatusCode, String)> {
    if let Some(p) = named_folder(state, "out_dir", raw, root)? {
        return Ok(p);
    }
    let config = state.config_read().clone();
    super::browse::allowed_or_missing_inside_roots(&config, raw)
        .map_err(|e| folder_refusal("out_dir", e.into_parts()))
}

/// `POST /v1/review/export` with `{format(s), out_dir?, holdout_cameras?, holdout_days?,
/// holdout_fraction?, benchmark_split?, include_unreviewed?, labels?, include_empty?}`.
pub(super) async fn export(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: ExportBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    let formats = match parse_formats(b.formats) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let fraction = b
        .holdout_fraction
        .unwrap_or(export::DEFAULT_HOLDOUT_FRACTION);
    if !(fraction.is_finite() && (0.0..=1.0).contains(&fraction)) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("holdout_fraction: must be between 0 and 1, got {fraction}"),
        );
    }
    blocking(move || {
        let (dir, classes, root) = context(&state);
        if let Some(message) = folder_problem(&state, &dir) {
            return folder_unavailable(message);
        }
        let filter = match class_filter(b.labels, b.include_empty, &classes) {
            Ok(f) => f,
            Err(e) => return fail(StatusCode::BAD_REQUEST, e),
        };
        let out_dir = match b.out_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            None => {
                exports_dir(&state).join(crate::collect::utc_file_ts(std::time::SystemTime::now()))
            }
            Some(raw) => match export_folder(&state, raw, &root) {
                Ok(p) => p,
                Err((s, m)) => return fail(s, m),
            },
        };
        let req = ExportRequest {
            formats,
            out_dir,
            split: SplitConfig {
                holdout_cameras: b.holdout_cameras,
                holdout_days: b.holdout_days,
                holdout_fraction: fraction,
            },
            benchmark_split: b.benchmark_split,
            include_unreviewed: b.include_unreviewed,
            hidden_classes: hidden(&state),
        };
        match export::run_filtered(&dir, &classes, &req, &filter) {
            Ok(r) => {
                tracing::info!(out = %r.out_dir.display(), frames = r.frames, "review export written");
                Json(serde_json::json!({ "success": true, "report": r })).into_response()
            }
            Err(e) => fail(StatusCode::UNPROCESSABLE_ENTITY, format!("Not exported: {e:#}")),
        }
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportBody {
    dir: String,
    #[serde(default)]
    treat_unchecked_as_reviewed: bool,
}

/// `POST /v1/review/import` with `{dir, treat_unchecked_as_reviewed?}`: X-AnyLabeling files of
/// exported frames back into their label files.
pub(super) async fn import(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: ImportBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let (dir, classes, root) = context(&state);
        if let Some(message) = folder_problem(&state, &dir) {
            return folder_unavailable(message);
        }
        let src = match named_folder(&state, "dir", &b.dir, &root) {
            Ok(Some(p)) if p.is_dir() => p,
            Ok(Some(_)) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("dir: '{}' is not a folder", b.dir),
                );
            }
            Ok(None) => {
                return fail(
                    StatusCode::NOT_FOUND,
                    format!("dir: '{}' does not exist; choose the folder again", b.dir),
                );
            }
            Err((s, m)) => return fail(s, m),
        };
        let hidden = hidden(&state);
        match xany::import_dir(
            &dir,
            &src,
            &classes,
            &hidden,
            b.treat_unchecked_as_reviewed,
            &now(),
        ) {
            Ok(r) => Json(serde_json::json!({ "success": true, "report": r })).into_response(),
            Err(e) => fail(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Not imported: {e:#}"),
            ),
        }
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FolderBody {
    dir: String,
}

/// `POST /v1/review/folder` with `{dir}`: the folder this page, its exports and the auto-labeler
/// work on (`collect.review_dir`), for frames an import saved outside the dataset folder. Empty or
/// the dataset folder itself = back to the dataset folder. Like the import's paths, it must be a
/// browsable folder, one picked in the native dialog or one already in use ([`allowed_folder`]).
pub(super) async fn folder(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: FolderBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let config = state.config_read().clone();
        let dataset = CollectSettings::for_config(&config).dir;
        let raw = b.dir.trim();
        let chosen = if raw.is_empty() {
            None
        } else {
            let p = match allowed_folder(&state, &config, raw) {
                Ok(p) => p,
                Err((s, m)) => return fail(s, format!("Folder not changed: {m}")),
            };
            if !p.is_dir() {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("Folder not changed: {raw} is not a folder"),
                );
            }
            let same = dataset
                .canonicalize()
                .is_ok_and(|d| super::browse::plain(&d) == p);
            (!same).then_some(p)
        };
        let result = save_config(&state, "review page", |c| {
            crate::collect::config::set_review_dir(c, chosen.clone())
        });
        match result {
            Ok(msg) => Json(serde_json::json!({
                "success": true,
                "message": msg,
                "dir": chosen.as_ref().unwrap_or(&dataset),
            }))
            .into_response(),
            Err(e) => fail(
                StatusCode::BAD_REQUEST,
                format!("Folder not changed: {e:#}"),
            ),
        }
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassesBody {
    classes: Vec<String>,
    /// `collect.hidden_classes`; absent = unchanged (pruned to the names still listed).
    #[serde(default)]
    hidden: Option<Vec<String>>,
}

/// The classes `old` wanted (its list without its hidden ones) that `new` does not: removed
/// from the list or newly hidden, in the old list's order.
fn newly_unwanted(old: &bulk::Wanted<'_>, new: &bulk::Wanted<'_>) -> Vec<String> {
    old.classes
        .iter()
        .filter(|c| old.class_of(c).is_some() && new.class_of(c).is_none())
        .cloned()
        .collect()
}

/// `POST /v1/review/classes` with `{classes: [...], hidden?: [...]}`: saves `collect.classes`
/// and, when given, `collect.hidden_classes` (each must be listed; "don't want": hidden from
/// Review and their pending boxes rejected); hidden names no longer listed are dropped. Answers
/// both. When the save makes classes unwanted (removed or newly hidden: `newly_unwanted`), the
/// pending model and auto boxes of unwanted classes on the unreviewed frames of the folder under
/// review are rejected right away (`auto_rejected: {frames, boxes, relabelled, failed}`,
/// [`bulk::bulk_reject_unwanted`]) and `reviewed_with_unwanted: {frames, boxes, classes}` counts
/// the reviewed frames still holding accepted boxes of unwanted classes (the page asks before
/// `POST /v1/review/reject-unwanted-reviewed`); both zero otherwise.
pub(super) async fn classes(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: ClassesBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((s, m)) => return fail(s, m),
    };
    blocking(move || {
        let classes: Vec<String> = b
            .classes
            .into_iter()
            .map(|c| c.trim().to_string())
            .collect();
        let hidden_req: Option<Vec<String>> = b
            .hidden
            .map(|h| h.into_iter().map(|c| c.trim().to_string()).collect());
        let mut hidden = Vec::new();
        let (mut old_classes, mut old_hidden) = (Vec::new(), Vec::new());
        let mut dir = PathBuf::new();
        let result = save_config(&state, "review page", |c| {
            old_classes = c.collect.classes.clone();
            old_hidden = c.collect.hidden_classes.clone();
            c.collect.classes = classes.clone();
            match &hidden_req {
                Some(h) => c.collect.hidden_classes = h.clone(),
                None => c.collect.prune_hidden(),
            }
            c.collect.validate(&c.data_root())?;
            hidden = c.collect.hidden_classes.clone();
            dir = c.collect.resolved_review_dir(&c.data_root());
            Ok(())
        });
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        };
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let old = bulk::Wanted {
            classes: &old_classes,
            hidden: &old_hidden,
        };
        let gone = newly_unwanted(&old, &wanted);
        let (rejected, reviewed) = if gone.is_empty() {
            (
                bulk::BulkOutcome::default(),
                bulk::ReviewedUnwanted::default(),
            )
        } else {
            let o = bulk::bulk_reject_unwanted(
                &dir,
                &wanted,
                bulk::Scope::Unreviewed,
                usize::MAX,
                false,
            );
            tracing::info!(
                classes = %gone.join(", "),
                boxes = o.boxes,
                frames = o.done_count,
                failed = o.failed_count,
                "review: classes no longer wanted; their pending boxes rejected"
            );
            (o, bulk::reviewed_with_unwanted(&dir, &wanted))
        };
        Json(serde_json::json!({
            "success": true,
            "message": msg,
            "classes": classes,
            "hidden": hidden,
            "classes_version": labels::classes_version(&classes),
            "newly_unwanted": gone,
            "auto_rejected": {
                "frames": rejected.done_count,
                "boxes": rejected.boxes,
                "relabelled": rejected.relabelled,
                "failed": rejected.failed,
            },
            "reviewed_with_unwanted": reviewed,
        }))
        .into_response()
    })
    .await
}

/// `GET /v1/review/class-catalog`: the class list, the hidden classes, the classes the enabled
/// models report, the models' classes and every class worth offering
/// ([`crate::training::autolabel::catalog::ClassCatalog`]; reads the models' class files).
pub(super) async fn class_catalog(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let config = state.config_read().clone();
        let cat = crate::training::autolabel::catalog::ClassCatalog::for_config(&config);
        let mut v = match serde_json::to_value(&cat) {
            Ok(v) => v,
            Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        };
        if let Some(o) = v.as_object_mut() {
            o.insert("success".into(), true.into());
        }
        Json(v).into_response()
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, get_text, test_state_with};
    use super::*;
    use crate::collect::record::{Det, Suppressed, empty_record};
    use axum::body::Body;
    use axum::http::Request;

    struct Ds {
        state: Arc<AppState>,
        dir: PathBuf,
        root: PathBuf,
    }

    impl Drop for Ds {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn ds() -> Ds {
        ds_with(|_| {})
    }

    /// [`ds`] with `edit` applied to the config.
    fn ds_with(edit: impl FnOnce(&mut crate::config::Config)) -> Ds {
        let root = std::env::temp_dir().join(format!("bop-review-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let (d, r) = (root.join("dataset"), root.clone());
        let state = test_state_with(move |c| {
            c.download_dir = Some(r);
            c.collect.dir = Some(d);
            edit(c);
        });
        Ds {
            state,
            dir: root.join("dataset"),
            root,
        }
    }

    /// A frame with a real 64x48 JPEG, a sidecar with one prediction and one suppressed box,
    /// queued for review.
    fn put_frame(
        dir: &std::path::Path,
        camera: &str,
        date: &str,
        stem: &str,
        margin_score: f32,
    ) -> String {
        let day = dir.join("ir").join(camera).join(date);
        std::fs::create_dir_all(&day).unwrap();
        let img = image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]));
        img.save(day.join(format!("{stem}.jpg"))).unwrap();
        let mut r = empty_record();
        r.id = stem.into();
        r.scene = "ir".into();
        r.camera = camera.into();
        r.timestamp = format!("{date}T01:02:03.004Z");
        r.width = 64;
        r.height = 48;
        r.threshold = 0.5;
        r.buckets = vec![Bucket::Suppressed];
        let person = Det {
            label: "person".into(),
            score: margin_score,
            bbox: [4.0, 4.0, 20.0, 40.0],
        };
        r.raw = vec![person.clone()];
        r.predictions = vec![person];
        r.suppressed = vec![Suppressed {
            label: "dog".into(),
            score: 0.6,
            bbox: [30.0, 30.0, 60.0, 46.0],
            reason: "static".into(),
        }];
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let rel = format!("ir/{camera}/{date}/{stem}.jpg");
        layout::append_review(dir, std::slice::from_ref(&rel)).unwrap();
        rel
    }

    #[tokio::test]
    async fn the_folder_under_review_can_be_another_folder() {
        let d = ds();
        let imported = d.root.join("imported");
        let rel = put_frame(
            &imported,
            "drive",
            "2026-10-21",
            "20261021T010203004Z_a",
            0.9,
        );
        let total = |state: Arc<AppState>| async move {
            let (_, v) = json(
                &state,
                Request::get("/v1/review/next").body(Body::empty()).unwrap(),
            )
            .await;
            v["total"].clone()
        };
        assert_eq!(total(d.state.clone()).await, 0);
        let body = serde_json::json!({ "dir": imported.display().to_string() }).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/folder", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            d.state.config_read().collect.review_dir,
            Some(super::super::browse::plain(
                &imported.canonicalize().unwrap()
            ))
        );
        assert_eq!(total(d.state.clone()).await, 1);
        // Its images are served too.
        let (s, _, _) = call(
            &d.state,
            Request::get(format!(
                "/dataset/image?path={}",
                encode_uri_component(&rel)
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        // The page offers the dataset folder to go back to.
        let (_, page) = get_text(&d.state, "/train/review").await;
        assert!(page.contains("Back to the dataset folder"), "{page}");
        // A file, a missing folder, a folder outside the browsable ones: refused, unchanged.
        let outside = std::env::temp_dir().join(format!("bop-review-out-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&outside).unwrap();
        let file = imported.join("ir/drive/2026-10-21/20261021T010203004Z_a.jpg");
        // (A folder outside the roots only on Unix: on Windows every drive is a root.)
        let mut refused = vec![file, d.root.join("nope")];
        if cfg!(unix) {
            refused.push(outside.clone());
        }
        for p in refused {
            let body = serde_json::json!({ "dir": p.display().to_string() }).to_string();
            let (s, v) = json(&d.state, post_json("/v1/review/folder", &body)).await;
            assert_ne!(s, StatusCode::OK, "{} {v}", p.display());
            assert!(d.state.config_read().collect.review_dir.is_some());
        }
        // The folder already under review stays accepted outside the roots (it passed the check
        // when chosen, often as a native pick, which expires after an hour).
        if cfg!(unix) {
            save_config(&d.state, "test", |c| {
                crate::collect::config::set_review_dir(c, Some(outside.clone()))
            })
            .unwrap();
            let body = serde_json::json!({ "dir": outside.display().to_string() }).to_string();
            let (s, v) = json(&d.state, post_json("/v1/review/folder", &body)).await;
            assert_eq!(s, StatusCode::OK, "{v}");
        }
        let _ = std::fs::remove_dir_all(&outside);
        // Empty = back to the dataset folder.
        let (s, _) = json(&d.state, post_json("/v1/review/folder", r#"{"dir": ""}"#)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(d.state.config_read().collect.review_dir, None);
        assert_eq!(total(d.state.clone()).await, 0);
    }

    /// One frame saved as reviewed, so an export has something to write.
    async fn reviewed_frame(d: &Ds) {
        let rel = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let q = encode_uri_component(&rel);
        let (_, l) = json(
            &d.state,
            Request::get(format!("/v1/review/labels?frame={q}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        let mut f = l["labels"].clone();
        f["status"] = "reviewed".into();
        let uri = format!("/v1/review/labels?frame={q}");
        let (s, v) = json(&d.state, post_json(&uri, &f.to_string())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
    }

    /// `out_dir` of an export and `dir` of an import reach the disk, so they obey the same rule
    /// as the other folder routes: refused outside the browsable folders until the folder is
    /// picked in the native dialog. (On Windows every drive is a root, so Unix only.)
    #[cfg(unix)]
    #[tokio::test]
    async fn export_and_import_folders_must_be_browsable_or_picked() {
        let d = ds();
        reviewed_frame(&d).await;
        let outside =
            std::env::temp_dir().join(format!("bop-review-fold-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&outside).unwrap();
        let text = outside.display().to_string();
        let export = serde_json::json!({"format": "coco", "out_dir": text}).to_string();
        let import = serde_json::json!({ "dir": text }).to_string();

        for (uri, body) in [
            ("/v1/review/export", &export),
            ("/v1/review/import", &import),
        ] {
            let (s, v) = json(&d.state, post_json(uri, body)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{uri}: {v}");
            let m = v["message"].as_str().unwrap();
            assert!(m.contains("outside the browsable folders"), "{m}");
        }
        // Nothing was written, and a child that does not exist yet is no way around it.
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
        let body =
            serde_json::json!({"format": "coco", "out_dir": outside.join("new")}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        assert!(!outside.join("new").exists());

        // A symlink inside a root that leads out of the roots is refused too.
        let link = d.root.join("link-out");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        for target in [link.clone(), link.join("new")] {
            let body = serde_json::json!({"format": "coco", "out_dir": target}).to_string();
            let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        }
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);

        // A native pick lets both through, for the folder and for a new folder inside it.
        super::super::pick::record_pick(&text);
        let (s, v) = json(&d.state, post_json("/v1/review/export", &export)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let (s, v) = json(&d.state, post_json("/v1/review/import", &import)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let body =
            serde_json::json!({"format": "coco", "out_dir": outside.join("new")}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(outside.join("new/export.json").is_file(), "{v}");
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// Exports still work where they worked: the default location, a folder in a root, a new
    /// folder in a root (the form's text field), and the data root's relative paths.
    #[tokio::test]
    async fn export_locations_inside_the_roots_keep_working() {
        let d = ds();
        reviewed_frame(&d).await;
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/export", r#"{"format": "coco"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let default = PathBuf::from(v["report"]["out_dir"].as_str().unwrap());
        assert!(default.starts_with(exports_dir(&d.state)), "{v}");

        let new = d.root.join("exports-here/deep");
        let body = serde_json::json!({"format": "coco", "out_dir": new}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(new.join("export.json").is_file(), "{v}");

        let body = r#"{"format": "coco", "out_dir": "relative-exports"}"#;
        let (s, v) = json(&d.state, post_json("/v1/review/export", body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(d.root.join("relative-exports/export.json").is_file(), "{v}");

        // Bad syntax and system folders stay a 400.
        for bad in ["/etc/x", "a\0b"] {
            let body = serde_json::json!({"format": "coco", "out_dir": bad}).to_string();
            let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}: {v}");
        }
        // An import needs a folder that exists.
        let body = serde_json::json!({ "dir": d.root.join("nope") }).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/import", &body)).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn json(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let (s, _, b) = call(state, req).await;
        (
            s,
            serde_json::from_slice(&b).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn page_and_queue_and_labels_round_trip() {
        let d = ds();
        let far = put_frame(
            &d.dir,
            "drive",
            "2026-10-21",
            "20261021T010203004Z_far",
            0.95,
        );
        let near = put_frame(
            &d.dir,
            "yard",
            "2026-10-22",
            "20261022T010203004Z_near",
            0.52,
        );
        let (s, page) = get_text(&d.state, "/train/review").await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"active\" aria-current=\"page\">Review</a>"),
            "nav"
        );
        assert!(page.contains("/static/review.js"));
        // Browse... on the folder under review (native picker, else the in-page browser of this
        // dialog); the export and X-AnyLabeling folders are on the Dataset tab.
        assert!(page.contains("data-browse=\"folder\" data-target=\"dir\""));
        let (_, data) = get_text(&d.state, "/train/dataset").await;
        assert!(data.contains("data-browse=\"folder\" data-target=\"out_dir\""));
        assert!(data.contains("data-browse=\"folder\" data-target=\"dir\""));
        assert!(page.contains("<dialog id=\"browse-dialog\">"));
        assert!(page.contains("/static/browse.js?v="));
        assert!(page.contains("raccoon"));
        assert!(
            page.contains("name=\"label\" id=\"rv-label\""),
            "class filter"
        );
        assert!(
            data.contains("Only frames with these classes"),
            "export class filter"
        );
        assert!(
            data.contains("name=\"include_empty\" checked"),
            "include empty by default"
        );

        let (s, v) = json(
            &d.state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["item"]["frame"], near, "most uncertain first");
        assert_eq!(v["total"], 2);
        assert_eq!(v["counts"]["remaining"], 2);
        let (_, v2) = json(
            &d.state,
            Request::get(format!(
                "/v1/review/next?after={}",
                encode_uri_component(&near)
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(v2["item"]["frame"], far);
        assert_eq!(v2["position"], 2);
        let (_, v3) = json(
            &d.state,
            Request::get("/v1/review/next?camera=drive")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(v3["total"], 1);
        assert_eq!(
            v3["labels"],
            serde_json::json!(["person"]),
            "unreviewed frames: the reply's labels"
        );
        let (_, v4) = json(
            &d.state,
            Request::get("/v1/review/next?label=person")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(v4["total"], 2, "both frames predicted a person");
        let (_, v4) = json(
            &d.state,
            Request::get("/v1/review/next?label=deer")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(v4["total"], 0);
        let (s, v4) = json(
            &d.state,
            Request::get("/v1/review/next?label=a%07b")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v4}");
        let (s, _) = json(
            &d.state,
            Request::get("/v1/review/next?bucket=nope")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        let q = encode_uri_component(&near);
        let (s, l) = json(
            &d.state,
            Request::get(format!("/v1/review/labels?frame={q}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(l["saved"], false);
        assert_eq!(l["width"], 64);
        assert_eq!(l["labels"]["boxes"].as_array().unwrap().len(), 2);
        assert_eq!(l["labels"]["boxes"][1]["reason"], "static");
        // Restore the suppressed dog, keep the person (pending -> accepted), save as reviewed.
        let mut file = l["labels"].clone();
        file["status"] = "reviewed".into();
        file["boxes"][1]["state"] = "accepted".into();
        let (s, saved) = json(
            &d.state,
            post_json(&format!("/v1/review/labels?frame={q}"), &file.to_string()),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{saved}");
        assert_eq!(saved["saved"], true);
        assert_eq!(saved["labels"]["boxes"][0]["state"], "accepted");
        let on_disk = std::fs::read_to_string(
            d.dir
                .join("ir/yard/2026-10-22/20261022T010203004Z_near.labels.json"),
        )
        .unwrap();
        assert!(on_disk.contains("\"reviewed_at\""), "{on_disk}");
        let (_, v) = json(
            &d.state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(
            v["item"]["frame"], far,
            "reviewed frames leave the default queue"
        );
        assert_eq!(v["counts"]["reviewed"], 1);
        assert_eq!(v["counts"]["per_class"]["dog"], 1);

        // An invalid save changes nothing.
        file["boxes"][0]["label"] = "unicorn".into();
        let (s, e) = json(
            &d.state,
            post_json(&format!("/v1/review/labels?frame={q}"), &file.to_string()),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(e["message"].as_str().unwrap().contains("unicorn"));
    }

    #[tokio::test]
    async fn bad_paths_cross_site_wrong_type_and_oversize() {
        let d = ds();
        let rel = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        std::fs::write(d.dir.join("secret.jpg"), b"x").unwrap();
        for bad in [
            "secret.jpg",
            "..%2Fsecret.jpg",
            "ir%2Fdrive%2F2026-10-21%2F..%2F..%2F..%2Fsecret.jpg",
            "ir%2Fdrive%2F2026-10-21%2F20261021T010203004Z_a.json",
            "ir%2Fdrive%2F2026-10-21%2F20261021T010203004Z_a.labels.json",
            "ir%2Fdrive%2F2026-10-21%2Fmissing.jpg",
            "%2Fetc%2Fpasswd",
        ] {
            let (s, _) = json(
                &d.state,
                Request::get(format!("/v1/review/labels?frame={bad}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::NOT_FOUND, "GET {bad}");
            let (s, _) = json(
                &d.state,
                post_json(&format!("/v1/review/labels?frame={bad}"), "{}"),
            )
            .await;
            assert_eq!(s, StatusCode::NOT_FOUND, "POST {bad}");
        }
        let (s, _) = json(
            &d.state,
            Request::get("/v1/review/labels")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let uri = format!("/v1/review/labels?frame={}", encode_uri_component(&rel));
        // Cross-site.
        let mut req = post_json(&uri, r#"{"status": "reviewed"}"#);
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let mut req = post_json("/v1/review/export", "{}");
        req.headers_mut()
            .insert("sec-fetch-site", "cross-site".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        // A form post (no JSON) is refused.
        let req = Request::post(&uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from("status=reviewed"))
            .unwrap();
        let (s, _) = json(&d.state, req).await;
        assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Oversize.
        let big = format!(
            r#"{{"status": "unreviewed", "pad": "{}"}}"#,
            "x".repeat(LABELS_BODY_LIMIT)
        );
        let (s, v) = json(&d.state, post_json(&uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
        let big = format!(r#"{{"dir": "{}"}}"#, "x".repeat(SMALL_BODY_LIMIT));
        let (s, _) = json(&d.state, post_json("/v1/review/import", &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
        // Remote peers never reach the admin routes.
        let req = super::super::test_util::from_peer(
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
            super::super::test_util::REMOTE,
        );
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert!(
            !d.dir
                .join("ir/drive/2026-10-21/20261021T010203004Z_a.labels.json")
                .exists()
        );
    }

    #[tokio::test]
    async fn export_import_and_classes() {
        let d = ds();
        let a = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let b = put_frame(&d.dir, "yard", "2026-10-22", "20261022T010203004Z_b", 0.9);
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/export", r#"{"format": "yolo"}"#),
        )
        .await;
        assert_eq!(
            s,
            StatusCode::UNPROCESSABLE_ENTITY,
            "nothing reviewed yet: {v}"
        );
        for (rel, empty) in [(&a, false), (&b, true)] {
            let q = encode_uri_component(rel);
            let (_, l) = json(
                &d.state,
                Request::get(format!("/v1/review/labels?frame={q}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            let mut f = l["labels"].clone();
            f["status"] = "reviewed".into();
            if empty {
                f["empty_confirmed"] = true.into();
                f["boxes"][0]["state"] = "rejected".into();
            }
            let (s, v) = json(
                &d.state,
                post_json(&format!("/v1/review/labels?frame={q}"), &f.to_string()),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{v}");
        }
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/export", r#"{"formats": ["yolo", "coco", "benchmark", "xanylabeling"], "holdout_cameras": ["yard"], "benchmark_split": "all"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["report"];
        assert_eq!(r["frames"], 2);
        assert_eq!((r["train"].as_u64(), r["val"].as_u64()), (Some(1), Some(1)));
        assert_eq!(r["empty_confirmed"], 1);
        assert_eq!(
            r["hard_negative_boxes"], 3,
            "the empty frame's person + both dogs"
        );
        let out = PathBuf::from(r["out_dir"].as_str().unwrap());
        assert!(out.starts_with(d.root.join("training").join("exports")));
        let na = "ir.drive.2026-10-21.20261021T010203004Z_a";
        let txt = std::fs::read_to_string(out.join(format!("labels/train/{na}.txt"))).unwrap();
        assert_eq!(txt, "0 0.187500 0.458333 0.250000 0.750000\n");
        let nb = "ir.yard.2026-10-22.20261022T010203004Z_b";
        assert_eq!(
            std::fs::read_to_string(out.join(format!("labels/val/{nb}.txt"))).unwrap(),
            ""
        );
        assert!(out.join(format!("images/val/{nb}.jpg")).is_file());
        assert!(
            std::fs::read_to_string(out.join("data.yaml"))
                .unwrap()
                .contains("  7: \"raccoon\"")
        );
        let coco: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("annotations/instances_train.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(coco["annotations"][0]["category_id"], 1);
        assert!(out.join("export.json").is_file());
        // The benchmark set loads through the benchmark's own loader.
        let bench = out.join("benchmark");
        assert_eq!(r["benchmark_dataset"], format!("dir:{}", bench.display()));
        let (set, warnings) = crate::benchmark::images::load_dir(&bench, None, None).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            set.ground_truth,
            crate::benchmark::images::GroundTruth::Manifest
        );
        assert_eq!(set.images.len(), 2);
        let neg = set
            .images
            .iter()
            .find(|i| i.tags.contains(&"hard-negative".to_string()))
            .unwrap();
        assert!(neg.gt.as_ref().unwrap().is_empty());
        assert!(neg.tags.contains(&"camera:yard".to_string()));

        // X-AnyLabeling: relabel a's person to raccoon in the exported file and import it back.
        let xdir = out.join("xanylabeling");
        let xp = xdir.join(format!("{na}.json"));
        let mut x: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&xp).unwrap()).unwrap();
        assert_eq!(x["checked"], true);
        x["shapes"][0]["label"] = "raccoon".into();
        std::fs::write(&xp, x.to_string()).unwrap();
        let body = serde_json::json!({"dir": xdir}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/import", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["report"]["imported"], 2, "{v}");
        let qa = encode_uri_component(&a);
        let (_, l) = json(
            &d.state,
            Request::get(format!("/v1/review/labels?frame={qa}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(l["labels"]["boxes"][0]["label"], "raccoon");
        assert_eq!(l["labels"]["boxes"][0]["model_label"], "person");
        assert_eq!(l["labels"]["status"], "reviewed");

        // A second export into the same, now non-empty folder goes into a new folder inside it.
        let body = serde_json::json!({"format": "coco", "out_dir": out}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let inner = PathBuf::from(v["report"]["out_dir"].as_str().unwrap());
        // (An existing folder comes back canonical, as from the other folder routes.)
        let canon = super::super::browse::plain(&out.canonicalize().unwrap());
        assert_eq!(inner.parent(), Some(canon.as_path()));
        assert!(inner.join("export.json").is_file());
        // The dataset folder itself is refused.
        let body = serde_json::json!({"format": "coco", "out_dir": d.dir}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(
            v["message"].as_str().unwrap().contains("dataset folder"),
            "{v}"
        );
        let (s, _) = json(
            &d.state,
            post_json("/v1/review/export", r#"{"format": "parquet"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = json(
            &d.state,
            post_json("/v1/review/export", r#"{"out_dir": "/etc/x"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Class filter: only frames with a raccoon (a, relabelled above), without the
        // confirmed-empty frame (b).
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/export",
                r#"{"format": "yolo", "labels": [" raccoon ", "raccoon", ""], "include_empty": false}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["report"]["frames"], 1);
        assert_eq!(v["report"]["filtered_out"], 1);
        assert_eq!(
            v["report"]["class_filter"],
            serde_json::json!({"labels": ["raccoon"], "include_empty": false})
        );
        let out2 = PathBuf::from(v["report"]["out_dir"].as_str().unwrap());
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out2.join("export.json")).unwrap())
                .unwrap();
        assert_eq!(written["class_filter"]["labels"][0], "raccoon");
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/export",
                r#"{"format": "yolo", "labels": ["unicorn"]}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let m = v["message"].as_str().unwrap();
        assert!(
            m.contains("'unicorn' is not in the class list") && m.contains("add 'unicorn'"),
            "{m}"
        );

        // Classes.
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/classes", r#"{"classes": ["person", " fox "]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(d.state.config_read().collect.classes, ["person", "fox"]);
        let (s, _) = json(
            &d.state,
            post_json("/v1/review/classes", r#"{"classes": ["a", "a"]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(d.state.config_read().collect.classes, ["person", "fox"]);
        // Hidden classes: saved when given, refused when not listed, pruned with the list.
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/classes",
                r#"{"classes": ["person", "fox", "bird"], "hidden": [" bird ", "fox"]}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["hidden"], serde_json::json!(["bird", "fox"]));
        assert_eq!(
            d.state.config_read().collect.hidden_classes,
            ["bird", "fox"]
        );
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/classes",
                r#"{"classes": ["person"], "hidden": ["fox"]}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"].as_str().unwrap().contains("hidden_classes"),
            "{v}"
        );
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/classes", r#"{"classes": ["person", "Bird"]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            v["hidden"],
            serde_json::json!(["Bird"]),
            "absent = kept, pruned"
        );
        assert_eq!(d.state.config_read().collect.hidden_classes, ["Bird"]);
        // The catalog.
        let (s, v) = json(&d.state, get("/v1/review/class-catalog")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["classes"], serde_json::json!(["person", "Bird"]));
        assert_eq!(v["hidden"], serde_json::json!(["Bird"]));
        assert!(v["relevant"].is_array() && v["global_object_filter"].is_array());
        assert!(v["models"].is_array());
        let entries = v["entries"].as_array().unwrap();
        assert_eq!(entries[0]["name"], "person");
        assert_eq!(entries[1]["name"], "Bird");
        assert_eq!(entries[1]["hidden"], true);
        assert_eq!(entries[1]["category"], "animal");
        assert!(
            entries
                .iter()
                .any(|e| e["name"] == "raccoon" && e["in_list"] == false)
        );
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn overlap_suggestions_round_trip_the_page_and_the_route_counts_every_status() {
        use crate::collect::labels::overlaps::{Suggestion, Verdict};
        let d = ds();
        let rel = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let q = encode_uri_component(&rel);
        let path = d
            .dir
            .join("ir/drive/2026-10-21/20261021T010203004Z_a.labels.json");
        let car = |bbox: [f32; 4]| labels::LabelBox {
            label: "car".into(),
            bbox,
            source: labels::Source::Auto,
            score: Some(0.5),
            state: labels::BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: labels::AutoInfo::default(),
        };
        let (big, small) = ([0.0, 0.0, 40.0, 40.0], [5.0, 5.0, 25.0, 25.0]);
        let s = Suggestion {
            a: big,
            b: small,
            verdict: Verdict::DropB,
            why: "A closer look found only the kept box".into(),
            kept_score: Some(0.9),
            other_score: Some(0.0),
            low_confidence: false,
            weak: false,
        };
        let f = LabelFile {
            status: labels::Status::Reviewed,
            reviewed_at: Some("2026-10-22T00:00:00Z".into()),
            boxes: vec![car(big), car(small)],
            overlap_suggestions: vec![s.clone()],
            ..LabelFile::default()
        };
        labels::write(&path, &f).unwrap();
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(l["overlap_live"], serde_json::json!([0]));
        assert_eq!(l["labels"]["overlap_suggestions"][0]["verdict"], "drop_b");
        // The page's save posts the file back as it read it.
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            l["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &l["labels"].to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["overlap_live"], serde_json::json!([0]));
        let back = labels::read(&path).unwrap().unwrap();
        assert_eq!(back.overlap_suggestions, [s]);

        // Dry run: reviewed frames count; a pair with an answer does not, unless asked.
        let body = r#"{"dry_run":true}"#;
        let (st, v) = json(&d.state, post_json("/v1/review/suggest-overlaps", body)).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(
            (v["pairs"].as_u64(), v["answered"].as_u64()),
            (Some(0), Some(1))
        );
        assert!(v["message"].as_str().unwrap().contains("tick Ask again"));
        let body = r#"{"dry_run":true,"again":true}"#;
        let (_, v) = json(&d.state, post_json("/v1/review/suggest-overlaps", body)).await;
        assert_eq!(
            (v["frames"].as_u64(), v["pairs"].as_u64()),
            (Some(1), Some(1))
        );
        assert_eq!(v["crops"], 1);
        // A skipped frame is passed over.
        let mut skipped = labels::read(&path).unwrap().unwrap();
        skipped.status = labels::Status::Skipped;
        labels::write(&path, &skipped).unwrap();
        let (_, v) = json(&d.state, post_json("/v1/review/suggest-overlaps", body)).await;
        assert_eq!(
            (v["pairs"].as_u64(), v["passed_over"].as_u64()),
            (Some(0), Some(1))
        );
        // A bad filter and an unknown field are refused.
        let bad = r#"{"dry_run":true,"scene":"dusk"}"#;
        let (st, _) = json(&d.state, post_json("/v1/review/suggest-overlaps", bad)).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        let bad = r#"{"status":"reviewed"}"#;
        let (st, _) = json(&d.state, post_json("/v1/review/suggest-overlaps", bad)).await;
        assert_ne!(st, StatusCode::OK);
    }

    #[tokio::test]
    async fn the_autolabel_preview_counts_what_the_page_lists_and_queues_nothing() {
        let d = ds_with(|c| {
            c.collect.classes = vec!["car".into(), "dog".into()];
            c.collect.hidden_classes = vec!["dog".into()];
        });
        let frame = |stem: &str, label: &str| {
            put_frame(&d.dir, "drive", "2026-10-21", stem, 0.9);
            let path = d
                .dir
                .join(format!("ir/drive/2026-10-21/{stem}.labels.json"));
            let at = |bbox: [f32; 4]| labels::LabelBox {
                label: label.into(),
                bbox,
                source: labels::Source::Auto,
                score: Some(0.5),
                state: labels::BoxState::Accepted,
                origin: None,
                reason: None,
                model_label: None,
                orig_bbox: None,
                difficult: false,
                reject_reason: None,
                auto: Default::default(),
            };
            let file = LabelFile {
                status: labels::Status::Reviewed,
                reviewed_at: Some("2026-10-22T00:00:00Z".into()),
                boxes: vec![at([1.0, 1.0, 30.0, 20.0]), at([2.0, 1.0, 30.0, 20.0])],
                ..LabelFile::default()
            };
            labels::write(&path, &file).unwrap();
        };
        frame("20261021T010203004Z_a", "car");
        frame("20261021T010204004Z_b", "dog");
        let post = |body: &str| {
            Request::post("/training/autolabel/preview")
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let (_, next) = json(
            &d.state,
            get("/v1/review/next?status=reviewed&auto=overlaps&source=all"),
        )
        .await;
        assert_eq!(next["total"], 1, "{next}");

        // The real form's fields; no GPU, download or login is asked for.
        let (st, v) = json(
            &d.state,
            post("mode=labelers&gdino=sam3&rfdetr=on&vlm=2b&status=reviewed&auto=overlaps&limit=1"),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true, "{v}");
        assert_eq!(v["matched"], next["total"], "{v}");
        assert_eq!(v["taken"], 1, "{v}");
        assert_eq!(v["taken_by_status"]["reviewed"], 1, "{v}");
        assert_eq!(v["over_limit"], 0, "{v}");
        assert_eq!(v["limit"], 1, "{v}");
        assert!(v.get("frames").is_none(), "{v}");
        assert!(
            v["estimate"]["text"]
                .as_str()
                .unwrap()
                .contains("GTX 1070 Ti"),
            "{v}"
        );
        assert!(
            v["summary"].as_str().unwrap().contains("This run takes 1"),
            "{v}"
        );
        assert!(
            v["warnings"][0].as_str().unwrap().contains("may reopen"),
            "{v}"
        );

        // The raw filter text of the Review tab's form.
        let (st, v) = json(
            &d.state,
            post("frames=status%3Dreviewed%2Csource%3Dall%2Climit%3D5&limit=9&rfdetr=on"),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(
            (v["matched"].as_u64(), v["limit"].as_u64()),
            (Some(2), Some(5))
        );

        // A threshold out of range is refused before anything is counted.
        let (st, v) = json(
            &d.state,
            post("mode=labelers&gdino=sam3&rfdetr=on&sam3_threshold=1.5"),
        )
        .await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"].as_str().unwrap().contains("sam3 threshold"),
            "{v}"
        );

        // The parser's errors are the real form's.
        let (st, v) = json(&d.state, post("frames=colour%3Dred")).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(v["success"], false, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("unknown frame filter"),
            "{v}"
        );

        // Nothing was queued (the queue is shared with other tests).
        let (_, t) = json(&d.state, get("/v1/training")).await;
        let queued = t["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["title"].as_str().unwrap().contains("status=reviewed"));
        assert!(!queued, "{}", t["tasks"]);
    }

    #[tokio::test]
    async fn saves_name_their_revision_and_stale_ones_are_refused() {
        let d = ds();
        let rel = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let q = encode_uri_component(&rel);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(l["revision"], "", "not saved yet");
        let file = l["labels"].to_string();
        let uri = |rev: &str| format!("/v1/review/labels?frame={q}&revision={rev}");
        let (s, saved) = json(&d.state, post_json(&uri(""), &file)).await;
        assert_eq!(s, StatusCode::OK, "{saved}");
        let r1 = saved["revision"].as_str().unwrap().to_string();
        assert_eq!(r1.len(), 16);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(l["revision"], r1.as_str());
        // A tab that opened the frame before that save.
        let (s, v) = json(&d.state, post_json(&uri(""), &file)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["conflict"], true);
        assert_eq!(v["success"], false);
        assert_eq!(v["message"], CONFLICT_MESSAGE);
        // An auto-label run merges boxes meanwhile: the old revision is stale.
        let path = d
            .dir
            .join("ir/drive/2026-10-21/20261021T010203004Z_a.labels.json");
        labels::update(&path, |f| {
            let mut f = f.unwrap();
            f.auto_run = Some(labels::AutoRun {
                at: "t".into(),
                labeler: "x".into(),
                classes: vec!["person".into()],
            });
            Ok(Some(f))
        })
        .unwrap();
        let (s, _) = json(&d.state, post_json(&uri(&r1), &file)).await;
        assert_eq!(s, StatusCode::CONFLICT);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(l["labels"]["auto_run"]["labeler"], "x", "not overwritten");
        let r2 = l["revision"].as_str().unwrap().to_string();
        let mut file = l["labels"].clone();
        file["status"] = "skipped".into();
        let (s, v) = json(&d.state, post_json(&uri(&r2), &file.to_string())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["labels"]["auto_run"]["labeler"], "x", "round-tripped");
        assert_eq!(
            v["labels"]["auto_run"]["classes"],
            serde_json::json!(["person"]),
            "the run's classes round-trip (the page says what it found nothing of)"
        );
        // Without a revision a save always overwrites (other clients).
        let (s, _) = json(
            &d.state,
            post_json(&format!("/v1/review/labels?frame={q}"), &file.to_string()),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn next_wraps_past_the_last_frame() {
        let d = ds();
        let a = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.52);
        let b = put_frame(&d.dir, "yard", "2026-10-22", "20261022T010203004Z_b", 0.95);
        let next = |after: &str| format!("/v1/review/next?after={}", encode_uri_component(after));
        let (_, v) = json(&d.state, get(&next(&a))).await;
        assert_eq!(
            (v["item"]["frame"].as_str(), v["wrapped"].as_bool()),
            (Some(b.as_str()), Some(false))
        );
        let (_, v) = json(&d.state, get(&next(&b))).await;
        assert_eq!(v["item"]["frame"], a.as_str());
        assert_eq!(v["position"], 1);
        assert_eq!(v["wrapped"], true);
        let (_, v) = json(&d.state, get("/v1/review/next?camera=yard")).await;
        assert_eq!(v["wrapped"], false);
        let (_, v) = json(
            &d.state,
            get(&format!(
                "/v1/review/next?camera=yard&after={}",
                encode_uri_component(&b)
            )),
        )
        .await;
        assert_eq!(
            (v["item"]["frame"].as_str(), v["wrapped"].as_bool()),
            (Some(b.as_str()), Some(true)),
            "one frame: itself again"
        );
        let (s, v) = json(&d.state, get("/v1/review/next?auto=maybe")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["message"].as_str().unwrap().starts_with("auto:"), "{v}");
    }

    /// Write `rel`'s pre-fill with its person prediction triaged `accept` by the labelers (or
    /// left without evidence).
    fn put_auto_labels(dir: &std::path::Path, rel: &str, accept: bool) {
        let frame = Frame::resolve(dir, rel).unwrap();
        let mut f = labels::prefill(
            &frame.record().unwrap(),
            &crate::collect::config::default_classes(),
            &[],
        );
        if accept {
            f.boxes[0].auto.triage = Some(labels::Triage::Accept);
            f.boxes[0].auto.agree = Some(true);
        }
        f.auto_run = Some(labels::AutoRun {
            at: "2026-10-22T00:00:00Z".into(),
            labeler: "mm-gdino-tiny + rfdetr-large".into(),
            classes: vec![],
        });
        labels::write(&frame.labels, &f).unwrap();
    }

    #[tokio::test]
    async fn bulk_accept_saves_only_the_frames_the_triage_settles() {
        let d = ds();
        let a = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let b = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_b", 0.9);
        let c = put_frame(&d.dir, "yard", "2026-10-22", "20261022T010203004Z_c", 0.9);
        let _never = put_frame(&d.dir, "yard", "2026-10-22", "20261022T010203004Z_n", 0.9);
        put_auto_labels(&d.dir, &a, true);
        // The model's person has no auto evidence: accepting unseen would trust the model.
        put_auto_labels(&d.dir, &b, false);
        put_auto_labels(&d.dir, &c, true);
        let (_, v) = json(&d.state, get("/v1/review/next")).await;
        assert_eq!(v["counts"]["auto_accept"], 2, "{v}");
        assert_eq!(v["counts"]["auto_review"], 0);
        for (auto, n) in [("accept", 2), ("nothing", 1), ("none", 1), ("review", 0)] {
            let (_, v) = json(&d.state, get(&format!("/v1/review/next?auto={auto}"))).await;
            assert_eq!(v["total"], n, "auto={auto}: {v}");
        }

        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/accept-auto",
                r#"{"dry_run": true, "scene": ""}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["eligible"].as_u64(), v["accepted"].as_u64()),
            (Some(2), Some(0))
        );
        assert_eq!(v["frames"].as_array().unwrap().len(), 2);
        let fa = Frame::resolve(&d.dir, &a)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap();
        assert_eq!(
            fa.status,
            labels::Status::Unreviewed,
            "a dry run writes nothing"
        );
        let (_, v) = json(
            &d.state,
            post_json(
                "/v1/review/accept-auto",
                r#"{"dry_run": true, "camera": "yard"}"#,
            ),
        )
        .await;
        assert_eq!(v["frames"], serde_json::json!([c]), "the filters apply");

        // At most `limit`.
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/accept-auto",
                r#"{"dry_run": false, "limit": 1}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["eligible"].as_u64(), v["accepted"].as_u64()),
            (Some(2), Some(1))
        );
        assert!(
            v["message"].as_str().unwrap().contains("1 more eligible"),
            "{v}"
        );
        let (_, v) = json(&d.state, post_json("/v1/review/accept-auto", "{}")).await;
        assert_eq!(
            (v["eligible"].as_u64(), v["accepted"].as_u64()),
            (Some(1), Some(1)),
            "{v}"
        );
        assert_eq!(v["failed"], serde_json::json!([]));
        for rel in [&a, &c] {
            let f = Frame::resolve(&d.dir, rel)
                .unwrap()
                .labels()
                .unwrap()
                .unwrap();
            assert_eq!(f.status, labels::Status::Reviewed, "{rel}");
            assert!(f.bulk_accepted);
            assert!(f.reviewed_at.is_some());
            let states: Vec<labels::BoxState> = f.boxes.iter().map(|b| b.state).collect();
            assert_eq!(
                states,
                [labels::BoxState::Accepted, labels::BoxState::Rejected],
                "the person accepted, the suppressed dog a hard negative"
            );
        }
        let fb = Frame::resolve(&d.dir, &b)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap();
        assert_eq!(fb.status, labels::Status::Unreviewed);
        let (_, v) = json(&d.state, post_json("/v1/review/accept-auto", "{}")).await;
        assert_eq!(
            (v["eligible"].as_u64(), v["accepted"].as_u64()),
            (Some(0), Some(0))
        );

        // The re-check under the lock: a frame that changed is left alone.
        let (dir, classes, _) = context(&d.state);
        assert!(!accept_one(&dir, &b, &classes, "t").unwrap());
        assert!(accept_one(&dir, "ir/x/2026-10-22/nope.jpg", &classes, "t").is_err());

        // Bad requests.
        for (body, field) in [
            (r#"{"dry_run": true, "colour": "red"}"#, "unknown field"),
            (r#"{"limit": 0}"#, "limit"),
            (r#"{"limit": 20001}"#, "limit"),
            (r#"{"scene": "dusk"}"#, "scene"),
        ] {
            let (s, v) = json(&d.state, post_json("/v1/review/accept-auto", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(
                v["message"].as_str().unwrap().contains(field),
                "{body}: {v}"
            );
        }
        let mut req = post_json("/v1/review/accept-auto", "{}");
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let big = format!(r#"{{"camera": "{}"}}"#, "x".repeat(SMALL_BODY_LIMIT));
        let (s, _) = json(&d.state, post_json("/v1/review/accept-auto", &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }
    #[tokio::test]
    async fn parked_and_rejected_spots_round_trip_and_a_camera_move_forgets_them() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Triage};
        let d = ds();
        let today = crate::collect::utc_date(std::time::SystemTime::now());
        // Decisions dated today: put_frame's fixed date would fall out of the window.
        let dated = |stem: &str| {
            let rel = put_frame(
                &d.dir,
                stem,
                None,
                vec![auto_box("car", 0.4, Triage::Review)],
            );
            let old = d.dir.join("ir/cam/2026-10-21");
            let new = d.dir.join(format!("ir/cam/{today}"));
            std::fs::create_dir_all(&new).unwrap();
            for ext in ["jpg", "json", "labels.json"] {
                std::fs::rename(
                    old.join(format!("{stem}.{ext}")),
                    new.join(format!("{stem}.{ext}")),
                )
                .unwrap();
            }
            rel.replace("2026-10-21", &today)
        };
        for stem in ["00000000T000000000Z_a", "00000000T000000000Z_b"] {
            let rel = dated(stem);
            let mut f = labels_of(&d.dir, &rel);
            f.status = labels::Status::Reviewed;
            f.boxes[0].state = BoxState::Rejected;
            labels::write(&Frame::resolve(&d.dir, &rel).unwrap().labels, &f).unwrap();
        }
        let (s, v) = json(&d.state, get("/v1/review/spots?camera=cam")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["rejected"][0]["rejections"], 2, "{v}");
        assert_eq!(v["live"], 1);
        let (s, _) = json(&d.state, get("/v1/review/spots?camera=..%2Fx")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // The page's "Real, parked": clamped to the frame, dated by the frame.
        let frame = put_frame(&d.dir, "20261021T010000000Z_c", None, vec![]);
        let body = serde_json::json!({
            "frame": frame, "label": "car", "bbox": [40.0, 4.0, 900.0, 40.0]
        })
        .to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/spots/parked", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["new"].as_bool(), v["spot"]["from"].as_str()),
            (Some(true), Some("2026-10-21"))
        );
        assert_eq!(v["spot"]["bbox"][2], 64.0);
        let (_, v) = json(&d.state, post_json("/v1/review/spots/parked", &body)).await;
        assert_eq!(v["new"], false, "the same spot again");
        assert!(d.dir.join("spots/cam.json").is_file());
        let (_, v) = json(&d.state, get("/v1/review/spots?camera=cam")).await;
        assert_eq!(v["parked"]["64x48"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["live"], 2);

        // Refused: a frame that is not one, a bad box, a bad class, an unknown field.
        for (body, status) in [
            (
                r#"{"frame":"../x.jpg","label":"car","bbox":[1,1,9,9]}"#,
                StatusCode::NOT_FOUND,
            ),
            (
                &serde_json::json!({"frame": frame, "label": "car", "bbox": [70, 1, 90, 9]})
                    .to_string(),
                StatusCode::BAD_REQUEST,
            ),
            (
                &serde_json::json!({"frame": frame, "label": "", "bbox": [1, 1, 9, 9]}).to_string(),
                StatusCode::BAD_REQUEST,
            ),
            (
                &serde_json::json!({"frame": frame, "label": "car", "bbox": [1, 1, 9, 9], "x": 1})
                    .to_string(),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let (s, v) = json(&d.state, post_json("/v1/review/spots/parked", body)).await;
            assert_eq!(s, status, "{body}: {v}");
        }

        // Retire one; again it is gone.
        let body = r#"{"camera":"cam","width":64,"height":48,"label":"car","bbox":[40,4,64,40]}"#;
        let (s, v) = json(&d.state, post_json("/v1/review/spots/retire", body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let (s, _) = json(&d.state, post_json("/v1/review/spots/retire", body)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // The camera moved: the rejected spot stops counting, a parked one retires.
        let body = serde_json::json!({
            "frame": frame, "label": "bus", "bbox": [1.0, 1.0, 20.0, 20.0]
        })
        .to_string();
        json(&d.state, post_json("/v1/review/spots/parked", &body)).await;
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/spots/moved", r#"{"camera":"cam"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["retired"], 1);
        let (_, v) = json(&d.state, get("/v1/review/spots?camera=cam")).await;
        assert_eq!(v["live"], 0, "{v}");
        assert!(v["moved_at"].is_string());
        assert!(v["rejected"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn repeats_history_spot_edits_and_repeat_only_frames() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Triage};
        let d = ds();
        let car = || auto_box("car", 0.4, Triage::Review); // [2, 2, 30, 40] of 64 x 48
        // A frame the reviewer looked at: the car there was rejected as a reflection.
        let seen = put_frame(&d.dir, "20261021T000000000Z_seen", None, vec![car()]);
        let frame = Frame::resolve(&d.dir, &seen).unwrap();
        let mut f = labels_of(&d.dir, &seen);
        f.status = labels::Status::Reviewed;
        f.boxes[0].state = BoxState::Rejected;
        f.boxes[0].reject_reason = Some(labels::RejectReason::Reflection);
        labels::write(&frame.labels, &f).unwrap();
        let a = put_frame(&d.dir, "20261021T010000000Z_a", None, vec![car()]);
        let b = put_frame(
            &d.dir,
            "20261021T020000000Z_b",
            None,
            vec![car(), auto_box("dog", 0.4, Triage::Review)],
        );

        // History: the camera's decisions; a bad camera name is refused.
        let (s, v) = json(&d.state, get("/v1/review/history?camera=cam")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["decisions"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["decisions"][0]["reject_reason"], "reflection");
        assert_eq!(v["decisions"][0]["frame"], seen.as_str());
        let (_, v) = json(&d.state, get("/v1/review/history?camera=other")).await;
        assert_eq!(v["decisions"], serde_json::json!([]));
        let (s, _) = json(&d.state, get("/v1/review/history?camera=..%2Fx")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Reject the spot everywhere: dry run first, then for real.
        let body = |extra: &str| {
            format!(
                r#"{{"camera": "cam", "label": "car", "bbox": [2, 2, 30, 40], "width": 64,
                    "height": 48{extra}}}"#
            )
        };
        let skip = format!(r#", "dry_run": true, "skip_frame": "{b}""#);
        let (_, v) = json(&d.state, post_json("/v1/review/reject-spot", &body(&skip))).await;
        assert_eq!(
            v["eligible"].as_u64(),
            Some(1),
            "the frame on screen is left out: {v}"
        );
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/reject-spot", &body(r#", "dry_run": true"#)),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["changed"].as_u64()
            ),
            (Some(2), Some(2), Some(0))
        );
        assert_eq!(labels_of(&d.dir, &a).boxes[0].state, BoxState::Pending);
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/reject-spot",
                &body(r#", "reject_reason": "static_object""#),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changed"].as_u64(), Some(2), "{v}");
        let la = labels_of(&d.dir, &a);
        assert_eq!(la.boxes[0].state, BoxState::Rejected);
        assert_eq!(
            la.boxes[0].reject_reason,
            Some(labels::RejectReason::StaticObject)
        );
        // The bulk rule's rejections do not count as a look; the cache was dropped by the writes.
        let (_, v) = json(&d.state, get("/v1/review/history?camera=cam")).await;
        assert_eq!(v["decisions"].as_array().unwrap().len(), 1, "{v}");

        // Repeat-only: a (nothing pending) sorts last in the queue and is saved; b (the dog) not.
        let (_, v) = json(&d.state, get("/v1/review/next?auto=repeats")).await;
        assert_eq!(v["item"]["frame"], a.as_str(), "{v}");
        assert_eq!(v["counts"]["repeats_only"].as_u64(), Some(1), "{v}");
        let (_, v) = json(&d.state, get("/v1/review/next")).await;
        assert_eq!(
            v["item"]["frame"],
            b.as_str(),
            "repeat-only frames come last: {v}"
        );
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/accept-repeats", r#"{"dry_run": true}"#),
        )
        .await;
        assert_eq!(
            (s, v["eligible"].as_u64()),
            (StatusCode::OK, Some(1)),
            "{v}"
        );
        let (s, v) = json(&d.state, post_json("/v1/review/accept-repeats", "{}")).await;
        assert_eq!(
            (s, v["accepted"].as_u64()),
            (StatusCode::OK, Some(1)),
            "{v}"
        );
        assert_eq!(labels_of(&d.dir, &a).status, labels::Status::Reviewed);
        assert_eq!(labels_of(&d.dir, &b).status, labels::Status::Unreviewed);

        // Accept-spot needs 0.8 and no reason; bad input is refused.
        for (uri, extra) in [
            ("/v1/review/accept-spot", r#", "min_iou": 0.7"#),
            (
                "/v1/review/accept-spot",
                r#", "reject_reason": "rain_snow""#,
            ),
            ("/v1/review/reject-spot", r#", "reject_reason": "aliens""#),
            ("/v1/review/reject-spot", r#", "min_iou": 0.1"#),
        ] {
            let (s, v) = json(&d.state, post_json(uri, &body(extra))).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{uri} {extra}: {v}");
        }
        let (s, _) = json(
            &d.state,
            post_json(
                "/v1/review/reject-spot",
                r#"{"camera": "cam", "label": "car", "bbox": [5, 5, 1, 1], "width": 64, "height": 48}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = json(
            &d.state,
            post_json("/v1/review/reject-spot", &body(r#", "nope": 1"#)),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "unknown fields are refused");
    }

    #[tokio::test]
    async fn restore_reopened_asks_first_and_leaves_frames_with_pending_boxes_alone() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Triage};
        let d = ds();
        let reopen = |stem: &str, boxes: Vec<labels::LabelBox>| {
            let rel = put_frame(&d.dir, stem, None, boxes);
            let frame = Frame::resolve(&d.dir, &rel).unwrap();
            let mut f = labels_of(&d.dir, &rel);
            f.reviewed_at = Some("2026-10-01T00:00:00Z".into());
            f.disagreement = Some("re-label by sam3: 1 new box".into());
            labels::write(&frame.labels, &f).unwrap();
            rel
        };
        let mut no = auto_box("car", 0.7, Triage::Review);
        no.state = BoxState::Rejected;
        let done = reopen("a", vec![no]);
        let waiting = reopen("b", vec![auto_box("car", 0.7, Triage::Review)]);
        let never = put_frame(&d.dir, "c", None, Vec::new());

        let uri = "/v1/review/restore-reopened";
        let (s, v) = json(&d.state, post_json(uri, r#"{"dry_run": true}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["eligible"].as_u64(), v["restored"].as_u64()),
            (Some(1), Some(0))
        );
        assert_eq!(v["frames"], serde_json::json!([done]));
        assert_eq!(labels_of(&d.dir, &done).status, labels::Status::Unreviewed);

        let (s, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["eligible"].as_u64(), v["restored"].as_u64()),
            (Some(1), Some(1))
        );
        assert!(v["message"].as_str().unwrap().contains("Restored 1"), "{v}");
        let f = labels_of(&d.dir, &done);
        assert_eq!(f.status, labels::Status::Reviewed);
        assert_eq!(f.disagreement, None);
        assert_eq!(f.boxes[0].state, BoxState::Rejected);
        assert_eq!(
            labels_of(&d.dir, &waiting).status,
            labels::Status::Unreviewed
        );
        assert!(labels_of(&d.dir, &waiting).disagreement.is_some());
        assert_eq!(labels_of(&d.dir, &never).status, labels::Status::Unreviewed);

        // The filters narrow it, bad input is refused.
        let (s, v) = json(&d.state, post_json(uri, r#"{"camera": "other"}"#)).await;
        assert_eq!(
            (s, v["eligible"].as_u64()),
            (StatusCode::OK, Some(0)),
            "{v}"
        );
        let (s, _) = json(&d.state, post_json(uri, r#"{"limit": 0}"#)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = json(&d.state, post_json(uri, r#"{"nope": 1}"#)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "unknown fields are refused");
    }

    #[tokio::test]
    async fn reject_low_rejects_only_pending_auto_boxes_below_the_score() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Triage};
        let d = ds();
        let a = put_frame(
            &d.dir,
            "a",
            None,
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("person", 0.2, Triage::Review),
                auto_box("car", 0.1, Triage::Review),
            ],
        );
        let b = put_frame(
            &d.dir,
            "b",
            None,
            vec![auto_box("dog", 0.25, Triage::Review)],
        );
        let high = put_frame(
            &d.dir,
            "c",
            None,
            vec![auto_box("dog", 0.8, Triage::Accept)],
        );
        let reviewed = put_frame(
            &d.dir,
            "r",
            None,
            vec![auto_box("dog", 0.1, Triage::Review)],
        );
        let frame = Frame::resolve(&d.dir, &reviewed).unwrap();
        let mut f = labels_of(&d.dir, &reviewed);
        f.status = labels::Status::Reviewed;
        labels::write(&frame.labels, &f).unwrap();

        let uri = "/v1/review/reject-low";
        let (s, v) = json(
            &d.state,
            post_json(uri, r#"{"dry_run": true, "min_score": 0.3}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(2), Some(3), Some(0)),
            "{v}"
        );
        assert_eq!(
            labels_of(&d.dir, &a).boxes[1].state,
            BoxState::Pending,
            "a dry run writes nothing"
        );

        let (s, v) = json(&d.state, post_json(uri, r#"{"min_score": 0.3}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(2), Some(3), Some(2)),
            "{v}"
        );
        assert_eq!(v["failed"], serde_json::json!([]));
        let fa = labels_of(&d.dir, &a);
        assert_eq!(fa.status, labels::Status::Unreviewed);
        let states: Vec<BoxState> = fa.boxes.iter().map(|b| b.state).collect();
        assert_eq!(
            states,
            [BoxState::Pending, BoxState::Rejected, BoxState::Rejected]
        );
        assert_eq!(fa.boxes[1].reason.as_deref(), Some(bulk::LOW_SCORE_REASON));
        assert_eq!(labels_of(&d.dir, &b).boxes[0].state, BoxState::Rejected);
        assert_eq!(labels_of(&d.dir, &high).boxes[0].state, BoxState::Pending);
        let fr = labels_of(&d.dir, &reviewed);
        assert_eq!(
            (fr.status, fr.boxes[0].state),
            (labels::Status::Reviewed, BoxState::Pending),
            "a reviewed frame is left alone"
        );
        let (_, v) = json(&d.state, post_json(uri, r#"{"min_score": 0.3}"#)).await;
        assert_eq!(
            (v["eligible"].as_u64(), v["rejected"].as_u64()),
            (Some(0), Some(0))
        );

        // Bad requests.
        for (body, field) in [
            (r#"{"dry_run": true}"#, "min_score"),
            (r#"{"min_score": 0}"#, "min_score"),
            (r#"{"min_score": 1.5}"#, "min_score"),
            (r#"{"min_score": "high"}"#, "invalid JSON body"),
            (r#"{"min_score": 0.3, "colour": "red"}"#, "unknown field"),
            (r#"{"min_score": 0.3, "limit": 0}"#, "limit"),
            (r#"{"min_score": 0.3, "scene": "dusk"}"#, "scene"),
        ] {
            let (s, v) = json(&d.state, post_json(uri, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(
                v["message"].as_str().unwrap().contains(field),
                "{body}: {v}"
            );
        }
        let mut req = post_json(uri, r#"{"min_score": 0.9}"#);
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let req = super::super::test_util::from_peer(
            post_json(uri, r#"{"min_score": 0.9}"#),
            super::super::test_util::REMOTE,
        );
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(labels_of(&d.dir, &high).boxes[0].state, BoxState::Pending);
        let big = format!(
            r#"{{"min_score": 0.3, "camera": "{}"}}"#,
            "x".repeat(SMALL_BODY_LIMIT)
        );
        let (s, _) = json(&d.state, post_json(uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn reject_stacked_spans_reviewed_frames_and_counts_what_is_left() {
        use crate::collect::labels::bulk::test_support::{
            auto_box, labels_of, put_frame, put_frame_at,
        };
        use labels::{BoxState, RejectReason, Status, Triage};
        let car = |bbox: [f32; 4], score: f32| labels::LabelBox {
            bbox,
            state: BoxState::Accepted,
            ..auto_box("car", score, Triage::Accept)
        };
        let (a, near, small) = (
            [0.0, 0.0, 40.0, 40.0],
            [0.0, 0.0, 40.0, 38.0],
            [5.0, 5.0, 25.0, 25.0],
        );
        let d = ds_with(|c| {
            c.collect.classes = ["person", "car"].map(String::from).to_vec();
        });
        let reviewed = put_frame(
            &d.dir,
            "r",
            None,
            vec![car(a, 0.9), car(near, 0.4), car(small, 0.3)],
        );
        let other = put_frame_at(
            &d.dir,
            "drive",
            "o",
            (64, 48),
            None,
            vec![car(a, 0.9), car(near, 0.4)],
        );
        for rel in [&reviewed, &other] {
            let frame = Frame::resolve(&d.dir, rel).unwrap();
            let mut f = labels_of(&d.dir, rel);
            f.status = Status::Reviewed;
            f.reviewed_at = Some("2026-10-01T00:00:00Z".into());
            labels::write(&frame.labels, &f).unwrap();
        }

        let uri = "/v1/review/reject-stacked";
        let (s, v) = json(&d.state, post_json(uri, r#"{"dry_run": true}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64(),
                v["overlaps_left"].as_u64()
            ),
            (Some(2), Some(2), Some(0), Some(1)),
            "{v}"
        );
        assert!(
            v["message"].as_str().unwrap().contains("Would reject 2"),
            "{v}"
        );
        assert_eq!(
            labels_of(&d.dir, &reviewed).boxes[1].state,
            BoxState::Accepted,
            "a dry run writes nothing"
        );

        // One camera first; the page's Frames filter does not narrow it.
        let one = r#"{"camera": "drive", "source": "queue"}"#;
        let (s, v) = json(&d.state, post_json(uri, one)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["rejected"], 1, "{v}");
        assert_eq!(v["overlaps_left"], 0, "{v}");
        assert_eq!(
            labels_of(&d.dir, &reviewed).boxes[1].state,
            BoxState::Accepted,
            "another camera"
        );

        let (s, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64(),
                v["overlaps_left"].as_u64()
            ),
            (Some(1), Some(1), Some(1), Some(1)),
            "{v}"
        );
        assert_eq!(v["failed"], serde_json::json!([]));
        assert_eq!(v["frames"], serde_json::json!([reviewed.clone()]));
        let m = v["message"].as_str().unwrap();
        assert!(
            m.contains("Rejected 1 stacked") && m.contains("overlapping boxes"),
            "{m}"
        );
        let f = labels_of(&d.dir, &reviewed);
        assert_eq!(f.status, Status::Reviewed);
        assert_eq!(f.reviewed_at.as_deref(), Some("2026-10-01T00:00:00Z"));
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
        assert_eq!(f.boxes[1].reject_reason, Some(RejectReason::Duplicate));
        assert_eq!(f.boxes[2].state, BoxState::Accepted, "nested: a reviewer's");

        // Run again: nothing to do, the overlap still waits. The queue's filter finds it.
        let (_, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["rejected"].as_u64(),
                v["overlaps_left"].as_u64()
            ),
            (Some(0), Some(0), Some(1)),
            "{v}"
        );
        let next = "/v1/review/next?auto=overlaps&status=reviewed";
        let (s, v) = json(
            &d.state,
            Request::builder().uri(next).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["item"]["frame"], serde_json::json!(reviewed), "{v}");
        assert_eq!(v["total"], 1, "{v}");

        // Bad requests.
        for (body, field) in [
            (r#"{"min_score": 0.3}"#, "unknown field"),
            (r#"{"limit": 0}"#, "limit"),
            (r#"{"limit": 20001}"#, "limit"),
            (r#"{"scene": "dusk"}"#, "scene"),
            (r#"{"source": "nope"}"#, "source"),
            (r#"{"dry_run": "yes"}"#, "invalid JSON body"),
        ] {
            let (s, v) = json(&d.state, post_json(uri, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(
                v["message"].as_str().unwrap().contains(field),
                "{body}: {v}"
            );
        }
        let mut req = post_json(uri, "{}");
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let req = super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        );
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let big = format!(r#"{{"camera": "{}"}}"#, "x".repeat(SMALL_BODY_LIMIT));
        let (s, _) = json(&d.state, post_json(uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn reject_other_class_rejects_only_pending_proposals_off_the_list() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Origin, Source, Triage};
        let model = |label: &str| labels::LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            auto: Default::default(),
            ..auto_box(label, 0.7, Triage::Review)
        };
        let d = ds_with(|c| {
            c.collect.classes = ["person", "car", "potted plant"].map(String::from).to_vec();
        });
        let a = put_frame(
            &d.dir,
            "a",
            None,
            vec![
                model("person"),
                model("clock"),
                model("pottedplant"),
                auto_box("clock", 0.9, Triage::Review),
            ],
        );
        let b = put_frame(&d.dir, "b", None, vec![model("parking meter")]);
        let on_list = put_frame(&d.dir, "c", None, vec![model("car")]);
        let reviewed = put_frame(&d.dir, "r", None, vec![model("clock")]);
        let frame = Frame::resolve(&d.dir, &reviewed).unwrap();
        let mut f = labels_of(&d.dir, &reviewed);
        f.status = labels::Status::Reviewed;
        f.boxes[0].state = BoxState::Accepted;
        labels::write(&frame.labels, &f).unwrap();

        let uri = "/v1/review/reject-other-class";
        let (s, v) = json(&d.state, post_json(uri, r#"{"dry_run": true}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["relabelled"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(2), Some(4), Some(1), Some(0)),
            "{v}"
        );
        assert_eq!(
            v["reviewed_with_unwanted"],
            serde_json::json!({"frames": 1, "boxes": 1, "classes": ["clock"]})
        );
        assert_eq!(
            labels_of(&d.dir, &a).boxes[1].state,
            BoxState::Pending,
            "a dry run writes nothing"
        );

        let (s, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["relabelled"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(2), Some(4), Some(1), Some(2)),
            "{v}"
        );
        assert_eq!(v["reviewed_with_unwanted"]["frames"], 1, "{v}");
        assert_eq!(v["failed"], serde_json::json!([]));
        assert!(
            v["message"].as_str().unwrap().contains("Renamed 1 box"),
            "{v}"
        );
        let fa = labels_of(&d.dir, &a);
        assert_eq!(fa.status, labels::Status::Unreviewed);
        let got: Vec<(&str, BoxState)> = fa
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.state))
            .collect();
        assert_eq!(
            got,
            [
                ("person", BoxState::Pending),
                ("clock", BoxState::Rejected),
                ("potted plant", BoxState::Pending),
                ("clock", BoxState::Rejected),
            ]
        );
        assert_eq!(
            fa.boxes[1].reason.as_deref(),
            Some(bulk::OTHER_CLASS_REASON)
        );
        assert_eq!(fa.boxes[2].model_label.as_deref(), Some("pottedplant"));
        assert_eq!(labels_of(&d.dir, &b).boxes[0].state, BoxState::Rejected);
        assert_eq!(
            labels_of(&d.dir, &on_list).boxes[0].state,
            BoxState::Pending
        );
        let fr = labels_of(&d.dir, &reviewed);
        assert_eq!(
            (fr.status, fr.boxes[0].state),
            (labels::Status::Reviewed, BoxState::Accepted),
            "a reviewed frame is left alone"
        );
        let (_, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(
            (v["eligible"].as_u64(), v["rejected"].as_u64()),
            (Some(0), Some(0))
        );

        // Bad requests.
        for (body, field) in [
            (r#"{"min_score": 0.3}"#, "unknown field"),
            (r#"{"limit": 0}"#, "limit"),
            (r#"{"scene": "dusk"}"#, "scene"),
            (r#"{"dry_run": "yes"}"#, "invalid JSON body"),
        ] {
            let (s, v) = json(&d.state, post_json(uri, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(
                v["message"].as_str().unwrap().contains(field),
                "{body}: {v}"
            );
        }
        let mut req = post_json(uri, "{}");
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let req = super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        );
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let big = format!(r#"{{"camera": "{}"}}"#, "x".repeat(SMALL_BODY_LIMIT));
        let (s, _) = json(&d.state, post_json(uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn hiding_a_class_rejects_its_pending_boxes_and_asks_about_reviewed_ones() {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Origin, Source, Status, Triage};
        let model = |label: &str| labels::LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            auto: Default::default(),
            ..auto_box(label, 0.7, Triage::Review)
        };
        let d = ds_with(|c| {
            c.collect.classes = ["person", "car", "dog"].map(String::from).to_vec();
        });
        let mut human = auto_box("car", 0.9, Triage::Review);
        human.source = Source::Human;
        human.score = None;
        let u = put_frame(
            &d.dir,
            "u",
            None,
            vec![
                model("person"),
                model("car"),
                auto_box("car", 0.9, Triage::Accept),
                auto_box("dog", 0.8, Triage::Review),
                human,
            ],
        );
        let r = put_frame(&d.dir, "r", None, vec![model("person"), model("car")]);
        let frame = Frame::resolve(&d.dir, &r).unwrap();
        let mut f = labels_of(&d.dir, &r);
        f.status = Status::Reviewed;
        for b in &mut f.boxes {
            b.state = BoxState::Accepted;
        }
        labels::write(&frame.labels, &f).unwrap();

        // An unchanged save rejects nothing.
        let uri = "/v1/review/classes";
        let (s, v) = json(
            &d.state,
            post_json(uri, r#"{"classes": ["person", "car", "dog"]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["auto_rejected"]["boxes"], 0, "{v}");
        assert_eq!(v["reviewed_with_unwanted"]["frames"], 0, "{v}");
        assert_eq!(labels_of(&d.dir, &u).boxes[1].state, BoxState::Pending);

        // `car` hidden, `dog` dropped from the list.
        let (s, v) = json(
            &d.state,
            post_json(uri, r#"{"classes": ["person", "car"], "hidden": ["car"]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["hidden"], serde_json::json!(["car"]));
        assert_eq!(v["newly_unwanted"], serde_json::json!(["car", "dog"]));
        assert_eq!(
            v["auto_rejected"],
            serde_json::json!({"frames": 1, "boxes": 3, "relabelled": 0, "failed": []}),
            "{v}"
        );
        assert_eq!(
            v["reviewed_with_unwanted"],
            serde_json::json!({"frames": 1, "boxes": 1, "classes": ["car"]})
        );
        let fu = labels_of(&d.dir, &u);
        assert_eq!(fu.status, Status::Unreviewed);
        let got: Vec<(&str, BoxState)> = fu
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.state))
            .collect();
        assert_eq!(
            got,
            [
                ("person", BoxState::Pending),
                ("car", BoxState::Rejected),
                ("car", BoxState::Rejected),
                ("dog", BoxState::Rejected),
                ("car", BoxState::Pending),
            ],
            "a human box is left to the reviewer"
        );
        assert!(
            fu.boxes[1..4]
                .iter()
                .all(|b| b.reason.as_deref() == Some(bulk::OTHER_CLASS_REASON))
        );
        assert_eq!(
            labels_of(&d.dir, &r).boxes[1].state,
            BoxState::Accepted,
            "reviewed frames wait for the question"
        );
        // A frame without a label file is pre-filled without the hidden class.
        let fresh = put_frame_with_record(&d, "fresh", &["person", "car"]);
        let q = encode_uri_component(&fresh);
        let (_, v) = json(
            &d.state,
            Request::get(format!("/v1/review/labels?frame={q}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        let labels: Vec<&str> = v["labels"]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["label"].as_str().unwrap())
            .collect();
        assert_eq!(labels, ["person"], "{v}");
        assert_eq!(v["hidden"], serde_json::json!(["car"]));

        // The reviewed frames, on request.
        let uri = "/v1/review/reject-unwanted-reviewed";
        let (s, v) = json(&d.state, post_json(uri, r#"{"dry_run": true}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(1), Some(1), Some(0)),
            "{v}"
        );
        assert_eq!(v["classes"], serde_json::json!(["car"]));
        assert_eq!(labels_of(&d.dir, &r).boxes[1].state, BoxState::Accepted);
        let (s, v) = json(&d.state, post_json(uri, "{}")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (
                v["eligible"].as_u64(),
                v["boxes"].as_u64(),
                v["rejected"].as_u64()
            ),
            (Some(1), Some(1), Some(1)),
            "{v}"
        );
        assert_eq!(v["frames"], serde_json::json!([r.clone()]));
        let fr = labels_of(&d.dir, &r);
        assert_eq!(
            (fr.status, fr.boxes[0].state, fr.boxes[1].state),
            (Status::Reviewed, BoxState::Accepted, BoxState::Rejected)
        );
        let (_, v) = json(&d.state, post_json(uri, r#"{"dry_run": true}"#)).await;
        assert_eq!(v["eligible"], 0, "{v}");

        // Bad requests and the guards.
        for (body, field) in [
            (r#"{"scene": "day"}"#, "unknown field"),
            (r#"{"limit": 0}"#, "limit"),
            (r#"{"dry_run": "yes"}"#, "invalid JSON body"),
        ] {
            let (s, v) = json(&d.state, post_json(uri, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(
                v["message"].as_str().unwrap().contains(field),
                "{body}: {v}"
            );
        }
        let mut req = post_json(uri, "{}");
        req.headers_mut()
            .insert("origin", "http://evil.example".parse().unwrap());
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let req = super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        );
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let big = format!(r#"{{"limit": 1{}}}"#, " ".repeat(SMALL_BODY_LIMIT));
        let (s, _) = json(&d.state, post_json(uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// A queued frame `<stem>` of `d` whose sidecar predicts `labels`, without a label file.
    fn put_frame_with_record(d: &Ds, stem: &str, labels: &[&str]) -> String {
        let rel = crate::collect::labels::bulk::test_support::put_frame(&d.dir, stem, None, vec![]);
        let frame = Frame::resolve(&d.dir, &rel).unwrap();
        std::fs::remove_file(&frame.labels).unwrap();
        let mut rec = frame.record().unwrap();
        rec.predictions = labels
            .iter()
            .zip([1.0_f32, 21.0, 41.0])
            .map(|(l, x)| Det {
                label: (*l).into(),
                score: 0.9,
                bbox: [x, 1.0, x + 14.0, 30.0],
            })
            .collect();
        std::fs::write(&frame.sidecar, serde_json::to_vec(&rec).unwrap()).unwrap();
        rel
    }

    #[tokio::test]
    async fn a_new_dataset_folder_is_not_an_unplugged_drive() {
        let d = ds();
        let new = d.root.join("not-yet");
        d.state
            .config
            .update("test", |c| {
                c.collect.dir = Some(new.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(folder_problem(&d.state, &new), None);
        // Its parent gone too is a drive that is not there.
        let gone = d.root.join("drive").join("frames");
        assert!(folder_problem(&d.state, &gone).is_some());
    }

    #[tokio::test]
    async fn a_missing_folder_under_review_is_named_not_taken_for_an_empty_dataset() {
        let d = ds();
        let rel = put_frame(&d.dir, "cam", "2026-10-22", "20261022T010203004Z_abcd", 0.9);
        d.state
            .config
            .update("test", |c| {
                c.collect.review_dir = Some(d.dir.clone());
                Ok(())
            })
            .unwrap();
        // The drive is unplugged.
        let moved = d.root.join("unplugged");
        std::fs::rename(&d.dir, &moved).unwrap();
        let (s, v) = json(
            &d.state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["folder_missing"], true, "{v}");
        let m = v["message"].as_str().unwrap();
        assert!(m.contains("not available"), "{m}");
        assert!(m.contains(&d.dir.display().to_string()), "{m}");
        assert!(m.contains("connect the drive"), "{m}");
        // A save says so too, rather than "not a frame of the dataset".
        let (s, v) = json(
            &d.state,
            Request::post(format!("/v1/review/labels?frame={rel}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&LabelFile::default()).unwrap(),
                ))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["folder_missing"], true, "{v}");
        assert!(
            v["message"].as_str().unwrap().contains("not available"),
            "{v}"
        );
        let (s, v) = json(
            &d.state,
            Request::post("/v1/review/export")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["folder_missing"], true, "{v}");
        // Back again: the frame is there, nothing happened to it.
        std::fs::rename(&moved, &d.dir).unwrap();
        let (s, v) = json(
            &d.state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(v.get("folder_missing").is_none(), "{v}");
        assert_eq!(v["total"], 1, "{v}");
    }

    #[tokio::test]
    async fn a_damaged_queue_line_shows_as_unreadable_and_costs_only_that_line() {
        let d = ds();
        put_frame(&d.dir, "cam", "2026-10-22", "20261022T010203004Z_abcd", 0.9);
        let queue = d.dir.join(layout::REVIEW_QUEUE_FILE);
        let mut bytes = std::fs::read(&queue).unwrap();
        bytes.extend_from_slice(b"\xff\xfe\n");
        std::fs::write(&queue, &bytes).unwrap();
        let (s, v) = json(
            &d.state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["total"].as_u64(), v["unreadable"].as_u64()),
            (Some(1), Some(1))
        );
    }

    #[tokio::test]
    async fn a_bulk_accept_through_the_route_honours_the_hidden_classes() {
        let d = ds_with(|c| c.collect.hidden_classes = vec!["dog".into()]);
        let rel = put_frame(&d.dir, "cam", "2026-10-22", "20261022T010203004Z_abcd", 0.9);
        let frame = Frame::resolve(&d.dir, &rel).unwrap();
        let pick = |label: &str, x: f32| {
            let mut b = crate::collect::labels::bulk::test_support::auto_box(
                label,
                0.9,
                crate::collect::labels::Triage::Accept,
            );
            b.bbox = [x, 2.0, x + 10.0, 40.0];
            b
        };
        let file = LabelFile {
            boxes: vec![pick("person", 1.0), pick("dog", 30.0)],
            ..LabelFile::default()
        };
        labels::write(&frame.labels, &file).unwrap();
        let (s, v) = json(
            &d.state,
            Request::post("/v1/review/accept-auto")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["accepted"], 1, "{v}");
        let l = frame.labels().unwrap().unwrap();
        assert_eq!(
            l.training().map(|b| b.label.as_str()).collect::<Vec<_>>(),
            ["person"]
        );
        assert_eq!(l.boxes[1].reason.as_deref(), Some(bulk::OTHER_CLASS_REASON));
    }

    #[tokio::test]
    async fn a_dataset_folder_nothing_was_collected_into_yet_is_just_empty() {
        // A fresh install: the default dataset folder does not exist yet.
        let root = std::env::temp_dir().join(format!("bop-review-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r));
        let (s, v) = json(
            &state,
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v.get("folder_missing").is_none(), "{v}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
