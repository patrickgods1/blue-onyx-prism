//! The Review page (`/review`, admin only) and its JSON API over the dataset collector's
//! frames ([`crate::collect::labels`]): `GET /v1/review/next` (the uncertainty-sorted queue
//! with filters and counts), `GET`/`POST /v1/review/labels?frame=` (a frame's label file, or
//! its pre-fill from the sidecar, with its revision; a save naming a stale revision is refused
//! with 409), `POST /v1/review/accept-auto` (bulk accept of frames the auto-labeler's triage
//! settles), `POST /v1/review/reject-low` (bulk reject of the pending auto boxes scoring
//! below a minimum, `reason: low_score`), `POST /v1/review/export` (YOLO / COCO / X-AnyLabeling / benchmark set), `POST
//! /v1/review/import` (X-AnyLabeling files), `POST /v1/review/classes` (`collect.classes`
//! and `collect.hidden_classes`) and `GET /v1/review/class-catalog` (the class picker's
//! catalog, [`crate::training::autolabel::catalog`]).
//! Every frame reference goes through
//! [`Frame::resolve`] (404 otherwise); POST bodies are JSON (415 otherwise) with per-route size
//! caps (413); the admin guard and the cross-site check of the router apply.

use super::{AppState, blocking, encode_uri_component, render, save_config};
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
/// Exports go under `<data root>/training/exports/<UTC time>` unless `out_dir` is given.
pub(crate) const EXPORTS_DIR: [&str; 2] = ["training", "exports"];

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

fn now() -> String {
    crate::collect::utc_rfc3339(std::time::SystemTime::now())
}

#[derive(Template)]
#[template(path = "review.html")]
struct ReviewTemplate {
    nav: &'static str,
    version: &'static str,
    dir: String,
    classes: Vec<String>,
    classes_json: String,
    classes_text: String,
    buckets: Vec<&'static str>,
    scenes: Vec<&'static str>,
    exports_dir: String,
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

/// `GET /review`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (dir, classes, root) = context(&state);
        let dataset = CollectSettings::for_config(&state.config_read()).dir;
        render(&ReviewTemplate {
            nav: "review",
            version: crate::VERSION,
            dir: dir.display().to_string(),
            custom_dir: dir != dataset,
            known_dirs: known_dirs(&state, &dataset),
            dataset_dir: dataset.display().to_string(),
            classes_json: serde_json::to_string(&classes).unwrap_or_else(|_| "[]".into()),
            classes_text: classes.join(", "),
            classes,
            buckets: Bucket::ALL.iter().map(|b| b.as_str()).collect(),
            scenes: layout::SCENES.to_vec(),
            exports_dir: EXPORTS_DIR
                .iter()
                .fold(root, |p, c| p.join(c))
                .display()
                .to_string(),
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
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    blocking(move || {
        let (dir, _, _) = context(&state);
        let built = queue::build(&dir, &filters);
        let after = q.get("after").map(String::as_str).filter(|a| !a.is_empty());
        let found = queue::next_after(&built.items, after);
        Json(serde_json::json!({
            "success": true,
            "item": found.map(|(_, i, _)| i),
            "position": found.map(|(p, _, _)| p + 1),
            "wrapped": found.is_some_and(|(_, _, w)| w),
            "total": built.items.len(),
            "counts": built.counts,
            "cameras": built.cameras,
            "labels": built.labels,
            "unreadable": built.unreadable,
            "upcoming": built.items.iter().skip(found.map_or(0, |(p, _, _)| p + 1)).take(5).map(|i| &i.frame).collect::<Vec<_>>(),
        }))
        .into_response()
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
fn labels_answer(frame: &Frame, classes: &[String], message: Option<String>) -> Response {
    let rec = match frame.record() {
        Ok(r) => r,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let (saved, revision) = match labels::read_with_revision(&frame.labels) {
        Ok(l) => l,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let is_saved = saved.is_some();
    let file = saved.unwrap_or_else(|| labels::prefill(&rec));
    let version = labels::classes_version(classes);
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
        "labels": file,
        "classes": classes,
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
        match Frame::resolve(&dir, &rel) {
            Some(f) => labels_answer(&f, &classes, None),
            None => not_found(),
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
        let Some(frame) = Frame::resolve(&dir, &rel) else {
            return not_found();
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
        labels_answer(&frame, &classes, Some(format!("Saved as {}.", status.as_str())))
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
    let filters = match Filters::from_query(&q) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let dry_run = b.dry_run;
    blocking(move || {
        let (dir, classes, _) = context(&state);
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
        let filter = match class_filter(b.labels, b.include_empty, &classes) {
            Ok(f) => f,
            Err(e) => return fail(StatusCode::BAD_REQUEST, e),
        };
        let out_dir = match b.out_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            None => EXPORTS_DIR
                .iter()
                .fold(root.clone(), |p, c| p.join(c))
                .join(crate::collect::utc_file_ts(std::time::SystemTime::now())),
            Some(raw) => match crate::config::validate_dir_setting("out_dir", raw, &root) {
                Ok(Some(p)) => p,
                Ok(None) => return fail(StatusCode::BAD_REQUEST, "out_dir: empty"),
                Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
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
        let src = match crate::config::validate_dir_setting("dir", &b.dir, &root) {
            Ok(Some(p)) if p.is_dir() => p,
            Ok(_) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("dir: '{}' is not a folder", b.dir),
                );
            }
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        };
        match xany::import_dir(&dir, &src, &classes, b.treat_unchecked_as_reviewed, &now()) {
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
/// browsable folder or one picked in the native dialog ([`super::browse::allowed_path`]).
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
            let p = match super::browse::allowed_path(&config, raw) {
                Ok(p) => p,
                Err(e) => {
                    let (s, m) = e.into_parts();
                    return fail(s, format!("Folder not changed: {m}"));
                }
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

/// `POST /v1/review/classes` with `{classes: [...], hidden?: [...]}`: saves `collect.classes`
/// and, when given, `collect.hidden_classes` (each must be listed); hidden names no longer
/// listed are dropped. Answers both.
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
        let result = save_config(&state, "review page", |c| {
            c.collect.classes = classes.clone();
            match &hidden_req {
                Some(h) => c.collect.hidden_classes = h.clone(),
                None => c.collect.prune_hidden(),
            }
            c.collect.validate(&c.data_root())?;
            hidden = c.collect.hidden_classes.clone();
            Ok(())
        });
        match result {
            Ok(msg) => Json(serde_json::json!({
                "success": true,
                "message": msg,
                "classes": classes,
                "hidden": hidden,
                "classes_version": labels::classes_version(&classes),
            }))
            .into_response(),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
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
        let root = std::env::temp_dir().join(format!("bop-review-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let (d, r) = (root.join("dataset"), root.clone());
        let state = test_state_with(move |c| {
            c.download_dir = Some(r);
            c.collect.dir = Some(d);
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
        let (_, page) = get_text(&d.state, "/review").await;
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
        let _ = std::fs::remove_dir_all(&outside);
        // Empty = back to the dataset folder.
        let (s, _) = json(&d.state, post_json("/v1/review/folder", r#"{"dir": ""}"#)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(d.state.config_read().collect.review_dir, None);
        assert_eq!(total(d.state.clone()).await, 0);
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
        let (s, page) = get_text(&d.state, "/review").await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"active\" aria-current=\"page\">Review</a>"),
            "nav"
        );
        assert!(page.contains("/static/review.js"));
        // Browse... on the export folder and the X-AnyLabeling folder (native picker, else the
        // in-page browser of this dialog).
        assert!(page.contains("data-browse=\"folder\" data-target=\"out_dir\""));
        assert!(page.contains("data-browse=\"folder\" data-target=\"dir\""));
        assert!(page.contains("<dialog id=\"browse-dialog\">"));
        assert!(page.contains("/static/browse.js?v="));
        assert!(page.contains("raccoon"));
        assert!(
            page.contains("name=\"label\" id=\"rv-label\""),
            "class filter"
        );
        assert!(
            page.contains("Only frames with these classes"),
            "export class filter"
        );
        assert!(
            page.contains("name=\"include_empty\" checked"),
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
            post_json("/v1/review/export", r#"{"formats": ["yolo", "coco", "benchmark", "xanylabeling"], "holdout_cameras": ["yard"]}"#),
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

        // A second export into the same, now non-empty folder is refused.
        let body = serde_json::json!({"format": "coco", "out_dir": out}).to_string();
        let (s, v) = json(&d.state, post_json("/v1/review/export", &body)).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(v["message"].as_str().unwrap().contains("not empty"));
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
        let mut f = labels::prefill(&frame.record().unwrap());
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
}
