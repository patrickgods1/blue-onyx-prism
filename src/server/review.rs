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
//!
//! This file holds what the routes share (answers, body and filter checks, the context of the
//! current config), the page, the queue (`next`) and a frame's labels; the other routes are split by
//! concern: [`bulk_routes`] (the bulk tools and the overlap check), [`spots`] (the reviewer's
//! history and the spot memory) and [`files`] (export, import, the folder under review, classes).

mod bulk_routes;
mod files;
mod spots;

pub(super) use bulk_routes::{
    accept_auto, accept_repeats, reject_low, reject_other_class, reject_stacked,
    reject_unwanted_reviewed, restore_reopened, suggest_overlaps,
};
pub(super) use files::{allowed_folder, class_catalog, classes, export, folder, import};
pub(super) use spots::{
    accept_spot, history, reject_spot, spots, spots_moved, spots_parked, spots_retire,
};

use super::{AppState, blocking, encode_uri_component, render};
use crate::collect::CollectSettings;
#[cfg(test)]
use crate::collect::labels::bulk::accept_one;
use crate::collect::labels::queue::Filters;
use crate::collect::labels::{self, Frame, LabelFile, bulk};
use crate::collect::{Bucket, layout};
use askama::Template;
use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, QueryRejection};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use files::known_dirs;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(test)]
use {super::save_config, files::exports_dir};

/// Largest `POST /v1/review/labels` body.
pub(super) const LABELS_BODY_LIMIT: usize = 256 * 1024;
/// Largest body of the other review POST routes.
pub(super) const SMALL_BODY_LIMIT: usize = 16 * 1024;

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

/// A refused request, answered as [`fail`]: a route returning [`Answer`] checks with `?`.
pub(super) struct Refused(Problem);

impl From<Problem> for Refused {
    fn from(p: Problem) -> Self {
        Self(p)
    }
}

impl IntoResponse for Refused {
    fn into_response(self) -> Response {
        let (status, message) = self.0;
        fail(status, message)
    }
}

/// What a route answers: its response, or the [`fail`] answer of a check that refused it.
pub(super) type Answer = Result<Response, Refused>;

/// `{"success": true}` with `fields` (an object), as JSON.
fn success(fields: serde_json::Value) -> Response {
    let mut v = fields;
    if let Some(o) = v.as_object_mut() {
        o.insert("success".into(), true.into());
    }
    Json(v).into_response()
}

/// Why the folder under review `dir` cannot be used, as the page words it; None when it can
/// ([`crate::collect::CollectConfig::review_folder_problem`]: an absent folder is not an empty
/// dataset).
pub(super) fn folder_problem(state: &AppState, dir: &std::path::Path) -> Option<String> {
    state.config_read().collect.review_folder_problem(dir)
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

/// The queue filters of `q` ([`Filters::from_query`]), or 400.
fn filters(q: &BTreeMap<String, String>) -> Result<Filters, Problem> {
    Filters::from_query(q).map_err(|e| (StatusCode::BAD_REQUEST, e))
}

/// The filters of a bulk route's body, `[bucket, scene, camera, label, source]` (None = any), with
/// the `fixed` ones the route sets itself (`status`, `auto`), or 400.
fn bulk_filters(given: [Option<String>; 5], fixed: &[(&str, &str)]) -> Result<Filters, Problem> {
    let keys = ["bucket", "scene", "camera", "label", "source"];
    let mut q: BTreeMap<String, String> = keys
        .into_iter()
        .zip(given)
        .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
        .collect();
    for (k, v) in fixed {
        q.insert((*k).to_string(), (*v).to_string());
    }
    filters(&q)
}

/// The most frames a bulk route changes: `given`, else `default`, within
/// 1..=[`ACCEPT_AUTO_MAX_LIMIT`], or 400.
fn bulk_limit(given: Option<usize>, default: usize) -> Result<usize, Problem> {
    let limit = given.unwrap_or(default);
    if (1..=ACCEPT_AUTO_MAX_LIMIT).contains(&limit) {
        Ok(limit)
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_AUTO_MAX_LIMIT}, got {limit}"),
        ))
    }
}

/// Ends a bulk route's `message` with the `failed` frames it could not change, when there are any:
/// `" {failed} {not_done} (listed); {fix} them by hand."`. The words are the route's own
/// (`"frame(s) not changed"`, `"review"`, ...): every message is pinned as it reads.
pub(super) fn push_failed(message: &mut String, failed: usize, not_done: &str, fix: &str) {
    if failed > 0 {
        message.push_str(&format!(
            " {failed} {not_done} (listed); {fix} them by hand."
        ));
    }
}

/// Ends a bulk route's `message` with the `eligible` frames past its `limit`, when there are any:
/// `" {n} {more} beyond the limit of {limit}; run it again."`, `more` in the route's words
/// (`"more frame(s)"`, `"more eligible"`, `"more"`).
pub(super) fn push_past_limit(message: &mut String, eligible: usize, limit: usize, more: &str) {
    if eligible > limit {
        message.push_str(&format!(
            " {} {more} beyond the limit of {limit}; run it again.",
            eligible - limit
        ));
    }
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

/// `GET /v1/review/next?bucket=&scene=&camera=&label=&status=&source=&auto=&refine_kind=&after=`
/// (`refine_kind` only with `auto=refine`; [`Filters::from_query`]): the most
/// uncertain frame matching the filters (after `after`, the frame shown now, when given; past
/// the last one the first again with `wrapped`), its position, the counts and the cameras and
/// classes for the filter lists. A big folder answers from [`labels::index`], with
/// `indexing: {done, total}` while its rows are still coming in; a small one is walked.
pub(super) async fn next(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Answer {
    let q = query_map(query);
    let mut filters = filters(&q)?;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        filters.wanted = Some((classes, hidden(&state)));
        let after = q.get("after").map(String::as_str).filter(|a| !a.is_empty());
        let found = labels::index::answer(&dir, &filters, after)
            .unwrap_or_else(|| labels::index::Answer::walk(&dir, &filters, after));
        let mut answer = serde_json::json!(found);
        if let Some(fields) = answer.as_object_mut() {
            fields.insert("success".into(), true.into());
        }
        if let (Some(message), Some(fields)) =
            (folder_problem(&state, &dir), answer.as_object_mut())
        {
            fields.insert("folder_missing".into(), true.into());
            fields.insert("message".into(), message.into());
        }
        Json(answer).into_response()
    })
    .await)
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
    size_floor: Option<labels::SizeFloor>,
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
    let live = labels::overlaps::live(&file, Some(&wanted));
    let overlap_live: Vec<usize> = live.iter().map(|l| l.index).collect();
    // The boxes of each live suggestion, indices into `labels.boxes` as served: where two boxes
    // share a position, the page must take the one `live` chose (RR-T9). `overlap_live` stays for
    // an older page.
    let overlap_live_boxes: Vec<serde_json::Value> = live
        .iter()
        .map(|l| serde_json::json!({"index": l.index, "a": l.a, "b": l.b}))
        .collect();
    let refine_live = refine_live(&file, &wanted);
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
        "overlap_live_boxes": overlap_live_boxes,
        "refine_live": refine_live,
        "labels": file,
        "classes": classes,
        "hidden": hidden,
        "size_floor": size_floor,
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

/// The suggestions of the dataset check on `file` a reviewer can act on, as `refine_live` of the
/// labels answer ([`labels::refine::live_json`], which the page's twin is tested against).
fn refine_live(file: &LabelFile, wanted: &bulk::Wanted<'_>) -> Vec<serde_json::Value> {
    labels::refine::live_json(file, Some(wanted))
}

/// `GET /v1/review/labels?frame=<scene>/<camera>/<date>/<stem>.jpg`.
pub(super) async fn labels_get(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
) -> Answer {
    let rel = frame_param(&query_map(query))?;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let size_floor = state.config_read().collect.size_floor;
        match Frame::resolve(&dir, &rel) {
            Some(f) => labels_answer(&f, &classes, &hidden, size_floor, None),
            None => no_frame(&state, &dir),
        }
    })
    .await)
}

/// Message of a save refused because the frame changed since the client read it.
pub(super) const CONFLICT_MESSAGE: &str = "This frame changed since you opened it (an auto-label \
     run or another tab saved it). Reload it to see the new boxes; your changes to it were not \
     saved.";

/// Which of the lists the jobs write a save's body named at all. A key left out (a script or
/// another client that rebuilds the file from the boxes) keeps the stored list; a key present,
/// even `[]`, is the client's list (the page always sends both).
#[derive(Debug, Default, Deserialize)]
struct SentLists {
    #[serde(default)]
    refine: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    overlap_suggestions: Option<serde::de::IgnoredAny>,
}

impl SentLists {
    /// The keys of `body` (a body that is no JSON object names none: it is refused anyway).
    fn of(body: &[u8]) -> Self {
        serde_json::from_slice(body).unwrap_or_default()
    }

    /// `file` with the stored file's lists in place of those the body left out.
    fn keep_stored(&self, file: &mut LabelFile, path: &std::path::Path) {
        if self.refine.is_some() && self.overlap_suggestions.is_some() {
            return;
        }
        let stored = match labels::read(path) {
            Ok(Some(s)) => s,
            Ok(None) => return,
            Err(e) => {
                tracing::debug!(
                    "review save: the stored label file did not read ({e:#}); its suggestions are not kept"
                );
                return;
            }
        };
        if self.refine.is_none() {
            file.refine = stored.refine;
        }
        if self.overlap_suggestions.is_none() {
            file.overlap_suggestions = stored.overlap_suggestions;
        }
    }
}

/// `POST /v1/review/labels?frame=...[&revision=<rev>]` with a label file as JSON: checked
/// ([`labels::normalize`]) and saved atomically; answers like the GET. With `revision` (the one
/// the GET answered; empty = not saved then) the save is refused with 409 and `conflict: true`
/// when the file changed meanwhile; without it, it always overwrites (X-AnyLabeling and other
/// clients). A body without a `refine` or `overlap_suggestions` key keeps the stored file's list
/// ([`SentLists`]): the suggestions and their answers are the jobs' and the reviewer's, not the
/// client's to drop by leaving them out.
pub(super) async fn labels_post(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let q = query_map(query);
    let rel = frame_param(&q)?;
    let expected = q.get("revision").map(|r| r.trim().to_string());
    let sent = body
        .as_ref()
        .map_or_else(|_| SentLists::default(), |b| SentLists::of(b));
    let mut file: LabelFile = json_body(&headers, body)?;
    Ok(blocking(move || {
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
        // Read before the write, outside its lock: with a revision the write is refused when the
        // file changed since; without one the save overwrites whatever came between, as always.
        sent.keep_stored(&mut file, &frame.labels);
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
            state.config_read().collect.size_floor,
            Some(format!("Saved as {}.", status.as_str())),
        )
    })
    .await)
}

/// Default most frames of one bulk accept.
pub(super) const ACCEPT_AUTO_DEFAULT_LIMIT: usize = 500;
/// Most frames of one bulk accept.
pub(super) const ACCEPT_AUTO_MAX_LIMIT: usize = 20_000;
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
        /// The dataset folder takes the review index ([`Way::Index`]).
        _indexed: Option<Indexed>,
    }

    /// How `GET /v1/review/next` answers in a test: the walk over the files, or the review index,
    /// which a folder this small takes only when a test asks ([`labels::index::tests::force`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Way {
        Walk,
        Index,
    }

    impl Way {
        fn ds(self) -> Ds {
            self.ds_with(|_| {})
        }

        fn ds_with(self, edit: impl FnOnce(&mut crate::config::Config)) -> Ds {
            let mut d = ds_with(edit);
            d._indexed = self.index(&d.dir);
            d
        }

        /// `dir` takes the index too, until the guard goes (another folder under review).
        fn index(self, dir: &std::path::Path) -> Option<Indexed> {
            (self == Self::Index).then(|| Indexed::new(dir))
        }
    }

    /// A folder the review index serves whatever its size, until this goes.
    struct Indexed(PathBuf);

    impl Indexed {
        fn new(dir: &std::path::Path) -> Self {
            labels::index::tests::force(dir);
            Self(dir.to_path_buf())
        }
    }

    impl Drop for Indexed {
        fn drop(&mut self) {
            let served = labels::index::tests::served(&self.0);
            labels::index::tests::unforce(&self.0);
            // An indexed variant that the walk answered would prove nothing about the index. A
            // folder that does not exist has no index to answer from.
            if !std::thread::panicking() && self.0.is_dir() {
                assert!(
                    served,
                    "the last /v1/review/next of {} was not answered by the index",
                    self.0.display()
                );
            }
        }
    }

    /// One test on a walked folder and, as `$indexed`, on one the review index serves: the same
    /// assertions hold either way.
    macro_rules! both_ways {
        ($walked:ident, $indexed:ident, $test:ident) => {
            #[tokio::test]
            async fn $walked() {
                $test(Way::Walk).await;
            }

            #[tokio::test]
            async fn $indexed() {
                $test(Way::Index).await;
            }
        };
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
            _indexed: None,
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

    both_ways!(
        the_folder_under_review_can_be_another_folder,
        the_folder_under_review_can_be_another_folder_indexed,
        the_folder_under_review_can_be_another_folder_in
    );

    async fn the_folder_under_review_can_be_another_folder_in(way: Way) {
        let d = way.ds();
        let imported = d.root.join("imported");
        let _imported = way.index(&imported);
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

    both_ways!(
        page_and_queue_and_labels_round_trip,
        page_and_queue_and_labels_round_trip_indexed,
        page_and_queue_and_labels_round_trip_in
    );

    async fn page_and_queue_and_labels_round_trip_in(way: Way) {
        let d = way.ds();
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
        // A request that is not signed in never reaches the admin routes.
        let req = super::super::test_util::anonymous(super::super::test_util::from_peer(
            Request::get("/v1/review/next").body(Body::empty()).unwrap(),
            super::super::test_util::REMOTE,
        ));
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
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

    /// A reviewed frame with one suggestion of the dataset check of every kind (and a dismissed
    /// one), classes person, car and dog: its labels path and file.
    fn refine_frame(d: &Ds) -> (String, PathBuf, LabelFile) {
        use crate::collect::labels::refine::{Answer, Kind, Suggestion};
        let rel = put_frame(&d.dir, "drive", "2026-10-21", "20261021T010203004Z_a", 0.9);
        let path = d
            .dir
            .join("ir/drive/2026-10-21/20261021T010203004Z_a.labels.json");
        let at = |label: &str, bbox: [f32; 4], state: labels::BoxState| labels::LabelBox {
            label: label.into(),
            bbox,
            source: labels::Source::Auto,
            score: Some(0.5),
            state,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: labels::AutoInfo::default(),
        };
        let low = labels::LabelBox {
            reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
            ..at("car", [30.0, 30.0, 50.0, 44.0], labels::BoxState::Rejected)
        };
        let s = |kind: Kind| Suggestion {
            kind,
            target: None,
            target_label: None,
            bbox: None,
            label: None,
            score: Some(0.875),
            why: "a fold model that never saw the frame found it at 0.88".into(),
            spot: None,
            answer: None,
            run: "refine-1".into(),
        };
        let f = LabelFile {
            status: labels::Status::Reviewed,
            reviewed_at: Some("2026-10-22T00:00:00Z".into()),
            boxes: vec![
                low,
                at("person", [4.0, 4.0, 20.0, 40.0], labels::BoxState::Accepted),
                at("car", [40.0, 2.0, 60.0, 20.0], labels::BoxState::Accepted),
            ],
            refine: vec![
                Suggestion {
                    bbox: Some([31.0, 30.0, 50.0, 44.0]),
                    label: Some("Car".into()),
                    ..s(Kind::Add)
                },
                Suggestion {
                    target: Some([40.0, 2.0, 60.0, 20.0]),
                    bbox: Some([41.0, 3.0, 59.0, 19.0]),
                    ..s(Kind::Tighten)
                },
                Suggestion {
                    target: Some([4.0, 4.0, 20.0, 40.0]),
                    target_label: Some("person".into()),
                    score: None,
                    ..s(Kind::Verify)
                },
                Suggestion {
                    target: Some([4.0, 4.0, 20.0, 40.0]),
                    label: Some("dog".into()),
                    spot: Some("drive:gate".into()),
                    ..s(Kind::Relabel)
                },
                Suggestion {
                    bbox: Some([2.0, 30.0, 12.0, 46.0]),
                    label: Some("dog".into()),
                    answer: Some(Answer::Dismissed),
                    ..s(Kind::Add)
                },
            ],
            ..LabelFile::default()
        };
        labels::write(&path, &f).unwrap();
        (rel, path, f)
    }

    fn refine_classes(c: &mut crate::config::Config) {
        c.collect.classes = vec!["person".into(), "car".into(), "dog".into()];
    }

    /// `refine_open` of `rel` in `auto=refine` (any status, every frame), None when not listed.
    async fn refine_open_of(d: &Ds, rel: &str) -> Option<u64> {
        let (s, v) = json(
            &d.state,
            get("/v1/review/next?auto=refine&status=any&source=all"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        (v["item"]["frame"] == rel).then(|| v["item"]["refine_open"].as_u64().unwrap_or(0))
    }

    both_ways!(
        refine_suggestions_are_served_live_and_a_page_answer_saves_as_rust_answers,
        refine_suggestions_are_served_live_and_a_page_answer_saves_as_rust_answers_indexed,
        refine_suggestions_are_served_live_and_a_page_answer_saves_as_rust_answers_in
    );

    async fn refine_suggestions_are_served_live_and_a_page_answer_saves_as_rust_answers_in(
        way: Way,
    ) {
        use crate::collect::labels::refine::{self, Answer};
        let d = way.ds_with(refine_classes);
        let (rel, path, f) = refine_frame(&d);
        let q = encode_uri_component(&rel);
        let classes = d.state.config_read().collect.classes.clone();
        let none: Vec<String> = Vec::new();
        let w = bulk::Wanted {
            classes: &classes,
            hidden: &none,
        };
        // (a) Served live: the four open ones, as `refine_live` builds them.
        let (st, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(st, StatusCode::OK, "{l}");
        assert_eq!(l["refine_live"], serde_json::json!(refine_live(&f, &w)));
        let live = l["refine_live"].as_array().unwrap();
        let brief: Vec<(u64, &str, serde_json::Value, serde_json::Value)> = live
            .iter()
            .map(|x| {
                (
                    x["index"].as_u64().unwrap(),
                    x["kind"].as_str().unwrap(),
                    x["target_box"].clone(),
                    x["replaces"].clone(),
                )
            })
            .collect();
        assert_eq!(
            brief,
            [
                (0, "add", serde_json::Value::Null, serde_json::json!([0])),
                (1, "tighten", serde_json::json!(2), serde_json::json!([])),
                (2, "verify", serde_json::json!(1), serde_json::json!([])),
                (3, "relabel", serde_json::json!(1), serde_json::json!([])),
            ]
        );
        assert_eq!(live[0]["label"], "Car");
        assert_eq!(live[2]["score"], serde_json::Value::Null);
        assert_eq!(live[2]["target_label"], "person");
        assert_eq!(live[3]["spot"], "drive:gate");
        assert_eq!(live[3]["run"], "refine-1");
        assert_eq!(live[1]["why"], f.refine[1].why.as_str());
        assert_eq!(refine_open_of(&d, &rel).await, Some(4));

        // (b) The page accepts the add as `acceptRefine` does: an accepted auto box with reason
        // `refine` in the place of the low-score reject, the class in the list's spelling, the
        // suggestion gone.
        let mut page = l["labels"].clone();
        page["boxes"][0] = serde_json::json!({
            "label": "car",
            "bbox": page["refine"][0]["bbox"].clone(),
            "source": "auto",
            "score": page["refine"][0]["score"].clone(),
            "state": "accepted",
            "reason": "refine",
        });
        page["refine"].as_array_mut().unwrap().remove(0);
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            l["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &page.to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let back = labels::read(&path).unwrap().unwrap();
        let mut want = f.clone();
        assert!(refine::accept(&mut want, 0, Some(&w)));
        let mut want = labels::normalize(want, &classes, 64, 48, "t").unwrap();
        want.reviewed_at.clone_from(&back.reviewed_at);
        assert_eq!(back, want);
        assert_eq!(back.boxes[0].reason.as_deref(), Some(refine::REFINE_REASON));
        // Served again: the indices after the removed one moved down by one.
        let kinds: Vec<(u64, &str)> = v["refine_live"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| (x["index"].as_u64().unwrap(), x["kind"].as_str().unwrap()))
            .collect();
        assert_eq!(kinds, [(0, "tighten"), (1, "verify"), (2, "relabel")]);
        assert_eq!(refine_open_of(&d, &rel).await, Some(3));

        // (c) A dismissal is the answer on the suggestion, kept through the save.
        let mut page = v["labels"].clone();
        page["refine"][1]["answer"] = "dismissed".into();
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            v["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &page.to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let back = labels::read(&path).unwrap().unwrap();
        assert_eq!(back.refine[1].answer, Some(Answer::Dismissed));
        let mut want = labels::read(&path).unwrap().unwrap();
        want.refine[1].answer = None;
        assert!(refine::dismiss(&mut want, 1));
        assert_eq!(back, want);
        assert_eq!(v["labels"]["refine"][1]["answer"], "dismissed");
        let kinds: Vec<u64> = v["refine_live"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x["index"].as_u64().unwrap())
            .collect();
        assert_eq!(kinds, [0, 2]);
        assert_eq!(refine_open_of(&d, &rel).await, Some(2));
        // One kind only: the tighten is there, the dismissed verify is not.
        let (_, n) = json(
            &d.state,
            get("/v1/review/next?auto=refine&refine_kind=verify&status=any&source=all"),
        )
        .await;
        assert_eq!(n["total"], 0, "{n}");
        let (_, n) = json(
            &d.state,
            get("/v1/review/next?auto=refine&refine_kind=tighten&status=any&source=all"),
        )
        .await;
        assert_eq!(
            (n["total"].as_u64(), n["item"]["refine_open"].as_u64()),
            (Some(1), Some(1)),
            "{n}"
        );
        // A kind without auto=refine is refused.
        let (st, n) = json(&d.state, get("/v1/review/next?refine_kind=add")).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{n}");
        assert!(refine_open_of(&d, &rel).await.is_some());
    }

    both_ways!(
        a_check_that_wrote_under_an_open_page_makes_its_save_a_conflict,
        a_check_that_wrote_under_an_open_page_makes_its_save_a_conflict_indexed,
        a_check_that_wrote_under_an_open_page_makes_its_save_a_conflict_in
    );

    async fn a_check_that_wrote_under_an_open_page_makes_its_save_a_conflict_in(way: Way) {
        use crate::collect::labels::refine::{self, Kind, Suggestion};
        let d = way.ds_with(refine_classes);
        let (rel, path, _) = refine_frame(&d);
        let q = encode_uri_component(&rel);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(refine_open_of(&d, &rel).await, Some(4));
        // The check writes under the label write lock while the page is open: suggestions only.
        let found = Suggestion {
            kind: Kind::Add,
            target: None,
            target_label: None,
            bbox: Some([44.0, 26.0, 60.0, 46.0]),
            label: Some("person".into()),
            score: Some(0.7),
            why: "found".into(),
            spot: None,
            answer: None,
            run: "refine-2".into(),
        };
        let before = labels::read(&path).unwrap().unwrap();
        let wrote = labels::update(&path, |old| {
            let mut f = old.unwrap();
            assert!(refine::apply(&mut f, vec![found.clone()], &[Kind::Add]).changed());
            Ok(Some(f))
        })
        .unwrap();
        assert!(wrote);
        let after = labels::read(&path).unwrap().unwrap();
        assert_eq!(after.boxes, before.boxes, "a check writes suggestions only");
        // The page saves what it read: refused, and the check's write stays.
        let mut page = l["labels"].clone();
        page["refine"][2]["answer"] = "kept".into();
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            l["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &page.to_string())).await;
        assert_eq!(st, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["conflict"], true);
        assert_eq!(labels::read(&path).unwrap().unwrap(), after);
        // The new add is live; the old unanswered add, not found again, went.
        assert!(after.refine.contains(&found));
        assert_eq!(refine_open_of(&d, &rel).await, Some(4));
    }

    both_ways!(
        a_save_without_the_suggestion_lists_keeps_them_and_a_sent_list_is_the_clients,
        a_save_without_the_suggestion_lists_keeps_them_and_a_sent_list_is_the_clients_indexed,
        a_save_without_the_suggestion_lists_keeps_them_and_a_sent_list_is_the_clients_in
    );

    async fn a_save_without_the_suggestion_lists_keeps_them_and_a_sent_list_is_the_clients_in(
        way: Way,
    ) {
        use crate::collect::labels::overlaps::{Suggestion, Verdict};
        let d = way.ds_with(refine_classes);
        let (rel, path, mut stored) = refine_frame(&d);
        stored.overlap_suggestions = vec![Suggestion {
            a: [4.0, 4.0, 20.0, 40.0],
            b: [5.0, 5.0, 20.0, 40.0],
            verdict: Verdict::Unclear,
            why: "looked at, no answer".into(),
            kept_score: None,
            other_score: None,
            low_confidence: false,
            weak: false,
        }];
        labels::write(&path, &stored).unwrap();
        let q = encode_uri_component(&rel);
        let without_lists = |v: &serde_json::Value| {
            let mut body = v["labels"].clone();
            let o = body.as_object_mut().unwrap();
            assert!(o.remove("refine").is_some() && o.remove("overlap_suggestions").is_some());
            body
        };
        // (a) A client that rebuilds the file from the boxes, no revision (X-AnyLabeling and
        // scripts): the suggestions, the dismissed one's answer and the overlap answer stay.
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        let uri = format!("/v1/review/labels?frame={q}");
        let (st, v) = json(&d.state, post_json(&uri, &without_lists(&l).to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let back = labels::read(&path).unwrap().unwrap();
        assert_eq!(back.refine, stored.refine);
        assert_eq!(back.overlap_suggestions, stored.overlap_suggestions);
        assert_eq!(refine_open_of(&d, &rel).await, Some(4));
        // (b) The same with a revision.
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            v["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &without_lists(&v).to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let back = labels::read(&path).unwrap().unwrap();
        assert_eq!(back.refine, stored.refine);
        assert_eq!(back.overlap_suggestions, stored.overlap_suggestions);
        // (c) A key that is there is the client's list, even an empty one.
        let mut body = v["labels"].clone();
        body["refine"] = serde_json::json!([]);
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            v["revision"].as_str().unwrap()
        );
        let (st, v) = json(&d.state, post_json(&uri, &body.to_string())).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let back = labels::read(&path).unwrap().unwrap();
        assert!(back.refine.is_empty());
        assert_eq!(back.overlap_suggestions, stored.overlap_suggestions);
        let mut body = v["labels"].clone();
        body["overlap_suggestions"] = serde_json::json!([]);
        let (st, _) = json(
            &d.state,
            post_json(&format!("/v1/review/labels?frame={q}"), &body.to_string()),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert!(
            labels::read(&path)
                .unwrap()
                .unwrap()
                .overlap_suggestions
                .is_empty()
        );
        // A null list is no list: refused, the file unchanged.
        let before = labels::read(&path).unwrap().unwrap();
        let mut body = v["labels"].clone();
        body["refine"] = serde_json::Value::Null;
        let (st, _) = json(
            &d.state,
            post_json(&format!("/v1/review/labels?frame={q}"), &body.to_string()),
        )
        .await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(labels::read(&path).unwrap().unwrap(), before);
        assert_eq!(refine_open_of(&d, &rel).await, None);
    }

    /// The message ends the bulk routes share, in each route's words, byte for byte.
    #[test]
    fn the_bulk_message_ends_keep_each_routes_words() {
        let end = |failed: usize, not_done: &str, fix: &str, eligible: usize, more: &str| {
            let mut m = String::from("Done.");
            push_failed(&mut m, failed, not_done, fix);
            push_past_limit(&mut m, eligible, 10, more);
            m
        };
        assert_eq!(end(0, "not saved", "review", 10, "more"), "Done.");
        assert_eq!(
            end(2, "not accepted", "review", 13, "more eligible"),
            "Done. 2 not accepted (listed); review them by hand. 3 more eligible beyond the \
             limit of 10; run it again."
        );
        assert_eq!(
            end(1, "frame(s) not changed", "fix", 11, "more frame(s)"),
            "Done. 1 frame(s) not changed (listed); fix them by hand. 1 more frame(s) beyond \
             the limit of 10; run it again."
        );
        assert_eq!(
            end(4, "not saved", "review", 20, "more"),
            "Done. 4 not saved (listed); review them by hand. 10 more beyond the limit of 10; \
             run it again."
        );
        let refused = |l: Option<usize>| bulk_limit(l, ACCEPT_AUTO_DEFAULT_LIMIT);
        assert_eq!(refused(None), Ok(ACCEPT_AUTO_DEFAULT_LIMIT));
        assert_eq!(
            refused(Some(ACCEPT_AUTO_MAX_LIMIT)),
            Ok(ACCEPT_AUTO_MAX_LIMIT)
        );
        assert_eq!(
            refused(Some(0)),
            Err((
                StatusCode::BAD_REQUEST,
                "limit: must be between 1 and 20000, got 0".to_string()
            ))
        );
        let f = bulk_filters(
            [None, Some("ir".into()), None, None, None],
            &[("status", "unreviewed")],
        )
        .unwrap();
        assert_eq!(
            f,
            Filters::from_query(&BTreeMap::from([
                ("scene".to_string(), "ir".to_string()),
                ("status".to_string(), "unreviewed".to_string()),
            ]))
            .unwrap()
        );
        assert_eq!(
            bulk_filters([None, Some("dusk".into()), None, None, None], &[]).map(|_| ()),
            Err((
                StatusCode::BAD_REQUEST,
                Filters::from_query(&BTreeMap::from([("scene".to_string(), "dusk".to_string())]))
                    .unwrap_err()
            ))
        );
    }

    /// RR-T9: two boxes at one position, the first relabelled by a reviewer, and a drop keyed to
    /// that position. `overlap_live` names only the suggestion; `overlap_live_boxes` names the boxes
    /// the server chose (the second, never the relabelled one), in the file as served.
    #[tokio::test]
    async fn a_live_suggestion_names_its_boxes_so_a_relabelled_twin_is_not_dropped() {
        use crate::collect::labels::overlaps::{Suggestion, Verdict};
        let d = ds_with(|c| c.collect.classes = vec!["car".into(), "truck".into()]);
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
            state: labels::BoxState::Pending,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: labels::AutoInfo::default(),
        };
        let (keep, inner) = ([0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 60.0, 60.0]);
        let mut relabelled = car(inner);
        relabelled.model_label = Some("truck".into());
        let mut hit = car(inner);
        hit.state = labels::BoxState::Accepted;
        let f = LabelFile {
            boxes: vec![relabelled, hit, car(keep)],
            overlap_suggestions: vec![Suggestion {
                a: keep,
                b: inner,
                verdict: Verdict::DropB,
                why: String::new(),
                kept_score: Some(0.8),
                other_score: Some(0.0),
                low_confidence: false,
                weak: false,
            }],
            ..LabelFile::default()
        };
        labels::write(&path, &f).unwrap();
        let (st, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        assert_eq!(st, StatusCode::OK, "{l}");
        // The old field, for an old page, is unchanged.
        assert_eq!(l["overlap_live"], serde_json::json!([0]));
        assert_eq!(
            l["overlap_live_boxes"],
            serde_json::json!([{"index": 0, "a": 2, "b": 1}])
        );
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

    both_ways!(
        next_wraps_past_the_last_frame,
        next_wraps_past_the_last_frame_indexed,
        next_wraps_past_the_last_frame_in
    );

    async fn next_wraps_past_the_last_frame_in(way: Way) {
        let d = way.ds();
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

    both_ways!(
        bulk_accept_saves_only_the_frames_the_triage_settles,
        bulk_accept_saves_only_the_frames_the_triage_settles_indexed,
        bulk_accept_saves_only_the_frames_the_triage_settles_in
    );

    async fn bulk_accept_saves_only_the_frames_the_triage_settles_in(way: Way) {
        let d = way.ds();
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

    both_ways!(
        repeats_history_spot_edits_and_repeat_only_frames,
        repeats_history_spot_edits_and_repeat_only_frames_indexed,
        repeats_history_spot_edits_and_repeat_only_frames_in
    );

    async fn repeats_history_spot_edits_and_repeat_only_frames_in(way: Way) {
        use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
        use labels::{BoxState, Triage};
        let d = way.ds();
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
        let req = super::super::test_util::anonymous(super::super::test_util::from_peer(
            post_json(uri, r#"{"min_score": 0.9}"#),
            super::super::test_util::REMOTE,
        ));
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        assert_eq!(labels_of(&d.dir, &high).boxes[0].state, BoxState::Pending);
        let big = format!(
            r#"{{"min_score": 0.3, "camera": "{}"}}"#,
            "x".repeat(SMALL_BODY_LIMIT)
        );
        let (s, _) = json(&d.state, post_json(uri, &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    both_ways!(
        reject_stacked_spans_reviewed_frames_and_counts_what_is_left,
        reject_stacked_spans_reviewed_frames_and_counts_what_is_left_indexed,
        reject_stacked_spans_reviewed_frames_and_counts_what_is_left_in
    );

    async fn reject_stacked_spans_reviewed_frames_and_counts_what_is_left_in(way: Way) {
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
        let d = way.ds_with(|c| {
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
        let req = super::super::test_util::anonymous(super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        ));
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
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
        let req = super::super::test_util::anonymous(super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        ));
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
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
        let req = super::super::test_util::anonymous(super::super::test_util::from_peer(
            post_json(uri, "{}"),
            super::super::test_util::REMOTE,
        ));
        let (s, _, _) = call(&d.state, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
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

    both_ways!(
        a_missing_folder_under_review_is_named_not_taken_for_an_empty_dataset,
        a_missing_folder_under_review_is_named_not_taken_for_an_empty_dataset_indexed,
        a_missing_folder_under_review_is_named_not_taken_for_an_empty_dataset_in
    );

    async fn a_missing_folder_under_review_is_named_not_taken_for_an_empty_dataset_in(way: Way) {
        let d = way.ds();
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

    both_ways!(
        a_damaged_queue_line_shows_as_unreadable_and_costs_only_that_line,
        a_damaged_queue_line_shows_as_unreadable_and_costs_only_that_line_indexed,
        a_damaged_queue_line_shows_as_unreadable_and_costs_only_that_line_in
    );

    async fn a_damaged_queue_line_shows_as_unreadable_and_costs_only_that_line_in(way: Way) {
        let d = way.ds();
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
    // ---- the review index behind `GET /v1/review/next` ----------------------------------------

    /// `GET /v1/review/next` with `params`.
    async fn next_of(d: &Ds, params: &[(&str, &str)]) -> serde_json::Value {
        let query = params
            .iter()
            .map(|(k, v)| format!("{k}={}", encode_uri_component(v)))
            .collect::<Vec<_>>()
            .join("&");
        let (s, v) = json(&d.state, get(&format!("/v1/review/next?{query}"))).await;
        assert_eq!(s, StatusCode::OK, "{query}: {v}");
        v
    }

    /// What [`next`] answers for `params` from a walk of the files, never the index.
    fn walked(d: &Ds, params: &[(&str, &str)]) -> serde_json::Value {
        let q: BTreeMap<String, String> = params
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let mut f = Filters::from_query(&q).unwrap();
        let (dir, classes, _) = context(&d.state);
        f.wanted = Some((classes, hidden(&d.state)));
        let after = q.get("after").map(String::as_str).filter(|a| !a.is_empty());
        let mut v = serde_json::json!(labels::index::Answer::walk(&dir, &f, after));
        v["success"] = true.into();
        v
    }

    /// The keys and value types of `v`, not its values (an array by its first element).
    fn shape(v: &serde_json::Value) -> serde_json::Value {
        use serde_json::Value;
        match v {
            Value::Object(m) => {
                Value::Object(m.iter().map(|(k, v)| (k.clone(), shape(v))).collect())
            }
            Value::Array(a) => Value::Array(a.first().map(shape).into_iter().collect()),
            Value::Null => "null".into(),
            Value::Bool(_) => "bool".into(),
            Value::Number(_) => "number".into(),
            Value::String(_) => "string".into(),
        }
    }

    /// Filters a Next is asked with, for answers compared whole with the walk's.
    const VIEWS: [&[(&str, &str)]; 7] = [
        &[],
        &[("status", "any")],
        &[("status", "reviewed")],
        &[("status", "skipped")],
        &[("auto", "accept")],
        &[("auto", "review")],
        &[("camera", "yard"), ("status", "any")],
    ];

    /// Every view of [`VIEWS`], and after its first frame, answered as the walk answers it.
    async fn same_as_the_walk(d: &Ds, when: &str) {
        for view in VIEWS {
            let got = next_of(d, view).await;
            assert_eq!(got, walked(d, view), "{when}, {view:?}");
            if let Some(first) = got["item"]["frame"].as_str() {
                let after = [view, &[("after", first)]].concat();
                assert_eq!(
                    next_of(d, &after).await,
                    walked(d, &after),
                    "{when}, {after:?}"
                );
            }
        }
    }

    /// Twelve queued frames of cameras drive and yard, each with one pending auto car scoring
    /// 0.2..0.75, every third triaged accept.
    fn put_auto_frames(dir: &std::path::Path) -> Vec<String> {
        use crate::collect::labels::bulk::test_support::{auto_box, put_frame_at};
        (0..12u8)
            .map(|i| {
                let triage = if i % 3 == 0 {
                    labels::Triage::Accept
                } else {
                    labels::Triage::Review
                };
                put_frame_at(
                    dir,
                    if i % 2 == 0 { "drive" } else { "yard" },
                    &format!("20261021T0102{i:02}004Z_f{i}"),
                    (64, 48),
                    None,
                    vec![auto_box("car", 0.2 + 0.05 * f32::from(i), triage)],
                )
            })
            .collect()
    }

    /// A save or a bulk action, then a Next: the index answers with what was just written,
    /// exactly as a walk of the files would, and it is the index answering.
    #[tokio::test]
    async fn the_index_answers_a_next_with_what_a_save_or_a_bulk_action_just_wrote() {
        let d = Way::Index.ds();
        let rels = put_auto_frames(&d.dir);
        same_as_the_walk(&d, "built").await;

        // A save through the page's route.
        let shown = next_of(&d, &[]).await;
        let frame = shown["item"]["frame"].as_str().unwrap().to_string();
        let q = encode_uri_component(&frame);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        let mut file = l["labels"].clone();
        file["status"] = "reviewed".into();
        file["boxes"][0]["state"] = "accepted".into();
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            l["revision"].as_str().unwrap()
        );
        let (s, saved) = json(&d.state, post_json(&uri, &file.to_string())).await;
        assert_eq!(s, StatusCode::OK, "{saved}");
        let v = next_of(&d, &[]).await;
        assert_ne!(
            v["item"]["frame"],
            frame.as_str(),
            "the saved frame left the queue at once"
        );
        assert_eq!(v["counts"]["reviewed"], 1, "{v}");
        same_as_the_walk(&d, "after a save").await;

        // A save as skipped.
        let frame = v["item"]["frame"].as_str().unwrap().to_string();
        let q = encode_uri_component(&frame);
        let (_, l) = json(&d.state, get(&format!("/v1/review/labels?frame={q}"))).await;
        let mut file = l["labels"].clone();
        file["status"] = "skipped".into();
        let uri = format!(
            "/v1/review/labels?frame={q}&revision={}",
            l["revision"].as_str().unwrap()
        );
        let (s, saved) = json(&d.state, post_json(&uri, &file.to_string())).await;
        assert_eq!(s, StatusCode::OK, "{saved}");
        assert_eq!(next_of(&d, &[("status", "skipped")]).await["total"], 1);
        same_as_the_walk(&d, "after a skip").await;

        // Bulk actions: reject the low ones, accept what the triage settles.
        let (s, v) = json(
            &d.state,
            post_json(
                "/v1/review/reject-low",
                r#"{"dry_run": false, "min_score": 0.4}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        same_as_the_walk(&d, "after reject-low").await;
        let (s, v) = json(
            &d.state,
            post_json("/v1/review/accept-auto", r#"{"dry_run": false}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["accepted"].as_u64().unwrap() > 0, "{v}");
        same_as_the_walk(&d, "after accept-auto").await;

        // The index answered all along: an edit no writer of this process made waits for the
        // minute's scan, so it is in the walk and not yet in the answer.
        let before = next_of(&d, &[("status", "any")]).await;
        let untouched = rels
            .iter()
            .find(|r| {
                crate::collect::labels::bulk::test_support::labels_of(&d.dir, r).status
                    == labels::Status::Unreviewed
            })
            .expect("a frame still unreviewed");
        let target = Frame::resolve(&d.dir, untouched).unwrap();
        let mut f = target.labels().unwrap().unwrap();
        f.status = labels::Status::Reviewed;
        std::fs::write(&target.labels, serde_json::to_vec(&f).unwrap()).unwrap();
        let after = next_of(&d, &[("status", "any")]).await;
        assert_eq!(after["counts"], before["counts"], "served by the index");
        assert_ne!(
            walked(&d, &[("status", "any")])["counts"],
            before["counts"],
            "the walk sees the direct edit"
        );
    }

    /// While the index is built, the answer says how far it is (`indexing: {done, total}`) and
    /// does not wait for it; once built, `indexing` is gone and the answer is the walk's.
    #[tokio::test]
    async fn indexing_is_in_the_answer_while_the_index_builds_and_gone_after() {
        let d = Way::Index.ds();
        put_auto_frames(&d.dir);
        let hold = labels::index::tests::hold(&d.dir);
        let v = next_of(&d, &[]).await;
        let progress = |v: &serde_json::Value| {
            let ix = v["indexing"]
                .as_object()
                .unwrap_or_else(|| panic!("indexing: {v}"));
            assert_eq!(ix.len(), 2, "{v}");
            (ix["done"].as_u64().unwrap(), ix["total"].as_u64().unwrap())
        };
        let (done, total) = progress(&v);
        assert!(done <= total, "{v}");
        assert!(hold.reached(), "the build stops at the hold");
        let v = next_of(&d, &[("status", "any")]).await;
        let (done, total) = progress(&v);
        assert!(done < 12 && total <= 12, "{v}");
        // A Next during the build moves on, past a frame it does not have yet too.
        let v = next_of(&d, &[("after", "ir/drive/2026-10-21/nothing.jpg")]).await;
        progress(&v);
        hold.release();
        let v = next_of(&d, &[]).await;
        assert!(v.get("indexing").is_none(), "{v}");
        assert_eq!(v, walked(&d, &[]));
        same_as_the_walk(&d, "built after a hold").await;
    }

    /// The answer has the same keys and value types whether the index or the walk gives it, and
    /// `indexing` only while the index is still filling in.
    #[tokio::test]
    async fn the_answer_has_one_shape_with_and_without_the_index() {
        let (walk, index) = (Way::Walk.ds(), Way::Index.ds());
        for d in [&walk, &index] {
            put_auto_frames(&d.dir);
            put_frame(
                &d.dir,
                "yard",
                "2026-10-22",
                "20261022T010203004Z_plain",
                0.52,
            );
        }
        for view in VIEWS {
            let (a, b) = (next_of(&walk, view).await, next_of(&index, view).await);
            assert_eq!(shape(&a), shape(&b), "{view:?}");
            assert!(
                a.get("indexing").is_none() && b.get("indexing").is_none(),
                "{b}"
            );
        }
        // During a build: the same keys, and `indexing` with two numbers.
        let held = Way::Index.ds();
        put_auto_frames(&held.dir);
        let hold = labels::index::tests::hold(&held.dir);
        let v = next_of(&held, &[]).await;
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.retain(|k| k != "indexing");
        let walked_keys: Vec<_> = next_of(&walk, &[])
            .await
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, walked_keys);
        assert_eq!(
            shape(&v["indexing"]),
            serde_json::json!({"done": "number", "total": "number"})
        );
        drop(hold);
    }

    /// A frame retention (or anyone) deleted behind the index's back is never the frame shown
    /// nor one of the frames the page loads ahead.
    #[tokio::test]
    async fn a_frame_deleted_from_disk_is_never_shown_or_loaded_ahead() {
        let d = Way::Index.ds();
        for i in 0..8u8 {
            put_frame(
                &d.dir,
                if i % 2 == 0 { "drive" } else { "yard" },
                "2026-10-22",
                &format!("20261022T0102{i:02}004Z_g{i}"),
                0.5 + 0.05 * f32::from(i),
            );
        }
        let v = next_of(&d, &[]).await;
        let first = v["item"]["frame"].as_str().unwrap().to_string();
        let ahead = v["upcoming"][1].as_str().unwrap().to_string();
        // Gone as retention takes a frame: image and sidecar, no label file to tell the index.
        for rel in [&first, &ahead] {
            let jpg = d.dir.join(rel);
            std::fs::remove_file(&jpg).unwrap();
            std::fs::remove_file(jpg.with_extension("json")).unwrap();
        }
        for after in [None, Some(first.as_str()), Some(ahead.as_str())] {
            let params: Vec<(&str, &str)> = after.map(|a| ("after", a)).into_iter().collect();
            let v = next_of(&d, &params).await;
            let shown: Vec<&str> = std::iter::once(&v["item"]["frame"])
                .chain(v["upcoming"].as_array().unwrap())
                .filter_map(serde_json::Value::as_str)
                .collect();
            assert!(!shown.is_empty(), "{v}");
            for rel in shown {
                assert!(rel != first && rel != ahead, "{rel} is gone: {v}");
                assert!(Frame::resolve(&d.dir, rel).is_some(), "{rel}: {v}");
            }
            assert_eq!(v, walked(&d, &params), "{after:?}");
        }
    }
}
