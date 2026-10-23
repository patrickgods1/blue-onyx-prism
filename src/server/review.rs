//! The Review page (`/review`, admin only) and its JSON API over the dataset collector's
//! frames ([`crate::collect::labels`]): `GET /v1/review/next` (the uncertainty-sorted queue
//! with filters and counts), `GET`/`POST /v1/review/labels?frame=` (a frame's label file, or
//! its pre-fill from the sidecar), `POST /v1/review/export` (YOLO / COCO / X-AnyLabeling /
//! benchmark set), `POST /v1/review/import` (X-AnyLabeling files) and `POST
//! /v1/review/classes` (`collect.classes`). Every frame reference goes through
//! [`Frame::resolve`] (404 otherwise); POST bodies are JSON (415 otherwise) with per-route size
//! caps (413); the admin guard and the cross-site check of the router apply.

use super::{AppState, blocking, encode_uri_component, render, save_config};
use crate::collect::CollectSettings;
use crate::collect::labels::export::{self, BenchSplit, ExportRequest, Format, SplitConfig};
use crate::collect::labels::queue::{self, Filters};
use crate::collect::labels::{self, Frame, LabelFile, xany};
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

/// Dataset folder, classes and data root of the current config.
fn context(state: &AppState) -> (PathBuf, Vec<String>, PathBuf) {
    let cfg = state.config_read();
    (
        CollectSettings::for_config(&cfg).dir,
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
}

/// `GET /review`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (dir, classes, root) = context(&state);
        render(&ReviewTemplate {
            nav: "review",
            version: crate::VERSION,
            dir: dir.display().to_string(),
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

/// `GET /v1/review/next?bucket=&scene=&camera=&status=&source=&after=`: the most uncertain
/// frame matching the filters (after `after`, the frame shown now, when given), its position,
/// and the counts.
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
            "item": found.map(|(_, i)| i),
            "position": found.map(|(p, _)| p + 1),
            "total": built.items.len(),
            "counts": built.counts,
            "cameras": built.cameras,
            "unreadable": built.unreadable,
            "upcoming": built.items.iter().skip(found.map_or(0, |(p, _)| p + 1)).take(5).map(|i| &i.frame).collect::<Vec<_>>(),
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

/// The labels answer for `frame` (saved, else pre-filled).
fn labels_answer(frame: &Frame, classes: &[String], message: Option<String>) -> Response {
    let rec = match frame.record() {
        Ok(r) => r,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let saved = match frame.labels() {
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

/// `POST /v1/review/labels?frame=...` with a label file as JSON: checked
/// ([`labels::normalize`]) and saved atomically; answers like the GET.
pub(super) async fn labels_post(
    State(state): State<Arc<AppState>>,
    query: Result<Query<BTreeMap<String, String>>, QueryRejection>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let rel = match frame_param(&query_map(query)) {
        Ok(r) => r,
        Err((s, m)) => return fail(s, m),
    };
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
        if let Err(e) = labels::write(&frame.labels, &f) {
            return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("Not saved: {e:#}"));
        }
        tracing::debug!(frame = %rel, status = status.as_str(), boxes = f.boxes.len(), "review labels saved");
        labels_answer(&frame, &classes, Some(format!("Saved as {}.", status.as_str())))
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
/// holdout_fraction?, benchmark_split?, include_unreviewed?}`.
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
        match export::run(&dir, &classes, &req) {
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
struct ClassesBody {
    classes: Vec<String>,
}

/// `POST /v1/review/classes` with `{classes: [...]}`: saves `collect.classes`.
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
        let result = save_config(&state, "review page", |c| {
            c.collect.classes = classes.clone();
            c.collect.validate(&c.data_root())
        });
        match result {
            Ok(msg) => Json(serde_json::json!({
                "success": true,
                "message": msg,
                "classes": classes,
                "classes_version": labels::classes_version(&classes),
            }))
            .into_response(),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
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
        assert!(page.contains("raccoon"));

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
    }
}
