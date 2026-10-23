//! The Dataset page (`/dataset`, admin only) of the dataset collector ([`crate::collect`]):
//! the collection and per-bucket toggles and limits (saved with [`save_config`], applied
//! live), counts by scene x bucket and per camera, disk use, the newest frames per bucket and
//! "Send to review" (appends to `<dir>/review-queue.jsonl`). Frames and sidecars are served
//! by `/dataset/image` and `/dataset/sidecar` only for paths that pass
//! [`crate::collect::layout::resolve`] (no traversal, no symlinks). `GET /v1/dataset` is the
//! same summary as JSON. Plain HTML forms: the page needs no script.

use super::{AppState, blocking, encode_uri_component, render, save_config};
use crate::collect::layout::{self, REVIEW_QUEUE_FILE, SCENES};
use crate::collect::{Bucket, CollectConfig, CollectSettings, Status};
use askama::Template;
use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;

const MB: f64 = 1024.0 * 1024.0;

/// Decode an `application/x-www-form-urlencoded` body into its fields in order (repeated keys
/// kept, which the page's frame checkboxes need).
pub(super) fn parse_form(body: &[u8]) -> Vec<(String, String)> {
    fn decode(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while let Some(&c) = b.get(i) {
            match c {
                b'+' => out.push(b' '),
                b'%' => {
                    let hex = b
                        .get(i + 1..i + 3)
                        .and_then(|h| std::str::from_utf8(h).ok())
                        .and_then(|h| u8::from_str_radix(h, 16).ok());
                    match hex {
                        Some(v) => {
                            out.push(v);
                            i += 2;
                        }
                        None => out.push(b'%'),
                    }
                }
                c => out.push(c),
            }
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    String::from_utf8_lossy(body)
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| match p.split_once('=') {
            Some((k, v)) => (decode(k), decode(v)),
            None => (decode(p), String::new()),
        })
        .collect()
}

fn field<'a>(fields: &'a [(String, String)], key: &str) -> Option<&'a str> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.trim())
}

/// Apply the page's settings form to `c`: checkboxes are on when present; a number field that
/// is missing or empty keeps its value. Errors name the field.
pub(super) fn apply_form(c: &mut CollectConfig, fields: &[(String, String)]) -> anyhow::Result<()> {
    use anyhow::Context;
    c.enabled = field(fields, "enabled").is_some();
    for b in Bucket::ALL {
        let key = b.as_str();
        let bc = c.buckets.get_mut(b);
        bc.enabled = field(fields, &format!("{key}.enabled")).is_some();
        if let Some(v) = field(fields, &format!("{key}.rate")).filter(|v| !v.is_empty()) {
            bc.rate = v
                .parse()
                .with_context(|| format!("{key} rate: '{v}' is not a number between 0 and 1"))?;
        }
        if let Some(v) = field(fields, &format!("{key}.interval")).filter(|v| !v.is_empty()) {
            bc.per_camera_min_interval_s = v.parse().with_context(|| {
                format!("{key} interval: '{v}' is not a whole number of seconds")
            })?;
        }
    }
    if let Some(v) = field(fields, "near_threshold.margin").filter(|v| !v.is_empty()) {
        c.buckets.near_threshold.margin = Some(
            v.parse()
                .with_context(|| format!("near-threshold margin: '{v}' is not a number"))?,
        );
    }
    if let Some(v) = field(fields, "max_gb").filter(|v| !v.is_empty()) {
        c.max_gb = v
            .parse()
            .with_context(|| format!("max size: '{v}' is not a number of GB"))?;
    }
    if let Some(v) = field(fields, "max_age_days").filter(|v| !v.is_empty()) {
        c.max_age_days = v
            .parse()
            .with_context(|| format!("max age: '{v}' is not a whole number of days"))?;
    }
    if let Some(v) = field(fields, "dedup_hamming").filter(|v| !v.is_empty()) {
        c.dedup_hamming = v
            .parse()
            .with_context(|| format!("dedup distance: '{v}' is not a whole number"))?;
    }
    if let Some(v) = field(fields, "dir") {
        c.dir = (!v.is_empty()).then(|| v.into());
    }
    Ok(())
}

struct BucketRow {
    key: &'static str,
    describe: &'static str,
    enabled: bool,
    rate: String,
    interval: u32,
    saved: u64,
}

struct SceneRow {
    scene: &'static str,
    counts: Vec<u64>,
}

struct CameraRow {
    name: String,
    frames: u64,
    mb: String,
    newest: String,
}

struct Thumb {
    rel: String,
    image_url: String,
    sidecar_url: String,
    caption: String,
}

struct ThumbGroup {
    key: &'static str,
    describe: &'static str,
    thumbs: Vec<Thumb>,
}

#[derive(Template)]
#[template(path = "dataset.html")]
struct DatasetTemplate {
    nav: &'static str,
    version: &'static str,
    message: Option<String>,
    error: Option<String>,
    enabled: bool,
    dir: String,
    dir_setting: String,
    frames: u64,
    used_mb: String,
    max_gb: String,
    max_age_days: u32,
    dedup_hamming: u32,
    near_margin: String,
    disabled: Option<String>,
    queued: usize,
    bucket_names: Vec<&'static str>,
    buckets: Vec<BucketRow>,
    scenes: Vec<SceneRow>,
    cameras: Vec<CameraRow>,
    groups: Vec<ThumbGroup>,
}

fn clean(x: f32) -> String {
    crate::config::clean_f32(x).to_string()
}

/// The page for the config as it is now (call off the async runtime: it may scan the folder).
fn view(state: &AppState, message: Option<String>, error: Option<String>) -> DatasetTemplate {
    let (c, settings) = {
        let cfg = state.config_read();
        (cfg.collect.clone(), CollectSettings::for_config(&cfg))
    };
    let Status {
        summary, disabled, ..
    } = crate::collect::global().status_of(&settings.dir);
    let buckets = Bucket::ALL
        .into_iter()
        .map(|b| {
            let bc = c.buckets.get(b);
            BucketRow {
                key: b.as_str(),
                describe: b.describe(),
                enabled: bc.enabled,
                rate: clean(bc.rate),
                interval: bc.per_camera_min_interval_s,
                saved: SCENES.iter().map(|s| summary.count(s, b)).sum(),
            }
        })
        .collect();
    let scenes = SCENES
        .into_iter()
        .map(|s| SceneRow {
            scene: s,
            counts: Bucket::ALL.iter().map(|b| summary.count(s, *b)).collect(),
        })
        .collect();
    let cameras = summary
        .by_camera
        .iter()
        .map(|(name, t)| CameraRow {
            name: name.clone(),
            frames: t.frames,
            mb: format!("{:.1}", t.bytes as f64 / MB),
            newest: t.newest.clone(),
        })
        .collect();
    let groups = Bucket::ALL
        .into_iter()
        .filter_map(|b| {
            let list = summary.newest.get(&b)?;
            let thumbs: Vec<Thumb> = list
                .iter()
                .map(|f| {
                    let q = encode_uri_component(&f.rel);
                    let labels = if f.labels.is_empty() {
                        "no detections".to_string()
                    } else {
                        f.labels.join(", ")
                    };
                    Thumb {
                        rel: f.rel.clone(),
                        image_url: format!("/dataset/image?path={q}"),
                        sidecar_url: format!("/dataset/sidecar?path={q}"),
                        caption: format!(
                            "{} \u{b7} {} \u{b7} {} \u{b7} {labels}",
                            f.timestamp, f.camera, f.scene
                        ),
                    }
                })
                .collect();
            (!thumbs.is_empty()).then(|| ThumbGroup {
                key: b.as_str(),
                describe: b.describe(),
                thumbs,
            })
        })
        .collect();
    DatasetTemplate {
        nav: "dataset",
        version: crate::VERSION,
        message,
        error,
        enabled: c.enabled,
        dir: settings.dir.display().to_string(),
        dir_setting: c
            .dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_default(),
        frames: summary.frames,
        used_mb: format!("{:.1}", summary.bytes as f64 / MB),
        max_gb: clean(c.max_gb),
        max_age_days: c.max_age_days,
        dedup_hamming: c.dedup_hamming,
        near_margin: clean(c.buckets.near_margin()),
        disabled,
        queued: layout::read_review(&settings.dir.join(REVIEW_QUEUE_FILE)).len(),
        bucket_names: Bucket::ALL.iter().map(|b| b.as_str()).collect(),
        buckets,
        scenes,
        cameras,
        groups,
    }
}

/// `GET /dataset`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, None, None))).await
}

/// `POST /dataset/settings`: the toggles and limits.
pub(super) async fn settings(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let fields = parse_form(&body);
        let result = save_config(&state, "dataset page", |c| {
            apply_form(&mut c.collect, &fields)?;
            c.collect.validate(&c.data_root())
        });
        match result {
            Ok(msg) => render(&view(&state, Some(msg), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not saved: {e:#}")))),
        }
    })
    .await
}

/// `POST /dataset/review`: queue the checked frames (`frame=<rel>`, repeated) for review.
pub(super) async fn review(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let rels: Vec<String> = parse_form(&body)
            .into_iter()
            .filter(|(k, _)| k == "frame")
            .map(|(_, v)| v)
            .collect();
        let dir = CollectSettings::for_config(&state.config_read()).dir;
        if rels.is_empty() {
            return render(&view(
                &state,
                None,
                Some("No frame selected: tick the frames to send to review first.".into()),
            ));
        }
        match layout::append_review(&dir, &rels) {
            Ok(n) => render(&view(
                &state,
                Some(format!(
                    "Queued {n} frame(s) for review ({} already queued) in {}.",
                    rels.len().saturating_sub(n),
                    dir.join(REVIEW_QUEUE_FILE).display()
                )),
                None,
            )),
            Err(e) => render(&view(&state, None, Some(format!("Nothing queued: {e:#}")))),
        }
    })
    .await
}

/// `POST /dataset/resume`: re-arm the collector after a disk stop.
pub(super) async fn resume(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        crate::collect::global().resume();
        render(&view(
            &state,
            Some("Collection resumed; it stops again if free space is still low.".into()),
            None,
        ))
    })
    .await
}

fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "success": false, "message": "not a frame of the dataset" })),
    )
        .into_response()
}

/// A frame's file `ext` for `?path=<scene>/<camera>/<date>/<stem>.jpg`.
async fn serve(
    state: Arc<AppState>,
    q: Option<HashMap<String, String>>,
    ext: &'static str,
) -> Response {
    let Some(rel) = q.and_then(|q| q.get("path").cloned()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "message": "give `path`" })),
        )
            .into_response();
    };
    blocking(move || {
        let dir = CollectSettings::for_config(&state.config_read()).dir;
        let Some(r) = layout::parse_rel(&rel).filter(|_| rel.ends_with(".jpg")) else {
            return not_found();
        };
        let wanted = if ext == "jpg" { r.jpg() } else { r.json() };
        let Some(path) = layout::resolve(&dir, &wanted, ext) else {
            return not_found();
        };
        match std::fs::read(&path) {
            Ok(bytes) => {
                let ctype = if ext == "jpg" {
                    "image/jpeg"
                } else {
                    "application/json"
                };
                (
                    [
                        (header::CONTENT_TYPE, ctype),
                        (header::CACHE_CONTROL, "private, max-age=600"),
                    ],
                    bytes,
                )
                    .into_response()
            }
            Err(_) => not_found(),
        }
    })
    .await
}

/// `GET /dataset/image?path=...`.
pub(super) async fn image(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    serve(state, query.ok().map(|Query(q)| q), "jpg").await
}

/// `GET /dataset/sidecar?path=...` (the frame's `.jpg` path; answers its `.json`).
pub(super) async fn sidecar(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    serve(state, query.ok().map(|Query(q)| q), "json").await
}

/// `GET /v1/dataset`: the settings, the summary and the disk stop reason.
pub(super) async fn json(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (c, settings) = {
            let cfg = state.config_read();
            (cfg.collect.clone(), CollectSettings::for_config(&cfg))
        };
        let s = crate::collect::global().status_of(&settings.dir);
        let queued = layout::read_review(&settings.dir.join(REVIEW_QUEUE_FILE)).len();
        Json(serde_json::json!({
            "success": true,
            "enabled": c.enabled,
            "dir": settings.dir,
            "settings": c,
            "disabled": s.disabled,
            "review_queue": queued,
            "summary": s.summary,
        }))
        .into_response()
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, get_text, test_state_with};
    use super::*;
    use axum::body::Body;
    use axum::http::Request;

    #[test]
    fn form_parsing_keeps_repeated_keys_and_decodes() {
        let f = parse_form(b"frame=ir%2Fa%2F2026-10-22%2Fx.jpg&frame=b+c&enabled=on&bad=%zz&k");
        assert_eq!(f[0], ("frame".into(), "ir/a/2026-10-22/x.jpg".into()));
        assert_eq!(f[1].1, "b c");
        assert_eq!(f[3].1, "%zz");
        assert_eq!(f[4], ("k".into(), String::new()));
    }

    #[test]
    fn settings_form_toggles_and_numbers() {
        let mut c = CollectConfig::default();
        let f = parse_form(
            b"enabled=on&positive.enabled=on&positive.rate=0.5&positive.interval=60&empty.rate=&max_gb=2&near_threshold.enabled=on&near_threshold.margin=0.05&dir=",
        );
        apply_form(&mut c, &f).unwrap();
        assert!(c.enabled);
        assert!(c.buckets.positive.enabled && !c.buckets.empty.enabled);
        assert_eq!(c.buckets.positive.rate, 0.5);
        assert_eq!(c.buckets.positive.per_camera_min_interval_s, 60);
        assert_eq!(c.buckets.empty.rate, 1.0, "an empty field keeps the value");
        assert_eq!(c.buckets.empty.per_camera_min_interval_s, 600);
        assert_eq!(c.max_gb, 2.0);
        assert_eq!(c.buckets.near_margin(), 0.05);
        assert_eq!(c.dir, None);
        let e = apply_form(&mut c, &parse_form(b"positive.rate=lots")).unwrap_err();
        assert!(e.to_string().contains("positive rate"), "{e}");
    }

    fn dataset_state() -> (Arc<AppState>, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("bop-ds-{}", uuid::Uuid::new_v4()));
        let d = dir.clone();
        let state = test_state_with(move |c| {
            c.collect.dir = Some(d);
        });
        (state, dir)
    }

    /// One frame (image + sidecar + index line) at `rel_day/stem`.
    fn put_frame(dir: &std::path::Path, rel_day: &str, stem: &str) -> String {
        let day = dir.join(rel_day);
        std::fs::create_dir_all(&day).unwrap();
        std::fs::write(day.join(format!("{stem}.jpg")), b"\xFF\xD8jpeg").unwrap();
        let mut r = crate::collect::record::empty_record();
        r.id = stem.into();
        r.buckets = vec![Bucket::Suppressed];
        r.scene = "ir".into();
        r.camera = "cam".into();
        r.timestamp = "2026-10-22T01:02:03.004Z".into();
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let line = crate::collect::record::IndexLine::of(&r, 10);
        layout::append_line(
            &day.join(layout::INDEX_FILE),
            &serde_json::to_string(&line).unwrap(),
        )
        .unwrap();
        format!("{rel_day}/{stem}.jpg")
    }

    fn post(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn page_shows_counts_and_thumbnails_and_serves_only_dataset_frames() {
        let (state, dir) = dataset_state();
        let rel = put_frame(&dir, "ir/cam/2026-10-22", "20261022T010203004Z_abcd");
        std::fs::write(dir.join("secret.jpg"), b"outside the layout").unwrap();
        let (s, page) = get_text(&state, "/dataset").await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"active\" aria-current=\"page\">Dataset</a>"),
            "nav"
        );
        assert!(
            page.contains("They contain whatever your cameras see"),
            "privacy note"
        );
        assert!(
            page.contains(&format!("value=\"{rel}\"")),
            "thumbnail checkbox"
        );
        assert!(
            page.contains(
                "/dataset/image?path=ir%2Fcam%2F2026-10-22%2F20261022T010203004Z_abcd.jpg"
            )
        );
        let (s, _, body) = call(
            &state,
            Request::get(format!(
                "/dataset/image?path={}",
                encode_uri_component(&rel)
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body, b"\xFF\xD8jpeg");
        let (s, _, body) = call(
            &state,
            Request::get(format!(
                "/dataset/sidecar?path={}",
                encode_uri_component(&rel)
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8_lossy(&body).contains("\"version\":1"));
        for bad in [
            "secret.jpg",
            "..%2Fsecret.jpg",
            "ir%2Fcam%2F2026-10-22%2F..%2F..%2F..%2Fsecret.jpg",
            "ir%2Fcam%2F2026-10-22%2F20261022T010203004Z_abcd.json",
            "%2Fetc%2Fpasswd",
            "ir%2Fcam%2F2026-10-22%2Fmissing.jpg",
        ] {
            let (s, _, _) = call(
                &state,
                Request::get(format!("/dataset/image?path={bad}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{bad}");
        }
        let (s, json) = get_text(&state, "/v1/dataset").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["summary"]["frames"], 1);
        assert_eq!(v["summary"]["by_scene"]["ir"]["suppressed"], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn send_to_review_appends_once_and_refuses_bad_paths() {
        let (state, dir) = dataset_state();
        let rel = put_frame(&dir, "ir/cam/2026-10-22", "20261022T010203004Z_abcd");
        let body = format!("frame={}", encode_uri_component(&rel));
        let (s, _, page) = call(&state, post("/dataset/review", &body)).await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8_lossy(&page).contains("Queued 1 frame(s)"));
        let (_, _, page) = call(&state, post("/dataset/review", &body)).await;
        assert!(
            String::from_utf8_lossy(&page)
                .contains("Queued 0 frame(s) for review (1 already queued)")
        );
        let q = layout::read_review(&dir.join(REVIEW_QUEUE_FILE));
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].frame, rel);
        assert_eq!(q[0].buckets, vec![Bucket::Suppressed]);
        let (_, _, page) = call(&state, post("/dataset/review", "frame=..%2F..%2Fcfg.json")).await;
        assert!(String::from_utf8_lossy(&page).contains("Nothing queued"));
        assert_eq!(layout::read_review(&dir.join(REVIEW_QUEUE_FILE)).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn settings_form_saves_the_collect_section() {
        let (state, dir) = dataset_state();
        let (s, _, page) = call(
            &state,
            post(
                "/dataset/settings",
                "enabled=on&positive.enabled=on&empty.enabled=on&empty.interval=120&max_gb=1",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let page = String::from_utf8_lossy(&page);
        assert!(page.contains("Saved to"), "{page}");
        let c = state.config_read().collect.clone();
        assert!(c.enabled && c.buckets.positive.enabled && !c.buckets.suppressed.enabled);
        assert_eq!(c.buckets.empty.per_camera_min_interval_s, 120);
        assert_eq!(c.max_gb, 1.0);
        let (_, _, page) = call(&state, post("/dataset/settings", "enabled=on&max_gb=0")).await;
        assert!(String::from_utf8_lossy(&page).contains("collect.max_gb"));
        assert_eq!(
            state.config_read().collect.max_gb,
            1.0,
            "an invalid form changes nothing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
