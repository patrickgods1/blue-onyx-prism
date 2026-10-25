//! The Training page's footage import and auto-label forms (`POST /training/import`,
//! `POST /training/autolabel`, also posted by the Review page), the import preview
//! (`GET /v1/training/import/preview`, the line under the path field) and the rows of their
//! jobs.
//! Both are planned like the CLI does ([`crate::training::import::plan_import`],
//! [`crate::training::autolabel::plan_autolabel`]) and queued on the page's training queue
//! (one GPU job at a time with the training runs). Admin only, like every training route. The
//! auto-label form answers JSON (`{success, message, id}`) to a client that accepts
//! `application/json` (the Review page's fetch), else the Training page.

use super::training::{all, field, num, service, view};
use super::{Shared, blocking, render};
use crate::resources::catalog;
use crate::training::autolabel::{
    AutolabelRequest, FrameFilter, GdinoSize, VlmSize, default_status,
};
use crate::training::import::{ImportRequest, ImportSource, Selection};
use crate::training::service::RunState;
use crate::training::task::{TaskKind, TaskRecord};
use axum::Json;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::path::PathBuf;
use std::sync::Arc;

/// A job of the queue as the page lists it.
pub(super) struct TaskRow {
    pub id: String,
    pub kind: &'static str,
    pub title: String,
    pub state: &'static str,
    pub busy: bool,
    pub percent: u8,
    pub describe: String,
    pub log_tail: String,
    pub created: String,
}

/// Pinned footage offered by the import form.
pub(super) struct FootageRow {
    pub key: &'static str,
    pub title: &'static str,
    pub license: &'static str,
    pub size: String,
}

pub(super) fn footage() -> Vec<FootageRow> {
    catalog::IMPORT_DATASETS
        .iter()
        .map(|r| FootageRow {
            key: catalog::install_key(r),
            title: r.title,
            license: r.license,
            size: catalog::format_size(r.size()),
        })
        .collect()
}

fn state_str(s: RunState) -> &'static str {
    match s {
        RunState::Queued => "queued",
        RunState::Running => "running",
        RunState::Done => "done",
        RunState::Failed => "failed",
        RunState::Cancelled => "cancelled",
    }
}

/// The newest jobs first, at most 30.
pub(super) fn task_rows(tasks: &[TaskRecord]) -> Vec<TaskRow> {
    tasks
        .iter()
        .rev()
        .take(30)
        .map(|r| TaskRow {
            id: r.id.clone(),
            kind: match r.kind {
                TaskKind::Import => "import",
                TaskKind::Autolabel => "auto-label",
            },
            title: r.title.clone(),
            state: state_str(r.state),
            busy: r.state.is_busy(),
            percent: r.percent,
            describe: r.describe(),
            log_tail: r
                .log
                .iter()
                .rev()
                .take(4)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
            created: r.created.clone(),
        })
        .collect()
}

/// The import of a form: `source` = `path | url | dataset` and its field, the options and the
/// confirmations (`confirm_license`, `confirm_large`). Errors name the field.
pub(crate) fn import_from_form(f: &[(String, String)]) -> anyhow::Result<ImportRequest> {
    let src = field(f, "source").unwrap_or("path");
    let source = match src {
        "path" => ImportSource::Path(PathBuf::from(
            field(f, "path").ok_or_else(|| anyhow::anyhow!("file or folder: give a path"))?,
        )),
        "url" => ImportSource::Url(
            field(f, "url")
                .ok_or_else(|| anyhow::anyhow!("URL: give the video's URL"))?
                .to_string(),
        ),
        "dataset" => ImportSource::Dataset(
            field(f, "dataset")
                .ok_or_else(|| anyhow::anyhow!("dataset: choose one"))?
                .to_string(),
        ),
        other => anyhow::bail!("source: '{other}' is not path, url or dataset"),
    };
    let mut r = ImportRequest::new(source);
    r.camera = field(f, "camera").map(str::to_string);
    if let Some(n) = num(f, "frames", "frames")? {
        r.frames = n;
    }
    if let Some(v) = num(f, "scene", "scene-change threshold")? {
        r.scene_threshold = v;
    }
    if let Some(v) = num(f, "max_fps", "max fps")? {
        r.max_fps = v;
    }
    if let Some(v) = num(f, "every", "periodic frame")? {
        r.every_s = v;
    }
    if let Some(v) = num(f, "dedup_window", "dedup window")? {
        r.dedup_window_s = v;
    }
    if let Some(s) = field(f, "select") {
        r.selection = Selection::parse(s)
            .ok_or_else(|| anyhow::anyhow!("selection: '{s}' is not siglip or dhash"))?;
    }
    if let Some(n) = num(f, "max_candidates", "max candidates per video")? {
        r.max_candidates = n;
    }
    r.balance = field(f, "no_balance").is_none();
    r.license = field(f, "license").map(str::to_string);
    r.license_confirmed = field(f, "confirm_license").is_some();
    r.allow_large = field(f, "confirm_large").is_some();
    r.out_dir = field(f, "out_dir").map(PathBuf::from);
    r.validate()?;
    Ok(r)
}

/// The frame-filter fields of the auto-label form, each a key of [`FrameFilter`].
const FILTER_FIELDS: [&str; 6] = ["status", "scene", "camera", "label", "auto", "limit"];

/// The frame filter text of an auto-label form: the raw `frames` text (the Training page's
/// "Other filters", the Review page's and older forms' whole filter) followed by the structured
/// fields `status`, `scene`, `camera`, `label`, `auto`, `limit` and `again` (ticked = `again=1`)
/// whose key the raw text does not already name. A comma in a field's value is refused (it
/// would split the filter).
pub(crate) fn frame_filter_text(f: &[(String, String)]) -> anyhow::Result<String> {
    let raw = field(f, "frames").unwrap_or("");
    let named: Vec<String> = raw
        .split(',')
        .filter_map(|p| p.split_once('='))
        .map(|(k, _)| k.trim().to_ascii_lowercase())
        .collect();
    let mut parts: Vec<String> = Vec::new();
    if !raw.is_empty() {
        parts.push(raw.to_string());
    }
    for k in FILTER_FIELDS {
        if let Some(v) = field(f, k)
            && !named.iter().any(|n| n == k)
        {
            if v.contains(',') {
                anyhow::bail!("{k}: '{v}' has a comma, which the frame filter cannot hold");
            }
            parts.push(format!("{k}={v}"));
        }
    }
    if field(f, "again").is_some() && !named.iter().any(|n| n == "again") {
        parts.push("again=1".into());
    }
    Ok(parts.join(","))
}

/// The auto-label run of a form: the frame filter ([`frame_filter_text`]), `mode` (`labelers`
/// ignores `with_model`, `model` needs it; absent = `with_model` decides, as the Review page and
/// older forms post), `gdino`, `vlm` (`off` = none), `rfdetr` / `siglip` (ticked = on; the
/// Review page's form sends them), `with_model`, `threshold`, `confirm_large`.
pub(crate) fn autolabel_from_form(f: &[(String, String)]) -> anyhow::Result<AutolabelRequest> {
    let with_model = match field(f, "mode") {
        Some("labelers") => None,
        Some("model") => Some(
            field(f, "with_model")
                .ok_or_else(|| anyhow::anyhow!("re-label: choose a trained model"))?
                .to_string(),
        ),
        Some(m) => anyhow::bail!("mode: '{m}' is not labelers or model"),
        None => field(f, "with_model").map(str::to_string),
    };
    let frames = FrameFilter::parse(&frame_filter_text(f)?, default_status(with_model.is_some()))?;
    let mut r = AutolabelRequest::new(frames);
    r.gdino = match field(f, "gdino").unwrap_or("tiny") {
        "off" => None,
        g => Some(
            GdinoSize::parse(g)
                .ok_or_else(|| anyhow::anyhow!("MM-Grounding-DINO: '{g}' is not a size"))?,
        ),
    };
    r.vlm = match field(f, "vlm").unwrap_or("2b") {
        "off" => None,
        v => Some(VlmSize::parse(v).ok_or_else(|| anyhow::anyhow!("VLM: '{v}' is not a size"))?),
    };
    r.rfdetr = field(f, "rfdetr").is_some();
    r.siglip = field(f, "siglip").is_some();
    r.with_model = with_model;
    if let Some(t) = num(f, "threshold", "threshold")? {
        r.threshold = t;
    }
    r.allow_large = field(f, "confirm_large").is_some();
    let classes: Vec<String> = all(f, "classes")
        .iter()
        .flat_map(|c| c.split(','))
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    r.classes = (!classes.is_empty()).then_some(classes);
    r.validate()?;
    Ok(r)
}

/// `POST /training/import`.
pub(super) async fn import(State(shared): State<Shared>, body: axum::body::Bytes) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let result = (|| -> anyhow::Result<String> {
            let mut req = import_from_form(&f)?;
            if let ImportSource::Path(p) = &req.source {
                // Only what the page's file browser may show.
                let allowed = super::browse::allowed_path(&config, &p.to_string_lossy())
                    .map_err(|e| anyhow::anyhow!("{}", e.into_parts().1))?;
                req.source = ImportSource::Path(allowed);
            }
            if let Some(d) = &req.out_dir {
                // The same check: a browsable folder or one picked in the native dialog.
                let allowed = super::browse::allowed_path(&config, &d.to_string_lossy())
                    .map_err(|e| anyhow::anyhow!("save frames to: {}", e.into_parts().1))?;
                req.out_dir = Some(allowed);
            }
            let cap = crate::training::capability_now();
            let job = crate::training::import::plan_import(&config, req, &cap)?;
            let title = crate::training::task::Task::title(&job);
            let shown = job.dataset_dir == crate::collect::CollectSettings::for_config(&config).dir;
            let out = job.dataset_dir.display().to_string();
            service(&shared)
                .enqueue_task(Box::new(job))
                .map_err(anyhow::Error::msg)?;
            Ok(if shown {
                format!(
                    "Queued: {title}. The frames appear on the Dataset and Review pages when it is done."
                )
            } else {
                format!(
                    "Queued: {title}. The frames are written to {out}; to review them, choose that \
                     folder on the Review page."
                )
            })
        })();
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not imported: {e:#}")))),
        }
    })
    .await
}

/// `GET /v1/training/import/preview?path=<p>[&camera=<c>][&max_candidates=<n>]`: what an import
/// of `path` would read ([`crate::training::import::preview::Preview`] plus `success` and the
/// resolved `path`); a missing path, one outside the browsable (or natively picked) folders or
/// nothing importable there is also 200, `{success: false, message}` (the page previews while the
/// path is typed, so these are not request errors). The path
/// goes through the same check as the import itself ([`super::browse::allowed_path`]).
///
/// The walk has a budget and runs one at a time ([`crate::training::import::preview`]): a large
/// folder answers `truncated: true` with a "large folder: ..." summary, and a request replaced
/// by a newer one while it waits or walks answers `{success: false, busy: true, message}` (the
/// page drops answers to stale requests anyway). A browser request marked `Sec-Fetch-Site:
/// cross-site` is 403: another site must not make the admin's browser walk the disks.
pub(super) async fn import_preview(
    State(shared): State<Shared>,
    headers: HeaderMap,
    Query(q): Query<Vec<(String, String)>>,
) -> Response {
    if super::guard::fetch_site_cross(&headers) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "success": false,
                "message": "Cross-site request refused: the import preview answers this server's own pages only."
            })),
        )
            .into_response();
    }
    blocking(move || {
        let state = shared.current();
        let config = state.config_read().clone();
        // A path being typed is not an error: 200 with success false and the reason.
        let fail = |message: String| {
            Json(serde_json::json!({"success": false, "message": message})).into_response()
        };
        let Some(raw) = field(&q, "path") else {
            return fail("give a file or folder path".into());
        };
        let path = match super::browse::allowed_path(&config, raw) {
            Ok(p) => p,
            Err(e) => {
                return fail(e.into_parts().1);
            }
        };
        let per_video = match num::<usize>(&q, "max_candidates", "max candidates per video") {
            Ok(n) => n.unwrap_or(0),
            Err(e) => return fail(e.to_string()),
        };
        use crate::training::import::preview;
        match preview::preview_latest(&path, field(&q, "camera"), per_video) {
            Ok(p) => {
                let mut v = serde_json::to_value(&p).unwrap_or_default();
                if let Some(o) = v.as_object_mut() {
                    o.insert("success".into(), true.into());
                    o.insert("path".into(), path.display().to_string().into());
                }
                Json(v).into_response()
            }
            Err(e) if e.downcast_ref::<preview::Busy>().is_some() => Json(serde_json::json!({
                "success": false, "busy": true, "message": e.to_string()
            }))
            .into_response(),
            Err(e) => fail(format!("{e:#}")),
        }
    })
    .await
}

/// Whether the client asks for JSON (`Accept` names `application/json`).
fn wants_json(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .any(|v| v.to_ascii_lowercase().contains("application/json"))
}

/// `POST /training/autolabel`: the Training page, or with `Accept: application/json` 200
/// `{success: true, message, id}` (the task id, polled in `GET /v1/training` `tasks[]`) / 400
/// `{success: false, message}`.
pub(super) async fn autolabel(
    State(shared): State<Shared>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let json = wants_json(&headers);
    blocking(move || {
        let state = shared.current();
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let result = (|| -> anyhow::Result<(String, String)> {
            let req = autolabel_from_form(&f)?;
            let cap = crate::training::capability_now();
            let mut job = crate::training::autolabel::plan_autolabel(&config, req, &cap)?;
            if job.request.with_model.is_some() {
                // The generation's runtimes (no second OpenVINO / ONNX Runtime instance).
                job.detector = Some(Arc::new(
                    crate::training::autolabel::relabel::LocalDetector {
                        config: config.clone(),
                        runtimes: state.registry.runtimes().cloned(),
                    },
                ));
            }
            let title = crate::training::task::Task::title(&job);
            let id = service(&shared)
                .enqueue_task(Box::new(job))
                .map_err(anyhow::Error::msg)?;
            Ok((
                format!(
                    "Queued: {title}. The boxes appear on the Review page as pending auto boxes \
                     when it is done."
                ),
                id,
            ))
        })();
        match (result, json) {
            (Ok((m, id)), true) => {
                Json(serde_json::json!({"success": true, "message": m, "id": id})).into_response()
            }
            (Err(e), true) => (
                StatusCode::BAD_REQUEST,
                Json(
                    serde_json::json!({"success": false, "message": format!("Not queued: {e:#}")}),
                ),
            )
                .into_response(),
            (Ok((m, _)), false) => render(&view(&state, Some(m), None)),
            (Err(e), false) => render(&view(&state, None, Some(format!("Not queued: {e:#}")))),
        }
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn form(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn import_forms_parse_and_name_the_field() {
        let r = import_from_form(&form(&[
            ("source", "url"),
            ("url", "https://example.org/v.mp4"),
            ("license", "CC BY 4.0"),
            ("confirm_license", "on"),
            ("frames", "50"),
            ("select", "dhash"),
        ]))
        .unwrap();
        assert_eq!(
            r.source,
            ImportSource::Url("https://example.org/v.mp4".into())
        );
        assert!(r.license_confirmed && !r.allow_large);
        assert_eq!((r.frames, r.selection), (50, Selection::Dhash));
        assert_eq!(
            r.max_candidates,
            crate::training::import::DEFAULT_MAX_CANDIDATES
        );
        let r = import_from_form(&form(&[("path", "clips"), ("max_candidates", "300")])).unwrap();
        assert_eq!((r.max_candidates, r.out_dir), (300, None));
        let r = import_from_form(&form(&[("path", "clips"), ("out_dir", "/data/frames")])).unwrap();
        assert_eq!(r.out_dir, Some(PathBuf::from("/data/frames")));
        let e = import_from_form(&form(&[("path", "x"), ("max_candidates", "0")])).unwrap_err();
        assert!(e.to_string().contains("max candidates"), "{e}");
        let e = import_from_form(&form(&[("path", "x"), ("max_candidates", "x")])).unwrap_err();
        assert!(e.to_string().contains("max candidates per video"), "{e}");
        let e = import_from_form(&form(&[("source", "path")])).unwrap_err();
        assert!(e.to_string().contains("give a path"), "{e}");
        let e = import_from_form(&form(&[("path", "x"), ("frames", "many")])).unwrap_err();
        assert!(e.to_string().contains("frames"), "{e}");
        let e = import_from_form(&form(&[("source", "ftp")])).unwrap_err();
        assert!(e.to_string().contains("source"), "{e}");
    }

    #[test]
    fn autolabel_forms_parse() {
        let r = autolabel_from_form(&form(&[
            ("frames", "scene=ir,limit=10"),
            ("gdino", "base"),
            ("vlm", "off"),
            ("rfdetr", "on"),
        ]))
        .unwrap();
        assert_eq!(r.frames.limit, 10);
        assert_eq!(r.gdino, Some(GdinoSize::Base));
        assert_eq!(r.vlm, None);
        assert!(r.rfdetr && !r.siglip);
        let m = autolabel_from_form(&form(&[("with_model", "night-x")])).unwrap();
        assert_eq!(m.frames.query["status"], "any");
        assert!(autolabel_from_form(&form(&[("frames", "colour=red")])).is_err());
        assert!(autolabel_from_form(&form(&[("gdino", "off"), ("vlm", "off")])).is_err());
    }

    #[test]
    fn the_frame_fields_compose_the_filter() {
        // The structured fields, in their order; empty ones are left out.
        let t = frame_filter_text(&form(&[
            ("status", "reviewed"),
            ("scene", "night"),
            ("camera", ""),
            ("label", "person"),
            ("auto", "none"),
            ("limit", "40"),
            ("again", "1"),
            ("frames", ""),
        ]))
        .unwrap();
        assert_eq!(
            t,
            "status=reviewed,scene=night,label=person,auto=none,limit=40,again=1"
        );
        let r = autolabel_from_form(&form(&[
            ("scene", "ir"),
            ("limit", "7"),
            ("frames", "bucket=empty, import=any"),
        ]))
        .unwrap();
        assert_eq!(r.frames.limit, 7);
        assert_eq!(r.frames.query["scene"], "ir");
        assert_eq!(r.frames.query["bucket"], "empty");
        assert_eq!(r.frames.import(), Some("any"));
        assert_eq!(r.frames.query["status"], "unreviewed");
        assert!(!r.frames.again());
        // A key of the raw text wins over its field (the Review page sends both `limit`s).
        let t = frame_filter_text(&form(&[
            ("frames", "scene=day,limit=50,again=1"),
            ("limit", "200"),
            ("scene", "night"),
            ("again", "1"),
        ]))
        .unwrap();
        assert_eq!(t, "scene=day,limit=50,again=1");
        // Nothing given: the defaults.
        assert_eq!(frame_filter_text(&form(&[("limit", "")])).unwrap(), "");
        let e = frame_filter_text(&form(&[("camera", "a,b")])).unwrap_err();
        assert!(e.to_string().contains("camera"), "{e}");
        // A bad value is refused by the filter with its key.
        let e = autolabel_from_form(&form(&[("auto", "maybe")])).unwrap_err();
        assert!(e.to_string().contains("auto"), "{e}");
    }

    #[test]
    fn the_mode_decides_between_labelers_and_a_model() {
        // Labelers: a chosen model in the hidden select is ignored.
        let r = autolabel_from_form(&form(&[
            ("mode", "labelers"),
            ("with_model", "night-x"),
            ("gdino", "tiny"),
            ("rfdetr", "on"),
        ]))
        .unwrap();
        assert_eq!(r.with_model, None);
        assert_eq!(r.frames.query["status"], "unreviewed");
        // Model: the labeler fields are posted too and ignored, as with `with_model` alone.
        let m = autolabel_from_form(&form(&[
            ("mode", "model"),
            ("with_model", "night-x"),
            ("gdino", "off"),
            ("vlm", "off"),
        ]))
        .unwrap();
        assert_eq!(m.with_model.as_deref(), Some("night-x"));
        assert_eq!(m.frames.query["status"], "any");
        assert!(m.resources().is_empty());
        let e = autolabel_from_form(&form(&[("mode", "model")])).unwrap_err();
        assert!(e.to_string().contains("choose a trained model"), "{e}");
        assert!(autolabel_from_form(&form(&[("mode", "both")])).is_err());
    }

    #[tokio::test]
    async fn the_forms_answer_with_the_page() {
        let _queue = super::super::training::QUEUE_TEST_LOCK.lock().await;
        use crate::server::test_util::{call, get_text, post_form, test_state_with};
        use axum::http::StatusCode;
        let root = std::env::temp_dir().join(format!("bop-train-tasks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("snaps")).unwrap();
        let mut rgb = Vec::new();
        for i in 0..(32 * 24) {
            let v = u8::try_from(i % 250).unwrap();
            rgb.extend([v, v / 2, 255 - v]);
        }
        let img = crate::image::RgbImage {
            width: 32,
            height: 24,
            rgb,
        };
        std::fs::write(
            root.join("snaps/Gate.20240101_120000.jpg"),
            crate::image::encode_jpeg(&img, 90).unwrap(),
        )
        .unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let (s, page) = get_text(&state, "/training").await;
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("Import footage"), "the import card");
        assert!(page.contains("Auto-label"), "the auto-label card");
        assert!(page.contains("meva-sample"));
        assert!(page.contains("/static/browse.js"));
        assert!(page.contains("import-preview"), "the preview line");
        // The preview of the folder: counts, the per-video budget and the summary line.
        std::fs::write(root.join("snaps/Drive.20240101_120000.mp4"), b"v").unwrap();
        let q = super::super::encode_uri_component(&root.join("snaps").display().to_string());
        let (s, json) = get_text(
            &state,
            &format!("/v1/training/import/preview?path={q}&max_candidates=300"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{json}");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(
            (v["videos"].as_u64(), v["images"].as_u64()),
            (Some(1), Some(1))
        );
        assert_eq!(v["cameras"][0]["camera"], "drive");
        assert_eq!(v["per_video"], 300);
        assert!(
            v["summary"]
                .as_str()
                .unwrap()
                .starts_with("1 video from 1 camera, 1 image ("),
            "{v}"
        );
        std::fs::remove_file(root.join("snaps/Drive.20240101_120000.mp4")).unwrap();
        // Outside the browsable folders, no path, a bad number: refused with a message.
        let (s, json) = get_text(&state, "/v1/training/import/preview?path=%2Fetc").await;
        assert_eq!(s, StatusCode::OK, "{json}");
        assert!(json.contains("\"success\":false"), "{json}");
        let (s, json) = get_text(&state, "/v1/training/import/preview").await;
        assert_eq!(s, StatusCode::OK);
        assert!(json.contains("give a file or folder path"), "{json}");
        // A cross-site GET is refused before anything is walked; same-origin passes.
        for (site, want) in [
            ("cross-site", StatusCode::FORBIDDEN),
            ("same-origin", StatusCode::OK),
            ("none", StatusCode::OK),
        ] {
            let req = axum::http::Request::get(format!("/v1/training/import/preview?path={q}"))
                .header("sec-fetch-site", site)
                .body(axum::body::Body::empty())
                .unwrap();
            let (s, _, b) = call(&state, req).await;
            assert_eq!(s, want, "{site}: {}", String::from_utf8_lossy(&b));
        }
        let (s, json) = get_text(
            &state,
            &format!("/v1/training/import/preview?path={q}&max_candidates=x"),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(json.contains("\"success\":false"), "{json}");
        // A denylisted URL is refused with the reason.
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/import",
                "source=url&url=http%3A%2F%2Fwww.insecam.org%2Fen%2F&license=x&confirm_license=on",
                true,
            ),
        )
        .await;
        let html = String::from_utf8(b).unwrap();
        assert!(
            html.contains("Not imported") && html.contains("unsecured"),
            "{html}"
        );
        // A path outside the browsable folders is refused.
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/import",
                "source=path&path=%2Fetc&confirm_license=on&select=dhash",
                true,
            ),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("Not imported"));
        // An image folder with the dHash selection imports without a GPU.
        let body = format!(
            "source=path&path={}&confirm_license=on&select=dhash&frames=5",
            super::super::encode_uri_component(&root.join("snaps").display().to_string())
        );
        let (_, _, b) = call(&state, post_form("/training/import", &body, true)).await;
        let html = String::from_utf8(b).unwrap();
        assert!(html.contains("Queued: import of snaps"), "{html}");
        // Wait for it (the queue is the process's), then the frame is in the dataset.
        let mut done = false;
        for _ in 0..200 {
            let (_, json) = get_text(&state, "/v1/training").await;
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            let mine = v["tasks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| {
                    t["title"]
                        .as_str()
                        .is_some_and(|x| x.contains("import of snaps"))
                })
                .cloned();
            if let Some(t) = mine
                && t["state"] != "queued"
                && t["state"] != "running"
            {
                assert_eq!(t["state"], "done", "{t}");
                done = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(done, "the import finished");
        let s = crate::collect::Summary::scan(&root.join("dataset"));
        assert_eq!(s.frames, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn the_autolabel_form_answers_json_when_asked() {
        let _queue = super::super::training::QUEUE_TEST_LOCK.lock().await;
        use crate::server::test_util::{call, test_state_with};
        use axum::body::Body;
        use axum::http::Request;
        let root = std::env::temp_dir().join(format!("bop-train-al-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let post = |body: &str, accept: &str| {
            Request::post("/training/autolabel")
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .header(header::ACCEPT, accept)
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        // A bad filter: 400 with the reason.
        let (s, _, b) = call(&state, post("frames=colour%3Dred", "application/json")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["success"], false);
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("unknown frame filter 'colour'"),
            "{v}"
        );
        // A re-label with a configured model needs no GPU: queued, with the task id.
        let (s, _, b) = call(
            &state,
            post(
                "with_model=ipcam-general&frames=limit%3D5",
                "application/json, text/plain, */*",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["success"], true, "{v}");
        assert!(
            v["message"].as_str().unwrap().starts_with("Queued: "),
            "{v}"
        );
        let id = v["id"].as_str().unwrap().to_string();
        assert!(!id.is_empty());
        let (_, json) = crate::server::test_util::get_text(&state, "/v1/training").await;
        let t: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            t["tasks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|x| x["id"] == id.as_str()),
            "{t}"
        );
        // A browser form post still gets the page.
        let (s, _, b) = call(&state, post("frames=colour%3Dred", "text/html")).await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("Not queued"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
