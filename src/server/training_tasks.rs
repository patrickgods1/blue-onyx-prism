//! The Training page's footage import and auto-label forms (`POST /training/import`,
//! `POST /training/autolabel`, also posted by the Review page) and the rows of their jobs.
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
use axum::extract::State;
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
    r.balance = field(f, "no_balance").is_none();
    r.license = field(f, "license").map(str::to_string);
    r.license_confirmed = field(f, "confirm_license").is_some();
    r.allow_large = field(f, "confirm_large").is_some();
    r.validate()?;
    Ok(r)
}

/// The auto-label run of a form: `frames` (the filter text), `gdino`, `vlm` (`off` = none),
/// `rfdetr` / `siglip` (ticked = on; the Review page's form sends them), `with_model`,
/// `threshold`, `confirm_large`.
pub(crate) fn autolabel_from_form(f: &[(String, String)]) -> anyhow::Result<AutolabelRequest> {
    let with_model = field(f, "with_model").map(str::to_string);
    let frames = FrameFilter::parse(
        field(f, "frames").unwrap_or(""),
        default_status(with_model.is_some()),
    )?;
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
            let cap = crate::training::capability_now();
            let job = crate::training::import::plan_import(&config, req, &cap)?;
            let title = crate::training::task::Task::title(&job);
            service(&shared)
                .enqueue_task(Box::new(job))
                .map_err(anyhow::Error::msg)?;
            Ok(format!(
                "Queued: {title}. The frames appear on the Dataset and Review pages when it is done."
            ))
        })();
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not imported: {e:#}")))),
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

    #[tokio::test]
    async fn the_forms_answer_with_the_page() {
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
