//! The Training page's footage import and auto-label forms (`POST /training/import`,
//! `POST /training/autolabel`, also posted by the Review page), the import preview
//! (`GET /v1/training/import/preview`, the line under the path field), the models an auto-label
//! run can use (`GET /v1/autolabel/models`) and the rows of their jobs.
//! Both are planned like the CLI does ([`crate::training::import::plan_import`],
//! [`crate::training::autolabel::plan_autolabel`]) and queued on the page's training queue
//! (one GPU job at a time with the training runs). Admin only, like every training route. The
//! auto-label form answers JSON (`{success, message, id}`) to a client that accepts
//! `application/json` (the Review page's fetch), else the Training page.

use super::training::{all, field, num, service, view};
use super::{AppState, Shared, blocking, render};
use crate::config::Config;
use crate::resources::catalog;
use crate::training::autolabel::models::{self as al_models, ConfigWriter};
use crate::training::autolabel::{
    AutolabelJob, AutolabelRequest, FrameFilter, GdinoSize, VlmSize, default_status,
};
use crate::training::finish::Finish;
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
        RunState::Paused => "paused",
        RunState::Done => "done",
        RunState::Failed => "failed",
        RunState::Cancelled => "cancelled",
    }
}

/// What a row says about `r`: [`TaskRecord::describe`], a queued job that follows an import
/// as "queued (after import <id>)".
fn describe_row(r: &TaskRecord, tasks: &[TaskRecord]) -> String {
    match (&r.after, r.state) {
        (Some(a), RunState::Queued)
            if tasks
                .iter()
                .any(|t| &t.id == a && t.kind == TaskKind::Import) =>
        {
            format!("queued (after import {a})")
        }
        _ => r.describe(),
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
            describe: describe_row(r, tasks),
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
/// whose key the raw text does not already name. `label` is a comma list of classes (any of
/// them), written `label=a|b`; a comma in another field's value is refused (it would split the
/// filter).
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
            // Class pickers post `label` as a comma list; the frame filter's any-of is `|`.
            let v = if k == "label" {
                v.split(',')
                    .map(str::trim)
                    .filter(|c| !c.is_empty())
                    .collect::<Vec<_>>()
                    .join("|")
            } else if v.contains(',') {
                anyhow::bail!("{k}: '{v}' has a comma, which the frame filter cannot hold");
            } else {
                v.to_string()
            };
            if !v.is_empty() {
                parts.push(format!("{k}={v}"));
            }
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
/// Review page's form sends them), `with_model` (any id of `GET /v1/autolabel/models`),
/// `threshold`, for the labelers `gpu` (`cuda:N`; empty or `auto` = the automatic pick), and for
/// a run with a model `device` (empty = the model's), `model_accept_score` (empty = off);
/// `confirm_large`, `confirm_license` (ticked = confirmed). The run options `keep_on_cancel`,
/// `vlm_batch`, `prefetch` and `vlm_min_on` (ticked = on; then `vlm_min_score` applies) count
/// only when the form posts `run_options` (the Training page's card); without it (the Review
/// page, older forms) they keep the request's defaults.
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
    if field(f, "run_options").is_some() {
        r.keep_on_cancel = field(f, "keep_on_cancel").is_some();
        r.vlm_batch = field(f, "vlm_batch").is_some();
        r.prefetch = field(f, "prefetch").is_some();
        r.vlm_min_score = if field(f, "vlm_min_on").is_some() {
            Some(
                num(f, "vlm_min_score", "VLM minimum score")?
                    .unwrap_or(crate::training::autolabel::SUGGESTED_VLM_MIN_SCORE),
            )
        } else {
            None
        };
    }
    if with_model.is_some() {
        // A model run's fields; the labelers' are ignored.
        r.device = field(f, "device").map(str::to_string);
        r.model_accept_score = num(f, "model_accept_score", "model accept score")?;
    } else {
        r.gpu = field(f, "gpu").map(str::to_string);
    }
    r.with_model = with_model;
    if let Some(t) = num(f, "threshold", "threshold")? {
        r.threshold = t;
    }
    r.allow_large = field(f, "confirm_large").is_some();
    r.confirm_license = field(f, "confirm_license").is_some();
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

/// The auto-label form's fields an import posts with `al.` (prefix stripped), without its frame
/// fields ([`FILTER_FIELDS`], `frames`, `again`): the run labels the import's frames
/// ([`crate::training::autolabel::follow_import`]).
fn chained_autolabel_fields(f: &[(String, String)]) -> Vec<(String, String)> {
    f.iter()
        .filter_map(|(k, v)| Some((k.strip_prefix("al.")?.to_string(), v.clone())))
        .filter(|(k, _)| {
            !FILTER_FIELDS.contains(&k.as_str()) && k.as_str() != "frames" && k.as_str() != "again"
        })
        .collect()
}

/// Default of `after_reject_score`.
const AFTER_REJECT_DEFAULT: f32 = 0.3;

/// The steps after an import of a form ([`Finish`]): `after_reject` (ticked; then
/// `after_reject_score`, 0.01..=1, default 0.3) and `after_accept` need `autolabel`,
/// `after_review_folder` needs `out_dir` (the import's folder `dataset_dir`). None when no step
/// is ticked. Errors name the option.
fn finish_from_form(
    f: &[(String, String)],
    chained: bool,
    out_dir: Option<&std::path::Path>,
    dataset_dir: &std::path::Path,
) -> anyhow::Result<Option<Finish>> {
    let reject = field(f, "after_reject").is_some();
    let accept = field(f, "after_accept").is_some();
    let folder = field(f, "after_review_folder").is_some();
    if !chained && reject {
        anyhow::bail!(
            "reject auto boxes scoring below: needs Auto-label after import; tick \"Auto-label the \
             imported frames\" or untick it"
        );
    }
    if !chained && accept {
        anyhow::bail!(
            "accept frames whose boxes are all auto: accept: needs Auto-label after import; tick \
             \"Auto-label the imported frames\" or untick it"
        );
    }
    let reject_below = if reject {
        let v = num::<f32>(f, "after_reject_score", "reject auto boxes scoring below")?
            .unwrap_or(AFTER_REJECT_DEFAULT);
        if !(v.is_finite() && (0.01..=1.0).contains(&v)) {
            anyhow::bail!("reject auto boxes scoring below: must be between 0.01 and 1, got {v}");
        }
        Some(v)
    } else {
        None
    };
    if folder && out_dir.is_none() {
        anyhow::bail!(
            "show this folder on the Review page: give a folder in Save frames to (the default \
             folder is the dataset folder the Review page shows already)"
        );
    }
    let fin = Finish {
        reject_below,
        accept_auto: accept,
        review_folder: folder.then(|| dataset_dir.to_path_buf()),
        config_writer: None,
    };
    Ok((!fin.is_empty()).then_some(fin))
}

/// `POST /training/import`. With `autolabel` ticked, the auto-label card's fields (posted with
/// an `al.` prefix) queue an auto-label of the import's frames right after it
/// ([`chained_autolabel_fields`], [`crate::training::task::Task::after`]); the follow-up steps
/// ([`finish_from_form`]) run after the last task. Everything is planned before anything is
/// queued: a bad auto-label setting imports nothing.
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
                // The same check (a browsable folder or one picked in the native dialog), or a
                // folder an earlier import wrote to / under review.
                let allowed = super::review::allowed_folder(&state, &config, &d.to_string_lossy())
                    .map_err(|(_, m)| anyhow::anyhow!("save frames to: {m}"))?;
                req.out_dir = Some(allowed);
            }
            let chained = field(&f, "autolabel").is_some();
            let has_out_dir = req.out_dir.is_some();
            let cap = crate::training::capability_now();
            let mut job = crate::training::import::plan_import(&config, req, &cap)?;
            let mut finish = finish_from_form(
                &f,
                chained,
                has_out_dir.then_some(job.dataset_dir.as_path()),
                &job.dataset_dir,
            )?;
            if let Some(fin) = &mut finish {
                fin.config_writer = Some(Arc::new(ServerConfigWriter {
                    shared: shared.clone(),
                }));
            }
            let switches = finish.as_ref().is_some_and(|f| f.review_folder.is_some());
            let autolabel = if chained {
                let mut plan = || -> anyhow::Result<AutolabelJob> {
                    let req = autolabel_from_form(&chained_autolabel_fields(&f))?;
                    let mut al = crate::training::autolabel::plan_autolabel(&config, req, &cap)?;
                    attach_seams(&state, &shared, &config, &mut al);
                    crate::training::autolabel::follow_import(&mut al, &job);
                    al.finish = finish.take();
                    Ok(al)
                };
                Some(plan().map_err(|e| anyhow::anyhow!("auto-label: {e:#}"))?)
            } else {
                job.finish = finish.take();
                None
            };
            let title = crate::training::task::Task::title(&job);
            let shown = job.dataset_dir == crate::collect::CollectSettings::for_config(&config).dir;
            let out = job.dataset_dir.display().to_string();
            let svc = service(&shared);
            let import_id = svc
                .enqueue_task(Box::new(job))
                .map_err(anyhow::Error::msg)?;
            let mut msg = match autolabel {
                Some(al) => {
                    let al_title = crate::training::task::Task::title(&al);
                    if let Err(e) = svc.enqueue_task(Box::new(al)) {
                        svc.cancel(&import_id);
                        anyhow::bail!("auto-label: {e} (the import was cancelled too)");
                    }
                    format!(
                        "Queued: {title}, then {al_title}. The auto-label runs only when the \
                         import finishes with new frames; its boxes appear on the Review page as \
                         pending auto boxes."
                    )
                }
                None => format!("Queued: {title}."),
            };
            if switches {
                msg.push_str(&format!(
                    " The Review page switches to {out} when it is done."
                ));
            } else if shown {
                msg.push_str(" The frames appear on the Dataset and Review pages when it is done.");
            } else {
                msg.push_str(&format!(
                    " The frames are written to {out}; to review them, choose that folder on the \
                     Review page."
                ));
            }
            Ok(msg)
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
            attach_seams(&state, &shared, &config, &mut job);
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

/// The server's seams of a run with a model: the generation's runtimes (no second OpenVINO /
/// ONNX Runtime instance) and the config writer of the model entry it adds.
fn attach_seams(state: &AppState, shared: &Shared, config: &Config, job: &mut AutolabelJob) {
    if job.request.with_model.is_some() {
        job.detector = Some(Arc::new(
            crate::training::autolabel::relabel::LocalDetector {
                config: config.clone(),
                runtimes: state.registry.runtimes().cloned(),
            },
        ));
        job.config_writer = Some(Arc::new(ServerConfigWriter {
            shared: shared.clone(),
        }));
    }
}

/// Saves the config entry a run with a local or catalog model adds (disabled) through
/// [`super::save_config`], on the generation current at that moment.
struct ServerConfigWriter {
    shared: Shared,
}

impl ConfigWriter for ServerConfigWriter {
    fn update(
        &self,
        edit: &mut dyn FnMut(&mut Config) -> anyhow::Result<()>,
    ) -> anyhow::Result<(Config, String)> {
        let state = self.shared.current();
        let saved = super::save_config(&state, "autolabel", |c| edit(c))?;
        let c = state.config_read().clone();
        Ok((c, saved))
    }
}

/// One device choice of `GET /v1/autolabel/models`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct DeviceChoice {
    pub value: String,
    pub label: String,
}

/// `auto` (with what it picks now), then every runnable device option.
fn device_choices(sel: &crate::backend::select::Selection) -> Vec<DeviceChoice> {
    let pick = sel
        .auto_pick()
        .map_or_else(|| "nothing runnable".to_string(), |o| o.spec.to_string());
    std::iter::once(DeviceChoice {
        value: "auto".into(),
        label: format!("auto ({pick})"),
    })
    .chain(
        sel.options
            .iter()
            .filter(|o| o.runnable)
            .map(|o| DeviceChoice {
                value: o.spec.to_string(),
                label: format!("{} \u{2014} {}", o.spec, o.label),
            }),
    )
    .collect()
}

/// `GET /v1/autolabel/models`: `{success, models, devices, default_threshold}`: every model a
/// run with a model can use ([`al_models::model_options`]: config, local files, catalog), the
/// runnable devices (`auto` first) and the default score threshold.
pub(super) async fn autolabel_models(State(shared): State<Shared>) -> Response {
    blocking(move || {
        let state = shared.current();
        let config = state.config_read().clone();
        let models = al_models::model_options(&config);
        let snap = super::devices::devices_snapshot(&state);
        Json(serde_json::json!({
            "success": true,
            "models": models,
            "devices": device_choices(&snap.selection),
            // As the f64 0.3, not the f32's 0.30000001192092896 the forms would show.
            "default_threshold": (f64::from(al_models::DEFAULT_THRESHOLD) * 100.0).round() / 100.0,
        }))
        .into_response()
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
        assert_eq!(
            m.frames.query["status"], "unreviewed",
            "first-pass labelling; self-training posts status=any"
        );
        assert_eq!((m.device.as_deref(), m.model_accept_score), (None, None));
        assert!(!m.confirm_license);
        // A model run's own fields.
        let m = autolabel_from_form(&form(&[
            ("mode", "model"),
            ("with_model", "resource:model:rfdetr-large"),
            ("device", "openvino:cpu"),
            ("model_accept_score", "0.8"),
            ("threshold", "0.4"),
            ("confirm_license", "on"),
            ("status", "any"),
        ]))
        .unwrap();
        assert_eq!(m.with_model.as_deref(), Some("resource:model:rfdetr-large"));
        assert_eq!(m.device.as_deref(), Some("openvino:cpu"));
        assert_eq!(m.model_accept_score, Some(0.8));
        assert!(m.confirm_license);
        assert_eq!(m.frames.query["status"], "any");
        // Empty fields are off; the labelers ignore them.
        let m = autolabel_from_form(&form(&[
            ("with_model", "x"),
            ("device", ""),
            ("model_accept_score", ""),
        ]))
        .unwrap();
        assert_eq!((m.device, m.model_accept_score), (None, None));
        let l = autolabel_from_form(&form(&[
            ("mode", "labelers"),
            ("with_model", "x"),
            ("device", "openvino:cpu"),
            ("model_accept_score", "0.9"),
            ("rfdetr", "on"),
        ]))
        .unwrap();
        assert_eq!((l.device, l.model_accept_score), (None, None));
        for (k, v) in [
            ("model_accept_score", "0.1"),
            ("model_accept_score", "1.5"),
            ("model_accept_score", "high"),
            ("device", "tpu"),
        ] {
            let e = autolabel_from_form(&form(&[("with_model", "x"), (k, v)])).unwrap_err();
            assert!(
                e.to_string().contains("accept score") || e.to_string().contains("device"),
                "{k}={v}: {e:#}"
            );
        }
        // The labelers' GPU; a model run ignores it (it has its device).
        let g = autolabel_from_form(&form(&[
            ("mode", "labelers"),
            ("rfdetr", "on"),
            ("gpu", "cuda:1"),
        ]))
        .unwrap();
        assert_eq!(g.gpu.as_deref(), Some("cuda:1"));
        let g = autolabel_from_form(&form(&[("rfdetr", "on"), ("gpu", "")])).unwrap();
        assert_eq!(g.gpu, None);
        let m = autolabel_from_form(&form(&[
            ("mode", "model"),
            ("with_model", "x"),
            ("gpu", "cuda:1"),
        ]))
        .unwrap();
        assert_eq!(m.gpu, None);
        let e = autolabel_from_form(&form(&[("rfdetr", "on"), ("gpu", "second")])).unwrap_err();
        assert!(e.to_string().contains("is not cuda:N"), "{e:#}");
        assert!(autolabel_from_form(&form(&[("frames", "colour=red")])).is_err());
        assert!(autolabel_from_form(&form(&[("gdino", "off"), ("vlm", "off")])).is_err());
    }

    #[test]
    fn autolabel_run_options_parse_ticked_and_unticked() {
        // The Review page and older forms post no run options: the defaults.
        let r = autolabel_from_form(&form(&[("rfdetr", "on")])).unwrap();
        assert!(r.keep_on_cancel && r.vlm_batch && r.prefetch);
        assert_eq!(r.vlm_min_score, None);
        // The Training page with every box unticked (absent); the score alone is ignored.
        let r = autolabel_from_form(&form(&[
            ("rfdetr", "on"),
            ("run_options", "1"),
            ("vlm_min_score", "0.3"),
        ]))
        .unwrap();
        assert!(!r.keep_on_cancel && !r.vlm_batch && !r.prefetch);
        assert_eq!(r.vlm_min_score, None);
        // Every box ticked.
        let r = autolabel_from_form(&form(&[
            ("rfdetr", "on"),
            ("run_options", "1"),
            ("keep_on_cancel", "on"),
            ("vlm_batch", "on"),
            ("prefetch", "on"),
            ("vlm_min_on", "on"),
            ("vlm_min_score", "0.25"),
        ]))
        .unwrap();
        assert!(r.keep_on_cancel && r.vlm_batch && r.prefetch);
        assert_eq!(r.vlm_min_score, Some(0.25));
        // Ticked with an empty score: the suggested 0.2; a bad one names the field.
        let r = autolabel_from_form(&form(&[
            ("rfdetr", "on"),
            ("run_options", "1"),
            ("vlm_min_on", "on"),
            ("vlm_min_score", ""),
        ]))
        .unwrap();
        assert_eq!(r.vlm_min_score, Some(0.2));
        for bad in ["0", "1.5", "low"] {
            let e = autolabel_from_form(&form(&[
                ("rfdetr", "on"),
                ("run_options", "1"),
                ("vlm_min_on", "on"),
                ("vlm_min_score", bad),
            ]))
            .unwrap_err();
            assert!(e.to_string().contains("VLM minimum score"), "{bad}: {e:#}");
        }
        // The keep box is shared by both modes (an unticked one is off in model mode too).
        let m = autolabel_from_form(&form(&[
            ("mode", "model"),
            ("with_model", "x"),
            ("run_options", "1"),
        ]))
        .unwrap();
        assert!(!m.keep_on_cancel);
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
        // A class picker's comma list becomes the filter's any-of and reaches the job.
        let t = frame_filter_text(&form(&[("label", "person, car,")])).unwrap();
        assert_eq!(t, "label=person|car");
        let r = autolabel_from_form(&form(&[("label", "person,car"), ("limit", "5")])).unwrap();
        assert_eq!(r.frames.limit, 5, "not split into key=value pairs");
        assert_eq!(
            r.frames.query.get("label").map(String::as_str),
            Some("person|car")
        );
        let want: std::collections::BTreeSet<String> =
            ["person".to_string(), "car".to_string()].into();
        assert_eq!(r.frames.filters().unwrap().label, want, "any of them");
        assert_eq!(frame_filter_text(&form(&[("label", " , ")])).unwrap(), "");
        assert!(autolabel_from_form(&form(&[("label", "ok,a\u{7}b")])).is_err());
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
        assert_eq!(m.frames.query["status"], "unreviewed");
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
    async fn the_models_route_lists_config_local_and_catalog_models() {
        use crate::server::test_util::{get_text, test_state_with};
        let root = std::env::temp_dir().join(format!("bop-al-models-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join("models/mine.onnx"), b"x").unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let (s, json) = get_text(&state, "/v1/autolabel/models").await;
        assert_eq!(s, StatusCode::OK, "{json}");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["default_threshold"], serde_json::json!(0.3));
        let models = v["models"].as_array().unwrap();
        assert_eq!(models[0]["source"], "config", "{v}");
        assert!(
            models
                .iter()
                .any(|m| m["id"] == "local:mine.onnx" && m["source"] == "local"),
            "{v}"
        );
        let rf = models
            .iter()
            .find(|m| m["id"] == "resource:model:rfdetr-base")
            .unwrap();
        for k in [
            "name",
            "installed",
            "in_config",
            "enabled",
            "family",
            "classes",
            "coco",
            "size_bytes",
            "needs_export",
            "large",
            "license",
            "note",
            "device",
        ] {
            assert!(rf.get(k).is_some(), "{k}: {rf}");
        }
        let devices = v["devices"].as_array().unwrap();
        assert_eq!(devices[0]["value"], "auto");
        assert!(
            devices[0]["label"].as_str().unwrap().starts_with("auto ("),
            "{v}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn the_server_writer_adds_disabled_entries_through_save_config() {
        use crate::server::test_util::test_state_with;
        let root = std::env::temp_dir().join(format!("bop-al-writer-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join("models/mine.onnx"), b"x").unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let st = state.clone();
        let (cfg, msg) = blocking(move || {
            let w = ServerConfigWriter { shared: st.into() };
            let c = w.update(&mut |_| Ok(())).unwrap().0;
            let plan = al_models::resolve(&c, "local:mine.onnx", None).unwrap();
            let mut msg = String::new();
            let out = w
                .update(&mut |c| {
                    msg = al_models::add_disabled(c, &plan)?;
                    Ok(())
                })
                .unwrap();
            (out.0, format!("{msg} {}", out.1))
        })
        .await;
        assert!(msg.contains("disabled") && msg.contains("Saved"), "{msg}");
        let m = cfg
            .models
            .iter()
            .find(|m| m.effective_name() == "mine")
            .unwrap();
        assert!(!m.enabled);
        // In the live config and on disk.
        assert!(
            state
                .config_read()
                .models
                .iter()
                .any(|m| m.effective_name() == "mine")
        );
        let disk = Config::load(&state.config_path).unwrap();
        assert!(
            disk.models
                .iter()
                .any(|m| m.effective_name() == "mine" && !m.enabled)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn device_choices_are_auto_then_the_runnable_options() {
        use crate::backend::select::{RuntimeProbe, select};
        let hw = crate::backend::detect::HardwareInfo::new("linux", "x86_64", vec![]);
        let probe = RuntimeProbe::openvino_only(&["CPU".to_string()]);
        let sel = select(&hw, &probe, true);
        let c = device_choices(&sel);
        assert_eq!(c[0].value, "auto");
        assert_eq!(c[0].label, "auto (openvino:cpu)");
        assert!(c.iter().any(|d| d.value == "openvino:cpu"), "{c:?}");
        assert!(
            c.iter().skip(1).all(|d| sel
                .options
                .iter()
                .any(|o| o.runnable && o.spec.to_string() == d.value)),
            "{c:?}"
        );
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
    #[test]
    fn chained_fields_and_follow_ups_parse() {
        let f = form(&[
            ("al.mode", "model"),
            ("al.with_model", "x"),
            ("al.status", "reviewed"),
            ("al.limit", "5"),
            ("al.frames", "import=other"),
            ("al.again", "1"),
            ("al.classes", "person"),
            ("frames", "40"),
        ]);
        let al = chained_autolabel_fields(&f);
        assert_eq!(
            al,
            form(&[
                ("mode", "model"),
                ("with_model", "x"),
                ("classes", "person")
            ]),
            "the frame fields are dropped, the import's own fields are not the run's"
        );
        let d = std::path::Path::new("/data/out");
        assert!(
            finish_from_form(&form(&[]), true, None, d)
                .unwrap()
                .is_none()
        );
        let fin = finish_from_form(
            &form(&[
                ("after_reject", "on"),
                ("after_accept", "on"),
                ("after_review_folder", "on"),
            ]),
            true,
            Some(d),
            d,
        )
        .unwrap()
        .unwrap();
        assert_eq!(fin.reject_below, Some(0.3), "the default score");
        assert!(fin.accept_auto);
        assert_eq!(fin.review_folder.as_deref(), Some(d));
        let fin = finish_from_form(
            &form(&[("after_reject", "on"), ("after_reject_score", "0.45")]),
            true,
            None,
            d,
        )
        .unwrap()
        .unwrap();
        assert_eq!(fin.reject_below, Some(0.45));
        // The score alone (box unticked) is no step.
        assert!(
            finish_from_form(&form(&[("after_reject_score", "0.5")]), false, None, d)
                .unwrap()
                .is_none()
        );
        // The folder switch alone, without auto-label.
        let fin = finish_from_form(&form(&[("after_review_folder", "on")]), false, Some(d), d)
            .unwrap()
            .unwrap();
        assert_eq!((fin.reject_below, fin.accept_auto), (None, false));
        for (f, chained, out, want) in [
            (
                form(&[("after_reject", "on")]),
                false,
                None,
                "needs Auto-label after import",
            ),
            (
                form(&[("after_accept", "on")]),
                false,
                None,
                "needs Auto-label after import",
            ),
            (
                form(&[("after_reject", "on"), ("after_reject_score", "0")]),
                true,
                None,
                "reject auto boxes scoring below: must be between 0.01 and 1",
            ),
            (
                form(&[("after_reject", "on"), ("after_reject_score", "1.5")]),
                true,
                None,
                "reject auto boxes scoring below",
            ),
            (
                form(&[("after_reject", "on"), ("after_reject_score", "low")]),
                true,
                None,
                "reject auto boxes scoring below: 'low' is not a number",
            ),
            (
                form(&[("after_review_folder", "on")]),
                true,
                None,
                "Save frames to",
            ),
        ] {
            let e = finish_from_form(&f, chained, out, d)
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{f:?}: {e}");
        }
    }

    #[test]
    fn a_job_after_an_import_says_so() {
        let rec = |id: &str, kind, state, after: Option<&str>| TaskRecord {
            id: id.into(),
            kind,
            title: String::new(),
            created: String::new(),
            state,
            stage: None,
            step: 0,
            steps: 1,
            percent: 0,
            detail: String::new(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
            after: after.map(str::to_string),
        };
        let tasks = vec![
            rec("imp", TaskKind::Import, RunState::Running, None),
            rec("al", TaskKind::Autolabel, RunState::Queued, Some("imp")),
            rec("al2", TaskKind::Autolabel, RunState::Queued, Some("al")),
            rec("al3", TaskKind::Autolabel, RunState::Queued, Some("gone")),
            rec("al4", TaskKind::Autolabel, RunState::Done, Some("imp")),
        ];
        let rows = task_rows(&tasks);
        let by = |id: &str| {
            rows.iter()
                .find(|r| r.id == id)
                .map(|r| r.describe.clone())
                .unwrap()
        };
        assert_eq!(by("al"), "queued (after import imp)");
        assert_eq!(by("al2"), "queued (after al)");
        assert_eq!(by("al3"), "queued (after gone)");
        assert!(by("al4").starts_with("done"), "{}", by("al4"));
    }

    #[tokio::test]
    async fn an_import_can_queue_an_auto_label_of_its_frames() {
        let _queue = super::super::training::QUEUE_TEST_LOCK.lock().await;
        use crate::server::test_util::{call, get_text, post_form, test_state_with};
        let root = std::env::temp_dir().join(format!("bop-train-chain-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("chainsnaps")).unwrap();
        let img = crate::image::RgbImage {
            width: 32,
            height: 24,
            rgb: vec![90; 32 * 24 * 3],
        };
        std::fs::write(
            root.join("chainsnaps/Gate.20240101_120000.jpg"),
            crate::image::encode_jpeg(&img, 90).unwrap(),
        )
        .unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let (_, page) = get_text(&state, "/training").await;
        assert!(page.contains("import-autolabel"), "the import card's box");
        assert!(page.contains("import-after"), "its follow-up options");
        let tasks = |state: Arc<AppState>| async move {
            let (_, json) = get_text(&state, "/v1/training").await;
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            v["tasks"].as_array().unwrap().clone()
        };
        let post = |state: Arc<AppState>, body: String| async move {
            let (_, _, b) = call(&state, post_form("/training/import", &body, true)).await;
            String::from_utf8(b).unwrap()
        };
        let src = format!(
            "source=path&path={}&confirm_license=on&select=dhash&frames=5",
            super::super::encode_uri_component(&root.join("chainsnaps").display().to_string())
        );

        // Bad auto-label settings, a follow-up without its option: nothing is queued.
        let before = tasks(state.clone()).await.len();
        for (extra, want) in [
            (
                "&autolabel=on&al.mode=labelers&al.gdino=off&al.vlm=off",
                "Not imported: auto-label",
            ),
            ("&autolabel=on&al.mode=model", "Not imported: auto-label"),
            ("&after_review_folder=on", "Save frames to"),
            ("&after_reject=on", "needs Auto-label after import"),
            (
                "&autolabel=on&al.mode=model&al.with_model=ipcam-general&after_reject=on\
                 &after_reject_score=2",
                "reject auto boxes scoring below",
            ),
        ] {
            let html = post(state.clone(), format!("{src}{extra}")).await;
            assert!(
                html.contains("Not imported") && html.contains(want),
                "{extra}: {html}"
            );
        }
        assert_eq!(tasks(state.clone()).await.len(), before, "nothing queued");

        // A model run needs no GPU: the import, then the auto-label of its frames.
        let html = post(
            state.clone(),
            format!(
                "{src}&autolabel=on&al.mode=model&al.with_model=ipcam-general&al.status=reviewed\
                 &al.limit=7&al.run_options=1&after_reject=on"
            ),
        )
        .await;
        assert!(
            html.contains("Queued: import of chainsnaps") && html.contains(", then label with"),
            "{html}"
        );
        let all = tasks(state.clone()).await;
        let imp = all
            .iter()
            .find(|t| {
                t["kind"] == "import"
                    && t["title"]
                        .as_str()
                        .is_some_and(|x| x.contains("import of chainsnaps"))
            })
            .unwrap()
            .clone();
        let imp_id = imp["id"].as_str().unwrap().to_string();
        let al = all
            .iter()
            .find(|t| t["after"] == imp_id.as_str())
            .unwrap()
            .clone();
        let al_id = al["id"].as_str().unwrap().to_string();
        assert_eq!(al["kind"], "autolabel");
        assert!(
            al["title"]
                .as_str()
                .unwrap()
                .contains(&format!("import {imp_id}")),
            "{al}"
        );
        // Cancel the chain (the import's cancel takes the auto-label with it while it waits);
        // the model run itself is not this test's business.
        for id in [&imp_id, &al_id] {
            let _ = call(
                &state,
                post_form("/training/cancel", &format!("id={id}"), true),
            )
            .await;
        }
        let mut settled = false;
        for _ in 0..400 {
            let all = tasks(state.clone()).await;
            let busy = all.iter().any(|t| {
                (t["id"] == imp_id.as_str() || t["id"] == al_id.as_str())
                    && ["queued", "running", "paused"].contains(&t["state"].as_str().unwrap_or(""))
            });
            if !busy {
                settled = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(settled, "the chain ended");
        let _ = std::fs::remove_dir_all(&root);
    }
}
