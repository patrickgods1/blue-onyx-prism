//! The Refine tab of the Train area (`GET /train/refine`, admin only): starts the dataset check
//! ([`crate::training::refine`]) and shows what it found. Three JSON routes serve the page
//! (`assets/train-refine.js`):
//!
//! - `GET /v1/refine`: the folder under review, the parsed `refine/report.json` (null when there is
//!   none, with `report_problem` when it is damaged or newer than this build), the open suggestions
//!   per kind (a walk of the label files, off the async workers, kept 60 s per folder and class
//!   list and dropped when a check starts or ends), the check in the queue, the newest finished one,
//!   and what the form needs of this machine;
//! - `GET /v1/refine/preview`: what a check would do ([`refine::preview`]), no GPU, download or
//!   write;
//! - `POST /v1/refine/check`: plans the check, hands it the registry's runtimes for the fold models'
//!   predictions (no second OpenVINO / ONNX Runtime instance) and queues it on the one-at-a-time
//!   queue. Refusals come back as `{success: false, message}` in the check's own words. A check
//!   that would replace open suggestions on the frames it checks asks first: 409 with
//!   `needs_confirm` and the counts, until the request says `replace_open: true`.
//!
//! The check writes suggestions and the report only; nothing here changes a box.

use super::train_area::{REFINE, tab_title};
use super::{AppState, Shared, blocking, render, training};
use crate::backend::Runtimes;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::queue::{self, Filters};
use crate::collect::labels::refine::{self as suggestions, Kind};
use crate::config::Config;
use crate::sync::LockExt;
use crate::training::capability::Capability;
use crate::training::eta::JobTiming;
use crate::training::refine::predict::LocalPredictor;
use crate::training::refine::report::{self, REPORT_VERSION, RefineReport};
use crate::training::refine::{self as check, RefineFrames, RefineJob, RefineRequest};
use crate::training::service::{RunRecord, RunState};
use crate::training::task::{Task, TaskKind, TaskRecord};
use askama::Template;
use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest request body of `POST /v1/refine/check`.
pub(super) const BODY_LIMIT: usize = 16 * 1024;
/// How long a walk's counts are kept.
const OPEN_TTL: Duration = Duration::from_secs(60);
/// Largest `report.json` read (a report with this many recall rows is not one this build wrote).
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message.into() })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------------------------
// The page

/// One option of the GPU select.
struct GpuOption {
    value: String,
    label: String,
}

/// The GPUs the form offers: only when there is a choice, as the Auto-label and Train forms do.
fn gpu_options(cap: &Capability) -> Vec<GpuOption> {
    if !cap.has_gpu_choice() {
        return Vec::new();
    }
    cap.cuda_options
        .iter()
        .map(|a| GpuOption {
            value: a.device_arg(),
            label: a.choice_label(),
        })
        .collect()
}

#[derive(Template)]
#[template(path = "train/refine.html")]
struct RefineTemplate {
    nav: &'static str,
    tab: &'static str,
    tab_title: &'static str,
    version: &'static str,
    scenes: [&'static str; 3],
    gpu_choices: Vec<GpuOption>,
    folds_min: usize,
    folds_max: usize,
    folds_default: usize,
    epochs_min: u32,
    epochs_max: u32,
    epochs_default: u32,
}

/// `GET /train/refine`. The cards are filled by `assets/train-refine.js` from `GET /v1/refine`,
/// so the page never waits for a walk of the label files.
pub(super) async fn page() -> Response {
    blocking(|| {
        render(&RefineTemplate {
            nav: "train",
            tab: REFINE,
            tab_title: tab_title(REFINE),
            version: crate::VERSION,
            scenes: crate::collect::layout::SCENES,
            gpu_choices: gpu_options(&crate::training::capability_now()),
            folds_min: *check::FOLDS.start(),
            folds_max: *check::FOLDS.end(),
            folds_default: check::DEFAULT_FOLDS,
            epochs_min: *check::EPOCHS.start(),
            epochs_max: *check::EPOCHS.end(),
            epochs_default: check::DEFAULT_EPOCHS,
        })
    })
    .await
}

// ---------------------------------------------------------------------------------------------
// GET /v1/refine

/// Live suggestions over the folder under review, per kind that exists today.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(super) struct Open {
    pub(super) add: usize,
    pub(super) verify: usize,
    pub(super) relabel: usize,
    /// Frames holding at least one.
    pub(super) frames: usize,
}

/// The open suggestions of the label files of `dir` ([`suggestions::live`]: answered, stale and
/// skipped-frame ones are not counted), judged against the class list as the Review routes are.
pub(super) fn count_open(dir: &Path, classes: &[String], hidden: &[String]) -> Open {
    let wanted = Wanted { classes, hidden };
    let per_frame = queue::labelled_map(dir, &Filters::default(), &|_| true, &|_rel, file| {
        let live = suggestions::live(&file, Some(&wanted));
        if live.is_empty() {
            return None;
        }
        let mut n = [0usize; 3];
        for l in &live {
            match l.suggestion.kind {
                Kind::Add => n[0] += 1,
                Kind::Verify => n[1] += 1,
                Kind::Relabel => n[2] += 1,
                Kind::Tighten | Kind::Unknown => {}
            }
        }
        (n.iter().sum::<usize>() > 0).then_some(n)
    });
    let mut open = Open::default();
    for n in per_frame {
        open.add += n[0];
        open.verify += n[1];
        open.relabel += n[2];
        open.frames += 1;
    }
    open
}

struct Kept {
    at: Instant,
    stamp: String,
    open: Open,
}

/// The walks' answers per folder and class list.
static OPEN: Mutex<BTreeMap<String, Kept>> = Mutex::new(BTreeMap::new());
/// One walk at a time: a second caller waits and finds the first one's answer.
static WALKING: Mutex<()> = Mutex::new(());

/// [`count_open`], kept [`OPEN_TTL`]. `stamp` names the checks on the queue (ids and states): when
/// one starts, ends or is cancelled the kept answer is dropped.
fn open_counts(dir: &Path, classes: &[String], hidden: &[String], stamp: &str) -> Open {
    let key = format!(
        "{}\u{1f}{}\u{1f}{}",
        dir.display(),
        classes.join("\u{1e}"),
        hidden.join("\u{1e}")
    );
    let fresh = |k: &Kept| k.at.elapsed() < OPEN_TTL && k.stamp == stamp;
    if let Some(k) = OPEN.lock_recover().get(&key).filter(|k| fresh(k)) {
        return k.open.clone();
    }
    let _walking = WALKING.lock_recover();
    if let Some(k) = OPEN.lock_recover().get(&key).filter(|k| fresh(k)) {
        return k.open.clone();
    }
    let open = count_open(dir, classes, hidden);
    OPEN.lock_recover().insert(
        key,
        Kept {
            at: Instant::now(),
            stamp: stamp.to_string(),
            open: open.clone(),
        },
    );
    open
}

/// The report in `dir` and what is wrong with it: `(None, None)` when there is none (never an
/// error), `(None, Some(why))` when it does not read or is a later build's.
pub(super) fn read_report(dir: &Path) -> (Option<RefineReport>, Option<String>) {
    let path = report::report_path(dir);
    let bytes = match std::fs::metadata(&path) {
        Ok(m) if m.len() > MAX_REPORT_BYTES => {
            return (
                None,
                Some(format!(
                    "{} is too large to be a report of the dataset check ({} bytes)",
                    path.display(),
                    m.len()
                )),
            );
        }
        Ok(_) => match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                return (
                    None,
                    Some(format!("{} could not be read: {e}", path.display())),
                );
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (None, None),
        Err(e) => {
            return (
                None,
                Some(format!("{} could not be read: {e}", path.display())),
            );
        }
    };
    match serde_json::from_slice::<RefineReport>(&bytes) {
        Ok(r) if r.version > REPORT_VERSION => (
            None,
            Some(format!(
                "{} was written by a newer version of this program (report format {}, this one \
                 reads up to {REPORT_VERSION}); update to see it",
                path.display(),
                r.version
            )),
        ),
        Ok(r) => (Some(r), None),
        Err(e) => (
            None,
            Some(format!(
                "{} is damaged and was not read ({e}); the next check writes a new one",
                path.display()
            )),
        ),
    }
}

/// The check in the queue (running, paused or queued) as the Jobs list words it, plus its state;
/// None when there is none.
pub(super) fn job_of(
    runs: &[RunRecord],
    tasks: &[TaskRecord],
    timing: &dyn Fn(&str) -> Option<JobTiming>,
) -> Option<serde_json::Value> {
    let rec = tasks
        .iter()
        .find(|t| t.kind == TaskKind::Refine && t.state.is_busy())?;
    let jobs = super::train_jobs::view(runs, tasks, timing);
    let line = jobs
        .active
        .iter()
        .chain(jobs.queued.iter())
        .find(|l| l.id == rec.id)?;
    let mut v = serde_json::to_value(line).ok()?;
    if let Some(o) = v.as_object_mut() {
        o.insert("state".into(), rec.state.as_str().into());
    }
    Some(v)
}

/// The newest check that is over: how it ended, in the words the queue recorded.
fn last_of(tasks: &[TaskRecord]) -> Option<serde_json::Value> {
    let t = tasks
        .iter()
        .rev()
        .find(|t| t.kind == TaskKind::Refine && !t.state.is_busy())?;
    Some(serde_json::json!({
        "id": t.id,
        "state": t.state.as_str(),
        "title": t.title,
        "summary": t.summary,
        "error": t.error,
        "finished": t.finished,
    }))
}

/// Ids and states of the checks on the queue: the kept counts are dropped when it changes.
fn queue_stamp(tasks: &[TaskRecord]) -> String {
    tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Refine)
        .map(|t| format!("{}:{}", t.id, t.state.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

/// What the form needs of this machine.
fn capability_bits(cap: &Capability) -> serde_json::Value {
    serde_json::json!({
        "can_check": cap.accelerator.is_some(),
        "why": cap.accelerator.is_none().then(|| cap.summary()),
        "gpus": gpu_options(cap)
            .into_iter()
            .map(|g| serde_json::json!({ "value": g.value, "label": g.label }))
            .collect::<Vec<_>>(),
    })
}

/// The answer of `GET /v1/refine`.
fn status_of(state: &AppState, cap: &Capability) -> serde_json::Value {
    let config = state.config_read().clone();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let runs = training::records(state);
    let tasks = training::tasks(state);
    let mut v = serde_json::json!({
        "success": true,
        "review_folder": dir.display().to_string(),
        "folder_missing": false,
        "report": null,
        "open": null,
        "job": job_of(&runs, &tasks, &training::job_timing),
        "last": last_of(&tasks),
        "capability": capability_bits(cap),
        "limits": {
            "folds": [check::FOLDS.start(), check::FOLDS.end()],
            "epochs": [check::EPOCHS.start(), check::EPOCHS.end()],
        },
    });
    let Some(o) = v.as_object_mut() else {
        return v;
    };
    // An unplugged drive is never an empty dataset: nothing is read from it.
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        o.insert("folder_missing".into(), true.into());
        o.insert("message".into(), why.into());
        return v;
    }
    let (report, problem) = read_report(&dir);
    o.insert(
        "report".into(),
        serde_json::to_value(report).unwrap_or_default(),
    );
    if let Some(p) = problem {
        o.insert("report_problem".into(), p.into());
    }
    let open = open_counts(
        &dir,
        &config.collect.classes,
        &config.collect.hidden_classes,
        &queue_stamp(&tasks),
    );
    o.insert(
        "open".into(),
        serde_json::to_value(open).unwrap_or_default(),
    );
    v
}

/// `GET /v1/refine`: see the module docs.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let cap = crate::training::capability_now();
        Json(status_of(&state, &cap)).into_response()
    })
    .await
}

// ---------------------------------------------------------------------------------------------
// The request

/// A check's request from the page's fields; the bounds are [`RefineRequest::validate`]'s.
fn build_request(
    folds: Option<usize>,
    epochs: Option<u32>,
    scenes: &[String],
    cameras: &[String],
    gpu: Option<String>,
    allow_large: bool,
) -> Result<RefineRequest, String> {
    let mut frames = RefineFrames::default();
    for s in scenes.iter().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        // The CLI's parse words the error.
        let one = RefineFrames::parse(&format!("scene={s}")).map_err(|e| format!("{e:#}"))?;
        frames.scenes.extend(one.scenes);
    }
    for c in cameras.iter().map(|c| c.trim()).filter(|c| !c.is_empty()) {
        if c.len() > 255 || c.chars().any(char::is_control) {
            return Err(
                "a camera name is a folder name: up to 255 characters, no control \
                        characters"
                    .into(),
            );
        }
        frames.cameras.push(c.to_string());
    }
    frames.scenes.sort();
    frames.scenes.dedup();
    frames.cameras.sort();
    frames.cameras.dedup();
    let request = RefineRequest {
        folds: folds.unwrap_or(check::DEFAULT_FOLDS),
        epochs: epochs.unwrap_or(check::DEFAULT_EPOCHS),
        frames,
        gpu: gpu.map(|g| g.trim().to_string()).filter(|g| !g.is_empty()),
        allow_large,
    };
    request.validate().map_err(|e| format!("{e:#}"))?;
    Ok(request)
}

/// Query of `GET /v1/refine/preview`: every field optional and read here, so a bad one is named
/// in the answer (`scene` and `camera` are comma lists).
#[derive(Debug, Default, Deserialize)]
pub(super) struct PreviewQuery {
    #[serde(default)]
    folds: Option<String>,
    #[serde(default)]
    epochs: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    gpu: Option<String>,
    #[serde(default)]
    allow_large: Option<String>,
}

fn whole<T: std::str::FromStr>(field: &str, v: Option<&str>) -> Result<Option<T>, String> {
    match v.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|_| format!("{field} must be a whole number, got '{v}'")),
    }
}

fn comma_list(v: Option<&str>) -> Vec<String> {
    v.map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default()
}

impl PreviewQuery {
    fn request(&self) -> Result<RefineRequest, String> {
        build_request(
            whole("folds", self.folds.as_deref())?,
            whole("epochs", self.epochs.as_deref())?,
            &comma_list(self.scene.as_deref()),
            &comma_list(self.camera.as_deref()),
            self.gpu.clone(),
            matches!(
                self.allow_large.as_deref().map(str::trim),
                Some("1" | "true" | "on" | "yes")
            ),
        )
    }
}

/// `GET /v1/refine/preview?folds=&epochs=&scene=&camera=&gpu=&allow_large=`: 200
/// `{success: true, ...}` ([`check::RefinePreview`]; `refused` says why a start would be refused,
/// `folder_missing` that the review folder is unplugged), or 400 `{success: false, message}` for a
/// bad field. Reads every label file of the folder, so the page asks on a click only.
pub(super) async fn preview(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PreviewQuery>,
) -> Response {
    blocking(move || {
        let request = match q.request() {
            Ok(r) => r,
            Err(m) => return fail(StatusCode::BAD_REQUEST, m),
        };
        let config = state.config_read().clone();
        let cap = crate::training::capability_now();
        match check::preview(&config, &request, &cap) {
            Ok(p) => {
                let mut v = serde_json::to_value(&p).unwrap_or_default();
                if let Some(o) = v.as_object_mut() {
                    o.insert("success".into(), true.into());
                    o.insert("text".into(), check::preview_text(&p).into());
                }
                Json(v).into_response()
            }
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        }
    })
    .await
}

/// Body of `POST /v1/refine/check`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckBody {
    #[serde(default)]
    folds: Option<usize>,
    #[serde(default)]
    epochs: Option<u32>,
    #[serde(default)]
    scenes: Vec<String>,
    #[serde(default)]
    cameras: Vec<String>,
    #[serde(default)]
    gpu: Option<String>,
    #[serde(default)]
    allow_large: bool,
    /// The user confirmed that the check replaces the open suggestions of the frames it checks
    /// (the second click after a `needs_confirm` refusal).
    #[serde(default)]
    replace_open: bool,
}

/// The JSON body of a POST: 413 over the cap, 415 without a JSON content type, 400 when it does
/// not parse (as the Review routes' own reader).
fn json_body<T: for<'de> Deserialize<'de>>(
    headers: &HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<T, (StatusCode, String)> {
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
    serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("the request is not valid: {e}"),
        )
    })
}

/// Why a check was not queued.
#[derive(Debug, PartialEq, Eq)]
struct Refusal {
    status: StatusCode,
    message: String,
    /// The folder under review is unplugged.
    folder_missing: bool,
    /// The check would replace open suggestions and the request did not say `replace_open`:
    /// (open suggestions, frames holding them).
    needs_confirm: Option<(usize, usize)>,
}

impl Refusal {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            folder_missing: false,
            needs_confirm: None,
        }
    }
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let body = match self.needs_confirm {
            Some((open, frames)) => serde_json::json!({
                "success": false,
                "needs_confirm": true,
                "open_suggestions": open,
                "open_frames": frames,
                "folder_missing": self.folder_missing,
                "message": self.message,
            }),
            None => serde_json::json!({
                "success": false,
                "folder_missing": self.folder_missing,
                "message": self.message,
            }),
        };
        (self.status, Json(body)).into_response()
    }
}

/// The page's words for a refusal the CLI words with its flag.
fn for_the_page(message: &str) -> String {
    message.replace("--allow-large", "\"Allow downloads over 500 MB\"")
}

/// Plan `request` and queue it through `enqueue`, which gets the job with its predictor set
/// (`runtimes`: the registry's). Refuses, before anything is queued, when a check is already
/// queued or running, the folder under review is unplugged, the frames could not be checked
/// ([`check::preview`]'s `refused`: too few reviewed frames, ...), the check would replace open
/// suggestions without `replace_open` (409 with `needs_confirm`: the preview's count, so no
/// second walk) or the plan fails (no GPU to train on, a large download not allowed). Returns the
/// task id and title.
fn start_check(
    config: &Config,
    request: RefineRequest,
    replace_open: bool,
    cap: &Capability,
    tasks: &[TaskRecord],
    runtimes: Option<Arc<Mutex<Runtimes>>>,
    enqueue: &dyn Fn(RefineJob) -> Result<String, String>,
) -> Result<(String, String), Refusal> {
    if let Some(t) = tasks
        .iter()
        .find(|t| t.kind == TaskKind::Refine && t.state.is_busy())
    {
        return Err(Refusal {
            status: StatusCode::CONFLICT,
            message: format!(
                "Not queued: a dataset check is already {} ({}). Wait for it, or cancel it \
                 first.",
                if t.state == RunState::Queued {
                    "queued"
                } else {
                    "running"
                },
                t.title
            ),
            folder_missing: false,
            needs_confirm: None,
        });
    }
    let dir = config.collect.resolved_review_dir(&config.data_root());
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        return Err(Refusal {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: format!("Not queued: {why}"),
            folder_missing: true,
            needs_confirm: None,
        });
    }
    let p = check::preview(config, &request, cap)
        .map_err(|e| Refusal::bad(format!("Not queued: {e:#}")))?;
    if let Some(why) = p.refused {
        return Err(Refusal::bad(format!("Not queued: {}", for_the_page(&why))));
    }
    // A rerun replaces the open suggestions of the frames it checks: asked once, never silently.
    if p.open_suggestions > 0 && !replace_open {
        return Err(Refusal {
            status: StatusCode::CONFLICT,
            message: format!(
                "Not queued: {}.",
                check::replace_warning(p.open_suggestions, p.open_frames)
            ),
            folder_missing: false,
            needs_confirm: Some((p.open_suggestions, p.open_frames)),
        });
    }
    let mut job = check::plan_refine(config, request, cap)
        .map_err(|e| Refusal::bad(format!("Not queued: {}", for_the_page(&format!("{e:#}")))))?;
    job.predictor = Some(Arc::new(LocalPredictor::new(config.clone(), runtimes)));
    let title = Task::title(&job);
    let id = enqueue(job).map_err(|e| Refusal::bad(format!("Not queued: {e}")))?;
    Ok((id, title))
}

/// `POST /v1/refine/check` with `{folds?, epochs?, scenes?, cameras?, gpu?, allow_large?,
/// replace_open?}`: queue the dataset check; 200 `{success: true, id, message}` (the task id, in
/// `GET /v1/train/jobs` and `GET /v1/training` `tasks[]`), or a refusal `{success: false, message,
/// folder_missing}`: 400 for a bad field, too few frames or a plan that cannot run, 409 for a check
/// already in the queue, 503 for an unplugged review folder. A check that would replace open
/// suggestions on the frames it checks is 409 `{success: false, needs_confirm: true,
/// open_suggestions, open_frames, folder_missing: false, message}` unless `replace_open` is true;
/// nothing is queued then.
pub(super) async fn start(
    State(shared): State<Shared>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: CheckBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    blocking(move || {
        let request = match build_request(
            b.folds,
            b.epochs,
            &b.scenes,
            &b.cameras,
            b.gpu,
            b.allow_large,
        ) {
            Ok(r) => r,
            Err(m) => return fail(StatusCode::BAD_REQUEST, m),
        };
        let state = shared.current();
        let config = state.config_read().clone();
        let cap = crate::training::capability_now();
        let tasks = training::tasks(&state);
        let queue = training::service(&shared);
        let runtimes = state.registry.runtimes().cloned();
        match start_check(
            &config,
            request,
            b.replace_open,
            &cap,
            &tasks,
            runtimes,
            &|job| queue.enqueue_task(Box::new(job)),
        ) {
            Ok((id, title)) => Json(serde_json::json!({
                "success": true,
                "id": id,
                "message": format!(
                    "Queued: {title}. It writes suggestions only: no box changes until you \
                     confirm one on the Review tab."
                ),
            }))
            .into_response(),
            Err(r) => r.into_response(),
        }
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, get_text, test_state_with};
    use super::*;
    use crate::collect::labels::bulk::test_support::put_frame_at;
    use crate::collect::labels::refine::{Answer, Suggestion};
    use crate::collect::labels::{self, BoxState, Frame, LabelBox, Source, Status};
    use axum::body::Body;
    use axum::http::Request;
    use std::path::PathBuf;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_refine_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A state whose dataset, training and download folders are `dir`'s.
    fn state_in(dir: &Path) -> Arc<AppState> {
        let dir = dir.to_path_buf();
        test_state_with(move |c| {
            c.download_dir = Some(dir.clone());
            c.collect.dir = Some(dir.join("dataset"));
            c.training.dir = Some(dir.join("training"));
        })
    }

    async fn json(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let (s, _, b) = call(state, req).await;
        (
            s,
            serde_json::from_slice(&b).unwrap_or(serde_json::Value::Null),
        )
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn car(bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            label: "car".into(),
            bbox,
            source: Source::Auto,
            score: Some(0.8),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    const CAR: [f32; 4] = [4.0, 4.0, 24.0, 20.0];
    const FAR: [f32; 4] = [40.0, 24.0, 60.0, 44.0];

    fn suggestion(kind: Kind, target: Option<[f32; 4]>, bbox: Option<[f32; 4]>) -> Suggestion {
        Suggestion {
            kind,
            target,
            target_label: target.map(|_| "car".into()),
            bbox,
            label: Some("car".into()),
            score: Some(0.7),
            why: "a fold model disagreed".into(),
            spot: None,
            answer: None,
            run: "r1".into(),
        }
    }

    /// A reviewed frame of `camera` with a car and `list` on it; its rel.
    fn frame_with(dir: &Path, camera: &str, stem: &str, list: Vec<Suggestion>) -> String {
        let rel = put_frame_at(dir, camera, stem, (64, 48), None, vec![car(CAR)]);
        let path = Frame::resolve(dir, &rel).unwrap().labels;
        let mut f = labels::read(&path).unwrap().unwrap();
        f.status = Status::Reviewed;
        f.reviewed_at = Some("2026-10-22T00:00:00Z".into());
        f.refine = list;
        labels::write(&path, &f).unwrap();
        rel
    }

    fn report_json() -> serde_json::Value {
        let mut r = RefineReport::new(
            "run1",
            4,
            check::recipe_info(
                &crate::training::recipe::TrainRecipe::new(
                    crate::resources::catalog::TrainFamily::Rfdetr,
                    "nano",
                )
                .unwrap(),
                "the default",
            ),
        );
        r.folds_done = 4;
        r.loose = 7;
        r.warnings = vec!["fold 2 of 4 trains on fewer than 50 boxes of dog (3)".into()];
        let mut tally = report::RecallTally::default();
        tally.add(
            "patio",
            "day",
            &[crate::training::refine::judge::BoxFact {
                class: "person".into(),
                short_px: 30.0,
                found: true,
            }],
        );
        r.finish(&tally);
        serde_json::to_value(&r).unwrap()
    }

    fn write_report(dir: &Path, text: &str) {
        let p = report::report_path(dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    fn task(id: &str, state: RunState) -> TaskRecord {
        TaskRecord {
            id: id.into(),
            kind: TaskKind::Refine,
            title: "Dataset check of all reviewed frames (4 folds)".into(),
            created: String::new(),
            state,
            stage: Some("fold 1".into()),
            step: 2,
            steps: 6,
            percent: 30,
            detail: "epoch 3 of 8".into(),
            log: vec![],
            summary: Some("0 suggestions".into()),
            result: None,
            error: None,
            finished: None,
            after: None,
            uses: Vec::new(),
        }
    }

    // ---- the page

    #[tokio::test]
    async fn the_page_renders_signed_in_and_sends_anonymous_browsers_to_sign_in() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, html) = get_text(&state, "/train/refine").await;
        assert_eq!(s, StatusCode::OK);
        for want in [
            "id=\"refine-form\"",
            "name=\"folds\" min=\"2\" max=\"8\" step=\"1\" value=\"4\"",
            "name=\"epochs\" min=\"1\" max=\"50\" step=\"1\" value=\"8\"",
            "id=\"refine-preview-btn\"",
            "id=\"refine-findings\"",
            "id=\"refine-coverage\"",
            "/static/train-refine.js?v=",
        ] {
            assert!(html.contains(want), "{want}");
        }
        // No cards for what comes later.
        for gone in ["static object", "size floor", "Tighten"] {
            assert!(!html.contains(gone), "{gone}");
        }
        let anon = super::super::test_util::anonymous(get("/train/refine"));
        let (s, h, _) = call(&state, anon).await;
        assert!(
            s == StatusCode::SEE_OTHER || s == StatusCode::UNAUTHORIZED,
            "{s}"
        );
        if s == StatusCode::SEE_OTHER {
            assert!(
                h[header::LOCATION].to_str().unwrap().starts_with("/login")
                    || h[header::LOCATION].to_str().unwrap().starts_with("/setup")
            );
        }
    }

    // ---- GET /v1/refine

    #[tokio::test]
    async fn without_a_report_the_answer_has_none_and_no_problem() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        assert!(
            v["report"].is_null() && v.get("report_problem").is_none(),
            "{v}"
        );
        assert_eq!(v["folder_missing"], false);
        assert_eq!(
            v["open"],
            serde_json::json!({"add":0,"verify":0,"relabel":0,"frames":0})
        );
        assert_eq!(v["limits"]["folds"], serde_json::json!([2, 8]));
        assert!(v["capability"]["can_check"].is_boolean());
        assert!(v["review_folder"].as_str().unwrap().ends_with("dataset"));
    }

    #[tokio::test]
    async fn a_report_is_parsed_and_a_damaged_one_is_null_with_a_problem() {
        let dir = scratch();
        let state = state_in(&dir);
        let review = dir.join("dataset");
        std::fs::create_dir_all(&review).unwrap();
        write_report(&review, &report_json().to_string());
        let (_, v) = json(&state, get("/v1/refine")).await;
        let r = &v["report"];
        assert_eq!(
            (r["folds"].as_u64(), r["folds_done"].as_u64()),
            (Some(4), Some(4)),
            "{v}"
        );
        assert_eq!(r["loose"], 7);
        assert_eq!(r["recall_by_group"][0]["camera"], "patio");
        assert_eq!(r["recall_by_group"][0]["boxes"], 1);
        assert_eq!(r["recall_by_size"].as_array().unwrap().len(), 5);
        assert_eq!(r["warnings"].as_array().unwrap().len(), 1);
        assert!(v.get("report_problem").is_none());

        write_report(&review, "{ this is not json");
        let (s, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(s, StatusCode::OK);
        assert!(v["report"].is_null());
        assert!(
            v["report_problem"].as_str().unwrap().contains("damaged"),
            "{v}"
        );

        // A later build's format is named, not guessed at.
        let mut newer = report_json();
        newer["version"] = (REPORT_VERSION + 1).into();
        write_report(&review, &newer.to_string());
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert!(v["report"].is_null());
        assert!(
            v["report_problem"].as_str().unwrap().contains("newer"),
            "{v}"
        );
    }

    #[tokio::test]
    async fn open_suggestions_are_counted_per_kind_and_answered_or_stale_ones_are_not() {
        let dir = scratch();
        let state = state_in(&dir);
        let review = dir.join("dataset");
        let add = || suggestion(Kind::Add, None, Some(FAR));
        let verify = || suggestion(Kind::Verify, Some(CAR), None);
        let relabel = || {
            let mut s = suggestion(Kind::Relabel, Some(CAR), None);
            s.label = Some("dog".into());
            s
        };
        frame_with(&review, "patio", "a1", vec![add(), verify()]);
        frame_with(&review, "patio", "a2", vec![relabel()]);
        // Answered: not open.
        let mut done = add();
        done.answer = Some(Answer::Dismissed);
        frame_with(&review, "front", "b1", vec![done]);
        // Stale: its target box is not where the suggestion says.
        frame_with(
            &review,
            "front",
            "b2",
            vec![suggestion(Kind::Verify, Some([1.0, 1.0, 9.0, 9.0]), None)],
        );
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(
            v["open"],
            serde_json::json!({"add": 1, "verify": 1, "relabel": 1, "frames": 2}),
            "{v}"
        );
    }

    #[test]
    fn counts_are_kept_until_the_queue_changes() {
        let dir = scratch();
        let review = dir.join("dataset");
        frame_with(
            &review,
            "patio",
            "a1",
            vec![suggestion(Kind::Add, None, Some(FAR))],
        );
        let classes = vec!["car".to_string()];
        let none: Vec<String> = Vec::new();
        assert_eq!(open_counts(&review, &classes, &none, "").add, 1);
        // A second frame is not seen inside the 60 s with the same queue ...
        frame_with(
            &review,
            "patio",
            "a2",
            vec![suggestion(Kind::Add, None, Some(FAR))],
        );
        assert_eq!(open_counts(&review, &classes, &none, "").add, 1);
        // ... but a check starting or ending drops it.
        assert_eq!(open_counts(&review, &classes, &none, "t1:running").add, 2);
    }

    #[tokio::test]
    async fn an_unavailable_review_folder_is_said_and_nothing_is_counted() {
        let dir = scratch();
        let d = dir.clone();
        let state = test_state_with(move |c| {
            c.download_dir = Some(d.clone());
            c.collect.review_dir = Some(d.join("unplugged"));
            c.training.dir = Some(d.join("training"));
        });
        let (s, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["folder_missing"], true, "{v}");
        assert!(v["message"].as_str().unwrap().contains("not available"));
        assert!(v["open"].is_null() && v["report"].is_null());
    }

    #[test]
    fn the_job_row_is_the_jobs_list_line_and_a_finished_check_is_last() {
        let tasks = [
            task("old", RunState::Failed),
            task("now", RunState::Running),
        ];
        let v = job_of(&[], &tasks, &|_| None).unwrap();
        assert_eq!(v["id"], "now");
        assert_eq!(v["state"], "running");
        assert_eq!(v["link"], "/train/refine#job-now");
        assert_eq!(v["kind"], "Dataset check");
        assert_eq!(v["percent"], 30);
        assert!(job_of(&[], &tasks[..1], &|_| None).is_none());
        let last = last_of(&tasks[..1]).unwrap();
        assert_eq!(
            (last["id"].as_str(), last["state"].as_str()),
            (Some("old"), Some("failed"))
        );
        assert!(last_of(&tasks[1..]).is_none());
        assert_ne!(queue_stamp(&tasks[..1]), queue_stamp(&tasks));
        let mut other = task("x", RunState::Running);
        other.kind = TaskKind::Autolabel;
        assert!(job_of(&[], &[other], &|_| None).is_none());
    }

    // ---- the request

    #[test]
    fn a_request_is_built_from_the_fields_and_checked() {
        let r = build_request(
            None,
            None,
            &["ir".into(), " day ".into(), "ir".into()],
            &["patio".into(), "".into(), "front".into()],
            Some(" ".into()),
            true,
        )
        .unwrap();
        assert_eq!((r.folds, r.epochs), (4, 8));
        assert_eq!(r.frames.scenes, ["day", "ir"]);
        assert_eq!(r.frames.cameras, ["front", "patio"]);
        assert!(r.gpu.is_none() && r.allow_large);
        for (folds, epochs, scene) in [
            (Some(1), None, "day"),
            (Some(9), None, "day"),
            (None, Some(0), "day"),
            (None, Some(51), "day"),
            (None, None, "dusk"),
        ] {
            assert!(
                build_request(folds, epochs, &[scene.into()], &[], None, false).is_err(),
                "{folds:?} {epochs:?} {scene}"
            );
        }
        let q = PreviewQuery {
            folds: Some("3".into()),
            scene: Some("ir,day".into()),
            camera: Some("a,b".into()),
            allow_large: Some("1".into()),
            ..PreviewQuery::default()
        };
        let r = q.request().unwrap();
        assert_eq!(r.folds, 3);
        assert_eq!(r.frames.cameras, ["a", "b"]);
        assert!(r.allow_large);
        let bad = PreviewQuery {
            folds: Some("x".into()),
            ..PreviewQuery::default()
        };
        assert!(
            bad.request()
                .unwrap_err()
                .contains("folds must be a whole number")
        );
    }

    // ---- GET /v1/refine/preview

    #[tokio::test]
    async fn the_preview_refuses_few_frames_in_the_checks_words_and_a_bad_field_is_a_400() {
        let dir = scratch();
        let state = state_in(&dir);
        let review = dir.join("dataset");
        frame_with(&review, "patio", "a1", vec![]);
        let (s, v) = json(&state, get("/v1/refine/preview")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        assert_eq!(v["frames"], 1);
        assert!(
            v["refused"]
                .as_str()
                .unwrap()
                .contains("at least 40 reviewed frames"),
            "{v}"
        );
        assert!(v["text"].as_str().unwrap().contains("refused:"));
        let (s, v) = json(&state, get("/v1/refine/preview?folds=9")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["success"], false);
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("folds must be between 2 and 8")
        );
        let (s, _) = json(&state, get("/v1/refine/preview?scene=dusk")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    /// 44 reviewed frames on four cameras (four camera days): enough for a check.
    fn enough_frames(review: &Path) {
        for cam in ["alpha", "bravo", "charlie", "delta"] {
            for i in 0..11 {
                frame_with(review, cam, &format!("f{i:02}"), vec![]);
            }
        }
    }

    #[tokio::test]
    async fn the_preview_of_a_dataset_that_can_be_checked_lists_its_folds() {
        let dir = scratch();
        let state = state_in(&dir);
        enough_frames(&dir.join("dataset"));
        let (s, v) = json(&state, get("/v1/refine/preview?folds=2&epochs=3")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["frames"], 44, "{v}");
        assert_eq!(v["camera_days"], 4);
        assert!(v["refused"].is_null(), "{v}");
        assert_eq!(v["folds"].as_array().unwrap().len(), 2);
        let total: u64 = v["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["frames"].as_u64().unwrap())
            .sum();
        assert_eq!(total, 44);
        assert_eq!(v["recipe"]["epochs"], 3);
        assert!(v["recipe"]["from"].as_str().unwrap().len() > 3);
    }

    // ---- POST /v1/refine/check

    fn host_config(dir: &Path) -> Config {
        let mut c = Config {
            download_dir: Some(dir.to_path_buf()),
            allow_large_downloads: true,
            ..Config::default()
        };
        c.seed_default_model();
        c.collect.dir = Some(dir.join("dataset"));
        c
    }

    fn host() -> Capability {
        crate::training::capability::host_for_tests()
    }

    #[test]
    fn a_check_is_planned_given_its_predictor_and_queued_once() {
        let dir = scratch();
        enough_frames(&dir.join("dataset"));
        let config = host_config(&dir);
        let seen = Mutex::new(Vec::new());
        let enqueue = |job: RefineJob| {
            assert!(
                job.predictor.is_some(),
                "the job runs on the registry's predictor"
            );
            assert_eq!(Task::kind(&job), TaskKind::Refine);
            seen.lock()
                .unwrap()
                .push((job.request.clone(), job.id.clone()));
            Ok(job.id.clone())
        };
        let req = RefineRequest {
            folds: 2,
            epochs: 3,
            ..RefineRequest::default()
        };
        let (id, title) =
            start_check(&config, req.clone(), false, &host(), &[], None, &enqueue).unwrap();
        assert!(
            title.starts_with("Dataset check") || title.to_lowercase().contains("check"),
            "{title}"
        );
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!((seen[0].0.folds, seen[0].0.epochs), (2, 3));
        assert_eq!(seen[0].1, id);
        drop(seen);

        // One at a time: a queued or running check refuses the next, a finished one does not.
        for state in [RunState::Queued, RunState::Running, RunState::Paused] {
            let e = start_check(
                &config,
                req.clone(),
                false,
                &host(),
                &[task("t1", state)],
                None,
                &enqueue,
            )
            .unwrap_err();
            assert_eq!(e.status, StatusCode::CONFLICT, "{state:?}");
            assert!(e.message.contains("already"), "{}", e.message);
        }
        assert!(
            start_check(
                &config,
                req,
                false,
                &host(),
                &[task("t1", RunState::Done)],
                None,
                &enqueue
            )
            .is_ok()
        );
    }

    #[test]
    fn a_check_that_cannot_run_is_refused_before_it_is_queued() {
        let dir = scratch();
        let never = |_: RefineJob| -> Result<String, String> { panic!("queued") };
        // Too few frames.
        let config = host_config(&dir);
        frame_with(&dir.join("dataset"), "patio", "a1", vec![]);
        let e = start_check(
            &config,
            RefineRequest::default(),
            false,
            &host(),
            &[],
            None,
            &never,
        )
        .unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(
            e.message
                .starts_with("Not queued: the dataset check needs at least 40"),
            "{}",
            e.message
        );
        // An unplugged folder.
        let mut gone = host_config(&dir);
        gone.collect.review_dir = Some(dir.join("unplugged"));
        let e = start_check(
            &gone,
            RefineRequest::default(),
            false,
            &host(),
            &[],
            None,
            &never,
        )
        .unwrap_err();
        assert_eq!(
            (e.status, e.folder_missing),
            (StatusCode::SERVICE_UNAVAILABLE, true)
        );
        // The queue's own refusal comes through.
        enough_frames(&dir.join("dataset"));
        let e = start_check(
            &config,
            RefineRequest::default(),
            false,
            &host(),
            &[],
            None,
            &|_| Err("the training thread has stopped".into()),
        )
        .unwrap_err();
        assert!(
            e.message.contains("the training thread has stopped"),
            "{}",
            e.message
        );
        // Large downloads: not allowed and the environment not set up -> refused in the page's words.
        let mut strict = host_config(&dir);
        strict.allow_large_downloads = false;
        let e = start_check(
            &strict,
            RefineRequest::default(),
            false,
            &host(),
            &[],
            None,
            &never,
        )
        .unwrap_err();
        assert!(!e.message.contains("--allow-large"), "{}", e.message);
    }

    /// [`enough_frames`] with open suggestions on two of them: 3 open on 2 frames, plus an
    /// answered one (not open).
    fn enough_frames_with_open(review: &Path) {
        enough_frames(review);
        let add = || suggestion(Kind::Add, None, Some(FAR));
        frame_with(
            review,
            "alpha",
            "f00",
            vec![add(), suggestion(Kind::Verify, Some(CAR), None)],
        );
        frame_with(review, "bravo", "f00", vec![add()]);
        let mut answered = add();
        answered.answer = Some(Answer::Dismissed);
        frame_with(review, "charlie", "f00", vec![answered]);
    }

    #[test]
    fn a_check_over_open_suggestions_asks_first_and_queues_with_replace_open() {
        let dir = scratch();
        let review = dir.join("dataset");
        let config = host_config(&dir);
        let never = |_: RefineJob| -> Result<String, String> { panic!("queued") };
        let queued = Mutex::new(0);
        let enqueue = |job: RefineJob| {
            *queued.lock().unwrap() += 1;
            Ok(job.id.clone())
        };
        let req = RefineRequest {
            folds: 2,
            epochs: 1,
            ..RefineRequest::default()
        };
        // Nothing open (an answered suggestion is not): no question, no flag needed.
        enough_frames(&review);
        let mut answered = suggestion(Kind::Add, None, Some(FAR));
        answered.answer = Some(Answer::Dismissed);
        frame_with(&review, "charlie", "f00", vec![answered]);
        start_check(&config, req.clone(), false, &host(), &[], None, &enqueue).unwrap();
        assert_eq!(*queued.lock().unwrap(), 1);
        // Open ones: refused with the counts, nothing queued.
        enough_frames_with_open(&review);
        let e = start_check(&config, req.clone(), false, &host(), &[], None, &never).unwrap_err();
        assert_eq!(e.status, StatusCode::CONFLICT);
        assert_eq!(e.needs_confirm, Some((3, 2)));
        assert!(!e.folder_missing);
        assert_eq!(
            e.message,
            format!("Not queued: {}.", check::replace_warning(3, 2))
        );
        // Confirmed: queued as before.
        start_check(&config, req.clone(), true, &host(), &[], None, &enqueue).unwrap();
        assert_eq!(*queued.lock().unwrap(), 2);
        // A check already in the queue is said first: the confirmation would not help.
        let e = start_check(
            &config,
            req,
            false,
            &host(),
            &[task("t1", RunState::Running)],
            None,
            &never,
        )
        .unwrap_err();
        assert_eq!((e.status, e.needs_confirm), (StatusCode::CONFLICT, None));
    }

    #[tokio::test]
    async fn the_route_answers_needs_confirm_and_takes_replace_open() {
        let dir = scratch();
        let d = dir.clone();
        let state = test_state_with(move |c| {
            c.download_dir = Some(d.clone());
            c.collect.dir = Some(d.join("dataset"));
            c.training.dir = Some(d.join("training"));
            // Nothing can be planned here: a confirmed start stops at the plan, never queued.
            c.allow_large_downloads = false;
        });
        enough_frames_with_open(&dir.join("dataset"));
        let (s, v) = json(&state, post_json("/v1/refine/check", r#"{"folds":2}"#)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(
            v,
            serde_json::json!({
                "success": false,
                "needs_confirm": true,
                "open_suggestions": 3,
                "open_frames": 2,
                "folder_missing": false,
                "message": format!("Not queued: {}.", check::replace_warning(3, 2)),
            })
        );
        // `replace_open: true` passes the question: what refuses now is the plan (no training
        // environment, downloads not allowed, or no GPU), never the question again.
        let (s, v) = json(
            &state,
            post_json("/v1/refine/check", r#"{"folds":2,"replace_open":true}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(v.get("needs_confirm").is_none(), "{v}");
        assert!(
            !training::tasks(&state)
                .iter()
                .any(|t| t.kind == TaskKind::Refine),
            "nothing queued"
        );
        // `replace_open: false` is the default; a wrong type or an unknown field is still a 400.
        let (s, v) = json(
            &state,
            post_json("/v1/refine/check", r#"{"replace_open":false}"#),
        )
        .await;
        assert_eq!(
            (s, v["needs_confirm"].as_bool()),
            (StatusCode::CONFLICT, Some(true))
        );
        for body in [
            r#"{"replace_open":"yes"}"#,
            r#"{"replace_open":true,"replace":true}"#,
        ] {
            let (s, v) = json(&state, post_json("/v1/refine/check", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(v.get("needs_confirm").is_none(), "{body}: {v}");
        }
    }

    #[tokio::test]
    async fn the_preview_route_says_what_a_start_would_replace() {
        let dir = scratch();
        let state = state_in(&dir);
        enough_frames_with_open(&dir.join("dataset"));
        let (s, v) = json(&state, get("/v1/refine/preview?folds=2&epochs=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["open_suggestions"].as_u64(), v["open_frames"].as_u64()),
            (Some(3), Some(2))
        );
        let warnings: Vec<&str> = v["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        assert_eq!(warnings[0], check::replace_warning(3, 2));
        assert!(
            warnings[1].starts_with("1 epoch per fold model is under half"),
            "{warnings:?}"
        );
        assert!(
            v["text"]
                .as_str()
                .unwrap()
                .contains("warning: this check replaces the 3")
        );
    }

    #[tokio::test]
    async fn the_route_refuses_bad_input_and_few_frames_and_needs_a_sign_in_and_a_same_site_post() {
        let dir = scratch();
        let state = state_in(&dir);
        frame_with(&dir.join("dataset"), "patio", "a1", vec![]);
        // Bad bounds, unknown field, not JSON, wrong type.
        for (body, status) in [
            (r#"{"folds":1}"#, StatusCode::BAD_REQUEST),
            (r#"{"epochs":99}"#, StatusCode::BAD_REQUEST),
            (r#"{"scenes":["dusk"]}"#, StatusCode::BAD_REQUEST),
            (r#"{"nope":1}"#, StatusCode::BAD_REQUEST),
            ("not json", StatusCode::BAD_REQUEST),
        ] {
            let (s, v) = json(&state, post_json("/v1/refine/check", body)).await;
            assert_eq!(s, status, "{body}: {v}");
            assert_eq!(v["success"], false, "{body}");
            assert!(
                v["message"].as_str().is_some_and(|m| !m.is_empty()),
                "{body}"
            );
        }
        let form = super::super::test_util::post_form("/v1/refine/check", "folds=4", false);
        let (s, _) = json(&state, form).await;
        assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // One reviewed frame: refused in the check's own words (before any GPU question).
        let (s, v) = json(&state, post_json("/v1/refine/check", "{}")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("at least 40 reviewed frames"),
            "{v}"
        );
        // Anonymous: 401, not a refusal of the check.
        let anon = super::super::test_util::anonymous(post_json("/v1/refine/check", "{}"));
        let (s, _) = json(&state, anon).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        // A cross-site POST is refused by the CSRF check, whatever the body.
        let cross = Request::post("/v1/refine/check")
            .header(header::CONTENT_TYPE, "application/json")
            .header("host", "localhost:32168")
            .header("origin", "http://evil.example")
            .body(Body::from("{}"))
            .unwrap();
        let (s, _) = json(&state, cross).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }

    /// The model entry list of the Models tab never gets a fold model: nothing in this module
    /// writes the config.
    #[test]
    fn nothing_here_writes_the_config() {
        let src = include_str!("refine.rs");
        let code = src.split("#[cfg(test)]").next().unwrap();
        for word in ["save_config", "ConfigStore", "config_store"] {
            assert!(!code.contains(word), "{word}");
        }
    }

    /// The Jobs tab links a check to its own tab.
    #[test]
    fn a_check_belongs_to_this_tab() {
        assert_eq!(super::super::train_area::task_tab(TaskKind::Refine), REFINE);
    }
}
