//! The Refine tab of the Train area (`GET /train/refine`, admin only): starts the dataset check
//! ([`crate::training::refine`]) and shows what it found. Three JSON routes serve the page
//! (`assets/train-refine.js`):
//!
//! - `GET /v1/refine`: the folder under review, the parsed `refine/report.json` of the last full
//!   check (`report`; null when there is none, with `report_problem` when it is damaged or newer
//!   than this build) and `refine/report-quick.json` of the last quick check (`quick_report`,
//!   `quick_report_problem`), the open suggestions
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
//! A quick check (`model` in the preview's query and the start's body, never with folds or epochs;
//! [`check::quick`]) runs a model trained here on the reviewed frames it never saw: `GET /v1/refine`
//! `capability.models` lists the usable ones (read from their training records, no frame walk, kept
//! a few seconds), and
//! the replace count of its 409 covers only the frames it checks.
//!
//! The check writes suggestions and the report only; nothing here changes a box. The Static
//! objects card's routes (`GET /v1/refine/spots`, `POST /v1/refine/accept-spot`,
//! `POST /v1/refine/static-cap`) are in [`super::refine_spots`], the one place of this tab that
//! writes boxes (through [`suggestions::accept`]) or the config.

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
use crate::training::service::{NotQueued, RunRecord, RunState};
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
/// Most walks' answers kept at once (folders under review x class lists).
const OPEN_KEEP: usize = 8;
/// Largest `report.json` read (a report with this many recall rows is not one this build wrote).
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn fail(status: StatusCode, message: impl Into<String>) -> Response {
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
    /// The "Propose better boxes for loose labels" box starts ticked: RF-DETR L's weights are on
    /// disk ([`check::tighten_weights_ready`]), the check's own default.
    tighten_default: bool,
}

/// `GET /train/refine`. The cards are filled by `assets/train-refine.js` from `GET /v1/refine`,
/// so the page never waits for a walk of the label files.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let config = state.config_read().clone();
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
            tighten_default: check::tighten_weights_ready(&config),
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
    /// Better-fitting boxes for loose labels (the check's closer look).
    pub(super) tighten: usize,
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
        let mut n = [0usize; 4];
        for l in &live {
            match l.suggestion.kind {
                Kind::Add => n[0] += 1,
                Kind::Verify => n[1] += 1,
                Kind::Relabel => n[2] += 1,
                Kind::Tighten => n[3] += 1,
                Kind::Unknown => {}
            }
        }
        (n.iter().sum::<usize>() > 0).then_some(n)
    });
    let mut open = Open::default();
    for n in per_frame {
        open.add += n[0];
        open.verify += n[1];
        open.relabel += n[2];
        open.tighten += n[3];
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
    remember(
        &mut OPEN.lock_recover(),
        key,
        Kept {
            at: Instant::now(),
            stamp: stamp.to_string(),
            open: open.clone(),
        },
    );
    open
}

/// Drop every kept count: a label write (the static spots' bulk accept) changed what a walk
/// would find, so the next read walks again.
pub(super) fn forget_counts() {
    OPEN.lock_recover().clear();
}

/// Put `kept` in `map` under `key`: answers past [`OPEN_TTL`] go first, then the oldest, so at
/// most [`OPEN_KEEP`] stay (each folder under review and class list ever asked about would
/// otherwise stay for the life of the process).
fn remember(map: &mut BTreeMap<String, Kept>, key: String, kept: Kept) {
    map.retain(|k, v| *k != key && v.at.elapsed() < OPEN_TTL);
    while map.len() >= OPEN_KEEP {
        let Some(oldest) = map.iter().min_by_key(|(_, v)| v.at).map(|(k, _)| k.clone()) else {
            break;
        };
        map.remove(&oldest);
    }
    map.insert(key, kept);
}

/// The full check's report in `dir` and what is wrong with it: `(None, None)` when there is none
/// (never an error), `(None, Some(why))` when it does not read or is a later build's.
pub(super) fn read_report(dir: &Path) -> (Option<RefineReport>, Option<String>) {
    read_report_at(&report::report_path(dir))
}

/// The quick check's report in `dir` (`refine/report-quick.json`), as [`read_report`] reads.
pub(super) fn read_quick_report(dir: &Path) -> (Option<RefineReport>, Option<String>) {
    read_report_at(&report::quick_report_path(dir))
}

/// The report the recall tables are drawn from (the Coverage and Too small cards, the coverage
/// route's recall columns, the floor's suggested input): the full check's when one reads, else the
/// quick check's; each says which it is by its `mode`.
pub(super) fn recall_report(dir: &Path) -> Option<RefineReport> {
    read_report(dir).0.or_else(|| read_quick_report(dir).0)
}

/// The report at `path`, as [`read_report`] reads it.
fn read_report_at(path: &Path) -> (Option<RefineReport>, Option<String>) {
    let bytes = match std::fs::metadata(path) {
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
        Ok(_) => match std::fs::read(path) {
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
    // The format first, alone: a later build's report may not parse as this build's at all, and
    // is then newer, not damaged.
    #[derive(Deserialize)]
    struct Format {
        version: u64,
    }
    if let Ok(Format { version }) = serde_json::from_slice::<Format>(&bytes)
        && version > u64::from(REPORT_VERSION)
    {
        return (
            None,
            Some(format!(
                "{} was written by a newer version of this program (report format {version}, \
                 this one reads up to {REPORT_VERSION}); update to see it",
                path.display(),
            )),
        );
    }
    match serde_json::from_slice::<RefineReport>(&bytes) {
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
pub(super) fn queue_stamp(tasks: &[TaskRecord]) -> String {
    tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Refine)
        .map(|t| format!("{}:{}", t.id, t.state.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

/// What the form needs of this machine; `tighten_default`: the closer look's weights are on disk,
/// so a check looks closer at its loose labels unless told not to. `can_check` is the full check's
/// (it trains, so it needs a GPU); a quick check runs its model on the model's own device, so
/// `models` (the configured models a quick check can use, [`check::quick::models`], newest first:
/// `{name, run, recipe, imgsz, enabled}`) is offered without one, and `unusable` names the models
/// with a training record it cannot use, with why. The frames each never saw are counted on Preview
/// (a walk of the label files). The list is kept a few seconds ([`quick_models`]).
fn capability_bits(
    cap: &Capability,
    tighten_default: bool,
    config: &Config,
    revision: u64,
) -> serde_json::Value {
    let (models, unusable) = quick_models(&MODELS, config, revision, MODELS_TTL);
    serde_json::json!({
        "can_check": cap.accelerator.is_some(),
        "tighten_default": tighten_default,
        "why": cap.accelerator.is_none().then(|| cap.summary()),
        "gpus": gpu_options(cap)
            .into_iter()
            .map(|g| serde_json::json!({ "value": g.value, "label": g.label }))
            .collect::<Vec<_>>(),
        "models": models,
        "unusable": unusable,
    })
}

/// How long the quick check's models list is kept: `GET /v1/refine` is read on every load of the
/// Refine and Dataset tabs, and the list reads `runs.json` and every model's training record.
const MODELS_TTL: Duration = Duration::from_secs(5);

/// The kept models list ([`quick_models`]): one answer, so it stays small.
struct KeptModels {
    at: Instant,
    key: String,
    value: (Vec<check::quick::QuickModel>, Vec<check::quick::Unusable>),
}

/// The process's kept models list.
static MODELS: Mutex<Option<KeptModels>> = Mutex::new(None);

/// What the kept list is valid for: the config's revision and data root, and each configured
/// model's training record as a stat sees it (size, time), so a record written or removed between
/// two reads is never served stale (a run folder or model file that goes is, for [`MODELS_TTL`]).
fn models_key(config: &Config, revision: u64) -> String {
    let mut key = format!("{revision}\u{1f}{}", config.data_root().display());
    for m in &config.models {
        let record = check::quick::record_path(config, m);
        let stamp = std::fs::metadata(&record)
            .map(|md| format!("{}:{:?}", md.len(), md.modified().ok()))
            .unwrap_or_else(|_| "-".into());
        key.push_str(&format!("\u{1f}{}={stamp}", record.display()));
    }
    key
}

/// [`check::quick::models`] of `config` (at `revision`), kept `ttl` ([`MODELS_TTL`]) in `slot`
/// while [`models_key`] is the same.
fn quick_models(
    slot: &Mutex<Option<KeptModels>>,
    config: &Config,
    revision: u64,
    ttl: Duration,
) -> (Vec<check::quick::QuickModel>, Vec<check::quick::Unusable>) {
    let key = models_key(config, revision);
    if let Some(k) = slot
        .lock_recover()
        .as_ref()
        .filter(|k| k.key == key && k.at.elapsed() < ttl)
    {
        return k.value.clone();
    }
    let value = check::quick::models(config);
    *slot.lock_recover() = Some(KeptModels {
        at: Instant::now(),
        key,
        value: value.clone(),
    });
    value
}

/// The answer of `GET /v1/refine`: the full check's report (`report`, `report_problem`) and the
/// quick check's (`quick_report`, `quick_report_problem`), each its mode's last one.
fn status_of(state: &AppState, cap: &Capability) -> serde_json::Value {
    let revision = state.config.revision();
    let config = state.config_read().clone();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let runs = training::records(state);
    let tasks = training::tasks(state);
    let mut v = serde_json::json!({
        "success": true,
        "review_folder": dir.display().to_string(),
        "folder_missing": false,
        "report": null,
        "quick_report": null,
        "open": null,
        "job": job_of(&runs, &tasks, &training::job_timing),
        "last": last_of(&tasks),
        "capability": capability_bits(
            cap,
            check::tighten_weights_ready(&config),
            &config,
            revision,
        ),
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
    for (key, (report, problem)) in [
        ("report", read_report(&dir)),
        ("quick_report", read_quick_report(&dir)),
    ] {
        o.insert(key.into(), serde_json::to_value(report).unwrap_or_default());
        if let Some(p) = problem {
            o.insert(format!("{key}_problem"), p.into());
        }
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

/// A check's fields as the page sends them.
#[derive(Debug, Default)]
struct Fields<'a> {
    folds: Option<usize>,
    epochs: Option<u32>,
    scenes: &'a [String],
    cameras: &'a [String],
    gpu: Option<String>,
    allow_large: bool,
    tighten: Option<bool>,
    /// A quick check's model; empty = none.
    model: Option<String>,
}

/// A check's request from the page's fields; the bounds are [`RefineRequest::validate`]'s. A quick
/// check (`model`) trains nothing, so folds or epochs with it are a mistake, said as one.
fn build_request(f: Fields<'_>) -> Result<RefineRequest, String> {
    let Fields {
        folds,
        epochs,
        scenes,
        cameras,
        gpu,
        allow_large,
        tighten,
        model,
    } = f;
    let model = model
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty());
    if model.is_some() && (folds.is_some() || epochs.is_some()) {
        return Err(
            "folds and epochs are for the full check: a quick check with a model trains nothing"
                .into(),
        );
    }
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
        tighten,
        model,
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
    /// `1` / `0`: the closer look at loose labels asked for or not; absent: the check's default.
    #[serde(default)]
    tighten: Option<String>,
    /// A quick check with this configured model; absent or empty: the full check.
    #[serde(default)]
    model: Option<String>,
}

/// A yes/no query field: `1`, `true`, `on`, `yes` or `0`, `false`, `off`, `no`; absent or empty:
/// None; anything else an error naming `field`.
fn yes_no(field: &str, v: Option<&str>) -> Result<Option<bool>, String> {
    match v.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some("1" | "true" | "on" | "yes") => Ok(Some(true)),
        Some("0" | "false" | "off" | "no") => Ok(Some(false)),
        Some(v) => Err(format!("{field} must be 1 or 0, got '{v}'")),
    }
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
        build_request(Fields {
            folds: whole("folds", self.folds.as_deref())?,
            epochs: whole("epochs", self.epochs.as_deref())?,
            scenes: &comma_list(self.scene.as_deref()),
            cameras: &comma_list(self.camera.as_deref()),
            gpu: self.gpu.clone(),
            allow_large: matches!(
                self.allow_large.as_deref().map(str::trim),
                Some("1" | "true" | "on" | "yes")
            ),
            tighten: yes_no("tighten", self.tighten.as_deref())?,
            model: self.model.clone(),
        })
    }
}

/// `GET /v1/refine/preview?folds=&epochs=&scene=&camera=&gpu=&allow_large=&tighten=&model=`: 200
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
    /// The closer look at loose labels (`true` / `false`); absent: the check's default (on when
    /// RF-DETR L's weights are on disk).
    #[serde(default)]
    tighten: Option<bool>,
    /// The user confirmed that the check replaces the open suggestions of the frames it checks
    /// (the second click after a `needs_confirm` refusal).
    #[serde(default)]
    replace_open: bool,
    /// A quick check with this configured model (no folds or epochs with it); absent: the full
    /// check.
    #[serde(default)]
    model: Option<String>,
}

/// The JSON body of a POST: 413 over the cap, 415 without a JSON content type, 400 when it does
/// not parse (as the Review routes' own reader).
pub(super) fn json_body<T: for<'de> Deserialize<'de>>(
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
    /// (open suggestions of the folds' kinds, open `tighten` the closer look may replace, frames
    /// holding them).
    needs_confirm: Option<(usize, usize, usize)>,
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
            Some((open, tightens, frames)) => serde_json::json!({
                "success": false,
                "needs_confirm": true,
                "open_suggestions": open,
                "open_tightens": tightens,
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

/// The rule a queue checks under its own lock before it takes a job
/// ([`crate::training::service::TrainingService::enqueue_task_unless`]): a reason to refuse, or
/// None.
type QueueRule<'a> = &'a dyn Fn(&[TaskRecord]) -> Option<String>;

/// How [`start_check`] queues a job: with the rule the queue must check under its own lock.
type Enqueue<'a> = &'a dyn Fn(RefineJob, QueueRule<'_>) -> Result<String, NotQueued>;

/// "A check is already queued or running" for the task list `tasks`, in the page's words; None
/// when no check is.
fn busy_message(tasks: &[TaskRecord]) -> Option<String> {
    let t = tasks
        .iter()
        .find(|t| t.kind == TaskKind::Refine && t.state.is_busy())?;
    Some(format!(
        "Not queued: a dataset check is already {} ({}). Wait for it, or cancel it first.",
        if t.state == RunState::Queued {
            "queued"
        } else {
            "running"
        },
        t.title
    ))
}

fn conflict(message: String) -> Refusal {
    Refusal {
        status: StatusCode::CONFLICT,
        message,
        folder_missing: false,
        needs_confirm: None,
    }
}

/// Plan `request` and queue it through `enqueue`, which gets the job with its predictor set
/// (`runtimes`: the registry's). Refuses, before anything is queued, when a check is already
/// queued or running, the folder under review is unplugged, the frames could not be checked
/// ([`check::preview`]'s `refused`: too few reviewed frames, ...), the check would replace open
/// suggestions without `replace_open` (409 with `needs_confirm`: the preview's count, so no
/// second walk) or the plan fails (no GPU to train on, a large download not allowed). Returns the
/// task id and title.
///
/// `tasks` is a copy taken before the walks of the preview and the plan (seconds): a check
/// another request queued meanwhile is not in it. So "a check is already queued" is asked twice:
/// first on `tasks` (no walk for nothing), then by the queue itself under its lock, together with
/// the queueing ([`busy_message`] as the [`QueueRule`]), so two starts at once queue one check.
fn start_check(
    config: &Config,
    request: RefineRequest,
    replace_open: bool,
    cap: &Capability,
    tasks: &[TaskRecord],
    runtimes: Option<Arc<Mutex<Runtimes>>>,
    enqueue: Enqueue<'_>,
) -> Result<(String, String), Refusal> {
    if let Some(m) = busy_message(tasks) {
        return Err(conflict(m));
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
    // A rerun replaces the open suggestions of the frames it checks (with the closer look, also
    // open `tighten` of the labels it answers): asked once, never silently.
    if p.open_suggestions + p.open_tightens > 0 && !replace_open {
        return Err(Refusal {
            status: StatusCode::CONFLICT,
            message: format!("Not queued: {}.", p.replace_text()),
            folder_missing: false,
            needs_confirm: Some((p.open_suggestions, p.open_tightens, p.open_frames)),
        });
    }
    let mut job = check::plan_refine(config, request, cap)
        .map_err(|e| Refusal::bad(format!("Not queued: {}", for_the_page(&format!("{e:#}")))))?;
    job.predictor = Some(Arc::new(LocalPredictor::new(config.clone(), runtimes)));
    let title = Task::title(&job);
    let id = enqueue(job, &busy_message).map_err(|e| match e {
        NotQueued::Refused(m) => conflict(m),
        NotQueued::Failed(m) => Refusal::bad(format!("Not queued: {m}")),
    })?;
    Ok((id, title))
}

/// `POST /v1/refine/check` with `{folds?, epochs?, scenes?, cameras?, gpu?, allow_large?,
/// replace_open?}`: queue the dataset check; 200 `{success: true, id, message}` (the task id, in
/// `GET /v1/train/jobs` and `GET /v1/training` `tasks[]`), or a refusal `{success: false, message,
/// folder_missing}`: 400 for a bad field, too few frames or a plan that cannot run, 409 for a check
/// already in the queue, 503 for an unplugged review folder. A check that would replace open
/// suggestions on the frames it checks (with the closer look, open `tighten` too: `open_tightens`,
/// a bound) is 409 `{success: false, needs_confirm: true, open_suggestions, open_tightens,
/// open_frames, folder_missing: false, message}` unless `replace_open` is true; nothing is queued
/// then.
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
        let request = match build_request(Fields {
            folds: b.folds,
            epochs: b.epochs,
            scenes: &b.scenes,
            cameras: &b.cameras,
            gpu: b.gpu,
            allow_large: b.allow_large,
            tighten: b.tighten,
            model: b.model,
        }) {
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
            &|job, rule| queue.enqueue_task_unless(Box::new(job), rule),
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
    use crate::training::refine::report::Mode;
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
                at_spot: false,
                under_floor: false,
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
            "id=\"refine-thin-body\"",
            "<h3>Where the data is thin</h3>",
            "id=\"refine-statics\"",
            "id=\"refine-cap-form\"",
            "name=\"cap\" min=\"0\" max=\"100000\" step=\"1\"",
            "id=\"refine-spot-confirm\"",
            "id=\"refine-floor\"",
            "id=\"refine-floor-form\"",
            "name=\"px\" min=\"1\" max=\"64\"",
            "name=\"input\" min=\"64\" max=\"4096\" step=\"1\"",
            "id=\"refine-floor-preview\"",
            "id=\"refine-floor-off\"",
            "id=\"refine-small-confirm\"",
            "id=\"refine-floor-restore\"",
            "id=\"refine-ignore-form\"",
            "name=\"mode\" value=\"background\"",
            "name=\"mode\" value=\"drop-frame\"",
            "/static/train-refine.js?v=",
            // RF-DETR L's weights are not here: the box starts unticked and says it downloads them.
            "<input type=\"checkbox\" id=\"refine-tighten\" name=\"tighten\"> Propose better boxes \
             for loose labels</label>",
            "downloads RF-DETR L (0.14 GB) the first time",
            // The mode choice: the full check by default, the quick check's model select hidden,
            // folds and epochs the full check's only.
            "<input type=\"radio\" name=\"check_mode\" value=\"folds\" checked> Full check (fold \
             models, about an hour)</label>",
            "<input type=\"radio\" name=\"check_mode\" value=\"model\" id=\"refine-mode-model\"> \
             Quick check with a model you trained (only frames it never saw)</label>",
            "<div class=\"tr-field\" id=\"refine-model-field\" hidden>",
            "<select id=\"refine-model\" name=\"model\"></select>",
        ] {
            assert!(html.contains(want), "{want}");
        }
        assert_eq!(
            html.matches("data-full-only").count(),
            2,
            "folds and epochs"
        );
        let folds = html.find("id=\"refine-folds\"").unwrap();
        let wrapper = html[..folds].rfind("<div").unwrap();
        assert!(html[wrapper..folds].contains("data-full-only"));
        // The Static objects card sits between Findings and Coverage.
        let at = |id: &str| html.find(id).unwrap();
        assert!(
            at("id=\"refine-findings\"") < at("id=\"refine-statics\"")
                && at("id=\"refine-statics\"") < at("id=\"refine-floor\"")
                && at("id=\"refine-floor\"") < at("id=\"refine-coverage\""),
        );
        // No cards for what comes later.
        assert!(!html.contains("Tighten"));
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

    /// RF-DETR L's weights on disk under `dir` (every part at its pinned size).
    fn put_zoom_weights(dir: &Path) {
        let config = Config {
            download_dir: Some(dir.to_path_buf()),
            ..Config::default()
        };
        let paths = crate::resources::export::train::TrainPaths::for_config(&config);
        let (res, at) = crate::training::overlaps::rfdetr_weights(&paths).unwrap();
        std::fs::create_dir_all(&at).unwrap();
        for p in res.parts {
            std::fs::File::create(at.join(p.file_name))
                .unwrap()
                .set_len(p.size)
                .unwrap();
        }
    }

    #[tokio::test]
    async fn the_look_is_ticked_when_its_weights_are_here_and_the_routes_take_the_field() {
        let dir = scratch();
        put_zoom_weights(&dir);
        let state = state_in(&dir);
        let (_, html) = get_text(&state, "/train/refine").await;
        assert!(
            html.contains(
                "<input type=\"checkbox\" id=\"refine-tighten\" name=\"tighten\" checked> \
                 Propose better boxes for loose labels"
            ),
            "{html}"
        );
        assert!(!html.contains("downloads RF-DETR L"));
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(v["capability"]["tighten_default"], true, "{v}");
        // The preview follows the default, or the field.
        enough_frames(&dir.join("dataset"));
        let (_, v) = json(&state, get("/v1/refine/preview?folds=2")).await;
        assert_eq!(v["tighten"], true, "{v}");
        let (_, v) = json(&state, get("/v1/refine/preview?folds=2&tighten=0")).await;
        assert_eq!(v["tighten"], false, "{v}");
        assert!(
            v["text"]
                .as_str()
                .unwrap()
                .contains("loose labels: counted only")
        );
        let (s, _) = json(&state, get("/v1/refine/preview?tighten=maybe")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        // A start takes a boolean, nothing else.
        let (s, v) = json(
            &state,
            post_json("/v1/refine/check", r#"{"tighten":"yes"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        let config = host_config(&dir);
        let seen = Mutex::new(Vec::new());
        let enqueue = |job: RefineJob, _: QueueRule<'_>| {
            seen.lock().unwrap().push(job.request.tighten);
            Ok(job.id.clone())
        };
        for asked in [None, Some(false), Some(true)] {
            let req = RefineRequest {
                folds: 2,
                epochs: 3,
                tighten: asked,
                ..RefineRequest::default()
            };
            start_check(&config, req, true, &host(), &[], None, &enqueue).unwrap();
        }
        assert_eq!(
            *seen.lock().unwrap(),
            [Some(true), Some(false), Some(true)],
            "the planned job holds the answer"
        );
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
            serde_json::json!({"add":0,"verify":0,"relabel":0,"tighten":0,"frames":0})
        );
        // RF-DETR L's weights are not here: the form's box starts unticked.
        assert_eq!(v["capability"]["tighten_default"], false);
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

    /// A report of `mode` written by `RefineReport::write` (as the job writes it), created at
    /// `created`, with `boxes` person boxes of the patio by day in its recall.
    fn written(review: &Path, mode: Mode, created: &str, boxes: u64) {
        let mut r: RefineReport = serde_json::from_value(report_json()).unwrap();
        let mut tally = report::RecallTally::default();
        let facts: Vec<_> = (0..boxes)
            .map(|_| crate::training::refine::judge::BoxFact {
                class: "person".into(),
                short_px: 30.0,
                found: true,
                at_spot: false,
                under_floor: false,
            })
            .collect();
        tally.add("patio", "day", &facts);
        r.finish(&tally);
        r.created = created.into();
        r.mode = mode;
        r.write(review).unwrap();
    }

    /// Each mode keeps its own last report: a quick check never replaces the full check's (whose
    /// recall the cards keep showing), nor the reverse.
    #[tokio::test]
    async fn the_full_and_the_quick_check_keep_their_own_last_report() {
        let dir = scratch();
        let state = state_in(&dir);
        let review = dir.join("dataset");
        std::fs::create_dir_all(&review).unwrap();
        // Quick only.
        written(&review, Mode::Model, "2026-11-10T10:00:00Z", 3);
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert!(v["report"].is_null(), "{v}");
        assert_eq!(v["quick_report"]["mode"], "model");
        assert!(!report::report_path(&review).exists());
        assert_eq!(recall_report(&review).unwrap().recall.boxes, 3);
        // Full after quick: both kept; the recall tables are the full check's.
        written(&review, Mode::Folds, "2026-11-11T10:00:00Z", 40);
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(v["report"]["mode"], "folds");
        assert_eq!(v["report"]["recall"]["boxes"], 40);
        assert_eq!(v["quick_report"]["recall"]["boxes"], 3);
        // Quick after full: still both, and the recall stays the full check's.
        written(&review, Mode::Model, "2026-11-12T10:00:00Z", 5);
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(v["report"]["recall"]["boxes"], 40, "{v}");
        assert_eq!(v["quick_report"]["recall"]["boxes"], 5);
        let r = recall_report(&review).unwrap();
        assert_eq!((r.mode, r.recall.boxes), (Mode::Folds, 40));
        // Each has its own problem string.
        std::fs::write(report::quick_report_path(&review), "{ broken").unwrap();
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert!(v["quick_report"].is_null());
        assert!(
            v["quick_report_problem"]
                .as_str()
                .unwrap()
                .contains("report-quick.json"),
            "{v}"
        );
        assert!(v.get("report_problem").is_none());
        assert_eq!(v["report"]["recall"]["boxes"], 40);
        // A report.json of a quick check (the first builds wrote it there) reads as what it says.
        std::fs::remove_file(report::quick_report_path(&review)).unwrap();
        written(&review, Mode::Folds, "2026-11-12T11:00:00Z", 1);
        let mut old: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report::report_path(&review)).unwrap()).unwrap();
        old["mode"] = "model".into();
        std::fs::write(report::report_path(&review), old.to_string()).unwrap();
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(v["report"]["mode"], "model");
    }

    /// The models list is kept a few seconds per config revision and training records, so the
    /// Dataset and Refine tabs' loads do not read every record each time.
    #[test]
    fn the_models_list_is_kept_until_the_revision_or_a_record_changes() {
        let f = check::quick::tests::fixture("kept", false);
        let config = check::quick::tests::config(&f);
        // A long life: the test never depends on how fast it runs.
        const HOUR: Duration = Duration::from_secs(3600);
        let slot = Mutex::new(None);
        let (first, _) = quick_models(&slot, &config, 1, HOUR);
        assert_eq!(first.len(), 1);
        // The run folder goes: still the kept answer (within the few seconds) ...
        let run = f.root.join("training").join(check::quick::tests::RUN);
        let moved = f.root.join("moved-run");
        std::fs::rename(&run, &moved).unwrap();
        assert_eq!(quick_models(&slot, &config, 1, HOUR).0.len(), 1);
        // ... until the revision changes.
        let (ok, bad) = quick_models(&slot, &config, 2, HOUR);
        assert!(ok.is_empty() && bad.len() == 1, "{ok:?} {bad:?}");
        std::fs::rename(&moved, &run).unwrap();
        assert!(quick_models(&slot, &config, 2, HOUR).0.is_empty(), "kept");
        // A record removed is never served stale.
        std::fs::remove_file(f.root.join("models").join("mine.provenance.json")).unwrap();
        let (ok, bad) = quick_models(&slot, &config, 2, HOUR);
        assert!(ok.is_empty() && bad.is_empty());
    }

    /// The format is read alone first: a later build's report that does not parse as this
    /// build's is newer, not damaged; this format's report that does not parse is damaged.
    #[test]
    fn a_newer_report_that_does_not_parse_here_is_newer_and_this_formats_is_damaged() {
        let dir = scratch();
        let mut newer = report_json();
        newer["version"] = (REPORT_VERSION + 1).into();
        newer["folds"] = "a field of another shape".into();
        newer.as_object_mut().unwrap().remove("recall_by_size");
        write_report(&dir, &newer.to_string());
        let (r, why) = read_report(&dir);
        let why = why.unwrap();
        assert!(r.is_none());
        assert!(
            why.contains("newer version") && why.contains("update to see it"),
            "{why}"
        );
        assert!(!why.contains("damaged"), "{why}");

        let mut broken = report_json();
        broken["folds"] = "a field of another shape".into();
        write_report(&dir, &broken.to_string());
        let (r, why) = read_report(&dir);
        assert!(r.is_none());
        assert!(why.unwrap().contains("damaged"));
        // No format at all, or not JSON: damaged.
        for text in [r#"{"folds": 4}"#, r#"{"version": "two"}"#, "[1, 2"] {
            write_report(&dir, text);
            let (r, why) = read_report(&dir);
            assert!(r.is_none(), "{text}");
            assert!(why.unwrap().contains("damaged"), "{text}");
        }
        // This format's report reads.
        write_report(&dir, &report_json().to_string());
        assert!(matches!(read_report(&dir), (Some(_), None)));
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
        frame_with(
            &review,
            "patio",
            "a2",
            vec![
                relabel(),
                suggestion(Kind::Tighten, Some(CAR), Some([5.0, 5.0, 23.0, 19.0])),
            ],
        );
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
            serde_json::json!({"add": 1, "verify": 1, "relabel": 1, "tighten": 1, "frames": 2}),
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

    #[test]
    fn at_most_a_handful_of_answers_are_kept_and_the_oldest_go_first() {
        let mut map = BTreeMap::new();
        let base = Instant::now();
        let kept = |secs: u64| Kept {
            at: base - OPEN_TTL + Duration::from_secs(secs),
            stamp: String::new(),
            open: Open::default(),
        };
        // One expired answer goes at the next insert whatever the room.
        map.insert(
            "expired".to_string(),
            Kept {
                at: base - OPEN_TTL - Duration::from_secs(1),
                ..kept(0)
            },
        );
        for i in 0..(OPEN_KEEP as u64 + 5) {
            remember(&mut map, format!("k{i:02}"), kept(30 + i));
            assert!(map.len() <= OPEN_KEEP, "{}", map.len());
        }
        assert!(!map.contains_key("expired"));
        let left: Vec<&String> = map.keys().collect();
        let want: Vec<String> = (5..(OPEN_KEEP as u64 + 5))
            .map(|i| format!("k{i:02}"))
            .collect();
        assert_eq!(left, want.iter().collect::<Vec<_>>(), "the newest stay");
        // The same key again replaces its answer, it does not push another out.
        remember(&mut map, "k12".into(), kept(59));
        assert_eq!(map.len(), OPEN_KEEP);
        assert!(map.contains_key("k05"));
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
        let r = build_request(Fields {
            scenes: &["ir".into(), " day ".into(), "ir".into()],
            cameras: &["patio".into(), "".into(), "front".into()],
            gpu: Some(" ".into()),
            allow_large: true,
            tighten: Some(false),
            ..Fields::default()
        })
        .unwrap();
        assert_eq!((r.folds, r.epochs), (4, 8));
        assert_eq!(r.frames.scenes, ["day", "ir"]);
        assert_eq!(r.frames.cameras, ["front", "patio"]);
        assert!(r.gpu.is_none() && r.allow_large);
        assert_eq!(r.tighten, Some(false));
        assert_eq!(r.model, None);
        for (folds, epochs, scene) in [
            (Some(1), None, "day"),
            (Some(9), None, "day"),
            (None, Some(0), "day"),
            (None, Some(51), "day"),
            (None, None, "dusk"),
        ] {
            let scenes = [scene.to_string()];
            assert!(
                build_request(Fields {
                    folds,
                    epochs,
                    scenes: &scenes,
                    ..Fields::default()
                })
                .is_err(),
                "{folds:?} {epochs:?} {scene}"
            );
        }
        // A quick check: the model's name, trimmed; an empty one is the full check; folds or
        // epochs with a model are refused (it trains nothing).
        let quick = |model: &str, folds: Option<usize>, epochs: Option<u32>| {
            build_request(Fields {
                folds,
                epochs,
                model: Some(model.into()),
                ..Fields::default()
            })
        };
        assert_eq!(
            quick(" mine ", None, None).unwrap().model.as_deref(),
            Some("mine")
        );
        assert_eq!(quick("  ", None, None).unwrap().model, None);
        for (f, e) in [(Some(4), None), (None, Some(8))] {
            let m = quick("mine", f, e).unwrap_err();
            assert!(m.contains("quick check with a model trains nothing"), "{m}");
        }
        assert!(quick(&"x".repeat(300), None, None).is_err());
        assert!(quick("a\u{7}b", None, None).is_err());
        let q = PreviewQuery {
            model: Some("mine".into()),
            ..PreviewQuery::default()
        };
        assert_eq!(q.request().unwrap().model.as_deref(), Some("mine"));
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
        assert_eq!(r.tighten, None, "absent: the check's default");
        for (v, want) in [("1", Some(true)), ("0", Some(false)), ("", None)] {
            let q = PreviewQuery {
                tighten: Some(v.into()),
                ..PreviewQuery::default()
            };
            assert_eq!(q.request().unwrap().tighten, want, "{v}");
        }
        let bad = PreviewQuery {
            tighten: Some("maybe".into()),
            ..PreviewQuery::default()
        };
        assert!(
            bad.request()
                .unwrap_err()
                .contains("tighten must be 1 or 0")
        );
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
        let enqueue = |job: RefineJob, rule: &dyn Fn(&[TaskRecord]) -> Option<String>| {
            assert!(
                job.predictor.is_some(),
                "the job runs on the registry's predictor"
            );
            assert_eq!(Task::kind(&job), TaskKind::Refine);
            assert!(rule(&[]).is_none());
            assert!(rule(&[task("t1", RunState::Running)]).is_some());
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

    /// A queue in memory that checks the rule under its own lock, as the training queue's
    /// `enqueue_task_unless` does.
    struct FakeQueue(Mutex<Vec<TaskRecord>>);

    impl FakeQueue {
        fn enqueue(&self, job: RefineJob, rule: QueueRule<'_>) -> Result<String, NotQueued> {
            let mut list = self.0.lock().unwrap();
            if let Some(why) = rule(&list) {
                return Err(NotQueued::Refused(why));
            }
            list.push(TaskRecord::new(&job, String::new()));
            Ok(job.id.clone())
        }
    }

    fn never(_: RefineJob, _: QueueRule<'_>) -> Result<String, NotQueued> {
        panic!("queued")
    }

    /// Two starts at once, both past the first look (the task list each read before its walks
    /// held no check): the queue's own check lets one through, the other is the 409.
    #[test]
    fn two_starts_at_once_queue_one_check_and_the_other_is_refused() {
        let dir = scratch();
        enough_frames(&dir.join("dataset"));
        let config = host_config(&dir);
        let queue = FakeQueue(Mutex::new(Vec::new()));
        let req = RefineRequest {
            folds: 2,
            epochs: 1,
            ..RefineRequest::default()
        };
        let gate = std::sync::Barrier::new(2);
        let answers: Vec<Result<(String, String), Refusal>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let (config, queue, gate, req) = (&config, &queue, &gate, req.clone());
                    s.spawn(move || {
                        gate.wait();
                        start_check(config, req, false, &host(), &[], None, &|job, rule| {
                            queue.enqueue(job, rule)
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(queue.0.lock().unwrap().len(), 1, "exactly one check queued");
        let refused: Vec<&Refusal> = answers.iter().filter_map(|a| a.as_ref().err()).collect();
        assert_eq!(refused.len(), 1, "{answers:?}");
        assert_eq!(refused[0].status, StatusCode::CONFLICT);
        assert!(
            refused[0]
                .message
                .starts_with("Not queued: a dataset check is already queued"),
            "{}",
            refused[0].message
        );
        assert_eq!(refused[0].needs_confirm, None);
    }

    #[test]
    fn a_check_that_cannot_run_is_refused_before_it_is_queued() {
        let dir = scratch();
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
            &|_, _| Err(NotQueued::Failed("the training thread has stopped".into())),
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
        let queued = Mutex::new(0);
        let enqueue = |job: RefineJob, _: QueueRule<'_>| {
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
        assert_eq!(e.needs_confirm, Some((3, 0, 2)));
        assert!(!e.folder_missing);
        assert_eq!(
            e.message,
            format!("Not queued: {}.", check::replace_warning(3, 0, 2))
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

    #[test]
    fn a_check_with_the_closer_look_asks_before_replacing_open_better_boxes() {
        let dir = scratch();
        let review = dir.join("dataset");
        let config = host_config(&dir);
        let queued = Mutex::new(0);
        let enqueue = |job: RefineJob, _: QueueRule<'_>| {
            *queued.lock().unwrap() += 1;
            Ok(job.id.clone())
        };
        enough_frames(&review);
        // An open better box of an earlier look, nothing else open.
        let tight = [5.0, 5.0, 20.0, 18.0];
        frame_with(
            &review,
            "alpha",
            "f00",
            vec![suggestion(Kind::Tighten, Some(CAR), Some(tight))],
        );
        let req = |tighten: bool| RefineRequest {
            folds: 2,
            epochs: 1,
            tighten: Some(tighten),
            ..RefineRequest::default()
        };
        // Without the look it is never touched: no question.
        start_check(&config, req(false), false, &host(), &[], None, &enqueue).unwrap();
        assert_eq!(*queued.lock().unwrap(), 1);
        // With the look it may be replaced: asked, counted as a bound.
        let e = start_check(&config, req(true), false, &host(), &[], None, &never).unwrap_err();
        assert_eq!(e.status, StatusCode::CONFLICT);
        assert_eq!(e.needs_confirm, Some((0, 1, 1)));
        assert_eq!(
            e.message,
            format!("Not queued: {}.", check::replace_warning(0, 1, 1))
        );
        assert!(
            e.message.contains("up to 1 open proposed box"),
            "{}",
            e.message
        );
        start_check(&config, req(true), true, &host(), &[], None, &enqueue).unwrap();
        assert_eq!(*queued.lock().unwrap(), 2);
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
                "open_tightens": 0,
                "open_frames": 2,
                "folder_missing": false,
                "message": format!("Not queued: {}.", check::replace_warning(3, 0, 2)),
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
        assert_eq!(warnings[0], check::replace_warning(3, 0, 2));
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

    // ---- the quick check

    /// A state over the quick check's fixture (a model `mine` trained here, disabled so the test
    /// registry does not load it).
    fn quick_state(f: &check::quick::tests::Fixture) -> Arc<AppState> {
        let fixed = check::quick::tests::config(f);
        test_state_with(move |c| {
            c.download_dir.clone_from(&fixed.download_dir);
            c.collect.dir.clone_from(&fixed.collect.dir);
            c.collect.classes.clone_from(&fixed.collect.classes);
            c.models = fixed
                .models
                .iter()
                .map(|m| crate::config::ModelConfig {
                    enabled: false,
                    ..m.clone()
                })
                .collect();
        })
    }

    #[tokio::test]
    async fn a_quick_check_lists_the_models_trained_here_previews_and_asks_about_its_frames_only() {
        let f = check::quick::tests::fixture("route", false);
        let state = quick_state(&f);
        // GET /v1/refine offers the model, with no frame walk for it.
        let (s, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let models = v["capability"]["models"].as_array().unwrap();
        assert_eq!(models.len(), 1, "{v}");
        assert_eq!(models[0]["name"], "mine");
        assert_eq!(models[0]["run"], check::quick::tests::RUN);
        assert_eq!(models[0]["imgsz"], 384);
        assert_eq!(models[0]["recipe"], "rfdetr-nano");
        assert_eq!(v["capability"]["unusable"], serde_json::json!([]));
        // The preview takes the model; folds or epochs with it are a 400.
        let (s, v) = json(&state, get("/v1/refine/preview?model=mine")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["mode"], "model");
        assert_eq!(v["frames"], 4);
        assert_eq!(v["model"]["trained_on"], 2);
        assert_eq!(v["model"]["validated_on"], 1);
        assert_eq!(v["folds"], serde_json::json!([]));
        assert!(v["refused"].is_null(), "{v}");
        assert!(
            v["text"]
                .as_str()
                .unwrap()
                .contains("with mine: 4 frames it never saw"),
            "{v}"
        );
        for q in ["model=mine&folds=2", "model=mine&epochs=3"] {
            let (s, v) = json(&state, get(&format!("/v1/refine/preview?{q}"))).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{q}: {v}");
        }
        let (s, v) = json(&state, get("/v1/refine/preview?model=nope")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("no configured model")
        );
        // Open suggestions on a frame it trained on are not this check's to replace; those on a
        // frame it never saw are, and the 409 counts them alone, in the model's words.
        let open = |rel: &str| {
            let p = Frame::resolve(&f.dataset, rel).unwrap().labels;
            let mut l = labels::read(&p).unwrap().unwrap();
            l.refine = vec![suggestion(
                Kind::Add,
                None,
                Some([200.0, 60.0, 260.0, 200.0]),
            )];
            labels::write(&p, &l).unwrap();
        };
        open(&f.rels["trained"][0]);
        open(&f.rels["new"][0]);
        let (s, v) = json(
            &state,
            post_json("/v1/refine/check", r#"{"model":"mine","folds":2}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        let (s, v) = json(&state, post_json("/v1/refine/check", r#"{"model":"mine"}"#)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["needs_confirm"], true);
        assert_eq!(
            (v["open_suggestions"].as_u64(), v["open_frames"].as_u64()),
            (Some(1), Some(1)),
            "{v}"
        );
        let m = v["message"].as_str().unwrap();
        assert!(
            m.starts_with(
                "Not queued: this check replaces the 1 open suggestion on 1 frame with what mine \
                 finds"
            ),
            "{m}"
        );
        // Confirmed, and with no GPU at all: planned with the registry's predictor and queued.
        let config = state.config_read().clone();
        let mut none = host();
        none.accelerator = None;
        none.cuda_options.clear();
        let seen = Mutex::new(Vec::new());
        let enqueue = |job: RefineJob, _: QueueRule<'_>| {
            assert!(job.predictor.is_some());
            assert!(job.train.is_none() && job.quick.is_some());
            seen.lock().unwrap().push(Task::title(&job));
            Ok(job.id.clone())
        };
        let req = RefineRequest {
            model: Some("mine".into()),
            ..RefineRequest::default()
        };
        let (_, title) =
            start_check(&config, req.clone(), true, &none, &[], None, &enqueue).unwrap();
        assert_eq!(
            title,
            "quick dataset check with mine (4 frames it never saw, all reviewed frames)"
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
        // A model without a training record is refused before anything is queued.
        std::fs::remove_file(f.root.join("models").join("mine.provenance.json")).unwrap();
        let e = start_check(&config, req, true, &none, &[], None, &never).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(
            e.message.contains("has no training record"),
            "{}",
            e.message
        );
        let (_, v) = json(&state, get("/v1/refine")).await;
        assert_eq!(v["capability"]["models"], serde_json::json!([]), "{v}");
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
