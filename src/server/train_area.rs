//! The Train area: one top-nav entry with eight tabs at their own URLs (Collect, Import,
//! Auto-label, Review, Dataset, Train, Models, Jobs). This file holds what the tabs share: the tab list,
//! `GET /train` (the most useful tab), the redirects of the old `/dataset`, `/review` and
//! `/training` pages, `GET /v1/train/summary` (the counts of the status strip and the one suggested
//! next action) with the pure rule [`next_step`], and the Dataset tab. The other tabs render in
//! [`super::dataset`], [`super::review`] and [`super::training`].
//!
//! Nothing here walks the frames on the request path: the walk ([`queue::tally`]) runs on its own
//! thread and its answer is kept ([`tally_of`]), so a large dataset delays neither the pages (the
//! strip is filled by `assets/train-area.js` after the page is up) nor this route.

use super::{AppState, blocking, encode_uri_component, render};
use crate::collect::CollectSettings;
use crate::collect::labels::queue::{self, Tally};
use crate::collect::layout::SCENES;
use crate::sync::LockExt;
use crate::training::service::{RunRecord, RunState};
use crate::training::task::TaskKind;
use askama::Template;
use axum::Json;
use axum::extract::{RawQuery, State};
use axum::response::{IntoResponse, Redirect, Response};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub(super) const COLLECT: &str = "collect";
pub(super) const IMPORT: &str = "import";
pub(super) const AUTOLABEL: &str = "autolabel";
pub(super) const REVIEW: &str = "review";
pub(super) const DATASET: &str = "dataset";
pub(super) const TRAIN: &str = "train";
pub(super) const MODELS: &str = "models";
pub(super) const JOBS: &str = "jobs";

/// The tabs in nav order: id, title.
pub(super) const TABS: [(&str, &str); 8] = [
    (COLLECT, "Collect"),
    (IMPORT, "Import"),
    (AUTOLABEL, "Auto-label"),
    (REVIEW, "Review"),
    (DATASET, "Dataset"),
    (TRAIN, "Train"),
    (MODELS, "Models"),
    (JOBS, "Jobs"),
];

/// The tab the Jobs card's forms came from (their hidden `tab` field), so their answer lands
/// where they were pressed; `default` for anything else.
pub(super) fn jobs_tab(field: Option<&str>, default: &'static str) -> &'static str {
    [IMPORT, AUTOLABEL, TRAIN, MODELS]
        .into_iter()
        .find(|t| field == Some(*t))
        .unwrap_or(default)
}

/// The title of tab `id`.
pub(super) fn tab_title(id: &str) -> &'static str {
    TABS.iter().find(|(t, _)| *t == id).map_or("Train", |t| t.1)
}

// ---------------------------------------------------------------------------------------------
// The next step

/// Reviewed frames that make a first export (and later ones) worth suggesting.
const ENOUGH_TO_EXPORT: u64 = 50;
/// Frames no auto-label run has seen before Auto-label is suggested: a few new frames from the
/// collector should not send the owner back there.
const ENOUGH_TO_AUTOLABEL: u64 = 20;

/// A job on the queue, as the strip words it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Job {
    /// "Training", "An import", ...
    pub(crate) what: &'static str,
    /// The tab with its progress.
    pub(crate) tab: &'static str,
    pub(crate) paused: bool,
}

/// What the strip shows and [`next_step`] decides from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct Summary {
    /// `collect.enabled`.
    pub(crate) collecting: bool,
    /// Frames in the dataset folder (the collector's own count).
    pub(crate) frames_collected: u64,
    /// The review counts below are known; false while the first walk of a folder runs.
    pub(crate) counted: bool,
    pub(crate) frames_to_review: u64,
    /// Of those, in the review queue (the Review tab's default list).
    pub(crate) frames_queued: u64,
    pub(crate) boxes_to_review: u64,
    pub(crate) never_auto_labelled: u64,
    pub(crate) frames_reviewed: u64,
    /// Review export folders offered to training.
    pub(crate) exports: u64,
    /// Frames of the newest export.
    pub(crate) exported_frames: u64,
    /// Every training run on the list.
    pub(crate) runs: u64,
    /// Runs that made a model.
    pub(crate) models: u64,
    /// The newest such model, when it is neither the default nor a night model or verifier.
    pub(crate) unused_model: Option<String>,
    /// The newest run ended in failure.
    pub(crate) last_run_failed: bool,
    /// The running job, else the first queued one.
    pub(crate) job: Option<Job>,
    /// This machine can train.
    pub(crate) can_train: bool,
    /// This machine can run the open-vocabulary labelers.
    pub(crate) can_autolabel: bool,
    /// Why the folder under review cannot be used (an unplugged drive), as the page words it.
    pub(crate) folder_problem: Option<String>,
}

/// The one suggested action: a plain sentence, the link that does it. Never an order: every tab
/// stays one click away.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct NextStep {
    /// Which rule answered (tests and the page's styling).
    pub(crate) key: &'static str,
    pub(crate) text: String,
    /// Where the action is: a tab, with a query or fragment when it helps.
    pub(crate) link: String,
    pub(crate) label: String,
    /// The tab `link` opens.
    pub(crate) tab: &'static str,
}

fn step(key: &'static str, tab: &'static str, text: String, link: String, label: &str) -> NextStep {
    NextStep {
        key,
        text,
        link,
        label: label.to_string(),
        tab,
    }
}

/// `n frames` / `1 frame`.
fn frames(n: u64) -> String {
    format!("{n} frame{}", if n == 1 { "" } else { "s" })
}

/// The next step of `s`, the first rule that applies: a job in progress, a finished model nobody
/// decided about, nothing to work on yet, enough new reviewed frames to export, frames to label
/// and review, an export not trained on, a failed run, and otherwise nothing to do.
pub(crate) fn next_step(s: &Summary) -> NextStep {
    if let Some(j) = &s.job {
        let verb = if j.paused { "paused" } else { "running" };
        return step(
            "job",
            j.tab,
            format!(
                "{} is {verb}. You can keep working on the other tabs.",
                j.what
            ),
            format!("/train/{}#jobs", j.tab),
            "See its progress",
        );
    }
    if let Some(why) = &s.folder_problem {
        return step(
            "folder_missing",
            REVIEW,
            why.clone(),
            "/train/review".to_string(),
            "Open Review",
        );
    }
    if let Some(m) = &s.unused_model {
        return step(
            "use_model",
            MODELS,
            format!("{m} finished training. Check its results and decide whether to use it."),
            "/train/models#use".to_string(),
            "See the results",
        );
    }
    let none_yet = s.frames_collected == 0;
    if !s.counted {
        if none_yet && !s.collecting {
            return collect_or_import(s);
        }
        return step(
            "counting",
            REVIEW,
            "Counting your frames...".to_string(),
            "/train/review".to_string(),
            "Open Review",
        );
    }
    if none_yet && s.frames_to_review + s.frames_reviewed == 0 {
        return collect_or_import(s);
    }
    let unexported = s.frames_reviewed.saturating_sub(s.exported_frames);
    if s.frames_reviewed >= ENOUGH_TO_EXPORT && unexported >= ENOUGH_TO_EXPORT {
        return export_step(unexported);
    }
    if s.frames_to_review > 0 {
        if s.can_autolabel && s.never_auto_labelled >= ENOUGH_TO_AUTOLABEL {
            return step(
                "autolabel",
                AUTOLABEL,
                format!(
                    "{} have no proposed boxes yet. Auto-label them, then you only confirm or \
                     correct.",
                    frames(s.never_auto_labelled)
                ),
                "/train/autolabel".to_string(),
                "Auto-label",
            );
        }
        let boxes = if s.boxes_to_review > 0 {
            format!(" ({} boxes)", s.boxes_to_review)
        } else {
            String::new()
        };
        let link = if s.frames_queued > 0 {
            "/train/review".to_string()
        } else {
            "/train/review?source=all".to_string()
        };
        return step(
            "review",
            REVIEW,
            format!("{}{boxes} to review.", frames(s.frames_to_review)),
            link,
            "Review them",
        );
    }
    if unexported > 0 {
        return export_step(unexported);
    }
    if s.last_run_failed {
        return step(
            "run_failed",
            TRAIN,
            "The last training run failed. Its log says why.".to_string(),
            "/train/train#jobs".to_string(),
            "See the run",
        );
    }
    if s.exports > 0 && s.runs == 0 {
        if s.can_train {
            return step(
                "train",
                TRAIN,
                "Your export is ready. Train a model on it.".to_string(),
                "/train/train".to_string(),
                "Set up training",
            );
        }
        return step(
            "cannot_train",
            TRAIN,
            "This machine cannot train: it needs an NVIDIA GPU or Apple silicon. Your exports can \
             still be used on another machine."
                .to_string(),
            "/train/train".to_string(),
            "Why not",
        );
    }
    if s.collecting {
        return step(
            "done",
            COLLECT,
            "All caught up. New frames will show here as they are collected.".to_string(),
            "/train/collect".to_string(),
            "See what was collected",
        );
    }
    step(
        "done",
        COLLECT,
        "All caught up. Turn on collecting to gather more frames from your cameras.".to_string(),
        "/train/collect".to_string(),
        "Turn on collecting",
    )
}

fn collect_or_import(s: &Summary) -> NextStep {
    if s.collecting {
        step(
            "wait_frames",
            IMPORT,
            "Collecting is on and no frames have arrived yet. You can import footage you already \
             have while you wait."
                .to_string(),
            "/train/import".to_string(),
            "Import footage",
        )
    } else {
        step(
            "start_collecting",
            COLLECT,
            "No frames yet. Turn on collecting, or import footage you already have.".to_string(),
            "/train/collect".to_string(),
            "Turn on collecting",
        )
    }
}

fn export_step(unexported: u64) -> NextStep {
    step(
        "export",
        DATASET,
        format!(
            "{} reviewed since the last export. Export them so training can use them.",
            frames(unexported)
        ),
        "/train/dataset#export".to_string(),
        "Export",
    )
}

// ---------------------------------------------------------------------------------------------
// The counts

/// One folder's walk, kept.
#[derive(Default)]
struct Kept {
    /// When the kept answer was made, how long it took; `None` before the first one.
    done: Option<(Instant, Duration, Arc<Tally>)>,
    /// A walk is running.
    walking: bool,
}

/// The walks, per folder: the review folder, and few others over a process's life.
static KEPT: Mutex<BTreeMap<PathBuf, Kept>> = Mutex::new(BTreeMap::new());
static WALKED: Condvar = Condvar::new();

/// How long a walk's answer is kept: at least 10 s, and 20 times what it took, so a walk of
/// seconds over a large dataset takes under 5% of a core however often the strip asks.
fn keep_for(took: Duration) -> Duration {
    (took * 20).max(Duration::from_secs(10))
}

/// Marks a walk over when dropped, panic included, so one failure does not stop the next.
struct WalkDone<'a>(&'a Path, Option<(Tally, Duration)>);

impl Drop for WalkDone<'_> {
    fn drop(&mut self) {
        if let Some(k) = KEPT.lock_recover().get_mut(self.0) {
            k.walking = false;
            if let Some((t, took)) = self.1.take() {
                k.done = Some((Instant::now(), took, Arc::new(t)));
            }
        }
        WALKED.notify_all();
    }
}

/// The [`Tally`] of `dir`: what is kept, however old, while a new walk runs on its own thread. With
/// nothing kept, waits up to `wait` for the walk (None past that: the page says it is counting and
/// asks again).
pub(super) fn tally_of(dir: &Path, wait: Duration) -> Option<Arc<Tally>> {
    let mut kept = KEPT.lock_recover();
    let k = kept.entry(dir.to_path_buf()).or_default();
    let fresh = k
        .done
        .as_ref()
        .is_some_and(|(at, took, _)| at.elapsed() < keep_for(*took));
    if !fresh && !k.walking {
        k.walking = true;
        let dir = dir.to_path_buf();
        std::thread::spawn(move || {
            let mut done = WalkDone(&dir, None);
            let start = Instant::now();
            let t = queue::tally(&dir);
            done.1 = Some((t, start.elapsed()));
        });
    }
    let until = Instant::now() + wait;
    loop {
        let k = kept.get(dir)?;
        if let Some((_, _, t)) = &k.done {
            return Some(t.clone());
        }
        let left = until.saturating_duration_since(Instant::now());
        if !k.walking || left.is_zero() {
            return None;
        }
        kept = WALKED
            .wait_timeout(kept, left)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

/// One review export folder, for the Dataset tab, the Train tab's list and the summary.
pub(super) struct ExportRow {
    pub(super) path: String,
    /// "2026-11-06 06:47 UTC", or the folder's name.
    pub(super) when: String,
    /// `frames` of its `export.json`; None without one.
    pub(super) frames: Option<u64>,
    /// "17 train, 9 held out".
    pub(super) split: String,
    /// "person 27, bicycle 1".
    pub(super) classes: String,
    pub(super) formats: String,
    /// The Train tab with this folder ticked.
    pub(super) train_link: String,
}

/// `20261106T064704883Z` as a date, the folder's name for any other.
fn export_when(name: &str) -> String {
    let digits = |r: std::ops::Range<usize>| {
        name.get(r)
            .filter(|p| p.bytes().all(|b| b.is_ascii_digit()))
    };
    match (
        digits(0..4),
        digits(4..6),
        digits(6..8),
        name.get(8..9),
        digits(9..11),
        digits(11..13),
    ) {
        (Some(y), Some(mo), Some(d), Some("T"), Some(h), Some(mi)) => {
            format!("{y}-{mo}-{d} {h}:{mi} UTC")
        }
        _ => name.to_string(),
    }
}

/// The row of export folder `dir`, its report read when it has one.
pub(super) fn export_row(dir: &Path) -> ExportRow {
    let report: Option<serde_json::Value> = std::fs::read(dir.join("export.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let num = |k: &str| report.as_ref().and_then(|r| r.get(k)?.as_u64());
    let mut per: Vec<(String, u64)> = report
        .as_ref()
        .and_then(|r| r.get("per_class")?.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    per.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ExportRow {
        train_link: format!(
            "/train/train?dataset={}",
            encode_uri_component(&dir.display().to_string())
        ),
        path: dir.display().to_string(),
        when: export_when(
            &dir.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        ),
        frames: num("frames"),
        split: match (num("train"), num("val")) {
            (Some(t), Some(v)) => format!("{t} train, {v} held out"),
            _ => String::new(),
        },
        classes: per
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", "),
        formats: report
            .as_ref()
            .and_then(|r| r.get("formats")?.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|f| f.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default(),
    }
}

/// The tab with the jobs of `kind`.
pub(super) const fn task_tab(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Import => IMPORT,
        TaskKind::Autolabel | TaskKind::Overlaps => AUTOLABEL,
    }
}

/// The job the strip names: a running one, else the first queued; training runs before the rest.
fn current_job(runs: &[RunRecord], tasks: &[crate::training::task::TaskRecord]) -> Option<Job> {
    let run = |s: RunState| {
        runs.iter().any(|r| r.state == s).then_some(Job {
            what: "Training",
            tab: TRAIN,
            paused: s == RunState::Paused,
        })
    };
    let task = |s: RunState| {
        tasks.iter().find(|t| t.state == s).map(|t| Job {
            what: match t.kind {
                TaskKind::Import => "An import",
                TaskKind::Autolabel => "An auto-label run",
                TaskKind::Overlaps => "An overlap check",
            },
            tab: task_tab(t.kind),
            paused: s == RunState::Paused,
        })
    };
    [RunState::Running, RunState::Paused, RunState::Queued]
        .into_iter()
        .find_map(|s| run(s).or_else(|| task(s)))
}

/// The newest model when nothing uses it: not the default, no night model or verifier.
fn unused_model(cfg: &crate::config::Config, runs: &[RunRecord]) -> Option<String> {
    let newest = runs
        .iter()
        .rev()
        .filter(|r| r.state == RunState::Done)
        .find_map(|r| r.output.as_ref())?;
    let name = newest.name.as_str();
    let same = |other: &str| other.eq_ignore_ascii_case(name);
    let current = crate::training::switch::current_default(cfg).unwrap_or_default();
    let used = same(&current)
        || cfg.models.iter().any(|m| {
            m.night_model.as_deref().is_some_and(same)
                || m.verify.as_ref().is_some_and(|v| same(&v.model))
        });
    (!used).then(|| name.to_string())
}

/// The strip's counts now. `wait`: see [`tally_of`].
pub(super) fn summary_of(state: &AppState, wait: Duration) -> (Summary, Option<Arc<Tally>>) {
    let config = state.config_read().clone();
    let review_dir = config.collect.resolved_review_dir(&config.data_root());
    let dataset_dir = CollectSettings::for_config(&config).dir;
    let tally = tally_of(&review_dir, wait);
    let brief = crate::collect::global().brief_of(&dataset_dir);
    let runs = super::training::runs(state);
    let tasks = super::training::tasks(state);
    let exports = crate::training::discover_exports(&config);
    let cap = crate::training::capability_now();
    let saved = config
        .training
        .datasets
        .iter()
        .filter(|d| !exports.contains(d))
        .count();
    let t = tally.as_deref();
    (
        Summary {
            folder_problem: super::review::folder_problem(state, &review_dir),
            collecting: config.collect.enabled,
            // A folder under review holds frames of its own, and imports count once indexed.
            frames_collected: brief.summary.frames.max(t.map_or(0, |t| t.frames)),
            counted: t.is_some(),
            frames_to_review: t.map_or(0, |t| t.to_review),
            frames_queued: t.map_or(0, |t| t.queued_to_review),
            boxes_to_review: t.map_or(0, |t| t.pending_boxes),
            never_auto_labelled: t.map_or(0, |t| t.never_auto_labelled),
            frames_reviewed: t.map_or(0, |t| t.reviewed),
            exported_frames: exports
                .first()
                .and_then(|d| export_row(d).frames)
                .unwrap_or(0),
            exports: (exports.len() + saved) as u64,
            runs: runs.len() as u64,
            models: runs
                .iter()
                .filter(|r| r.state == RunState::Done && r.output.is_some())
                .count() as u64,
            unused_model: unused_model(&config, &runs),
            last_run_failed: runs.last().is_some_and(|r| r.state == RunState::Failed),
            job: current_job(&runs, &tasks),
            can_train: cap.available,
            can_autolabel: cap.accelerator.is_some(),
        },
        tally,
    )
}

/// `GET /v1/train/summary`: the strip's counts and [`next_step`]. `counted: false` while the first
/// walk of a large folder runs; ask again.
pub(super) async fn summary(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (s, _) = summary_of(&state, Duration::from_millis(1500));
        let next = next_step(&s);
        Json(serde_json::json!({ "success": true, "summary": s, "next": next })).into_response()
    })
    .await
}

/// `GET /train`: the tab [`next_step`] points at. It never waits long: a folder still being
/// counted opens Review, which counts for itself.
pub(super) async fn entry(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let (s, _) = summary_of(&state, Duration::from_millis(400));
        let link = next_step(&s).link;
        // The server's redirect drops the fragment on some clients; the tab is what matters.
        let at = link.split('#').next().unwrap_or(&link).to_string();
        Redirect::temporary(&at).into_response()
    })
    .await
}

/// `target`, keeping the query string of the old URL.
fn moved(target: &str, query: Option<&str>) -> Response {
    match query.filter(|q| !q.is_empty()) {
        Some(q) => Redirect::temporary(&format!("{target}?{q}")).into_response(),
        None => Redirect::temporary(target).into_response(),
    }
}

/// `GET /dataset`, kept for bookmarks.
pub(super) async fn old_dataset(RawQuery(q): RawQuery) -> Response {
    moved("/train/collect", q.as_deref())
}

/// `GET /review`, kept for bookmarks.
pub(super) async fn old_review(RawQuery(q): RawQuery) -> Response {
    moved("/train/review", q.as_deref())
}

/// `GET /training`, kept for bookmarks. A fragment stays with the browser; `train-area.js` sends
/// `/train/train#import` and the like on to the card's new tab.
pub(super) async fn old_training(RawQuery(q): RawQuery) -> Response {
    moved("/train/train", q.as_deref())
}

// ---------------------------------------------------------------------------------------------
// The Dataset tab

/// Most camera columns of the coverage table; the rest add up in "Other cameras".
const COVERAGE_CAMERAS: usize = 8;

/// A class in the coverage table.
struct CoverageRow {
    class: String,
    total: u64,
    /// In the order of [`SCENES`].
    scenes: Vec<u64>,
    /// In the order of `coverage_cameras`.
    cameras: Vec<u64>,
    other: u64,
}

/// The coverage table of `t`: classes with the most boxes first, the cameras with the most boxes
/// as columns.
fn coverage(t: &Tally) -> (Vec<CoverageRow>, Vec<String>, bool) {
    let mut per_camera: BTreeMap<&str, u64> = BTreeMap::new();
    for row in t.coverage.values() {
        for (c, n) in &row.by_camera {
            *per_camera.entry(c).or_default() += n;
        }
    }
    let mut cameras: Vec<(&str, u64)> = per_camera.into_iter().collect();
    cameras.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let shown: Vec<String> = cameras
        .iter()
        .take(COVERAGE_CAMERAS)
        .map(|c| c.0.to_string())
        .collect();
    let mut rows: Vec<CoverageRow> = t
        .coverage
        .iter()
        .map(|(class, row)| {
            let cams: Vec<u64> = shown
                .iter()
                .map(|c| row.by_camera.get(c).copied().unwrap_or(0))
                .collect();
            let sum: u64 = cams.iter().sum();
            CoverageRow {
                class: class.clone(),
                total: row.boxes,
                scenes: SCENES
                    .iter()
                    .map(|s| row.by_scene.get(*s).copied().unwrap_or(0))
                    .collect(),
                other: row.boxes - sum,
                cameras: cams,
            }
        })
        .collect();
    rows.sort_by(|a, b| b.total.cmp(&a.total).then_with(|| a.class.cmp(&b.class)));
    let other = cameras.len() > COVERAGE_CAMERAS;
    (rows, shown, other)
}

#[derive(Template)]
#[template(path = "train/dataset.html")]
struct DatasetTabTemplate {
    nav: &'static str,
    tab: &'static str,
    tab_title: &'static str,
    version: &'static str,
    classes_json: String,
    classes_text: String,
    exports_dir: String,
    exports: Vec<ExportRow>,
    /// The folder the coverage counts describe.
    dir: String,
    /// Still counted: the coverage table says so instead of waiting.
    counting: bool,
    reviewed: u64,
    coverage: Vec<CoverageRow>,
    coverage_cameras: Vec<String>,
    coverage_other: bool,
    coverage_scenes: [&'static str; 3],
}

/// `GET /train/dataset`.
pub(super) async fn dataset_page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let config = state.config_read().clone();
        let dir = config.collect.resolved_review_dir(&config.data_root());
        let tally = tally_of(&dir, Duration::from_millis(1000));
        let exports = crate::training::discover_exports(&config)
            .iter()
            .map(|d| export_row(d))
            .collect();
        let classes = config.collect.classes;
        let (rows, cameras, other) = tally
            .as_deref()
            .map_or_else(|| (Vec::new(), Vec::new(), false), coverage);
        render(&DatasetTabTemplate {
            nav: "train",
            tab: DATASET,
            tab_title: tab_title(DATASET),
            version: crate::VERSION,
            classes_json: serde_json::to_string(&classes).unwrap_or_else(|_| "[]".into()),
            classes_text: classes.join(", "),
            exports_dir: crate::training::training_dir(&state.config_read())
                .join("exports")
                .display()
                .to_string(),
            exports,
            dir: dir.display().to_string(),
            counting: tally.is_none(),
            reviewed: tally.as_ref().map_or(0, |t| t.reviewed),
            coverage: rows,
            coverage_cameras: cameras,
            coverage_other: other,
            coverage_scenes: SCENES,
        })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{call, get_text, post_form, test_state_with};
    use super::super::training::{QUEUE_TEST_LOCK, own_queue};
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};

    fn counted() -> Summary {
        Summary {
            counted: true,
            can_train: true,
            can_autolabel: true,
            ..Summary::default()
        }
    }

    fn key(s: &Summary) -> &'static str {
        next_step(s).key
    }

    #[test]
    fn with_nothing_it_points_at_collecting_or_importing() {
        let n = next_step(&counted());
        assert_eq!(
            (n.key, n.tab, n.link.as_str()),
            ("start_collecting", COLLECT, "/train/collect")
        );
        let s = Summary {
            collecting: true,
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.tab), ("wait_frames", IMPORT));
    }

    /// An unplugged drive is not an empty dataset: never advice to collect or import.
    #[test]
    fn a_missing_review_folder_is_said_before_any_advice_on_frames() {
        let why = "The folder under review is not available: /v/x; connect the drive".to_string();
        let s = Summary {
            folder_problem: Some(why.clone()),
            ..Summary::default()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.tab), ("folder_missing", REVIEW));
        assert_eq!(n.text, why);
        // A running job still comes first: it is what the user can act on.
        let busy = Summary {
            job: Some(Job {
                tab: IMPORT,
                what: "An import",
                paused: false,
            }),
            ..s
        };
        assert_eq!(next_step(&busy).key, "job");
    }

    #[test]
    fn frames_nobody_labelled_go_to_auto_label_when_this_machine_can() {
        let s = Summary {
            frames_collected: 188,
            frames_to_review: 188,
            never_auto_labelled: 188,
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.tab), ("autolabel", AUTOLABEL));
        assert!(n.text.starts_with("188 frames have"), "{}", n.text);
        // Without a GPU, or for a few new frames, Review is the answer.
        let no_gpu = Summary {
            can_autolabel: false,
            ..s.clone()
        };
        assert_eq!(key(&no_gpu), "review");
        let few = Summary {
            never_auto_labelled: ENOUGH_TO_AUTOLABEL - 1,
            ..s
        };
        assert_eq!(key(&few), "review");
    }

    #[test]
    fn frames_to_review_say_how_many_and_open_the_list_that_holds_them() {
        let s = Summary {
            frames_collected: 40,
            frames_to_review: 12,
            frames_queued: 12,
            boxes_to_review: 30,
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.link.as_str()), ("review", "/train/review"));
        assert_eq!(n.text, "12 frames (30 boxes) to review.");
        // Collected but never sent to review: the default list would be empty.
        let n = next_step(&Summary {
            frames_queued: 0,
            ..s
        });
        assert_eq!(n.link, "/train/review?source=all");
    }

    #[test]
    fn enough_new_reviewed_frames_come_before_more_review() {
        let s = Summary {
            frames_collected: 300,
            frames_to_review: 100,
            frames_reviewed: 60,
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.tab), ("export", DATASET));
        assert_eq!(n.link, "/train/dataset#export");
        // Under the bar, reviewing goes on.
        let s = Summary {
            frames_reviewed: ENOUGH_TO_EXPORT - 1,
            ..s
        };
        assert_eq!(key(&s), "review");
        // Already exported: only what is new counts.
        let s = Summary {
            frames_reviewed: 60,
            exported_frames: 55,
            exports: 1,
            ..s
        };
        assert_eq!(key(&s), "review");
    }

    #[test]
    fn nothing_left_to_review_exports_the_few_reviewed_frames() {
        let s = Summary {
            frames_collected: 8,
            frames_reviewed: 8,
            ..counted()
        };
        assert_eq!(key(&s), "export");
        assert!(next_step(&s).text.contains("8 frames reviewed"));
        let one = Summary {
            frames_reviewed: 1,
            ..s
        };
        assert!(next_step(&one).text.starts_with("1 frame reviewed"));
    }

    #[test]
    fn an_export_nothing_trained_on_goes_to_train_or_says_why_not() {
        let s = Summary {
            frames_collected: 80,
            frames_reviewed: 80,
            exported_frames: 80,
            exports: 1,
            ..counted()
        };
        assert_eq!(next_step(&s).tab, TRAIN);
        assert_eq!(key(&s), "train");
        let no = Summary {
            can_train: false,
            ..s
        };
        assert_eq!(key(&no), "cannot_train");
    }

    #[test]
    fn a_finished_model_comes_before_more_work() {
        let s = Summary {
            frames_collected: 300,
            frames_to_review: 100,
            unused_model: Some("custom-yolo26s".into()),
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.tab), ("use_model", MODELS));
        assert!(n.text.contains("custom-yolo26s"));
        assert!(n.link.ends_with("#use"));
    }

    #[test]
    fn a_job_comes_first_and_names_its_tab() {
        let s = Summary {
            frames_to_review: 5,
            unused_model: Some("m".into()),
            job: Some(Job {
                what: "An import",
                tab: IMPORT,
                paused: true,
            }),
            ..counted()
        };
        let n = next_step(&s);
        assert_eq!((n.key, n.link.as_str()), ("job", "/train/import#jobs"));
        assert!(n.text.contains("An import is paused"), "{}", n.text);
    }

    #[test]
    fn a_failed_run_is_named_once_the_frames_are_done() {
        let s = Summary {
            frames_collected: 80,
            frames_reviewed: 80,
            exported_frames: 80,
            exports: 1,
            runs: 1,
            last_run_failed: true,
            ..counted()
        };
        assert_eq!(key(&s), "run_failed");
    }

    #[test]
    fn with_everything_done_it_says_so() {
        let s = Summary {
            frames_collected: 80,
            frames_reviewed: 80,
            exported_frames: 80,
            exports: 1,
            runs: 1,
            models: 1,
            ..counted()
        };
        assert_eq!(key(&s), "done");
        assert!(next_step(&s).text.contains("Turn on collecting"));
        let on = Summary {
            collecting: true,
            ..s
        };
        assert_eq!(key(&on), "done");
        assert!(!next_step(&on).text.contains("Turn on"));
    }

    #[test]
    fn while_counting_it_does_not_guess() {
        let s = Summary {
            counted: false,
            frames_collected: 5000,
            ..counted()
        };
        assert_eq!(key(&s), "counting");
        // Nothing collected and collection off needs no count.
        let s = Summary {
            counted: false,
            ..counted()
        };
        assert_eq!(key(&s), "start_collecting");
    }

    #[test]
    fn every_link_is_a_tab_of_the_area() {
        for s in [
            counted(),
            Summary {
                frames_to_review: 3,
                ..counted()
            },
            Summary {
                frames_reviewed: 3,
                ..counted()
            },
        ] {
            let n = next_step(&s);
            assert!(n.link.starts_with(&format!("/train/{}", n.tab)), "{n:?}");
            assert!(TABS.iter().any(|(t, _)| *t == n.tab));
        }
    }

    #[test]
    fn the_jobs_forms_land_on_their_tab() {
        assert_eq!(jobs_tab(Some("import"), TRAIN), IMPORT);
        assert_eq!(jobs_tab(Some("models"), TRAIN), MODELS);
        assert_eq!(jobs_tab(None, MODELS), MODELS);
        assert_eq!(jobs_tab(Some("../x"), TRAIN), TRAIN);
        assert_eq!(jobs_tab(Some("review"), TRAIN), TRAIN);
    }

    #[test]
    fn export_rows_read_the_report_and_survive_without_one() {
        let dir = scratch();
        let e = dir.join("20261106T064704883Z");
        std::fs::create_dir_all(&e).unwrap();
        std::fs::write(
            e.join("export.json"),
            r#"{"frames":26,"train":17,"val":9,"formats":["yolo","benchmark"],
                "per_class":{"bicycle":1,"person":27}}"#,
        )
        .unwrap();
        let r = export_row(&e);
        assert_eq!(r.when, "2026-11-06 06:47 UTC");
        assert_eq!(r.frames, Some(26));
        assert_eq!(r.split, "17 train, 9 held out");
        assert_eq!(r.classes, "person 27, bicycle 1");
        assert_eq!(r.formats, "yolo, benchmark");
        let plain = dir.join("my-export");
        std::fs::create_dir_all(&plain).unwrap();
        let r = export_row(&plain);
        assert_eq!((r.when.as_str(), r.frames), ("my-export", None));
        assert!(r.split.is_empty() && r.classes.is_empty());
    }

    #[test]
    fn coverage_lists_classes_by_boxes_and_folds_the_cameras_past_the_limit() {
        let mut t = Tally::default();
        let mut person = queue::ClassCoverage {
            boxes: 12,
            ..Default::default()
        };
        person.by_scene.insert("day".into(), 9);
        person.by_scene.insert("ir".into(), 3);
        for i in 0..10 {
            person
                .by_camera
                .insert(format!("cam{i}"), if i < 2 { 2 } else { 1 });
        }
        t.coverage.insert("person".into(), person);
        t.coverage.insert(
            "car".into(),
            queue::ClassCoverage {
                boxes: 1,
                by_scene: [("night".to_string(), 1)].into(),
                by_camera: [("cam0".to_string(), 1)].into(),
            },
        );
        let (rows, cameras, other) = coverage(&t);
        assert_eq!(rows[0].class, "person");
        assert_eq!(rows[0].scenes, vec![9, 0, 3], "{SCENES:?}");
        assert_eq!(cameras.len(), COVERAGE_CAMERAS);
        assert!(other);
        assert_eq!(rows[0].other, 2);
        assert_eq!(rows[1].scenes, vec![0, 1, 0]);
    }

    /// The tabs' URLs as a server sees them.
    const PAGES: [(&str, &str); 8] = [
        ("/train/collect", "collect"),
        ("/train/import", "import"),
        ("/train/autolabel", "autolabel"),
        ("/train/review", "review"),
        ("/train/dataset", "dataset"),
        ("/train/train", "train"),
        ("/train/models", "models"),
        ("/train/jobs", "jobs"),
    ];

    /// A folder of its own for a test (under the system's temp folder).
    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_area_{}", uuid::Uuid::new_v4()));
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

    /// Finishes the first walk of `state`'s review folder: the routes wait 400 ms or 1.5 s for it,
    /// a busy machine can take longer, and what is kept is answered at once however old.
    fn walk_finished(state: &AppState) {
        let config = state.config_read().clone();
        let dir = config.collect.resolved_review_dir(&config.data_root());
        assert!(
            tally_of(&dir, Duration::from_secs(60)).is_some(),
            "no walk of {} in 60 s",
            dir.display()
        );
    }

    async fn status_and_location(state: &Arc<AppState>, url: &str) -> (StatusCode, String) {
        let (s, h, _) = call(state, Request::get(url).body(Body::empty()).unwrap()).await;
        let at = h
            .get(header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        (s, at)
    }

    #[tokio::test]
    async fn the_old_pages_redirect_and_keep_their_query() {
        let dir = scratch();
        let state = state_in(&dir);
        for (old, new) in [
            ("/dataset", "/train/collect"),
            ("/review", "/train/review"),
            ("/training", "/train/train"),
        ] {
            let (s, at) = status_and_location(&state, old).await;
            assert_eq!(
                (s, at.as_str()),
                (StatusCode::TEMPORARY_REDIRECT, new),
                "{old}"
            );
            let (s, at) =
                status_and_location(&state, &format!("{old}?auto=overlaps&camera=a%20b")).await;
            assert_eq!(s, StatusCode::TEMPORARY_REDIRECT);
            assert_eq!(at, format!("{new}?auto=overlaps&camera=a%20b"));
        }
    }

    #[tokio::test]
    async fn train_goes_to_the_tab_of_the_next_step() {
        let dir = scratch();
        let state = state_in(&dir);
        let _queue = own_queue(&state).await;
        walk_finished(&state);
        let (s, at) = status_and_location(&state, "/train").await;
        assert_eq!(s, StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(at, "/train/collect");
    }

    #[tokio::test]
    async fn the_summary_route_answers_counts_and_a_next_step() {
        let dir = scratch();
        let state = state_in(&dir);
        let _queue = own_queue(&state).await;
        walk_finished(&state);
        let (s, body) = get_text(&state, "/v1/train/summary").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["summary"]["counted"], true, "{v}");
        assert_eq!(v["summary"]["frames_to_review"], 0);
        assert_eq!(v["next"]["key"], "start_collecting");
        assert_eq!(v["next"]["link"], "/train/collect");
    }

    #[tokio::test]
    async fn the_jobs_route_answers_with_no_queue_yet() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, body) = get_text(&state, "/v1/train/jobs").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["success"], true);
        assert!(v["jobs"]["active"].is_array(), "{v}");
        assert!(v["jobs"]["queued"].is_array(), "{v}");
    }

    #[tokio::test]
    async fn the_jobs_tab_says_when_nothing_runs_and_every_tab_links_to_it() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, html) = get_text(&state, "/train/jobs").await;
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains(r#"id="train-jobs-empty""#), "{html}");
        assert!(html.contains(r#"id="train-jobs""#), "{html}");
        for (url, _) in PAGES {
            let (_, page) = get_text(&state, url).await;
            assert!(page.contains(r#"<a href="/train/jobs""#), "{url}");
            assert!(page.contains(r#"id="train-jobs-count""#), "{url}");
        }
    }

    #[tokio::test]
    async fn every_tab_renders_with_its_assets_and_the_nav_marks_it() {
        let dir = scratch();
        let state = state_in(&dir);
        for (url, tab) in PAGES {
            let (s, html) = get_text(&state, url).await;
            assert_eq!(s, StatusCode::OK, "{url}");
            assert!(html.contains("href=\"/train/collect\""), "{url}: nav");
            assert!(
                html.contains(&format!(
                    "<a href=\"{url}\" class=\"active\" aria-current=\"page\">"
                )),
                "{url}: sub-tab"
            );
            assert!(html.contains(&format!("id=\"train-strip\" data-tab=\"{tab}\"")));
            assert!(
                html.contains("href=\"/train\" class=\"active\" aria-current=\"page\">Train<"),
                "{url}: top nav"
            );
            let mut rest = html.as_str();
            let mut seen = 0;
            while let Some(i) = rest.find("/static/") {
                rest = &rest[i..];
                let end = rest.find('"').unwrap();
                let (path, after) = rest.split_at(end);
                rest = after;
                assert!(path.contains("?v="), "{url}: {path} has no version");
                let (s, _) = get_text(&state, path).await;
                assert_eq!(s, StatusCode::OK, "{url}: {path}");
                seen += 1;
            }
            assert!(seen >= 3, "{url}: {seen} assets");
        }
    }

    /// What a tab must and must not hold: the split of the old pages.
    #[tokio::test]
    async fn each_tab_holds_its_own_content() {
        let dir = scratch();
        let state = state_in(&dir);
        let cases: [(&str, &[&str], &[&str]); 7] = [
            (
                "/train/collect",
                &["action=\"/dataset/settings\"", "name=\"enabled\""],
                &["id=\"rv-export\"", "id=\"import-form\""],
            ),
            (
                "/train/import",
                &["id=\"import-form\"", "id=\"jobs\""],
                &["id=\"autolabel-form\"", "id=\"train\""],
            ),
            (
                "/train/autolabel",
                &["id=\"autolabel-form\"", "id=\"jobs\"", "id=\"al-advanced\""],
                &["id=\"import-form\"", "id=\"train\""],
            ),
            (
                "/train/review",
                &[
                    "id=\"rv-canvas\"",
                    "id=\"rv-filters\"",
                    "id=\"rv-bulk\"",
                    "id=\"rv-autolabel-link\"",
                ],
                &[
                    "id=\"rv-autolabel\"",
                    "id=\"rv-export\"",
                    "id=\"rv-classes\"",
                    "id=\"jobs\"",
                ],
            ),
            (
                "/train/dataset",
                &[
                    "id=\"rv-export\"",
                    "id=\"rv-import\"",
                    "id=\"rv-classes\"",
                    "id=\"coverage\"",
                ],
                &["id=\"rv-canvas\"", "id=\"jobs\""],
            ),
            (
                "/train/train",
                &["id=\"train\"", "id=\"jobs\"", "id=\"train-advanced\""],
                &["id=\"import-form\"", "id=\"use\""],
            ),
            (
                "/train/models",
                &["id=\"use\"", "id=\"jobs\""],
                &["id=\"train\"", "id=\"autolabel-form\""],
            ),
        ];
        for (url, has, hasnt) in cases {
            let (_, html) = get_text(&state, url).await;
            for h in has {
                assert!(html.contains(h), "{url} should hold {h}");
            }
            for h in hasnt {
                assert!(!html.contains(h), "{url} should not hold {h}");
            }
        }
    }

    /// A form's answer is the tab it belongs to, whichever address it posted to.
    #[tokio::test]
    async fn forms_answer_with_their_tab() {
        // These forms reach `training::service`, which moves the process's queue to this test's
        // folder and would replace the jobs another test is waiting on.
        let _queue = QUEUE_TEST_LOCK.lock().await;
        let dir = scratch();
        let state = state_in(&dir);
        let on_tab = |html: &str, tab: &str| html.contains(&format!("data-tab=\"{tab}\""));
        for (url, form, tab) in [
            ("/training/queue", "dataset=%2Fx", "train"),
            ("/training/datasets", "add=%2Fno%2Fsuch", "train"),
            ("/training/import", "source=path", "import"),
            ("/training/autolabel", "mode=model", "autolabel"),
            ("/training/use", "id=nope&role=night", "models"),
            ("/training/revert", "forget=1", "models"),
            ("/training/cancel", "id=nope", "train"),
            ("/training/cancel", "id=nope&tab=import", "import"),
            ("/training/pause", "id=nope&tab=autolabel", "autolabel"),
            ("/training/remove", "id=nope&tab=models", "models"),
            ("/dataset/settings", "enabled=on&max_gb=0", "collect"),
        ] {
            let (_, _, b) = call(&state, post_form(url, form, true)).await;
            let html = String::from_utf8(b).unwrap();
            assert!(on_tab(&html, tab), "{url} {form}: not on {tab}");
        }
    }

    #[test]
    fn a_walk_answers_and_is_kept_while_the_next_one_runs() {
        let dir = scratch();
        let first = tally_of(&dir, Duration::from_secs(60)).expect("a walk of an empty folder");
        assert_eq!(first.frames, 0);
        // Kept: the same answer, no new walk needed.
        let again = tally_of(&dir, Duration::ZERO).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        // Past its time it still answers at once, while a new walk runs behind it.
        let old = Instant::now().checked_sub(Duration::from_secs(60)).unwrap();
        if let Some(k) = KEPT.lock_recover().get_mut(&dir) {
            k.done.as_mut().unwrap().0 = old;
        }
        let stale = tally_of(&dir, Duration::ZERO).unwrap();
        assert!(Arc::ptr_eq(&first, &stale));
    }

    #[test]
    fn slow_walks_are_kept_longer() {
        assert_eq!(keep_for(Duration::from_millis(5)), Duration::from_secs(10));
        assert_eq!(keep_for(Duration::from_secs(3)), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn the_advanced_sections_start_closed() {
        let dir = scratch();
        let state = state_in(&dir);
        let (_, html) = get_text(&state, "/train/autolabel").await;
        assert!(html.contains("<details class=\"tr-more\" id=\"al-advanced\">"));
        let (_, html) = get_text(&state, "/train/train").await;
        assert!(html.contains("<details class=\"tr-more\" id=\"train-advanced\">"));
        let (_, html) = get_text(&state, "/train/collect").await;
        assert!(html.contains("<details class=\"tr-more\" id=\"collect-advanced\">"));
        let (_, html) = get_text(&state, "/train/review").await;
        assert!(html.contains("<details class=\"card rv-bulk\" id=\"rv-bulk\">"));
        assert!(html.contains("<details class=\"help\">"));
    }
}
