//! The Jobs tab of the Train area (`GET /train/jobs`) and what fills it (`GET /v1/train/jobs`,
//! drawn by `assets/train-area.js`): every job a person started that is still working, in one
//! list, with its step, percent, time left and a link to where it lives:
//!
//! - the training queue: imports, auto-label runs, overlap checks, dataset checks and training
//!   runs (their rows on their own tabs, `#job-<id>`);
//! - side jobs beside it ([`super::side_jobs`]): dataset exports and X-AnyLabeling imports of the
//!   Dataset tab (`side-<n>`);
//! - a benchmark run (`/benchmark/run`, id `benchmark`), from the benchmark runner's progress;
//! - model exports, queued or running (the Config page's exported models, id
//!   `model-export-<resource>`), from the exporter's state.
//!
//! The ids never collide: queue ids are times, the others carry their prefix. Everything is read
//! from memory (the queue's records, the side jobs, the benchmark runner, the exporter), no disk,
//! so the page can ask every few seconds.
//!
//! The estimates say what they cover. Training comes from its epochs and is a range, from the
//! early stop if nothing improves any more to the epoch cap, and leaves out the export and the
//! benchmark after it. Any other step extrapolates that step's own percent
//! ([`crate::training::eta`]), and the steps after it are unknown, so a step that is not the job's
//! last says "in this step".

use super::side_jobs::SideView;
use super::train_area::{JOBS, TRAIN, tab_title, task_tab};
use super::{AppState, blocking, render, training};
use crate::benchmark::service::Running as BenchRunning;
use crate::resources::export::{ExportState, ExportStatus};
use crate::training::estimate::{format_duration, to_u64};
use crate::training::eta::JobTiming;
use crate::training::job::TrainStage;
use crate::training::progress::TrainProgress;
use crate::training::service::{RunRecord, RunState};
use crate::training::task::{TaskKind, TaskRecord};
use askama::Template;
use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

/// Under this an estimate is "under a minute" and a range's two ends read as one.
const SAME_ESTIMATE_S: f64 = 60.0;

/// The time line of a queued job.
const WAITING: &str = "waiting for its turn";

#[derive(Template)]
#[template(path = "train/running.html")]
struct JobsTabTemplate {
    nav: &'static str,
    tab: &'static str,
    tab_title: &'static str,
    version: &'static str,
}

/// `GET /train/jobs`.
pub(super) async fn page() -> Response {
    render(&JobsTabTemplate {
        nav: "train",
        tab: JOBS,
        tab_title: tab_title(JOBS),
        version: crate::VERSION,
    })
}

/// A job of the list: on the queue or beside it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct JobLine {
    pub(super) id: String,
    /// "Training", "Import", "Dataset export", "Benchmark", ...
    pub(super) kind: &'static str,
    /// The run's model name, the task's title, the export's formats and folder, ...
    pub(super) title: String,
    pub(super) paused: bool,
    /// Waiting for its turn: nothing to measure yet.
    pub(super) queued: bool,
    /// `Step 5 of 8`; empty before the first step.
    pub(super) step: String,
    /// The step's name, capitalised: `Labelling`.
    pub(super) phase: String,
    /// The step's own short line: `312 of 900 frames`, `Epoch 12 of 50`.
    pub(super) detail: String,
    /// Of the whole job.
    pub(super) percent: u8,
    /// `about 12 min left in this step`, `running for 3 min` until it can tell.
    pub(super) time: String,
    /// The tab's (or page's) title, for the link.
    pub(super) tab: &'static str,
    /// The job's row on its tab, or the card or page that shows it.
    pub(super) link: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub(super) struct Jobs {
    pub(super) active: Vec<JobLine>,
    pub(super) queued: Vec<JobLine>,
}

/// `Labelling` from `labelling`.
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or_else(String::new, |f| {
        f.to_uppercase().chain(c).collect::<String>()
    })
}

/// What the training stage has left, `least` to `most` seconds.
fn training_time(least: f64, most: f64) -> String {
    if most < SAME_ESTIMATE_S {
        "under a minute of training left".to_string()
    } else if most - least < SAME_ESTIMATE_S.max(most * 0.1) {
        format!("about {} of training left", format_duration(most))
    } else if least < SAME_ESTIMATE_S {
        format!("up to {} of training left", format_duration(most))
    } else {
        format!(
            "{} to {} of training left",
            format_duration(least),
            format_duration(most)
        )
    }
}

/// The time column of a job; `timing` is None when the clock has not started.
fn time_text(
    paused: bool,
    last_step: bool,
    timing: Option<JobTiming>,
    training: Option<&TrainProgress>,
) -> String {
    let Some(t) = timing else {
        return "starting".to_string();
    };
    let ran = format_duration(t.elapsed_s);
    if paused {
        return format!("paused after {ran}");
    }
    let estimate = match training {
        Some(p) => p
            .time_left(t.paused_s)
            .map(|(least, most)| training_time(least, most)),
        None => t.step_left_s.map(|s| {
            let scope = if last_step { "" } else { " in this step" };
            if s < SAME_ESTIMATE_S {
                format!("under a minute left{scope}")
            } else {
                format!("about {} left{scope}", format_duration(s))
            }
        }),
    };
    estimate.unwrap_or_else(|| format!("running for {ran}"))
}

fn step_text(step: usize, steps: usize) -> String {
    if step == 0 || steps == 0 {
        String::new()
    } else {
        format!("Step {step} of {steps}")
    }
}

fn link(tab: &str, id: &str) -> String {
    format!("/train/{tab}#job-{id}")
}

fn run_line(r: &RunRecord, timing: Option<JobTiming>) -> JobLine {
    // In the training stage only the epochs tell; its percent jumps at the export.
    let training =
        (r.stage == Some(TrainStage::Train)).then(|| r.progress.clone().unwrap_or_default());
    JobLine {
        id: r.id.clone(),
        kind: "Training",
        title: r.name.clone(),
        paused: r.state == RunState::Paused,
        queued: r.state == RunState::Queued,
        step: step_text(r.step, r.steps),
        phase: r.stage.map_or_else(String::new, |s| capitalised(s.label())),
        detail: training
            .as_ref()
            .and_then(TrainProgress::brief)
            .map_or_else(|| r.detail.clone(), |b| capitalised(&b)),
        percent: r.percent,
        time: if r.state == RunState::Queued {
            WAITING.to_string()
        } else {
            time_text(
                r.state == RunState::Paused,
                r.step >= r.steps,
                timing,
                training.as_ref(),
            )
        },
        tab: tab_title(TRAIN),
        link: link(TRAIN, &r.id),
    }
}

/// A task's kind in a job line.
const fn task_kind(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Import => "Import",
        TaskKind::Autolabel => "Auto-label",
        TaskKind::Overlaps => "Overlap check",
        TaskKind::Refine => "Dataset check",
    }
}

fn task_line(t: &TaskRecord, timing: Option<JobTiming>, tasks: &[TaskRecord]) -> JobLine {
    let tab = task_tab(t.kind);
    JobLine {
        id: t.id.clone(),
        kind: task_kind(t.kind),
        title: t.title.clone(),
        paused: t.state == RunState::Paused,
        queued: t.state == RunState::Queued,
        step: step_text(t.step, t.steps),
        phase: t.stage.as_deref().map_or_else(String::new, capitalised),
        detail: t.detail.clone(),
        percent: t.percent,
        time: if t.state == RunState::Queued {
            match &t.after {
                Some(a) => {
                    let dep = tasks.iter().find(|d| d.id == *a);
                    format!("starts after {}", dep.map_or(a, |d| &d.title))
                }
                None => WAITING.to_string(),
            }
        } else {
            time_text(t.state == RunState::Paused, t.step >= t.steps, timing, None)
        },
        tab: tab_title(tab),
        link: link(tab, &t.id),
    }
}

/// What works beside the queue, read at `now` ([`beside`]).
#[derive(Debug, Clone)]
pub(super) struct Beside {
    pub(super) side: Vec<SideView>,
    pub(super) benchmark: Option<BenchRunning>,
    /// The exporter's busy (queued or running) exports.
    pub(super) exports: Vec<ExportStatus>,
    pub(super) now: Instant,
}

impl Default for Beside {
    fn default() -> Self {
        Self {
            side: Vec::new(),
            benchmark: None,
            exports: Vec::new(),
            now: Instant::now(),
        }
    }
}

/// What works beside the queue now: memory only, one short lock each.
fn beside(state: &AppState) -> Beside {
    let now = Instant::now();
    let exports = state
        .resources
        .as_ref()
        .and_then(|c| c.provisioner.exporter())
        .map(|e| e.statuses())
        .unwrap_or_default();
    Beside {
        side: state.side_jobs.views(now),
        benchmark: state.benchmark.running(),
        exports: exports.into_iter().filter(|s| s.state.is_busy()).collect(),
        now,
    }
}

/// A clock that only says how long a job ran, and its estimate of the rest when it has one.
fn plain_timing(elapsed_s: f64, left_s: Option<f64>) -> JobTiming {
    JobTiming {
        elapsed_s,
        paused_s: 0.0,
        step_left_s: left_s,
    }
}

/// Seconds from `since` to `now`.
fn seconds(since: Instant, now: Instant) -> f64 {
    now.saturating_duration_since(since).as_secs_f64()
}

fn side_line(v: &SideView) -> JobLine {
    let r = v.report.as_ref();
    JobLine {
        id: v.id.clone(),
        kind: v.kind.label(),
        title: v.title.clone(),
        paused: false,
        queued: false,
        // A job of one step has no "Step 1 of 1".
        step: r
            .filter(|r| r.steps > 1)
            .map_or_else(String::new, |r| step_text(r.step, r.steps)),
        phase: r.map_or_else(String::new, |r| capitalised(r.stage)),
        detail: r.map_or_else(String::new, |r| r.detail.clone()),
        percent: r.map_or(0, |r| r.percent),
        time: time_text(
            false,
            r.is_none_or(|r| r.step >= r.steps),
            r.map(|_| v.timing),
            None,
        ),
        tab: tab_title(v.kind.tab()),
        link: format!("/train/{}#{}", v.kind.tab(), v.kind.anchor()),
    }
}

/// Models a benchmark line names before "and N more".
const BENCH_TITLE_MODELS: usize = 3;

/// The benchmark run's title: its models.
fn bench_title(models: &[String]) -> String {
    match models.len() {
        0 => "The enabled models".to_string(),
        n if n <= BENCH_TITLE_MODELS => models.join(", "),
        n => format!(
            "{} and {} more",
            models
                .iter()
                .take(BENCH_TITLE_MODELS)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            n - BENCH_TITLE_MODELS
        ),
    }
}

/// The running benchmark, from the runner's own progress (`GET /v1/benchmark`).
fn bench_line(b: &BenchRunning, now: Instant) -> JobLine {
    let p = b.progress.as_ref();
    let phase = match p.map(|p| p.stage.as_str()) {
        None => String::new(),
        Some("download") => "Downloading image sets".to_string(),
        Some("pseudo-gt") => "Reference labels".to_string(),
        Some("benchmark") => "Measuring".to_string(),
        Some(other) => capitalised(other),
    };
    let step = p
        .filter(|p| p.stage == "benchmark" && p.runs_total > 0)
        .map_or_else(String::new, |p| {
            let run = (p.runs_done + 1).min(p.runs_total);
            format!("Run {run} of {}", p.runs_total)
        });
    let detail = p.map_or_else(String::new, |p| {
        if !p.text.is_empty() {
            capitalised(&p.text)
        } else if p.model.is_empty() {
            String::new()
        } else if p.total > 0 {
            format!(
                "{} on {}: {} {} of {}",
                p.model, p.device, p.phase, p.done, p.total
            )
        } else {
            format!("{} on {}: {}", p.model, p.device, p.phase)
        }
    });
    let timing = b
        .started
        .filter(|_| p.is_some())
        .map(|s| plain_timing(seconds(s, now), p.and_then(|p| p.eta_secs)));
    JobLine {
        id: "benchmark".to_string(),
        kind: "Benchmark",
        title: bench_title(&b.models),
        paused: false,
        queued: false,
        step,
        phase,
        detail,
        percent: p.map_or(0, |p| {
            u8::try_from(to_u64(p.percent.min(100.0))).unwrap_or(100)
        }),
        // The runner's estimate covers the whole run.
        time: time_text(false, true, timing, None),
        tab: "Benchmark",
        link: "/benchmark/run".to_string(),
    }
}

/// A model export of the exporter that is queued or running (None for any other state).
fn export_line(s: &ExportStatus, now: Instant) -> Option<JobLine> {
    let base = |queued: bool| JobLine {
        id: format!("model-export-{}", s.id),
        kind: "Model export",
        title: s.title.to_string(),
        paused: false,
        queued,
        step: String::new(),
        phase: String::new(),
        detail: String::new(),
        percent: 0,
        time: WAITING.to_string(),
        tab: "Resources",
        link: "/config#res-models-export".to_string(),
    };
    match &s.state {
        ExportState::Queued => Some(base(true)),
        ExportState::Running {
            stage,
            step,
            steps,
            percent,
            detail,
        } => Some(JobLine {
            step: step_text(*step, *steps),
            phase: capitalised(stage.label()),
            detail: detail.clone(),
            percent: *percent,
            time: time_text(
                false,
                step >= steps,
                s.started.map(|t| plain_timing(seconds(t, now), None)),
                None,
            ),
            ..base(false)
        }),
        _ => None,
    }
}

/// Every job of `runs` and `tasks` on the queue (`timing` is the queue's clock) and what works
/// `beside` it: queue jobs first, then side jobs, the benchmark and model exports.
pub(super) fn view(
    runs: &[RunRecord],
    tasks: &[TaskRecord],
    timing: &dyn Fn(&str) -> Option<JobTiming>,
    beside: &Beside,
) -> Jobs {
    let lines = |keep: &dyn Fn(RunState) -> bool| -> Vec<JobLine> {
        runs.iter()
            .filter(|r| keep(r.state))
            .map(|r| run_line(r, timing(&r.id)))
            .chain(
                tasks
                    .iter()
                    .filter(|t| keep(t.state))
                    .map(|t| task_line(t, timing(&t.id), tasks)),
            )
            .collect()
    };
    let exports = beside
        .exports
        .iter()
        .filter_map(|s| export_line(s, beside.now));
    let (queued_exports, running_exports): (Vec<JobLine>, Vec<JobLine>) =
        exports.partition(|l| l.queued);
    let mut active = lines(&RunState::is_active);
    active.extend(beside.side.iter().map(side_line));
    active.extend(beside.benchmark.iter().map(|b| bench_line(b, beside.now)));
    active.extend(running_exports);
    let mut queued = lines(&|s| s == RunState::Queued);
    queued.extend(queued_exports);
    Jobs { active, queued }
}

/// Query of `GET /v1/train/jobs`: `task=<id>` adds that task's record (`task`, null when there
/// is none), `tasks=busy` the records of every queued, running or paused task (`tasks`), both
/// without `log` and `result` ([`task_brief`]). For a page that follows one job (Review) without
/// the runs and histories of `GET /v1/training`.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct JobsQuery {
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    tasks: Option<String>,
}

/// A task's record as the Jobs answer carries it: what a follower shows (state, stage, step,
/// percent, detail, summary, error), not its log or result.
fn task_brief(t: &TaskRecord) -> serde_json::Value {
    let mut v = serde_json::to_value(t).unwrap_or(serde_json::Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.remove("log");
        o.remove("result");
    }
    v
}

/// The answer of `GET /v1/train/jobs` for `q`: `jobs` ([`view`]), and the task records `q` asks
/// for.
fn answer(
    runs: &[RunRecord],
    tasks: &[TaskRecord],
    timing: &dyn Fn(&str) -> Option<JobTiming>,
    beside: &Beside,
    q: &JobsQuery,
) -> serde_json::Value {
    let mut v = serde_json::Map::new();
    v.insert("success".into(), true.into());
    v.insert(
        "jobs".into(),
        serde_json::to_value(view(runs, tasks, timing, beside)).unwrap_or_default(),
    );
    if let Some(id) = q.task.as_deref() {
        let t = tasks.iter().find(|t| t.id == id);
        v.insert("task".into(), t.map_or(serde_json::Value::Null, task_brief));
    }
    if q.tasks.as_deref() == Some("busy") {
        let busy = tasks.iter().filter(|t| t.state.is_busy()).map(task_brief);
        v.insert("tasks".into(), busy.collect());
    }
    serde_json::Value::Object(v)
}

/// `GET /v1/train/jobs` (`?task=<id>`, `?tasks=busy`: [`JobsQuery`]). Off the async workers: the
/// queue's records are cloned under its locks and, without a started queue, `runs.json` and
/// `tasks.json` are read; what works beside the queue is read from memory ([`beside`]); the
/// answer is serialized there too.
pub(super) async fn jobs(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<JobsQuery>,
) -> Response {
    blocking(move || {
        let v = answer(
            &training::records(&state),
            &training::tasks(&state),
            &training::job_timing,
            &beside(&state),
            &q,
        );
        Json(v).into_response()
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::dataset::SceneFilter;

    const NO_CLOCK: fn(&str) -> Option<JobTiming> = |_| None;

    /// Nothing beside the queue.
    fn none() -> Beside {
        Beside::default()
    }

    fn clock(elapsed_s: f64, step_left_s: Option<f64>) -> JobTiming {
        JobTiming {
            elapsed_s,
            paused_s: 0.0,
            step_left_s,
        }
    }

    fn task(id: &str, kind: TaskKind, state: RunState) -> TaskRecord {
        TaskRecord {
            id: id.into(),
            kind,
            title: "Label 900 frames".into(),
            created: String::new(),
            state,
            stage: Some("labelling".into()),
            step: 4,
            steps: 5,
            percent: 72,
            detail: "312 of 900 frames".into(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
            after: None,
            uses: Vec::new(),
        }
    }

    fn run(id: &str, state: RunState) -> RunRecord {
        training::tests::record(id, SceneFilter::default(), None, state, None)
    }

    #[test]
    fn a_task_line_says_step_detail_time_and_where_it_lives() {
        let tasks = [task("t1", TaskKind::Autolabel, RunState::Running)];
        let v = view(&[], &tasks, &|_| Some(clock(300.0, Some(720.0))), &none());
        assert_eq!(v.active.len(), 1);
        let l = &v.active[0];
        assert_eq!(l.kind, "Auto-label");
        assert_eq!(l.step, "Step 4 of 5");
        assert_eq!(l.phase, "Labelling");
        assert_eq!(l.detail, "312 of 900 frames");
        assert_eq!(l.percent, 72);
        assert_eq!(l.time, "about 12 min left in this step");
        assert_eq!(l.tab, "Auto-label");
        assert_eq!(l.link, "/train/autolabel#job-t1");
    }

    #[test]
    fn each_kind_links_to_its_own_tab() {
        let tasks = [
            task("i", TaskKind::Import, RunState::Running),
            task("o", TaskKind::Overlaps, RunState::Paused),
            task("f", TaskKind::Refine, RunState::Running),
        ];
        let runs = [run("r", RunState::Running)];
        let v = view(&runs, &tasks, &NO_CLOCK, &none());
        let links: Vec<_> = v.active.iter().map(|l| l.link.as_str()).collect();
        assert_eq!(
            links,
            [
                "/train/train#job-r",
                "/train/import#job-i",
                "/train/autolabel#job-o",
                "/train/refine#job-f"
            ]
        );
        assert!(v.active[2].paused);
    }

    #[test]
    fn only_the_last_step_speaks_for_the_whole_job() {
        let mut t = task("t", TaskKind::Import, RunState::Running);
        t.step = 7;
        t.steps = 7;
        let v = view(&[], &[t], &|_| Some(clock(10.0, Some(30.0))), &none());
        assert_eq!(v.active[0].time, "under a minute left");
    }

    #[test]
    fn with_no_estimate_the_line_says_how_long_it_ran() {
        let t = clock(180.0, None);
        assert_eq!(time_text(false, false, Some(t), None), "running for 3 min");
        assert_eq!(time_text(false, false, None, None), "starting");
        assert_eq!(
            time_text(true, false, Some(clock(180.0, Some(60.0))), None),
            "paused after 3 min"
        );
    }

    #[test]
    fn training_time_is_a_range_from_its_epochs_not_the_steps_percent() {
        let p = TrainProgress {
            epoch: 11,
            epochs: 50,
            elapsed_s: 1100.0,
            stop_epoch: Some(20),
            ..TrainProgress::default()
        };
        let t = clock(1200.0, Some(5.0));
        assert_eq!(
            time_text(false, false, Some(t), Some(&p)),
            "18 min to 73 min of training left"
        );
        // Before the first batch, and during the export: no guess from the stage's percent.
        let none = TrainProgress::default();
        assert_eq!(
            time_text(false, false, Some(t), Some(&none)),
            "running for 20 min"
        );
    }

    #[test]
    fn a_training_range_collapses_when_its_ends_meet() {
        assert_eq!(
            training_time(1000.0, 1000.0),
            "about 17 min of training left"
        );
        assert_eq!(training_time(20.0, 1000.0), "up to 17 min of training left");
        assert_eq!(training_time(10.0, 30.0), "under a minute of training left");
    }

    #[test]
    fn queued_jobs_are_listed_with_a_link_and_finished_ones_not_at_all() {
        let mut after = task("a", TaskKind::Autolabel, RunState::Queued);
        after.after = Some("imp1".into());
        after.stage = None;
        after.step = 0;
        after.detail = String::new();
        let tasks = [
            after,
            task("imp1", TaskKind::Import, RunState::Done),
            task("c", TaskKind::Import, RunState::Queued),
            task("d", TaskKind::Import, RunState::Done),
            task("e", TaskKind::Overlaps, RunState::Failed),
        ];
        let runs = [
            run("r1", RunState::Queued),
            run("r2", RunState::Cancelled),
            run("r3", RunState::Done),
        ];
        let v = view(&runs, &tasks, &NO_CLOCK, &none());
        assert!(v.active.is_empty());
        let q: Vec<_> = v
            .queued
            .iter()
            .map(|q| (q.id.as_str(), q.time.as_str(), q.link.as_str(), q.queued))
            .collect();
        assert_eq!(
            q,
            [
                ("r1", WAITING, "/train/train#job-r1", true),
                (
                    "a",
                    "starts after Label 900 frames",
                    "/train/autolabel#job-a",
                    true
                ),
                ("c", WAITING, "/train/import#job-c", true),
            ]
        );
        assert_eq!(v.queued[1].step, "");
    }

    #[test]
    fn a_follower_gets_its_task_without_log_or_result_and_the_full_answer_stays() {
        let mut done = task("d", TaskKind::Autolabel, RunState::Done);
        done.log = vec!["a long line".into(); 50];
        done.result = Some(serde_json::json!({ "frames": 900 }));
        done.summary = Some("labelled 900 frames".into());
        let tasks = [done, task("o", TaskKind::Overlaps, RunState::Running)];
        let plain = answer(&[], &tasks, &NO_CLOCK, &none(), &JobsQuery::default());
        assert!(plain.get("task").is_none() && plain.get("tasks").is_none());
        assert_eq!(
            plain["jobs"],
            serde_json::to_value(view(&[], &tasks, &NO_CLOCK, &none())).unwrap()
        );
        let q = JobsQuery {
            task: Some("d".into()),
            tasks: Some("busy".into()),
        };
        let v = answer(&[], &tasks, &NO_CLOCK, &none(), &q);
        assert_eq!(v["task"]["state"], "done");
        assert_eq!(v["task"]["summary"], "labelled 900 frames");
        assert!(v["task"].get("log").is_none() && v["task"].get("result").is_none());
        let busy: Vec<_> = v["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["id"].clone())
            .collect();
        assert_eq!(busy, ["o"]);
        assert_eq!(v["tasks"][0]["kind"], "overlaps");
        let gone = JobsQuery {
            task: Some("nope".into()),
            tasks: None,
        };
        assert!(answer(&[], &tasks, &NO_CLOCK, &none(), &gone)["task"].is_null());
    }

    fn export_status(
        id: &'static str,
        state: ExportState,
        started: Option<Instant>,
    ) -> ExportStatus {
        ExportStatus {
            id,
            title: "YOLO26 s",
            state,
            log: vec!["a child line".into()],
            started,
        }
    }

    /// Everything beside the queue at once: an export of the Dataset tab, a benchmark run, a
    /// running and a queued model export, beside a training run.
    fn busy_beside() -> (
        Beside,
        Arc<crate::server::side_jobs::SideJobs>,
        crate::server::side_jobs::SideJob,
    ) {
        use crate::benchmark::service::ProgressView;
        use crate::resources::export::Stage;
        use crate::server::side_jobs::{SideJobs, SideKind, export_report};
        let now = Instant::now();
        let ago = |s: u64| now.checked_sub(std::time::Duration::from_secs(s)).unwrap();
        let jobs = Arc::new(SideJobs::default());
        let job = jobs.start(SideKind::DatasetExport, "YOLO, COCO to D:/exports/a".into());
        job.report(export_report(
            crate::collect::labels::export::ExportStep::Writing { done: 50, of: 100 },
        ));
        let beside = Beside {
            side: jobs.views(now),
            benchmark: Some(BenchRunning {
                models: vec!["a".into(), "b".into(), "c".into(), "d".into()],
                progress: Some(ProgressView {
                    stage: "benchmark".into(),
                    model: "a".into(),
                    device: "openvino:gpu".into(),
                    phase: "timing".into(),
                    done: 40,
                    total: 200,
                    runs_done: 2,
                    runs_total: 8,
                    percent: 31.6,
                    eta_secs: Some(600.0),
                    ..ProgressView::default()
                }),
                started: Some(ago(300)),
            }),
            exports: vec![
                export_status(
                    "model:yolo26s",
                    ExportState::Running {
                        stage: Stage::Packages,
                        step: 3,
                        steps: 8,
                        percent: 40,
                        detail: "downloading torch".into(),
                    },
                    Some(ago(240)),
                ),
                export_status("model:yolo26m", ExportState::Queued, None),
            ],
            now,
        };
        (beside, jobs, job)
    }

    #[test]
    fn exports_imports_benchmarks_and_model_exports_are_listed_beside_the_queue() {
        let (beside, _jobs, _job) = busy_beside();
        let runs = [run("r", RunState::Running)];
        let v = view(&runs, &[], &NO_CLOCK, &beside);
        let ids: Vec<_> = v.active.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(
            ids,
            ["r", "side-1", "benchmark", "model-export-model:yolo26s"]
        );

        let side = &v.active[1];
        assert_eq!(side.kind, "Dataset export");
        assert_eq!(side.title, "YOLO, COCO to D:/exports/a");
        assert_eq!(side.step, "Step 2 of 3");
        assert_eq!(side.phase, "Writing files");
        assert_eq!(side.detail, "50 of 100 files written");
        assert_eq!(side.percent, 61);
        assert_eq!(
            (side.tab, side.link.as_str()),
            ("Dataset", "/train/dataset#export")
        );
        assert!(side.time.starts_with("running for"), "{}", side.time);

        let b = &v.active[2];
        assert_eq!(b.kind, "Benchmark");
        assert_eq!(b.title, "a, b, c and 1 more");
        assert_eq!(b.step, "Run 3 of 8");
        assert_eq!(b.phase, "Measuring");
        assert_eq!(b.detail, "a on openvino:gpu: timing 40 of 200");
        assert_eq!(b.percent, 32);
        assert_eq!(b.time, "about 10 min left");
        assert_eq!((b.tab, b.link.as_str()), ("Benchmark", "/benchmark/run"));

        let e = &v.active[3];
        assert_eq!(e.kind, "Model export");
        assert_eq!(e.title, "YOLO26 s");
        assert_eq!(e.step, "Step 3 of 8");
        assert_eq!(e.phase, "Installing the pinned Python packages");
        assert_eq!(e.detail, "downloading torch");
        assert_eq!(e.percent, 40);
        assert_eq!(e.time, "running for 4 min");
        assert_eq!(e.link, "/config#res-models-export");

        let q: Vec<_> = v
            .queued
            .iter()
            .map(|l| (l.id.as_str(), l.queued, l.time.as_str()))
            .collect();
        assert_eq!(q, [("model-export-model:yolo26m", true, WAITING)]);

        // One id per line across every source.
        let mut all: Vec<_> = v
            .active
            .iter()
            .chain(&v.queued)
            .map(|l| l.id.clone())
            .collect();
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n);
    }

    #[test]
    fn a_side_job_ends_its_line_when_its_guard_drops() {
        let (beside, jobs, job) = busy_beside();
        assert!(
            view(&[], &[], &NO_CLOCK, &beside)
                .active
                .iter()
                .any(|l| l.id == "side-1")
        );
        drop(job);
        let now = Beside {
            side: jobs.views(Instant::now()),
            ..Beside::default()
        };
        assert!(view(&[], &[], &NO_CLOCK, &now).active.is_empty());
    }

    #[test]
    fn a_side_job_before_its_first_report_is_starting_and_a_benchmark_before_its_first_too() {
        use crate::server::side_jobs::{SideJobs, SideKind};
        let jobs = Arc::new(SideJobs::default());
        let _g = jobs.start(SideKind::XanyImport, "From D:/x".into());
        let beside = Beside {
            side: jobs.views(Instant::now()),
            benchmark: Some(BenchRunning {
                models: Vec::new(),
                progress: None,
                started: Some(Instant::now()),
            }),
            ..Beside::default()
        };
        let v = view(&[], &[], &NO_CLOCK, &beside);
        let l = &v.active[0];
        assert_eq!(
            (l.kind, l.time.as_str(), l.percent),
            ("X-AnyLabeling import", "starting", 0)
        );
        assert_eq!(l.step, "");
        assert_eq!(l.link, "/train/dataset#xany-import");
        let b = &v.active[1];
        assert_eq!(
            (b.title.as_str(), b.time.as_str()),
            ("The enabled models", "starting")
        );
        // A finished or failed model export is no job.
        let gone = export_status("x", ExportState::Installed, None);
        assert!(export_line(&gone, Instant::now()).is_none());
    }

    #[test]
    fn nothing_on_the_queue_is_an_empty_summary() {
        let done = [task("d", TaskKind::Import, RunState::Done)];
        assert_eq!(view(&[], &done, &NO_CLOCK, &none()), Jobs::default());
    }
}
