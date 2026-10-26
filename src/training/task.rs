//! Jobs of the training queue other than a training run: footage imports
//! ([`super::import`]) and auto-label runs ([`super::autolabel`]). They share the queue's one
//! worker thread with the training runs ([`super::service::TrainingService`]), so a GPU or the
//! training environment is used by one job at a time, and each holds the cross-process
//! training lock (`tools/.train.lock`) while it runs. A [`Task`] reports numbered stages and
//! log lines; its [`TaskRecord`] (state, progress, log tail, result) is kept like a run's, in
//! `<training dir>/tasks.json`.

use super::service::RunState;
use crate::resources::export::{CommandRunner, Fetcher};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// File of the task records in the training folder.
pub const TASKS_FILE: &str = "tasks.json";

/// What a task does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskKind {
    Import,
    Autolabel,
}

impl TaskKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            TaskKind::Import => "import",
            TaskKind::Autolabel => "autolabel",
        }
    }
}

/// The seams a task runs on (the queue's).
#[derive(Clone, Copy)]
pub struct TaskContext<'a> {
    pub runner: &'a dyn CommandRunner,
    pub fetcher: &'a dyn Fetcher,
}

impl std::fmt::Debug for TaskContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskContext").finish_non_exhaustive()
    }
}

/// What a task reports while it runs.
#[derive(Debug)]
pub enum TaskReport<'a> {
    /// Stage `step` of `steps` (1-based), `percent` of the whole task.
    Stage {
        stage: &'a str,
        step: usize,
        steps: usize,
        percent: u8,
        detail: &'a str,
    },
    Line(&'a str),
}

/// A finished task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcome {
    /// One line for the page and the CLI.
    pub summary: String,
    /// The task's report (counts, frames, warnings).
    pub result: serde_json::Value,
}

/// A queued job (see the module docs).
pub trait Task: Send {
    fn id(&self) -> &str;
    fn kind(&self) -> TaskKind;
    /// "import of clip.mp4", "auto-label 20 frames".
    fn title(&self) -> String;
    /// Number of stages.
    fn steps(&self) -> usize;
    /// Run every stage. A cancelled run returns an error carrying
    /// [`crate::resources::export::Cancelled`], or `Ok` when it stopped cleanly and kept partial
    /// work: the record is then Cancelled with that outcome's summary and result.
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> anyhow::Result<TaskOutcome>;
}

/// One task as the Training page and the CLI show it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: String,
    pub kind: TaskKind,
    pub title: String,
    pub created: String,
    pub state: RunState,
    #[serde(default)]
    pub stage: Option<String>,
    pub step: usize,
    pub steps: usize,
    pub percent: u8,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub log: Vec<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub finished: Option<String>,
}

impl TaskRecord {
    pub fn new(task: &dyn Task, created: String) -> Self {
        Self {
            id: task.id().to_string(),
            kind: task.kind(),
            title: task.title(),
            created,
            state: RunState::Queued,
            stage: None,
            step: 0,
            steps: task.steps(),
            percent: 0,
            detail: String::new(),
            log: Vec::new(),
            summary: None,
            result: None,
            error: None,
            finished: None,
        }
    }

    /// "extracting frames (2/5): 12 of 40" / "failed while ...: ..." / "done: ...".
    pub fn describe(&self) -> String {
        match self.state {
            RunState::Queued => "queued".into(),
            RunState::Running => match &self.stage {
                Some(s) if self.detail.is_empty() => format!("{s} ({}/{})", self.step, self.steps),
                Some(s) => format!("{s} ({}/{}): {}", self.step, self.steps, self.detail),
                None => "starting".into(),
            },
            RunState::Paused => match &self.stage {
                Some(s) if self.detail.is_empty() => {
                    format!("paused ({s} {}/{})", self.step, self.steps)
                }
                Some(s) => format!("paused ({s} {}/{}): {}", self.step, self.steps, self.detail),
                None => "paused".into(),
            },
            RunState::Done => format!("done: {}", self.summary.as_deref().unwrap_or("")),
            RunState::Failed => format!(
                "failed{}: {}",
                self.stage
                    .as_ref()
                    .map(|s| format!(" while {s}"))
                    .unwrap_or_default(),
                self.error.as_deref().unwrap_or("unknown error")
            ),
            RunState::Cancelled => match self.summary.as_deref() {
                Some(s) if !s.is_empty() => format!("cancelled: {s}"),
                _ => "cancelled".into(),
            },
        }
    }
}

/// `tasks.json` beside `runs.json`.
pub fn tasks_file(runs_file: &Path) -> PathBuf {
    runs_file.with_file_name(TASKS_FILE)
}

/// Load the records of `path`; busy ones become interrupted failures.
pub fn load_records(path: &Path) -> Vec<TaskRecord> {
    let mut v: Vec<TaskRecord> = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    for r in &mut v {
        if r.state.is_busy() {
            r.state = RunState::Failed;
            r.error = Some(
                "interrupted: the service stopped while this job was queued or running; start \
                 it again"
                    .into(),
            );
        }
    }
    v
}

/// Save `records` to `path` (atomic; a failure only warns).
pub fn save_records(path: &Path, records: &[TaskRecord]) {
    let tmp = path.with_extension("json.tmp");
    let r = serde_json::to_vec_pretty(records)
        .map_err(anyhow::Error::from)
        .and_then(|b| {
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p)?;
            }
            std::fs::write(&tmp, b)?;
            std::fs::rename(&tmp, path)?;
            Ok(())
        });
    if let Err(e) = r {
        tracing::warn!("saving {}: {e:#}", path.display());
    }
}

/// Overall percent of stage `step` (1-based) of `steps` at `pct` percent, with equal stages.
pub fn overall(step: usize, steps: usize, pct: u8) -> u8 {
    let steps = steps.max(1);
    let done = step.saturating_sub(1).min(steps);
    let p = (done * 100 + usize::from(pct.min(100))) / steps;
    u8::try_from(p.min(100)).unwrap_or(100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overall_percent_and_describe() {
        assert_eq!(overall(1, 4, 0), 0);
        assert_eq!(overall(1, 4, 100), 25);
        assert_eq!(overall(3, 4, 50), 62);
        assert_eq!(overall(4, 4, 100), 100);
        assert_eq!(overall(9, 4, 100), 100);
        let mut r = TaskRecord {
            id: "x".into(),
            kind: TaskKind::Import,
            title: "import".into(),
            created: String::new(),
            state: RunState::Running,
            stage: Some("extracting frames".into()),
            step: 2,
            steps: 5,
            percent: 30,
            detail: "12 frames".into(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
        };
        assert_eq!(r.describe(), "extracting frames (2/5): 12 frames");
        r.state = RunState::Paused;
        assert_eq!(r.describe(), "paused (extracting frames 2/5): 12 frames");
        r.state = RunState::Cancelled;
        assert_eq!(r.describe(), "cancelled");
        r.summary = Some("kept 7 labelled frames".into());
        assert_eq!(r.describe(), "cancelled: kept 7 labelled frames");
        r.summary = None;
        r.state = RunState::Failed;
        r.error = Some("boom".into());
        assert_eq!(r.describe(), "failed while extracting frames: boom");
    }

    #[test]
    fn records_round_trip_and_busy_ones_are_interrupted() {
        let dir = std::env::temp_dir().join(format!("bop-tasks-{}", uuid::Uuid::new_v4()));
        let p = tasks_file(&dir.join("runs.json"));
        assert!(p.ends_with(TASKS_FILE));
        let r = TaskRecord {
            id: "a".into(),
            kind: TaskKind::Autolabel,
            title: "t".into(),
            created: "c".into(),
            state: RunState::Paused,
            stage: None,
            step: 0,
            steps: 3,
            percent: 0,
            detail: String::new(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""state":"paused""#), "{json}");
        assert_eq!(serde_json::from_str::<TaskRecord>(&json).unwrap(), r);
        save_records(&p, &[r]);
        let back = load_records(&p);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].state, RunState::Failed);
        assert!(back[0].error.as_deref().unwrap().contains("interrupted"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
