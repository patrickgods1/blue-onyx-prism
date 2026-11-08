//! Jobs of the training queue other than a training run: footage imports
//! ([`super::import`]), auto-label runs ([`super::autolabel`]) and overlap checks
//! ([`super::overlaps`]). They share the queue's one worker thread with the training runs ([`super::service::TrainingService`]), so a GPU or the
//! training environment is used by one job at a time, and each holds the cross-process
//! training lock (`tools/.train.lock`) while it runs. A [`Task`] reports numbered stages and
//! log lines; its [`TaskRecord`] (state, progress, log tail, result) is kept like a run's, in
//! `<training dir>/tasks.json`. A task can follow another ([`Task::after`]: an auto-label run
//! queued with its import): it runs only when that one finished as it needs
//! ([`Task::skip_reason`]), else it ends Cancelled with the reason as its summary.
//!
//! [`load_or_move_aside`] and [`save_file`] are how `tasks.json` and `runs.json` reach the disk: a
//! damaged file or a record this build cannot read is set aside as `<name>.corrupt-<time>` before
//! the next save could write over it. Each saved record carries a `format` ([`RECORD_FORMAT`]); the
//! file stays a bare JSON list, so an older build (which ignores unknown fields) still reads it.

use super::service::RunState;
use crate::resources::export::{CommandRunner, Fetcher};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// File of the task records in the training folder.
pub const TASKS_FILE: &str = "tasks.json";
/// Layout version of a `runs.json` / `tasks.json` record, the `format` key [`save_file`] writes
/// on each. Bumped only when an older build would misread a record of this one; a new field
/// keeps it. A record of a later format is set aside unread ([`load_or_move_aside`]); 0.2.0
/// wrote none, which is this format.
pub const RECORD_FORMAT: u32 = 1;
/// Finished task records kept in `tasks.json` (the oldest go first); queued and running ones are
/// always kept. A task owns no files the list is needed for.
pub const MAX_FINISHED_TASKS: usize = 200;

/// Drop the oldest finished records of `tasks` (in list order, which is queue order) past
/// [`MAX_FINISHED_TASKS`]; busy ones stay whatever their number.
pub fn cap_finished(tasks: &mut Vec<TaskRecord>) {
    let finished = tasks.iter().filter(|r| !r.state.is_busy()).count();
    let mut extra = finished.saturating_sub(MAX_FINISHED_TASKS);
    tasks.retain(|r| {
        if extra > 0 && !r.state.is_busy() {
            extra -= 1;
            false
        } else {
            true
        }
    });
}

/// What a task does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskKind {
    Import,
    Autolabel,
    Overlaps,
}

impl TaskKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            TaskKind::Import => "import",
            TaskKind::Autolabel => "autolabel",
            TaskKind::Overlaps => "overlaps",
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
    /// Stage `step` of `steps` (1-based), `percent` of the whole task, `step_percent` of the stage.
    Stage {
        stage: &'a str,
        step: usize,
        steps: usize,
        percent: u8,
        step_percent: u8,
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
    /// Ids of the resources it reads while queued or running, for the ones removable by hand
    /// (kept in its record: [`TaskRecord::uses`]).
    fn uses(&self) -> Vec<String> {
        Vec::new()
    }
    /// The task this one follows (queued before it): it runs only when that one allows it
    /// ([`Self::skip_reason`]).
    fn after(&self) -> Option<&str> {
        None
    }
    /// Why this task does not run after `dep` (the record of [`Self::after`]); None = run. By
    /// default it runs only after a finished (Done) task.
    fn skip_reason(&self, dep: &TaskRecord) -> Option<String> {
        default_skip_reason(dep)
    }
}

/// [`Task::skip_reason`]'s default: "not run: the import it follows failed" unless `dep` is
/// Done.
pub fn default_skip_reason(dep: &TaskRecord) -> Option<String> {
    let what = match dep.state {
        RunState::Done => return None,
        RunState::Cancelled => "was cancelled",
        RunState::Failed => "failed",
        RunState::Queued | RunState::Running | RunState::Paused => "did not finish",
    };
    Some(format!(
        "not run: the {} it follows {what}",
        dep.kind.as_str()
    ))
}

/// One task as the Train tab and the CLI show it.
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
    /// The task it follows ([`Task::after`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// [`Task::uses`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uses: Vec<String>,
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
            after: task.after().map(str::to_string),
            uses: task.uses(),
        }
    }

    /// "extracting frames (2/5): 12 of 40" / "failed while ...: ..." / "done: ...".
    pub fn describe(&self) -> String {
        match self.state {
            RunState::Queued => match &self.after {
                Some(a) => format!("queued (after {a})"),
                None => "queued".into(),
            },
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

/// What [`load_or_move_aside`] read.
#[derive(Debug)]
pub struct Loaded<T> {
    pub records: Vec<T>,
    /// False when something damaged could not be set aside: it is still only in the file, so
    /// nothing may be saved over that.
    pub writable: bool,
}

/// Reads of a job record file before it counts as unreadable.
const READ_TRIES: u32 = 3;
/// Wait between those reads.
const READ_RETRY: std::time::Duration = std::time::Duration::from_millis(50);

/// The records of the JSON list at `path`. A missing file is an empty list. Anything else that
/// cannot be used is kept where the next save cannot reach it, logged as an error:
///
/// - a file that was read and is not a JSON list moves to `<name>.corrupt-<time>`, and the history
///   starts empty;
/// - a file that cannot be read stays where it is: a failed read says nothing about its content,
///   and antivirus or a backup can hold a file for a moment on Windows. After [`READ_TRIES`] tries
///   the history is empty and nothing is saved over the file until a restart;
/// - a record of an otherwise good file that does not parse (a [`TaskKind`] of a later version,
///   after a downgrade), or of a later `format` than [`RECORD_FORMAT`] (it might parse into
///   something it does not mean), is copied, alone, to such a file, and the others load. The file
///   itself keeps it until the next save, so a later load finds it again: the same records are not
///   copied twice.
///
/// Every reader comes through here, including those that never save (the page before the queue
/// starts, `train list`): a damaged file is damaged for all of them.
pub fn load_or_move_aside<T: DeserializeOwned>(path: &Path) -> Loaded<T> {
    let bytes = match read_retrying(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Loaded {
                records: Vec::new(),
                writable: true,
            };
        }
        Err(e) => {
            tracing::error!(
                "{} cannot be read ({e}): job history is empty for now and nothing is saved over \
                 the file. Check that no other program holds it, then restart",
                path.display()
            );
            return Loaded {
                records: Vec::new(),
                writable: false,
            };
        }
    };
    let list: Vec<serde_json::Value> = match serde_json::from_slice(&bytes) {
        Ok(l) => l,
        Err(e) => return move_aside(path, &format!("is damaged ({e})")),
    };
    let mut records = Vec::with_capacity(list.len());
    let mut unread = Vec::new();
    let mut first_error = None;
    for v in list {
        match record_format(&v).and_then(|()| T::deserialize(&v).map_err(|e| e.to_string())) {
            Ok(r) => records.push(r),
            Err(e) => {
                first_error.get_or_insert(e);
                unread.push(v);
            }
        }
    }
    let writable = match first_error {
        Some(e) => keep_aside(path, &unread, &e),
        None => true,
    };
    Loaded { records, writable }
}

/// Whether this build reads record `v` by its `format`: none (0.2.0) or at most
/// [`RECORD_FORMAT`].
fn record_format(v: &serde_json::Value) -> Result<(), String> {
    match v.get(FORMAT_KEY) {
        None => Ok(()),
        Some(f) if f.as_u64().is_some_and(|f| f <= u64::from(RECORD_FORMAT)) => Ok(()),
        Some(f) => Err(format!(
            "record format {f}, this version reads format {RECORD_FORMAT} and older"
        )),
    }
}

/// The key of [`RECORD_FORMAT`] in a saved record.
const FORMAT_KEY: &str = "format";

/// A record as [`save_file`] writes it: `format` first, then the record's own fields.
#[derive(Serialize)]
struct Stamped<'a, T> {
    format: u32,
    #[serde(flatten)]
    record: &'a T,
}

/// [`std::fs::read`], tried again for a moment on any failure but "not found".
fn read_retrying(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut tries = 1;
    loop {
        match std::fs::read(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound && tries < READ_TRIES => {
                tries += 1;
                std::thread::sleep(READ_RETRY);
            }
            done => return done,
        }
    }
}

/// `<name>.corrupt-<time>` beside `path`, in no use yet.
fn aside_path(path: &Path) -> PathBuf {
    let mut base = path.file_name().unwrap_or_default().to_os_string();
    base.push(format!(
        ".corrupt-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    ));
    let free = (0..)
        .map(|i| {
            let mut name = base.clone();
            if i > 0 {
                name.push(format!("-{i}"));
            }
            path.with_file_name(name)
        })
        .find(|p| !p.exists());
    free.unwrap_or_else(|| path.with_file_name(base))
}

/// Move the whole of `path` aside. A copy will do when it cannot be renamed (Windows: another
/// process has it open): the original can then be written over.
fn move_aside<T>(path: &Path, why: &str) -> Loaded<T> {
    let aside = aside_path(path);
    let moved = std::fs::rename(path, &aside).or_else(|e| match std::fs::copy(path, &aside) {
        Ok(_) => Ok(()),
        Err(_) => Err(e),
    });
    if let Err(e) = &moved {
        tracing::error!(
            "{} {why} and could not be moved aside to {} ({e}); job history starts empty and \
             nothing is saved over the file until a restart: check the permissions of its folder",
            path.display(),
            aside.display()
        );
    } else {
        tracing::error!(
            "{} {why}: moved to {}; job history starts empty. Nothing was deleted: restore that \
             file by hand if it can be repaired",
            path.display(),
            aside.display()
        );
    }
    Loaded {
        records: Vec::new(),
        writable: moved.is_ok(),
    }
}

/// Copy `unread`, the records of `path` this build cannot read, to a file of their own, unless an
/// earlier load did. Returns whether they are safe there.
fn keep_aside(path: &Path, unread: &[serde_json::Value], error: &str) -> bool {
    let Ok(bytes) = serde_json::to_vec_pretty(unread) else {
        return false;
    };
    let mut prefix = path.file_name().unwrap_or_default().to_os_string();
    prefix.push(".corrupt-");
    let prefix = prefix.to_string_lossy().into_owned();
    let copied = path
        .parent()
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .any(|e| std::fs::read(e.path()).is_ok_and(|b| b == bytes));
    if copied {
        return true;
    }
    let aside = aside_path(path);
    match crate::collect::layout::write_atomic(&aside, &bytes) {
        Ok(()) => {
            tracing::error!(
                "{}: {} record(s) this version cannot read ({error}), most likely written by a \
                 later version; kept in {}, the others loaded",
                path.display(),
                unread.len(),
                aside.display()
            );
            true
        }
        Err(e) => {
            tracing::error!(
                "{}: {} record(s) this version cannot read ({error}) could not be copied to {} \
                 ({e}); nothing is saved over the file until a restart: check the disk and the \
                 permissions of its folder",
                path.display(),
                unread.len(),
                aside.display()
            );
            false
        }
    }
}

/// Write `records` to `path` as a JSON list, each stamped with its `format` ([`RECORD_FORMAT`]):
/// through a temporary file, on the disk before the rename, so a crash leaves the old file or the
/// new one. A failure only warns.
pub fn save_file<T: Serialize>(path: &Path, records: &[T]) {
    let tmp = path.with_extension("json.tmp");
    let stamped: Vec<Stamped<'_, T>> = records
        .iter()
        .map(|record| Stamped {
            format: RECORD_FORMAT,
            record,
        })
        .collect();
    let r = serde_json::to_vec_pretty(&stamped)
        .map_err(anyhow::Error::from)
        .and_then(|b| {
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p)?;
            }
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&b)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, path)?;
            Ok(())
        });
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!("saving {}: {e:#}", path.display());
    }
}

/// [`load_or_move_aside`]; busy records become interrupted failures.
pub fn open_records(path: &Path) -> Loaded<TaskRecord> {
    let mut loaded = load_or_move_aside::<TaskRecord>(path);
    for r in &mut loaded.records {
        if r.state.is_busy() {
            r.state = RunState::Failed;
            r.error = Some(
                "interrupted: the service stopped while this job was queued or running; start \
                 it again"
                    .into(),
            );
        }
    }
    loaded
}

/// [`open_records`], for a reader that does not save.
pub fn load_records(path: &Path) -> Vec<TaskRecord> {
    open_records(path).records
}

/// Save `records` to `path` ([`save_file`]).
pub fn save_records(path: &Path, records: &[TaskRecord]) {
    save_file(path, records);
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
            after: None,
            uses: Vec::new(),
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
    fn the_newest_finished_tasks_are_kept_and_busy_ones_always() {
        let mut tasks: Vec<TaskRecord> = Vec::new();
        // A queued and a running task among the oldest, then 2 more finished than the cap.
        for i in 0..MAX_FINISHED_TASKS + 4 {
            let mut r = record(&format!("t{i}"));
            r.state = match i {
                1 => RunState::Queued,
                2 => RunState::Running,
                3 => RunState::Failed,
                _ => RunState::Done,
            };
            tasks.push(r);
        }
        cap_finished(&mut tasks);
        let ids: Vec<&str> = tasks.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(tasks.len(), MAX_FINISHED_TASKS + 2);
        assert_eq!(
            ids[..3],
            ["t1", "t2", "t4"],
            "t0 and t3 were the oldest finished"
        );
        assert_eq!(
            tasks.iter().filter(|r| !r.state.is_busy()).count(),
            MAX_FINISHED_TASKS
        );
        assert_eq!(*ids.last().unwrap(), format!("t{}", MAX_FINISHED_TASKS + 3));
        // Under the cap nothing goes, busy or not.
        let before = tasks.clone();
        cap_finished(&mut tasks);
        assert_eq!(tasks, before);
        let mut busy: Vec<TaskRecord> = (0..MAX_FINISHED_TASKS + 5)
            .map(|i| {
                let mut r = record(&format!("q{i}"));
                r.state = RunState::Queued;
                r
            })
            .collect();
        cap_finished(&mut busy);
        assert_eq!(busy.len(), MAX_FINISHED_TASKS + 5);
    }

    #[test]
    fn a_tasks_json_from_before_uses_still_parses() {
        let old = r#"[{"id":"a","kind":"autolabel","title":"t","created":"c","state":"done",
            "step":1,"steps":1,"percent":100}]"#;
        let r: Vec<TaskRecord> = serde_json::from_str(old).unwrap();
        assert!(r[0].uses.is_empty());
        let json = serde_json::to_string(&r[0]).unwrap();
        assert!(!json.contains("uses"), "{json}");
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
            after: None,
            uses: Vec::new(),
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

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-records-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn record(id: &str) -> TaskRecord {
        TaskRecord {
            id: id.into(),
            kind: TaskKind::Import,
            title: "import".into(),
            created: String::new(),
            state: RunState::Done,
            stage: None,
            step: 1,
            steps: 1,
            percent: 100,
            detail: String::new(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
            after: None,
            uses: Vec::new(),
        }
    }

    /// The `<name>.corrupt-<time>` files beside `path`, oldest name first.
    fn asides(path: &Path) -> Vec<PathBuf> {
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let mut v: Vec<PathBuf> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with(&format!("{name}.corrupt-"))
            })
            .collect();
        v.sort();
        v
    }

    #[test]
    fn a_missing_file_is_an_empty_history() {
        let dir = scratch("missing");
        let p = dir.join(TASKS_FILE);
        assert!(load_records(&p).is_empty());
        assert!(asides(&p).is_empty());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_file_goes_aside_and_the_next_save_cannot_touch_it() {
        // Cut by a crash, emptied by one, or not the list this version writes.
        for bytes in [&b"[{\"id\":\"cut"[..], b"", b"{\"id\": \"x\"}", b"\xff\xfe"] {
            let dir = scratch("corrupt");
            let p = dir.join(TASKS_FILE);
            std::fs::write(&p, bytes).unwrap();
            assert!(load_records(&p).is_empty());
            assert!(!p.exists(), "moved, not copied");
            let aside = asides(&p);
            assert_eq!(aside.len(), 1);
            assert_eq!(std::fs::read(&aside[0]).unwrap(), bytes);
            save_records(&p, &[record("a")]);
            assert_eq!(load_records(&p).len(), 1);
            assert_eq!(std::fs::read(&aside[0]).unwrap(), bytes, "still there");
            assert_eq!(asides(&p).len(), 1);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A read that fails says nothing about the file: antivirus or a backup can hold it for a
    /// moment. It stays where it is, and nothing is saved over it.
    #[test]
    fn a_file_that_cannot_be_read_stays_and_is_not_written_over() {
        let dir = scratch("unreadable");
        let p = dir.join(TASKS_FILE);
        // Reading a folder fails on every OS, with something other than "not found".
        std::fs::create_dir(&p).unwrap();
        let l: Loaded<TaskRecord> = load_or_move_aside(&p);
        assert!(l.records.is_empty());
        assert!(!l.writable);
        assert!(p.exists());
        assert!(asides(&p).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_this_build_cannot_read_stays_aside_and_the_rest_load() {
        let dir = scratch("one-record");
        let p = dir.join(TASKS_FILE);
        // What a downgrade finds: a kind added by a later version.
        let mut newer = serde_json::to_value(record("newer")).unwrap();
        newer["kind"] = "transcode".into();
        let list = serde_json::json!([record("a"), newer, record("b")]);
        std::fs::write(&p, serde_json::to_vec(&list).unwrap()).unwrap();
        let ids = |v: &[TaskRecord]| v.iter().map(|r| r.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&load_records(&p)), ["a", "b"]);
        let aside = asides(&p);
        assert_eq!(aside.len(), 1);
        let kept: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&aside[0]).unwrap()).unwrap();
        assert_eq!(kept, [newer.clone()]);
        // Loading again (the page reads the file while no queue is running) adds no copy.
        assert_eq!(ids(&load_records(&p)), ["a", "b"]);
        assert_eq!(asides(&p).len(), 1);
        // The next save drops it from the file, not from the disk.
        save_records(&p, &load_records(&p));
        assert_eq!(ids(&load_records(&p)), ["a", "b"]);
        assert_eq!(asides(&p).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saved_records_carry_their_format_and_the_file_stays_a_bare_list() {
        let dir = scratch("format");
        let p = dir.join(TASKS_FILE);
        let mut r = record("a");
        r.uses = vec!["sam3".into()];
        save_records(&p, &[r.clone(), record("b")]);
        let text = std::fs::read_to_string(&p).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let list = v.as_array().expect("a bare JSON list");
        assert!(
            list.iter().all(|x| x[FORMAT_KEY] == RECORD_FORMAT),
            "{text}"
        );
        assert_eq!(
            text.matches("\"format\"").count(),
            2,
            "once per record: {text}"
        );
        assert!(
            serde_json::to_value(&r).unwrap().get(FORMAT_KEY).is_none(),
            "the record has no field of that name: the stamp is the only one"
        );
        // This build reads it back unchanged...
        assert_eq!(load_records(&p), [r.clone(), record("b")]);
        // ...and so does a reader that knows no `format` and ignores unknown fields, as 0.2.0
        // did (today's struct, read the old way: the whole file as one list).
        let old_way: Vec<TaskRecord> = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(old_way, [r, record("b")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_of_a_later_format_is_set_aside_unread_and_the_rest_load() {
        let dir = scratch("later-format");
        let p = dir.join(TASKS_FILE);
        let stamp = |id: &str, format: serde_json::Value| {
            let mut v = serde_json::to_value(record(id)).unwrap();
            v[FORMAT_KEY] = format;
            v
        };
        // A later format parses as a record here, and is still not read: its fields may mean
        // something else now. A field this build does not know is fine at a known format.
        let mut known = stamp("known", RECORD_FORMAT.into());
        known["later_field"] = "ignored".into();
        let later = stamp("later", (RECORD_FORMAT + 1).into());
        let odd = stamp("odd", "1".into());
        let list = serde_json::json!([
            record("old"),
            known,
            later.clone(),
            odd.clone(),
            stamp("zero", 0.into())
        ]);
        std::fs::write(&p, serde_json::to_vec(&list).unwrap()).unwrap();
        let l = load_or_move_aside::<TaskRecord>(&p);
        let ids: Vec<&str> = l.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["old", "known", "zero"]);
        assert!(l.writable, "the unread ones are safe aside");
        let aside = asides(&p);
        assert_eq!(aside.len(), 1);
        let kept: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&aside[0]).unwrap()).unwrap();
        assert_eq!(kept, [later, odd]);
        assert!(
            record_format(&stamp("x", (RECORD_FORMAT + 1).into()))
                .unwrap_err()
                .contains(&format!("this version reads format {RECORD_FORMAT}"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_corrupt_file_does_not_replace_the_first() {
        let dir = scratch("twice");
        let p = dir.join(TASKS_FILE);
        std::fs::write(&p, b"first").unwrap();
        load_records(&p);
        std::fs::write(&p, b"second").unwrap();
        load_records(&p);
        let texts: std::collections::BTreeSet<Vec<u8>> = asides(&p)
            .iter()
            .map(|a| std::fs::read(a).unwrap())
            .collect();
        assert_eq!(
            texts,
            [b"first".to_vec(), b"second".to_vec()]
                .into_iter()
                .collect()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
