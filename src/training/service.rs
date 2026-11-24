//! [`TrainingService`]: the queue of training runs. One thread runs one [`TrainJob`] at a time
//! ([`super::job::run_training`]), the rest wait; any run can be cancelled (a queued one is
//! dropped, a running one has its process tree killed). Every run's record ([`RunRecord`]: state,
//! stage, progress, log tail, dataset counts, the installed model, the benchmark) is kept in
//! memory and saved to `<training dir>/runs.json`, so the Train tab shows past runs after a
//! restart; a run that was queued or running when the process stopped is marked interrupted. A
//! damaged file is set aside, never written over ([`task::load_or_move_aside`]).
//! While idle the queue can move to another training folder ([`TrainingService::set_store`]).
//! Footage imports and auto-label runs ([`super::task::Task`]) go through the same queue and
//! thread (one GPU job at a time); their records are kept in `tasks.json`.
//!
//! One owner per folder. A queue writes its folder's files only while it holds the folder's
//! [`task::QueueLock`] (`.queue.lock`), from the moment that folder becomes its store until it
//! moves, stops or is dropped. A queue that finds the lock held by another process (the service
//! beside a `train` command) is a *guest* ([`TrainingService::foreign_owner`]): it shows that
//! folder's records as the files hold them, writes nothing and refuses every job and removal
//! until it gets the lock ([`TrainingService::refresh`]); the CLI refuses to run at all
//! ([`TrainingService::start_owner`]). Only the owner marks busy records interrupted when it opens
//! the folder (`open_records`), and a run's own busy marker (`<run dir>.busy`, held by
//! [`super::job`] for the whole run) tells a run another process still trains (left alone,
//! failed with [`STILL_RUNNING`]) from one a crash left. Readers ([`load_records`],
//! [`load_bare_records`]) never write, and show a busy record interrupted only when its owner is
//! provably gone. Remove with files and Retry export refuse a run whose marker is held
//! ([`run_in_use`]).
//!
//! What reaches the disk: a state or stage change at once, progress at most every
//! [`PROGRESS_SAVE_EVERY`] (a crash loses at most that much of the progress shown, never a state
//! change or a result), and a run's per-epoch metrics in its own `history.jsonl`, appended per
//! epoch ([`history`]); `runs.json` holds none of them. No run record is dropped (a run owns its
//! folder and model); `tasks.json` keeps the newest [`task::MAX_FINISHED_TASKS`] finished tasks.
//! A job's log tail grows only through [`push_log`], which also counts the lines in memory
//! (never in the files), so the CLI's `--verbose` ([`LogFollower`]) keeps following it once the
//! tail is full.
//!
//! The running job can be paused ([`TrainingService::pause`]): its helper processes (Python,
//! ffmpeg, run through the queue's [`CommandRunner`] wrapped in a [`PausableRunner`]) are
//! suspended where they are, keeping their memory, loaded models and GPU allocations, and no new
//! helper starts; work the service does itself between helpers (frame analysis, an in-process
//! benchmark, a download) finishes first. The queue keeps the job's place (nothing else starts)
//! until [`TrainingService::resume`] or a cancel. A running run can also finish early
//! ([`TrainingService::finish`]): the trainer stops after its current epoch and exports the best.
//!
//! Stopping. [`TrainingService::shutdown`] is the CLI's stop: the running job is cancelled as by
//! the user (a cancelled training takes its checkpoints along) and the thread joined.
//! [`TrainingService::stop_for_exit`] is the service's: the running job's process tree is killed
//! the same way, but the run is *interrupted* ([`super::job::run_training_or_exit`]): it keeps its
//! work folder and checkpoint and is recorded as a failure that says the service stopped (the row
//! offers Retry export when there is a checkpoint), exactly as [`open_records`] marks a run a crash
//! left busy; queued runs and tasks are written that way at once. The thread is waited for until a
//! deadline, then left behind, and the queue never writes its files again (a later queue of the
//! same process, after an in-process restart, owns them). A stopped queue refuses new jobs without
//! recording them; a job whose message cannot reach the thread is recorded as failed, never left
//! queued.
//!
//! Panics. A panic in a job (dataset assembly, an import, an auto-label, the progress parser) is
//! caught on the queue's thread: the job's record fails with the panic's message, `current` is
//! cleared (a drop guard, so every exit path clears it), its files are kept, and the next job runs.

use super::eta::{JobClock, JobTiming};
use super::job::{
    BenchSummary, Hooks, TrainError, TrainJob, TrainOutput, TrainReport, TrainRequest, TrainStage,
    not_in_config,
};
use super::progress::{EpochPoint, TrainProgress};
use super::task::{self, Task, TaskContext, TaskRecord, TaskReport};
use crate::model::name::same_model_name;
use crate::resources::export::{CommandRunner, Fetcher, PausableRunner};
use crate::storage::busy;
use crate::sync::LockExt;
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub mod history;

/// File of the run records in the training folder.
pub const RUNS_FILE: &str = "runs.json";
/// Log lines kept per run while it runs, and after it failed (the row's Log box shows why).
const LOG_TAIL: usize = 60;
/// Log lines a run keeps on its record once it ended Done or Cancelled ([`trim_finished_log`]):
/// more than the row's Log box shows (6), and the full text is in `<run>/train.log`. Runs are
/// never dropped, so this is what each finished record costs `runs.json` from then on.
pub const FINISHED_LOG_TAIL: usize = 15;

/// Trim a run that ended Done or Cancelled to its last [`FINISHED_LOG_TAIL`] log lines; a failed
/// or busy run keeps its [`LOG_TAIL`].
fn trim_finished_log(r: &mut RunRecord) {
    if matches!(r.state, RunState::Done | RunState::Cancelled) {
        let skip = r.log.len().saturating_sub(FINISHED_LOG_TAIL);
        r.log.drain(..skip);
    }
}
/// Characters of a child's line kept on a record.
const LOG_LINE_CHARS: usize = 300;

/// Add `line` (its first [`LOG_LINE_CHARS`] characters) to the log tail `log`, dropping the oldest
/// line once it holds [`LOG_TAIL`], and count it in `seq`: the number of lines ever pushed, which
/// [`LogFollower`] needs to tell new lines from old once the tail is full (QA-039). The one way a
/// run's or a task's log grows.
pub fn push_log(log: &mut Vec<String>, seq: &mut u64, line: &str) {
    if log.len() >= LOG_TAIL {
        let over = log.len() + 1 - LOG_TAIL;
        log.drain(..over);
    }
    log.push(line.chars().take(LOG_LINE_CHARS).collect());
    *seq = seq.saturating_add(1);
}

/// Follows a log tail that [`push_log`] keeps, for a reader that polls it (the CLI's
/// `--verbose`): each poll gives the lines pushed since the last one, each once and in order.
/// Lines pushed and evicted between two polls (more than [`LOG_TAIL`] of them) are skipped: the
/// tail no longer holds them (the full text is in the run's `train.log`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LogFollower {
    seen: u64,
}

impl LogFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// The lines of `log` not returned before, given `seq`, the push count `log` was read at.
    pub fn new_lines<'a>(&mut self, log: &'a [String], seq: u64) -> &'a [String] {
        let fresh = seq.saturating_sub(self.seen);
        self.seen = self.seen.max(seq);
        let n = usize::try_from(fresh).map_or(log.len(), |f| f.min(log.len()));
        log.get(log.len() - n..).unwrap_or_default()
    }
}

/// How often a running job's progress (percent, epoch, detail, log tail) is saved when nothing
/// else changed; state changes, stage changes and results are saved at once.
pub const PROGRESS_SAVE_EVERY: Duration = Duration::from_secs(5 * 60);

/// Whether a job's progress report is saved now: on a new stage (or step) at once, else when
/// [`PROGRESS_SAVE_EVERY`] passed since the last save.
pub fn progress_save_due(new_stage: bool, since_save: Duration) -> bool {
    new_stage || since_save >= PROGRESS_SAVE_EVERY
}

/// Where a run is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Queued,
    Running,
    /// Running, with its helper processes suspended ([`TrainingService::pause`]).
    Paused,
    Done,
    Failed,
    Cancelled,
}

impl RunState {
    /// The state as the pages and the API spell it: equal to its serde (snake_case) name.
    pub fn as_str(self) -> &'static str {
        match self {
            RunState::Queued => "queued",
            RunState::Running => "running",
            RunState::Paused => "paused",
            RunState::Done => "done",
            RunState::Failed => "failed",
            RunState::Cancelled => "cancelled",
        }
    }

    pub fn is_busy(self) -> bool {
        matches!(
            self,
            RunState::Queued | RunState::Running | RunState::Paused
        )
    }

    /// Running or paused: the job holds the queue's thread.
    pub fn is_active(self) -> bool {
        matches!(self, RunState::Running | RunState::Paused)
    }
}

/// Why [`TrainingService::enqueue_task_unless`] did not queue a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotQueued {
    /// The caller's rule refused it, in the rule's words.
    Refused(String),
    /// The queue could not take it (an id already listed, the training thread gone).
    Failed(String),
}

/// Dataset counts of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DatasetCounts {
    pub classes: Vec<String>,
    pub train_images: usize,
    /// Unique training frames (0 in records of older versions).
    #[serde(default)]
    pub train_frames: usize,
    pub valid_images: usize,
    /// [`super::dataset::Stats::valid_groups`] (empty in records of older versions).
    #[serde(default)]
    pub valid_groups: Vec<String>,
    pub heldout_images: usize,
    pub heldout_dir: Option<PathBuf>,
    pub warnings: Vec<String>,
    /// [`super::dataset::BalanceReport::summary`] when the run balanced its training frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balance_summary: Option<String>,
}

impl DatasetCounts {
    /// The counts of a prepared dataset (recorded when the job reports it).
    pub fn of(p: &super::dataset::Prepared) -> Self {
        Self {
            classes: p.classes.clone(),
            train_images: p.train_images,
            train_frames: p.train_frames,
            valid_images: p.valid_images,
            valid_groups: p.valid_groups.clone(),
            heldout_images: p.heldout_images,
            heldout_dir: p.heldout_dir.clone(),
            warnings: p.warnings.clone(),
            balance_summary: p.balance_summary.clone(),
        }
    }
}

/// One run as the Train tab and `train list` show it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: String,
    pub name: String,
    pub created: String,
    pub request: TrainRequest,
    /// The accelerator it runs on (`describe()`).
    pub device: String,
    pub state: RunState,
    pub stage: Option<TrainStage>,
    pub step: usize,
    pub steps: usize,
    pub percent: u8,
    pub detail: String,
    #[serde(default)]
    pub progress: Option<TrainProgress>,
    #[serde(default)]
    pub log: Vec<String>,
    #[serde(default)]
    pub dataset: Option<DatasetCounts>,
    #[serde(default)]
    pub output: Option<TrainOutput>,
    #[serde(default)]
    pub registered: Option<String>,
    #[serde(default)]
    pub benchmark: Option<BenchSummary>,
    #[serde(default)]
    pub benchmark_error: Option<String>,
    /// The current default model on the same frames ([`super::job::Baseline`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<super::job::Baseline>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub finished: Option<String>,
    pub run_dir: PathBuf,
    /// Queued or running again to export its checkpoint ([`TrainingService::enqueue_retry`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub retrying: bool,
    /// What its training used: data loader workers, the machine, commit and RAM
    /// ([`super::workers::RunResources`]; None before the training stage and in records from
    /// before it was recorded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<super::workers::RunResources>,
}

impl RunRecord {
    /// The metric that picks the run's best epoch: the progress' once it trains (a fall-back to
    /// mAP included), else the request's.
    pub fn select_metric(&self) -> super::recipe::SelectMetric {
        self.progress
            .as_ref()
            .filter(|p| p.settings.is_some())
            .map_or(
                self.request.recipe.select_metric,
                super::progress::TrainProgress::select_metric,
            )
    }

    fn new(job: &TrainJob) -> Self {
        Self {
            id: job.id.clone(),
            name: job.name.clone(),
            created: now(),
            request: job.request.clone(),
            device: job.accelerator.describe(),
            state: RunState::Queued,
            stage: None,
            step: 0,
            steps: TrainStage::ALL.len(),
            percent: 0,
            detail: String::new(),
            progress: None,
            log: Vec::new(),
            dataset: None,
            output: None,
            registered: None,
            benchmark: None,
            benchmark_error: None,
            baseline: None,
            error: None,
            finished: None,
            run_dir: job.run_dir.clone(),
            retrying: false,
            resources: None,
        }
    }

    /// "training (5/8): epoch 3/50, ..." / "failed while training: ...".
    pub fn describe(&self) -> String {
        match self.state {
            RunState::Queued => "queued".into(),
            RunState::Running => match self.stage {
                Some(s) if self.detail.is_empty() => {
                    format!("{} ({}/{})", s.label(), self.step, self.steps)
                }
                Some(s) => format!(
                    "{} ({}/{}): {}",
                    s.label(),
                    self.step,
                    self.steps,
                    self.detail
                ),
                None => "starting".into(),
            },
            RunState::Paused => match self.stage {
                Some(s) if self.detail.is_empty() => {
                    format!("paused ({} {}/{})", s.label(), self.step, self.steps)
                }
                Some(s) => format!(
                    "paused ({} {}/{}): {}",
                    s.label(),
                    self.step,
                    self.steps,
                    self.detail
                ),
                None => "paused".into(),
            },
            RunState::Done => format!(
                "done: {}",
                self.output
                    .as_ref()
                    .map_or_else(String::new, |o| o.onnx.display().to_string())
            ),
            RunState::Failed => format!(
                "failed{}: {}",
                self.stage
                    .map(|s| format!(" while {}", s.label()))
                    .unwrap_or_default(),
                self.error.as_deref().unwrap_or("unknown error")
            ),
            RunState::Cancelled => match self.output.as_ref() {
                Some(o) if self.registered.is_none() => {
                    format!("cancelled after the install. {}", not_in_config(o))
                }
                Some(o) => format!("cancelled after the install: {} is in the config", o.name),
                None => "cancelled".into(),
            },
        }
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// A new run id: `YYYYMMDD-HHMMSS-<4 hex>`.
pub fn new_run_id() -> String {
    let u = uuid::Uuid::new_v4().simple().to_string();
    format!(
        "{}-{}",
        crate::clock::name_stamp(std::time::SystemTime::now()),
        u.get(..4).unwrap_or("0000")
    )
}

/// What the owner's open says of a busy run whose folder another process still holds (a run of an
/// older version, or of a process outside the queue): it is no longer this queue's to follow.
pub const STILL_RUNNING: &str =
    "still running in another process when this queue opened the folder; its files are kept";

/// The owner's load ([`task::load_or_move_aside`]; the caller holds the folder's
/// [`task::QueueLock`]). Each run's busy marker ([`busy::probe`] of its folder) decides a busy
/// record: held by another process -> failed with [`STILL_RUNNING`]; a crash's -> interrupted
/// ([`interrupt`]) and the marker deleted; none -> interrupted. A crash's marker of a finished
/// record is deleted too (a panic in a run, a process that ended under a run it had left behind).
/// Also returns whether a record changed (the owner saves the file then).
fn open_records(path: &Path) -> (task::Loaded<RunRecord>, bool) {
    let mut loaded = task::load_or_move_aside::<RunRecord>(path);
    let mut changed = false;
    for r in &mut loaded.records {
        let busy_now = r.state.is_busy();
        changed |= busy_now;
        match busy::probe(&r.run_dir) {
            busy::State::Live => {
                if busy_now {
                    r.state = RunState::Failed;
                    r.error = Some(STILL_RUNNING.into());
                    r.retrying = false;
                }
            }
            busy::State::Crashed(claim) => {
                if busy_now {
                    interrupt(r);
                }
                claim.finish();
            }
            busy::State::Ended => {
                if busy_now {
                    interrupt(r);
                }
            }
        }
    }
    (loaded, changed)
}

/// Why run `r` cannot be removed with its files or retried: a training in another process (or a
/// queue thread this process left behind) holds its folder's busy marker. Checked for every record
/// whatever its stored state; a crash's marker is let go at once (the owner's next open deletes
/// it).
pub fn run_in_use(r: &RunRecord) -> Option<String> {
    matches!(busy::probe(&r.run_dir), busy::State::Live).then(|| {
        format!(
            "a training in another process is using run {} ({}) and its folder {}; wait for it \
             to finish (or stop that `train` command), then try again",
            r.name,
            r.id,
            r.run_dir.display()
        )
    })
}

/// Why a guest queue ([`TrainingService::foreign_owner`]) takes no job and removes nothing.
fn guest_refusal(dir: &Path) -> String {
    format!(
        "another blue-onyx-prism process (a `train` command) is using the training folder {}; \
         wait for it to finish, then try again",
        dir.display()
    )
}

/// Why a `train` command does not run beside a queue that owns its folder
/// ([`TrainingService::start_owner`]).
pub fn owner_refusal(dir: &Path) -> String {
    format!(
        "the service's training queue owns {}; queue it on the Train tab, or stop the service and \
         run the command again",
        dir.display()
    )
}

/// A run the service stopped under (a crash read back at the next start, or
/// [`TrainingService::stop_for_exit`]): failed, saying how far it got and what to do; its files
/// stay (Retry export when it left a checkpoint).
fn interrupt(r: &mut RunRecord) {
    if let Some(p) = r.progress.as_mut() {
        history::refresh_best(p, &r.run_dir);
    }
    r.state = RunState::Failed;
    r.error = Some(match (&r.output, &r.registered) {
        (Some(o), None) => {
            format!("interrupted after the install. {}", not_in_config(o))
        }
        (Some(o), Some(_)) => format!(
            "interrupted after {} was added to the config: the model is ready, only its \
             benchmark did not finish; run it on the Benchmark page",
            o.name
        ),
        (None, _) => "interrupted: the service stopped while this run was queued or \
                      running; use Retry export if the row offers it, else queue it again"
            .into(),
    });
    r.retrying = false;
}

/// What a job that panicked tells the user (the panic itself is in the log).
fn internal_error(panic: &str, run: bool) -> String {
    let next = if run {
        "use Retry export if the row offers it, else queue the run again"
    } else {
        "start the job again"
    };
    format!(
        "internal error: {panic}. Its files are kept. Please report this with the bug-report \
         zip (blue-onyx-prism diagnostics), then {next}"
    )
}

/// The message of a panic's payload (`panic!` with a literal or a format string).
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".into())
}

/// [`TrainingService::while_idle`]'s rule: why not, given the run and task records as they stand.
pub type IdleRule<'a> = dyn Fn(&[RunRecord], &[TaskRecord]) -> Option<String> + 'a;

/// Why a stopped queue takes no job.
const STOPPED: &str = "the training queue has stopped (the service is stopping or restarting); \
                       queue it again once the service is back";
/// Why a job whose message could not reach the queue's thread failed.
const THREAD_GONE: &str = "the training thread has stopped, so this job never ran; restart the \
                           service and queue it again";

/// [`load_bare_records`] with each run's epochs ([`history::attach`]): `train list`, the page
/// before the queue starts.
pub fn load_records(path: &Path) -> Vec<RunRecord> {
    let mut runs = load_bare_records(path);
    for r in &mut runs {
        history::attach(r, None);
    }
    runs
}

/// [`load_bare_records`] with each run's epochs shared ([`history::share`]): the page's poll
/// before the queue starts.
pub fn load_shown(path: &Path) -> Vec<Shown> {
    load_bare_records(path)
        .into_iter()
        .map(|r| share_with(r, None))
        .collect()
}

/// The run records of `path` as stored, for a reader that never saves (the Jobs list, the
/// storage card, a dataset check's plan, the Review queue's steer, a guest queue): the records
/// that parse ([`task::read_records`]: a damaged file is no records and stays where it is, nothing
/// is ever renamed, moved aside, copied or written; an older record keeps its epochs, migrated
/// only by the owner). A busy record is shown interrupted (`interrupt`, in memory) only when its
/// owner is provably gone: no queue holds the folder's lock ([`task::queue_owned`]) and no process
/// holds its run's busy marker; otherwise it keeps its state.
pub fn load_bare_records(path: &Path) -> Vec<RunRecord> {
    let mut runs = task::read_records::<RunRecord>(path);
    if runs.iter().any(|r| r.state.is_busy()) && !task::queue_owned(&task::queue_lock_file(path)) {
        for r in runs.iter_mut().filter(|r| r.state.is_busy()) {
            if !matches!(busy::probe(&r.run_dir), busy::State::Live) {
                interrupt(r);
            }
        }
    }
    runs
}

/// Folders of `training_dir` holding a run's files (`train.log` or `history.jsonl`) that no
/// record of `records` names, by folder or by run id: runs whose record an earlier version lost
/// (two processes saving `runs.json` over each other). Report only: nothing removes them.
pub fn unlisted_run_dirs(training_dir: &Path, records: &[RunRecord]) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(training_dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .filter(|p| {
            p.join(super::job::LOG_FILE).is_file() || p.join(history::HISTORY_FILE).is_file()
        })
        .filter(|p| {
            !records.iter().any(|r| {
                r.run_dir == *p
                    || p.file_name()
                        .is_some_and(|n| n == std::ffi::OsStr::new(&r.id))
            })
        })
        .collect();
    out.sort();
    out
}

/// The queue's state. Lock order (a lock is only ever taken while holding locks before it):
/// `admit` > `persist_lock` > `runs` > `tasks` > `stores`. `current` is held over others only by
/// [`TrainingService::set_paused`] (`current` > `runs` / `tasks` > `clock`, with no `persist_lock`
/// or `stores` under it); `cancelled`, `clock`, `live`, `run_log_seq` and `task_log_seq` are
/// leaves.
struct Inner {
    runner: Arc<dyn CommandRunner>,
    fetcher: Arc<dyn Fetcher>,
    hooks: Arc<dyn Hooks>,
    runs: Mutex<Vec<RunRecord>>,
    /// Waited on with `runs` only ([`TrainingService::wait`]): a condvar must always be used with
    /// the same mutex (std may panic otherwise, and does on macOS).
    runs_changed: Condvar,
    /// Waited on with `tasks` only ([`TrainingService::wait_task`]).
    tasks_changed: Condvar,
    current: Mutex<Option<Current>>,
    cancelled: Mutex<HashSet<String>>,
    /// `runs.json` and `tasks.json` beside it (None = memory only); moved by
    /// [`TrainingService::set_store`] while idle.
    stores: Mutex<Stores>,
    /// Serializes snapshot + write, so the file always ends with the latest state (and a
    /// store move never splits an update from its write).
    persist_lock: Mutex<()>,
    /// Taken by every enqueue (runs, retries, tasks) for its check and push, and by
    /// [`TrainingService::while_idle`] for its whole action: an action that needs the queue idle
    /// (removing the training environment, the SAM 3 weights) never has a job queued under it.
    /// Taken before any other lock of the queue.
    admit: Mutex<()>,
    /// No job starts any more (set by [`TrainingService::shutdown`] and
    /// [`TrainingService::stop_for_exit`]); new jobs are refused.
    stop: AtomicBool,
    /// The stop is for the process's exit: a job that ends now is interrupted, not cancelled
    /// ([`TrainingService::stop_for_exit`]). Set before the job's cancel flag.
    exiting: AtomicBool,
    tasks: Mutex<Vec<TaskRecord>>,
    /// The clock of the job that last held the queue's thread. A leaf lock: nothing is taken
    /// under it.
    clock: Mutex<Option<(String, JobClock)>>,
    /// The running run's epochs (its record holds none, [`history`]): readers get these rather
    /// than the file, which may lag or have failed. A leaf lock.
    live: Mutex<Option<(String, Arc<Vec<EpochPoint>>)>>,
    /// Log lines pushed so far per run id ([`push_log`]'s counter; memory only): what
    /// [`LogFollower`] needs once the tail is full. A leaf lock, taken under `runs`.
    run_log_seq: Mutex<HashMap<String, u64>>,
    /// The same per task id, taken under `tasks`.
    task_log_seq: Mutex<HashMap<String, u64>>,
}

/// The job on the queue's thread and its flags.
struct Current {
    id: String,
    cancel: Arc<AtomicBool>,
    /// Suspends the job's helper processes while set ([`PausableRunner`]).
    pause: Arc<AtomicBool>,
}

impl Current {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            cancel: Arc::new(AtomicBool::new(false)),
            pause: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Debug, Default)]
struct Stores {
    runs: Option<PathBuf>,
    tasks: Option<PathBuf>,
    /// This file holds damage that could not be set aside, or another process owns it (a
    /// guest): this process does not write it ([`task::Loaded::writable`]), and its records live
    /// in memory.
    hold_runs: bool,
    hold_tasks: bool,
    /// The folder's owner lock, held for as long as this is the store (None: memory only, a
    /// guest, or a folder whose lock cannot work, warned).
    _lock: Option<task::QueueLock>,
    /// Another process's queue owns the folder: its records are read, never written, and every
    /// job and removal is refused ([`guest_refusal`]).
    guest: bool,
}

/// What [`Stores::open`] read.
struct Opened {
    stores: Stores,
    runs: Vec<RunRecord>,
    tasks: Vec<TaskRecord>,
    /// The owner changed run records as it opened them (busy ones interrupted, histories of an
    /// older `runs.json` moved to their files, [`history::migrate`]): `runs.json` is to be saved
    /// once.
    runs_changed: bool,
    /// Busy task records were interrupted: `tasks.json` is to be saved once.
    tasks_changed: bool,
}

impl Stores {
    /// The stores at `runs` and the records they hold. With the folder's lock (tried for
    /// `grace`) this queue is its owner: busy records a crash left are interrupted
    /// ([`open_records`]) and histories moved out ([`history::migrate`]). Held by another queue:
    /// a guest, with the records as a reader sees them ([`load_bare_records`],
    /// [`task::load_records`]). A lock that cannot work (a read-only folder, a file system
    /// without locks) is warned and the queue owns the folder without it, as before locks.
    fn open(runs: Option<PathBuf>, grace: Duration) -> Opened {
        let Some(path) = runs else {
            return Opened {
                stores: Self::default(),
                runs: Vec::new(),
                tasks: Vec::new(),
                runs_changed: false,
                tasks_changed: false,
            };
        };
        let lock_file = task::queue_lock_file(&path);
        match task::QueueLock::try_take(&lock_file, grace) {
            Ok(Some(lock)) => Self::own(path, Some(lock)),
            Ok(None) => Self::visit(path),
            Err(e) => {
                tracing::warn!(
                    "cannot lock {} ({e}): this queue writes the training folder without the \
                     check that keeps a second blue-onyx-prism process (the service and a `train` \
                     command) from writing it too; check the permissions of the folder",
                    lock_file.display()
                );
                Self::own(path, None)
            }
        }
    }

    /// The owner's open of `runs` and `tasks.json` beside it.
    fn own(runs: PathBuf, lock: Option<task::QueueLock>) -> Opened {
        let tasks = task::tasks_file(&runs);
        let (mut run_records, interrupted) = open_records(&runs);
        let mut task_records = task::load_or_move_aside::<TaskRecord>(&tasks);
        let tasks_changed = task::interrupt_busy(&mut task_records.records) > 0;
        let migrated = history::migrate(&mut run_records.records) > 0;
        Opened {
            stores: Self {
                hold_runs: !run_records.writable,
                hold_tasks: !task_records.writable,
                runs: Some(runs),
                tasks: Some(tasks),
                _lock: lock,
                guest: false,
            },
            runs: run_records.records,
            tasks: task_records.records,
            runs_changed: interrupted || migrated,
            tasks_changed,
        }
    }

    /// A guest's open: the records as the files hold them, nothing to write.
    fn visit(runs: PathBuf) -> Opened {
        let tasks = task::tasks_file(&runs);
        Opened {
            runs: load_bare_records(&runs),
            tasks: task::load_records(&tasks),
            stores: Self {
                hold_runs: true,
                hold_tasks: true,
                runs: Some(runs),
                tasks: Some(tasks),
                _lock: None,
                guest: true,
            },
            runs_changed: false,
            tasks_changed: false,
        }
    }

    /// The training folder of a guest ([`Self::guest`]).
    fn guest_dir(&self) -> Option<PathBuf> {
        self.runs.as_deref().filter(|_| self.guest).map(|p| {
            p.parent()
                .map_or_else(|| p.to_path_buf(), Path::to_path_buf)
        })
    }

    fn runs_to_write(&self) -> Option<PathBuf> {
        self.runs.clone().filter(|_| !self.hold_runs)
    }

    fn tasks_to_write(&self) -> Option<PathBuf> {
        self.tasks.clone().filter(|_| !self.hold_tasks)
    }
}

impl Inner {
    fn update(&self, id: &str, f: impl FnOnce(&mut RunRecord)) {
        let mut runs = self.runs.lock_recover();
        if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
            f(r);
        }
        drop(runs);
        self.runs_changed.notify_all();
    }

    /// Save what an owner's open changed ([`Opened::runs_changed`], [`Opened::tasks_changed`];
    /// the caller holds `persist_lock`, or no other thread has the queue yet).
    fn save_opened(&self, runs: bool, tasks: bool) {
        if runs {
            self.write_runs();
        }
        if tasks {
            self.write_tasks();
        }
    }

    /// Wake every waiter (a store move, a stop).
    fn notify_all(&self) {
        self.runs_changed.notify_all();
        self.tasks_changed.notify_all();
    }

    /// Refused with [`guest_refusal`] while another process owns the folder.
    fn refuse_guest(&self) -> Result<(), String> {
        match self.stores.lock_recover().guest_dir() {
            Some(dir) => Err(guest_refusal(&dir)),
            None => Ok(()),
        }
    }

    fn persist(&self) {
        let _order = self.persist_lock.lock_recover();
        self.write_runs();
    }

    /// Write the runs (the caller holds `persist_lock`). The path is copied out first: `stores`
    /// is never held while `runs` is taken or the file written (the lock order on [`Inner`]).
    fn write_runs(&self) {
        let path = self.stores.lock_recover().runs_to_write();
        if let Some(p) = path {
            let runs = self.runs.lock_recover().clone();
            task::save_file(&p, &runs);
        }
    }

    /// [`Self::update`] and [`Self::persist`] as one step: a record that stops being busy is
    /// written before the store can move.
    fn update_persist(&self, id: &str, f: impl FnOnce(&mut RunRecord)) {
        let _order = self.persist_lock.lock_recover();
        self.update(id, f);
        self.write_runs();
    }

    /// A reader's copy of `r` with its epochs ([`history::attach`]).
    fn with_history(&self, r: RunRecord) -> RunRecord {
        let live = self.live.lock_recover().clone();
        attach_with(r, live.as_ref())
    }

    fn set_live(&self, id: &str, h: Option<Arc<Vec<EpochPoint>>>) {
        *self.live.lock_recover() = h.map(|h| (id.to_string(), h));
    }

    /// Start the clock of job `id`.
    fn start_clock(&self, id: &str) {
        *self.clock.lock_recover() = Some((id.to_string(), JobClock::new(Instant::now())));
    }

    fn on_clock(&self, f: impl FnOnce(&mut JobClock)) {
        if let Some((_, c)) = self.clock.lock_recover().as_mut() {
            f(c);
        }
    }

    fn update_task(&self, id: &str, f: impl FnOnce(&mut TaskRecord)) {
        let mut tasks = self.tasks.lock_recover();
        if let Some(r) = tasks.iter_mut().find(|r| r.id == id) {
            f(r);
        }
        drop(tasks);
        self.tasks_changed.notify_all();
    }

    /// Add line `l` to run `id`'s log tail ([`push_log`]).
    fn push_run_log(&self, id: &str, l: &str) {
        let mut runs = self.runs.lock_recover();
        if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
            let mut seqs = self.run_log_seq.lock_recover();
            push_log(&mut r.log, seqs.entry(id.to_string()).or_default(), l);
        }
        drop(runs);
        self.runs_changed.notify_all();
    }

    /// Add line `l` to task `id`'s log tail ([`push_log`]).
    fn push_task_log(&self, id: &str, l: &str) {
        let mut tasks = self.tasks.lock_recover();
        if let Some(r) = tasks.iter_mut().find(|r| r.id == id) {
            let mut seqs = self.task_log_seq.lock_recover();
            push_log(&mut r.log, seqs.entry(id.to_string()).or_default(), l);
        }
        drop(tasks);
        self.tasks_changed.notify_all();
    }

    fn persist_tasks(&self) {
        let _order = self.persist_lock.lock_recover();
        self.write_tasks();
    }

    /// Write the tasks (the caller holds `persist_lock`; the path copied out first, as in
    /// [`Self::write_runs`]).
    fn write_tasks(&self) {
        let path = self.stores.lock_recover().tasks_to_write();
        if let Some(p) = path {
            let tasks = self.tasks.lock_recover().clone();
            task::save_records(&p, &tasks);
        }
    }

    /// [`Self::update_task`] and [`Self::persist_tasks`] as one step.
    fn update_persist_task(&self, id: &str, f: impl FnOnce(&mut TaskRecord)) {
        let _order = self.persist_lock.lock_recover();
        self.update_task(id, f);
        self.write_tasks();
    }

    /// Mark every run and task whose state `pick` selects interrupted ([`interrupt`],
    /// [`interrupt_task`]) and write both files; returns how many.
    fn interrupt_where(&self, pick: impl Fn(RunState) -> bool) -> usize {
        let _order = self.persist_lock.lock_recover();
        let stamp = now();
        let mut n = 0;
        for r in self
            .runs
            .lock_recover()
            .iter_mut()
            .filter(|r| pick(r.state))
        {
            interrupt(r);
            r.finished.get_or_insert_with(|| stamp.clone());
            n += 1;
        }
        for r in self
            .tasks
            .lock_recover()
            .iter_mut()
            .filter(|r| pick(r.state))
        {
            task::interrupt(r);
            r.finished.get_or_insert_with(|| stamp.clone());
            n += 1;
        }
        self.notify_all();
        self.write_runs();
        self.write_tasks();
        n
    }

    /// Never write `runs.json` or `tasks.json` again, and let go of the folder's lock (a stopped
    /// queue whose thread may still end a job: the files are the next queue's, in this process
    /// after an in-process restart or in another).
    fn forget_stores(&self) {
        let _order = self.persist_lock.lock_recover();
        let old = std::mem::take(&mut *self.stores.lock_recover());
        drop(old);
        self.notify_all();
    }

    /// The stage run `id`'s record is in.
    fn stage_of(&self, id: &str) -> Option<TrainStage> {
        self.runs
            .lock_recover()
            .iter()
            .find(|r| r.id == id)
            .and_then(|r| r.stage)
    }
}

/// Clears the queue's `current` job (and the live epochs) when dropped: on the normal path, and
/// while a panic unwinds out of the job.
struct ClearCurrent<'a>(&'a Inner);

impl Drop for ClearCurrent<'_> {
    fn drop(&mut self) {
        *self.0.current.lock_recover() = None;
        // Readers go to the file from now on (or to the record, which holds the epochs itself
        // when the file could not be written).
        self.0.set_live("", None);
    }
}

/// A run record without its epochs and the epochs, shared ([`TrainingService::shown`]).
pub type Shown = (RunRecord, Arc<Vec<EpochPoint>>);

/// `r` and its epochs, shared: `live`'s when it is that run's ([`history::share`]).
fn share_with(mut r: RunRecord, live: Option<&(String, Arc<Vec<EpochPoint>>)>) -> Shown {
    let live = live.filter(|(id, _)| *id == r.id).map(|(_, h)| h);
    let h = history::share(&mut r, live);
    (r, h)
}

/// `r` with its epochs: `live`'s when it is that run's ([`history::attach`]).
fn attach_with(mut r: RunRecord, live: Option<&(String, Arc<Vec<EpochPoint>>)>) -> RunRecord {
    let live = live.filter(|(id, _)| *id == r.id).map(|(_, h)| h);
    history::attach(&mut r, live);
    r
}

enum Msg {
    Job(Box<TrainJob>),
    Task(Box<dyn Task>),
    Stop,
}

/// The training queue (see the module docs).
pub struct TrainingService {
    inner: Arc<Inner>,
    tx: Sender<Msg>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for TrainingService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingService")
            .field("store", &self.inner.stores.lock_recover().runs)
            .finish_non_exhaustive()
    }
}

impl TrainingService {
    /// Start the worker thread. `store`: the `runs.json` to load and keep (None = memory only).
    #[allow(
        clippy::expect_used,
        reason = "invariant: one more thread at first use; without it training cannot work, so \
                  failing to spawn is a controlled abort"
    )]
    pub fn start(
        runner: Arc<dyn CommandRunner>,
        fetcher: Arc<dyn Fetcher>,
        hooks: Arc<dyn Hooks>,
        store: Option<PathBuf>,
    ) -> Arc<Self> {
        let opened = Stores::open(store, crate::resources::export::train::LOCK_GRACE);
        let inner = Arc::new(Inner {
            runner,
            fetcher,
            hooks,
            runs: Mutex::new(opened.runs),
            runs_changed: Condvar::new(),
            tasks_changed: Condvar::new(),
            current: Mutex::new(None),
            clock: Mutex::new(None),
            live: Mutex::new(None),
            run_log_seq: Mutex::new(HashMap::new()),
            task_log_seq: Mutex::new(HashMap::new()),
            cancelled: Mutex::new(HashSet::new()),
            stores: Mutex::new(opened.stores),
            persist_lock: Mutex::new(()),
            admit: Mutex::new(()),
            stop: AtomicBool::new(false),
            exiting: AtomicBool::new(false),
            tasks: Mutex::new(opened.tasks),
        });
        {
            let _order = inner.persist_lock.lock_recover();
            inner.save_opened(opened.runs_changed, opened.tasks_changed);
        }
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&inner);
        let thread = std::thread::Builder::new()
            .name("model-training".into())
            .spawn(move || worker_loop(&worker, &rx))
            .expect("spawning the training thread");
        Arc::new(Self {
            inner,
            tx,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// [`Self::start`] for a command that must own `store`'s folder (the CLI's `train run`,
    /// `retry-export`, import, auto-label, overlaps and dataset check): when another process's
    /// queue owns it, the queue is shut down unwritten and the answer is [`owner_refusal`].
    pub fn start_owner(
        runner: Arc<dyn CommandRunner>,
        fetcher: Arc<dyn Fetcher>,
        hooks: Arc<dyn Hooks>,
        store: PathBuf,
    ) -> Result<Arc<Self>, String> {
        let svc = Self::start(runner, fetcher, hooks, Some(store));
        let guest = svc.inner.stores.lock_recover().guest_dir();
        match guest {
            Some(dir) => {
                svc.shutdown();
                Err(owner_refusal(&dir))
            }
            None => Ok(svc),
        }
    }

    /// Another process's queue owns this queue's folder: its records are shown as the files hold
    /// them, nothing is written, and every job and removal is refused until [`Self::refresh`]
    /// gets the folder's lock.
    pub fn foreign_owner(&self) -> bool {
        self.inner.stores.lock_recover().guest
    }

    /// A guest ([`Self::foreign_owner`]) tries once more to own its folder: with the lock it
    /// loads the folder as at start (as the owner: busy records a crash left are interrupted, the
    /// file saved); else it reads the files again, so it shows the owner's records as they stand.
    /// Nothing for an owner, a memory-only or a stopped queue.
    pub fn refresh(&self) {
        // An owner's reads never wait for a save (`persist_lock`).
        if !self.foreign_owner() {
            return;
        }
        let _order = self.inner.persist_lock.lock_recover();
        let mut runs = self.inner.runs.lock_recover();
        let mut tasks = self.inner.tasks.lock_recover();
        let mut stores = self.inner.stores.lock_recover();
        if !stores.guest || self.stopped() {
            return;
        }
        let Some(path) = stores.runs.clone() else {
            return;
        };
        let opened = Stores::open(Some(path), Duration::ZERO);
        let owner = !opened.stores.guest;
        let save = (opened.runs_changed, opened.tasks_changed);
        (*stores, *runs, *tasks) = (opened.stores, opened.runs, opened.tasks);
        drop((stores, tasks, runs));
        if owner {
            tracing::info!("the training queue owns its folder again");
            // `_order` is held: no other write can come between.
            self.inner.save_opened(save.0, save.1);
        }
        self.inner.notify_all();
    }

    /// A queue that is stopped from the start: no thread, no store, no records; every job is
    /// refused. What a server that is stopping hands a late request instead of a new queue.
    pub fn closed(
        runner: Arc<dyn CommandRunner>,
        fetcher: Arc<dyn Fetcher>,
        hooks: Arc<dyn Hooks>,
    ) -> Arc<Self> {
        let (tx, _) = crossbeam_channel::unbounded();
        Arc::new(Self {
            inner: Arc::new(Inner {
                runner,
                fetcher,
                hooks,
                runs: Mutex::new(Vec::new()),
                runs_changed: Condvar::new(),
                tasks_changed: Condvar::new(),
                current: Mutex::new(None),
                clock: Mutex::new(None),
                live: Mutex::new(None),
                run_log_seq: Mutex::new(HashMap::new()),
                task_log_seq: Mutex::new(HashMap::new()),
                cancelled: Mutex::new(HashSet::new()),
                stores: Mutex::new(Stores::default()),
                persist_lock: Mutex::new(()),
                admit: Mutex::new(()),
                stop: AtomicBool::new(true),
                exiting: AtomicBool::new(true),
                tasks: Mutex::new(Vec::new()),
            }),
            tx,
            thread: Mutex::new(None),
        })
    }

    /// The queue was stopped ([`Self::shutdown`], [`Self::stop_for_exit`], [`Self::closed`]).
    pub fn stopped(&self) -> bool {
        self.inner.stop.load(Ordering::SeqCst)
    }

    /// The `runs.json` the queue keeps (None = memory only).
    pub fn store(&self) -> Option<PathBuf> {
        self.inner.stores.lock_recover().runs.clone()
    }

    /// Keep the runs in `store` (and `tasks.json` beside it) from now on: the records shown
    /// become that folder's (loaded like at start); the old folder's files stay where they are.
    /// The new folder's lock is taken before the old one is let go. `Ok(false)` when it already
    /// is the store; refused while a run or task is queued or running (its records belong to the
    /// old folder), and when another process's queue owns the new folder (the queue stays where
    /// it is).
    pub fn set_store(&self, store: &Path) -> Result<bool, String> {
        let _order = self.inner.persist_lock.lock_recover();
        let mut runs = self.inner.runs.lock_recover();
        let mut tasks = self.inner.tasks.lock_recover();
        let mut stores = self.inner.stores.lock_recover();
        if stores.runs.as_deref() == Some(store) {
            return Ok(false);
        }
        if self.stopped() {
            return Err(STOPPED.into());
        }
        if runs.iter().any(|r| r.state.is_busy()) || tasks.iter().any(|r| r.state.is_busy()) {
            return Err(
                "a training run, import or auto-label job is queued or running; wait for it or \
                 cancel it first"
                    .into(),
            );
        }
        let opened = Stores::open(Some(store.to_path_buf()), Duration::ZERO);
        if let Some(dir) = opened.stores.guest_dir() {
            return Err(guest_refusal(&dir));
        }
        let save = (opened.runs_changed, opened.tasks_changed);
        // The old folder's lock goes with the old stores, after the new one was taken.
        (*stores, *runs, *tasks) = (opened.stores, opened.runs, opened.tasks);
        drop((stores, tasks, runs));
        // `_order` is held: no other write can come between.
        self.inner.save_opened(save.0, save.1);
        self.inner.notify_all();
        Ok(true)
    }

    /// Queue a run. Refused while another queued or running run trains the same model name.
    pub fn enqueue(&self, job: TrainJob) -> Result<String, String> {
        let id = job.id.clone();
        let _admit = self.inner.admit.lock_recover();
        {
            let mut runs = self.inner.runs.lock_recover();
            // Under the runs lock: a stop that comes later finds this record and marks it.
            if self.stopped() {
                return Err(STOPPED.into());
            }
            // Under the runs lock too: a store move cannot come between the check and the push.
            self.inner.refuse_guest()?;
            if runs
                .iter()
                .any(|r| r.state.is_busy() && same_model_name(&r.name, &job.name))
            {
                return Err(format!(
                    "a run training {} is already queued or running",
                    job.name
                ));
            }
            if runs.iter().any(|r| r.id == id) {
                return Err(format!("run {id} exists"));
            }
            // Never dropped: a run owns its folder and model, which only Remove deletes.
            runs.push(RunRecord::new(&job));
        }
        self.inner.persist();
        self.inner.runs_changed.notify_all();
        if self.tx.send(Msg::Job(Box::new(job))).is_err() {
            self.inner.update_persist(&id, |r| {
                if r.state == RunState::Queued {
                    r.state = RunState::Failed;
                    r.error = Some(THREAD_GONE.into());
                    r.finished = Some(now());
                }
            });
            return Err(THREAD_GONE.into());
        }
        Ok(id)
    }

    /// Queue the export of a failed run's checkpoint ([`super::retry::plan_retry`]) on its own
    /// record: same id, folder and history. A cancelled retry goes back to the failure it came from.
    pub fn enqueue_retry(&self, job: TrainJob) -> Result<String, String> {
        let id = job.id.clone();
        let _admit = self.inner.admit.lock_recover();
        let before;
        {
            let mut runs = self.inner.runs.lock_recover();
            if self.stopped() {
                return Err(STOPPED.into());
            }
            self.inner.refuse_guest()?;
            if runs
                .iter()
                .any(|r| r.state.is_busy() && same_model_name(&r.name, &job.name))
            {
                return Err(format!(
                    "a run training {} is already queued or running",
                    job.name
                ));
            }
            let Some(r) = runs.iter_mut().find(|r| r.id == id) else {
                return Err(format!("no run {id}"));
            };
            if r.state != RunState::Failed {
                return Err(format!("run {id} is {}, not failed", r.describe()));
            }
            if let Some(why) = run_in_use(r) {
                return Err(why);
            }
            before = (r.percent, std::mem::take(&mut r.detail), r.finished.take());
            r.state = RunState::Queued;
            r.retrying = true;
            r.percent = 0;
        }
        self.inner.persist();
        self.inner.runs_changed.notify_all();
        if self.tx.send(Msg::Job(Box::new(job))).is_err() {
            // Back to the failure it was, its checkpoint still there to retry.
            let (percent, detail, finished) = before;
            self.inner.update_persist(&id, |r| {
                if r.state == RunState::Queued {
                    r.state = RunState::Failed;
                    r.retrying = false;
                    r.percent = percent;
                    r.detail = detail;
                    r.finished = finished;
                }
            });
            return Err(THREAD_GONE.into());
        }
        Ok(id)
    }

    /// Queue an import or auto-label task (after the runs and tasks queued before it).
    pub fn enqueue_task(&self, task: Box<dyn Task>) -> Result<String, String> {
        self.enqueue_task_unless(task, &|_| None)
            .map_err(|e| match e {
                NotQueued::Refused(m) | NotQueued::Failed(m) => m,
            })
    }

    /// [`Self::enqueue_task`], unless `refuse` answers a reason for the task list as it stands.
    /// `refuse` runs under the list's own lock, so no other enqueue comes between its decision
    /// and the queueing: two starts of a job that runs one at a time (the dataset check) cannot
    /// both pass. `refuse` gets the records only; it must take no lock of this queue.
    pub fn enqueue_task_unless(
        &self,
        task: Box<dyn Task>,
        refuse: &dyn Fn(&[TaskRecord]) -> Option<String>,
    ) -> Result<String, NotQueued> {
        let id = task.id().to_string();
        let _admit = self.inner.admit.lock_recover();
        {
            let mut tasks = self.inner.tasks.lock_recover();
            if self.stopped() {
                return Err(NotQueued::Failed(STOPPED.into()));
            }
            self.inner.refuse_guest().map_err(NotQueued::Failed)?;
            if let Some(why) = refuse(&tasks) {
                return Err(NotQueued::Refused(why));
            }
            if tasks.iter().any(|r| r.id == id) {
                return Err(NotQueued::Failed(format!("job {id} exists")));
            }
            tasks.push(TaskRecord::new(task.as_ref(), now()));
            task::cap_finished(&mut tasks);
        }
        self.inner.persist_tasks();
        self.inner.tasks_changed.notify_all();
        if self.tx.send(Msg::Task(task)).is_err() {
            self.inner.update_persist_task(&id, |r| {
                if r.state == RunState::Queued {
                    r.state = RunState::Failed;
                    r.error = Some(THREAD_GONE.into());
                    r.finished = Some(now());
                }
            });
            return Err(NotQueued::Failed(THREAD_GONE.into()));
        }
        Ok(id)
    }

    /// Put `r` in the task list as it is (replacing the record of its id): page tests show
    /// busy jobs without running one.
    #[cfg(test)]
    pub(crate) fn put_task_record(&self, r: TaskRecord) {
        let mut tasks = self.inner.tasks.lock_recover();
        tasks.retain(|t| t.id != r.id);
        tasks.push(r);
        drop(tasks);
        self.inner.tasks_changed.notify_all();
    }

    /// The import and auto-label records.
    pub fn tasks(&self) -> Vec<TaskRecord> {
        self.inner.tasks.lock_recover().clone()
    }

    pub fn get_task(&self, id: &str) -> Option<TaskRecord> {
        self.inner
            .tasks
            .lock_recover()
            .iter()
            .find(|r| r.id == id)
            .cloned()
    }

    /// Block until task `id` is no longer busy or `timeout` passes.
    pub fn wait_task(&self, id: &str, timeout: Duration) -> Option<TaskRecord> {
        self.wait_task_logged(id, timeout).map(|(r, _)| r)
    }

    /// [`Self::wait_task`] with the number of log lines pushed to the task so far, read with the
    /// record (for [`LogFollower::new_lines`]).
    pub fn wait_task_logged(&self, id: &str, timeout: Duration) -> Option<(TaskRecord, u64)> {
        let deadline = Instant::now() + timeout;
        let mut tasks = self.inner.tasks.lock_recover();
        loop {
            let r = tasks.iter().find(|r| r.id == id).cloned();
            let seq = || {
                self.inner
                    .task_log_seq
                    .lock_recover()
                    .get(id)
                    .copied()
                    .unwrap_or(0)
            };
            if r.as_ref().is_none_or(|r| !r.state.is_busy()) {
                return r.map(|r| (r, seq()));
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return r.map(|r| (r, seq()));
            };
            tasks = self
                .inner
                .tasks_changed
                .wait_timeout(tasks, left.min(Duration::from_secs(1)))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Cancel a queued or running run or task; returns whether there was one. The queued tasks
    /// that follow a cancelled task ([`Task::after`]) are cancelled with it. A guest
    /// ([`Self::foreign_owner`]) cancels nothing: the jobs it shows are another process's.
    pub fn cancel(&self, id: &str) -> bool {
        if self.foreign_owner() {
            return false;
        }
        if let Some(st) = self.get_task(id).map(|r| r.state) {
            let done = match st {
                RunState::Queued => self.cancel_queued_task(id, None),
                RunState::Running | RunState::Paused => self.flag_current(id),
                _ => false,
            };
            if done {
                self.cancel_followers(id);
            }
            return done;
        }
        let Some(st) = self.get_bare(id).map(|r| r.state) else {
            return false;
        };
        match st {
            RunState::Queued => self.cancel_queued_run(id),
            RunState::Running | RunState::Paused => self.flag_current(id),
            _ => false,
        }
    }

    /// Cancel a run read as queued. The queue may have started it since: then it is the running
    /// run that must stop, and only it may change its record ([`Self::flag_current`]).
    pub(super) fn cancel_queued_run(&self, id: &str) -> bool {
        self.inner.cancelled.lock_recover().insert(id.to_string());
        let mut queued = false;
        self.inner.update_persist(id, |r| {
            if r.state == RunState::Queued {
                queued = true;
                r.state = if r.retrying {
                    RunState::Failed
                } else {
                    RunState::Cancelled
                };
                r.retrying = false;
                r.finished = Some(now());
            }
        });
        queued || self.cancel_started(id)
    }

    /// The job `id` left the queue while it was being cancelled: drop the mark that was for the
    /// queue, or a later job of that id would be skipped, and stop the job. `current` is set before
    /// a job turns Running, so a running job is found; a finished one answers false.
    fn cancel_started(&self, id: &str) -> bool {
        self.inner.cancelled.lock_recover().remove(id);
        self.flag_current(id)
    }

    /// Mark the queued task `id` Cancelled (with `summary`) and drop it when the worker gets it.
    fn cancel_queued_task(&self, id: &str, summary: Option<String>) -> bool {
        self.inner.cancelled.lock_recover().insert(id.to_string());
        let mut queued = false;
        self.inner.update_persist_task(id, |r| {
            if r.state == RunState::Queued {
                queued = true;
                r.state = RunState::Cancelled;
                r.summary = summary;
                r.finished = Some(now());
            }
        });
        queued || self.cancel_started(id)
    }

    /// Cancel the queued tasks that follow `id`, and the ones that follow those.
    fn cancel_followers(&self, id: &str) {
        let mut todo = vec![id.to_string()];
        while let Some(dep) = todo.pop() {
            let kind = self.get_task(&dep).map_or("task", |r| r.kind.as_str());
            let followers: Vec<String> = self
                .inner
                .tasks
                .lock_recover()
                .iter()
                .filter(|r| r.state == RunState::Queued && r.after.as_deref() == Some(&dep))
                .map(|r| r.id.clone())
                .collect();
            for f in followers {
                let why = format!("not run: the {kind} it follows was cancelled");
                let _ = self.cancel_queued_task(&f, Some(why));
                todo.push(f);
            }
        }
    }

    /// Set the cancel flag of the running job `id` (a paused one is resumed by its runner
    /// to be stopped).
    fn flag_current(&self, id: &str) -> bool {
        let cur = self.inner.current.lock_recover();
        match cur.as_ref() {
            Some(c) if c.id == id => {
                c.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Finish the running run `id` after its current epoch, exporting its best one: writes the
    /// stop file the trainer checks after each epoch ([`super::job::stop_file`]); a paused run is
    /// resumed first. Refused (with the reason) unless `id` is the run on the queue's thread in
    /// its training stage.
    pub fn finish(&self, id: &str) -> Result<String, String> {
        let Some(r) = self.get_bare(id) else {
            return Err(if self.get_task(id).is_some() {
                format!("{id} is an import or auto-label job; only a training run can finish early")
            } else {
                format!("no run {id}")
            });
        };
        if !r.state.is_active() {
            return Err(format!(
                "run {id} is {}; only a running run can finish early",
                r.describe()
            ));
        }
        if r.retrying {
            return Err(format!("run {id} is exporting its best checkpoint"));
        }
        if r.stage != Some(TrainStage::Train) {
            return Err(format!(
                "run {id} is not training yet ({}); cancel it instead, or wait for its training \
                 stage",
                r.stage.map_or("starting", |s| s.label())
            ));
        }
        let p = r.progress.as_ref();
        if let Some(t) = p.and_then(TrainProgress::stop_text) {
            return Err(format!("run {id} already stopped training: {t}"));
        }
        if p.is_some_and(|p| p.phase.is_some()) {
            return Err(format!("run {id} has finished training and is exporting"));
        }
        let cancelling = {
            let cur = self.inner.current.lock_recover();
            match cur.as_ref().filter(|c| c.id == id) {
                Some(c) => c.cancel.load(Ordering::Relaxed),
                None => return Err(format!("run {id} is not the running job")),
            }
        };
        if cancelling {
            return Err(format!("run {id} is being cancelled"));
        }
        let path = super::job::stop_file(&r.run_dir);
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&path, now()));
        if let Err(e) = written {
            return Err(format!(
                "cannot write the stop request {}: {e}",
                path.display()
            ));
        }
        if r.state == RunState::Paused {
            self.resume(id);
        }
        tracing::info!(run = %id, "finish requested");
        Ok(format!(
            "Finishing {id} after the current epoch; its best epoch is exported, installed and \
             benchmarked."
        ))
    }

    /// Pause the running run or task `id` (see the module docs); false when it is not running
    /// (queued, already paused, finished, unknown).
    pub fn pause(&self, id: &str) -> bool {
        self.set_paused(id, true)
    }

    /// Continue the paused run or task `id`; false when it is not paused.
    pub fn resume(&self, id: &str) -> bool {
        self.set_paused(id, false)
    }

    /// Flip the current job `id` between Running and Paused. Under the `current` lock, so the
    /// flip cannot land after the worker finished the job (it clears `current` before writing
    /// the final state, which then overwrites Paused).
    fn set_paused(&self, id: &str, paused: bool) -> bool {
        let (from, to) = if paused {
            (RunState::Running, RunState::Paused)
        } else {
            (RunState::Paused, RunState::Running)
        };
        let cur = self.inner.current.lock_recover();
        let Some(c) = cur.as_ref().filter(|c| c.id == id) else {
            return false;
        };
        if c.cancel.load(Ordering::Relaxed) {
            return false;
        }
        let mut flipped = false;
        let mut flip = |state: &mut RunState| {
            if *state == from {
                *state = to;
                flipped = true;
            }
        };
        let is_task = self.inner.tasks.lock_recover().iter().any(|r| r.id == id);
        if is_task {
            self.inner.update_task(id, |r| flip(&mut r.state));
        } else {
            self.inner.update(id, |r| flip(&mut r.state));
        }
        if flipped {
            c.pause.store(paused, Ordering::Relaxed);
            self.inner.on_clock(|c| {
                if paused {
                    c.pause(Instant::now());
                } else {
                    c.resume(Instant::now());
                }
            });
        }
        drop(cur);
        if flipped {
            tracing::info!(job = %id, "{}", if paused { "paused" } else { "resumed" });
            if is_task {
                self.inner.persist_tasks();
            } else {
                self.inner.persist();
            }
        }
        flipped
    }

    /// Every run, with its epochs ([`history::attach`]).
    pub fn runs(&self) -> Vec<RunRecord> {
        let runs = self.inner.runs.lock_recover().clone();
        let live = self.inner.live.lock_recover().clone();
        runs.into_iter()
            .map(|r| attach_with(r, live.as_ref()))
            .collect()
    }

    /// Every run as stored, without its epochs: for readers that do not show them (the Jobs list,
    /// the storage card). A record of an older `runs.json` not migrated yet still holds its own.
    pub fn records(&self) -> Vec<RunRecord> {
        self.inner.runs.lock_recover().clone()
    }

    /// Every run as stored, each with its epochs shared rather than copied ([`history::share`]:
    /// the running job's live `Arc`, else the history file's cached one, the same `Arc` while the
    /// file is unchanged): for readers that only serialize the epochs (the page's poll). The
    /// records hold no history of their own.
    pub fn shown(&self) -> Vec<Shown> {
        let runs = self.inner.runs.lock_recover().clone();
        let live = self.inner.live.lock_recover().clone();
        runs.into_iter()
            .map(|r| share_with(r, live.as_ref()))
            .collect()
    }

    /// The epochs of run `id`, shared ([`Self::shown`]); None for no such run.
    pub fn history(&self, id: &str) -> Option<Arc<Vec<EpochPoint>>> {
        let r = self.get_bare(id)?;
        let live = self.inner.live.lock_recover().clone();
        Some(share_with(r, live.as_ref()).1)
    }

    /// How long job `id` has run and how long its step should still take; None unless it is the
    /// job on the queue's thread.
    pub fn timing(&self, id: &str) -> Option<JobTiming> {
        let clock = self.inner.clock.lock_recover();
        let (_, c) = clock.as_ref().filter(|(job, _)| job == id)?;
        Some(c.timing(Instant::now()))
    }

    /// Run `id`, with its epochs ([`history::attach`]).
    pub fn get(&self, id: &str) -> Option<RunRecord> {
        self.get_bare(id).map(|r| self.inner.with_history(r))
    }

    /// Run `id` as stored, without its epochs ([`Self::records`]).
    pub fn record(&self, id: &str) -> Option<RunRecord> {
        self.get_bare(id)
    }

    /// Run `id` as stored (no epochs): for the checks that read its state.
    fn get_bare(&self, id: &str) -> Option<RunRecord> {
        self.inner
            .runs
            .lock_recover()
            .iter()
            .find(|r| r.id == id)
            .cloned()
    }

    /// Run `act` while no run or task is queued or running and none can be queued (every enqueue
    /// waits for it), unless `refuse` answers a reason for the records as they stand. Refused with
    /// a message when the queue is busy (`refuse` is asked only when it is idle) or stopped.
    /// `refuse` must take no lock of this queue. `act` runs with no lock of the queue held but
    /// `admit`: it may use the queue (`set_store`), never enqueue, and may take its time (readers
    /// of the queue are not held up, only enqueues).
    pub fn while_idle<T>(
        &self,
        refuse: &IdleRule<'_>,
        act: impl FnOnce() -> T,
    ) -> Result<T, String> {
        let _admit = self.inner.admit.lock_recover();
        if self.stopped() {
            return Err(STOPPED.into());
        }
        self.inner.refuse_guest()?;
        if self.busy() {
            return Err(
                "a training run, import or auto-label job is queued or running; wait for it or \
                 cancel it first"
                    .into(),
            );
        }
        let runs = self.inner.runs.lock_recover().clone();
        let tasks = self.inner.tasks.lock_recover().clone();
        if let Some(why) = refuse(&runs, &tasks) {
            return Err(why);
        }
        drop((runs, tasks));
        Ok(act())
    }

    /// A run or task is queued or running.
    pub fn busy(&self) -> bool {
        self.inner
            .runs
            .lock_recover()
            .iter()
            .any(|r| r.state.is_busy())
            || self
                .inner
                .tasks
                .lock_recover()
                .iter()
                .any(|r| r.state.is_busy())
    }

    /// Forget the finished run `id` and delete its folder ([`super::retry::delete_run_files`]),
    /// in that order of checks: nothing is forgotten when the files cannot go. Refused while a
    /// training in another process holds the folder ([`run_in_use`]), whatever the record says.
    /// Returns the record.
    pub fn remove_with_files(
        &self,
        id: &str,
        delete: impl FnOnce(&RunRecord) -> anyhow::Result<()>,
    ) -> Result<RunRecord, String> {
        if self.stopped() {
            return Err(STOPPED.into());
        }
        self.inner.refuse_guest()?;
        let Some(r) = self.get_bare(id) else {
            return Err(format!("no run {id}"));
        };
        if r.state.is_busy() {
            return Err("the run is queued or running; cancel it first".into());
        }
        if let Some(why) = run_in_use(&r) {
            return Err(why);
        }
        delete(&r).map_err(|e| {
            format!(
                "the files of run {id} could not be deleted: {e:#}. The run stays in the list; \
                 close what uses them and try again"
            )
        })?;
        self.remove(id)?;
        Ok(r)
    }

    /// Forget a finished run or task (its files stay). Refused once the queue stopped: its
    /// files are no longer written, so the record would come back.
    pub fn remove(&self, id: &str) -> Result<(), String> {
        if self.stopped() {
            return Err(STOPPED.into());
        }
        self.inner.refuse_guest()?;
        {
            let mut tasks = self.inner.tasks.lock_recover();
            if let Some(i) = tasks.iter().position(|r| r.id == id) {
                if tasks.get(i).is_some_and(|r| r.state.is_busy()) {
                    return Err("the job is queued or running; cancel it first".into());
                }
                tasks.remove(i);
                drop(tasks);
                self.inner.persist_tasks();
                return Ok(());
            }
        }
        let mut runs = self.inner.runs.lock_recover();
        match runs.iter().position(|r| r.id == id) {
            Some(i) if runs.get(i).is_some_and(|r| r.state.is_busy()) => {
                Err("the run is queued or running; cancel it first".into())
            }
            Some(i) => {
                let gone = runs.remove(i);
                drop(runs);
                self.inner.persist();
                history::forget(&history::history_path(&gone.run_dir));
                Ok(())
            }
            None => Err(format!("no run {id}")),
        }
    }

    /// Block until `id` is no longer busy or `timeout` passes.
    pub fn wait(&self, id: &str, timeout: Duration) -> Option<RunRecord> {
        self.wait_logged(id, timeout).map(|(r, _)| r)
    }

    /// [`Self::wait`] with the number of log lines pushed to the run so far, read with the
    /// record (for [`LogFollower::new_lines`]).
    pub fn wait_logged(&self, id: &str, timeout: Duration) -> Option<(RunRecord, u64)> {
        let deadline = Instant::now() + timeout;
        let mut runs = self.inner.runs.lock_recover();
        loop {
            let r = runs.iter().find(|r| r.id == id).cloned();
            let done = r.as_ref().is_none_or(|r| !r.state.is_busy());
            if done || Instant::now() >= deadline {
                let seq = self
                    .inner
                    .run_log_seq
                    .lock_recover()
                    .get(id)
                    .copied()
                    .unwrap_or(0);
                drop(runs);
                return r.map(|r| (self.inner.with_history(r), seq));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            runs = self
                .inner
                .runs_changed
                .wait_timeout(runs, left.min(Duration::from_secs(1)))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Stop: the running run is cancelled (its processes killed), the thread joined, and the
    /// folder let go (`Inner::forget_stores`): its records stay readable in memory, its files
    /// are the next queue's.
    pub fn shutdown(&self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        if let Some(c) = self.inner.current.lock_recover().as_ref() {
            c.cancel.store(true, Ordering::Relaxed);
        }
        let _ = self.tx.send(Msg::Stop);
        if let Some(h) = self.thread.lock_recover().take() {
            let _ = h.join();
        }
        self.inner.forget_stores();
    }

    /// Stop for the process's exit (see the module docs): queued jobs are written as interrupted,
    /// the running one has its process tree killed and is recorded as interrupted with its files
    /// kept, and the thread is waited for until `deadline`. Past it the thread is left behind
    /// (its record is written as interrupted now) and the queue writes nothing again. Returns
    /// whether the thread ended in time. Idempotent.
    pub fn stop_for_exit(&self, deadline: Instant) -> bool {
        // `exiting` before the cancel flag: the job that sees the cancel sees why.
        self.inner.exiting.store(true, Ordering::SeqCst);
        self.inner.stop.store(true, Ordering::SeqCst);
        let queued = self.inner.interrupt_where(|s| s == RunState::Queued);
        let running = self.inner.current.lock_recover().as_ref().map(|c| {
            c.cancel.store(true, Ordering::SeqCst);
            c.id.clone()
        });
        let _ = self.tx.send(Msg::Stop);
        let ended = self.wait_thread(deadline);
        if !ended {
            tracing::warn!(
                job = running.as_deref().unwrap_or("-"),
                "the training queue did not stop in time; its job is recorded as interrupted \
                 and ends with the process"
            );
            self.inner.interrupt_where(RunState::is_busy);
        }
        self.inner.forget_stores();
        if running.is_some() || queued > 0 {
            tracing::info!(
                job = running.as_deref().unwrap_or("-"),
                queued,
                "training queue stopped for exit; interrupted jobs keep their files"
            );
        }
        ended
    }

    /// Join the queue's thread if it ends by `deadline`; else let it go (a join would block).
    fn wait_thread(&self, deadline: Instant) -> bool {
        let Some(h) = self.thread.lock_recover().take() else {
            return true;
        };
        loop {
            if h.is_finished() {
                let _ = h.join();
                return true;
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            std::thread::sleep(left.min(Duration::from_millis(10)));
        }
    }

    /// End the queue's thread without stopping the queue: what a thread that died leaves, for
    /// the tests of the enqueue paths that find no thread.
    #[cfg(test)]
    pub(crate) fn end_thread_only(&self) {
        let _ = self.tx.send(Msg::Stop);
        if let Some(h) = self.thread.lock_recover().take() {
            let _ = h.join();
        }
    }
}

impl Drop for TrainingService {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The dataset folders (`<run>/data`) of the runs of `job`'s comparison group that started before
/// it (not queued, not `job`), in queue order: where an identical dataset may already be
/// ([`super::dataset::write_reusing`] checks that it is, file by file). Empty without a group, and
/// for an export retry (it writes no dataset). Pure.
pub fn group_data_dirs(records: &[RunRecord], job: &TrainJob) -> Vec<PathBuf> {
    let Some(group) = job.request.compare_group.as_deref() else {
        return Vec::new();
    };
    if job.export_only.is_some() {
        return Vec::new();
    }
    records
        .iter()
        .filter(|r| {
            r.id != job.id
                && r.state != RunState::Queued
                && r.request.compare_group.as_deref() == Some(group)
                && r.run_dir != job.run_dir
        })
        .map(|r| r.run_dir.join("data"))
        .collect()
}

fn worker_loop(inner: &Inner, rx: &Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        if inner.stop.load(Ordering::SeqCst) {
            return;
        }
        let (id, is_task, finished) = match &msg {
            Msg::Stop => return,
            Msg::Job(j) => (j.id.clone(), false, "training run finished"),
            Msg::Task(t) => (t.id().to_string(), true, finished_reason(t.kind())),
        };
        // A panic anywhere in a job ends that job, never the queue.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match msg {
            Msg::Job(j) => run_job(inner, *j),
            Msg::Task(t) => {
                if !inner.cancelled.lock_recover().remove(t.id()) {
                    run_task(inner, t.as_ref());
                }
            }
            Msg::Stop => {}
        }));
        if let Err(p) = outcome {
            contain_panic(inner, &id, is_task, &panic_text(p.as_ref()));
        }
        // The job and what it held are dropped: give the freed heap back (audit 2026-11, phase
        // 5; coalesced, off this thread).
        crate::backend::memory::trim_soon(finished);
        if inner.stop.load(Ordering::SeqCst) {
            return;
        }
    }
}

/// The trim reason ([`crate::backend::memory::trim_soon`]) of a finished task of `kind`.
fn finished_reason(kind: task::TaskKind) -> &'static str {
    match kind {
        task::TaskKind::Import => "import finished",
        task::TaskKind::Autolabel => "auto-label finished",
        task::TaskKind::Overlaps => "overlaps finished",
        task::TaskKind::Refine => "dataset check finished",
    }
}

/// A job's panic that got past its own catch: its record, still busy, fails.
fn contain_panic(inner: &Inner, id: &str, is_task: bool, panic: &str) {
    tracing::error!(job = %id, "the training queue caught a panic ({panic}); the job failed and the queue goes on");
    let error = internal_error(panic, !is_task);
    if is_task {
        inner.update_persist_task(id, |r| {
            if r.state.is_busy() {
                r.state = RunState::Failed;
                r.error = Some(error);
                r.finished = Some(now());
            }
        });
    } else {
        inner.update_persist(id, |r| {
            if r.state.is_busy() {
                r.state = RunState::Failed;
                r.error = Some(error);
                r.retrying = false;
                r.finished = Some(now());
            }
        });
    }
}

/// Run one training run on the queue's thread (see [`worker_loop`]).
fn run_job(inner: &Inner, mut job: TrainJob) {
    if inner.cancelled.lock_recover().remove(&job.id) {
        return;
    }
    let id = job.id.clone();
    let cur = Current::new(&id);
    let (flag, pause) = (cur.cancel.clone(), cur.pause.clone());
    *inner.current.lock_recover() = Some(cur);
    let clear = ClearCurrent(inner);
    // Queued -> Running under the runs lock: a cancel that marked it Cancelled after the
    // `cancelled` check above wins (else Running would overwrite it and the cancel be lost).
    let mut start = false;
    inner.update(&id, |r| {
        if r.state == RunState::Queued {
            r.state = RunState::Running;
            start = true;
        }
    });
    if !start {
        drop(clear);
        inner.cancelled.lock_recover().remove(&id);
        return;
    }
    inner.persist();
    job.reuse_from = group_data_dirs(&inner.runs.lock_recover(), &job);
    inner.start_clock(&id);
    // What an automatic worker count learns from: the runs that recorded their resources.
    if job.workers == super::job::Workers::Auto && job.past_runs.is_empty() {
        job.past_runs = inner
            .runs
            .lock_recover()
            .iter()
            .filter(|r| r.resources.is_some() && r.id != id)
            .cloned()
            .collect();
    }
    tracing::info!(run = %id, model = %job.name, "training started: {} on {}", job.request.recipe.label(), job.accelerator.describe());
    let started = Instant::now();
    let mut last_stage = None;
    let mut last_save = Instant::now();
    // An export retry trains nothing: it never touches the history its run already has.
    let mut epochs = job
        .export_only
        .is_none()
        .then(|| history::HistoryLog::new(&job.run_dir));
    let mut report = |p: TrainReport<'_>| match p {
        TrainReport::Stage {
            stage,
            percent,
            detail,
        } => {
            let new_stage = last_stage != Some(stage);
            if new_stage {
                tracing::info!(run = %id, "training stage {}/{}: {}", stage.number(), TrainStage::ALL.len(), stage.label());
                last_stage = Some(stage);
            }
            inner.on_clock(|c| c.report(Instant::now(), stage.number(), percent));
            inner.update(&id, |r| {
                r.stage = Some(stage);
                r.step = stage.number();
                r.percent = stage.overall(percent);
                r.detail = detail.chars().take(300).collect();
            });
            if progress_save_due(new_stage, last_save.elapsed()) {
                inner.persist();
                last_save = Instant::now();
            }
        }
        TrainReport::Line(l) => {
            tracing::debug!(target: "blue_onyx_prism::training::child", run = %id, "{l}");
            inner.push_run_log(&id, l);
        }
        TrainReport::Progress(tp) => {
            let stored = match epochs.as_mut() {
                Some(log) => {
                    if log.sync(tp) {
                        inner.set_live(&id, Some(Arc::new(tp.history.clone())));
                    }
                    log.stored(tp)
                }
                None => tp.clone(),
            };
            inner.update(&id, |r| r.progress = Some(stored));
        }
        TrainReport::Batch(b) => {
            tracing::warn!(run = %id, "out of GPU memory: retrying at batch {b}");
            inner.set_live(&id, None);
            inner.update_persist(&id, |r| {
                r.request.recipe.batch = b;
                r.progress = None;
            });
        }
        TrainReport::Prepared(p) => {
            inner.update_persist(&id, |r| r.dataset = Some(DatasetCounts::of(p)));
        }
        TrainReport::Installed(o) => {
            inner.update_persist(&id, |r| r.output = Some(o.clone()));
        }
        TrainReport::Registered(m) => {
            inner.update_persist(&id, |r| r.registered = Some(m.to_string()));
        }
        TrainReport::Resources(x) => {
            inner.update_persist(&id, |r| r.resources = Some(x.clone()));
        }
    };
    let runner = PausableRunner {
        inner: inner.runner.as_ref(),
        pause: &pause,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::job::run_training_or_exit(
            &job,
            &runner,
            inner.fetcher.as_ref(),
            inner.hooks.as_ref(),
            &mut report,
            &flag,
            &inner.exiting,
        )
    }))
    .unwrap_or_else(|p| {
        let panic = panic_text(p.as_ref());
        tracing::error!(run = %id, "training panicked: {panic}");
        Err(TrainError {
            stage: inner.stage_of(&id),
            message: internal_error(&panic, true),
            cancelled: false,
        })
    });
    // The run ended here: a marker a panic left unlocked ([`busy`]) is a crash's no more.
    if let busy::State::Crashed(claim) = busy::probe(&job.run_dir) {
        claim.finish();
    }
    drop(clear);
    match result {
        Ok(out) => {
            tracing::info!(run = %id, model = %job.name, "training finished in {:.0} s: {}", started.elapsed().as_secs_f64(), out.output.onnx.display());
            inner.update_persist(&id, |r| {
                r.state = RunState::Done;
                r.percent = 100;
                r.output = Some(out.output);
                r.registered = Some(out.registered);
                r.benchmark = out.benchmark;
                r.benchmark_error = out.benchmark_error;
                r.baseline = out.baseline;
                r.error = None;
                r.retrying = false;
                r.finished = Some(now());
                trim_finished_log(r);
            });
        }
        // Stopped for the process's exit: interrupted, its files kept for a retry.
        Err(e) if inner.exiting.load(Ordering::SeqCst) => {
            tracing::warn!(run = %id, "training interrupted: the service is stopping");
            inner.update_persist(&id, |r| {
                r.stage = e.stage.or(r.stage);
                interrupt(r);
                r.finished = Some(now());
            });
        }
        Err(TrainError {
            cancelled: true, ..
        }) => {
            tracing::warn!(run = %id, "training cancelled");
            inner.update_persist(&id, |r| {
                // A retry goes back to its failure, unless it got as far as installing: then
                // that failure is over.
                r.state = if r.retrying && r.output.is_none() {
                    RunState::Failed
                } else {
                    RunState::Cancelled
                };
                r.retrying = false;
                r.finished = Some(now());
                trim_finished_log(r);
            });
        }
        Err(e) => {
            tracing::warn!(run = %id, "training failed{}: {}", e.stage.map(|s| format!(" while {}", s.label())).unwrap_or_default(), e.message);
            inner.update_persist(&id, |r| {
                r.state = RunState::Failed;
                r.stage = e.stage;
                r.error = Some(e.message);
                r.retrying = false;
                r.finished = Some(now());
            });
        }
    }
}

/// Run one task on the queue's thread (see [`worker_loop`]).
fn run_task(inner: &Inner, t: &dyn Task) {
    let id = t.id().to_string();
    // A task that follows another runs only when that one allows it (a record removed since
    // does not stop it).
    if let Some(dep) = t.after() {
        let rec = inner
            .tasks
            .lock_recover()
            .iter()
            .find(|r| r.id == dep)
            .cloned();
        if let Some(why) = rec.and_then(|r| t.skip_reason(&r)) {
            tracing::info!(job = %id, "{} {why}", t.kind().as_str());
            inner.cancelled.lock_recover().remove(&id);
            inner.update_persist_task(&id, |r| {
                if r.state == RunState::Queued {
                    r.state = RunState::Cancelled;
                    r.summary = Some(why);
                    r.finished = Some(now());
                }
            });
            return;
        }
    }
    let cur = Current::new(&id);
    let (flag, pause) = (cur.cancel.clone(), cur.pause.clone());
    *inner.current.lock_recover() = Some(cur);
    let clear = ClearCurrent(inner);
    // Queued -> Running under the tasks lock, so a cancel just before the start is not lost.
    let mut start = false;
    inner.update_task(&id, |r| {
        if r.state == RunState::Queued {
            r.state = RunState::Running;
            start = true;
        }
    });
    if !start {
        drop(clear);
        inner.cancelled.lock_recover().remove(&id);
        return;
    }
    inner.persist_tasks();
    inner.start_clock(&id);
    tracing::info!(job = %id, "{} started: {}", t.kind().as_str(), t.title());
    let started = Instant::now();
    let mut last_step = 0usize;
    let mut last_save = Instant::now();
    let mut report = |p: TaskReport<'_>| match p {
        TaskReport::Stage {
            stage,
            step,
            steps,
            percent,
            step_percent,
            detail,
        } => {
            let new_step = step != last_step;
            if new_step {
                tracing::info!(job = %id, "stage {step}/{steps}: {stage}");
                last_step = step;
            }
            inner.on_clock(|c| c.report(Instant::now(), step, step_percent));
            inner.update_task(&id, |r| {
                r.stage = Some(stage.to_string());
                r.step = step;
                r.steps = steps;
                r.percent = percent;
                r.detail = detail.chars().take(300).collect();
            });
            if progress_save_due(new_step, last_save.elapsed()) {
                inner.persist_tasks();
                last_save = Instant::now();
            }
        }
        TaskReport::Line(l) => {
            tracing::debug!(target: "blue_onyx_prism::training::child", job = %id, "{l}");
            inner.push_task_log(&id, l);
        }
    };
    let runner = PausableRunner {
        inner: inner.runner.as_ref(),
        pause: &pause,
    };
    let ctx = TaskContext {
        runner: &runner,
        fetcher: inner.fetcher.as_ref(),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        t.run(ctx, &mut report, &flag)
    }))
    .unwrap_or_else(|p| {
        let panic = panic_text(p.as_ref());
        tracing::error!(job = %id, "{} panicked: {panic}", t.kind().as_str());
        Err(anyhow::anyhow!(internal_error(&panic, false)))
    });
    drop(clear);
    match result {
        // Stopped for the process's exit: interrupted (what it kept, it keeps).
        Ok(out) if inner.exiting.load(Ordering::SeqCst) => {
            tracing::warn!(job = %id, "{} interrupted: the service is stopping ({})", t.kind().as_str(), out.summary);
            inner.update_persist_task(&id, |r| {
                task::interrupt(r);
                r.summary = Some(out.summary);
                r.result = Some(out.result);
                r.finished = Some(now());
            });
        }
        Err(_) if inner.exiting.load(Ordering::SeqCst) => {
            tracing::warn!(job = %id, "{} interrupted: the service is stopping", t.kind().as_str());
            inner.update_persist_task(&id, |r| {
                task::interrupt(r);
                r.finished = Some(now());
            });
        }
        // Cancelled, but the task stopped cleanly and kept what it had (its summary says what).
        Ok(out) if flag.load(Ordering::Relaxed) => {
            tracing::warn!(job = %id, "{} cancelled: {}", t.kind().as_str(), out.summary);
            inner.update_persist_task(&id, |r| {
                r.state = RunState::Cancelled;
                r.summary = Some(out.summary);
                r.result = Some(out.result);
                r.finished = Some(now());
            });
        }
        Ok(out) => {
            tracing::info!(job = %id, "{} finished in {:.0} s: {}", t.kind().as_str(), started.elapsed().as_secs_f64(), out.summary);
            inner.update_persist_task(&id, |r| {
                r.state = RunState::Done;
                r.percent = 100;
                r.summary = Some(out.summary);
                r.result = Some(out.result);
                r.finished = Some(now());
            });
        }
        Err(e)
            if e.downcast_ref::<crate::resources::export::Cancelled>()
                .is_some()
                || flag.load(Ordering::Relaxed) =>
        {
            tracing::warn!(job = %id, "{} cancelled", t.kind().as_str());
            inner.update_persist_task(&id, |r| {
                r.state = RunState::Cancelled;
                r.finished = Some(now());
            });
        }
        Err(e) => {
            tracing::warn!(job = %id, "{} failed: {e:#}", t.kind().as_str());
            inner.update_persist_task(&id, |r| {
                r.state = RunState::Failed;
                r.error = Some(format!("{e:#}"));
                r.finished = Some(now());
            });
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod run_state_tests {
    use super::RunState;

    #[test]
    fn as_str_is_the_serde_name_of_every_state() {
        let all = [
            RunState::Queued,
            RunState::Running,
            RunState::Paused,
            RunState::Done,
            RunState::Failed,
            RunState::Cancelled,
        ];
        // An exhaustive match: a new variant breaks this build until it is listed above.
        for s in all {
            match s {
                RunState::Queued
                | RunState::Running
                | RunState::Paused
                | RunState::Done
                | RunState::Failed
                | RunState::Cancelled => {}
            }
            assert_eq!(
                serde_json::to_string(&s).unwrap(),
                format!("\"{}\"", s.as_str())
            );
        }
    }
}
