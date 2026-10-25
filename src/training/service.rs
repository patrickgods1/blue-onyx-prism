//! [`TrainingService`]: the queue of training runs. One thread runs one [`TrainJob`] at a time
//! ([`super::job::run_training`]), the rest wait; any run can be cancelled (a queued one is
//! dropped, a running one has its process tree killed). Every run's record ([`RunRecord`]: state,
//! stage, progress, log tail, dataset counts, the installed model, the benchmark) is kept in
//! memory and saved to `<training dir>/runs.json`, so the Training page shows past runs after a
//! restart; a run that was queued or running when the process stopped is marked interrupted.
//! While idle the queue can move to another training folder ([`TrainingService::set_store`]).
//! Footage imports and auto-label runs ([`super::task::Task`]) go through the same queue and
//! thread (one GPU job at a time); their records are kept in `tasks.json`.

use super::job::{
    BenchSummary, Hooks, TrainError, TrainJob, TrainOutput, TrainReport, TrainRequest, TrainStage,
};
use super::progress::TrainProgress;
use super::task::{self, Task, TaskContext, TaskRecord, TaskReport};
use crate::resources::export::{CommandRunner, Fetcher};
use crate::sync::LockExt;
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// File of the run records in the training folder.
pub const RUNS_FILE: &str = "runs.json";
/// Records kept (oldest finished ones dropped first).
pub const MAX_RECORDS: usize = 100;
/// Log lines kept per run.
const LOG_TAIL: usize = 60;

/// Where a run is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl RunState {
    pub fn is_busy(self) -> bool {
        matches!(self, RunState::Queued | RunState::Running)
    }
}

/// Dataset counts of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DatasetCounts {
    pub classes: Vec<String>,
    pub train_images: usize,
    pub valid_images: usize,
    pub heldout_images: usize,
    pub heldout_dir: Option<PathBuf>,
    pub warnings: Vec<String>,
}

/// One run as the Training page and `train list` show it.
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
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub finished: Option<String>,
    pub run_dir: PathBuf,
}

impl RunRecord {
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
            error: None,
            finished: None,
            run_dir: job.run_dir.clone(),
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
            RunState::Cancelled => "cancelled".into(),
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
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        u.get(..4).unwrap_or("0000")
    )
}

/// Load the records of `path`; busy ones become interrupted failures.
pub fn load_records(path: &Path) -> Vec<RunRecord> {
    let mut v: Vec<RunRecord> = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    for r in &mut v {
        if r.state.is_busy() {
            r.state = RunState::Failed;
            r.error = Some(
                "interrupted: the service stopped while this run was queued or running; queue \
                 it again"
                    .into(),
            );
        }
    }
    v
}

fn save_records(path: &Path, runs: &[RunRecord]) {
    let tmp = path.with_extension("json.tmp");
    let r = serde_json::to_vec_pretty(runs)
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

struct Inner {
    runner: Arc<dyn CommandRunner>,
    fetcher: Arc<dyn Fetcher>,
    hooks: Arc<dyn Hooks>,
    runs: Mutex<Vec<RunRecord>>,
    changed: Condvar,
    current: Mutex<Option<(String, Arc<AtomicBool>)>>,
    cancelled: Mutex<HashSet<String>>,
    /// `runs.json` and `tasks.json` beside it (None = memory only); moved by
    /// [`TrainingService::set_store`] while idle.
    stores: Mutex<Stores>,
    /// Serializes snapshot + write, so the file always ends with the latest state (and a
    /// store move never splits an update from its write).
    persist_lock: Mutex<()>,
    stop: AtomicBool,
    tasks: Mutex<Vec<TaskRecord>>,
}

#[derive(Debug, Clone, Default)]
struct Stores {
    runs: Option<PathBuf>,
    tasks: Option<PathBuf>,
}

impl Stores {
    fn new(runs: Option<PathBuf>) -> Self {
        Self {
            tasks: runs.as_deref().map(task::tasks_file),
            runs,
        }
    }
}

impl Inner {
    fn update(&self, id: &str, f: impl FnOnce(&mut RunRecord)) {
        let mut runs = self.runs.lock_recover();
        if let Some(r) = runs.iter_mut().find(|r| r.id == id) {
            f(r);
        }
        drop(runs);
        self.changed.notify_all();
    }

    fn persist(&self) {
        let _order = self.persist_lock.lock_recover();
        self.write_runs();
    }

    /// Write the runs (the caller holds `persist_lock`).
    fn write_runs(&self) {
        if let Some(p) = self.stores.lock_recover().runs.clone() {
            let runs = self.runs.lock_recover().clone();
            save_records(&p, &runs);
        }
    }

    /// [`Self::update`] and [`Self::persist`] as one step: a record that stops being busy is
    /// written before the store can move.
    fn update_persist(&self, id: &str, f: impl FnOnce(&mut RunRecord)) {
        let _order = self.persist_lock.lock_recover();
        self.update(id, f);
        self.write_runs();
    }

    fn update_task(&self, id: &str, f: impl FnOnce(&mut TaskRecord)) {
        let mut tasks = self.tasks.lock_recover();
        if let Some(r) = tasks.iter_mut().find(|r| r.id == id) {
            f(r);
        }
        drop(tasks);
        self.changed.notify_all();
    }

    fn persist_tasks(&self) {
        let _order = self.persist_lock.lock_recover();
        self.write_tasks();
    }

    /// Write the tasks (the caller holds `persist_lock`).
    fn write_tasks(&self) {
        if let Some(p) = self.stores.lock_recover().tasks.clone() {
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
        let stores = Stores::new(store);
        let runs = stores.runs.as_deref().map(load_records).unwrap_or_default();
        let tasks = stores
            .tasks
            .as_deref()
            .map(task::load_records)
            .unwrap_or_default();
        let inner = Arc::new(Inner {
            runner,
            fetcher,
            hooks,
            runs: Mutex::new(runs),
            changed: Condvar::new(),
            current: Mutex::new(None),
            cancelled: Mutex::new(HashSet::new()),
            stores: Mutex::new(stores),
            persist_lock: Mutex::new(()),
            stop: AtomicBool::new(false),
            tasks: Mutex::new(tasks),
        });
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

    /// The `runs.json` the queue keeps (None = memory only).
    pub fn store(&self) -> Option<PathBuf> {
        self.inner.stores.lock_recover().runs.clone()
    }

    /// Keep the runs in `store` (and `tasks.json` beside it) from now on: the records shown
    /// become that folder's (loaded like at start); the old folder's files stay where they are.
    /// `Ok(false)` when it already is the store; refused while a run or task is queued or
    /// running (its records belong to the old folder).
    pub fn set_store(&self, store: &Path) -> Result<bool, String> {
        let _order = self.inner.persist_lock.lock_recover();
        let mut runs = self.inner.runs.lock_recover();
        let mut tasks = self.inner.tasks.lock_recover();
        let mut stores = self.inner.stores.lock_recover();
        if stores.runs.as_deref() == Some(store) {
            return Ok(false);
        }
        if runs.iter().any(|r| r.state.is_busy()) || tasks.iter().any(|r| r.state.is_busy()) {
            return Err(
                "a training run, import or auto-label job is queued or running; wait for it or \
                 cancel it first"
                    .into(),
            );
        }
        *stores = Stores::new(Some(store.to_path_buf()));
        *runs = load_records(store);
        *tasks = stores
            .tasks
            .as_deref()
            .map(task::load_records)
            .unwrap_or_default();
        drop((stores, tasks, runs));
        self.inner.changed.notify_all();
        Ok(true)
    }

    /// Queue a run. Refused while another queued or running run trains the same model name.
    pub fn enqueue(&self, job: TrainJob) -> Result<String, String> {
        let id = job.id.clone();
        {
            let mut runs = self.inner.runs.lock_recover();
            if runs.iter().any(|r| r.state.is_busy() && r.name == job.name) {
                return Err(format!(
                    "a run training {} is already queued or running",
                    job.name
                ));
            }
            if runs.iter().any(|r| r.id == id) {
                return Err(format!("run {id} exists"));
            }
            runs.push(RunRecord::new(&job));
            // Drop the oldest finished records past the cap.
            while runs.len() > MAX_RECORDS {
                match runs.iter().position(|r| !r.state.is_busy()) {
                    Some(i) => {
                        runs.remove(i);
                    }
                    None => break,
                }
            }
        }
        self.inner.persist();
        self.inner.changed.notify_all();
        self.tx
            .send(Msg::Job(Box::new(job)))
            .map_err(|_| "the training thread has stopped".to_string())?;
        Ok(id)
    }

    /// Queue an import or auto-label task (after the runs and tasks queued before it).
    pub fn enqueue_task(&self, task: Box<dyn Task>) -> Result<String, String> {
        let id = task.id().to_string();
        {
            let mut tasks = self.inner.tasks.lock_recover();
            if tasks.iter().any(|r| r.id == id) {
                return Err(format!("job {id} exists"));
            }
            tasks.push(TaskRecord::new(task.as_ref(), now()));
            while tasks.len() > MAX_RECORDS {
                match tasks.iter().position(|r| !r.state.is_busy()) {
                    Some(i) => {
                        tasks.remove(i);
                    }
                    None => break,
                }
            }
        }
        self.inner.persist_tasks();
        self.inner.changed.notify_all();
        self.tx
            .send(Msg::Task(task))
            .map_err(|_| "the training thread has stopped".to_string())?;
        Ok(id)
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
        let deadline = Instant::now() + timeout;
        let mut tasks = self.inner.tasks.lock_recover();
        loop {
            let r = tasks.iter().find(|r| r.id == id).cloned();
            if r.as_ref().is_none_or(|r| !r.state.is_busy()) {
                return r;
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return r;
            };
            tasks = self
                .inner
                .changed
                .wait_timeout(tasks, left.min(Duration::from_secs(1)))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Cancel a queued or running run or task; returns whether there was one.
    pub fn cancel(&self, id: &str) -> bool {
        if let Some(st) = self.get_task(id).map(|r| r.state) {
            return match st {
                RunState::Queued => {
                    self.inner.cancelled.lock_recover().insert(id.to_string());
                    self.inner.update_persist_task(id, |r| {
                        r.state = RunState::Cancelled;
                        r.finished = Some(now());
                    });
                    true
                }
                RunState::Running => self.flag_current(id),
                _ => false,
            };
        }
        let Some(st) = self.get(id).map(|r| r.state) else {
            return false;
        };
        match st {
            RunState::Queued => {
                self.inner.cancelled.lock_recover().insert(id.to_string());
                self.inner.update_persist(id, |r| {
                    r.state = RunState::Cancelled;
                    r.finished = Some(now());
                });
                true
            }
            RunState::Running => self.flag_current(id),
            _ => false,
        }
    }

    /// Set the cancel flag of the running job `id`.
    fn flag_current(&self, id: &str) -> bool {
        let cur = self.inner.current.lock_recover();
        match cur.as_ref() {
            Some((cid, flag)) if cid == id => {
                flag.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    pub fn runs(&self) -> Vec<RunRecord> {
        self.inner.runs.lock_recover().clone()
    }

    pub fn get(&self, id: &str) -> Option<RunRecord> {
        self.inner
            .runs
            .lock_recover()
            .iter()
            .find(|r| r.id == id)
            .cloned()
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

    /// Forget a finished run or task (its files stay).
    pub fn remove(&self, id: &str) -> Result<(), String> {
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
                runs.remove(i);
                drop(runs);
                self.inner.persist();
                Ok(())
            }
            None => Err(format!("no run {id}")),
        }
    }

    /// Block until `id` is no longer busy or `timeout` passes.
    pub fn wait(&self, id: &str, timeout: Duration) -> Option<RunRecord> {
        let deadline = Instant::now() + timeout;
        let mut runs = self.inner.runs.lock_recover();
        loop {
            let r = runs.iter().find(|r| r.id == id).cloned();
            if r.as_ref().is_none_or(|r| !r.state.is_busy()) {
                return r;
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return r;
            };
            runs = self
                .inner
                .changed
                .wait_timeout(runs, left.min(Duration::from_secs(1)))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Stop: the running run is cancelled (its processes killed), the thread joined.
    pub fn shutdown(&self) {
        self.inner.stop.store(true, Ordering::Relaxed);
        if let Some((_, flag)) = self.inner.current.lock_recover().as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
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

fn worker_loop(inner: &Inner, rx: &Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        let job = match msg {
            Msg::Stop => return,
            Msg::Job(j) => *j,
            Msg::Task(t) => {
                if inner.stop.load(Ordering::Relaxed) {
                    return;
                }
                if !inner.cancelled.lock_recover().remove(t.id()) {
                    run_task(inner, t.as_ref());
                }
                if inner.stop.load(Ordering::Relaxed) {
                    return;
                }
                continue;
            }
        };
        if inner.stop.load(Ordering::Relaxed) {
            return;
        }
        if inner.cancelled.lock_recover().remove(&job.id) {
            continue;
        }
        let id = job.id.clone();
        let flag = Arc::new(AtomicBool::new(false));
        *inner.current.lock_recover() = Some((id.clone(), flag.clone()));
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
            *inner.current.lock_recover() = None;
            inner.cancelled.lock_recover().remove(&id);
            continue;
        }
        inner.persist();
        tracing::info!(run = %id, model = %job.name, "training started: {} on {}", job.request.recipe.label(), job.accelerator.describe());
        let started = Instant::now();
        let mut last_stage = None;
        let mut last_save = Instant::now();
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
                inner.update(&id, |r| {
                    r.stage = Some(stage);
                    r.step = stage.number();
                    r.percent = stage.overall(percent);
                    r.detail = detail.chars().take(300).collect();
                });
                if new_stage || last_save.elapsed() > Duration::from_secs(30) {
                    inner.persist();
                    last_save = Instant::now();
                }
            }
            TrainReport::Line(l) => {
                tracing::debug!(target: "blue_onyx_prism::training::child", run = %id, "{l}");
                inner.update(&id, |r| {
                    if r.log.len() >= LOG_TAIL {
                        r.log.remove(0);
                    }
                    r.log.push(l.chars().take(300).collect());
                });
            }
            TrainReport::Progress(tp) => inner.update(&id, |r| r.progress = Some(tp.clone())),
            TrainReport::Prepared(p) => inner.update(&id, |r| {
                r.dataset = Some(DatasetCounts {
                    classes: p.classes.clone(),
                    train_images: p.train_images,
                    valid_images: p.valid_images,
                    heldout_images: p.heldout_images,
                    heldout_dir: p.heldout_dir.clone(),
                    warnings: p.warnings.clone(),
                });
            }),
        };
        let result = super::job::run_training(
            &job,
            inner.runner.as_ref(),
            inner.fetcher.as_ref(),
            inner.hooks.as_ref(),
            &mut report,
            &flag,
        );
        *inner.current.lock_recover() = None;
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
                    r.finished = Some(now());
                });
            }
            Err(TrainError {
                cancelled: true, ..
            }) => {
                tracing::warn!(run = %id, "training cancelled");
                inner.update_persist(&id, |r| {
                    r.state = RunState::Cancelled;
                    r.finished = Some(now());
                });
            }
            Err(e) => {
                tracing::warn!(run = %id, "training failed{}: {}", e.stage.map(|s| format!(" while {}", s.label())).unwrap_or_default(), e.message);
                inner.update_persist(&id, |r| {
                    r.state = RunState::Failed;
                    r.stage = e.stage;
                    r.error = Some(e.message);
                    r.finished = Some(now());
                });
            }
        }
        if inner.stop.load(Ordering::Relaxed) {
            return;
        }
    }
}

/// Run one task on the queue's thread (see [`worker_loop`]).
fn run_task(inner: &Inner, t: &dyn Task) {
    let id = t.id().to_string();
    let flag = Arc::new(AtomicBool::new(false));
    *inner.current.lock_recover() = Some((id.clone(), flag.clone()));
    // Queued -> Running under the tasks lock, so a cancel just before the start is not lost.
    let mut start = false;
    inner.update_task(&id, |r| {
        if r.state == RunState::Queued {
            r.state = RunState::Running;
            start = true;
        }
    });
    if !start {
        *inner.current.lock_recover() = None;
        inner.cancelled.lock_recover().remove(&id);
        return;
    }
    inner.persist_tasks();
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
            detail,
        } => {
            let new_step = step != last_step;
            if new_step {
                tracing::info!(job = %id, "stage {step}/{steps}: {stage}");
                last_step = step;
            }
            inner.update_task(&id, |r| {
                r.stage = Some(stage.to_string());
                r.step = step;
                r.steps = steps;
                r.percent = percent;
                r.detail = detail.chars().take(300).collect();
            });
            if new_step || last_save.elapsed() > Duration::from_secs(30) {
                inner.persist_tasks();
                last_save = Instant::now();
            }
        }
        TaskReport::Line(l) => {
            tracing::debug!(target: "blue_onyx_prism::training::child", job = %id, "{l}");
            inner.update_task(&id, |r| {
                if r.log.len() >= LOG_TAIL {
                    r.log.remove(0);
                }
                r.log.push(l.chars().take(300).collect());
            });
        }
    };
    let ctx = TaskContext {
        runner: inner.runner.as_ref(),
        fetcher: inner.fetcher.as_ref(),
    };
    let result = t.run(ctx, &mut report, &flag);
    *inner.current.lock_recover() = None;
    match result {
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
