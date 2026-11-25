//! The Import, Auto-label, Train and Models tabs (`/train/...`, admin only, docs/TRAINING.md): what this machine can train
//! (capability, environment state), the dataset picker (review export folders), the family /
//! size picker with estimates and hardware fit, the run queue with live progress
//! (`TrainArea.follow` in `assets/train-area.js` polls `GET /v1/training`), finished runs with their benchmark and the
//! "Use as default / night model / verify model" actions (written with [`save_config`]).
//!
//! The queue ([`TrainingService`]) is one per process at a time, started on first use. It acts on
//! the run (`runner`'s one `run_server` call) that started it: its downloads go through that run's
//! download manager, looked up per download, and its [`Hooks`] add a finished model to the config
//! through [`save_config`] (the `add-model` code) and benchmark it on its held-out frames with the
//! generation's benchmark runner, both on the generation current at that moment. When the run
//! stops, [`stop_for_exit`] interrupts the job (its checkpoint kept), shuts the queue down and
//! empties its slot: nothing of the finished run is kept, and the next run of the same process (an
//! in-process restart of the Windows service) starts a fresh queue on its own state.
//!
//! The queue owns its training folder ([`crate::training::task::QueueLock`]). While a `train`
//! command of another process owns it, the queue is a guest ([`TrainingService::foreign_owner`]):
//! the page shows that process's records, read afresh at each call, every job and removal is
//! refused with a message naming the folder, and each call tries the lock again
//! ([`TrainingService::refresh`]).
//!
//! The module is split by concern (QA-134): this file holds the process's queue slot and the
//! record readers every other route shares; `hooks` what the queue acts on (its [`Hooks`], the
//! run it follows, its downloads); `view` the page (`TrainingTemplate`, built from pure
//! stages); `fields` the queue form's readers; `routes` the `/train/*`, `/training/*` and
//! `/v1/training*` handlers; `poll` the page tick's light answer.
//!
//! [`Hooks`]: crate::training::job::Hooks
//! [`save_config`]: super::save_config

use super::{AppState, Shared};
use crate::sync::LockExt;
use crate::training::service::{RunRecord, TrainingService};
use hooks::{RunFetcher, ServerHooks, Target};
use std::sync::Arc;

mod fields;
mod hooks;
mod poll;
mod routes;
#[cfg(test)]
pub(super) mod tests;
mod view;

pub(super) use routes::{
    autolabel_tab, balance_preview, cancel, compare_json, dataset_classes, datasets, finish,
    import_tab, json, metric_columns, models_tab, pause, queue, remove, remove_env, resume,
    retry_export, set_dir, train_tab, use_model,
};
pub(super) use view::flash;

/// Built-in benchmark sets offered besides the held-out frames.
const EXTRA_SETS: [&str; 3] = ["nir-night", "exdark-ir", "exdark-night"];

/// The process's training queue: started on first use, emptied by [`stop_for_exit`].
static QUEUE: QueueSlot = QueueSlot::new();

/// Serializes the tests that queue jobs or move the process's queue ([`QUEUE`]).
#[cfg(test)]
pub(super) static QUEUE_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// [`QUEUE_TEST_LOCK`] with the queue idle, then moved to `state`'s training folder, for a test
/// that asserts what the queue holds: [`QUEUE`] is one for the process, so it holds what the
/// last job left whatever folder the test configured, and only [`service`] moves it.
#[cfg(test)]
pub(super) async fn own_queue(state: &Arc<AppState>) -> tokio::sync::MutexGuard<'static, ()> {
    let queue = QUEUE_TEST_LOCK.lock().await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while QUEUE.get().is_some_and(|s| s.busy()) {
        assert!(std::time::Instant::now() < deadline, "the queue stays busy");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    service(&Shared::from(state.clone()));
    queue
}

/// The process (or this run, for an in-process restart) is stopping: the running job's child tree
/// is killed with the run's retry checkpoint kept (not a user Cancel), queued jobs are written as
/// interrupted, the queue is shut down by `deadline` ([`TrainingService::stop_for_exit`]) and the
/// slot emptied, so a later run starts a fresh queue on its own state and nothing keeps this run's
/// [`AppState`] alive. Idempotent.
#[allow(
    dead_code,
    reason = "called from runner::stop through server::stop_training_for_exit (audit 2026-11 phase 1)"
)]
pub(crate) fn stop_for_exit(deadline: std::time::Instant) {
    QUEUE.stop_for_exit(deadline);
}

/// The process's training queue, started on first use. A run that is stopping gets a closed
/// queue ([`TrainingService::closed`]) that refuses every job: a late request never starts a
/// queue that would outlive its run.
pub(super) fn service(shared: &Shared) -> Arc<TrainingService> {
    let svc = QUEUE.get_or_start(shared);
    let want = crate::training::runs_file(&shared.current().config_read());
    sync_store(&svc, &want);
    svc
}

/// The started queue, if any (readers: they show the files otherwise).
pub(super) fn started_queue() -> Option<Arc<TrainingService>> {
    QUEUE.get()
}

/// A started queue and what it acts on.
struct Queue {
    svc: Arc<TrainingService>,
    target: Arc<Target>,
}

#[derive(Default)]
struct Slot {
    queue: Option<Queue>,
    /// The run whose queue was stopped last (compared by address only, never upgraded): a late
    /// request of that run gets a closed queue.
    stopped: Option<std::sync::Weak<std::sync::RwLock<Arc<AppState>>>>,
}

/// The resettable home of the process's queue ([`QUEUE`]; tests make their own).
pub(super) struct QueueSlot {
    slot: std::sync::Mutex<Slot>,
}

impl QueueSlot {
    pub(super) const fn new() -> Self {
        Self {
            slot: std::sync::Mutex::new(Slot {
                queue: None,
                stopped: None,
            }),
        }
    }

    pub(super) fn get(&self) -> Option<Arc<TrainingService>> {
        self.slot
            .lock_recover()
            .queue
            .as_ref()
            .map(|q| q.svc.clone())
    }

    /// The queue, started on `shared`'s run when there is none (a closed one while that run
    /// stops).
    pub(super) fn get_or_start(&self, shared: &Shared) -> Arc<TrainingService> {
        let mut slot = self.slot.lock_recover();
        if let Some(q) = &slot.queue {
            return q.svc.clone();
        }
        let state = shared.current();
        let target = Target::new(shared);
        let stopped_run = slot
            .stopped
            .as_ref()
            .is_some_and(|w| std::ptr::eq(w.as_ptr(), Arc::as_ptr(&shared.current)));
        if state.shutdown.is_cancelled() || stopped_run {
            target.detach();
            return TrainingService::closed(
                Arc::new(crate::resources::export::SystemRunner),
                Arc::new(RunFetcher::new(target.clone())),
                Arc::new(ServerHooks { target }),
            );
        }
        let store = crate::training::runs_file(&state.config_read());
        drop(state);
        let svc = TrainingService::start(
            Arc::new(crate::resources::export::SystemRunner),
            Arc::new(RunFetcher::new(target.clone())),
            Arc::new(ServerHooks {
                target: target.clone(),
            }),
            Some(store),
        );
        slot.queue = Some(Queue {
            svc: svc.clone(),
            target,
        });
        svc
    }

    /// [`stop_for_exit`] on this slot.
    pub(super) fn stop_for_exit(&self, deadline: std::time::Instant) {
        let q = {
            let mut slot = self.slot.lock_recover();
            let q = slot.queue.take();
            if let Some(sh) = q.as_ref().and_then(|q| q.target.shared()) {
                slot.stopped = Some(Arc::downgrade(&sh.current));
            }
            q
        };
        let Some(q) = q else { return };
        // The hooks keep the run while its job is being stopped (a register or benchmark in
        // progress ends on the cancel), then let go.
        q.svc.stop_for_exit(deadline);
        q.target.detach();
    }
}

/// Keep the started queue on the configured training folder (`training.dir` saved on this page,
/// or edited on disk: then at the next queue, import, auto-label or remove): it moves when idle
/// ([`TrainingService::set_store`]); while a job is queued or running, or while another process
/// owns the new folder, it stays and moves on a later call. A guest on the right folder tries to
/// own it again ([`TrainingService::refresh`]). Reads (the page, `GET /v1/training`) never move it.
fn sync_store(svc: &TrainingService, want: &std::path::Path) {
    if svc.store().as_deref() != Some(want) {
        if let Err(e) = svc.set_store(want) {
            tracing::debug!("training folder {} not used yet: {e}", want.display());
        }
    } else {
        svc.refresh();
    }
}

/// The started queue, its records brought up to date first when it is a guest (another process
/// owns its folder: [`TrainingService::refresh`] reads that process's records, or takes the folder
/// over once it is free).
fn started_fresh() -> Option<Arc<TrainingService>> {
    let s = started_queue()?;
    s.refresh();
    Some(s)
}

/// The import and auto-label jobs (the queue's, else `tasks.json` read as a reader:
/// [`crate::training::task::load_records`] never writes it).
pub(super) fn tasks(state: &AppState) -> Vec<crate::training::task::TaskRecord> {
    match started_fresh() {
        Some(s) => s.tasks(),
        None => crate::training::task::load_records(&crate::training::task::tasks_file(
            &crate::training::runs_file(&state.config_read()),
        )),
    }
}

/// A training run or task of this process's queue is queued or running, so it may be using the
/// GPU. A `train` command in another process is seen once this process's queue has started on
/// its folder (the queue is then a guest showing that process's records); before that it is not.
pub(crate) fn queue_busy() -> bool {
    started_fresh().is_some_and(|s| s.busy())
}

/// The clock of job `id` ([`TrainingService::timing`]).
pub(super) fn job_timing(id: &str) -> Option<crate::training::eta::JobTiming> {
    started_queue()?.timing(id)
}

/// [`runs`] without their epochs, for readers that do not show them (the Jobs list, the storage
/// card): no history file is read and no history cloned. Before the queue starts, `runs.json` as a
/// reader sees it ([`crate::training::service::load_bare_records`]: never written, a run another
/// process trains keeps its state).
pub(super) fn records(state: &AppState) -> Vec<RunRecord> {
    match started_fresh() {
        Some(s) => s.records(),
        None => crate::training::service::load_bare_records(&crate::training::runs_file(
            &state.config_read(),
        )),
    }
}

/// [`records`] for a reader that must never touch the files: the started queue's runs, else the
/// records of `runs.json` that parse ([`crate::training::task::read_records`]), the file left as it
/// is, damaged or not (the queue moves a damaged one aside when it starts).
pub(super) fn records_read_only(state: &AppState) -> Vec<RunRecord> {
    match started_queue() {
        Some(s) => s.records(),
        None => {
            crate::training::task::read_records(&crate::training::runs_file(&state.config_read()))
        }
    }
}

/// The queue if it was started (the page shows `runs.json` otherwise, as a reader:
/// [`crate::training::service::load_records`]).
pub(super) fn runs(state: &AppState) -> Vec<RunRecord> {
    match started_fresh() {
        Some(s) => s.runs(),
        None => crate::training::service::load_records(&crate::training::runs_file(
            &state.config_read(),
        )),
    }
}

/// [`runs`] with each run's epochs shared, not copied
/// ([`crate::training::service::TrainingService::shown`]): for the page's poll, which only
/// serializes them.
pub(super) fn shown(state: &AppState) -> Vec<crate::training::service::Shown> {
    match started_fresh() {
        Some(s) => s.shown(),
        None => {
            crate::training::service::load_shown(&crate::training::runs_file(&state.config_read()))
        }
    }
}
