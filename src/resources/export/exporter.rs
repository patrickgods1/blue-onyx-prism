//! [`Exporter`]: one export at a time in a background thread (the service), plus the CLI entry
//! points ([`export_now`], [`cli_fetcher`]), the CLI's signal handling (`CtrlC`) and the
//! "needs export" message.
//!
//! The thread survives a panicking export (caught, the export `Failed`) and, should it still
//! end, leaves nothing `Queued` or `Running` (QA resources-6). Each request carries a ticket, so
//! a cancelled export that is queued again runs once, and a cancel that lands while the worker
//! picks the export up is never lost (QA resources-7). [`Exporter::request_stop`] /
//! [`Exporter::shutdown_by`] stop it within the caller's deadline (the service's stop).

use super::*;
use std::collections::HashMap;

/// Called with the result after a successful export (e.g. add it to the config).
pub type OnDone = Box<dyn FnOnce(&ExportOutput) + Send>;

/// One queued export.
pub struct ExportRequest {
    pub job: ExportJob,
    pub on_done: Option<OnDone>,
}

impl std::fmt::Debug for ExportRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportRequest")
            .field("job", &self.job)
            .field("on_done", &self.on_done.is_some())
            .finish()
    }
}

/// Where an export is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExportState {
    Queued,
    Running {
        stage: Stage,
        /// 1-based stage number of [`Stage::ALL`].
        step: usize,
        steps: usize,
        /// Overall percent.
        percent: u8,
        detail: String,
    },
    Installed,
    Failed {
        stage: Option<Stage>,
        error: String,
    },
    Cancelled,
}

impl ExportState {
    pub fn is_busy(&self) -> bool {
        matches!(self, ExportState::Queued | ExportState::Running { .. })
    }

    /// "exporting: installing the pinned Python packages (3/8): downloading torch".
    pub fn describe(&self) -> String {
        match self {
            ExportState::Queued => "export queued".to_string(),
            ExportState::Running {
                stage,
                step,
                steps,
                detail,
                ..
            } => {
                if detail.is_empty() {
                    format!("exporting: {} ({step}/{steps})", stage.label())
                } else {
                    format!("exporting: {} ({step}/{steps}): {detail}", stage.label())
                }
            }
            ExportState::Installed => "exported".to_string(),
            ExportState::Failed {
                stage: Some(s),
                error,
            } => format!("export failed while {}: {error}", s.label()),
            ExportState::Failed { stage: None, error } => format!("export failed: {error}"),
            ExportState::Cancelled => "export cancelled".to_string(),
        }
    }
}

/// Shared status of one export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportStatus {
    pub id: &'static str,
    pub title: &'static str,
    #[serde(flatten)]
    pub state: ExportState,
    /// Last lines of the child processes' output.
    pub log: Vec<String>,
    /// When the export started running (None while queued): the Jobs tab's elapsed time.
    #[serde(skip)]
    pub started: Option<Instant>,
}

/// Lines of output kept per export.
const LOG_TAIL: usize = 40;

/// How long [`Exporter::shutdown`] (and the drop of the last handle) waits for the export
/// thread: a cancelled export kills its tree within about two seconds (`SystemRunner`'s TERM
/// grace) and removes its staging folder.
pub const SHUTDOWN_WAIT: Duration = Duration::from_secs(10);

/// What a queued or running export is told when the export thread is gone.
const THREAD_GONE: &str = "the export thread stopped unexpectedly (an internal error, see the \
                           log); restart the service, then export again";

struct Shared {
    runner: Arc<dyn CommandRunner>,
    fetcher: Arc<dyn Fetcher>,
    statuses: Mutex<BTreeMap<&'static str, ExportStatus>>,
    changed: Condvar,
    /// The running export's id and cancel flag. Lock order: `statuses`, then `current`.
    current: Mutex<Option<(&'static str, Arc<AtomicBool>)>>,
    /// The ticket of each model's latest request (QA resources-7), changed only while
    /// `statuses` is held (lock order: `statuses`, then `tickets`): the worker runs a queued
    /// message only while its ticket is the latest and the status is `Queued`, so a cancelled
    /// request that was queued again runs once, under the new ticket.
    tickets: Mutex<HashMap<&'static str, u64>>,
    /// The next request's ticket.
    next_ticket: std::sync::atomic::AtomicU64,
    stop: AtomicBool,
    /// The worker thread has exited (stopped or panicked): nothing more is queued.
    dead: AtomicBool,
    /// Runs on the worker between receiving a request and picking it up (tests: a cancel
    /// landing in that window).
    #[cfg(test)]
    pickup_hook: Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl Shared {
    fn update(&self, id: &'static str, f: impl FnOnce(&mut ExportStatus)) {
        let mut map = self.statuses.lock_recover();
        if let Some(s) = map.get_mut(id) {
            f(s);
        }
        drop(map);
        self.changed.notify_all();
    }

    fn set_state(&self, id: &'static str, state: ExportState) {
        self.update(id, |s| s.state = state);
    }

    /// Every busy status becomes `to(..)` (the thread is gone or stopping).
    fn end_busy(&self, to: impl Fn() -> ExportState) {
        let mut map = self.statuses.lock_recover();
        for s in map.values_mut().filter(|s| s.state.is_busy()) {
            s.state = to();
        }
        drop(map);
        self.changed.notify_all();
    }
}

enum Msg {
    Job {
        ticket: u64,
        req: Box<ExportRequest>,
    },
    Stop,
}

/// Background exporter: one thread, one export at a time, the rest queued.
pub struct Exporter {
    shared: Arc<Shared>,
    tx: Sender<Msg>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for Exporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exporter").finish_non_exhaustive()
    }
}

impl Exporter {
    #[allow(
        clippy::expect_used,
        reason = "invariant: the OS can create one more thread at startup; without it the \
                  process cannot work at all, so failing to spawn is a controlled abort"
    )]
    pub fn start(runner: Arc<dyn CommandRunner>, fetcher: Arc<dyn Fetcher>) -> Self {
        let shared = Arc::new(Shared {
            runner,
            fetcher,
            statuses: Mutex::new(BTreeMap::new()),
            changed: Condvar::new(),
            current: Mutex::new(None),
            tickets: Mutex::new(HashMap::new()),
            next_ticket: std::sync::atomic::AtomicU64::new(1),
            stop: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            #[cfg(test)]
            pickup_hook: Mutex::new(None),
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("model-export".into())
            .spawn(move || worker_loop(&worker, &rx))
            .expect("spawning the export thread (OS thread limit or out of memory)");
        Self {
            shared,
            tx,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// Queue an export. Refused while the same model is queued or exporting, and once the export
    /// thread has stopped (then nothing is left `Queued`).
    pub fn enqueue(&self, req: ExportRequest) -> Result<(), String> {
        let res = req.job.resource;
        let ticket = {
            let mut map = self.shared.statuses.lock_recover();
            if self.shared.dead.load(Ordering::SeqCst) || self.shared.stop.load(Ordering::SeqCst) {
                return Err(
                    "the export thread has stopped (the service is stopping or hit an \
                            internal error); restart the service, then export again"
                        .to_string(),
                );
            }
            if map.get(res.id).is_some_and(|s| s.state.is_busy()) {
                return Err(format!("{} is already being exported", res.title));
            }
            let ticket = self.shared.next_ticket.fetch_add(1, Ordering::Relaxed);
            map.insert(
                res.id,
                ExportStatus {
                    id: res.id,
                    title: res.title,
                    state: ExportState::Queued,
                    log: Vec::new(),
                    started: None,
                },
            );
            self.shared.tickets.lock_recover().insert(res.id, ticket);
            ticket
        };
        self.shared.changed.notify_all();
        if self
            .tx
            .send(Msg::Job {
                ticket,
                req: Box::new(req),
            })
            .is_err()
        {
            let mut map = self.shared.statuses.lock_recover();
            let latest = self.shared.tickets.lock_recover().get(res.id) == Some(&ticket);
            if let Some(s) = map.get_mut(res.id)
                && latest
            {
                s.state = ExportState::Failed {
                    stage: None,
                    error: THREAD_GONE.to_string(),
                };
            }
            drop(map);
            self.shared.changed.notify_all();
            return Err("the export thread has stopped".to_string());
        }
        Ok(())
    }

    /// Cancel a queued or running export. Returns whether there was one.
    pub fn cancel(&self, id: &str) -> bool {
        let mut map = self.shared.statuses.lock_recover();
        let Some(st) = map.values_mut().find(|s| s.id.eq_ignore_ascii_case(id)) else {
            return false;
        };
        let done = match st.state {
            ExportState::Queued => {
                // The worker skips a message whose status is no longer `Queued`.
                st.state = ExportState::Cancelled;
                true
            }
            ExportState::Running { .. } => {
                let cur = self.shared.current.lock_recover();
                match cur.as_ref() {
                    Some((cid, flag)) if cid.eq_ignore_ascii_case(id) => {
                        flag.store(true, Ordering::Relaxed);
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        };
        drop(map);
        self.shared.changed.notify_all();
        done
    }

    pub fn status(&self, id: &str) -> Option<ExportStatus> {
        let map = self.shared.statuses.lock_recover();
        map.values()
            .find(|s| s.id.eq_ignore_ascii_case(id))
            .cloned()
    }

    pub fn statuses(&self) -> Vec<ExportStatus> {
        let map = self.shared.statuses.lock_recover();
        map.values().cloned().collect()
    }

    /// An export is queued or running.
    pub fn busy(&self) -> bool {
        self.statuses().iter().any(|s| s.state.is_busy())
    }

    /// Drop a finished status (after its files were removed).
    pub fn forget(&self, id: &str) {
        let mut map = self.shared.statuses.lock_recover();
        map.retain(|k, s| !(k.eq_ignore_ascii_case(id) && !s.state.is_busy()));
    }

    /// Block until `id` is no longer busy or `timeout` passes.
    pub fn wait(&self, id: &str, timeout: Option<Duration>) -> Option<ExportStatus> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut map = self.shared.statuses.lock_recover();
        loop {
            let st = map
                .values()
                .find(|s| s.id.eq_ignore_ascii_case(id))
                .cloned();
            if st.as_ref().is_none_or(|s| !s.state.is_busy()) {
                return st;
            }
            let wait = match deadline {
                Some(d) => match d.checked_duration_since(Instant::now()) {
                    Some(left) => left,
                    None => return st,
                },
                None => Duration::from_secs(1),
            };
            map = self
                .shared
                .changed
                .wait_timeout(map, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Run `hook` on the worker between receiving each request and picking it up (tests).
    #[cfg(test)]
    pub(crate) fn set_pickup_hook(&self, hook: impl Fn() + Send + 'static) {
        *self.shared.pickup_hook.lock_recover() = Some(Box::new(hook));
    }

    /// Start the stop without waiting: no new export, every queued one `Cancelled`, the running
    /// one cancelled (its process tree killed, its staging removed as on a user cancel).
    /// Idempotent.
    pub fn request_stop(&self) {
        {
            let mut map = self.shared.statuses.lock_recover();
            self.shared.stop.store(true, Ordering::SeqCst);
            for s in map.values_mut() {
                if s.state == ExportState::Queued {
                    s.state = ExportState::Cancelled;
                }
            }
            if let Some((_, flag)) = self.shared.current.lock_recover().as_ref() {
                flag.store(true, Ordering::Relaxed);
            }
        }
        self.shared.changed.notify_all();
        let _ = self.tx.send(Msg::Stop);
    }

    /// Stop the worker (a running export is cancelled: its child process is killed) and join
    /// it for at most [`SHUTDOWN_WAIT`].
    pub fn shutdown(&self) {
        self.shutdown_by(Instant::now() + SHUTDOWN_WAIT);
    }

    /// [`Self::request_stop`], then join the worker until `deadline`; a worker still busy then
    /// (a child that does not die, a slow staging removal) is left to finish alone, logged.
    /// Returns whether the thread finished. Idempotent.
    pub fn shutdown_by(&self, deadline: Instant) -> bool {
        self.request_stop();
        let handle = self.thread.lock_recover().take();
        let Some(h) = handle else {
            return true;
        };
        while !h.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if h.is_finished() {
            let _ = h.join();
            true
        } else {
            let running = self
                .shared
                .current
                .lock_recover()
                .as_ref()
                .map(|(id, _)| *id);
            tracing::warn!(
                "the model export thread is still busy{} after the stop request; leaving it to \
                 finish in the background",
                running.map(|id| format!(" with {id}")).unwrap_or_default()
            );
            *self.thread.lock_recover() = Some(h);
            false
        }
    }
}

impl Drop for Exporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Runs when the export thread exits, normally or by a panic outside a job (QA resources-6):
/// marks the exporter dead so later enqueues fail at once, clears `current`, and ends every busy
/// status (`Cancelled` on a stop, `Failed` otherwise), so nothing waits on a dead queue.
struct ExporterExit<'a>(&'a Shared);

impl Drop for ExporterExit<'_> {
    fn drop(&mut self) {
        let shared = self.0;
        shared.dead.store(true, Ordering::SeqCst);
        *shared.current.lock_recover() = None;
        if shared.stop.load(Ordering::SeqCst) && !std::thread::panicking() {
            shared.end_busy(|| ExportState::Cancelled);
        } else {
            tracing::error!("model export: {THREAD_GONE}");
            shared.end_busy(|| ExportState::Failed {
                stage: None,
                error: THREAD_GONE.to_string(),
            });
        }
    }
}

/// The text of a panic payload.
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

fn worker_loop(shared: &Shared, rx: &Receiver<Msg>) {
    let _exit = ExporterExit(shared);
    while let Ok(msg) = rx.recv() {
        let (ticket, req) = match msg {
            Msg::Stop => return,
            Msg::Job { ticket, req } => (ticket, *req),
        };
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let id = req.job.resource.id;
        #[cfg(test)]
        if let Some(hook) = shared.pickup_hook.lock_recover().as_ref() {
            hook();
        }
        let flag = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        // Pick up under the statuses lock: a cancel either sees `Queued` before this (then the
        // status is no longer `Queued` here and the message is skipped) or `Running` with its
        // flag in `current` (then it sets the flag).
        {
            let mut map = shared.statuses.lock_recover();
            let latest = shared.tickets.lock_recover().get(id) == Some(&ticket);
            let Some(st) = map.get_mut(id) else {
                continue;
            };
            if !latest || st.state != ExportState::Queued {
                continue;
            }
            st.state = ExportState::Running {
                stage: Stage::Uv,
                step: 1,
                steps: Stage::ALL.len(),
                percent: 0,
                detail: String::new(),
            };
            st.started = Some(started);
            *shared.current.lock_recover() = Some((id, flag.clone()));
        }
        shared.changed.notify_all();
        let mut last_stage: Option<Stage> = None;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_one(shared, req, &flag, started, &mut last_stage);
        }));
        *shared.current.lock_recover() = None;
        if let Err(p) = ran {
            let what = panic_text(p.as_ref());
            tracing::error!(model = %id, "model export stopped on an internal error: {what}");
            shared.set_state(
                id,
                ExportState::Failed {
                    stage: last_stage,
                    error: format!(
                        "the export stopped on an internal error ({what}); this is a bug, \
                         please report it with the log. Export again to retry"
                    ),
                },
            );
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
    }
}

/// Run one picked-up export to its end and record the result.
fn run_one(
    shared: &Shared,
    req: ExportRequest,
    flag: &AtomicBool,
    started: Instant,
    last_stage: &mut Option<Stage>,
) {
    let res = req.job.resource;
    let id = res.id;
    tracing::info!(
        model = %res.model_name().unwrap_or(id),
        "model export started ({}, {})",
        catalog::export_recipe(res).map_or("?", |r| r.exporter()),
        res.license
    );
    let mut report = |p: Progress<'_>| match p {
        Progress::Stage {
            stage,
            percent,
            detail,
        } => {
            if *last_stage != Some(stage) {
                tracing::info!(model = %res.model_name().unwrap_or(id), "export stage {}/{}: {}", stage.number(), Stage::ALL.len(), stage.label());
                *last_stage = Some(stage);
            }
            shared.set_state(
                id,
                ExportState::Running {
                    stage,
                    step: stage.number(),
                    steps: Stage::ALL.len(),
                    percent: stage.overall(percent),
                    detail: detail.chars().take(200).collect(),
                },
            );
        }
        Progress::Line { line, .. } => {
            tracing::debug!(target: "blue_onyx_prism::resources::export::child", model = %res.model_name().unwrap_or(id), "{line}");
            shared.update(id, |s| {
                if s.log.len() == LOG_TAIL {
                    s.log.remove(0);
                }
                s.log.push(line.chars().take(300).collect());
            });
        }
    };
    let result = run_export(
        &req.job,
        shared.runner.as_ref(),
        shared.fetcher.as_ref(),
        &mut report,
        flag,
    );
    match result {
        Ok(out) => {
            tracing::info!(
                model = %res.model_name().unwrap_or(id),
                "model export finished in {:.0} s: {} ({} classes)",
                started.elapsed().as_secs_f64(),
                out.onnx.display(),
                out.classes
            );
            // Before the state flips, so a waiter sees the callback's effect.
            if let Some(cb) = req.on_done {
                cb(&out);
            }
            shared.set_state(id, ExportState::Installed);
        }
        Err(e) if e.cancelled => {
            tracing::warn!(model = %res.model_name().unwrap_or(id), "model export cancelled");
            shared.set_state(id, ExportState::Cancelled);
        }
        Err(e) => {
            tracing::warn!(
                model = %res.model_name().unwrap_or(id),
                "model export failed{}: {}",
                e.stage.map(|s| format!(" while {}", s.label())).unwrap_or_default(),
                e.message
            );
            shared.set_state(
                id,
                ExportState::Failed {
                    stage: e.stage,
                    error: e.message,
                },
            );
        }
    }
}

/// Run one export now (CLI): progress lines and the child output go to stderr.
pub fn export_now(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    verbose: bool,
) -> Result<ExportOutput> {
    let cancel = Arc::new(AtomicBool::new(false));
    let _ctrl_c = CtrlC::cancels(Arc::clone(&cancel));
    let mut last: Option<(Stage, u8)> = None;
    let name = job.resource.model_name().unwrap_or(job.resource.id);
    let mut report = |p: Progress<'_>| {
        let mut err = std::io::stderr().lock();
        match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                let overall = stage.overall(percent);
                // One line per stage change and per 10% of overall progress.
                let show = match last {
                    Some((s, o)) => s != stage || overall / 10 != o / 10,
                    None => true,
                };
                if show {
                    let _ = writeln!(
                        err,
                        "[{name}] {overall:>3}% {}/{} {}: {detail}",
                        stage.number(),
                        Stage::ALL.len(),
                        stage.label()
                    );
                    last = Some((stage, overall));
                }
            }
            Progress::Line { line, .. } => {
                if verbose {
                    let _ = writeln!(err, "    {line}");
                }
            }
        }
    };
    run_export(job, runner, fetcher, &mut report, &cancel).map_err(|e| {
        anyhow!(
            "export of {name} failed{}: {}",
            e.stage
                .map(|s| format!(" while {}", s.label()))
                .unwrap_or_default(),
            e.message
        )
    })
}

/// Ctrl-C handler of a CLI export (RS-11): the first Ctrl-C sets the cancel flag, so the running
/// stage stops and its process tree is killed; a second one kills every running process tree
/// ([`kill_live_trees`]) and exits at once (status 130). On Unix a SIGTERM (`kill <pid>`) or
/// SIGHUP (the terminal closed) counts as a Ctrl-C (audit 2026-11, RA recovery-1): without that
/// they would end the CLI and leave `uv` / `python`, which lead their own process groups,
/// running with the CLI's locks released.
///
/// The signals are handled by one process-wide thread, started by the first guard: tokio never
/// gives a signal back to its default action, so after the last guard is dropped the thread
/// still ends the process on one (killing any running tree first, then exiting with 128 + the
/// signal number, as the default action would).
pub(crate) struct CtrlC {
    /// The guard that was active when this one started (restored on drop).
    previous: Option<Active>,
    armed: bool,
}

/// The guard being honoured now.
#[derive(Clone)]
struct Active {
    cancel: Arc<AtomicBool>,
    what: &'static str,
}

/// The active guard, and whether its cancel was already requested.
static ACTIVE: Mutex<Option<(Active, bool)>> = Mutex::new(None);

/// One of the signals a CLI job listens to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliSignal {
    Interrupt,
    #[cfg(unix)]
    Terminate,
    #[cfg(unix)]
    Hangup,
}

impl CliSignal {
    fn name(self) -> &'static str {
        match self {
            CliSignal::Interrupt => "Ctrl-C",
            #[cfg(unix)]
            CliSignal::Terminate => "SIGTERM",
            #[cfg(unix)]
            CliSignal::Hangup => "SIGHUP",
        }
    }

    /// The exit status of the default action (128 + the signal number).
    fn exit_code(self) -> i32 {
        match self {
            CliSignal::Interrupt => 130,
            #[cfg(unix)]
            CliSignal::Terminate => 143,
            #[cfg(unix)]
            CliSignal::Hangup => 129,
        }
    }
}

/// What a signal does now (pure, tested): cancel the active job, or end the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignalAction {
    Cancel,
    Exit,
}

fn signal_action(active: Option<&(Active, bool)>) -> SignalAction {
    match active {
        Some((_, false)) => SignalAction::Cancel,
        _ => SignalAction::Exit,
    }
}

fn on_signal(sig: CliSignal) {
    let mut active = ACTIVE.lock_recover();
    match signal_action(active.as_ref()) {
        SignalAction::Cancel => {
            if let Some((a, asked)) = active.as_mut() {
                eprintln!(
                    "{}: cancelling {} and stopping its processes (press Ctrl-C again to quit \
                     at once)",
                    sig.name(),
                    a.what
                );
                a.cancel.store(true, Ordering::SeqCst);
                *asked = true;
            }
        }
        SignalAction::Exit => {
            drop(active);
            let n = kill_live_trees();
            if n > 0 {
                eprintln!(
                    "{}: stopped {n} running process tree(s); quitting",
                    sig.name()
                );
            }
            std::process::exit(sig.exit_code());
        }
    }
}

/// Start the process-wide signal thread once; false when it could not start.
fn signal_thread() -> bool {
    static STARTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STARTED.get_or_init(|| {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<bool>();
        let spawned = std::thread::Builder::new()
            .name("ctrl-c".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::warn!("Ctrl-C will not cancel the job cleanly: {e}");
                        let _ = ready_tx.send(false);
                        return;
                    }
                };
                rt.block_on(listen(ready_tx));
            });
        match spawned {
            Ok(_) => ready_rx.recv().unwrap_or(false),
            Err(e) => {
                tracing::warn!("Ctrl-C will not cancel the job cleanly: {e}");
                false
            }
        }
    })
}

/// Wait for the next signal of `s` (never, when it could not be registered).
#[cfg(unix)]
async fn recv_opt(s: &mut Option<tokio::signal::unix::Signal>) -> Option<()> {
    match s.as_mut() {
        Some(s) => s.recv().await,
        None => std::future::pending().await,
    }
}

/// The signal thread's loop: registers the handlers (then tells `ready`), handles each signal.
async fn listen(ready: std::sync::mpsc::Sender<bool>) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut int = match signal(SignalKind::interrupt()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Ctrl-C will not cancel the job cleanly: {e}");
                let _ = ready.send(false);
                return;
            }
        };
        let mut term = signal(SignalKind::terminate()).ok();
        let mut hup = signal(SignalKind::hangup()).ok();
        let _ = ready.send(true);
        loop {
            let sig = tokio::select! {
                Some(()) = int.recv() => CliSignal::Interrupt,
                Some(()) = recv_opt(&mut term) => CliSignal::Terminate,
                Some(()) = recv_opt(&mut hup) => CliSignal::Hangup,
                else => return,
            };
            on_signal(sig);
        }
    }
    #[cfg(windows)]
    {
        let mut int = match tokio::signal::windows::ctrl_c() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Ctrl-C will not cancel the job cleanly: {e}");
                let _ = ready.send(false);
                return;
            }
        };
        let _ = ready.send(true);
        while int.recv().await.is_some() {
            on_signal(CliSignal::Interrupt);
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = ready.send(false);
    }
}

impl CtrlC {
    fn cancels(cancel: Arc<AtomicBool>) -> Self {
        Self::cancels_named(cancel, "the export")
    }

    /// [`Self::cancels`] for another long CLI job (`what`: "the training").
    pub(crate) fn cancels_named(cancel: Arc<AtomicBool>, what: &'static str) -> Self {
        if !signal_thread() {
            return Self {
                previous: None,
                armed: false,
            };
        }
        let previous = ACTIVE
            .lock_recover()
            .replace((Active { cancel, what }, false))
            .map(|(a, _)| a);
        Self {
            previous,
            armed: true,
        }
    }
}

impl Drop for CtrlC {
    fn drop(&mut self) {
        if self.armed {
            *ACTIVE.lock_recover() = self.previous.take().map(|a| (a, false));
        }
    }
}

/// A [`ManagerFetcher`] on a fresh download manager for `root` (CLI).
pub fn cli_fetcher(root: &Path) -> ManagerFetcher {
    let mut opts = manager::ManagerOptions::new(root);
    opts.on_progress = Some(manager::cli_progress());
    ManagerFetcher::new(Arc::new(Manager::start(opts)))
}

/// "needs export" text for a configured model file that an export would produce. `outdated`: the
/// file is there but from another version (see [`is_outdated`]).
pub fn needs_export_message(res: &Resource, path: &Path, outdated: bool) -> String {
    let why = if outdated {
        "is from an older version of this model"
    } else {
        "is not there yet"
    };
    format!(
        "needs export: {} {why}. {} ({}) is exported on this machine on request: click Export \
         for it on the Config page (Resources, exported models) or run \
         `blue-onyx-prism fetch --resource {} --allow-large`",
        path.display(),
        res.title,
        res.license,
        res.id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signal_cancels_the_job_once_then_ends_the_process() {
        let active = Active {
            cancel: Arc::new(AtomicBool::new(false)),
            what: "the export",
        };
        assert_eq!(
            signal_action(Some(&(active.clone(), false))),
            SignalAction::Cancel
        );
        // The second one, or any without a job (tokio never restores the default action).
        assert_eq!(signal_action(Some(&(active, true))), SignalAction::Exit);
        assert_eq!(signal_action(None), SignalAction::Exit);
        assert_eq!(CliSignal::Interrupt.exit_code(), 130);
        #[cfg(unix)]
        {
            assert_eq!(CliSignal::Terminate.exit_code(), 128 + libc::SIGTERM);
            assert_eq!(CliSignal::Hangup.exit_code(), 128 + libc::SIGHUP);
        }
    }
}
