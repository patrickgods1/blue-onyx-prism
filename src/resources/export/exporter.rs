//! [`Exporter`]: one export at a time in a background thread (the service), plus the CLI entry
//! points ([`export_now`], [`cli_fetcher`]) and the "needs export" message.

use super::*;

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
}

/// Lines of output kept per export.
const LOG_TAIL: usize = 40;

struct Shared {
    runner: Arc<dyn CommandRunner>,
    fetcher: Arc<dyn Fetcher>,
    statuses: Mutex<BTreeMap<&'static str, ExportStatus>>,
    changed: Condvar,
    /// The running export's id and cancel flag.
    current: Mutex<Option<(&'static str, Arc<AtomicBool>)>>,
    /// Queued exports cancelled before they started.
    cancelled: Mutex<HashSet<&'static str>>,
    stop: AtomicBool,
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
}

enum Msg {
    Job(Box<ExportRequest>),
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
            cancelled: Mutex::new(HashSet::new()),
            stop: AtomicBool::new(false),
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

    /// Queue an export. Refused while the same model is queued or exporting.
    pub fn enqueue(&self, req: ExportRequest) -> Result<(), String> {
        let res = req.job.resource;
        {
            let mut map = self.shared.statuses.lock_recover();
            if map.get(res.id).is_some_and(|s| s.state.is_busy()) {
                return Err(format!("{} is already being exported", res.title));
            }
            map.insert(
                res.id,
                ExportStatus {
                    id: res.id,
                    title: res.title,
                    state: ExportState::Queued,
                    log: Vec::new(),
                },
            );
        }
        self.shared.cancelled.lock_recover().remove(res.id);
        self.shared.changed.notify_all();
        self.tx
            .send(Msg::Job(Box::new(req)))
            .map_err(|_| "the export thread has stopped".to_string())
    }

    /// Cancel a queued or running export. Returns whether there was one.
    pub fn cancel(&self, id: &str) -> bool {
        let Some(st) = self.status(id) else {
            return false;
        };
        match st.state {
            ExportState::Queued => {
                self.shared.cancelled.lock_recover().insert(st.id);
                self.shared.set_state(st.id, ExportState::Cancelled);
                true
            }
            ExportState::Running { .. } => {
                let cur = self.shared.current.lock_recover();
                if let Some((cid, flag)) = cur.as_ref()
                    && cid.eq_ignore_ascii_case(id)
                {
                    flag.store(true, Ordering::Relaxed);
                    return true;
                }
                false
            }
            _ => false,
        }
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

    /// Stop the worker (a running export is cancelled: its child process is killed) and join.
    pub fn shutdown(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some((_, flag)) = self.shared.current.lock_recover().as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
        let _ = self.tx.send(Msg::Stop);
        let handle = self.thread.lock_recover().take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl Drop for Exporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(shared: &Shared, rx: &Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        let req = match msg {
            Msg::Stop => return,
            Msg::Job(r) => *r,
        };
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let res = req.job.resource;
        let id = res.id;
        let skipped = shared.cancelled.lock_recover().remove(id);
        if skipped {
            continue;
        }
        let flag = Arc::new(AtomicBool::new(false));
        *shared.current.lock_recover() = Some((id, flag.clone()));
        tracing::info!(
            model = %res.model_name().unwrap_or(id),
            "model export started ({}, {})",
            catalog::export_recipe(res).map_or("?", |r| r.exporter()),
            res.license
        );
        let started = Instant::now();
        let mut last_stage: Option<Stage> = None;
        let mut report = |p: Progress<'_>| match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                if last_stage != Some(stage) {
                    tracing::info!(model = %res.model_name().unwrap_or(id), "export stage {}/{}: {}", stage.number(), Stage::ALL.len(), stage.label());
                    last_stage = Some(stage);
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
                tracing::info!(target: "blue_onyx_prism::resources::export::child", model = %res.model_name().unwrap_or(id), "{line}");
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
            &flag,
        );
        *shared.current.lock_recover() = None;
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
        if shared.stop.load(Ordering::Relaxed) {
            return;
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
/// stage stops and its process tree is killed; a second one exits at once (status 130). Stops
/// listening when dropped. Without it, Ctrl-C would end the CLI but leave `uv` / `python` (in
/// their own process group) running.
struct CtrlC {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CtrlC {
    fn cancels(cancel: Arc<AtomicBool>) -> Self {
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("ctrl-c".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::warn!("Ctrl-C will not cancel the export cleanly: {e}");
                        return;
                    }
                };
                rt.block_on(async move {
                    tokio::select! {
                        _ = &mut rx => return,
                        r = tokio::signal::ctrl_c() => if r.is_err() { return },
                    }
                    eprintln!(
                        "Ctrl-C: cancelling the export and stopping its processes (press Ctrl-C \
                         again to quit at once)"
                    );
                    cancel.store(true, Ordering::SeqCst);
                    tokio::select! {
                        _ = &mut rx => {}
                        r = tokio::signal::ctrl_c() => if r.is_ok() {
                            std::process::exit(130);
                        },
                    }
                });
            });
        match thread {
            Ok(t) => Self {
                stop: Some(tx),
                thread: Some(t),
            },
            Err(e) => {
                tracing::warn!("Ctrl-C will not cancel the export cleanly: {e}");
                Self {
                    stop: None,
                    thread: None,
                }
            }
        }
    }
}

impl Drop for CtrlC {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
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
