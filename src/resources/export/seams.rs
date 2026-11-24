//! Seams of the export stage machine: [`CommandRunner`] (child processes, the real
//! [`SystemRunner`] with process-tree kill, pause / resume and bounded output; [`PausableRunner`]
//! binds a pause flag to any runner) and [`Fetcher`] (verified downloads through the download
//! manager, a cancel stopping the download itself). Tests swap both for the fake in
//! `super::fake`.
//!
//! [`SystemRunner`] is the one spawn path of every long-running child (export, training, import,
//! auto-label, refine): its `Tree` ties the child's whole process tree to the service (own
//! process group + Linux parent-death signal, a Windows Job joined before the child runs, a
//! stdin pipe the scripts watch for EOF, an entry in `<data root>/.children.json`, a kill on
//! drop), and [`kill_live_trees`] ends every running tree on a forced exit. The runner waits on
//! the child in the thread that spawned it (required by `PR_SET_PDEATHSIG`, see
//! [`crate::backend::process`]); after the child exits, its output is drained for a bounded
//! time only.

use super::*;

/// One child process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    pub stage: Stage,
    pub program: PathBuf,
    pub args: Vec<OsString>,
    /// Set on top of the (sanitized) inherited environment.
    pub env: Vec<(String, OsString)>,
    pub cwd: PathBuf,
    pub timeout: Duration,
    /// Start the child below normal priority (Windows `BELOW_NORMAL_PRIORITY_CLASS`, Unix nice
    /// of at least 10; [`crate::backend::priority`]); the processes it starts inherit it. Set
    /// for the training environment's helpers (training, import, auto-label), which keep every
    /// core busy for hours; the model export's commands keep the normal priority.
    pub low_priority: bool,
}

impl Cmd {
    /// `program arg arg ...` for logs and tests.
    pub fn display(&self) -> String {
        std::iter::once(self.program.to_string_lossy().into_owned())
            .chain(self.args.iter().map(|a| a.to_string_lossy().into_owned()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Runs child processes (the real one spawns them; tests fake it).
pub trait CommandRunner: Send + Sync {
    /// Run `cmd` to completion, passing each output line (stdout and stderr) to `on_line`. Fails
    /// on a non-zero exit, on timeout and when `cancel` is set (the child is killed).
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()>;

    /// [`Self::run`] that also follows `pause`: while it is set the child's whole process tree is
    /// suspended (it keeps its memory, open files and GPU allocations) and continues where it was
    /// when it clears; paused time does not count toward [`Cmd::timeout`]. Entered while `pause`
    /// is set, it waits before starting the child. `cancel` works while paused. The default
    /// ignores `pause` (test fakes).
    fn run_pausable(
        &self,
        cmd: &Cmd,
        on_line: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
        pause: &AtomicBool,
    ) -> Result<()> {
        let _ = pause;
        self.run(cmd, on_line, cancel)
    }
}

/// A runner whose every child follows one pause flag ([`CommandRunner::run_pausable`]): the
/// training queue hands it to its jobs, so their code needs no pause support of its own. (Its
/// own `run_pausable` is the default: a second flag passed to it is ignored.)
pub struct PausableRunner<'a> {
    pub inner: &'a dyn CommandRunner,
    pub pause: &'a AtomicBool,
}

impl std::fmt::Debug for PausableRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PausableRunner")
            .field("paused", &self.pause.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl CommandRunner for PausableRunner<'_> {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        self.inner.run_pausable(cmd, on_line, cancel, self.pause)
    }
}

/// Installs a verified resource (the real one goes through the download manager).
pub trait Fetcher: Send + Sync {
    /// Install `res` into `dir` (archives extracted, plain files placed), reporting
    /// `(bytes, total)`. Returns when installed; fails on a download error or `cancel`.
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()>;
}

/// Longest output line kept from a child (bytes); longer segments are cut and marked `...`
/// (RS-08: progress bars rewrite one line for hours, a misbehaving child never emits a newline).
pub(crate) const MAX_OUTPUT_LINE: usize = 4096;
/// Output lines buffered between the reader threads and the runner; a child that outruns the
/// log blocks on its pipe instead of growing memory.
pub(crate) const MAX_OUTPUT_LINES: usize = 2000;

/// Whether the child inherits `key` from the service's environment (RS-07). The child starts
/// from an empty environment; only what a process needs to run, find its temp dir and reach the
/// network through the operator's proxy comes through, plus [`Cmd::env`]. Everything that could
/// redirect Python, uv, pip, the dynamic loader (`LD_*`, `DYLD_*`), the model libraries
/// (`YOLO_*`, `HF_*`, `TORCH_*`) or the CA bundle of `requests`/curl is dropped.
/// `SSL_CERT_FILE`/`SSL_CERT_DIR` pass through when the operator set them (a private CA for the
/// proxy); nothing sets them by default. `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER` pass through
/// so PyTorch numbers the GPUs as the service's CUDA probe did: the training GPU picker hands the
/// child `--device cuda:N` with the probe's ordinal. `PROCESSOR_ARCHITECTURE` (and the WOW64
/// `PROCESSOR_ARCHITEW6432`) pass through because Python's `platform.machine()` reads them on
/// Windows: without them it is `''`, and polars (imported by Ultralytics when it saves a training
/// checkpoint) then refuses to load ("unknown feature flag: 'sse3'").
pub(super) fn inherited(key: &str) -> bool {
    let k = key.to_ascii_uppercase();
    matches!(
        k.as_str(),
        "PATH"
            | "HOME"
            | "USERPROFILE"
            | "SYSTEMROOT"
            | "SYSTEMDRIVE"
            | "COMSPEC"
            | "PATHEXT"
            | "PROCESSOR_ARCHITECTURE"
            | "PROCESSOR_ARCHITEW6432"
            | "APPDATA"
            | "LOCALAPPDATA"
            | "PROGRAMDATA"
            | "TEMP"
            | "TMP"
            | "TMPDIR"
            | "LANG"
            | "HTTP_PROXY"
            | "HTTPS_PROXY"
            | "NO_PROXY"
            | "ALL_PROXY"
            | "SSL_CERT_FILE"
            | "SSL_CERT_DIR"
            | "CUDA_VISIBLE_DEVICES"
            | "CUDA_DEVICE_ORDER"
    ) || k.starts_with("LC_")
}

/// Output lines of a failed child kept for [`super::failure::classify`] (the cause of a Python
/// failure is often far above its last lines: a faulthandler stack, then a long traceback).
pub(crate) const FAILURE_LINES: usize = 200;
/// Of those, the lines quoted in the error message.
pub(crate) const FAILURE_TAIL: usize = 8;

/// A child process exited unsuccessfully. Displays as "`program` failed (`status`); last output:
/// <the last [`FAILURE_TAIL`] lines>"; [`Self::lines`] has up to [`FAILURE_LINES`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildFailed {
    /// File name of the program.
    pub program: String,
    /// "exit code: 255".
    pub status: String,
    /// The last output lines, oldest first.
    pub lines: Vec<String>,
}

impl std::fmt::Display for ChildFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = self.lines.len().saturating_sub(FAILURE_TAIL);
        write!(
            f,
            "{} failed ({}); last output: {}",
            self.program,
            self.status,
            self.lines.get(from..).unwrap_or_default().join(" | ")
        )
    }
}

impl std::error::Error for ChildFailed {}

/// Spawns real processes.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemRunner;

/// Set to `1` in every child's environment: the child's stdin is the parent-watch pipe (a pipe
/// the service never writes), so the Python scripts may watch it for EOF. A script run by hand (stdin a
/// terminal) does not have it and does not watch.
pub const PARENT_PIPE_ENV: &str = "BOP_PARENT_PIPE";

/// After a child exited, how long its helpers may keep the output pipes open before the rest
/// of the tree is killed.
const ORPHAN_GRACE: Duration = Duration::from_secs(2);
/// After that kill, how much longer the output is drained before the runner stops waiting for
/// the pipes (RA resources-8: a helper outside the group or job holds them; the reader threads
/// are left to end with it).
const DRAIN_AFTER_KILL: Duration = Duration::from_secs(5);
/// Between SIGTERM and SIGKILL when a tree is stopped (Unix): uv drops its temp files.
#[cfg(unix)]
const TERM_GRACE: Duration = Duration::from_millis(1500);

/// Process groups of the trees running now (Unix; empty on Windows, where each tree's Job
/// Object dies with the process): [`kill_live_trees`] ends them on the way out of a forced exit.
static LIVE: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// SIGKILL every process tree a [`SystemRunner`] of this process is running now (each after a
/// SIGCONT, so paused trees die too), for exits that skip the runners' own cleanup: the second
/// Ctrl-C of a CLI command, the service's forced exit. Returns how many groups were signalled.
/// Windows: nothing to do, the trees' Job Objects kill them when the process exits.
pub fn kill_live_trees() -> usize {
    let groups: Vec<u32> = LIVE.lock_recover().clone();
    for &g in &groups {
        if let Err(e) = crate::backend::process::kill_group(g) {
            tracing::warn!("could not end process group {g}: {e}");
        }
    }
    groups.len()
}

/// Keeps a tree's group in [`LIVE`] while it runs.
struct Live(u32);

impl Live {
    fn add(pgid: u32) -> Self {
        if cfg!(unix) {
            LIVE.lock_recover().push(pgid);
        }
        Self(pgid)
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let mut live = LIVE.lock_recover();
        if let Some(i) = live.iter().position(|&g| g == self.0) {
            live.swap_remove(i);
        }
    }
}

/// The data root a command's children are recorded in ([`crate::children`]): the parent of
/// the `tools` folder its program lives in (uv and the Python environments are all under
/// `<data root>/tools/`, see [`Paths`]). None (not recorded) for a program elsewhere.
pub(crate) fn record_root(program: &Path) -> Option<PathBuf> {
    program
        .ancestors()
        .skip(1)
        .find(|a| a.file_name().is_some_and(|n| n == TOOLS_DIR))
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

/// A running child and what ties its whole process tree to it (audit 2026-11, RA resources-1,
/// -8, -9, backend-8):
/// - the tree: the process group the child leads on Unix ([`crate::backend::process::prepare`],
///   which on Linux also asks for SIGKILL when the spawning thread dies), a kill-on-close Job
///   Object on Windows ([`crate::backend::job`]) that the child joins before its first
///   instruction (spawned suspended, assigned, then resumed; a job that cannot be made or
///   joined fails the spawn rather than run a tree that would survive a cancel);
/// - the parent watch: the child's stdin is a pipe whose write end only this struct holds and
///   never writes, so when the service dies every descendant that inherited stdin reads EOF
///   (the scripts exit on it, [`PARENT_PIPE_ENV`]);
/// - its entry in `<data root>/.children.json` ([`crate::children`], for a program under
///   `tools/`) so the next start ends it after a crash;
/// - dropped before the child was reaped (an early `?`, a panic in a line handler), the tree is
///   killed (SIGCONT + SIGKILL to the group; Windows: the job).
///
/// Signals go through `killpg(2)` ([`crate::backend::process::signal_group`]), never an
/// external `kill` program.
struct Tree {
    child: std::process::Child,
    /// The child has been waited for.
    reaped: bool,
    /// Write end of the child's stdin, never written: closed when the tree is dropped.
    parent_pipe: Option<std::process::ChildStdin>,
    #[cfg(windows)]
    job: crate::backend::job::Job,
    /// The tree's threads while it is paused (resumed on drop).
    #[cfg(windows)]
    suspended: Option<crate::backend::job::Suspended>,
    /// The tree is paused (Unix: SIGSTOP sent).
    #[cfg(unix)]
    stopped: bool,
    _record: Option<crate::children::Recorded>,
    _live: Live,
}

impl Tree {
    fn spawn(cmd: &Cmd) -> Result<Self> {
        use std::process::{Command, Stdio};
        let mut c = Command::new(&cmd.program);
        c.args(&cmd.args)
            .current_dir(&cmd.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear();
        for (k, v) in std::env::vars_os() {
            if inherited(&k.to_string_lossy()) {
                c.env(&k, &v);
            }
        }
        // BE-15: the service prepends the runtime library dirs (OpenVINO, ONNX Runtime, CUDA) to
        // PATH on Windows; the export's Python must not pick up those DLLs. None elsewhere.
        if let Some(path) = crate::backend::libs::original_path() {
            c.env("PATH", path);
        }
        for (k, v) in &cmd.env {
            c.env(k, v);
        }
        c.env(PARENT_PIPE_ENV, "1");
        // Own process group (pgid = pid) so the tree can be signalled as one; Linux: SIGKILL
        // when this (spawning and waiting) thread dies.
        crate::backend::process::prepare(&mut c);
        #[cfg(unix)]
        if cmd.low_priority {
            crate::backend::priority::lower_child(&mut c);
        }
        #[cfg(windows)]
        let job = {
            use std::os::windows::process::CommandExt;
            /// CREATE_SUSPENDED: the child runs nothing before it is in the job.
            const CREATE_SUSPENDED: u32 = 0x0000_0004;
            // CREATE_NO_WINDOW: no console window when the service runs it; below normal
            // priority for a low-priority command (inherited by the processes it starts).
            c.creation_flags(crate::backend::priority::child_creation_flags(
                crate::command::CREATE_NO_WINDOW | CREATE_SUSPENDED,
                cmd.low_priority,
            ));
            crate::backend::job::Job::new().with_context(|| {
                format!(
                    "could not create a job object for {}; without it its helper processes \
                     would survive a cancel. Restart the service; if it persists, the service \
                     may be running inside a job that forbids nested jobs",
                    cmd.program.display()
                )
            })?
        };
        let child = c
            .spawn()
            .with_context(|| format!("starting {}", cmd.program.display()))?;
        #[cfg(windows)]
        let mut child = child;
        #[cfg(windows)]
        if let Err(e) = job
            .assign(&child)
            .and_then(|()| crate::backend::job::resume_process(child.id()).map(|_| ()))
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e.context(format!(
                "could not put {} in a job object (its helpers would survive a cancel), so it \
                 was not run",
                cmd.program.display()
            )));
        }
        let mut tree = Tree {
            reaped: false,
            parent_pipe: None,
            #[cfg(windows)]
            job,
            #[cfg(windows)]
            suspended: None,
            #[cfg(unix)]
            stopped: false,
            _record: None,
            _live: Live::add(child.id()),
            child,
        };
        tree.parent_pipe = tree.child.stdin.take();
        tree._record = record_root(&cmd.program).and_then(|root| {
            let kind = cmd
                .program
                .file_stem()
                .map_or_else(|| "child".to_string(), |n| n.to_string_lossy().into_owned());
            crate::children::Registry::new(root)
                .record(&kind, tree.child.id())
                .map_err(|e| {
                    tracing::warn!(
                        "{} is not listed for the startup clean-up: {e:#}; if the service is \
                         killed while it runs, it may keep running",
                        cmd.program.display()
                    );
                })
                .ok()
        });
        Ok(tree)
    }

    /// Process group of the tree (Unix: the child leads it).
    #[cfg(unix)]
    fn pgid(&self) -> u32 {
        self.child.id()
    }

    /// Send `sig` to the group, logging a failure (a group that is gone is not one).
    #[cfg(unix)]
    fn signal(&self, sig: crate::backend::process::Signal) -> bool {
        match crate::backend::process::signal_group(self.pgid(), sig) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    "could not send {sig:?} to process group {}: {e}",
                    self.pgid()
                );
                false
            }
        }
    }

    /// Pause the child and every process it spawned (Unix: STOP to the group; Windows: every
    /// thread of the job suspended). A failure is logged: the tree keeps running.
    fn suspend(&mut self) {
        #[cfg(unix)]
        if self.signal(crate::backend::process::Signal::Stop) {
            self.stopped = true;
        } else {
            tracing::warn!("the process tree was not paused and keeps running");
        }
        #[cfg(windows)]
        if self.suspended.is_none() {
            match self.job.suspend() {
                Ok(s) => {
                    tracing::debug!("paused {} threads of the process tree", s.len());
                    self.suspended = Some(s);
                }
                Err(e) => tracing::warn!("the process tree was not paused: {e:#}"),
            }
        }
    }

    /// Continue a tree paused by [`Self::suspend`].
    fn resume(&mut self) {
        #[cfg(unix)]
        if self.stopped {
            self.signal(crate::backend::process::Signal::Cont);
            self.stopped = false;
        }
        #[cfg(windows)]
        if let Some(s) = self.suspended.take() {
            s.resume();
        }
    }

    /// Kill the child and every process it spawned, then reap the child. Unix: SIGTERM and
    /// SIGCONT to the group (a paused tree acts on the TERM only once it runs), up to
    /// [`TERM_GRACE`] for the leader, then SIGCONT + SIGKILL to the group. The group must be
    /// signalled before `wait` reaps the leader (the pgid lives on with the leader's zombie).
    fn kill_tree(&mut self) {
        if self.reaped {
            return;
        }
        #[cfg(unix)]
        {
            use crate::backend::process::Signal;
            self.signal(Signal::Term);
            self.signal(Signal::Cont);
            let until = Instant::now() + TERM_GRACE;
            while Instant::now() < until {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        self.kill_now();
    }

    /// End the tree at once (no grace) and reap the child.
    fn kill_now(&mut self) {
        if self.reaped {
            return;
        }
        #[cfg(unix)]
        {
            // kill_group sends SIGCONT first: a paused tree dies too.
            if let Err(e) = crate::backend::process::kill_group(self.pgid()) {
                tracing::warn!("could not end process group {}: {e}", self.pgid());
            }
            self.stopped = false;
        }
        #[cfg(windows)]
        if let Err(e) = self.job.kill() {
            tracing::warn!("could not end the process tree's job: {e:#}");
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.reaped = true;
        // Windows: close the (now terminated) threads' handles.
        self.resume();
    }

    /// The child exited (`try_wait` returned its status): it is reaped.
    fn exited(&mut self) {
        self.reaped = true;
        #[cfg(unix)]
        {
            self.stopped = false;
        }
    }

    /// Kill what is left of the tree after the child itself exited (helpers still holding the
    /// output pipes; the group outlives its reaped leader while a member lives).
    fn kill_orphans(&self) {
        #[cfg(unix)]
        if let Err(e) = crate::backend::process::kill_group(self.pgid()) {
            tracing::warn!("could not end process group {}: {e}", self.pgid());
        }
        #[cfg(windows)]
        let _ = self.job.kill();
    }

    /// Close the parent-watch pipe now, as the service's death would (tests).
    #[cfg(all(test, unix))]
    fn close_parent_pipe(&mut self) {
        self.parent_pipe = None;
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        if !self.reaped {
            tracing::warn!(
                "process {} was still running when its runner returned; ending its tree",
                self.child.id()
            );
            self.kill_now();
        }
    }
}

/// How often the runner looks at its pause / cancel flags and the child.
const POLL: Duration = Duration::from_millis(100);

impl CommandRunner for SystemRunner {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        self.run_pausable(cmd, on_line, cancel, &AtomicBool::new(false))
    }

    fn run_pausable(
        &self,
        cmd: &Cmd,
        on_line: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
        pause: &AtomicBool,
    ) -> Result<()> {
        // No new helper starts while the job is paused.
        if pause.load(Ordering::Relaxed) {
            tracing::info!("paused before running {}", cmd.display());
            while pause.load(Ordering::Relaxed) {
                if cancel.load(Ordering::Relaxed) {
                    return Err(anyhow!(Cancelled));
                }
                std::thread::sleep(POLL);
            }
        }
        tracing::info!("running {}", cmd.display());
        let mut tree = Tree::spawn(cmd)?;
        let (tx, rx) = crossbeam_channel::bounded::<String>(MAX_OUTPUT_LINES);
        let mut readers = Vec::new();
        if let Some(out) = tree.child.stdout.take() {
            readers.push(spawn_reader(out, tx.clone()));
        }
        if let Some(err) = tree.child.stderr.take() {
            readers.push(spawn_reader(err, tx.clone()));
        }
        drop(tx);
        let mut deadline = Instant::now() + cmd.timeout;
        // Since when the tree is paused (that time is added to the deadline on resume).
        let mut paused_at: Option<Instant> = None;
        let mut tail: VecDeque<String> = VecDeque::new();
        let mut keep = |line: String, on_line: &mut dyn FnMut(&str)| {
            on_line(&line);
            if tail.len() == FAILURE_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        };
        let status = loop {
            match rx.recv_timeout(POLL) {
                Ok(line) => keep(line, on_line),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Output closed: wait for the exit (still honoring cancel and the timeout).
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if cancel.load(Ordering::Relaxed) {
                tree.kill_tree();
                return Err(anyhow!(Cancelled));
            }
            match (pause.load(Ordering::Relaxed), paused_at) {
                (true, None) => {
                    tracing::info!("pausing {}", cmd.display());
                    tree.suspend();
                    paused_at = Some(Instant::now());
                }
                (false, Some(since)) => {
                    tracing::info!(
                        "resuming {} after {:.0} s",
                        cmd.display(),
                        since.elapsed().as_secs_f64()
                    );
                    tree.resume();
                    deadline += since.elapsed();
                    paused_at = None;
                }
                _ => {}
            }
            if paused_at.is_none() && Instant::now() >= deadline {
                tree.kill_tree();
                bail!(
                    "{} timed out after {} min and was stopped",
                    cmd.program.display(),
                    cmd.timeout.as_secs() / 60
                );
            }
            // An error here drops the tree unreaped: its Drop kills it.
            if let Some(status) = tree.child.try_wait()? {
                tree.exited();
                break status;
            }
        };
        // Drain until both readers hang up (a bounded channel: the readers may still be
        // blocked on `send`), then join them. A helper the child left behind still holding
        // the pipes would keep them open forever: after ORPHAN_GRACE the rest of the tree is
        // killed (the group only gets a signal while it has members, so the pgid cannot have
        // been reused), and DRAIN_AFTER_KILL later (a helper outside the group or job) the
        // runner stops waiting and leaves the reader threads to end with the pipe. A cancel
        // stops the drain too.
        let grace = Instant::now() + ORPHAN_GRACE;
        let give_up = grace + DRAIN_AFTER_KILL;
        let mut orphans_killed = false;
        let mut drained = false;
        let mut cancelled = false;
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => keep(line, on_line),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    drained = true;
                    break;
                }
            }
            if cancel.load(Ordering::Relaxed) {
                tree.kill_orphans();
                cancelled = true;
                break;
            }
            let now = Instant::now();
            if !orphans_killed && now >= grace {
                tracing::warn!(
                    "{} left helper processes behind; stopping them",
                    cmd.display()
                );
                tree.kill_orphans();
                orphans_killed = true;
            }
            if now >= give_up {
                tracing::warn!(
                    "{} exited {} s ago, but a process outside its tree still holds its \
                     output; no longer waiting for it (process group {})",
                    cmd.display(),
                    (ORPHAN_GRACE + DRAIN_AFTER_KILL).as_secs(),
                    tree.child.id()
                );
                break;
            }
        }
        if drained {
            for r in readers {
                let _ = r.join();
            }
        } else {
            // The readers end when the pipe closes or their next send fails.
            drop(rx);
            drop(readers);
        }
        if cancelled {
            return Err(anyhow!(Cancelled));
        }
        if !status.success() {
            return Err(anyhow::Error::new(ChildFailed {
                program: cmd.program.file_name().map_or_else(
                    || cmd.program.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                ),
                status: status.to_string(),
                lines: tail.into_iter().collect(),
            }));
        }
        Ok(())
    }
}

/// Reads one output stream line by line (`\n` and `\r` both end a line: progress bars rewrite
/// one line with `\r`), keeping at most [`MAX_OUTPUT_LINE`] bytes of each, and sends the
/// non-empty lines; returns when the stream or the receiver closes.
fn spawn_reader(
    stream: impl std::io::Read + Send + 'static,
    tx: Sender<String>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = std::io::BufReader::new(stream);
        let mut line = Vec::with_capacity(256);
        let mut cut = false;
        loop {
            let data = match reader.fill_buf() {
                Ok(d) if !d.is_empty() => d,
                _ => break,
            };
            let n = data.len();
            let mut closed = false;
            for &b in data {
                if b == b'\n' || b == b'\r' {
                    if !send_line(&tx, &line, cut) {
                        closed = true;
                        break;
                    }
                    line.clear();
                    cut = false;
                } else if line.len() < MAX_OUTPUT_LINE {
                    line.push(b);
                } else {
                    cut = true;
                }
            }
            reader.consume(n);
            if closed {
                return;
            }
        }
        send_line(&tx, &line, cut);
    })
}

/// Send one captured line (ANSI stripped, trimmed, `...` when cut); false when the receiver is
/// gone.
fn send_line(tx: &Sender<String>, raw: &[u8], cut: bool) -> bool {
    let mut line = strip_ansi(&String::from_utf8_lossy(raw))
        .trim_end()
        .to_string();
    if line.is_empty() {
        return true;
    }
    if cut {
        line.push_str("...");
    }
    tx.send(line).is_ok()
}

/// `text` without ANSI escape sequences (Ultralytics colors its log lines).
pub(super) fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI: ESC [ params final-byte (0x40..=0x7e); other escapes: drop ESC + one char.
            if chars.peek() == Some(&'[') {
                chars.next();
                for n in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&n) {
                        break;
                    }
                }
            } else {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Verified downloads through the shared [`Manager`] (same queue, lock, resume and SHA-256 checks
/// as every other download).
///
/// A cancel stops the download, not just the wait (RA resources-4): the job is queued as one
/// waiter's ([`Manager::enqueue_waiting`]) and [`manager::Waiter::give_up`] cancels it unless a
/// Download button, the service's provisioning or another fetch still wants it. A failed attempt
/// that the manager retries (`Failed { retry_at: Some(..) }`, the service) is waited out, still
/// honouring the cancel; only a final failure fails the fetch.
#[derive(Debug)]
pub struct ManagerFetcher {
    manager: Arc<Manager>,
}

impl ManagerFetcher {
    pub fn new(manager: Arc<Manager>) -> Self {
        Self { manager }
    }
}

impl Fetcher for ManagerFetcher {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()> {
        let job = Job::into_dir(res, dir.to_path_buf());
        let key = job.key();
        // A remembered "installed" from before a removal would make enqueue a no-op.
        self.manager.forget(&key);
        let waiter = self.manager.enqueue_waiting(job);
        let mut told_retry: Option<u32> = None;
        loop {
            if cancel.load(Ordering::Relaxed) {
                if waiter.give_up() {
                    tracing::info!("stopped the download of {}", res.title);
                }
                return Err(anyhow!(Cancelled));
            }
            match self.manager.status(&key).map(|s| s.state) {
                Some(DownloadState::Installed) => return Ok(()),
                Some(DownloadState::Failed {
                    error,
                    retry_at: None,
                    ..
                }) => {
                    bail!("download of {} failed: {error}", res.title)
                }
                // Not final (QA-037): the manager tries again after its backoff; keep waiting.
                Some(
                    state @ DownloadState::Failed {
                        retry_at: Some(_),
                        attempts,
                        ..
                    },
                ) => {
                    if told_retry != Some(attempts) {
                        told_retry = Some(attempts);
                        // "download of X failed: ...; retrying in N s", the model-state text.
                        tracing::info!("{} (cancel to stop it)", state.describe_for(res.title));
                    }
                }
                Some(DownloadState::Downloading { bytes, total }) => progress(bytes, total),
                _ if !self.manager.is_alive() => bail!(
                    "download of {} was not started: the download manager is not running; \
                     restart the service (or rerun the command)",
                    res.title
                ),
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// [`ManagerFetcher`] on a real download manager and a loopback server.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod fetcher_tests {
    use super::*;
    use crate::resources::catalog::{Layout, Part, Provides};
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    const BODY: &[u8] = b"weights of a model that downloads on the second try";

    /// Answers the first request with a 503, every later one with [`BODY`]. Returns the URL and
    /// the request count.
    fn fails_once() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/weights.bin", listener.local_addr().unwrap());
        let count = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&count);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if s.read(&mut byte).map_or(true, |n| n == 0) {
                        break;
                    }
                    head.push(byte[0]);
                }
                if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    let _ = write!(
                        s,
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                         Connection: close\r\n\r\n"
                    );
                } else {
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        BODY.len()
                    );
                    let _ = s.write_all(BODY);
                }
                let _ = s.flush();
            }
        });
        (url, count)
    }

    fn resource(url: &str) -> &'static Resource {
        use sha2::{Digest, Sha256};
        let sha: String = Sha256::digest(BODY)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let part = Part {
            url: Box::leak(url.to_string().into_boxed_str()),
            sha256: Box::leak(sha.into_boxed_str()),
            size: BODY.len() as u64,
            file_name: "weights.bin",
            archive: None,
            layout: Layout::File,
        };
        Box::leak(Box::new(Resource {
            id: "model:fetcher-test",
            kind: ResourceKind::Model,
            version: "1",
            platform: None,
            parts: Box::leak(vec![part].into_boxed_slice()),
            dest: "models",
            provides: Provides::Model {
                name: "fetcher-test",
                family: crate::model::ModelFamilyKind::Yolo5,
            },
            title: "Fetcher test weights",
            description: "test",
            license: "MIT",
        }))
    }

    /// QA-037: a failed attempt the manager retries (`Failed { retry_at: Some(..) }`, the
    /// service's manager) is waited out; the fetch succeeds with the retry. The backoff is long
    /// enough that the fetch's 200 ms poll sees the failed state between the attempts.
    #[test]
    fn a_retried_download_failure_does_not_fail_the_fetch() {
        let (url, requests) = fails_once();
        let res = resource(&url);
        let root = crate::test_dirs::TempDir::new("fetch-retry");
        let manager = Arc::new(Manager::start(manager::ManagerOptions {
            retry: true,
            backoff: |_| Duration::from_millis(1500),
            max_attempts: Some(3),
            ..manager::ManagerOptions::new(root.to_path_buf())
        }));
        let events = manager.subscribe();
        let fetcher = ManagerFetcher::new(Arc::clone(&manager));
        let dir = root.join("models");
        let result = fetcher.fetch(res, &dir, &mut |_, _| {}, &AtomicBool::new(false));
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(std::fs::read(dir.join("weights.bin")).unwrap(), BODY);
        assert_eq!(requests.load(Ordering::SeqCst), 2, "one failure, one retry");
        let first = events.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(
            matches!(
                first,
                manager::Event::Failed {
                    retry_at: Some(_),
                    ..
                }
            ),
            "{first:?}"
        );
        manager.shutdown();
    }
}

/// [`SystemRunner`] on a real process tree: cancel, timeout, pause, resume and a helper left behind,
/// with a ticking grandchild so that "the whole tree" means more than the process the runner spawned.
/// The bodies run on every OS and only [`Ticker`] differs. On Windows this is the only test of the
/// job object's suspend, resume and kill ([`crate::backend::job`]), through [`Tree`]; Unix uses
/// process groups, and `export::tests` covers those as well.
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tree_tests {
    use super::*;
    use std::sync::Mutex;

    /// A child that starts a grandchild, which ticks about once a second: one line on stdout, read
    /// by the runner, and one more line in [`Self::file`], which outlives the runner's interest. The
    /// file is watched by its size (no handle held, so no sharing rules: a second handle may not
    /// read a file a Windows process has locked). The loop ends by itself after two minutes, so a
    /// failing test leaves nothing running; no test lets it run that long.
    struct Ticker {
        dir: PathBuf,
        file: PathBuf,
    }

    impl Ticker {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("bop-tree-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            #[cfg(windows)]
            std::fs::write(
                dir.join("tick.cmd"),
                "@echo off\r\nfor /L %%i in (1,1,120) do (\r\n  echo tick\r\n  \
                 echo tick>>ticks.txt\r\n  ping -n 2 127.0.0.1 >NUL\r\n)\r\n",
            )
            .unwrap();
            // PIPE ignored: when the runner stops reading, only the kill may end the loop.
            #[cfg(unix)]
            std::fs::write(
                dir.join("tick.sh"),
                "trap '' PIPE\ni=0\nwhile [ $i -lt 120 ]; do\n  echo tick\n  \
                 echo tick >>ticks.txt\n  sleep 1\n  i=$((i+1))\ndone\n",
            )
            .unwrap();
            let file = dir.join("ticks.txt");
            Self { dir, file }
        }

        /// The tree: a shell that waits for a second shell, which runs the loop.
        fn cmd(&self, timeout: Duration) -> Cmd {
            #[cfg(windows)]
            let (program, args) = ("cmd", vec!["/C", "cmd", "/C", "tick.cmd"]);
            #[cfg(unix)]
            let (program, args) = ("/bin/sh", vec!["-c", "sh tick.sh & wait"]);
            Cmd {
                stage: Stage::Export,
                program: program.into(),
                args: args.iter().map(OsString::from).collect(),
                env: vec![],
                cwd: self.dir.clone(),
                timeout,
                low_priority: false,
            }
        }

        fn size(&self) -> u64 {
            std::fs::metadata(&self.file).map_or(0, |m| m.len())
        }

        /// Assert that the file stops growing: quiet for several ticks, within a long bound (a
        /// surviving ticker never gets there).
        fn assert_stops(&self, what: &str) {
            let quiet = Duration::from_secs(4);
            let until = Instant::now() + Duration::from_secs(45);
            let (mut last, mut since) = (self.size(), Instant::now());
            while since.elapsed() < quiet {
                assert!(
                    Instant::now() < until,
                    "{what}: the grandchild kept ticking"
                );
                std::thread::sleep(Duration::from_millis(100));
                let now = self.size();
                if now != last {
                    (last, since) = (now, Instant::now());
                }
            }
        }
    }

    impl Drop for Ticker {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A [`SystemRunner::run_pausable`] on its own thread, with its flags and the output lines.
    struct Running {
        lines: Arc<Mutex<Vec<String>>>,
        cancel: Arc<AtomicBool>,
        pause: Arc<AtomicBool>,
        worker: std::thread::JoinHandle<Result<()>>,
    }

    impl Running {
        fn start(cmd: Cmd) -> Self {
            let lines = Arc::new(Mutex::new(Vec::new()));
            let cancel = Arc::new(AtomicBool::new(false));
            let pause = Arc::new(AtomicBool::new(false));
            let worker = {
                let (lines, cancel, pause) = (lines.clone(), cancel.clone(), pause.clone());
                std::thread::spawn(move || {
                    SystemRunner.run_pausable(
                        &cmd,
                        &mut |l| lines.lock().unwrap().push(l.to_string()),
                        &cancel,
                        &pause,
                    )
                })
            };
            Self {
                lines,
                cancel,
                pause,
                worker,
            }
        }

        fn count(&self) -> usize {
            self.lines.lock().unwrap().len()
        }

        /// Wait for `n` lines. A failure here is a tree that does not run, whatever it says of the
        /// flags.
        fn wait_for(&self, n: usize) {
            let until = Instant::now() + Duration::from_secs(60);
            while self.count() < n {
                assert!(
                    Instant::now() < until && !self.worker.is_finished(),
                    "{} of {n} lines; finished: {}",
                    self.count(),
                    self.worker.is_finished()
                );
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        /// Set the pause and wait until the tree is quiet (longer than a tick, the lines in flight
        /// in). Returns the lines and the file's size from then on.
        fn pause(&self, t: &Ticker) -> (usize, u64) {
            self.pause.store(true, Ordering::Relaxed);
            let until = Instant::now() + Duration::from_secs(45);
            let mut last = (self.count(), t.size());
            let mut since = Instant::now();
            while since.elapsed() < Duration::from_secs(2) {
                assert!(Instant::now() < until, "the tree kept ticking when paused");
                std::thread::sleep(Duration::from_millis(50));
                let now = (self.count(), t.size());
                if now != last {
                    (last, since) = (now, Instant::now());
                }
            }
            last
        }

        /// The runner's result. A runner that never returns fails here rather than hanging the
        /// test binary.
        fn finish(self) -> Result<()> {
            let until = Instant::now() + Duration::from_secs(120);
            while !self.worker.is_finished() {
                assert!(Instant::now() < until, "the runner did not return");
                std::thread::sleep(Duration::from_millis(20));
            }
            self.worker.join().unwrap()
        }
    }

    #[test]
    fn a_cancel_stops_the_whole_tree() {
        let t = Ticker::new("cancel");
        let run = Running::start(t.cmd(Duration::from_secs(600)));
        run.wait_for(2);
        run.cancel.store(true, Ordering::Relaxed);
        let e = run.finish().unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        t.assert_stops("cancel");
    }

    #[test]
    fn a_timeout_stops_the_whole_tree() {
        let t = Ticker::new("timeout");
        let run = Running::start(t.cmd(Duration::from_secs(3)));
        let e = run.finish().unwrap_err();
        assert!(format!("{e:#}").contains("timed out"), "{e:#}");
        t.assert_stops("timeout");
    }

    #[test]
    fn a_pause_holds_the_whole_tree_and_a_resume_continues_it() {
        let t = Ticker::new("pause");
        let run = Running::start(t.cmd(Duration::from_secs(600)));
        run.wait_for(2);
        let (lines, size) = run.pause(&t);
        // Several ticks' worth of silence: the grandchild does the ticking.
        std::thread::sleep(Duration::from_secs(4));
        assert_eq!(run.count(), lines, "output while paused");
        assert_eq!(t.size(), size, "the grandchild ran while paused");
        run.pause.store(false, Ordering::Relaxed);
        run.wait_for(lines + 2);
        assert!(t.size() > size, "no tick after the resume");
        // Paused again, a cancel still reaches the tree, and nothing it held comes back to life.
        run.pause(&t);
        run.cancel.store(true, Ordering::Relaxed);
        let e = run.finish().unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        t.assert_stops("cancel while paused");
    }

    /// The pause alone outlasts the child's `TIMEOUT`, yet the child goes on after it, and the runner
    /// still times it out once it has used up the rest.
    #[test]
    fn paused_time_does_not_count_toward_the_timeout() {
        const TIMEOUT: Duration = Duration::from_secs(15);
        let t = Ticker::new("paused-timeout");
        let run = Running::start(t.cmd(TIMEOUT));
        run.wait_for(1);
        let (lines, _) = run.pause(&t);
        std::thread::sleep(TIMEOUT + Duration::from_secs(1));
        if run.worker.is_finished() {
            let why = run.finish().map_err(|e| format!("{e:#}"));
            panic!("the runner ended during the pause: {why:?}");
        }
        run.pause.store(false, Ordering::Relaxed);
        // Had the deadline stayed, the first poll after the resume would end the child.
        run.wait_for(lines + 3);
        let e = run.finish().unwrap_err();
        assert!(format!("{e:#}").contains("timed out"), "{e:#}");
        t.assert_stops("timeout after a pause");
    }

    /// Entered while paused, the child does not start until the pause clears; a cancel while it
    /// waits returns without starting it.
    #[test]
    fn nothing_starts_during_a_pause() {
        let t = Ticker::new("entered-paused");
        let pause = Arc::new(AtomicBool::new(true));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = {
            let (pause, cancel) = (pause.clone(), cancel.clone());
            let cmd = t.cmd(Duration::from_secs(600));
            std::thread::spawn(move || {
                SystemRunner.run_pausable(&cmd, &mut |_| {}, &cancel, &pause)
            })
        };
        std::thread::sleep(POLL * 5);
        assert_eq!(t.size(), 0, "started during the pause");
        assert!(!worker.is_finished());
        cancel.store(true, Ordering::Relaxed);
        let e = worker.join().unwrap().unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        assert_eq!(t.size(), 0);
    }

    /// A helper that outlives a successful child, holding the output pipes, is stopped after the
    /// grace period: the run ends long before the helper would, and well before its timeout.
    #[test]
    fn a_helper_left_behind_is_stopped() {
        let t = Ticker::new("orphan");
        const TIMEOUT: Duration = Duration::from_secs(600);
        #[cfg(windows)]
        let args = ["/C", "start", "/B", "cmd", "/C", "tick.cmd"];
        #[cfg(unix)]
        let args = ["-c", "sh tick.sh & exit 0"];
        let mut cmd = t.cmd(TIMEOUT);
        cmd.args = args.iter().map(OsString::from).collect();
        let run = Running::start(cmd);
        run.finish().unwrap();
        t.assert_stops("a helper left behind");
    }

    #[test]
    fn record_root_is_the_parent_of_the_tools_folder() {
        let root = Path::new("/data/root");
        assert_eq!(
            record_root(&root.join("tools/uv/uv")),
            Some(root.to_path_buf())
        );
        assert_eq!(
            record_root(&root.join("tools/export-env/bin/python")),
            Some(root.to_path_buf())
        );
        // The nearest `tools` above the program counts, not one in the data root's own path.
        assert_eq!(
            record_root(Path::new("/home/u/tools/bop/tools/train-env/bin/python")),
            Some(PathBuf::from("/home/u/tools/bop"))
        );
        assert_eq!(record_root(Path::new("/bin/sh")), None);
        assert_eq!(record_root(Path::new("tools")), None);
    }

    /// A shell command as a [`Cmd`] in `dir`.
    #[cfg(unix)]
    fn sh_cmd(dir: &Path, script: &str) -> Cmd {
        Cmd {
            stage: Stage::Export,
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            env: vec![],
            cwd: dir.to_path_buf(),
            timeout: Duration::from_secs(120),
            low_priority: false,
        }
    }

    /// Whether `pid` is gone (or a zombie) within 15 s.
    #[cfg(unix)]
    fn gone(pid: u32) -> bool {
        let until = Instant::now() + Duration::from_secs(15);
        loop {
            let out = std::process::Command::new("ps")
                .args(["-o", "stat=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            let stat = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if stat.is_empty() || stat.starts_with('Z') {
                return true;
            }
            if Instant::now() > until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// The first line of the tree's stdout (a pid the script echoes).
    #[cfg(unix)]
    fn first_line(tree: &mut Tree) -> String {
        use std::io::BufRead as _;
        let out = tree.child.stdout.take().unwrap();
        let mut line = String::new();
        std::io::BufReader::new(out).read_line(&mut line).unwrap();
        line.trim().to_string()
    }

    /// The parent watch (RA resources-1): the child's stdin is a pipe nothing writes to, it is
    /// told so by `BOP_PARENT_PIPE=1`, and a child waiting on it (the scripts' watcher; here a
    /// shell `cat`) sees EOF exactly when the write end closes, as it does when the service dies.
    #[cfg(unix)]
    #[test]
    fn the_parent_pipe_ends_a_child_that_watches_it() {
        let t = Ticker::new("pipe");
        let mut tree = Tree::spawn(&sh_cmd(
            &t.dir,
            "[ \"$BOP_PARENT_PIPE\" = 1 ] || exit 9; cat >/dev/null; exit 7",
        ))
        .unwrap();
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            tree.child.try_wait().unwrap().is_none(),
            "the child did not wait on its stdin"
        );
        tree.close_parent_pipe();
        let until = Instant::now() + Duration::from_secs(15);
        let status = loop {
            if let Some(s) = tree.child.try_wait().unwrap() {
                tree.exited();
                break s;
            }
            assert!(Instant::now() < until, "EOF never reached the child");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(status.code(), Some(7), "{status}");
    }

    /// The same through the real script watcher: `watch_parent()` of each script, started with
    /// `BOP_PARENT_PIPE=1`, ends the process (and its group) once the pipe closes.
    #[cfg(unix)]
    #[test]
    fn the_scripts_exit_when_the_parent_pipe_closes() {
        let Some(python) = crate::training::progress::test_python::python3(
            "the_scripts_exit_when_the_parent_pipe_closes",
        ) else {
            return;
        };
        let t = Ticker::new("py-pipe");
        let scripts = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts");
        for name in ["export_model", "train_model", "autolabel", "import_footage"] {
            // A grandchild in the script's group: it must die with it.
            let code = format!(
                "import importlib.util, subprocess, sys, time\n\
                 spec = importlib.util.spec_from_file_location('m', '{scripts}/{name}.py')\n\
                 m = importlib.util.module_from_spec(spec)\n\
                 spec.loader.exec_module(m)\n\
                 kid = subprocess.Popen(['sleep', '60'], stdin=subprocess.DEVNULL)\n\
                 print(kid.pid, flush=True)\n\
                 m.watch_parent()\n\
                 time.sleep(60)\n"
            );
            let cmd = Cmd {
                program: python.into(),
                args: vec!["-I".into(), "-c".into(), code.into()],
                ..sh_cmd(&t.dir, "")
            };
            let mut tree = Tree::spawn(&cmd).unwrap();
            let kid: u32 = first_line(&mut tree)
                .parse()
                .unwrap_or_else(|e| panic!("{name}: no pid line ({e})"));
            std::thread::sleep(Duration::from_millis(300));
            assert!(
                tree.child.try_wait().unwrap().is_none(),
                "{name} exited early"
            );
            let t0 = Instant::now();
            tree.close_parent_pipe();
            let until = Instant::now() + Duration::from_secs(20);
            while tree.child.try_wait().unwrap().is_none() {
                assert!(Instant::now() < until, "{name} kept running");
                std::thread::sleep(Duration::from_millis(20));
            }
            tree.exited();
            // The pipe, not the 2 s parent poll (the parent never changed here).
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "{name}: {:?}",
                t0.elapsed()
            );
            assert!(gone(kid), "{name}: its child {kid} survived");
        }
    }

    /// RA resources-1 (+extra): a tree dropped before its child was reaped is killed with its
    /// whole group, also when it is paused.
    #[cfg(unix)]
    #[test]
    fn dropping_a_running_or_paused_tree_kills_its_group() {
        let t = Ticker::new("drop");
        for paused in [false, true] {
            let mut tree = Tree::spawn(&sh_cmd(&t.dir, "sleep 60 & echo $!; wait")).unwrap();
            let leader = tree.child.id();
            let kid: u32 = first_line(&mut tree).parse().unwrap();
            assert!(LIVE.lock_recover().contains(&leader));
            if paused {
                tree.suspend();
                assert!(tree.stopped);
            }
            drop(tree);
            assert!(
                gone(kid),
                "paused={paused}: grandchild {kid} survived the drop"
            );
            assert!(
                gone(leader),
                "paused={paused}: leader {leader} survived the drop"
            );
            assert!(!LIVE.lock_recover().contains(&leader));
        }
    }

    /// A child under `<root>/tools/` is in `<root>/.children.json` while it runs and gone after.
    #[cfg(unix)]
    #[test]
    fn a_running_child_is_recorded_until_it_is_reaped() {
        let t = Ticker::new("record");
        let tools = t.dir.join(TOOLS_DIR);
        std::fs::create_dir_all(&tools).unwrap();
        let sh = tools.join("sh");
        std::os::unix::fs::symlink("/bin/sh", &sh).unwrap();
        let reg = crate::children::Registry::new(&t.dir);
        let cmd = Cmd {
            program: sh,
            ..sh_cmd(&t.dir, "sleep 2")
        };
        let run = std::thread::spawn(move || {
            SystemRunner.run(&cmd, &mut |_| {}, &AtomicBool::new(false))
        });
        let until = Instant::now() + Duration::from_secs(20);
        let entry = loop {
            if let Some(e) = reg.entries().unwrap().into_iter().next() {
                break e;
            }
            assert!(Instant::now() < until, "never recorded");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(entry.kind, "sh");
        assert_eq!(entry.owner_pid, std::process::id());
        run.join().unwrap().unwrap();
        assert!(
            reg.entries().unwrap().is_empty(),
            "still listed after the reap"
        );
    }

    /// RA resources-8: a helper that left the tree's process group and holds the output pipes
    /// does not hang the runner: it stops waiting after the bounded drain.
    #[cfg(unix)]
    #[test]
    fn a_helper_outside_the_group_does_not_hang_the_runner() {
        let t = Ticker::new("escaped");
        // `set -m` in bash: the background job gets a process group of its own (dash, Ubuntu's
        // `/bin/sh`, only warns without a terminal and keeps the job in the group).
        let cmd = sh_cmd(&t.dir, "bash -c 'set -m; sleep 60 & echo $!; exit 0'");
        let mut lines = Vec::new();
        let t0 = Instant::now();
        SystemRunner
            .run(
                &cmd,
                &mut |l| lines.push(l.to_string()),
                &AtomicBool::new(false),
            )
            .unwrap();
        let took = t0.elapsed();
        let helper: u32 = lines
            .iter()
            .find_map(|l| l.trim().parse().ok())
            .unwrap_or_else(|| panic!("no helper pid in {lines:?}"));
        let _ = crate::backend::process::kill_pid(helper);
        assert!(took < Duration::from_secs(30), "{took:?}");
        assert!(took >= ORPHAN_GRACE + DRAIN_AFTER_KILL, "{took:?}");
    }

    /// The drain after the exit honours a cancel too: set once the child has exited, while the
    /// escaped helper still holds the pipes, it ends the run before the orphan grace.
    #[cfg(unix)]
    #[test]
    fn a_cancel_ends_the_drain() {
        let t = Ticker::new("drain-cancel");
        let cmd = sh_cmd(&t.dir, "bash -c 'set -m; sleep 60 & echo $!; exit 0'");
        let cancel = Arc::new(AtomicBool::new(false));
        let setter = {
            let cancel = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(800));
                cancel.store(true, Ordering::Relaxed);
            })
        };
        let mut lines = Vec::new();
        let t0 = Instant::now();
        let e = SystemRunner
            .run(&cmd, &mut |l| lines.push(l.to_string()), &cancel)
            .unwrap_err();
        setter.join().unwrap();
        let helper: u32 = lines
            .iter()
            .find_map(|l| l.trim().parse().ok())
            .unwrap_or_else(|| panic!("no helper pid in {lines:?}"));
        let _ = crate::backend::process::kill_pid(helper);
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        assert!(
            t0.elapsed() < ORPHAN_GRACE + Duration::from_millis(500),
            "{:?}",
            t0.elapsed()
        );
    }
}
