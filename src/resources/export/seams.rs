//! Seams of the export stage machine: [`CommandRunner`] (child processes, the real
//! [`SystemRunner`] with process-tree kill, pause / resume and bounded output; [`PausableRunner`]
//! binds a pause flag to any runner) and [`Fetcher`] (verified downloads through the download
//! manager). Tests swap both for the fake in `super::fake`.

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
/// child `--device cuda:N` with the probe's ordinal.
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

/// A running child and the handle that kills (and suspends) its whole process tree: the process
/// group on Unix (the child is its leader), a Job Object with kill-on-close on Windows
/// ([`crate::backend::job`]).
struct Tree {
    child: std::process::Child,
    #[cfg(windows)]
    job: Option<crate::backend::job::Job>,
    /// The tree's threads while it is paused (resumed on drop).
    #[cfg(windows)]
    suspended: Option<crate::backend::job::Suspended>,
}

impl Tree {
    fn spawn(cmd: &Cmd) -> Result<Self> {
        use std::process::{Command, Stdio};
        let mut c = Command::new(&cmd.program);
        c.args(&cmd.args)
            .current_dir(&cmd.cwd)
            .stdin(Stdio::null())
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
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Own process group (pgid = pid) so `kill_tree` reaches every descendant.
            c.process_group(0);
            if cmd.low_priority {
                crate::backend::priority::lower_child(&mut c);
            }
        }
        #[cfg(windows)]
        let job = {
            use std::os::windows::process::CommandExt;
            // CREATE_NO_WINDOW: no console window when the service runs it; below normal
            // priority for a low-priority command (inherited by the processes it starts).
            c.creation_flags(crate::backend::priority::child_creation_flags(
                0x0800_0000,
                cmd.low_priority,
            ));
            match crate::backend::job::Job::new() {
                Ok(j) => Some(j),
                Err(e) => {
                    tracing::warn!("no job object for the export process tree: {e:#}");
                    None
                }
            }
        };
        let child = c
            .spawn()
            .with_context(|| format!("starting {}", cmd.program.display()))?;
        #[cfg(windows)]
        if let Some(j) = &job
            && let Err(e) = j.assign(&child)
        {
            tracing::warn!("export child not assigned to its job object: {e:#}");
        }
        Ok(Tree {
            child,
            #[cfg(windows)]
            job,
            #[cfg(windows)]
            suspended: None,
        })
    }

    /// Pause the child and every process it spawned (Unix: STOP to the group; Windows: every
    /// thread of the job suspended).
    fn suspend(&mut self) {
        #[cfg(unix)]
        signal_group(self.child.id(), "STOP");
        #[cfg(windows)]
        match &self.job {
            Some(j) if self.suspended.is_none() => match j.suspend() {
                Ok(s) => {
                    tracing::debug!("paused {} threads of the process tree", s.len());
                    self.suspended = Some(s);
                }
                Err(e) => tracing::warn!("the process tree was not paused: {e:#}"),
            },
            Some(_) => {}
            None => tracing::warn!(
                "the process tree cannot be paused: it has no job object (see the earlier warning)"
            ),
        }
    }

    /// Continue a tree paused by [`Self::suspend`].
    fn resume(&mut self) {
        #[cfg(unix)]
        signal_group(self.child.id(), "CONT");
        #[cfg(windows)]
        if let Some(s) = self.suspended.take() {
            s.resume();
        }
    }

    /// Kill the child and every process it spawned, then reap the child.
    fn kill_tree(&mut self) {
        #[cfg(unix)]
        {
            // The group must be signalled before `wait` reaps the leader (the pgid lives on
            // with the leader's zombie). TERM first, so uv can drop its temp files, KILL after
            // a grace period.
            let pgid = self.child.id();
            signal_group(pgid, "TERM");
            // A paused (stopped) tree only acts on the TERM once it runs again.
            signal_group(pgid, "CONT");
            let until = Instant::now() + Duration::from_millis(1500);
            while Instant::now() < until {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            signal_group(pgid, "KILL");
        }
        #[cfg(windows)]
        if let Some(j) = &self.job {
            let _ = j.kill();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Windows: close the (now terminated) threads' handles.
        self.resume();
    }
}

impl Tree {
    /// Kill what is left of the tree after the child itself exited (helpers still holding the
    /// output pipes).
    fn kill_orphans(&self) {
        #[cfg(unix)]
        signal_group(self.child.id(), "KILL");
        #[cfg(windows)]
        if let Some(j) = &self.job {
            let _ = j.kill();
        }
    }
}

/// `kill -<signal> -- -<pgid>` through the `kill` binary (no libc binding needed here).
#[cfg(unix)]
fn signal_group(pgid: u32, signal: &str) {
    use std::process::{Command, Stdio};
    let _ = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg("--")
        .arg(format!("-{pgid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
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
            if let Some(status) = tree.child.try_wait()? {
                break status;
            }
        };
        // Drain until both readers hang up (a bounded channel: the readers may still be
        // blocked on `send`), then join them. A helper the child left behind still holding
        // the pipes would keep them open forever: after a grace period the rest of the tree
        // is killed (the group only gets a signal while it has members, so the pgid cannot
        // have been reused).
        let grace = Instant::now() + Duration::from_secs(2);
        let mut orphans_killed = false;
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => keep(line, on_line),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
            if !orphans_killed && Instant::now() >= grace {
                tracing::warn!(
                    "{} left helper processes behind; stopping them",
                    cmd.display()
                );
                tree.kill_orphans();
                orphans_killed = true;
            }
        }
        for r in readers {
            let _ = r.join();
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
        self.manager.enqueue(job);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!(Cancelled));
            }
            match self.manager.status(&key).map(|s| s.state) {
                Some(DownloadState::Installed) => return Ok(()),
                Some(DownloadState::Failed { error, .. }) => {
                    bail!("download of {} failed: {error}", res.title)
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
