//! The one bounded runner for short helper programs (sysctl, netstat, lsof, launchctl,
//! PowerShell, `id`; audit 2026-11): stdout and stderr are drained on threads while waiting so a
//! full pipe never blocks the child, output is capped, the child's process tree is killed at the
//! deadline, and a grandchild that keeps a pipe open cannot hang the caller.
//!
//! - [`run_within`] / [`run_capped`]: stdin null, both outputs piped and read by one thread each
//!   (the first `limit` bytes kept, the rest read and dropped, [`DEFAULT_OUTPUT_LIMIT`] by
//!   default). On Unix the child leads its own process group ([`crate::backend::process::prepare`]);
//!   on Windows it starts with [`CREATE_NO_WINDOW`].
//! - The wait ends when the child exits, the timeout passes or the caller's `abandoned` flag is
//!   set. In the last two cases the child is stopped: on Unix SIGTERM to its group, up to
//!   [`TERM_GRACE`] for the leader to exit, then SIGCONT + SIGKILL to the group
//!   ([`crate::backend::process::kill_group`], so members that ignored the TERM die too) and a
//!   kill of the leader; on Windows `Child::kill` (the direct child only: the tools run here start
//!   no helpers of their own, and a leftover holding the pipe is covered by the bounded join).
//! - After the wait the reader threads get at most [`JOIN_GRACE`] to see the end of their pipe.
//!   A grandchild that inherited the pipe (`sh -c 'sleep 5 & echo x'`) keeps it open; then the
//!   output read so far is returned and the reader thread is left to finish on its own (it holds
//!   at most `limit` bytes and ends when the last writer closes the pipe).
//! - [`euid`]: this process's effective uid through `/usr/bin/id -u` (cached; no `unsafe` outside
//!   `backend/`), the one copy for the service control and the file dialog.
//!
//! The waiting itself (`wait_for`) is written against the small `Waitable` trait, so its
//! stop-on-error / timeout / abandon rules are tested with fakes.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// Windows process creation flag: no console window for a console tool started by the service,
/// the tray or another console-less process. The one copy of the constant.
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Bytes of stdout and of stderr each that [`run_within`] keeps (the rest is read and dropped):
/// every helper's answer fits, and a runaway tool cannot grow the service's memory.
pub const DEFAULT_OUTPUT_LIMIT: usize = 64 * 1024;

/// How long a stopped child's process group has between SIGTERM and SIGKILL (Unix): a dialog
/// closes itself and `sudo` relays the TERM to its child within it.
pub const TERM_GRACE: Duration = Duration::from_secs(2);

/// How long the reader threads may take to reach the end of their pipes once the child has
/// ended. Output a tool wrote before exiting is already in the pipe and is read in
/// microseconds; only a grandchild holding the pipe open makes the wait run out.
pub const JOIN_GRACE: Duration = Duration::from_secs(1);

/// The first and the longest pause between two looks at the child: short tools (`id -u`,
/// `launchctl print`) answer in a few milliseconds, a dialog waits minutes.
const POLL_FIRST: Duration = Duration::from_millis(2);
const POLL_MAX: Duration = Duration::from_millis(100);

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    /// The child exited by itself (any exit code or signal).
    Exited,
    /// The timeout passed; the child was stopped.
    TimedOut,
    /// The caller's `abandoned` flag was set (an HTTP request went away); the child was stopped.
    Abandoned,
}

/// What a run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutput {
    /// Exit code; `None` when the child was stopped, ended by a signal, or still running.
    pub code: Option<i32>,
    /// The first `limit` bytes of stdout.
    pub stdout: Vec<u8>,
    /// The first `limit` bytes of stderr.
    pub stderr: Vec<u8>,
    /// Stdout had more than `limit` bytes (the rest was dropped).
    pub stdout_truncated: bool,
    pub ended: Ended,
}

impl RunOutput {
    /// The child exited by itself with code 0.
    pub fn success(&self) -> bool {
        self.ended == Ended::Exited && self.code == Some(0)
    }

    /// Stdout as text (lossy UTF-8).
    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Stderr as text (lossy UTF-8).
    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// Run `cmd` until it exits, `timeout` passes or `abandoned` is set (then its tree is stopped),
/// keeping [`DEFAULT_OUTPUT_LIMIT`] bytes of each output. `Err` only when it cannot be started
/// or its state cannot be read (it is stopped then); a timeout is `Ok` with
/// [`Ended::TimedOut`]. `cmd`'s stdin, stdout and stderr settings are replaced.
pub fn run_within(
    cmd: Command,
    timeout: Duration,
    abandoned: Option<&AtomicBool>,
) -> std::io::Result<RunOutput> {
    run_capped(cmd, timeout, abandoned, DEFAULT_OUTPUT_LIMIT)
}

/// [`run_within`] keeping `limit` bytes of stdout and of stderr each.
pub fn run_capped(
    cmd: Command,
    timeout: Duration,
    abandoned: Option<&AtomicBool>,
    limit: usize,
) -> std::io::Result<RunOutput> {
    run_capped_with(cmd, timeout, abandoned, limit, |_| Ok(()))
}

/// [`run_capped`] with `guard` called right after the spawn: what it returns is kept until the
/// child has been reaped (a Windows Job Object holding the child's tree, a
/// [`crate::children::Recorded`] entry). A guard that fails stops the child and is the error.
pub fn run_capped_with<G>(
    mut cmd: Command,
    timeout: Duration,
    abandoned: Option<&AtomicBool>,
    limit: usize,
    guard: impl FnOnce(&std::process::Child) -> std::io::Result<G>,
) -> std::io::Result<RunOutput> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    crate::backend::process::prepare(&mut cmd);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn()?;
    let _guard = match guard(&child) {
        Ok(g) => g,
        Err(e) => {
            stop(&mut child);
            return Err(e);
        }
    };
    let out = match child.stdout.take().map(|r| drain(r, limit)).transpose() {
        Ok(d) => d,
        Err(e) => {
            stop(&mut child);
            return Err(e);
        }
    };
    let err = match child.stderr.take().map(|r| drain(r, limit)).transpose() {
        Ok(d) => d,
        Err(e) => {
            stop(&mut child);
            return Err(e);
        }
    };
    let (status, ended) = wait_for(&mut child, timeout, abandoned)?;
    let until = Instant::now() + JOIN_GRACE;
    let (stdout, stdout_truncated) = out.map(|d| d.collect(until)).unwrap_or_default();
    let (stderr, _) = err.map(|d| d.collect(until)).unwrap_or_default();
    Ok(RunOutput {
        code: status.and_then(|s| s.code()),
        stdout,
        stderr,
        stdout_truncated,
        ended,
    })
}

/// What a reader thread has kept so far: `None` once [`Drain::collect`] took it (later bytes
/// are dropped).
#[derive(Debug, Default)]
struct Kept {
    bytes: Option<Vec<u8>>,
    truncated: bool,
}

/// A reader thread on one pipe of the child.
#[derive(Debug)]
struct Drain {
    kept: Arc<Mutex<Kept>>,
    /// Disconnected when the thread has reached the end of the pipe (or stopped on an error).
    done: mpsc::Receiver<()>,
}

/// Read `r` to its end on a new thread, keeping the first `limit` bytes.
fn drain(mut r: impl Read + Send + 'static, limit: usize) -> std::io::Result<Drain> {
    let kept = Arc::new(Mutex::new(Kept {
        bytes: Some(Vec::new()),
        truncated: false,
    }));
    let (tx, done) = mpsc::channel::<()>();
    let shared = kept.clone();
    std::thread::Builder::new()
        .name("command-drain".into())
        .spawn(move || {
            let _tx = tx;
            let mut buf = [0u8; 8192];
            while let Ok(n) = r.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut k = shared.lock().unwrap_or_else(|e| e.into_inner());
                let k = &mut *k;
                if let Some(bytes) = k.bytes.as_mut() {
                    let room = limit.saturating_sub(bytes.len());
                    bytes.extend_from_slice(buf.get(..n.min(room)).unwrap_or_default());
                    k.truncated |= n > room;
                }
            }
        })?;
    Ok(Drain { kept, done })
}

impl Drain {
    /// What was read by `until` (or the end of the pipe, if sooner), and whether bytes past the
    /// limit were dropped. A thread still reading then is left to finish alone.
    fn collect(self, until: Instant) -> (Vec<u8>, bool) {
        let _ = self
            .done
            .recv_timeout(until.saturating_duration_since(Instant::now()));
        let mut k = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        (k.bytes.take().unwrap_or_default(), k.truncated)
    }
}

/// Stop a running child (see the module doc): Unix SIGTERM to its process group, up to
/// [`TERM_GRACE`] for the leader, then SIGCONT + SIGKILL to the group; then a kill of the child
/// and a wait for it (it is never left a zombie).
///
/// The group is killed even when the leader exited during the grace (it was reaped then), so a
/// member that ignored the TERM dies too. Its id cannot name another group while one of its
/// members lives; with none left the call finds nothing (`ESRCH`), short of the PID being reused
/// as a new group's id within those two seconds.
fn stop(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        use crate::backend::process::{Signal, kill_group, signal_group};
        let group = child.id();
        let _ = signal_group(group, Signal::Term);
        let until = Instant::now() + TERM_GRACE;
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = kill_group(group);
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// A running child as [`wait_for`] sees it (a fake in the tests).
trait Waitable {
    type Status;
    fn try_wait(&mut self) -> std::io::Result<Option<Self::Status>>;
    fn stop(&mut self);
}

impl Waitable for std::process::Child {
    type Status = std::process::ExitStatus;
    fn try_wait(&mut self) -> std::io::Result<Option<Self::Status>> {
        std::process::Child::try_wait(self)
    }
    fn stop(&mut self) {
        stop(self);
    }
}

/// Wait until `p` exits, `timeout` passes or `abandoned` is set; in the last two cases, and
/// when its state cannot be read, the process is stopped (it is never left running).
fn wait_for<W: Waitable>(
    p: &mut W,
    timeout: Duration,
    abandoned: Option<&AtomicBool>,
) -> std::io::Result<(Option<W::Status>, Ended)> {
    let deadline = Instant::now() + timeout;
    let mut pause = POLL_FIRST;
    loop {
        match p.try_wait() {
            Ok(Some(s)) => return Ok((Some(s), Ended::Exited)),
            Ok(None) => {}
            Err(e) => {
                p.stop();
                return Err(e);
            }
        }
        if abandoned.is_some_and(|a| a.load(Ordering::Relaxed)) {
            p.stop();
            return Ok((None, Ended::Abandoned));
        }
        let now = Instant::now();
        if now >= deadline {
            p.stop();
            return Ok((None, Ended::TimedOut));
        }
        std::thread::sleep(pause.min(deadline.saturating_duration_since(now)));
        pause = (pause * 2).min(POLL_MAX);
    }
}

/// `name` (e.g. `netstat.exe`) in `System32` of the Windows folder (`SystemRoot`, else
/// `C:\Windows`): by absolute path, so a same-named program on the search path or in the
/// working folder never runs instead. Built on every OS (tests).
pub fn system32(name: &str) -> std::path::PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    std::path::Path::new(&root).join("System32").join(name)
}

/// How long `id -u` may take.
#[cfg(unix)]
const ID_TIMEOUT: Duration = Duration::from_secs(2);

/// This process's effective uid, from `/usr/bin/id -u` (no `unsafe` outside `backend/`): asked
/// once per process, `None` when `id` cannot run, fails or does not answer in time.
#[cfg(unix)]
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux")),
    allow(
        dead_code,
        reason = "used by the macOS / Linux service control and dialog"
    )
)]
pub fn euid() -> Option<u32> {
    static EUID: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *EUID.get_or_init(|| {
        let mut cmd = Command::new("/usr/bin/id");
        cmd.arg("-u");
        match run_within(cmd, ID_TIMEOUT, None) {
            Ok(out) => parse_uid(&out),
            Err(e) => {
                tracing::debug!("could not run /usr/bin/id -u: {e}");
                None
            }
        }
    })
}

/// The uid printed by a successful `id -u`.
#[cfg_attr(not(unix), allow(dead_code, reason = "Unix only; tested everywhere"))]
fn parse_uid(out: &RunOutput) -> Option<u32> {
    if !out.success() {
        return None;
    }
    out.stdout_text().trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn sh(script: &str) -> Command {
        let mut c = Command::new("/bin/sh");
        c.args(["-c", script]);
        c
    }

    /// Whether process `pid` exists (`kill -0`), reaped or not.
    #[cfg(unix)]
    fn alive(pid: u32) -> bool {
        Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// Poll `alive(pid)` until it is false or `within` passes.
    #[cfg(unix)]
    fn gone_within(pid: u32, within: Duration) -> bool {
        let until = Instant::now() + within;
        while Instant::now() < until {
            if !alive(pid) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        !alive(pid)
    }

    #[cfg(unix)]
    #[test]
    fn small_output_exit_code_and_stderr() {
        let out = run_within(
            sh("printf 'out\\n'; printf 'err' >&2; exit 0"),
            Duration::from_secs(10),
            None,
        )
        .unwrap();
        assert_eq!(out.ended, Ended::Exited);
        assert!(out.success());
        assert_eq!(out.stdout_text(), "out\n");
        assert_eq!(out.stderr_text(), "err");
        assert!(!out.stdout_truncated);
        let out = run_within(sh("exit 7"), Duration::from_secs(10), None).unwrap();
        assert_eq!((out.ended, out.code), (Ended::Exited, Some(7)));
        assert!(!out.success());
    }

    /// The guard lives until the child is reaped; a failing guard stops the child and is the
    /// error.
    #[cfg(unix)]
    #[test]
    fn the_guard_lives_until_the_child_is_reaped() {
        struct Flag(Arc<AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let seen = Arc::new(Mutex::new(None::<u32>));
        let (d, s) = (dropped.clone(), seen.clone());
        let out = run_capped_with(
            sh("sleep 0.3; echo done"),
            Duration::from_secs(10),
            None,
            DEFAULT_OUTPUT_LIMIT,
            move |child| {
                *s.lock().unwrap() = Some(child.id());
                Ok(Flag(d))
            },
        )
        .unwrap();
        assert_eq!(out.stdout_text(), "done\n");
        assert!(
            dropped.load(Ordering::SeqCst),
            "guard not dropped after the run"
        );
        assert!(seen.lock().unwrap().is_some());

        let pid = Arc::new(Mutex::new(0u32));
        let p = pid.clone();
        let t = Instant::now();
        let e = run_capped_with(
            sh("sleep 30"),
            Duration::from_secs(60),
            None,
            DEFAULT_OUTPUT_LIMIT,
            move |child| -> std::io::Result<()> {
                *p.lock().unwrap() = child.id();
                Err(std::io::Error::other("no job object"))
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("no job object"), "{e}");
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "the child was waited for"
        );
        let pid = *pid.lock().unwrap();
        assert!(
            gone_within(pid, Duration::from_secs(5)),
            "child {pid} survived"
        );
    }

    #[test]
    fn a_program_that_cannot_start_is_an_error() {
        let e = run_within(
            Command::new("/nonexistent/blue-onyx-prism-no-such-tool"),
            Duration::from_secs(1),
            None,
        )
        .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
    }

    /// More than a pipe buffer (and more than 1 MiB) is drained while waiting: the child exits,
    /// its code is kept and the output is capped at the limit.
    #[cfg(unix)]
    #[test]
    fn large_output_completes_and_is_capped() {
        let t = Instant::now();
        let out = run_within(
            sh("yes | head -c 2000000; exit 3"),
            Duration::from_secs(20),
            None,
        )
        .unwrap();
        assert_eq!(out.ended, Ended::Exited);
        assert_eq!(out.code, Some(3));
        assert_eq!(out.stdout.len(), DEFAULT_OUTPUT_LIMIT);
        assert!(out.stdout_truncated);
        assert!(out.stdout.starts_with(b"y\ny\n"));
        assert!(t.elapsed() < Duration::from_secs(15), "{:?}", t.elapsed());
        // The cap is a parameter, and output under it is kept whole.
        let out = run_capped(
            sh("head -c 200000 /dev/zero; head -c 300000 /dev/zero >&2"),
            Duration::from_secs(20),
            None,
            1024 * 1024,
        )
        .unwrap();
        assert_eq!((out.stdout.len(), out.stderr.len()), (200_000, 300_000));
        assert!(!out.stdout_truncated);
        let out = run_capped(sh("printf abcdef"), Duration::from_secs(10), None, 4).unwrap();
        assert_eq!(out.stdout, b"abcd");
        assert!(out.stdout_truncated);
    }

    #[cfg(windows)]
    #[test]
    fn large_output_completes_and_is_capped_on_windows() {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new(system32("cmd.exe"));
        c.raw_arg("/d /c for /L %i in (1,1,20000) do @echo xxxxxxxx");
        let t = Instant::now();
        let out = run_within(c, Duration::from_secs(60), None).unwrap();
        assert_eq!(out.ended, Ended::Exited);
        assert_eq!(out.code, Some(0));
        assert_eq!(out.stdout.len(), DEFAULT_OUTPUT_LIMIT);
        assert!(out.stdout_truncated);
        assert!(t.elapsed() < Duration::from_secs(50));
        let mut c = Command::new(system32("cmd.exe"));
        c.raw_arg("/d /c ping -n 30 127.0.0.1 >NUL");
        let t = Instant::now();
        let out = run_within(c, Duration::from_millis(300), None).unwrap();
        assert_eq!(out.ended, Ended::TimedOut);
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    /// A timeout stops the whole process group: the grandchild is gone afterwards too, and the
    /// output written before the deadline is kept.
    #[cfg(unix)]
    #[test]
    fn a_timeout_kills_the_group() {
        let t = Instant::now();
        let out = run_within(
            sh("sleep 60 & echo $!; sleep 60"),
            Duration::from_millis(500),
            None,
        )
        .unwrap();
        assert_eq!(out.ended, Ended::TimedOut);
        assert_eq!(out.code, None);
        assert!(
            t.elapsed()
                < Duration::from_millis(500) + TERM_GRACE + JOIN_GRACE + Duration::from_secs(2),
            "{:?}",
            t.elapsed()
        );
        let grandchild: u32 = out.stdout_text().trim().parse().unwrap();
        assert!(
            gone_within(grandchild, Duration::from_secs(5)),
            "grandchild {grandchild} survived the timeout"
        );
    }

    /// A member that ignores SIGTERM is killed after the grace.
    #[cfg(unix)]
    #[test]
    fn a_member_ignoring_term_is_killed() {
        let out = run_within(
            sh("(trap '' TERM; sleep 60) & echo $!; sleep 60"),
            Duration::from_millis(300),
            None,
        )
        .unwrap();
        assert_eq!(out.ended, Ended::TimedOut);
        let pid: u32 = out.stdout_text().trim().parse().unwrap();
        assert!(gone_within(pid, Duration::from_secs(5)), "{pid} survived");
    }

    /// A grandchild that keeps stdout open after the child exited does not hang the caller: the
    /// output so far comes back after the bounded join.
    #[cfg(unix)]
    #[test]
    fn a_grandchild_holding_the_pipe_does_not_hang() {
        let t = Instant::now();
        let out = run_within(
            sh("sleep 5 & echo $!; echo started"),
            Duration::from_secs(10),
            None,
        )
        .unwrap();
        let took = t.elapsed();
        assert_eq!(out.ended, Ended::Exited);
        assert_eq!(out.code, Some(0));
        let text = out.stdout_text();
        let mut lines = text.lines();
        let pid: u32 = lines.next().unwrap().parse().unwrap();
        assert_eq!(lines.next(), Some("started"));
        assert!(took < JOIN_GRACE + Duration::from_secs(2), "{took:?}");
        assert!(took >= JOIN_GRACE, "the join waited for the pipe: {took:?}");
        let _ = Command::new("/bin/kill").arg(pid.to_string()).status();
    }

    /// Setting the abandoned flag stops the child.
    #[cfg(unix)]
    #[test]
    fn the_abandoned_flag_stops_it() {
        let flag = Arc::new(AtomicBool::new(false));
        let setter = {
            let flag = flag.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                flag.store(true, Ordering::Relaxed);
            })
        };
        let t = Instant::now();
        let out = run_within(sh("sleep 60"), Duration::from_secs(60), Some(&flag)).unwrap();
        setter.join().unwrap();
        assert_eq!(out.ended, Ended::Abandoned);
        assert!(t.elapsed() < Duration::from_secs(10));
        // Already set: stopped at the first look.
        let set = AtomicBool::new(true);
        let out = run_within(sh("sleep 60"), Duration::from_secs(60), Some(&set)).unwrap();
        assert_eq!(out.ended, Ended::Abandoned);
    }

    /// A process whose state cannot be read is stopped before the error is returned.
    #[test]
    fn wait_errors_stop_the_process() {
        struct FakeProc {
            calls: usize,
            fail_at: usize,
            stopped: bool,
        }
        impl Waitable for FakeProc {
            type Status = i32;
            fn try_wait(&mut self) -> std::io::Result<Option<i32>> {
                self.calls += 1;
                if self.calls >= self.fail_at {
                    Err(std::io::Error::other("waitpid failed"))
                } else {
                    Ok(None)
                }
            }
            fn stop(&mut self) {
                self.stopped = true;
            }
        }
        let never = AtomicBool::new(false);
        for fail_at in [1, 2] {
            let mut p = FakeProc {
                calls: 0,
                fail_at,
                stopped: false,
            };
            let e = wait_for(&mut p, Duration::from_secs(60), Some(&never)).unwrap_err();
            assert_eq!(e.to_string(), "waitpid failed");
            assert!(p.stopped, "stopped after a try_wait error ({fail_at})");
        }
        // An exit is not stopped; a timeout is; an abandon is.
        struct Exits(bool);
        impl Waitable for Exits {
            type Status = i32;
            fn try_wait(&mut self) -> std::io::Result<Option<i32>> {
                Ok(Some(0))
            }
            fn stop(&mut self) {
                self.0 = true;
            }
        }
        let mut p = Exits(false);
        assert_eq!(
            wait_for(&mut p, Duration::from_secs(60), Some(&never)).unwrap(),
            (Some(0), Ended::Exited)
        );
        assert!(!p.0);
        let mut p = FakeProc {
            calls: 0,
            fail_at: usize::MAX,
            stopped: false,
        };
        assert_eq!(
            wait_for(&mut p, Duration::ZERO, None).unwrap(),
            (None, Ended::TimedOut)
        );
        assert!(p.stopped);
        let gone = AtomicBool::new(true);
        let mut p = FakeProc {
            calls: 0,
            fail_at: usize::MAX,
            stopped: false,
        };
        assert_eq!(
            wait_for(&mut p, Duration::from_secs(60), Some(&gone)).unwrap(),
            (None, Ended::Abandoned)
        );
        assert!(p.stopped);
    }

    #[test]
    fn uid_needs_a_clean_exit_and_a_number() {
        let out = |code, ended, text: &str| RunOutput {
            code,
            stdout: text.as_bytes().to_vec(),
            stderr: Vec::new(),
            stdout_truncated: false,
            ended,
        };
        assert_eq!(parse_uid(&out(Some(0), Ended::Exited, "501\n")), Some(501));
        assert_eq!(parse_uid(&out(Some(0), Ended::Exited, "0\n")), Some(0));
        assert_eq!(parse_uid(&out(Some(1), Ended::Exited, "0\n")), None);
        assert_eq!(parse_uid(&out(None, Ended::TimedOut, "0")), None);
        assert_eq!(parse_uid(&out(Some(0), Ended::Exited, "root")), None);
    }

    #[cfg(unix)]
    #[test]
    fn euid_is_this_process_uid() {
        let want = Command::new("/usr/bin/id").arg("-u").output().unwrap();
        let want: u32 = String::from_utf8_lossy(&want.stdout)
            .trim()
            .parse()
            .unwrap();
        assert_eq!(euid(), Some(want));
    }

    #[test]
    fn system32_is_under_the_windows_folder() {
        let p = system32("netstat.exe");
        assert!(p.ends_with(std::path::Path::new("System32").join("netstat.exe")));
        assert!(p.is_absolute() || !cfg!(windows), "{}", p.display());
    }
}
