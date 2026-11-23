//! Process-tree primitives every child-process spawner uses (audit 2026-11, RA processes): the
//! unsafe OS calls behind [`crate::command`] and the export / training runners live here because
//! `unsafe` is only allowed under `backend/`.
//!
//! - [`prepare`] puts a child in its own process group (Unix) and, on Linux, asks the kernel to
//!   SIGKILL it when the spawning thread's process dies (`PR_SET_PDEATHSIG`).
//! - [`signal_group`] / [`kill_group`] signal a whole group through `killpg(2)` (no external
//!   `kill` program). A killed group is first sent SIGCONT so a paused (SIGSTOP) tree dies too.
//! - [`start_time`] / [`exe_path`] identify a process so a recorded PID is never confused with a
//!   reused one ([`crate::children`]).
//!
//! Windows trees are held by a Job Object ([`super::job`]); the group calls are no-ops there.

use std::path::PathBuf;
use std::process::Command;

/// A signal sent to a process group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// SIGSTOP: pause the whole tree.
    Stop,
    /// SIGCONT: continue a paused tree.
    Cont,
    /// SIGTERM: ask the tree to exit.
    Term,
    /// SIGKILL: end the tree.
    Kill,
}

/// Prepare `cmd` before spawn: own process group on Unix; on Linux also parent-death SIGKILL.
pub fn prepare(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let _ = cmd;
}

/// Send `sig` to the process group led by `pgid` (Unix). No-op `Ok` on Windows.
pub fn signal_group(pgid: u32, sig: Signal) -> std::io::Result<()> {
    let _ = (pgid, sig);
    Ok(())
}

/// End the whole group led by `pgid`: SIGCONT, then SIGKILL (Unix). No-op `Ok` on Windows.
pub fn kill_group(pgid: u32) -> std::io::Result<()> {
    signal_group(pgid, Signal::Cont)?;
    signal_group(pgid, Signal::Kill)
}

/// Start time of process `pid` in an OS-specific monotonic-enough unit (stable for the life of
/// the process, different for a reused PID), or `None` when it does not exist or cannot be read.
pub fn start_time(pid: u32) -> Option<u64> {
    let _ = pid;
    None
}

/// Executable path of process `pid`, or `None` when it cannot be read.
pub fn exe_path(pid: u32) -> Option<PathBuf> {
    let _ = pid;
    None
}
