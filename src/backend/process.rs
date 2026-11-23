//! Process-tree primitives every child-process spawner uses (audit 2026-11, RA processes): the
//! unsafe OS calls behind [`crate::command`], [`crate::children`] and the export / training
//! runners live here because `unsafe` is only allowed under `backend/`.
//!
//! - [`prepare`] puts a child in its own process group (Unix) and, on Linux, asks the kernel to
//!   SIGKILL it when its parent dies (`PR_SET_PDEATHSIG`).
//! - [`signal_group`] / [`kill_group`] signal a whole group through `killpg(2)` (no external
//!   `kill` program). A killed group is first sent SIGCONT so a paused (SIGSTOP) tree dies too.
//!   [`kill_pid`] ends one process (CONT + KILL on Unix, `TerminateProcess` on Windows).
//! - [`start_time`] / [`exe_path`] identify a process so a recorded PID is never confused with a
//!   reused one ([`crate::children`]): Linux `/proc/<pid>/stat` field 22 and `/proc/<pid>/exe`,
//!   macOS `proc_pidinfo(PROC_PIDTBSDINFO)` and `proc_pidpath`, Windows `GetProcessTimes` and
//!   `QueryFullProcessImageNameW`.
//!
//! Safety rails: a PID or process group of 0 or 1, one that does not fit `pid_t`, this process
//! itself or this process's own group is refused (`InvalidInput`): `kill(0, ..)` / `killpg(0, ..)`
//! would signal our own group and `kill(-1, ..)` every process we may signal. A group or process
//! that no longer exists (`ESRCH`) is `Ok`: it is already what the caller wanted.
//!
//! **Linux caveat (`PR_SET_PDEATHSIG`).** The kernel sends the death signal when the *thread*
//! that forked the child exits, not when the whole process does. A child prepared here must
//! therefore be spawned from a thread that outlives it: the runners in [`crate::command`] and
//! `resources::export` wait on the child in the thread that spawned it, which is the intended use.
//! Never spawn a prepared child from a pool thread that may retire while the child runs (tokio's
//! `spawn_blocking` threads exit after an idle timeout): the child would be killed with it.
//! The death signal is also cleared by the kernel when the child execs a set-user-ID binary.
//!
//! Windows trees are held by a Job Object (`backend::job`, Windows only); the group calls are no-ops there.

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

#[cfg(unix)]
impl Signal {
    /// The libc signal number.
    fn raw(self) -> libc::c_int {
        match self {
            Signal::Stop => libc::SIGSTOP,
            Signal::Cont => libc::SIGCONT,
            Signal::Term => libc::SIGTERM,
            Signal::Kill => libc::SIGKILL,
        }
    }
}

/// Prepare `cmd` before spawn: own process group on Unix; on Linux also SIGKILL when the parent
/// dies (see the module doc's caveat: spawn from a thread that outlives the child).
pub fn prepare(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(target_os = "linux")]
    linux_parent_death(cmd);
    let _ = cmd;
}

/// `PR_SET_PDEATHSIG(SIGKILL)` in the forked child, then a check that the parent did not die
/// before the call (the child would then have been reparented and never get the signal).
#[cfg(target_os = "linux")]
fn linux_parent_death(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // Captured in the parent, before fork: the child compares its parent with it.
    let parent = libc::pid_t::try_from(std::process::id()).ok();
    let sig = libc::c_ulong::try_from(libc::SIGKILL).unwrap_or(9);
    // Runs in the forked child before exec: only async-signal-safe calls (`prctl`, `getppid`,
    // `_exit`; `last_os_error` only reads errno), no lock, no allocation; captures only `Copy`
    // integers.
    let hook = move || -> std::io::Result<()> {
        // SAFETY: PR_SET_PDEATHSIG takes one unsigned long signal number; the remaining variadic
        // arguments are unused by this option.
        if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, sig) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        if let Some(parent) = parent {
            // SAFETY: getppid never fails and is async-signal-safe.
            if unsafe { libc::getppid() } != parent {
                // The parent died between fork and prctl: we were reparented and the death
                // signal will never come.
                // SAFETY: _exit is async-signal-safe and ends only this child.
                unsafe { libc::_exit(1) };
            }
        }
        Ok(())
    };
    // SAFETY: `hook` is async-signal-safe (see above), as `pre_exec` requires.
    unsafe {
        cmd.pre_exec(hook);
    }
}

/// A PID / process group the signal calls accept: > 1, fits `pid_t`, not this process.
#[cfg(unix)]
fn checked_pid(pid: u32) -> std::io::Result<libc::pid_t> {
    let refuse = |why: &str| {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to signal process {pid}: {why} (a caller bug; nothing was sent)"),
        ))
    };
    let Ok(p) = libc::pid_t::try_from(pid) else {
        return refuse("not a valid process id");
    };
    if p <= 1 {
        return refuse("0 and 1 would signal our own group or init");
    }
    if pid == std::process::id() {
        return refuse("it is this process");
    }
    Ok(p)
}

/// `Ok` for a call that returned 0 or failed with `ESRCH` (no such process / group).
#[cfg(unix)]
fn signalled(rc: libc::c_int) -> std::io::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let e = std::io::Error::last_os_error();
    if e.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(e)
    }
}

/// Send `sig` to the process group led by `pgid` (Unix). A group that no longer exists is `Ok`;
/// this process's own group is refused. No-op `Ok` on Windows (the Job Object holds the tree).
pub fn signal_group(pgid: u32, sig: Signal) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let pg = checked_pid(pgid)?;
        // SAFETY: getpgrp never fails and has no preconditions.
        if pg == unsafe { libc::getpgrp() } {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to signal process group {pgid}: it is this process's own group \
                     (the child was not spawned through backend::process::prepare)"
                ),
            ));
        }
        // SAFETY: killpg only sends a signal; `pg` is a checked group id > 1 that is not ours.
        signalled(unsafe { libc::killpg(pg, sig.raw()) })
    }
    #[cfg(not(unix))]
    {
        let _ = (pgid, sig);
        Ok(())
    }
}

/// End the whole group led by `pgid`: SIGCONT (a stopped tree must run to die), then SIGKILL
/// (Unix). No-op `Ok` on Windows.
pub fn kill_group(pgid: u32) -> std::io::Result<()> {
    signal_group(pgid, Signal::Cont)?;
    signal_group(pgid, Signal::Kill)
}

/// End one process: SIGCONT then SIGKILL on Unix, `TerminateProcess` (exit code 1) on Windows.
/// A process that no longer exists is `Ok`; this process itself is refused. Callers that kill a
/// recorded PID must first prove it is still the same process ([`start_time`], [`exe_path`]).
pub fn kill_pid(pid: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let p = checked_pid(pid)?;
        // SAFETY: kill only sends a signal to the checked pid (> 1, not this process).
        signalled(unsafe { libc::kill(p, libc::SIGCONT) })?;
        // SAFETY: as above.
        signalled(unsafe { libc::kill(p, libc::SIGKILL) })
    }
    #[cfg(windows)]
    {
        windows_kill(pid)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }
}

#[cfg(windows)]
fn windows_kill(pid: u32) -> std::io::Result<()> {
    use windows::Win32::Foundation::{CloseHandle, ERROR_INVALID_PARAMETER};
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};
    if pid == 0 || pid == std::process::id() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to end process {pid}: it is this process or the idle process"),
        ));
    }
    // SAFETY: plain open by id; the handle is closed below.
    let h = match unsafe { OpenProcess(PROCESS_TERMINATE, false, pid) } {
        Ok(h) => h,
        // No such process: already gone.
        Err(e) if e.code() == ERROR_INVALID_PARAMETER.to_hresult() => return Ok(()),
        Err(e) => return Err(std::io::Error::other(format!("OpenProcess({pid}): {e}"))),
    };
    // SAFETY: `h` was opened with PROCESS_TERMINATE above and stays open for the call.
    let r = unsafe { TerminateProcess(h, 1) };
    // SAFETY: `h` is closed exactly once.
    let _ = unsafe { CloseHandle(h) };
    r.map_err(|e| std::io::Error::other(format!("TerminateProcess({pid}): {e}")))
}

/// Start time of process `pid` in an OS-specific unit (Linux: clock ticks since boot; macOS:
/// microseconds since the epoch; Windows: FILETIME 100 ns units), stable for the life of the
/// process and different for a reused PID, or `None` when it does not exist (Windows: has
/// exited) or cannot be read.
pub fn start_time(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        parse_stat_start(&stat)
    }
    #[cfg(target_os = "macos")]
    {
        macos_start_time(pid)
    }
    #[cfg(windows)]
    {
        windows_start_time(pid)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = pid;
        None
    }
}

/// Field 22 (`starttime`) of a `/proc/<pid>/stat` line. The command name (field 2) is in
/// parentheses and may itself hold spaces and `)`, so the fields are counted after the last `)`.
#[cfg_attr(
    not(target_os = "linux"),
    allow(dead_code, reason = "used on Linux and in tests")
)]
fn parse_stat_start(stat: &str) -> Option<u64> {
    let (_, rest) = stat.rsplit_once(')')?;
    // `rest` starts at field 3 (state), so field 22 is the 20th.
    rest.split_whitespace().nth(22 - 3)?.parse().ok()
}

#[cfg(target_os = "macos")]
fn macos_start_time(pid: u32) -> Option<u64> {
    let pid = libc::c_int::try_from(pid).ok()?;
    let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>()).ok()?;
    let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::zeroed();
    // SAFETY: `info` is live, writable and `size` bytes long, the size of the flavor's struct;
    // the kernel writes at most that many bytes during the call.
    let n = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    if n != size {
        return None;
    }
    // SAFETY: the struct is plain integers, zeroed above and filled in by the successful call.
    let info = unsafe { info.assume_init() };
    info.pbi_start_tvsec
        .checked_mul(1_000_000)?
        .checked_add(info.pbi_start_tvusec)
}

/// A process handle closed on drop.
#[cfg(windows)]
struct OwnedProcess(windows::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl OwnedProcess {
    fn open(
        pid: u32,
        access: windows::Win32::System::Threading::PROCESS_ACCESS_RIGHTS,
    ) -> Option<Self> {
        // SAFETY: plain open by id; the handle is owned by the returned value.
        unsafe { windows::Win32::System::Threading::OpenProcess(access, false, pid) }
            .ok()
            .map(OwnedProcess)
    }
}

#[cfg(windows)]
impl Drop for OwnedProcess {
    fn drop(&mut self) {
        // SAFETY: the handle came from OpenProcess and is closed exactly once.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.0) };
    }
}

#[cfg(windows)]
fn windows_start_time(pid: u32) -> Option<u64> {
    use windows::Win32::Foundation::{FILETIME, WAIT_TIMEOUT};
    use windows::Win32::System::Threading::{
        GetProcessTimes, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
        WaitForSingleObject,
    };
    let h = OwnedProcess::open(pid, PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE)?;
    // An exited process stays openable while any handle to it is open: it is not running.
    // SAFETY: `h.0` is a valid process handle with SYNCHRONIZE access; a zero timeout only polls.
    if unsafe { WaitForSingleObject(h.0, 0) } != WAIT_TIMEOUT {
        return None;
    }
    let (mut created, mut exited, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // SAFETY: `h.0` is valid with PROCESS_QUERY_LIMITED_INFORMATION; the four FILETIMEs are live
    // and writable for the call.
    unsafe { GetProcessTimes(h.0, &mut created, &mut exited, &mut kernel, &mut user) }.ok()?;
    Some((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime))
}

/// Executable path of process `pid`, or `None` when it cannot be read (gone, a zombie on
/// Linux, or not ours to inspect). Linux reports a replaced binary with a ` (deleted)` suffix,
/// which then no longer matches a recorded path (the safe direction).
pub fn exe_path(pid: u32) -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_link(format!("/proc/{pid}/exe")).ok()
    }
    #[cfg(target_os = "macos")]
    {
        macos_exe_path(pid)
    }
    #[cfg(windows)]
    {
        windows_exe_path(pid)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = pid;
        None
    }
}

#[cfg(target_os = "macos")]
fn macos_exe_path(pid: u32) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let pid = libc::c_int::try_from(pid).ok()?;
    let mut buf = vec![0u8; usize::try_from(libc::PROC_PIDPATHINFO_MAXSIZE).ok()?];
    let cap = u32::try_from(buf.len()).ok()?;
    // SAFETY: `buf` is live and writable for `cap` bytes; the call writes at most that many.
    let n = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr().cast(), cap) };
    let n = usize::try_from(n).ok().filter(|&n| n > 0)?;
    let bytes = buf.get(..n)?;
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(windows)]
fn windows_exe_path(pid: u32) -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows::Win32::System::Threading::{
        PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
    };
    let h = OwnedProcess::open(pid, PROCESS_QUERY_LIMITED_INFORMATION)?;
    // The longest path Windows allows (\\?\ paths), in UTF-16 units.
    let mut buf = vec![0u16; 32_768];
    let mut len = u32::try_from(buf.len()).ok()?;
    // SAFETY: `h.0` is valid with PROCESS_QUERY_LIMITED_INFORMATION; `buf` is live and writable
    // for `len` units, and `len` is updated to the characters written (without the NUL).
    unsafe {
        QueryFullProcessImageNameW(
            h.0,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut len,
        )
    }
    .ok()?;
    let wide = buf.get(..usize::try_from(len).ok()?)?;
    Some(PathBuf::from(std::ffi::OsString::from_wide(wide)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_start_is_field_22_counted_after_the_last_paren() {
        // Fields 3.. after the name; starttime (22) = 987654.
        let tail = "S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 987654 20 21";
        assert_eq!(
            parse_stat_start(&format!("42 (sleep) {tail}")),
            Some(987_654)
        );
        // A name holding spaces and parentheses.
        assert_eq!(
            parse_stat_start(&format!("42 (a) (b c)) {tail}")),
            Some(987_654)
        );
        assert_eq!(parse_stat_start("42 (sleep) S 1 2"), None);
        assert_eq!(parse_stat_start("no parens at all"), None);
        assert_eq!(
            parse_stat_start("42 (x) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 nan"),
            None
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn exe_path_of_self_is_the_current_exe() {
        let canon = |p: PathBuf| std::fs::canonicalize(&p).unwrap_or(p);
        let mine = exe_path(std::process::id()).expect("own exe path");
        let current = std::env::current_exe().expect("current_exe");
        assert_eq!(canon(mine), canon(current));
    }

    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn start_time_of_self_is_stable() {
        let a = start_time(std::process::id()).expect("own start time");
        assert_eq!(start_time(std::process::id()), Some(a));
    }

    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn absent_process_has_no_identity() {
        // Above any pid_max (Linux 2^22, macOS 99999); Windows ids are multiples of 4.
        let pid = 4_000_001;
        assert_eq!(start_time(pid), None);
        assert_eq!(exe_path(pid), None);
    }

    #[cfg(unix)]
    fn spawn(script: &str) -> std::process::Child {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg(script);
        cmd.stdin(std::process::Stdio::null());
        prepare(&mut cmd);
        cmd.spawn().expect("spawn sh")
    }

    /// Whether `pid` still exists as a process that is not a zombie.
    #[cfg(unix)]
    fn running(pid: u32) -> bool {
        let out = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .expect("ps");
        let stat = String::from_utf8_lossy(&out.stdout);
        let stat = stat.trim();
        !stat.is_empty() && !stat.starts_with('Z')
    }

    #[cfg(unix)]
    fn wait_gone(pid: u32) -> bool {
        (0..100).any(|_| {
            if running(pid) {
                std::thread::sleep(std::time::Duration::from_millis(50));
                false
            } else {
                true
            }
        })
    }

    /// Pids of the processes in group `pgid` other than the leader.
    #[cfg(unix)]
    fn members(pgid: u32) -> Vec<u32> {
        let out = Command::new("ps")
            .args(["-A", "-o", "pid=,pgid="])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let pid: u32 = it.next()?.parse().ok()?;
                let pg: u32 = it.next()?.parse().ok()?;
                (pg == pgid && pid != pgid).then_some(pid)
            })
            .collect()
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn child_identity_is_stable_and_differs_from_ours() {
        let mut child = spawn("exec sleep 30");
        let pid = child.id();
        let a = start_time(pid).expect("child start time");
        // `spawn` returns once `sh` runs; its `exec sleep` comes a moment later (on Linux the
        // exe still reads `dash` right after spawn). Exec keeps the process, so the start time
        // must not change while the exe does.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut exe = exe_path(pid).expect("child exe");
        while exe.file_name().and_then(|n| n.to_str()) != Some("sleep")
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
            exe = exe_path(pid).expect("child exe");
        }
        assert_eq!(start_time(pid), Some(a));
        assert_ne!(Some(a), start_time(std::process::id()));
        assert_eq!(
            exe.file_name().and_then(|n| n.to_str()),
            Some("sleep"),
            "{exe:?}"
        );
        kill_group(pid).expect("kill");
        let _ = child.wait();
    }

    #[cfg(unix)]
    #[test]
    fn kill_group_ends_the_leader_and_its_children() {
        let mut child = spawn("sleep 30 & sleep 30");
        let pid = child.id();
        let mut kids = Vec::new();
        for _ in 0..100 {
            kids = members(pid);
            if !kids.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!kids.is_empty(), "background sleep not seen");
        kill_group(pid).expect("kill_group");
        let status = child.wait().expect("wait");
        assert!(!status.success(), "leader was not killed: {status}");
        for k in kids {
            assert!(wait_gone(k), "group member {k} survived");
        }
    }

    #[cfg(unix)]
    #[test]
    fn kill_group_ends_a_stopped_group() {
        let mut child = spawn("sleep 30 & wait");
        let pid = child.id();
        let mut kids = Vec::new();
        for _ in 0..100 {
            kids = members(pid);
            if !kids.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        signal_group(pid, Signal::Stop).expect("stop");
        kill_group(pid).expect("kill_group");
        let status = child.wait().expect("wait");
        assert!(!status.success());
        for k in kids {
            assert!(wait_gone(k), "stopped member {k} survived");
        }
    }

    #[cfg(unix)]
    #[test]
    fn signalling_a_gone_group_or_pid_is_ok() {
        let mut child = spawn("exit 0");
        let pid = child.id();
        child.wait().expect("wait");
        signal_group(pid, Signal::Term).expect("gone group is Ok");
        kill_group(pid).expect("gone group is Ok");
        kill_pid(pid).expect("gone pid is Ok");
    }

    #[cfg(unix)]
    #[test]
    fn kill_pid_ends_a_stopped_process() {
        let mut child = spawn("exec sleep 30");
        let pid = child.id();
        signal_group(pid, Signal::Stop).expect("stop");
        kill_pid(pid).expect("kill_pid");
        let status = child.wait().expect("wait");
        assert!(!status.success());
    }

    #[cfg(unix)]
    #[test]
    fn own_process_group_and_reserved_ids_are_refused() {
        let me = std::process::id();
        for pid in [0, 1, me, u32::MAX] {
            let e = kill_pid(pid).expect_err("refused");
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{pid}: {e}");
        }
        for pg in [0, 1, u32::MAX] {
            let e = signal_group(pg, Signal::Cont).expect_err("refused");
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{pg}: {e}");
        }
        // SAFETY: getpgrp has no preconditions.
        let own = u32::try_from(unsafe { libc::getpgrp() }).expect("pgrp");
        if own > 1 {
            let e = signal_group(own, Signal::Cont).expect_err("own group refused");
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn kill_pid_ends_a_child_and_refuses_self() {
        let mut child = Command::new("cmd")
            .args(["/C", "ping -n 30 127.0.0.1 >NUL"])
            .spawn()
            .expect("spawn");
        let pid = child.id();
        assert!(start_time(pid).is_some());
        kill_pid(pid).expect("kill_pid");
        let _ = child.wait();
        assert_eq!(start_time(pid), None);
        assert!(kill_pid(std::process::id()).is_err());
    }
}
