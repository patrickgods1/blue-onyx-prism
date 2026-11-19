//! Lower the process priority for a benchmark so a CPU-bound run leaves the desktop responsive.
//!
//! - Windows: `SetPriorityClass(BELOW_NORMAL_PRIORITY_CLASS)` on the whole process, the previous
//!   class (read with `GetPriorityClass` first; 0 = failure, then nothing is restored) restored
//!   when the [`Guard`] drops.
//! - Unix: `setpriority(PRIO_PROCESS, 0, 10)` in the CLI only, before the runtimes exist: on
//!   Linux nice is per thread and inherited by the threads created afterwards. OpenVINO's TBB
//!   pool and ONNX Runtime's pools are not ours, so a running service cannot lower them after the
//!   fact: in the service this is a no-op (`Nice=` in the systemd unit is the user's lever). An
//!   unprivileged process cannot raise its priority again, so the CLI keeps nice 10 until it
//!   exits (the guard's drop does nothing).
//! - Other OSes: no-op.
//!
//! Child processes (the training / import / auto-label helpers, `Cmd::low_priority` in
//! `resources::export`) start below normal instead: [`child_creation_flags`] adds
//! `BELOW_NORMAL_PRIORITY_CLASS` to the creation flags on Windows (the processes they start,
//! e.g. PyTorch's DataLoader workers, inherit it); [`lower_child`] raises the child's nice value
//! to at least [`CHILD_NICE`] between fork and exec on Unix (inherited by its threads and
//! children).
//!
//! Background threads of the service (the review index, the review queue's walk workers, the
//! Train area's tally, the storage measurement, retention) lower only themselves with
//! [`lower_this_thread`] ([`ThreadRole::Background`]): Windows
//! `SetThreadPriority(THREAD_PRIORITY_BELOW_NORMAL)`; Linux, where nice is per thread, the
//! thread's own nice raised to at least [`BACKGROUND_NICE`] (`setpriority(PRIO_PROCESS, tid)`;
//! never lowered, so a service under `Nice=15` keeps 15); other OSes nothing (macOS
//! `setpriority` would move the whole process). Detection, prep, worker and HTTP threads are
//! [`ThreadRole::Detection`] and never change. A lowered thread stays lowered until it exits (an
//! unprivileged Linux thread cannot get its nice back), so only threads that do nothing else are
//! lowered, never a pooled HTTP or blocking thread.
//!
//! Only the CPU scheduling priority changes: `PROCESS_MODE_BACKGROUND_BEGIN` / EcoQoS are
//! deliberately not used because they also lower I/O and memory priority and change clocks,
//! which skews timings.

/// Restores the previous priority class on drop (Windows); does nothing elsewhere.
#[derive(Debug)]
#[must_use = "the priority is restored when the guard drops"]
pub struct Guard {
    /// Priority class to restore (Windows), None = nothing to restore.
    #[cfg(windows)]
    restore: Option<u32>,
}

impl Guard {
    fn noop() -> Self {
        Self {
            #[cfg(windows)]
            restore: None,
        }
    }
}

/// Lower the process priority for a benchmark run when `enabled` (`benchmark.low_priority`,
/// CLI `--no-low-priority`). `in_service` = the call comes from the running service (web UI),
/// not the CLI. Keep the returned guard alive for the whole run.
pub fn lower_for_benchmark(enabled: bool, in_service: bool) -> Guard {
    if !lowers(enabled, in_service) {
        if enabled {
            tracing::debug!(
                "benchmark: process priority unchanged (the runtimes' threads already exist; \
                 use Nice= in the service unit)"
            );
        }
        return Guard::noop();
    }
    imp::lower()
}

/// Whether [`lower_for_benchmark`] changes the priority (pure): never when disabled, and on Unix
/// not in the service, whose runtime threads already exist (see the module docs).
fn lowers(enabled: bool, in_service: bool) -> bool {
    enabled && !(cfg!(unix) && in_service)
}

/// `BELOW_NORMAL_PRIORITY_CLASS`, a process creation flag on Windows.
pub const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;

/// Nice value (at least) of a low-priority child process on Unix.
pub const CHILD_NICE: i32 = 10;

/// Windows creation flags of a child: `base` (e.g. `CREATE_NO_WINDOW`) plus
/// [`BELOW_NORMAL_PRIORITY_CLASS`] when `low_priority` (pure).
pub const fn child_creation_flags(base: u32, low_priority: bool) -> u32 {
    if low_priority {
        base | BELOW_NORMAL_PRIORITY_CLASS
    } else {
        base
    }
}

/// Start `cmd`'s process with a nice value of at least [`CHILD_NICE`] (never lowers a higher
/// one, e.g. a service already running under `Nice=15`). A failure leaves the inherited
/// priority (an unprivileged process can always raise its nice value, so it does not fail in
/// practice).
#[cfg(unix)]
pub fn lower_child(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: the hook runs in the forked child before exec and only makes the two
    // async-signal-safe syscalls getpriority / setpriority on the child itself; it allocates
    // nothing and touches no shared state.
    unsafe {
        cmd.pre_exec(|| {
            let current = libc::getpriority(libc::PRIO_PROCESS, 0);
            if current < CHILD_NICE {
                let _ = libc::setpriority(libc::PRIO_PROCESS, 0, CHILD_NICE);
            }
            Ok(())
        });
    }
}

/// Nice value (at least) of a background thread on Linux ([`lower_this_thread`]).
pub const BACKGROUND_NICE: i32 = 10;

/// What a thread does, for [`thread_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadRole {
    /// Detection, prep, worker and HTTP threads: never lowered.
    Detection,
    /// A UI or index walk, retention, a measurement: below normal.
    Background,
}

/// The operating systems [`thread_action`] tells apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Linux,
    /// macOS and every other OS: no per-thread priority call is used.
    Other,
}

impl Os {
    /// The OS this build runs on.
    pub const fn current() -> Self {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "linux") {
            Os::Linux
        } else {
            Os::Other
        }
    }
}

/// What [`lower_this_thread`] does to the calling thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadAction {
    /// Nothing.
    Keep,
    /// `SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)`.
    BelowNormal,
    /// Raise the thread's nice to at least this value (never lower it).
    NiceAtLeast(i32),
}

/// The action for a thread of `role` on `os` (pure): background threads below normal on Windows
/// and Linux, everything else kept.
pub const fn thread_action(os: Os, role: ThreadRole) -> ThreadAction {
    match (os, role) {
        (_, ThreadRole::Detection) | (Os::Other, _) => ThreadAction::Keep,
        (Os::Windows, ThreadRole::Background) => ThreadAction::BelowNormal,
        (Os::Linux, ThreadRole::Background) => ThreadAction::NiceAtLeast(BACKGROUND_NICE),
    }
}

/// The nice value a thread at `current` gets for `action` (pure): `None` = no call.
pub const fn nice_target(current: i32, action: ThreadAction) -> Option<i32> {
    match action {
        ThreadAction::NiceAtLeast(n) if current < n => Some(n),
        _ => None,
    }
}

/// Apply [`thread_action`] for `role` to the calling thread (best effort: a failure is logged at
/// debug level and the thread keeps its priority). True when the priority changed. Call it first
/// thing on a thread the caller spawned for background work; see the module docs.
pub fn lower_this_thread(role: ThreadRole) -> bool {
    let action = thread_action(Os::current(), role);
    let changed = imp::apply_thread(action);
    if let Err(e) = &changed {
        tracing::debug!("could not lower the thread's priority ({e}); it keeps its priority");
    }
    changed.unwrap_or(false)
}

/// The calling thread's scheduling level, for tests: `GetThreadPriority` on Windows, the nice
/// value on Linux, 0 elsewhere.
#[cfg(all(test, windows))]
pub(crate) fn thread_level() -> i32 {
    imp::thread_priority()
}

/// See the Windows twin.
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn thread_level() -> i32 {
    // SAFETY: reads the calling thread's nice value (Linux: per thread).
    unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) }
}

/// See the Windows twin.
#[cfg(all(test, not(any(windows, target_os = "linux"))))]
pub(crate) fn thread_level() -> i32 {
    0
}

#[cfg(windows)]
mod imp {
    use super::{Guard, ThreadAction};

    pub(super) fn apply_thread(action: ThreadAction) -> std::io::Result<bool> {
        use windows::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
        };
        if action != ThreadAction::BelowNormal {
            return Ok(false);
        }
        // SAFETY: GetCurrentThread returns a pseudo-handle for the calling thread that needs no
        // closing; SetThreadPriority only changes that thread's scheduling priority.
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) }
            .map(|()| true)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// The calling thread's priority (`GetThreadPriority`), for tests.
    #[cfg(test)]
    pub(super) fn thread_priority() -> i32 {
        use windows::Win32::System::Threading::{GetCurrentThread, GetThreadPriority};
        // SAFETY: reads the calling thread's priority through its pseudo-handle.
        unsafe { GetThreadPriority(GetCurrentThread()) }
    }

    use windows::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, GetPriorityClass, PROCESS_CREATION_FLAGS,
        SetPriorityClass,
    };

    pub(super) fn lower() -> Guard {
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing; both calls
        // only read/write this process's priority class.
        let previous = unsafe { GetPriorityClass(GetCurrentProcess()) };
        if previous == 0 {
            tracing::warn!(
                "benchmark: could not read the process priority class ({}); keeping the normal \
                 priority",
                std::io::Error::last_os_error()
            );
            return Guard::noop();
        }
        if previous == BELOW_NORMAL_PRIORITY_CLASS.0 {
            return Guard::noop();
        }
        // SAFETY: see above.
        match unsafe { SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) } {
            Ok(()) => {
                tracing::info!(
                    "benchmark: process priority lowered to below normal (benchmark.low_priority)"
                );
                Guard {
                    restore: Some(previous),
                }
            }
            Err(e) => {
                tracing::warn!(
                    "benchmark: could not lower the process priority ({e}); the run keeps the \
                     normal priority"
                );
                Guard::noop()
            }
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let Some(class) = self.restore.take() else {
                return;
            };
            // SAFETY: see `lower`.
            match unsafe { SetPriorityClass(GetCurrentProcess(), PROCESS_CREATION_FLAGS(class)) } {
                Ok(()) => tracing::info!("benchmark: process priority restored"),
                Err(e) => tracing::warn!(
                    "benchmark: could not restore the process priority ({e}); restart the \
                     service to get the normal priority back"
                ),
            }
        }
    }
}

#[cfg(unix)]
mod imp {
    use super::{Guard, ThreadAction};

    pub(super) fn apply_thread(action: ThreadAction) -> std::io::Result<bool> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: gettid has no preconditions and only returns the calling thread's id.
            let tid = unsafe { libc::gettid() };
            let who = libc::id_t::try_from(tid).map_err(std::io::Error::other)?;
            // SAFETY: errno is thread-local; clearing it first tells a nice of -1 from an error.
            unsafe { *libc::__errno_location() = 0 };
            // SAFETY: getpriority / setpriority with PRIO_PROCESS and a thread id read and change
            // only that thread's nice value (Linux: nice is per thread).
            let current = unsafe { libc::getpriority(libc::PRIO_PROCESS, who) };
            if current == -1 && std::io::Error::last_os_error().raw_os_error() != Some(0) {
                return Err(std::io::Error::last_os_error());
            }
            let Some(target) = super::nice_target(current, action) else {
                return Ok(false);
            };
            // SAFETY: see above.
            if unsafe { libc::setpriority(libc::PRIO_PROCESS, who, target) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(true)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = action;
            Ok(false)
        }
    }

    /// Nice value of a low-priority benchmark.
    const NICE: libc::c_int = 10;

    pub(super) fn lower() -> Guard {
        // SAFETY: setpriority only changes the scheduling priority of the calling thread
        // (Linux) / process; no memory is touched.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, NICE) };
        if rc == 0 {
            tracing::info!(
                "benchmark: process priority lowered to nice {NICE} (benchmark.low_priority)"
            );
        } else {
            tracing::warn!(
                "benchmark: could not lower the process priority ({}); the run keeps the \
                 normal priority",
                std::io::Error::last_os_error()
            );
        }
        Guard::noop()
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    use super::{Guard, ThreadAction};

    pub(super) fn apply_thread(_action: ThreadAction) -> std::io::Result<bool> {
        Ok(false)
    }

    pub(super) fn lower() -> Guard {
        Guard::noop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_flags_add_below_normal_only_when_asked() {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        assert_eq!(child_creation_flags(CREATE_NO_WINDOW, true), 0x0800_4000);
        assert_eq!(
            child_creation_flags(CREATE_NO_WINDOW, false),
            CREATE_NO_WINDOW
        );
        assert_eq!(child_creation_flags(0, true), BELOW_NORMAL_PRIORITY_CLASS);
        #[cfg(windows)]
        assert_eq!(
            BELOW_NORMAL_PRIORITY_CLASS,
            windows::Win32::System::Threading::BELOW_NORMAL_PRIORITY_CLASS.0
        );
    }

    /// The nice value of the calling thread (Linux: nice is per thread, so a fresh thread sees
    /// no other test's change).
    #[cfg(target_os = "linux")]
    fn own_nice() -> i32 {
        // SAFETY: reads the calling thread's scheduling priority.
        unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) }
    }

    #[test]
    fn disabled_is_noop() {
        assert!(!lowers(false, false) && !lowers(false, true));
        assert!(lowers(true, false), "the CLI lowers when enabled");
        #[cfg(windows)]
        assert!(lower_for_benchmark(false, false).restore.is_none());
        #[cfg(target_os = "linux")]
        std::thread::spawn(|| {
            let before = own_nice();
            drop(lower_for_benchmark(false, false));
            assert_eq!(own_nice(), before);
        })
        .join()
        .unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn guard_sets_and_restores() {
        use windows::Win32::System::Threading::{
            BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, GetPriorityClass,
        };
        // SAFETY: reads this process's priority class.
        let class = || unsafe { GetPriorityClass(GetCurrentProcess()) };
        let before = class();
        assert_ne!(before, 0);
        let guard = lower_for_benchmark(true, true);
        assert_eq!(class(), BELOW_NORMAL_PRIORITY_CLASS.0);
        drop(guard);
        assert_eq!(class(), before);
    }

    #[test]
    fn thread_action_table() {
        use ThreadAction::{BelowNormal, Keep, NiceAtLeast};
        use ThreadRole::{Background, Detection};
        let table = [
            (Os::Windows, Detection, Keep),
            (Os::Windows, Background, BelowNormal),
            (Os::Linux, Detection, Keep),
            (Os::Linux, Background, NiceAtLeast(BACKGROUND_NICE)),
            (Os::Other, Detection, Keep),
            (Os::Other, Background, Keep),
        ];
        for (os, role, want) in table {
            assert_eq!(thread_action(os, role), want, "{os:?} {role:?}");
        }
        // Nice only ever goes up: a thread already at or above the target is left alone.
        assert_eq!(nice_target(0, NiceAtLeast(10)), Some(10));
        assert_eq!(nice_target(-5, NiceAtLeast(10)), Some(10));
        assert_eq!(nice_target(10, NiceAtLeast(10)), None);
        assert_eq!(nice_target(15, NiceAtLeast(10)), None);
        assert_eq!(nice_target(0, Keep), None);
        assert_eq!(nice_target(0, BelowNormal), None);
        let here = Os::current();
        #[cfg(windows)]
        assert_eq!(here, Os::Windows);
        #[cfg(target_os = "linux")]
        assert_eq!(here, Os::Linux);
        #[cfg(not(any(windows, target_os = "linux")))]
        assert_eq!(here, Os::Other);
    }

    /// A background thread is lowered, a detection thread and the spawning thread are not.
    #[test]
    fn only_the_calling_background_thread_is_lowered() {
        let level = thread_level;
        let before = level();
        let (detection, background) = std::thread::spawn(move || {
            let d = std::thread::spawn(move || {
                let changed = lower_this_thread(ThreadRole::Detection);
                (changed, level())
            })
            .join()
            .unwrap();
            let b = std::thread::spawn(move || {
                let changed = lower_this_thread(ThreadRole::Background);
                (changed, level())
            })
            .join()
            .unwrap();
            (d, b)
        })
        .join()
        .unwrap();
        assert_eq!(
            detection,
            (false, before),
            "detection threads keep their priority"
        );
        assert_eq!(level(), before, "the spawning thread is untouched");
        #[cfg(windows)]
        assert_eq!(
            background,
            (
                true,
                windows::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL.0
            )
        );
        #[cfg(target_os = "linux")]
        assert!(background.1 >= BACKGROUND_NICE, "{background:?}");
        #[cfg(not(any(windows, target_os = "linux")))]
        assert_eq!(background, (false, before));
    }

    #[cfg(unix)]
    #[test]
    fn service_guard_is_noop() {
        assert!(!lowers(true, true));
        #[cfg(target_os = "linux")]
        std::thread::spawn(|| {
            let before = own_nice();
            drop(lower_for_benchmark(true, true));
            assert_eq!(own_nice(), before, "the service keeps its priority");
            // The CLI path in the same fresh thread does lower it: the check above can fail.
            drop(lower_for_benchmark(true, false));
            assert!(own_nice() >= 10, "nice {}", own_nice());
        })
        .join()
        .unwrap();
    }
}
