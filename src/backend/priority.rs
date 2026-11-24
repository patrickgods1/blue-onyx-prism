//! Lower the process priority while a benchmark runs on a CPU device, so a CPU-bound run leaves
//! the desktop responsive.
//!
//! - Only CPU devices (`openvino:cpu`, `ort:cpu`, legacy `CPU`; [`lowers_device`], pure): a GPU
//!   device's run keeps the normal priority. On Windows the priority class is the whole
//!   process's, including the threads answering Blue Iris, and a GPU run leaves the CPU mostly
//!   idle, so lowering it would only slow detection down for nothing.
//! - Windows: [`DevicePriority`] (one per benchmark run) sets
//!   `SetPriorityClass(BELOW_NORMAL_PRIORITY_CLASS)` when a CPU device starts (its compile, warm-up
//!   and timing) and restores the previous class (read with `GetPriorityClass` first; 0 =
//!   failure, then nothing is restored) when a non-CPU device starts or the run ends.
//! - Unix: `setpriority(PRIO_PROCESS, 0, 10)` (only when the nice value is below 10, never
//!   lowering a higher one, `benchmark_nice`) in the CLI only, before the runtimes exist, and
//!   only for a run whose devices are all CPUs ([`lower_up_front`]): on Linux nice is per thread
//!   and inherited by the threads created afterwards. OpenVINO's TBB pool and ONNX Runtime's pools
//!   are not ours, so a running service cannot lower them after the fact: in the service this is
//!   a no-op (`Nice=` in the systemd unit is the user's lever). An unprivileged process cannot
//!   raise its priority again, so per-device switching is impossible there and the CLI keeps
//!   nice 10 until it exits (the guard's drop does nothing).
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

use crate::sync::LockExt;
use std::sync::Mutex;

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

/// Whether a running process can lower its priority for one device and raise it again after
/// (Windows' priority class); elsewhere the CLI lowers a CPU-only run up front
/// ([`lower_up_front`]) and the service never does.
pub const PER_DEVICE: bool = cfg!(windows);

/// Whether a benchmark on the device spec `device` runs below normal priority (pure): CPU
/// devices only (`openvino:cpu`, `ort:cpu`, legacy `CPU`). `auto`, GPUs, NPUs and an
/// unparsable spec keep the normal priority (`auto` is decided again on the device it loaded
/// on).
pub fn lowers_device(device: &str) -> bool {
    super::spec::parse(device).is_ok_and(|s| s.is_cpu())
}

/// Which priority a benchmark device run got (log lines and the Benchmark page's progress).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    /// Below normal while this CPU device runs.
    BelowNormal,
    /// Normal: not a CPU device.
    Normal,
    /// Normal: `benchmark.low_priority` is off.
    Off,
    /// Unchanged: this OS cannot raise the priority again after the device, so a running
    /// process does not lower it per device (see the module docs).
    Unchanged,
}

impl Applied {
    /// Short text for the progress line ("below normal priority (CPU device)").
    pub fn describe(self) -> &'static str {
        match self {
            Applied::BelowNormal => "below normal priority (CPU device)",
            Applied::Normal => "normal priority (not a CPU device)",
            Applied::Off => "normal priority (low priority off)",
            Applied::Unchanged => "priority unchanged (not adjustable per device on this OS)",
        }
    }
}

/// The priority of a benchmark on `device` (pure): `enabled` = `benchmark.low_priority` (and
/// no `--no-low-priority`), `per_device` = [`PER_DEVICE`].
pub fn decide(enabled: bool, device: &str, per_device: bool) -> Applied {
    if !enabled {
        Applied::Off
    } else if !lowers_device(device) {
        Applied::Normal
    } else if !per_device {
        Applied::Unchanged
    } else {
        Applied::BelowNormal
    }
}

/// The priority of one benchmark run, switched per device: [`DevicePriority::enter`] before a
/// device compiles and again on the device it loaded on. Below normal while a CPU device runs,
/// restored when a non-CPU device starts and when this drops (end of run, error, panic).
#[derive(Debug)]
pub struct DevicePriority {
    enabled: bool,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Held while lowered.
    guard: Option<Guard>,
    /// Last decision and its device (logged once per change).
    last: Option<(String, Applied)>,
}

impl DevicePriority {
    /// `enabled` = `benchmark.low_priority` (and no `--no-low-priority`).
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            state: Mutex::default(),
        }
    }

    /// Set the priority for a run on `device` (a device spec) and return what applied.
    pub fn enter(&self, device: &str) -> Applied {
        let applied = decide(self.enabled, device, PER_DEVICE);
        let mut st = self.state.lock_recover();
        match applied {
            Applied::BelowNormal => {
                if st.guard.is_none() {
                    st.guard = Some(imp::lower());
                }
            }
            // Dropping the guard restores the previous class.
            _ => st.guard = None,
        }
        let changed = st
            .last
            .as_ref()
            .is_none_or(|(d, a)| d != device || *a != applied);
        if changed {
            match applied {
                Applied::Off => tracing::debug!("benchmark on {device}: {}", applied.describe()),
                _ => tracing::info!(
                    "benchmark on {device}: {} (benchmark.low_priority lowers CPU devices only)",
                    applied.describe()
                ),
            }
            st.last = Some((device.to_string(), applied));
        }
        applied
    }

    /// The last decision ([`Self::enter`]), None before the first device.
    pub fn current(&self) -> Option<Applied> {
        self.state.lock_recover().last.as_ref().map(|(_, a)| *a)
    }
}

/// The CLI on Unix: lower the process before the runtimes exist (their thread pools inherit the
/// nice value) when `enabled` and every device of the run is a CPU (`cpu_only`). Elsewhere
/// (Windows: [`DevicePriority`] per device) a no-op. Keep the returned guard alive for the whole
/// run.
pub fn lower_up_front(enabled: bool, cpu_only: bool) -> Guard {
    if !lowers_up_front(enabled, cpu_only, PER_DEVICE) {
        return Guard::noop();
    }
    imp::lower()
}

/// Whether [`lower_up_front`] lowers (pure): only where the priority cannot be switched per
/// device, for an enabled CPU-only run.
fn lowers_up_front(enabled: bool, cpu_only: bool, per_device: bool) -> bool {
    enabled && cpu_only && !per_device
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

/// The nice value [`lower_up_front`] sets on Unix for a process at `current` (pure): at least
/// [`CHILD_NICE`], never lowering a higher one (a CLI run under `nice -n 15` keeps 15; QA-042).
#[cfg_attr(
    not(unix),
    allow(dead_code, reason = "used by the Unix `imp::lower` and tests")
)]
const fn benchmark_nice(current: i32) -> Option<i32> {
    nice_target(current, ThreadAction::NiceAtLeast(CHILD_NICE))
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
                tracing::debug!("benchmark: process priority class set to below normal");
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
                Ok(()) => tracing::debug!("benchmark: process priority class restored"),
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

    pub(super) fn lower() -> Guard {
        // SAFETY: getpriority only reads the calling process's (Linux: thread's) nice value. A
        // -1 is either nice -1 or an error; both lead to the setpriority call below, as before
        // this read existed, so errno need not be cleared.
        let current = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
        let Some(target) = super::benchmark_nice(current) else {
            tracing::info!(
                "benchmark: the process already runs at nice {current} (at least {}); its \
                 priority is left as it is (benchmark.low_priority)",
                super::CHILD_NICE
            );
            return Guard::noop();
        };
        // SAFETY: setpriority only changes the scheduling priority of the calling thread
        // (Linux) / process; no memory is touched.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, target) };
        if rc == 0 {
            tracing::info!(
                "benchmark: process priority lowered to nice {target} for this CPU-only run \
                 (benchmark.low_priority)"
            );
        } else {
            tracing::warn!(
                "benchmark: could not lower the process priority ({}); the run keeps nice \
                 {current}",
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

    #[test]
    fn only_cpu_devices_lower() {
        for cpu in ["openvino:cpu", "ort:cpu", "CPU", "cpu", " OpenVINO:CPU "] {
            assert!(lowers_device(cpu), "{cpu}");
        }
        for other in [
            "auto",
            "openvino:gpu",
            "openvino:gpu.1",
            "GPU",
            "GPU.0",
            "openvino:npu",
            "NPU",
            "ort:cuda",
            "ort:cuda:1",
            "ort:tensorrt",
            "ort:directml:0",
            "ort:coreml",
            "",
            "nonsense",
        ] {
            assert!(!lowers_device(other), "{other}");
        }
    }

    /// The decision table: off, then CPU only, then whether the OS can switch per device.
    #[test]
    fn decision_table() {
        use Applied::*;
        let rows = [
            // (enabled, device, per_device) -> applied
            (true, "openvino:cpu", true, BelowNormal),
            (true, "ort:cpu", true, BelowNormal),
            (true, "CPU", true, BelowNormal),
            (true, "openvino:gpu.0", true, Normal),
            (true, "ort:cuda:0", true, Normal),
            (true, "ort:directml", true, Normal),
            (true, "ort:coreml", true, Normal),
            (true, "openvino:npu", true, Normal),
            (true, "auto", true, Normal),
            (false, "openvino:cpu", true, Off),
            (false, "ort:cuda", true, Off),
            (true, "openvino:cpu", false, Unchanged),
            (true, "openvino:gpu", false, Normal),
            (false, "ort:cpu", false, Off),
        ];
        for (enabled, device, per_device, want) in rows {
            assert_eq!(
                decide(enabled, device, per_device),
                want,
                "{enabled} {device} {per_device}"
            );
        }
        // Up front (Unix CLI): only an enabled CPU-only run where per-device switching is
        // impossible.
        assert!(lowers_up_front(true, true, false));
        assert!(!lowers_up_front(true, false, false));
        assert!(!lowers_up_front(false, true, false));
        assert!(!lowers_up_front(true, true, true));
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
        let p = DevicePriority::new(false);
        assert_eq!(p.current(), None);
        assert_eq!(p.enter("openvino:cpu"), Applied::Off);
        assert_eq!(p.current(), Some(Applied::Off));
        assert!(p.state.lock_recover().guard.is_none());
        #[cfg(target_os = "linux")]
        std::thread::spawn(|| {
            let before = own_nice();
            drop(lower_up_front(false, true));
            drop(lower_up_front(true, false));
            assert_eq!(own_nice(), before);
        })
        .join()
        .unwrap();
    }

    /// A CPU device lowers the process, a GPU device restores it, the end of the run too.
    #[cfg(windows)]
    #[test]
    fn device_priority_switches_and_restores() {
        use windows::Win32::System::Threading::{
            BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, GetPriorityClass,
        };
        // SAFETY: reads this process's priority class.
        let class = || unsafe { GetPriorityClass(GetCurrentProcess()) };
        let before = class();
        assert_ne!(before, 0);
        let p = DevicePriority::new(true);
        assert_eq!(p.enter("openvino:cpu"), Applied::BelowNormal);
        assert_eq!(class(), BELOW_NORMAL_PRIORITY_CLASS.0);
        // Again on the same or another CPU device: still lowered, one guard.
        assert_eq!(p.enter("ort:cpu"), Applied::BelowNormal);
        assert_eq!(class(), BELOW_NORMAL_PRIORITY_CLASS.0);
        assert_eq!(p.enter("openvino:gpu.0"), Applied::Normal);
        assert_eq!(class(), before, "restored before the GPU device runs");
        assert_eq!(p.enter("CPU"), Applied::BelowNormal);
        assert_eq!(class(), BELOW_NORMAL_PRIORITY_CLASS.0);
        drop(p);
        assert_eq!(class(), before, "restored at the end of the run");
        // Up front is a no-op where the priority switches per device.
        drop(lower_up_front(true, true));
        assert_eq!(class(), before);
    }

    // QA-042: the Unix benchmark set nice 10 unconditionally, raising the priority of a CLI run
    // started under `nice -n 15` (as root) or failing with EACCES and a false log line.
    #[test]
    fn benchmark_nice_never_lowers_a_higher_nice() {
        assert_eq!(benchmark_nice(0), Some(CHILD_NICE));
        assert_eq!(benchmark_nice(-1), Some(CHILD_NICE));
        assert_eq!(benchmark_nice(CHILD_NICE - 1), Some(CHILD_NICE));
        assert_eq!(benchmark_nice(CHILD_NICE), None);
        assert_eq!(benchmark_nice(15), None);
        assert_eq!(benchmark_nice(19), None);
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
    fn unix_never_switches_per_device() {
        let p = DevicePriority::new(true);
        assert_eq!(p.enter("openvino:cpu"), Applied::Unchanged);
        assert_eq!(p.enter("openvino:gpu"), Applied::Normal);
        #[cfg(target_os = "linux")]
        std::thread::spawn(|| {
            let before = own_nice();
            assert_eq!(
                DevicePriority::new(true).enter("ort:cpu"),
                Applied::Unchanged
            );
            assert_eq!(own_nice(), before, "a running process keeps its priority");
            // The CLI's up-front path for a CPU-only run in the same fresh thread does lower
            // it: the check above can fail.
            drop(lower_up_front(true, true));
            assert!(own_nice() >= 10, "nice {}", own_nice());
        })
        .join()
        .unwrap();
    }
}
