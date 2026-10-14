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
    if !enabled {
        return Guard::noop();
    }
    imp::lower(in_service)
}

#[cfg(windows)]
mod imp {
    use super::Guard;
    use windows::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, GetPriorityClass, PROCESS_CREATION_FLAGS,
        SetPriorityClass,
    };

    pub(super) fn lower(_in_service: bool) -> Guard {
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
    use super::Guard;

    /// Nice value of a low-priority benchmark.
    const NICE: libc::c_int = 10;

    pub(super) fn lower(in_service: bool) -> Guard {
        if in_service {
            tracing::debug!(
                "benchmark: process priority unchanged (the runtimes' threads already exist; \
                 use Nice= in the service unit)"
            );
            return Guard::noop();
        }
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
    use super::Guard;

    pub(super) fn lower(_in_service: bool) -> Guard {
        Guard::noop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_is_noop() {
        let g = lower_for_benchmark(false, false);
        drop(g);
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

    #[cfg(not(windows))]
    #[test]
    fn service_guard_is_noop() {
        let guard = lower_for_benchmark(true, true);
        drop(guard);
    }
}
