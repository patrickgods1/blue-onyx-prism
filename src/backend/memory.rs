//! Total physical memory, the commit charge and this process's own memory from the OS, without
//! spawning a process.
//!
//! Windows asks `GlobalMemoryStatusEx` (the FFI call is why this lives in `backend/`, the only
//! place `unsafe` is allowed); other platforms return `None` and `system_info` reads
//! `/proc/meminfo` or `sysctl` there instead. [`process_memory`] asks `K32GetProcessMemoryInfo`
//! on Windows and `proc_pid_rusage` on macOS; Linux reads `/proc/self/status` in
//! [`crate::metrics`], with no FFI. Never panics; a failed call is `None`.

/// The system commit charge (Windows): what every process may commit in total (RAM + paging
/// file) and what is still free. A process that needs more fails with WinError 1455 ("the paging
/// file is too small") even when RAM is free, which is what spawned PyTorch processes hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commit {
    /// Commit limit in bytes (`ullTotalPageFile`: physical memory + paging files).
    pub limit: u64,
    /// Commit still available in bytes (`ullAvailPageFile`).
    pub available: u64,
}

impl Commit {
    /// Bytes committed now (`limit - available`).
    pub fn in_use(&self) -> u64 {
        self.limit.saturating_sub(self.available)
    }
}

/// What this process holds in memory now, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Pages in RAM: the working set on Windows, the resident size on macOS.
    pub resident: u64,
    /// macOS only: the physical footprint, what Activity Monitor calls Memory. It counts
    /// compressed and swapped pages and GPU-backed memory charged to the process, which the
    /// resident size misses.
    pub footprint: Option<u64>,
}

/// [`ProcessMemory`] from `K32GetProcessMemoryInfo`; None when the call fails.
#[cfg(windows)]
pub fn process_memory() -> Option<ProcessMemory> {
    use windows::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::GetCurrentProcess;
    let cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>()).ok()?;
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb,
        ..PROCESS_MEMORY_COUNTERS::default()
    };
    // SAFETY: the pseudo-handle of the current process is always valid and needs no closing;
    // `counters` is live and writable with `cb` set to its size, and is only written during the call.
    let done = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) };
    done.as_bool().then_some(ProcessMemory {
        resident: counters.WorkingSetSize as u64,
        footprint: None,
    })
}

/// [`ProcessMemory`] from `proc_pid_rusage`; None when the call fails.
#[cfg(target_os = "macos")]
pub fn process_memory() -> Option<ProcessMemory> {
    let pid = libc::c_int::try_from(std::process::id()).ok()?;
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `usage` is live, writable and sized for the flavor asked for; the kernel only writes
    // that struct, during the call (the parameter's type is a C API quirk: it is the struct's address).
    let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, usage.as_mut_ptr().cast()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: the struct is plain integers, zeroed above and filled in by the successful call.
    let usage = unsafe { usage.assume_init() };
    Some(ProcessMemory {
        resident: usage.ri_resident_size,
        footprint: Some(usage.ri_phys_footprint),
    })
}

/// None: Linux reads `/proc/self/status` ([`crate::metrics::ProcessStats`]).
#[cfg(not(any(windows, target_os = "macos")))]
pub fn process_memory() -> Option<ProcessMemory> {
    None
}

/// One `GlobalMemoryStatusEx` call; None when it fails.
#[cfg(windows)]
fn status() -> Option<windows::Win32::System::SystemInformation::MEMORYSTATUSEX> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).ok()?,
        ..MEMORYSTATUSEX::default()
    };
    // SAFETY: `status` is a live, writable MEMORYSTATUSEX with `dwLength` set to its size, as
    // the API requires; the call only writes into it for its duration.
    unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
    Some(status)
}

/// Total physical memory in bytes as reported by the OS (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms or when the call fails.
#[cfg(windows)]
pub fn total_physical_bytes() -> Option<u64> {
    Some(status()?.ullTotalPhys).filter(|b| *b > 0)
}

/// Total physical memory in bytes as reported by the OS (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms or when the call fails.
#[cfg(not(windows))]
pub fn total_physical_bytes() -> Option<u64> {
    None
}

/// The commit limit and the commit still available now (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms (Linux overcommits and macOS grows its swap, so there is no fixed
/// limit to plan against) or when the call fails.
#[cfg(windows)]
pub fn commit_bytes() -> Option<Commit> {
    let s = status()?;
    Some(Commit {
        limit: s.ullTotalPageFile,
        available: s.ullAvailPageFile,
    })
    .filter(|c| c.limit > 0)
}

/// The commit limit and the commit still available now (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms (Linux overcommits and macOS grows its swap, so there is no fixed
/// limit to plan against) or when the call fails.
#[cfg(not(windows))]
pub fn commit_bytes() -> Option<Commit> {
    None
}

#[cfg(test)]
mod tests {
    /// Grows by a large touched allocation. Other tests of this binary free memory at any moment,
    /// so one attempt showing the growth is enough; a counter that does not move fails them all.
    #[cfg(any(windows, target_os = "macos"))]
    #[test]
    fn process_memory_is_non_zero_and_grows_after_a_large_allocation() {
        const SIZE: usize = 256 << 20;
        let read = || super::process_memory().unwrap();
        let first = read();
        assert!(first.resident > 1 << 20, "{first:?}");
        assert_eq!(first.footprint.is_some(), cfg!(target_os = "macos"));
        assert!(first.footprint.is_none_or(|b| b > 1 << 20), "{first:?}");
        let grew = (0..5).any(|_| {
            let before = read();
            let block = vec![1u8; SIZE];
            let after = read();
            std::hint::black_box(&block);
            let by = |a: Option<u64>, b: Option<u64>| {
                a.zip(b)
                    .is_none_or(|(a, b)| b.saturating_sub(a) >= SIZE as u64 / 2)
            };
            after.resident.saturating_sub(before.resident) >= SIZE as u64 / 2
                && by(before.footprint, after.footprint)
        });
        assert!(grew, "no growth seen over 5 allocations of {SIZE} bytes");
    }

    #[cfg(not(any(windows, target_os = "macos")))]
    #[test]
    fn process_memory_is_left_to_proc_elsewhere() {
        assert_eq!(super::process_memory(), None);
    }

    #[test]
    fn total_physical_bytes_is_plausible() {
        let got = super::total_physical_bytes();
        if cfg!(windows) {
            // Any machine running the tests has more than 256 MiB.
            assert!(got.is_some_and(|b| b > 256 << 20), "{got:?}");
        } else {
            assert_eq!(got, None);
        }
    }

    #[test]
    fn commit_is_plausible() {
        let got = super::commit_bytes();
        if cfg!(windows) {
            let c = got.unwrap();
            // The limit covers RAM (plus any paging file); what is free fits in it.
            assert!(
                c.limit >= super::total_physical_bytes().unwrap() / 2,
                "{c:?}"
            );
            assert!(c.available <= c.limit, "{c:?}");
            assert_eq!(c.in_use(), c.limit - c.available);
        } else {
            assert_eq!(got, None);
        }
    }
}
