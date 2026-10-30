//! Total physical memory and the commit charge from the OS, without spawning a process.
//!
//! Windows asks `GlobalMemoryStatusEx` (the FFI call is why this lives in `backend/`, the only
//! place `unsafe` is allowed); other platforms return `None` and `system_info` reads
//! `/proc/meminfo` or `sysctl` there instead. Never panics; a failed call is `None`.

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
