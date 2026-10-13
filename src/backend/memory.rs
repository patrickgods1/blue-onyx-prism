//! Total physical memory from the OS, without spawning a process.
//!
//! Windows asks `GlobalMemoryStatusEx` (the FFI call is why this lives in `backend/`, the only
//! place `unsafe` is allowed); other platforms return `None` and `system_info` reads
//! `/proc/meminfo` or `sysctl` there instead. Never panics; a failed call is `None`.

/// Total physical memory in bytes as reported by the OS (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms or when the call fails.
#[cfg(windows)]
pub fn total_physical_bytes() -> Option<u64> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).ok()?,
        ..MEMORYSTATUSEX::default()
    };
    // SAFETY: `status` is a live, writable MEMORYSTATUSEX with `dwLength` set to its size, as
    // the API requires; the call only writes into it for its duration.
    unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
    Some(status.ullTotalPhys).filter(|b| *b > 0)
}

/// Total physical memory in bytes as reported by the OS (`GlobalMemoryStatusEx` on Windows);
/// `None` on other platforms or when the call fails.
#[cfg(not(windows))]
pub fn total_physical_bytes() -> Option<u64> {
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
}
