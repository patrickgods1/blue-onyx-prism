//! Total and available physical memory, the commit charge, the paging files and this process's
//! own memory from the OS, without spawning a process.
//!
//! Windows asks `GlobalMemoryStatusEx` and the registry (the FFI calls are why this lives in
//! `backend/`, the only place `unsafe` is allowed); other platforms return `None` and
//! `system_info` reads `/proc/meminfo` or `sysctl` there instead. [`process_memory`] asks
//! `K32GetProcessMemoryInfo` on Windows and `proc_pid_rusage` on macOS; Linux reads
//! `/proc/self/status` in [`crate::metrics`], with no FFI. Never panics; a failed call is `None`.
//!
//! Paging files: [`parse_paging_files`] reads the registry's `PagingFiles` lines and
//! [`growable_bytes`] (pure) how far they can still grow. That is commit a process gets on
//! demand: Windows grows a paging file whose maximum is above its current size as the commit
//! charge nears the limit (measured on a 16 GB box with `c:\pagefile.sys 2935 24576`: the file
//! went from 7.6 to 16.9 GB during one training run, with no WinError 1455). Each volume keeps
//! [`PAGING_DISK_MARGIN_BYTES`] free; a system-managed file may reach [`system_managed_max`]
//! (3 x RAM, at least 4 GB, Windows' own rule).

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

/// Physical memory (Windows): what is installed and what is available now (free + standby).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Physical {
    /// Bytes of RAM (`ullTotalPhys`).
    pub total: u64,
    /// Bytes available now (`ullAvailPhys`): what new working sets get without paging others out.
    pub available: u64,
}

/// Bytes in a MiB (the registry's paging file sizes are MB = MiB).
const MIB: u64 = 1 << 20;

/// Free space kept on a paging file's volume when counting how far the file can grow: Windows
/// grows the file until the disk is nearly full, and other writers need that room. The paging
/// file is usually on the system drive, which on a Blue Iris box often holds the recordings too:
/// an auto-label job grew `c:\pagefile.sys` to its 24 GB maximum and left C: with 236 MB free
/// (2026-11-18), so 4 GB was not enough to protect them.
pub const PAGING_DISK_MARGIN_BYTES: u64 = 16 << 30;

/// One line of the registry's `PagingFiles` value
/// (`HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management`):
/// `<path> <initial MB> <maximum MB>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagingFile {
    /// As written, e.g. `c:\pagefile.sys`; `?:\pagefile.sys` = Windows picks the drive.
    pub path: String,
    /// Initial size in MB (0 when system-managed).
    pub initial_mb: u64,
    /// Maximum size in MB (0 when system-managed).
    pub max_mb: u64,
    /// Windows sizes the file: a `?:\` path ("Automatically manage paging file size for all
    /// drives" writes `?:\pagefile.sys`), sizes `0 0` or no sizes ("System managed size").
    pub system_managed: bool,
}

impl PagingFile {
    /// Initial == maximum: the file never grows.
    pub fn fixed(&self) -> bool {
        !self.system_managed && self.initial_mb == self.max_mb
    }

    /// The largest the file may become, in bytes.
    pub fn max_bytes(&self, total_phys: u64) -> u64 {
        if self.system_managed {
            system_managed_max(total_phys)
        } else {
            self.max_mb.saturating_mul(MIB)
        }
    }

    /// The drive letter of the path, uppercase; None for `?:` (Windows picks).
    pub fn drive(&self) -> Option<char> {
        self.path
            .chars()
            .next()
            .filter(char::is_ascii_alphabetic)
            .map(|c| c.to_ascii_uppercase())
    }
}

/// The maximum of a system-managed paging file: 3 x RAM, at least 4 GB.
pub fn system_managed_max(total_phys: u64) -> u64 {
    total_phys.saturating_mul(3).max(4 << 30)
}

/// `X:\...` or `?:\...`.
fn is_paging_path(path: &str) -> bool {
    matches!(path.as_bytes(), [d, b':', b'\\', _, ..] if d.is_ascii_alphabetic() || *d == b'?')
}

/// The uppercase drive letter `s` starts with (`C:...`).
fn drive_letter(s: &str) -> Option<char> {
    match s.as_bytes() {
        [d, b':', ..] if d.is_ascii_alphabetic() => Some(char::from(d.to_ascii_uppercase())),
        _ => None,
    }
}

/// The registry's `PagingFiles` lines as [`PagingFile`]s (pure). `?:\pagefile.sys`, sizes `0 0`
/// or no sizes are system-managed; a line that is not `<X:\path> [<initial> <max>]` with whole
/// numbers and initial <= max is skipped.
pub fn parse_paging_files(lines: &[String]) -> Vec<PagingFile> {
    lines
        .iter()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let path = parts.next()?;
            if !is_paging_path(path) {
                return None;
            }
            let sizes: Vec<&str> = parts.collect();
            let (initial_mb, max_mb) = match sizes.as_slice() {
                [] => (0, 0),
                [a, b] => (a.parse::<u64>().ok()?, b.parse::<u64>().ok()?),
                _ => return None,
            };
            if initial_mb > max_mb {
                return None;
            }
            let system_managed = path.starts_with('?') || max_mb == 0;
            Some(PagingFile {
                path: path.to_string(),
                initial_mb: if system_managed { 0 } else { initial_mb },
                max_mb: if system_managed { 0 } else { max_mb },
                system_managed,
            })
        })
        .collect()
}

/// The volume root (`C:\`) of `file`: its own drive letter; for `?:` the drive of the first
/// `ExistingPageFiles` entry (`\??\C:\pagefile.sys`, the file Windows actually uses), else
/// `system_drive` (`SystemDrive`, `C:`). None when none of them names a drive.
pub fn volume_root(
    file: &PagingFile,
    existing: &[String],
    system_drive: Option<&str>,
) -> Option<String> {
    file.drive()
        .or_else(|| {
            existing
                .iter()
                .find_map(|e| drive_letter(e.trim().trim_start_matches(r"\??\")))
        })
        .or_else(|| system_drive.and_then(|d| drive_letter(d.trim())))
        .map(|c| format!("{c}:\\"))
}

/// Bytes the paging files can still grow by (pure): over `files`, the maximum less the current
/// size, each volume's total capped by its free space (`free_space` of the root `volume_of`
/// names) less [`PAGING_DISK_MARGIN_BYTES`].
///
/// The current sizes are what the commit limit holds beyond RAM (`commit_limit - total_phys`;
/// the limit runs a few hundred MB below RAM + paging files, so this errs a little high), shared
/// out in proportion to the initial sizes (a system-managed file weighs its maximum, so its share
/// is large and its room small; all weights 0 = equal shares). Fixed files (initial == maximum)
/// give 0, and so does a file whose volume or free space cannot be read: nothing is assumed.
pub fn growable_bytes(
    files: &[PagingFile],
    commit_limit: u64,
    total_phys: u64,
    volume_of: impl Fn(&PagingFile) -> Option<String>,
    free_space: impl Fn(&str) -> Option<u64>,
) -> u64 {
    let current_total = u128::from(commit_limit.saturating_sub(total_phys));
    let weight = |f: &PagingFile| {
        u128::from(if f.system_managed {
            f.max_bytes(total_phys)
        } else {
            f.initial_mb.saturating_mul(MIB)
        })
    };
    let weights: u128 = files.iter().map(weight).sum();
    let count = files.len() as u128;
    // Room per volume root, in file order.
    let mut volumes: Vec<(String, u64)> = Vec::new();
    for f in files {
        // All weights 0 = equal shares (`count` > 0 inside the loop).
        let share = current_total
            .saturating_mul(weight(f))
            .checked_div(weights)
            .unwrap_or_else(|| current_total.checked_div(count).unwrap_or(0));
        let current = u64::try_from(share).unwrap_or(u64::MAX);
        let room = if f.fixed() {
            0
        } else {
            f.max_bytes(total_phys).saturating_sub(current)
        };
        let Some(root) = volume_of(f) else { continue };
        let key = root.to_ascii_uppercase();
        match volumes.iter_mut().find(|(k, _)| *k == key) {
            Some((_, r)) => *r = r.saturating_add(room),
            None => volumes.push((key, room)),
        }
    }
    volumes
        .iter()
        .map(|(root, room)| {
            let free = free_space(root).map_or(0, |b| b.saturating_sub(PAGING_DISK_MARGIN_BYTES));
            (*room).min(free)
        })
        .fold(0, u64::saturating_add)
}

/// The registry's `PagingFiles` (parsed) and `ExistingPageFiles` (the files in use,
/// `\??\C:\pagefile.sys`) values; None when `PagingFiles` cannot be read. Not cached: the value
/// is read once per training run.
#[cfg(windows)]
pub fn paging_registry() -> Option<(Vec<PagingFile>, Vec<String>)> {
    use windows::core::w;
    let files = read_multi_sz(w!("PagingFiles"))?;
    let existing = read_multi_sz(w!("ExistingPageFiles")).unwrap_or_default();
    Some((parse_paging_files(&files), existing))
}

/// None: Windows only.
#[cfg(not(windows))]
pub fn paging_registry() -> Option<(Vec<PagingFile>, Vec<String>)> {
    None
}

/// A REG_MULTI_SZ value under `Memory Management` as its strings; None when it cannot be read.
#[cfg(windows)]
fn read_multi_sz(value: windows::core::PCWSTR) -> Option<Vec<String>> {
    use windows::Win32::Foundation::{ERROR_MORE_DATA, ERROR_SUCCESS};
    use windows::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_MULTI_SZ, RegGetValueW};
    use windows::core::w;
    let key = w!(r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management");
    // Paging file lists are a few dozen characters; on ERROR_MORE_DATA (a longer value) grow to
    // the size the call asked for, a few times, then give up.
    let mut buf: Vec<u16> = vec![0; 512];
    for _ in 0..4 {
        let mut bytes = u32::try_from(buf.len() * 2).ok()?;
        // SAFETY: `key` and `value` are NUL-terminated wide strings; `buf` is live and writable
        // for `bytes` bytes, which the call sets to what it wrote (or needs); nothing is kept.
        let rc = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                key,
                value,
                RRF_RT_REG_MULTI_SZ,
                None,
                Some(buf.as_mut_ptr().cast()),
                Some(&mut bytes),
            )
        };
        if rc == ERROR_MORE_DATA {
            buf = vec![0; (bytes as usize).div_ceil(2) + 64];
            continue;
        }
        if rc != ERROR_SUCCESS {
            return None;
        }
        buf.truncate(bytes as usize / 2);
        return Some(
            buf.split(|c| *c == 0)
                .filter(|s| !s.is_empty())
                .map(String::from_utf16_lossy)
                .collect(),
        );
    }
    None
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

/// Installed and available physical memory (`GlobalMemoryStatusEx` on Windows); `None` on other
/// platforms or when the call fails.
#[cfg(windows)]
pub fn physical_bytes() -> Option<Physical> {
    let s = status()?;
    Some(Physical {
        total: s.ullTotalPhys,
        available: s.ullAvailPhys,
    })
    .filter(|p| p.total > 0)
}

/// Installed and available physical memory (`GlobalMemoryStatusEx` on Windows); `None` on other
/// platforms or when the call fails.
#[cfg(not(windows))]
pub fn physical_bytes() -> Option<Physical> {
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

    #[test]
    fn physical_is_plausible() {
        let got = super::physical_bytes();
        if cfg!(windows) {
            let p = got.unwrap();
            assert_eq!(Some(p.total), super::total_physical_bytes());
            assert!(p.available > 0 && p.available <= p.total, "{p:?}");
        } else {
            assert_eq!(got, None);
        }
    }

    /// The live registry: every Windows installation has the value (empty when paging is off).
    #[test]
    fn paging_registry_reads_on_windows_only() {
        let got = super::paging_registry();
        if cfg!(windows) {
            let (files, _existing) = got.unwrap();
            for f in &files {
                assert!(f.system_managed || f.max_mb >= f.initial_mb, "{f:?}");
            }
        } else {
            assert_eq!(got, None);
        }
    }

    fn lines(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    fn file(path: &str, initial_mb: u64, max_mb: u64) -> PagingFile {
        PagingFile {
            path: path.into(),
            initial_mb,
            max_mb,
            system_managed: false,
        }
    }

    use super::{PAGING_DISK_MARGIN_BYTES, PagingFile, growable_bytes, parse_paging_files};

    const GB: u64 = 1 << 30;

    #[test]
    fn paging_files_parse_custom_system_managed_and_skip_garbage() {
        // This box's value.
        let got = parse_paging_files(&lines(&[r"c:\pagefile.sys 2935 24576"]));
        assert_eq!(got, vec![file(r"c:\pagefile.sys", 2935, 24576)]);
        assert!(!got[0].fixed());
        assert_eq!(got[0].drive(), Some('C'));
        assert_eq!(got[0].max_bytes(16 * GB), 24 * GB);
        // Fixed.
        let fixed = &parse_paging_files(&lines(&[r"D:\pagefile.sys 5600 5600"]))[0];
        assert!(fixed.fixed());
        // System-managed: all drives (`?:`), sizes 0 0, no sizes.
        let sm = parse_paging_files(&lines(&[
            r"?:\pagefile.sys",
            r"c:\pagefile.sys 0 0",
            r"E:\pagefile.sys",
        ]));
        assert_eq!(sm.len(), 3);
        assert!(sm.iter().all(|f| f.system_managed && !f.fixed()), "{sm:?}");
        assert_eq!(sm[0].drive(), None);
        assert_eq!(sm[2].drive(), Some('E'));
        // 3 x RAM, at least 4 GB.
        assert_eq!(sm[0].max_bytes(16 * GB), 48 * GB);
        assert_eq!(sm[0].max_bytes(GB), 4 * GB);
        // A `?:` path keeps system-managed even with sizes.
        assert!(parse_paging_files(&lines(&[r"?:\pagefile.sys 100 200"]))[0].system_managed);
        // Garbage: skipped, the good lines kept.
        let got = parse_paging_files(&lines(&[
            "",
            "   ",
            "pagefile.sys 100 200",
            r"c:pagefile.sys 100 200",
            r"1:\pagefile.sys 100 200",
            r"c:\pagefile.sys 100",
            r"c:\pagefile.sys a b",
            r"c:\pagefile.sys -1 200",
            r"c:\pagefile.sys 300 200",
            r"c:\pagefile.sys 100 200 300",
            r"c:\",
            r"  d:\pagefile.sys   1024   2048  ",
        ]));
        assert_eq!(got, vec![file(r"d:\pagefile.sys", 1024, 2048)]);
    }

    #[test]
    fn volume_roots_follow_the_letter_the_existing_file_or_the_system_drive() {
        let root = super::volume_root;
        let sm = &parse_paging_files(&lines(&[r"?:\pagefile.sys"]))[0];
        let existing = lines(&[r"\??\D:\pagefile.sys"]);
        assert_eq!(
            root(&file(r"e:\p.sys", 1, 2), &existing, Some("C:")),
            Some(r"E:\".into())
        );
        assert_eq!(root(sm, &existing, Some("C:")), Some(r"D:\".into()));
        assert_eq!(root(sm, &[], Some("c:")), Some(r"C:\".into()));
        assert_eq!(root(sm, &[], None), None);
        assert_eq!(root(sm, &lines(&["junk"]), Some("")), None);
    }

    /// Commit limit = RAM + the paging file (as GlobalMemoryStatusEx sees it), free space per root.
    fn grow(files: &[PagingFile], limit: u64, ram: u64, free: &[(&str, u64)]) -> u64 {
        growable_bytes(
            files,
            limit,
            ram,
            |f| super::volume_root(f, &lines(&[r"\??\C:\pagefile.sys"]), Some("C:")),
            |root| free.iter().find(|(r, _)| *r == root).map(|(_, b)| *b),
        )
    }

    #[test]
    fn growth_is_the_maximum_less_the_current_size_within_the_free_disk() {
        let ram = 16 * GB;
        let this_box = parse_paging_files(&lines(&[r"c:\pagefile.sys 2935 24576"]));
        // A 7.5 GB file now (limit 23.5), 24 GB max: 16.5 GB to go; plenty of disk.
        assert_eq!(
            grow(&this_box, ram + 15 * GB / 2, ram, &[(r"C:\", 100 * GB)]),
            33 * GB / 2
        );
        // 31 GB free on C: keeps 16: 15.
        assert_eq!(
            grow(&this_box, ram + 15 * GB / 2, ram, &[(r"C:\", 31 * GB)]),
            15 * GB
        );
        // 19 GB free keeps 16: 3; 15 GB free (this box on 2026-11-18): none.
        assert_eq!(
            grow(&this_box, ram + 15 * GB / 2, ram, &[(r"C:\", 19 * GB)]),
            3 * GB
        );
        assert_eq!(
            grow(&this_box, ram + 15 * GB / 2, ram, &[(r"C:\", 15 * GB)]),
            0
        );
        // A full disk (or under the margin): no growth.
        assert_eq!(grow(&this_box, ram + 7 * GB, ram, &[(r"C:\", 3 * GB)]), 0);
        assert_eq!(
            grow(
                &this_box,
                ram + 7 * GB,
                ram,
                &[(r"C:\", PAGING_DISK_MARGIN_BYTES)]
            ),
            0
        );
        // The free space cannot be read: no growth.
        assert_eq!(grow(&this_box, ram + 7 * GB, ram, &[]), 0);
        // Already at its maximum (or the limit says more): none.
        assert_eq!(
            grow(&this_box, ram + 25 * GB, ram, &[(r"C:\", 100 * GB)]),
            0
        );
        // Fixed: none, whatever the disk.
        let fixed = parse_paging_files(&lines(&[r"c:\pagefile.sys 5600 5600"]));
        assert_eq!(grow(&fixed, ram + 3 * GB, ram, &[(r"C:\", 100 * GB)]), 0);
        // No paging file: none.
        assert_eq!(grow(&[], ram, ram, &[(r"C:\", 100 * GB)]), 0);
        // System-managed (`?:`, on C: per ExistingPageFiles): up to 3 x RAM.
        let sm = parse_paging_files(&lines(&[r"?:\pagefile.sys"]));
        assert_eq!(grow(&sm, ram + 8 * GB, ram, &[(r"C:\", 100 * GB)]), 40 * GB);
        assert_eq!(grow(&sm, ram + 8 * GB, ram, &[(r"C:\", 22 * GB)]), 6 * GB);
        // Two files: the current 12 GB shared 1:2 by initial size (4 + 8); each volume its own cap.
        let two = parse_paging_files(&lines(&[
            r"c:\pagefile.sys 4096 16384",
            r"d:\pagefile.sys 8192 8192",
        ]));
        let plenty = [(r"C:\", 100 * GB), (r"D:\", 100 * GB)];
        assert_eq!(grow(&two, ram + 12 * GB, ram, &plenty), 12 * GB);
        let tight = [(r"C:\", 21 * GB), (r"D:\", 100 * GB)];
        assert_eq!(grow(&two, ram + 12 * GB, ram, &tight), 5 * GB);
        // Initial sizes of 0 (weights 0): equal shares of the current 4 GB.
        let zero = parse_paging_files(&lines(&[r"c:\a.sys 0 8192", r"d:\b.sys 0 8192"]));
        assert_eq!(grow(&zero, ram + 4 * GB, ram, &plenty), 12 * GB);
        // Two growable files on one volume share its free space.
        let same = parse_paging_files(&lines(&[
            r"c:\pagefile.sys 4096 16384",
            r"C:\pagefile2.sys 4096 16384",
        ]));
        assert_eq!(
            grow(&same, ram + 8 * GB, ram, &[(r"C:\", 100 * GB)]),
            24 * GB
        );
        assert_eq!(
            grow(&same, ram + 8 * GB, ram, &[(r"C:\", 26 * GB)]),
            10 * GB
        );
    }
}
