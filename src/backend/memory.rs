//! Total and available physical memory, the commit charge, the paging files, the page-in rate and
//! this process's own memory from the OS, without spawning a process.
//!
//! Windows asks `GlobalMemoryStatusEx` and the registry (the FFI calls are why this lives in
//! `backend/`, the only place `unsafe` is allowed); other platforms return `None` and
//! `system_info` reads `/proc/meminfo` or `sysctl` there instead. [`process_memory`] asks
//! `K32GetProcessMemoryInfo` on Windows and `proc_pid_rusage` on macOS; Linux reads
//! `/proc/self/status` in [`crate::metrics`], with no FFI. [`PageIns`] is the system's page-in
//! (hard fault) rate: PDH on Windows, `/proc/vmstat` on Linux, none elsewhere. Never panics; a
//! failed call is `None`. [`trim`] asks the allocator to give freed heap memory back to the OS
//! (glibc `malloc_trim`, macOS `malloc_zone_pressure_relief`, Windows `HeapCompact`);
//! [`trim_soon`] asks for one from a short-lived `bop-trim` thread, coalescing bursts.
//! [`tune_allocator`] (Linux glibc, first thing in every binary's `main`) caps the malloc arenas
//! and fixes the mmap threshold unless the environment already chose ([`allocator_plan`], pure).
//!
//! Paging files: [`parse_paging_files`] reads the registry's `PagingFiles` lines and
//! [`growable_bytes`] (pure) how far they can still grow. That is commit a process gets on
//! demand: Windows grows a paging file whose maximum is above its current size as the commit
//! charge nears the limit (measured on a 16 GB box with `c:\pagefile.sys 2935 24576`: the file
//! went from 7.6 to 16.9 GB during one training run, with no WinError 1455). Each volume keeps
//! [`PAGING_DISK_MARGIN_BYTES`] free; a system-managed file may reach [`system_managed_max`]
//! (3 x RAM, at least 4 GB, Windows' own rule).

use crate::sync::LockExt;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

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
    /// Memory the process owns alone, whether in RAM or not (audit 2026-11, memgap-2): Windows'
    /// private bytes (`PrivateUsage`, the commit charge WinError 1455 is about; the OS trims the
    /// working set under pressure without lowering it), the physical footprint on macOS. Linux
    /// reads `RssAnon` in [`crate::metrics::ProcessStats`].
    pub private: Option<u64>,
}

/// [`ProcessMemory`] from `K32GetProcessMemoryInfo` (the `_EX` counters, for `PrivateUsage`);
/// None when the call fails.
#[cfg(windows)]
pub fn process_memory() -> Option<ProcessMemory> {
    use windows::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;
    let cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>()).ok()?;
    let mut counters = PROCESS_MEMORY_COUNTERS_EX {
        cb,
        ..PROCESS_MEMORY_COUNTERS_EX::default()
    };
    // SAFETY: the pseudo-handle of the current process is always valid and needs no closing;
    // `counters` is live and writable for `cb` bytes (its own size). The EX struct starts with
    // the plain one's fields, and the API takes it through this pointer type when `cb` is the EX
    // size (documented for GetProcessMemoryInfo); it is only written during the call.
    let done = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            std::ptr::from_mut(&mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
            cb,
        )
    };
    done.as_bool().then_some(ProcessMemory {
        resident: counters.WorkingSetSize as u64,
        footprint: None,
        private: Some(counters.PrivateUsage as u64),
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
        // macOS has no separate private-bytes counter; the footprint is what the process owns.
        private: Some(usage.ri_phys_footprint),
    })
}

/// None: Linux reads `/proc/self/status` ([`crate::metrics::ProcessStats`]).
#[cfg(not(any(windows, target_os = "macos")))]
pub fn process_memory() -> Option<ProcessMemory> {
    None
}

/// What one [`trim`] call did and could measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trimmed {
    /// The allocator was asked to give memory back (false on musl and other platforms, where
    /// [`trim`] is a no-op).
    pub asked: bool,
    /// Bytes the allocator reports it released: macOS only (`malloc_zone_pressure_relief`'s
    /// answer). glibc's `malloc_trim` answers only "something / nothing" and `HeapCompact` the
    /// largest free block, neither of which is a released amount.
    pub released: Option<u64>,
    /// This process's resident memory before and after the call (working set on Windows, the
    /// resident size on macOS, `VmRSS` on Linux), when it can be read.
    pub resident_before: Option<u64>,
    pub resident_after: Option<u64>,
    /// macOS only: the physical footprint before and after (what Activity Monitor shows).
    pub footprint_before: Option<u64>,
    pub footprint_after: Option<u64>,
    /// Private bytes before and after ([`ProcessMemory::private`]; `RssAnon` on Linux): on
    /// Windows the figure that shows whether the trim lowered the commit, which the working set
    /// alone cannot.
    pub private_before: Option<u64>,
    pub private_after: Option<u64>,
}

/// Give freed heap memory back to the OS, once, on the calling thread (audit 2026-11, memgap-1):
/// Linux glibc `malloc_trim(0)` (also returns free pages in the middle of every arena), macOS
/// `malloc_zone_pressure_relief(NULL, 0)` (every zone), Windows `HeapCompact` on the process heap
/// only (the heap Rust std and the UCRT malloc of ONNX Runtime / OpenVINO use; other modules'
/// heaps may be `HEAP_NO_SERIALIZE` and are never touched; never `EmptyWorkingSet`, which lowers
/// no commit and makes the next detection fault the model back in). No-op elsewhere.
///
/// It recovers only what the allocator holds free: live runtime pools (ORT's CPU arena, OpenVINO
/// compiled models) count as in use. Cheap, but it takes the allocator's locks while it walks
/// them, so callers run it after a big release (a retired generation or worker), never per
/// request. Logs one debug line with the before / after resident memory.
pub fn trim() -> Trimmed {
    let before = crate::metrics::ProcessStats::read();
    let (asked, released) = trim_now();
    let after = crate::metrics::ProcessStats::read();
    let out = Trimmed {
        asked,
        released,
        resident_before: before.resident_memory_bytes,
        resident_after: after.resident_memory_bytes,
        footprint_before: before.memory_footprint_bytes,
        footprint_after: after.memory_footprint_bytes,
        private_before: before.private_memory_bytes,
        private_after: after.private_memory_bytes,
    };
    let mb = |b: Option<u64>| b.map_or_else(|| "?".to_string(), |b| format!("{:.1}", mib(b)));
    tracing::debug!(
        "memory trim: asked={asked} released_mb={} resident_mb={} -> {} private_mb={} -> {} \
         footprint_mb={} -> {}",
        mb(out.released),
        mb(out.resident_before),
        mb(out.resident_after),
        mb(out.private_before),
        mb(out.private_after),
        mb(out.footprint_before),
        mb(out.footprint_after),
    );
    out
}

/// Quiet time after the last [`trim_soon`] request before the trim runs: a burst of releases (a
/// generation's workers retiring one by one, the last sign-ins of a burst) becomes one trim.
pub const TRIM_QUIET: Duration = Duration::from_secs(2);
/// A trim runs at most this long after the first request of a burst, however long requests keep
/// coming (a steady stream must not postpone it forever).
pub const TRIM_MAX_DELAY: Duration = Duration::from_secs(15);
/// Least time between two trims: each takes the allocator's locks while it walks the heap, so a
/// caller that asks often (many small jobs) costs at most one walk per this interval.
pub const TRIM_MIN_GAP: Duration = Duration::from_secs(10);

/// Trims run by [`trim_soon`]'s thread so far (tests and the debug log).
static TRIMS_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Default)]
struct TrimState {
    /// First and latest request of the pending burst; `None` = nothing pending.
    first: Option<Instant>,
    latest: Option<Instant>,
    reasons: Vec<&'static str>,
    /// When the previous trim ran.
    last: Option<Instant>,
    /// A `bop-trim` thread is alive (it exits once nothing is pending).
    thread: bool,
}

static TRIM: (Mutex<TrimState>, Condvar) = (
    Mutex::new(TrimState {
        first: None,
        latest: None,
        reasons: Vec::new(),
        last: None,
        thread: false,
    }),
    Condvar::new(),
);

/// When the pending burst's trim runs (pure): [`TRIM_QUIET`] after its latest request, at most
/// [`TRIM_MAX_DELAY`] after its first, and never sooner than [`TRIM_MIN_GAP`] after the last trim.
pub fn trim_due(first: Instant, latest: Instant, last: Option<Instant>) -> Instant {
    let quiet = (latest + TRIM_QUIET).min(first + TRIM_MAX_DELAY);
    last.map_or(quiet, |l| quiet.max(l + TRIM_MIN_GAP))
}

/// Ask for a [`trim`] soon, off the calling thread (audit 2026-11, phase 5 trim points). Returns at
/// once; requests close together become one trim ([`trim_due`]). For callers that just released a
/// lot (a retired worker or generation, a finished job, benchmark or export, the end of a sign-in
/// burst). `reason` names the caller in the debug log. The `bop-trim` thread lives only while a
/// trim is pending; a failed thread spawn skips that trim (logged).
pub fn trim_soon(reason: &'static str) {
    let (lock, wake) = &TRIM;
    let mut s = lock.lock_recover();
    let now = Instant::now();
    s.first.get_or_insert(now);
    s.latest = Some(now);
    if !s.reasons.contains(&reason) {
        s.reasons.push(reason);
    }
    if s.thread {
        wake.notify_all();
        return;
    }
    s.thread = true;
    drop(s);
    let spawned = std::thread::Builder::new()
        .name("bop-trim".into())
        .spawn(trim_thread);
    if let Err(e) = spawned {
        tracing::debug!("memory trim skipped: could not start its thread ({e})");
        let mut s = lock.lock_recover();
        s.thread = false;
        s.first = None;
        s.latest = None;
        s.reasons.clear();
    }
}

fn trim_thread() {
    let (lock, wake) = &TRIM;
    let mut s = lock.lock_recover();
    loop {
        let (Some(first), Some(latest)) = (s.first, s.latest) else {
            s.thread = false;
            return;
        };
        let due = trim_due(first, latest, s.last);
        let now = Instant::now();
        if now < due {
            s = wake
                .wait_timeout(s, due - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            continue;
        }
        let reasons = std::mem::take(&mut s.reasons);
        s.first = None;
        s.latest = None;
        drop(s);
        tracing::debug!("memory trim for: {}", reasons.join(", "));
        let _ = trim();
        TRIMS_RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        s = lock.lock_recover();
        s.last = Some(Instant::now());
    }
}

/// How many trims [`trim_soon`] has run in this process.
pub fn trims_run() -> u64 {
    TRIMS_RUN.load(std::sync::atomic::Ordering::Relaxed)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 20)
}

/// glibc malloc arenas [`tune_allocator`] allows when the environment does not say (the systemd
/// unit and the Docker image have set `MALLOC_ARENA_MAX=2` since 0.2): glibc otherwise gives each
/// allocating thread its own arena, up to 8 per core, and the HTTP, worker, prep, collector and
/// runtime threads all touch image-sized buffers, so a long run leaves many fragmented arenas
/// that are rarely given back and resident memory creeps up for days. Two keep that bounded for
/// a little allocator contention.
pub const ARENA_MAX: i32 = 2;

/// The fixed glibc mmap threshold [`tune_allocator`] sets when the environment does not say:
/// 8 MiB. Blocks at least this large are mmapped and go back to the OS when freed.
///
/// glibc's threshold is dynamic by default: it starts at 128 KiB and rises to the size of every
/// mmapped block freed, up to 32 MiB, so after the first 4K frame (24.9 MB decoded) or 1280-input
/// tensor (19.7 MB) every later buffer of that size is served from an arena and stays there after
/// a burst (and the trim threshold follows to twice that). Fixing it also stops that climb and
/// keeps the trim threshold at 128 KiB. 8 MiB sits above every per-request buffer of a 640 model
/// (the 4.9 MB tensor, a 1080p frame's 6.2 MB, a 2.8 MB output), so the hot path pays no
/// mmap / munmap and ~1,200 zeroed page faults per buffer; frame-sized tensors are pooled
/// ([`crate::prep::TensorPool`]) and the decode buffer is reused per thread, so what goes to mmap
/// is the rare big one-off (a 1440p or 4K frame that is kept, a benchmark image) that should not
/// pin heap after it is freed.
pub const MMAP_THRESHOLD: i32 = 8 << 20;

/// The GLIBC_TUNABLES names of the two settings (`name=value` entries separated by `:`).
const ARENA_TUNABLE: &str = "glibc.malloc.arena_max";
const MMAP_TUNABLE: &str = "glibc.malloc.mmap_threshold";

/// What [`tune_allocator`] sets (`None`: left as the environment chose).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorPlan {
    /// `mallopt(M_ARENA_MAX, ..)`.
    pub arena_max: Option<i32>,
    /// `mallopt(M_MMAP_THRESHOLD, ..)`.
    pub mmap_threshold: Option<i32>,
}

/// What to set given the environment `var` reads (pure, every OS): [`ARENA_MAX`] unless
/// `MALLOC_ARENA_MAX` is set or `GLIBC_TUNABLES` names `glibc.malloc.arena_max`; [`MMAP_THRESHOLD`]
/// unless `MALLOC_MMAP_THRESHOLD_` is set or `GLIBC_TUNABLES` names `glibc.malloc.mmap_threshold`.
/// A value set by the user (any value: glibc has already applied or rejected it) always wins.
pub fn allocator_plan(var: impl Fn(&str) -> Option<std::ffi::OsString>) -> AllocatorPlan {
    let tunables = var("GLIBC_TUNABLES");
    let tunables = tunables.as_ref().map(|t| t.to_string_lossy());
    let tuned = |name: &str| {
        tunables.as_deref().is_some_and(|t| {
            t.split(':')
                .any(|entry| entry.split('=').next().is_some_and(|n| n.trim() == name))
        })
    };
    let unset = |env: &str, tunable: &str| var(env).is_none() && !tuned(tunable);
    AllocatorPlan {
        arena_max: unset("MALLOC_ARENA_MAX", ARENA_TUNABLE).then_some(ARENA_MAX),
        mmap_threshold: unset("MALLOC_MMAP_THRESHOLD_", MMAP_TUNABLE).then_some(MMAP_THRESHOLD),
    }
}

/// Linux glibc: apply [`allocator_plan`] of this process's environment with `mallopt` (audit
/// 2026-11, memgap-3 / platform-16), so a tarball run, another supervisor and every CLI command get
/// what the systemd unit and the Docker image set. Call it first thing in `main`, after
/// `panic::install()` and before any thread starts: an arena cap set later does not bound arenas
/// already made. `mallopt` does not survive `exec`, so child processes keep their own defaults.
/// No-op elsewhere (macOS and Windows have no per-thread arenas of this kind).
pub fn tune_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let plan = allocator_plan(|k| std::env::var_os(k));
        if let Some(n) = plan.arena_max {
            // SAFETY: mallopt takes glibc's own locks; M_ARENA_MAX with a positive count is a
            // documented parameter, and a refused value (return 0) leaves glibc's default.
            let _ = unsafe { libc::mallopt(libc::M_ARENA_MAX, n) };
        }
        if let Some(n) = plan.mmap_threshold {
            // SAFETY: as above; M_MMAP_THRESHOLD with a value under glibc's 32 MiB maximum.
            let _ = unsafe { libc::mallopt(libc::M_MMAP_THRESHOLD, n) };
        }
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_now() -> (bool, Option<u64>) {
    // SAFETY: malloc_trim takes glibc's arena locks itself and is safe to call from any thread;
    // 0 = keep no extra padding at the top of the main heap.
    let _ = unsafe { libc::malloc_trim(0) };
    (true, None)
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// libsystem_malloc (macOS 10.7+): ask `zone` (null = every zone) to release free memory,
    /// `goal` bytes (0 = as much as possible); returns the bytes released.
    fn malloc_zone_pressure_relief(
        zone: *mut libc::malloc_zone_t,
        goal: libc::size_t,
    ) -> libc::size_t;
}

#[cfg(target_os = "macos")]
fn trim_now() -> (bool, Option<u64>) {
    // SAFETY: a null zone means "every registered zone"; the allocator takes its own locks.
    let released = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    (true, u64::try_from(released).ok())
}

#[cfg(windows)]
fn trim_now() -> (bool, Option<u64>) {
    use windows::Win32::System::Memory::{GetProcessHeap, HEAP_FLAGS, HeapCompact};
    // SAFETY: GetProcessHeap has no preconditions; the process heap lives as long as the process.
    let Ok(heap) = (unsafe { GetProcessHeap() }) else {
        return (false, None);
    };
    // SAFETY: the process heap is serialized (no HEAP_NO_SERIALIZE), so HeapCompact may run
    // beside other threads' allocations; it returns the largest free block (not a released size).
    let _ = unsafe { HeapCompact(heap, HEAP_FLAGS(0)) };
    (true, None)
}

#[cfg(not(any(
    all(target_os = "linux", target_env = "gnu"),
    target_os = "macos",
    windows
)))]
fn trim_now() -> (bool, Option<u64>) {
    (false, None)
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

/// The system's page-in rate, pages read from disk per second because a process touched memory
/// that was not in RAM (hard faults): what a box short of RAM does instead of failing. Each
/// [`PageIns::rate`] is the average since the previous call (or since [`PageIns::open`]).
///
/// Windows: the PDH counter `\Memory\Pages Input/sec` (English name, so a localized Windows
/// finds it too), the figure Resource Monitor's "Hard Faults/sec" is close to. Linux:
/// `pgmajfault` of `/proc/vmstat` ([`parse_vmstat_majfault`]; major faults, which read a page
/// from disk or swap). Elsewhere none: [`PageIns::open`] is None. Owned by one thread (the PDH
/// handles are not `Send`).
pub struct PageIns {
    #[cfg(windows)]
    query: windows::Win32::System::Performance::PDH_HQUERY,
    #[cfg(windows)]
    counter: windows::Win32::System::Performance::PDH_HCOUNTER,
    #[cfg(target_os = "linux")]
    last: (u64, std::time::Instant),
}

impl std::fmt::Debug for PageIns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageIns").finish_non_exhaustive()
    }
}

#[cfg(windows)]
impl PageIns {
    /// Opens the counter and takes its first reading; None when PDH refuses (a broken
    /// performance counter registry).
    pub fn open() -> Option<Self> {
        use windows::Win32::System::Performance::{
            PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
            PdhOpenQueryW,
        };
        use windows::core::{PCWSTR, w};
        let mut query = PDH_HQUERY::default();
        // SAFETY: a null data source = live data; `query` is a live out-parameter written only
        // during the call.
        if unsafe { PdhOpenQueryW(PCWSTR::null(), 0, &mut query) } != 0 {
            return None;
        }
        let mut counter = PDH_HCOUNTER::default();
        // SAFETY: `query` was opened above; the path is a NUL-terminated wide string literal;
        // `counter` is a live out-parameter written only during the call.
        let added = unsafe {
            PdhAddEnglishCounterW(query, w!(r"\Memory\Pages Input/sec"), 0, &mut counter)
        };
        // SAFETY: `query` is open; collecting fills the counter's first raw value (a rate
        // needs two).
        if added != 0 || unsafe { PdhCollectQueryData(query) } != 0 {
            // SAFETY: `query` is open and closed once here; nothing uses it afterwards.
            unsafe { PdhCloseQuery(query) };
            return None;
        }
        Some(Self { query, counter })
    }

    /// Pages read in per second since the last call; None when a reading fails.
    pub fn rate(&mut self) -> Option<f64> {
        use windows::Win32::System::Performance::{
            PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE, PDH_FMT_DOUBLE,
            PdhCollectQueryData, PdhGetFormattedCounterValue,
        };
        // SAFETY: `self.query` stays open until drop.
        if unsafe { PdhCollectQueryData(self.query) } != 0 {
            return None;
        }
        let mut value = PDH_FMT_COUNTERVALUE::default();
        // SAFETY: `self.counter` belongs to the open query; `value` is a live out-parameter
        // written only during the call; the type out-parameter is not asked for.
        let rc =
            unsafe { PdhGetFormattedCounterValue(self.counter, PDH_FMT_DOUBLE, None, &mut value) };
        if rc != 0 || !matches!(value.CStatus, PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA) {
            return None;
        }
        // SAFETY: PDH_FMT_DOUBLE asked for the `doubleValue` member, which the call filled.
        let v = unsafe { value.Anonymous.doubleValue };
        (v.is_finite() && v >= 0.0).then_some(v)
    }
}

#[cfg(windows)]
impl Drop for PageIns {
    fn drop(&mut self) {
        // SAFETY: the query was opened in `open` and is closed once, here (closing it also
        // removes its counter).
        unsafe { windows::Win32::System::Performance::PdhCloseQuery(self.query) };
    }
}

#[cfg(target_os = "linux")]
impl PageIns {
    /// Takes the first reading of `/proc/vmstat`; None when it cannot be read.
    pub fn open() -> Option<Self> {
        let n = read_majfault()?;
        Some(Self {
            last: (n, std::time::Instant::now()),
        })
    }

    /// Major faults per second since the last call; None when a reading fails.
    pub fn rate(&mut self) -> Option<f64> {
        let n = read_majfault()?;
        let now = std::time::Instant::now();
        let (before, at) = std::mem::replace(&mut self.last, (n, now));
        let secs = now.duration_since(at).as_secs_f64();
        (secs > 0.0).then(|| n.saturating_sub(before) as f64 / secs)
    }
}

#[cfg(target_os = "linux")]
fn read_majfault() -> Option<u64> {
    parse_vmstat_majfault(&std::fs::read_to_string("/proc/vmstat").ok()?)
}

#[cfg(not(any(windows, target_os = "linux")))]
impl PageIns {
    /// None: no page-in counter is read on this platform.
    pub fn open() -> Option<Self> {
        None
    }

    /// None (never opened here).
    pub fn rate(&mut self) -> Option<f64> {
        None
    }
}

/// `pgmajfault` of `/proc/vmstat` text (pure; `<name> <count>` lines).
pub fn parse_vmstat_majfault(text: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let (k, v) = l.split_once(' ')?;
        (k == "pgmajfault").then(|| v.trim().parse().ok()).flatten()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn trim_returns_and_reports_what_it_did() {
        let block = vec![1u8; 32 << 20];
        std::hint::black_box(&block);
        drop(block);
        let t = super::trim();
        let expected = cfg!(any(
            all(target_os = "linux", target_env = "gnu"),
            target_os = "macos",
            windows
        ));
        assert_eq!(t.asked, expected, "{t:?}");
        assert_eq!(t.released.is_some(), cfg!(target_os = "macos"), "{t:?}");
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            assert!(
                t.resident_before.is_some() && t.resident_after.is_some(),
                "{t:?}"
            );
        }
        assert_eq!(
            t.footprint_after.is_some(),
            cfg!(target_os = "macos"),
            "{t:?}"
        );
        // A second call right after is harmless.
        let _ = super::trim();
    }

    #[test]
    fn trim_due_coalesces_a_burst_and_keeps_the_gap() {
        use super::{TRIM_MAX_DELAY, TRIM_MIN_GAP, TRIM_QUIET, trim_due};
        use std::time::Duration;
        let t0 = std::time::Instant::now();
        // One request: quiet time after it.
        assert_eq!(trim_due(t0, t0, None), t0 + TRIM_QUIET);
        // A later request in the burst pushes it back...
        let t1 = t0 + Duration::from_secs(1);
        assert_eq!(trim_due(t0, t1, None), t1 + TRIM_QUIET);
        // ...but never past the burst's cap.
        let late = t0 + TRIM_MAX_DELAY + Duration::from_secs(5);
        assert_eq!(trim_due(t0, late, None), t0 + TRIM_MAX_DELAY);
        // A recent trim holds the next one off for the gap.
        assert_eq!(trim_due(t0, t0, Some(t0)), t0 + TRIM_MIN_GAP);
        // An old one does not.
        let past = t0.checked_sub(Duration::from_secs(60));
        if let Some(past) = past {
            assert_eq!(trim_due(t0, t0, Some(past)), t0 + TRIM_QUIET);
        }
    }

    /// The thread runs one trim for a burst and exits once nothing is pending.
    #[test]
    fn trim_soon_runs_one_trim_for_a_burst_then_stops_its_thread() {
        let before = super::trims_run();
        for _ in 0..5 {
            super::trim_soon("test burst");
        }
        let deadline = std::time::Instant::now() + super::TRIM_MAX_DELAY + super::TRIM_MIN_GAP;
        while super::trims_run() == before && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(super::trims_run() > before, "no trim ran");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let alive = super::TRIM.0.lock().map(|s| s.thread).unwrap_or(true);
            if !alive {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "bop-trim thread did not exit"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

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

    /// Private bytes are read where the OS has them, and are plausible: at least a MiB, and on
    /// macOS the footprint (its stand-in). Growth is not asserted (other tests free memory).
    #[cfg(any(windows, target_os = "macos"))]
    #[test]
    fn private_bytes_are_read_where_supported() {
        let m = super::process_memory().unwrap();
        assert!(m.private.is_some_and(|b| b > 1 << 20), "{m:?}");
        if cfg!(target_os = "macos") {
            assert_eq!(m.private, m.footprint);
        }
        let t = super::trim();
        assert!(
            t.private_before.is_some() && t.private_after.is_some(),
            "{t:?}"
        );
    }

    #[test]
    fn allocator_plan_leaves_what_the_environment_chose() {
        use super::{AllocatorPlan, MMAP_THRESHOLD, allocator_plan};
        use std::ffi::OsString;
        let plan = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, OsString)> = vars
                .iter()
                .map(|(k, v)| ((*k).to_string(), OsString::from(v)))
                .collect();
            allocator_plan(move |k| vars.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()))
        };
        let both = AllocatorPlan {
            arena_max: Some(2),
            mmap_threshold: Some(MMAP_THRESHOLD),
        };
        let none = AllocatorPlan {
            arena_max: None,
            mmap_threshold: None,
        };
        // Nothing set: both.
        assert_eq!(plan(&[]), both);
        // Unrelated variables and tunables change nothing.
        assert_eq!(
            plan(&[
                ("MALLOC_TRIM_THRESHOLD_", "1"),
                (
                    "GLIBC_TUNABLES",
                    "glibc.malloc.tcache_count=0:glibc.malloc.arena_test=4"
                ),
            ]),
            both
        );
        // The variables (any value, even empty) win.
        let arena_only = AllocatorPlan {
            arena_max: None,
            ..both
        };
        let mmap_only = AllocatorPlan {
            mmap_threshold: None,
            ..both
        };
        assert_eq!(plan(&[("MALLOC_ARENA_MAX", "4")]), arena_only);
        assert_eq!(plan(&[("MALLOC_ARENA_MAX", "")]), arena_only);
        assert_eq!(plan(&[("MALLOC_MMAP_THRESHOLD_", "131072")]), mmap_only);
        assert_eq!(
            plan(&[("MALLOC_ARENA_MAX", "2"), ("MALLOC_MMAP_THRESHOLD_", "1")]),
            none
        );
        // Without the trailing underscore it is not glibc's variable.
        assert_eq!(plan(&[("MALLOC_MMAP_THRESHOLD", "1")]), both);
        // Tunables, alone or in a list, by exact name.
        assert_eq!(
            plan(&[("GLIBC_TUNABLES", "glibc.malloc.arena_max=1")]),
            arena_only
        );
        assert_eq!(
            plan(&[(
                "GLIBC_TUNABLES",
                "glibc.malloc.tcache_count=0:glibc.malloc.mmap_threshold=65536"
            )]),
            mmap_only
        );
        assert_eq!(
            plan(&[(
                "GLIBC_TUNABLES",
                "glibc.malloc.mmap_threshold=65536:glibc.malloc.arena_max=8"
            )]),
            none
        );
        assert_eq!(
            plan(&[("GLIBC_TUNABLES", "glibc.malloc.arena_max_x=1")]),
            both
        );
        // A variable and a tunable, one each.
        assert_eq!(
            plan(&[
                ("MALLOC_ARENA_MAX", "2"),
                ("GLIBC_TUNABLES", "glibc.malloc.mmap_threshold=1048576"),
            ]),
            none
        );
        // The threshold stays under glibc's 32 MiB maximum (a larger one is refused).
        const { assert!(MMAP_THRESHOLD > 0 && MMAP_THRESHOLD <= 32 << 20) };
    }

    /// Harmless to call again (threads already run here): it only sets glibc parameters.
    #[test]
    fn tune_allocator_is_callable_anywhere() {
        super::tune_allocator();
        let v = vec![0u8; 16 << 20];
        std::hint::black_box(&v);
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

    #[test]
    fn vmstat_majfault_is_parsed() {
        let text = "nr_free_pages 1234\npgmajfault 98765\npgmajfault_s 1\n";
        assert_eq!(super::parse_vmstat_majfault(text), Some(98765));
        assert_eq!(super::parse_vmstat_majfault("pgfault 5\n"), None);
        assert_eq!(super::parse_vmstat_majfault("pgmajfault x\n"), None);
    }

    /// The live counter: open on Windows and Linux, a rate that is a plain non-negative number.
    #[test]
    fn page_ins_read_where_supported() {
        let got = super::PageIns::open();
        if cfg!(any(windows, target_os = "linux")) {
            let mut c = got.expect("the page-in counter opens");
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = c.rate();
            assert!(r.is_some_and(|r| r.is_finite() && r >= 0.0), "{r:?}");
        } else {
            assert!(got.is_none());
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
