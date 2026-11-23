//! Registry of child processes the service started (`<data root>/.children.json`; audit 2026-11,
//! orphan sweep). Every long-running child (export, training, auto-label, import, refine) is
//! recorded with its PID, start time and executable when it starts ([`Registry::record`]) and
//! forgotten when it is reaped (the [`Recorded`] guard's drop); a start after a crash kills only
//! recorded entries whose PID, start time and exe all still match
//! ([`crate::backend::process::start_time`] / `exe_path`), so a reused PID is never touched.
//!
//! - Each entry also names its owner: the PID and start time of the process that spawned it.
//!   [`Registry::sweep`] acts only on entries whose owner is gone, so a second live service
//!   process (a CLI `train` beside the service, a service and its replacement) keeps its own.
//! - The decision is pure ([`decide`], tested with a fake [`Probe`]): owner alive = keep; child's
//!   PID reused (start time differs) = forget (a group id is not reused while its group has
//!   members, so nothing of that group is left); everything matches = kill (the group the child
//!   leads, then the PID).
//! - Child gone (or alive but no longer provably the recorded program: exe differs or
//!   unreadable, e.g. a zombie): the rest of its group may still run (`uv` under a dead leader,
//!   ffmpeg under yt-dlp). Never a blind `killpg` then: only the group's members that are
//!   provably ours are killed, one by one by PID ([`Action::KillMembers`]): listed in group
//!   `pid` ([`process::group_members`], Linux and macOS), started at or after the recorded child,
//!   exe under `<data root>/tools` (uv, the Python installs and the environments all live
//!   there), and the entry was recorded in this boot ([`process::boot_id`]; start times of
//!   another boot prove nothing). None left = forget. Windows lists nothing: the Job held the
//!   tree.
//! - Every read-modify-write holds `<data root>/.children.lock` (fs4, through
//!   [`crate::fsx::lock_wait`]) plus a process mutex, and writes through
//!   [`crate::fsx::write_atomic`]. A lock the file system cannot take falls back to the process
//!   mutex (warned once). A damaged file is treated as empty and rewritten (it only holds PIDs;
//!   logged once); a file of a newer format is refused (written by a newer version) and never
//!   rewritten.
//!
//! The exe is read when the child is recorded (right after `spawn`, which returns once the child
//! has exec'd). A child that later execs another program (`sh -c 'exec ...'`, a launcher shim
//! such as a macOS framework Python re-executing `Python.app`) then no longer matches the record:
//! it is not killed as a tree, only by PID as a member of its own group when its new program
//! lies under `tools` (see above), else forgotten. Record the process that stays.
//!
//! File: `{ "format": 1, "children": [{ "pid", "start_time", "exe", "kind", "started_at" (UTC
//! RFC 3339), "owner_pid", "owner_start_time", "boot" (optional, absent on Windows and in
//! entries of earlier versions, which then get no group-member sweep) }] }`. `start_time` units
//! are the OS's (see `process::start_time`), so the file is only meaningful on the machine that
//! wrote it.
//!
//! [`sweep_orphans_once`] is what the service (`runner`) and the CLI commands that run jobs
//! (`fetch`, `train ...`) call before they take a job lock or trust a busy marker.

use crate::backend::process;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The registry file's format.
pub const FORMAT: u32 = 1;
/// The registry file, in the data root.
pub const FILE: &str = ".children.json";
/// Its lock file, in the data root.
pub const LOCK_FILE: &str = ".children.lock";
/// How long a writer waits for another process holding the lock (writes take milliseconds).
const LOCK_WAIT: Duration = Duration::from_secs(10);
const LOCK_POLL: Duration = Duration::from_millis(20);

/// Serializes this process's writers (also the fallback when the file lock cannot work).
static WRITE: Mutex<()> = Mutex::new(());
static WARNED_DAMAGED: AtomicBool = AtomicBool::new(false);
static WARNED_NO_LOCK: AtomicBool = AtomicBool::new(false);

/// One recorded child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub pid: u32,
    /// [`process::start_time`] of the child when it was recorded.
    pub start_time: u64,
    /// [`process::exe_path`] of the child when it was recorded.
    pub exe: PathBuf,
    /// What it is (`export`, `training`, `autolabel`, ...), for logs.
    pub kind: String,
    /// When it was recorded, UTC RFC 3339.
    pub started_at: String,
    /// The process that spawned it.
    pub owner_pid: u32,
    pub owner_start_time: u64,
    /// [`process::boot_id`] when it was recorded (`None`: unknown; no group-member sweep).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boot: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct File {
    format: u32,
    #[serde(default)]
    children: Vec<Entry>,
}

/// What the OS says about a PID; the real one is [`OsProbe`], tests use a fake.
pub trait Probe {
    fn start_time(&self, pid: u32) -> Option<u64>;
    fn exe_path(&self, pid: u32) -> Option<PathBuf>;
    /// The processes of group `pgid` (`None`: this OS cannot list them).
    fn group_members(&self, pgid: u32) -> Option<Vec<u32>>;
    /// This boot's identifier (`None`: unknown).
    fn boot_id(&self) -> Option<String>;
}

/// [`Probe`] over [`process::start_time`] / [`process::exe_path`].
#[derive(Debug, Clone, Copy, Default)]
pub struct OsProbe;

impl Probe for OsProbe {
    fn start_time(&self, pid: u32) -> Option<u64> {
        process::start_time(pid)
    }
    fn exe_path(&self, pid: u32) -> Option<PathBuf> {
        process::exe_path(pid)
    }
    fn group_members(&self, pgid: u32) -> Option<Vec<u32>> {
        process::group_members(pgid)
    }
    fn boot_id(&self) -> Option<String> {
        static BOOT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        BOOT.get_or_init(process::boot_id).clone()
    }
}

/// What [`Registry::sweep`] does with an entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Its owner is alive: leave the entry and the child alone.
    Keep,
    /// The owner is gone and nothing of the child is left that is provably ours: drop the entry.
    Forget,
    /// The owner is gone and the child still matches: kill its group and it, drop the entry.
    Kill,
    /// The owner is gone, the child is gone (or no longer provably the recorded program), and
    /// these members of its group are provably ours ([`group_orphans`]): kill each by PID, drop
    /// the entry.
    KillMembers(Vec<u32>),
}

/// The pure sweep decision for one entry; `tools` are the data root's `tools` folder (as
/// written and canonical), under which a group member's exe must lie.
pub fn decide(entry: &Entry, tools: &[PathBuf], probe: &impl Probe) -> Action {
    if probe.start_time(entry.owner_pid) == Some(entry.owner_start_time) {
        return Action::Keep;
    }
    match probe.start_time(entry.pid) {
        // The PID now names another process. A group id is not handed out while its group has
        // members, so the child's group is empty.
        Some(t) if t != entry.start_time => return Action::Forget,
        Some(_) if probe.exe_path(entry.pid).as_deref() == Some(entry.exe.as_path()) => {
            return Action::Kill;
        }
        _ => {}
    }
    let members = group_orphans(entry, tools, probe);
    if members.is_empty() {
        Action::Forget
    } else {
        Action::KillMembers(members)
    }
}

/// The members of the group `entry.pid` led that are provably ours: the entry was recorded in
/// this boot, and each member started at or after the recorded child and runs a program under
/// one of `tools`. Sorted, without duplicates; never this process. Empty when the OS cannot list
/// groups.
pub fn group_orphans(entry: &Entry, tools: &[PathBuf], probe: &impl Probe) -> Vec<u32> {
    let same_boot = entry
        .boot
        .as_deref()
        .is_some_and(|b| probe.boot_id().as_deref() == Some(b));
    if !same_boot || tools.is_empty() {
        return Vec::new();
    }
    let Some(listed) = probe.group_members(entry.pid) else {
        return Vec::new();
    };
    let me = std::process::id();
    let mut found: Vec<u32> = listed
        .into_iter()
        .filter(|&pid| pid != me && pid > 1)
        .filter(|&pid| probe.start_time(pid).is_some_and(|t| t >= entry.start_time))
        .filter(|&pid| {
            probe
                .exe_path(pid)
                .is_some_and(|exe| tools.iter().any(|t| exe.starts_with(t)))
        })
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// `<data_root>/tools`, as written and canonical (exe paths the OS reports are resolved, a data
/// root may not be: `/var` vs `/private/var` on macOS).
pub fn tools_dirs(data_root: &Path) -> Vec<PathBuf> {
    let tools = data_root.join(crate::resources::export::TOOLS_DIR);
    let mut dirs = vec![tools.clone()];
    if let Ok(real) = std::fs::canonicalize(&tools)
        && real != tools
    {
        dirs.push(real);
    }
    dirs
}

/// What a sweep kills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// A live recorded child: its group, then the PID.
    Tree(u32),
    /// One process of a group whose leader is gone: this PID only, never the group.
    Process(u32),
}

/// The real kill of a [`Target`].
fn os_kill(target: Target) -> std::result::Result<(), String> {
    match target {
        // The child leads its own group (`process::prepare`): end the tree, then the PID in case
        // it left the group (both are tried; the first error is reported).
        Target::Tree(pid) => {
            let group = process::kill_group(pid);
            let one = process::kill_pid(pid);
            group.and(one)
        }
        // Never the group: its leader is gone, so only proven members are ended.
        Target::Process(pid) => process::kill_pid(pid),
    }
    .map_err(|e| e.to_string())
}

/// How many times a sweep lists a leaderless group again after killing its members (a member
/// may have started another process meanwhile).
const MEMBER_ROUNDS: usize = 4;

/// A child [`Registry::sweep`] killed (or tried to).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swept {
    pub pid: u32,
    pub kind: String,
    pub exe: PathBuf,
    /// The kill failed (the entry is dropped all the same).
    pub error: Option<String>,
}

/// What one sweep did.
#[derive(Debug, Default)]
pub struct SweepReport {
    /// Orphans killed.
    pub killed: Vec<Swept>,
    /// Stale entries dropped without a kill (child gone, PID reused or exe differs).
    pub forgotten: usize,
    /// Entries of live owners left in place.
    pub kept: usize,
    /// The sweep could not run (lock, read or write failure, newer format); nothing was killed
    /// unless `killed` says so.
    pub error: Option<String>,
}

/// `<data root>/.children.json`.
#[derive(Debug, Clone)]
pub struct Registry {
    root: PathBuf,
}

/// This process's identity, read once.
fn me() -> Result<(u32, u64)> {
    static ME: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let pid = std::process::id();
    let start = ME
        .get_or_init(|| process::start_time(pid))
        .context("this process's start time cannot be read, so its children cannot be recorded")?;
    Ok((pid, start))
}

impl Registry {
    pub fn new(data_root: impl Into<PathBuf>) -> Self {
        Self {
            root: data_root.into(),
        }
    }

    /// The registry file's path.
    pub fn path(&self) -> PathBuf {
        self.root.join(FILE)
    }

    /// Record `child_pid` (already spawned) as a `kind` child of this process. The returned guard
    /// removes the entry when dropped or [`Recorded::forget`]ten: hold it until the child is
    /// reaped. Fails when the child's identity cannot be read (it already exited) or the file
    /// cannot be written; the child itself is untouched.
    pub fn record(&self, kind: &str, child_pid: u32) -> Result<Recorded> {
        let (owner_pid, owner_start_time) = me()?;
        self.record_as(kind, child_pid, owner_pid, owner_start_time, &OsProbe)
    }

    fn record_as(
        &self,
        kind: &str,
        child_pid: u32,
        owner_pid: u32,
        owner_start_time: u64,
        probe: &impl Probe,
    ) -> Result<Recorded> {
        let start_time = probe.start_time(child_pid).with_context(|| {
            format!("child {child_pid} ({kind}) has no start time: it exited already")
        })?;
        let exe = probe.exe_path(child_pid).with_context(|| {
            format!("child {child_pid} ({kind}): its executable path cannot be read")
        })?;
        let entry = Entry {
            pid: child_pid,
            start_time,
            exe,
            kind: kind.to_string(),
            started_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            owner_pid,
            owner_start_time,
            boot: probe.boot_id(),
        };
        self.modify(|children| {
            children.retain(|c| !(c.pid == entry.pid && c.start_time == entry.start_time));
            children.push(entry.clone());
        })?;
        Ok(Recorded {
            registry: self.clone(),
            pid: child_pid,
            start_time,
            done: false,
        })
    }

    /// Kill the recorded children of processes that are gone (when PID, start time and exe still
    /// match), and what is provably ours of the group of a child that is gone ([`decide`]), and
    /// drop every stale entry. Entries of live owners, this process's included, stay.
    pub fn sweep(&self) -> SweepReport {
        let tools = tools_dirs(&self.root);
        self.sweep_with(&OsProbe, &tools, &mut os_kill)
    }

    fn sweep_with(
        &self,
        probe: &impl Probe,
        tools: &[PathBuf],
        kill: &mut dyn FnMut(Target) -> std::result::Result<(), String>,
    ) -> SweepReport {
        let mut report = SweepReport::default();
        let done = self.modify(|children| {
            children.retain(|entry| match decide(entry, tools, probe) {
                Action::Keep => {
                    report.kept += 1;
                    true
                }
                Action::Forget => {
                    report.forgotten += 1;
                    false
                }
                Action::KillMembers(first) => {
                    kill_members(entry, first, tools, probe, kill, &mut report.killed);
                    false
                }
                Action::Kill => {
                    let error = kill(Target::Tree(entry.pid)).err();
                    match &error {
                        None => tracing::info!(
                            "ended orphaned {} process {} ({}) left by an earlier run",
                            entry.kind,
                            entry.pid,
                            entry.exe.display()
                        ),
                        Some(e) => tracing::warn!(
                            "could not end orphaned {} process {} ({}): {e}. End it by hand \
                             if it is still running",
                            entry.kind,
                            entry.pid,
                            entry.exe.display()
                        ),
                    }
                    report.killed.push(Swept {
                        pid: entry.pid,
                        kind: entry.kind.clone(),
                        exe: entry.exe.clone(),
                        error,
                    });
                    false
                }
            });
        });
        if let Err(e) = done {
            report.error = Some(format!("{e:#}"));
        }
        report
    }

    /// The entries now (for diagnostics and tests). A damaged file reads as empty.
    pub fn entries(&self) -> Result<Vec<Entry>> {
        Ok(self.load()?.children)
    }

    fn remove(&self, pid: u32, start_time: u64) -> Result<()> {
        self.modify(|children| {
            children.retain(|c| !(c.pid == pid && c.start_time == start_time));
        })
    }

    fn load(&self) -> Result<File> {
        let path = self.path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(File {
                    format: FORMAT,
                    children: Vec::new(),
                });
            }
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        match serde_json::from_slice::<File>(&bytes) {
            Ok(f) if f.format > FORMAT => bail!(
                "{} has format {}, newer than this version's {FORMAT}: it was written by a newer \
                 Blue Onyx Prism. Children are not recorded or swept until that version runs \
                 again (or the file is deleted)",
                path.display(),
                f.format
            ),
            Ok(f) => Ok(f),
            Err(e) => {
                if !WARNED_DAMAGED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "{} is damaged ({e}); it only lists child process ids, so it is started \
                         over (a child an earlier run left will not be swept)",
                        path.display()
                    );
                }
                Ok(File {
                    format: FORMAT,
                    children: Vec::new(),
                })
            }
        }
    }

    /// Read, change and write the list under the locks.
    fn modify(&self, change: impl FnOnce(&mut Vec<Entry>)) -> Result<()> {
        let _process = WRITE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("creating the data root {}", self.root.display()))?;
        let lock_path = self.root.join(LOCK_FILE);
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("opening {}", lock_path.display()))?;
        let deadline = Instant::now() + LOCK_WAIT;
        match crate::fsx::lock_wait(&lock, crate::fsx::LockKind::Exclusive, deadline, LOCK_POLL) {
            Ok(true) => {}
            Ok(false) => bail!(
                "{} is held by another process for more than {} s; the child-process list was \
                 not updated. Retry; if it persists, look for a stuck blue-onyx-prism process",
                lock_path.display(),
                LOCK_WAIT.as_secs()
            ),
            Err(e) => {
                if !WARNED_NO_LOCK.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "{} cannot be locked ({e}); the child-process list is guarded within \
                         this process only",
                        lock_path.display()
                    );
                }
            }
        }
        let mut file = self.load()?;
        let before = file.children.clone();
        change(&mut file.children);
        if file.children == before && self.path().exists() {
            return Ok(());
        }
        file.format = FORMAT;
        let bytes = serde_json::to_vec_pretty(&file).context("serializing the child list")?;
        write(&self.path(), &bytes)
    }
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::fsx::write_atomic(path, bytes).with_context(|| {
        format!(
            "writing {} (is the data root writable and the disk not full?)",
            path.display()
        )
    })
}

/// Kill `first`, members of the group of `entry`'s gone child, by PID; then list the group
/// again (up to [`MEMBER_ROUNDS`] times) for processes they started meanwhile. Each kill is
/// logged and pushed to `killed`.
fn kill_members(
    entry: &Entry,
    first: Vec<u32>,
    tools: &[PathBuf],
    probe: &impl Probe,
    kill: &mut dyn FnMut(Target) -> std::result::Result<(), String>,
    killed: &mut Vec<Swept>,
) {
    let mut done: Vec<u32> = Vec::new();
    let mut batch = first;
    for _ in 0..MEMBER_ROUNDS {
        batch.retain(|p| !done.contains(p));
        if batch.is_empty() {
            return;
        }
        for pid in std::mem::take(&mut batch) {
            let exe = probe.exe_path(pid).unwrap_or_default();
            let error = kill(Target::Process(pid)).err();
            match &error {
                None => tracing::info!(
                    "ended process {pid} ({}) left in the process group of {} process {} by an \
                     earlier run",
                    exe.display(),
                    entry.kind,
                    entry.pid
                ),
                Some(e) => tracing::warn!(
                    "could not end process {pid} ({}), left in the process group of {} process {} \
                     by an earlier run: {e}. End it by hand if it is still running",
                    exe.display(),
                    entry.kind,
                    entry.pid
                ),
            }
            killed.push(Swept {
                pid,
                kind: entry.kind.clone(),
                exe,
                error,
            });
            done.push(pid);
        }
        batch = group_orphans(entry, tools, probe);
    }
}

/// Data roots already swept by [`sweep_orphans_once`] in this process.
static SWEPT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Kill the children that earlier, now dead, processes left in `data_root`'s `.children.json`
/// ([`Registry::sweep`]), once per data root and process, and log what happened; `None` when
/// this process swept `data_root` already. Call it before anything that trusts a job's lock or
/// busy marker: the service before its retention sweeper starts, the CLI's `fetch` and `train`
/// commands before they run (their job locks die with a crashed process while its children may
/// still write; the sweep ends those children first).
pub fn sweep_orphans_once(data_root: &Path) -> Option<SweepReport> {
    {
        let mut swept = SWEPT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if swept.iter().any(|r| r == data_root) {
            return None;
        }
        swept.push(data_root.to_path_buf());
    }
    let report = Registry::new(data_root).sweep();
    if let Some(e) = &report.error {
        tracing::warn!(
            "the orphaned-process check of {} could not run: {e}. A training, export or \
             auto-label process an earlier run left may still be running; end it by hand if so",
            data_root.display()
        );
    }
    if !report.killed.is_empty() {
        tracing::info!(
            killed = report.killed.len(),
            forgotten = report.forgotten,
            "ended processes an earlier run left behind (listed above)"
        );
    } else if report.forgotten > 0 {
        tracing::debug!(
            forgotten = report.forgotten,
            "dropped child-process entries of earlier runs (those processes are gone)"
        );
    }
    Some(report)
}

/// A recorded child; dropping it (or [`Self::forget`]) removes the entry. Hold it until the
/// child has been reaped.
#[derive(Debug)]
pub struct Recorded {
    registry: Registry,
    pid: u32,
    start_time: u64,
    done: bool,
}

impl Recorded {
    /// The recorded PID.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Remove the entry now and say whether that worked.
    pub fn forget(mut self) -> Result<()> {
        self.done = true;
        self.registry.remove(self.pid, self.start_time)
    }
}

impl Drop for Recorded {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if let Err(e) = self.registry.remove(self.pid, self.start_time) {
            tracing::warn!(
                "child {} stays listed in the child-process file: {e:#}. The next start drops it",
                self.pid
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Fake {
        start: HashMap<u32, u64>,
        exe: HashMap<u32, PathBuf>,
        groups: HashMap<u32, Vec<u32>>,
        boot: Option<String>,
    }

    impl Fake {
        fn with(mut self, pid: u32, start: u64, exe: &str) -> Self {
            self.start.insert(pid, start);
            self.exe.insert(pid, PathBuf::from(exe));
            self
        }
        fn group(mut self, pgid: u32, members: &[u32]) -> Self {
            self.groups.insert(pgid, members.to_vec());
            self
        }
        fn booted(mut self, id: &str) -> Self {
            self.boot = Some(id.to_string());
            self
        }
    }

    impl Probe for Fake {
        fn start_time(&self, pid: u32) -> Option<u64> {
            self.start.get(&pid).copied()
        }
        fn exe_path(&self, pid: u32) -> Option<PathBuf> {
            self.exe.get(&pid).cloned()
        }
        fn group_members(&self, pgid: u32) -> Option<Vec<u32>> {
            Some(self.groups.get(&pgid).cloned().unwrap_or_default())
        }
        fn boot_id(&self) -> Option<String> {
            self.boot.clone()
        }
    }

    /// The data root's `tools` folder of these tests.
    fn tools() -> Vec<PathBuf> {
        vec![PathBuf::from("/tools")]
    }

    fn entry(pid: u32, start: u64, exe: &str, owner: u32, owner_start: u64) -> Entry {
        Entry {
            pid,
            start_time: start,
            exe: PathBuf::from(exe),
            kind: "export".into(),
            started_at: "2026-11-22T00:00:00Z".into(),
            owner_pid: owner,
            owner_start_time: owner_start,
            boot: None,
        }
    }

    const PY: &str = "/tools/env/bin/python";

    #[test]
    fn decide_covers_every_case() {
        let e = entry(500, 77, PY, 100, 11);
        // Owner alive: keep, whatever the child is.
        let alive = Fake::default().with(100, 11, "/svc").with(500, 77, PY);
        assert_eq!(decide(&e, &tools(), &alive), Action::Keep);
        let alive_child_gone = Fake::default().with(100, 11, "/svc");
        assert_eq!(decide(&e, &tools(), &alive_child_gone), Action::Keep);
        // Owner pid reused by another process (start differs): the owner is gone.
        let reused_owner = Fake::default().with(100, 12, "/other").with(500, 77, PY);
        assert_eq!(decide(&e, &tools(), &reused_owner), Action::Kill);
        // Owner gone, child matches: kill.
        let orphan = Fake::default().with(500, 77, PY);
        assert_eq!(decide(&e, &tools(), &orphan), Action::Kill);
        // Child pid reused (start differs): forget.
        let reused = Fake::default().with(500, 78, PY);
        assert_eq!(decide(&e, &tools(), &reused), Action::Forget);
        // Exe differs: forget.
        let other_exe = Fake::default().with(500, 77, "/usr/bin/vim");
        assert_eq!(decide(&e, &tools(), &other_exe), Action::Forget);
        // Exe unreadable: not provably ours.
        let mut no_exe = Fake::default().with(500, 77, PY);
        no_exe.exe.clear();
        assert_eq!(decide(&e, &tools(), &no_exe), Action::Forget);
        // Child gone: forget.
        assert_eq!(decide(&e, &tools(), &Fake::default()), Action::Forget);
    }

    /// The leader of a recorded group is gone and its owner too: only the members that are
    /// provably ours (this boot, started at or after the leader, exe under `tools`) are killed.
    #[test]
    fn decide_kills_only_provable_members_of_a_leaderless_group() {
        let mut e = entry(500, 77, PY, 100, 11);
        e.boot = Some("b1".into());
        let os = || {
            Fake::default()
                .booted("b1")
                .with(600, 80, "/tools/uv/uv") // ours
                .with(601, 70, "/tools/uv/uv") // started before the leader
                .with(602, 90, "/usr/bin/ffmpeg") // not under tools
                .with(603, 91, PY) // exe unreadable (below)
                .group(500, &[602, 600, 601, 603, 604]) // 604: gone since the listing
        };
        let mut probe = os();
        probe.exe.remove(&603);
        assert_eq!(decide(&e, &tools(), &probe), Action::KillMembers(vec![600]));
        // Our own process is never listed as a target, whatever it runs.
        let me = std::process::id();
        let with_me = os().with(me, 99, PY).group(500, &[me, 600]);
        assert_eq!(
            decide(&e, &tools(), &with_me),
            Action::KillMembers(vec![600])
        );
        // Another boot: start times prove nothing.
        assert_eq!(decide(&e, &tools(), &os().booted("b2")), Action::Forget);
        // Recorded by an earlier version (no boot): no member sweep.
        let mut old = e.clone();
        old.boot = None;
        assert_eq!(decide(&old, &tools(), &os()), Action::Forget);
        // Owner alive: nothing.
        assert_eq!(
            decide(&e, &tools(), &os().with(100, 11, "/svc")),
            Action::Keep
        );
        // Leader PID reused: its group is empty by definition, members are someone else's.
        assert_eq!(decide(&e, &tools(), &os().with(500, 5, PY)), Action::Forget);
        // Leader alive and still the recorded program: the whole tree, as before.
        assert_eq!(decide(&e, &tools(), &os().with(500, 77, PY)), Action::Kill);
        // Leader alive but now another program (a launcher that re-executed itself): it and
        // the members are killed by PID when their exe is under tools.
        let shim = os()
            .with(500, 77, "/tools/python/Python.app/Contents/MacOS/Python")
            .group(500, &[500, 600]);
        assert_eq!(
            decide(&e, &tools(), &shim),
            Action::KillMembers(vec![500, 600])
        );
        let outside = os().with(500, 77, "/usr/bin/osascript").group(500, &[500]);
        assert_eq!(decide(&e, &tools(), &outside), Action::Forget);
        // No group listing on this OS (Windows): forget.
        assert_eq!(decide(&e, &[], &os()), Action::Forget);
    }

    #[test]
    fn a_sweep_kills_leaderless_members_one_by_one_and_drops_the_entry() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let at_record = Fake::default().booted("b1").with(500, 77, PY);
        std::mem::forget(
            reg.record_as("export", 500, 100, 11, &at_record)
                .expect("record"),
        );
        assert_eq!(
            reg.entries().expect("entries")[0].boot.as_deref(),
            Some("b1")
        );
        // Leader gone; two members left, one not ours.
        let now = Fake::default()
            .booted("b1")
            .with(600, 80, "/tools/uv/uv")
            .with(601, 81, "/usr/bin/vim")
            .group(500, &[600, 601]);
        let mut killed = Vec::new();
        let report = reg.sweep_with(&now, &tools(), &mut |target| {
            killed.push(target);
            Ok(())
        });
        // Listed again after the kill (the fake does not change), never killed twice, and the
        // group itself is never signalled.
        assert_eq!(killed, vec![Target::Process(600)]);
        assert_eq!(report.killed.len(), 1);
        assert_eq!(report.killed[0].pid, 600);
        assert_eq!(report.killed[0].exe, PathBuf::from("/tools/uv/uv"));
        assert_eq!(report.error, None);
        assert!(reg.entries().expect("entries").is_empty());
    }

    /// The real OS: a recorded leader exits and leaves a member in its group. While the owner
    /// lives the sweep keeps it; once the owner is gone the member is killed by PID. (macOS kills
    /// a copied system binary, so the member is the system `sleep` and the "tools" folder of
    /// this sweep is its folder; the tools rule itself is covered by the fake-probe tests.)
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_real_leaderless_member_is_killed_only_when_its_owner_is_gone() {
        use std::io::{BufRead as _, Write as _};
        let dir = temp();
        let sleep = ["/bin/sleep", "/usr/bin/sleep"]
            .into_iter()
            .find_map(|p| std::fs::canonicalize(p).ok())
            .expect("a sleep binary");
        let tools = vec![sleep.parent().expect("folder").to_path_buf()];
        // The leader starts the member, prints its pid, and waits for a line on stdin.
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(format!(
                "'{}' 30 >/dev/null 2>&1 & echo $!; read x",
                sleep.display()
            ))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        process::prepare(&mut cmd);
        let mut leader = cmd.spawn().expect("spawn");
        let mut line = String::new();
        std::io::BufReader::new(leader.stdout.take().expect("stdout"))
            .read_line(&mut line)
            .expect("read");
        let member: u32 = line.trim().parse().expect("member pid");
        let reg = Registry::new(dir.path());
        std::mem::forget(reg.record("export", leader.id()).expect("record"));
        // The leader exits; the member stays in its group.
        let mut stdin = leader.stdin.take().expect("stdin");
        stdin.write_all(b"\n").expect("write");
        drop(stdin);
        leader.wait().expect("wait");
        let alive = |pid: u32| process::start_time(pid).is_some();
        assert!(alive(member), "the member ended with its leader");
        // Our own entry: kept, the member untouched.
        let report = reg.sweep_with(&OsProbe, &tools, &mut os_kill);
        assert_eq!(report.kept, 1, "{report:?}");
        assert!(alive(member));
        // The owner is gone (another start time recorded for it).
        reg.modify(|c| {
            for e in c.iter_mut() {
                e.owner_start_time = e.owner_start_time.wrapping_add(1);
            }
        })
        .expect("modify");
        let report = reg.sweep_with(&OsProbe, &tools, &mut os_kill);
        assert_eq!(report.error, None);
        let pids: Vec<u32> = report.killed.iter().map(|k| k.pid).collect();
        assert_eq!(pids, vec![member], "{report:?}");
        assert_eq!(report.killed[0].error, None);
        let gone = (0..100).any(|_| {
            // A killed child of init may linger as a zombie for a moment.
            if process::exe_path(member).is_none() || !alive(member) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
            false
        });
        assert!(gone, "member {member} survived");
        assert!(reg.entries().expect("entries").is_empty());
    }

    /// A scratch folder removed on drop.
    struct Temp(PathBuf);

    impl Temp {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp() -> Temp {
        let d =
            std::env::temp_dir().join(format!("bop_children_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).expect("temp dir");
        Temp(d)
    }

    #[test]
    fn record_and_drop_remove_the_entry() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let probe = Fake::default().with(500, 77, PY).with(501, 78, PY);
        let a = reg
            .record_as("export", 500, 100, 11, &probe)
            .expect("record");
        let b = reg
            .record_as("training", 501, 100, 11, &probe)
            .expect("record");
        let listed = reg.entries().expect("entries");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].exe, PathBuf::from(PY));
        assert_eq!(listed[1].kind, "training");
        assert!(chrono::DateTime::parse_from_rfc3339(&listed[0].started_at).is_ok());
        drop(a);
        assert_eq!(reg.entries().expect("entries").len(), 1);
        assert_eq!(b.pid(), 501);
        b.forget().expect("forget");
        assert!(reg.entries().expect("entries").is_empty());
        let text = std::fs::read_to_string(reg.path()).expect("file");
        let v: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v["format"], 1);
        assert_eq!(v["children"], serde_json::json!([]));
    }

    #[test]
    fn record_fails_for_a_child_that_already_exited() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let err = reg
            .record_as("export", 500, 100, 11, &Fake::default())
            .expect_err("no identity");
        assert!(format!("{err:#}").contains("exited already"), "{err:#}");
        let mut no_exe = Fake::default().with(500, 77, PY);
        no_exe.exe.clear();
        assert!(reg.record_as("export", 500, 100, 11, &no_exe).is_err());
        assert!(!reg.path().exists());
    }

    #[test]
    fn record_creates_a_missing_data_root() {
        let dir = temp();
        let reg = Registry::new(dir.path().join("a/b"));
        let probe = Fake::default().with(500, 77, PY);
        let r = reg
            .record_as("export", 500, 100, 11, &probe)
            .expect("record");
        assert_eq!(reg.entries().expect("entries").len(), 1);
        drop(r);
    }

    #[test]
    fn sweep_kills_matching_orphans_and_keeps_live_owners() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let at_record = Fake::default()
            .with(500, 77, PY)
            .with(501, 78, PY)
            .with(502, 79, PY)
            .with(503, 80, PY)
            .with(504, 81, PY);
        // Dead owner 100: 500 still matches, 501's pid was reused, 502's exe changed, 503 gone.
        // Live owner 200: 504 stays.
        let mut guards = Vec::new();
        for pid in [500, 501, 502, 503] {
            guards.push(
                reg.record_as("export", pid, 100, 11, &at_record)
                    .expect("record"),
            );
        }
        guards.push(
            reg.record_as("training", 504, 200, 22, &at_record)
                .expect("record"),
        );
        // A crash: no guard runs.
        guards.into_iter().for_each(std::mem::forget);
        let now = Fake::default()
            .with(200, 22, "/svc")
            .with(500, 77, PY)
            .with(501, 999, PY)
            .with(502, 79, "/bin/other")
            .with(504, 81, PY);
        let mut killed = Vec::new();
        let report = reg.sweep_with(&now, &tools(), &mut |target| {
            killed.push(target);
            Ok(())
        });
        assert_eq!(killed, vec![Target::Tree(500)]);
        assert_eq!(report.killed.len(), 1);
        assert_eq!(report.killed[0].pid, 500);
        assert_eq!(report.killed[0].error, None);
        assert_eq!(report.forgotten, 3);
        assert_eq!(report.kept, 1);
        assert_eq!(report.error, None);
        let left = reg.entries().expect("entries");
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].pid, 504);
    }

    #[test]
    fn a_failed_kill_is_reported_and_the_entry_dropped() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let probe = Fake::default().with(500, 77, PY);
        std::mem::forget(
            reg.record_as("export", 500, 100, 11, &probe)
                .expect("record"),
        );
        let report = reg.sweep_with(&probe, &tools(), &mut |_| Err("EPERM".into()));
        assert_eq!(report.killed.len(), 1);
        assert_eq!(report.killed[0].error.as_deref(), Some("EPERM"));
        assert!(reg.entries().expect("entries").is_empty());
    }

    #[test]
    fn a_damaged_file_is_started_over() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        std::fs::write(reg.path(), b"{not json").expect("write");
        assert!(reg.entries().expect("entries").is_empty());
        let report = reg.sweep_with(&Fake::default(), &tools(), &mut |_| Ok(()));
        assert_eq!(report.error, None);
        let probe = Fake::default().with(500, 77, PY);
        let r = reg
            .record_as("export", 500, 100, 11, &probe)
            .expect("record");
        assert_eq!(reg.entries().expect("entries").len(), 1);
        drop(r);
    }

    #[test]
    fn a_newer_format_is_refused_and_left_alone() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let newer = br#"{"format": 2, "children": [], "future": true}"#;
        std::fs::write(reg.path(), newer).expect("write");
        let probe = Fake::default().with(500, 77, PY);
        let err = reg
            .record_as("export", 500, 100, 11, &probe)
            .expect_err("newer refused");
        assert!(format!("{err:#}").contains("newer"), "{err:#}");
        let report = reg.sweep_with(&probe, &tools(), &mut |_| panic!("no kill"));
        assert!(report.error.is_some());
        assert_eq!(std::fs::read(reg.path()).expect("read"), newer);
    }

    #[test]
    fn an_unchanged_sweep_does_not_rewrite() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let probe = Fake::default().with(100, 11, "/svc").with(500, 77, PY);
        std::mem::forget(
            reg.record_as("export", 500, 100, 11, &probe)
                .expect("record"),
        );
        let before = std::fs::metadata(reg.path())
            .expect("meta")
            .modified()
            .expect("mtime");
        std::thread::sleep(Duration::from_millis(20));
        let report = reg.sweep_with(&probe, &tools(), &mut |_| panic!("no kill"));
        assert_eq!(report.kept, 1);
        let after = std::fs::metadata(reg.path())
            .expect("meta")
            .modified()
            .expect("mtime");
        assert_eq!(before, after);
    }

    #[test]
    fn concurrent_records_all_land() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let probe: Fake = (0..16).fold(Fake::default(), |f, i| f.with(1000 + i, 5, PY));
        std::thread::scope(|s| {
            for i in 0..16u32 {
                let (reg, probe) = (&reg, &probe);
                s.spawn(move || {
                    std::mem::forget(
                        reg.record_as("export", 1000 + i, 100, 11, probe)
                            .expect("record"),
                    );
                });
            }
        });
        assert_eq!(reg.entries().expect("entries").len(), 16);
    }

    /// The real OS path: a recorded child whose owner is "gone" (a wrong owner start time) is
    /// killed with its group; our own live entries are kept.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn sweep_kills_a_real_orphan_and_keeps_our_own() {
        let dir = temp();
        let reg = Registry::new(dir.path());
        let spawn = || {
            // `sleep` itself, not `sh -c 'exec sleep'`: the exe is read at record time, and a
            // child that execs another program after that no longer matches (it is forgotten).
            let mut cmd = std::process::Command::new("sleep");
            cmd.arg("30");
            process::prepare(&mut cmd);
            cmd.spawn().expect("spawn")
        };
        let (mut orphan, mut ours) = (spawn(), spawn());
        let (me, my_start) = me().expect("me");
        std::mem::forget(
            reg.record_as(
                "export",
                orphan.id(),
                me,
                my_start.wrapping_add(1),
                &OsProbe,
            )
            .expect("record orphan"),
        );
        let kept = reg.record("training", ours.id()).expect("record ours");
        let report = reg.sweep();
        assert_eq!(report.error, None);
        assert_eq!(report.killed.len(), 1, "{report:?}");
        assert_eq!(report.killed[0].pid, orphan.id());
        assert_eq!(report.killed[0].error, None);
        assert_eq!(report.kept, 1);
        let status = orphan.wait().expect("wait");
        assert!(!status.success());
        assert_eq!(
            ours.try_wait().expect("try_wait"),
            None,
            "our own child was killed"
        );
        process::kill_group(ours.id()).expect("kill");
        let _ = ours.wait();
        drop(kept);
        assert!(reg.entries().expect("entries").is_empty());
    }
}
