//! Download manager (docs/PLAN.md, "Download manager"): one worker thread, a queue, one download
//! at a time, with shared progress.
//!
//! For each [`Job`]:
//! 1. Skip when the target already holds a matching [`Manifest`] (`.installed.json`).
//! 2. Take the cross-process lock `<root>/.downloads.lock` (`File::try_lock`, an OS-level lock
//!    that is released when the process dies), so the CLI and the service never download at the
//!    same time. Re-check step 1 once it is held.
//! 3. Stream each part over HTTPS to `<file>.partial` (archives under `<root>/.downloads/`,
//!    model files next to their destination), resuming with `Range` when the server allows it.
//! 4. Check the exact size and the pinned SHA-256. A mismatch deletes the file and fails the job
//!    with "hash mismatch" (not retried automatically).
//! 5. Archives: extract the whitelisted entries ([`super::extract`]) into a staging directory
//!    next to the target, write the manifest there, then rename it into place. Model files are
//!    renamed into place one by one.
//! 6. Optionally mark an ONNX Runtime flavor active (`onnxruntime/active.txt`).
//!
//! Failures are retried with a 30 s -> 5 min exponential backoff ([`backoff`]) when
//! [`ManagerOptions::retry`] is set (the service); the CLI fails fast. Retries are bounded by
//! [`ManagerOptions::max_attempts`] (the service uses [`SERVICE_MAX_ATTEMPTS`], 12 attempts over
//! about 43 minutes): after the last one the job is `Failed { retry_at: None }` with the manual
//! `fetch` command in its message, and a blocked model fails instead of waiting forever. Queuing
//! the job again (the Download button of the Config page, a config save, a restart) starts a
//! fresh series of attempts.
//!
//! If the worker thread cannot start its runtime or HTTP client, or panics, every job it will
//! never run is marked `Failed { retry_at: None }` and later [`Manager::enqueue`] calls fail at
//! once ([`Manager::is_alive`]), so nothing waits on a dead queue. [`Manager::shutdown`] sets the
//! stop flag, which the download, hash and extraction loops check, and joins the worker for at
//! most `SHUTDOWN_JOIN` before leaving it to finish in the background.
//!
//! Every state change is
//! published in [`Status`] (shared, for `/stats` and `/v1/resources`), through the optional
//! [`ManagerOptions::on_progress`] hook (CLI progress bars) and, for terminal states, as an
//! [`Event`] to every [`Manager::subscribe`]r (the registry in phase 7.3).
//!
//! # Trust model
//!
//! Everything downloadable is pinned in `super::catalog` (reviewed commits only): an `https://`
//! URL, the exact size and the SHA-256 of every file. The client refuses any other scheme for
//! the first URL and for every redirect hop (at most `MAX_REDIRECTS`); `http://` to the
//! loopback address is accepted only in the crate's own test build, for the local fixture
//! server. A response is failed, permanently, as soon as its `Content-Length` / `Content-Range`
//! or the running byte count disagrees with the pinned size, so a swapped file never costs more
//! than one attempt; a resumed download (`Range`) is hashed as a whole file again. A size or
//! hash mismatch deletes the file and is not retried automatically. Nothing downloaded is
//! executed by this module: archives go through the whitelist in [`super::extract`] into a
//! staging directory that is swapped into place only after every entry passed; model files are
//! read by the inference runtimes.

use super::catalog::{ArchiveKind, Layout, Resource, ResourceKind};
use super::extract::{self, Target};
use crate::sync::LockExt;
use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Manifest written into a completed install directory.
pub const MANIFEST_FILE: &str = ".installed.json";
/// Cross-process download lock, under the download root.
pub(crate) const LOCK_FILE: &str = ".downloads.lock";
/// Partial and verified archive downloads, under the download root.
pub(crate) const DOWNLOADS_DIR: &str = ".downloads";

/// First retry delay.
pub(crate) const BACKOFF_MIN: Duration = Duration::from_secs(30);
/// Longest retry delay.
pub(crate) const BACKOFF_MAX: Duration = Duration::from_secs(300);

/// Attempts per job in the service before it gives up (30 + 60 + 120 + 240 + 7 x 300 s of
/// backoff between them: about 43 minutes).
pub const SERVICE_MAX_ATTEMPTS: u32 = 12;
/// How long [`Manager::shutdown`] waits for the worker before detaching it.
pub(crate) const SHUTDOWN_JOIN: Duration = Duration::from_secs(5);

/// Command that downloads `res` by hand (with `--allow-large` when it needs it).
pub fn fetch_command(res: &Resource) -> String {
    format!(
        "blue-onyx-prism fetch --resource {}{}",
        res.id,
        if res.is_large() { " --allow-large" } else { "" }
    )
}

/// Delay before retry number `attempt` (1-based): 30 s, 60 s, 120 s, 240 s, then 300 s.
pub fn backoff(attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(16);
    BACKOFF_MIN
        .checked_mul(1u32 << shift)
        .unwrap_or(BACKOFF_MAX)
        .min(BACKOFF_MAX)
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

/// `.installed.json`: what was installed into a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub version: String,
    /// SHA-256 of every downloaded part, in catalog order (empty for local archives).
    pub sha256: Vec<String>,
    /// Installed files, relative to the directory.
    pub files: Vec<String>,
}

/// Manifest location for `resource` installed in `dir`: `<dir>/.installed.json` for runtimes
/// (their directory is exclusive), `<dir>/.<model>.installed.json` for model files (a models
/// directory holds many).
pub fn manifest_path(resource: &Resource, dir: &Path) -> PathBuf {
    match resource.model_name() {
        Some(name) => dir.join(format!(".{name}{MANIFEST_FILE}")),
        None => dir.join(MANIFEST_FILE),
    }
}

/// Read a manifest file; None when absent or unreadable.
pub fn read_manifest(path: &Path) -> Option<Manifest> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn write_manifest(path: &Path, m: &Manifest) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(m)?)
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}

/// `resource` (this version) is installed in `dir` according to its manifest; for models every
/// file must also still exist.
pub fn is_installed(resource: &Resource, dir: &Path) -> bool {
    let Some(m) = read_manifest(&manifest_path(resource, dir)) else {
        return false;
    };
    m.id == resource.id
        && m.version == resource.version
        && (!resource.kind.is_plain_files()
            || resource
                .parts
                .iter()
                .all(|p| dir.join(p.file_name).is_file()))
}

// ---------------------------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------------------------

/// Exclusive cross-process download lock (`<root>/.downloads.lock`). Released on drop (and by
/// the OS when the process exits).
#[derive(Debug)]
pub(crate) struct DownloadLock {
    _file: std::fs::File,
}

impl DownloadLock {
    /// Take the lock if it is free; None when another process (or handle) holds it.
    pub fn try_acquire(root: &Path) -> Result<Option<Self>> {
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let path = root.join(LOCK_FILE);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => {
                // For humans looking at a stuck lock; the OS lock is what counts.
                let _ = file.set_len(0);
                let _ = writeln!(file, "{}", std::process::id());
                Ok(Some(Self { _file: file }))
            }
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("locking {}", path.display()))
            }
        }
    }

    /// Take the lock, polling for up to `wait` (forever when None) unless `stop` is set.
    pub fn acquire(root: &Path, wait: Option<Duration>, stop: &AtomicBool) -> Result<Self> {
        let started = Instant::now();
        let mut logged = false;
        loop {
            if let Some(l) = Self::try_acquire(root)? {
                return Ok(l);
            }
            if !logged {
                tracing::info!(
                    "another process is downloading ({}); waiting",
                    root.join(LOCK_FILE).display()
                );
                logged = true;
            }
            if stop.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            if wait.is_some_and(|w| started.elapsed() >= w) {
                return Err(anyhow!(LockBusy));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

/// The download lock is held by another process.
#[derive(Debug, thiserror::Error)]
#[error("another process holds the download lock")]
pub(crate) struct LockBusy;

/// A downloaded file does not match its pinned hash (not retried automatically).
#[derive(Debug, thiserror::Error)]
#[error(
    "hash mismatch for {file}: expected sha256 {expected}, got {actual}; the file was deleted. \
     The server sent other bytes than the pinned release (a re-uploaded file, a proxy or \
     antivirus rewriting downloads, or tampering); try again later, and report it if it persists"
)]
pub(crate) struct HashMismatch {
    pub file: String,
    pub expected: String,
    pub actual: String,
}

/// The server offers a file of another size than the pinned one (not retried automatically):
/// the pin is stale or the URL was swapped; retrying would only download it again.
#[derive(Debug, thiserror::Error)]
#[error(
    "size mismatch for {file}: the server offers {offered} bytes, pinned {expected}. The release \
     file changed upstream or a proxy serves another file; try again later, and report it if it \
     persists"
)]
pub(crate) struct SizeMismatch {
    pub file: String,
    pub expected: u64,
    /// What the server offered, or `>expected` when the body simply kept coming.
    pub offered: String,
}

/// Retry this error later? Hash and size mismatches and refused URLs are permanent.
fn retryable(e: &anyhow::Error) -> bool {
    !(e.downcast_ref::<HashMismatch>().is_some()
        || e.downcast_ref::<SizeMismatch>().is_some()
        || e.downcast_ref::<UrlRefused>().is_some()
        || e.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_redirect())
        }))
}

#[derive(Debug, thiserror::Error)]
#[error(
    "refusing to download {0}: only https URLs are allowed; the pinned URL or a redirect left \
     https (a changed server or an intercepting proxy); update Blue Onyx Prism or report it"
)]
struct UrlRefused(String);

/// Most redirect hops followed for one download (GitHub and Hugging Face use one or two).
pub(crate) const MAX_REDIRECTS: usize = 5;

// ---------------------------------------------------------------------------------------------
// Jobs, status, events
// ---------------------------------------------------------------------------------------------

/// One resource to install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub resource: &'static Resource,
    /// Install directory (runtimes: exclusive directory; models: the models directory).
    pub target: PathBuf,
    /// After installing an ONNX Runtime flavor, make it the active one (`active.txt` in the
    /// target's parent directory).
    pub activate_ort: bool,
    /// Install from these local archives (one per part) instead of downloading; no hash check.
    pub local_archives: Option<Vec<PathBuf>>,
    /// Version recorded in `VERSION` and the manifest (default: the resource's).
    pub version: Option<String>,
    /// Keep downloaded archives in `<root>/.downloads` after installing.
    pub keep_downloads: bool,
}

impl Job {
    /// Install `resource` into `<root>/<resource.dest>`.
    pub fn new(resource: &'static Resource, root: &Path) -> Self {
        Self::into_dir(resource, root.join(resource.dest))
    }

    /// Install `resource` into `dir` (models: the directory the files go in).
    pub fn into_dir(resource: &'static Resource, dir: PathBuf) -> Self {
        Self {
            resource,
            target: dir,
            activate_ort: false,
            local_archives: None,
            version: None,
            keep_downloads: false,
        }
    }

    /// Queue key: resource id and target directory.
    pub fn key(&self) -> String {
        format!("{}@{}", self.resource.id, self.target.display())
    }

    fn version(&self) -> &str {
        self.version.as_deref().unwrap_or(self.resource.version)
    }
}

/// Where a job is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum State {
    Queued,
    /// `bytes` of `total` (all parts) downloaded.
    Downloading {
        bytes: u64,
        total: u64,
    },
    Verifying,
    Extracting,
    Installed,
    /// `retry_at` (Unix seconds) is set when an automatic retry is scheduled.
    Failed {
        error: String,
        retry_at: Option<u64>,
        attempts: u32,
    },
}

impl State {
    /// Installed, or failed without a scheduled retry.
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            State::Installed | State::Failed { retry_at: None, .. }
        )
    }

    /// "downloading 42% (35/83 MB)", "installed", ...
    pub fn describe(&self) -> String {
        match self {
            State::Queued => "queued".to_string(),
            State::Downloading { bytes, total } => {
                let pct = if *total > 0 { bytes * 100 / total } else { 0 };
                format!(
                    "downloading {pct}% ({}/{} MB)",
                    bytes / 1_000_000,
                    total.div_ceil(1_000_000)
                )
            }
            State::Verifying => "verifying".to_string(),
            State::Extracting => "extracting".to_string(),
            State::Installed => "installed".to_string(),
            State::Failed {
                error, retry_at, ..
            } => match retry_at.and_then(|t| {
                let now = unix_now();
                (t > now).then(|| t - now)
            }) {
                Some(s) => format!("failed: {error} (retrying in {s} s)"),
                None => format!("failed: {error}"),
            },
        }
    }
}

impl State {
    /// Model-state text naming the resource: "downloading OpenVINO runtime 42% (18/44 MB)",
    /// "download of OpenVINO runtime failed: ...; retrying in 60 s", ...
    pub fn describe_for(&self, title: &str) -> String {
        match self {
            State::Queued => format!("waiting to download {title}"),
            State::Downloading { bytes, total } => {
                let pct = if *total > 0 { bytes * 100 / total } else { 0 };
                format!(
                    "downloading {title} {pct}% ({}/{} MB)",
                    bytes / 1_000_000,
                    total.div_ceil(1_000_000)
                )
            }
            State::Verifying => format!("verifying {title}"),
            State::Extracting => format!("installing {title}"),
            State::Installed => format!("{title} installed"),
            State::Failed {
                error, retry_at, ..
            } => match retry_at.and_then(|t| {
                let now = unix_now();
                Some(t.saturating_sub(now))
            }) {
                Some(s) => format!("download of {title} failed: {error}; retrying in {s} s"),
                None => format!("download of {title} failed: {error}"),
            },
        }
    }
}

/// Shared status of one job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Status {
    /// Queue key ([`Job::key`]).
    pub key: String,
    pub id: &'static str,
    pub title: &'static str,
    pub size: u64,
    pub target: PathBuf,
    #[serde(flatten)]
    pub state: State,
}

/// Terminal outcomes, sent to subscribers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Installed {
        key: String,
        id: &'static str,
        target: PathBuf,
    },
    Failed {
        key: String,
        id: &'static str,
        error: String,
        /// When the next automatic attempt happens (None = no retry).
        retry_at: Option<SystemTime>,
    },
}

/// Progress hook: called on every state change (downloads at most every 100 ms).
pub(crate) type ProgressFn = Arc<dyn Fn(&Status) + Send + Sync>;

/// Manager settings.
#[derive(Clone)]
pub struct ManagerOptions {
    /// Download root: holds the lock file and `.downloads/`.
    pub root: PathBuf,
    /// Retry failed jobs with [`backoff`] (the service); the CLI fails fast.
    pub retry: bool,
    /// How long to wait for another process's lock before failing the attempt (None = wait).
    pub lock_wait: Option<Duration>,
    pub on_progress: Option<ProgressFn>,
    /// Retry schedule (tests shorten it).
    pub backoff: fn(u32) -> Duration,
    /// With `retry`: give up after this many attempts of one job (None = retry forever). The
    /// service uses [`SERVICE_MAX_ATTEMPTS`].
    pub max_attempts: Option<u32>,
}

impl ManagerOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            retry: false,
            lock_wait: None,
            on_progress: None,
            backoff,
            max_attempts: None,
        }
    }
}

impl std::fmt::Debug for ManagerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagerOptions")
            .field("root", &self.root)
            .field("retry", &self.retry)
            .field("lock_wait", &self.lock_wait)
            .field("max_attempts", &self.max_attempts)
            .finish_non_exhaustive()
    }
}

struct Shared {
    opts: ManagerOptions,
    statuses: Mutex<BTreeMap<String, Status>>,
    changed: Condvar,
    subscribers: Mutex<Vec<Sender<Event>>>,
    stop: AtomicBool,
    /// The worker thread has exited (stopped, or died: see `dead_reason`).
    dead: AtomicBool,
    /// Why the worker died (None after a normal stop).
    dead_reason: Mutex<Option<String>>,
}

impl Shared {
    fn set(&self, job: &Job, state: State) {
        let status = Status {
            key: job.key(),
            id: job.resource.id,
            title: job.resource.title,
            size: job.resource.size(),
            target: job.target.clone(),
            state,
        };
        if let Some(hook) = &self.opts.on_progress {
            hook(&status);
        }
        let mut map = self.statuses.lock_recover();
        map.insert(status.key.clone(), status);
        drop(map);
        self.changed.notify_all();
    }

    fn state(&self, key: &str) -> Option<State> {
        let map = self.statuses.lock_recover();
        map.get(key).map(|s| s.state.clone())
    }

    fn emit(&self, ev: Event) {
        let mut subs = self.subscribers.lock_recover();
        subs.retain(|s| s.send(ev.clone()).is_ok());
    }

    /// Mark `job` failed for good (no retry) and tell the subscribers.
    fn fail(&self, job: &Job, error: String, attempts: u32) {
        crate::metrics::DOWNLOADS
            .failures
            .fetch_add(1, Ordering::Relaxed);
        self.set(
            job,
            State::Failed {
                error: error.clone(),
                retry_at: None,
                attempts,
            },
        );
        self.emit(Event::Failed {
            key: job.key(),
            id: job.resource.id,
            error,
            retry_at: None,
        });
    }

    /// Text for jobs the dead worker will never run.
    fn worker_gone_message(&self) -> String {
        let reason = self.dead_reason.lock_recover().clone();
        match reason {
            Some(r) => format!(
                "the download worker stopped ({r}); this is usually a missing TLS certificate \
                 store or an exhausted thread/file limit; restart the service (or rerun the \
                 command) after fixing it"
            ),
            None => "the download manager was stopped before this download finished; start it \
                     again (restart the service or rerun the command)"
                .to_string(),
        }
    }

    /// Fail every job that is not final (the worker is gone and will never finish them).
    fn fail_unfinished(&self) {
        let error = self.worker_gone_message();
        let pending: Vec<Status> = {
            let map = self.statuses.lock_recover();
            map.values()
                .filter(|s| !s.state.is_final())
                .cloned()
                .collect()
        };
        for st in pending {
            let attempts = match st.state {
                State::Failed { attempts, .. } => attempts,
                _ => 0,
            };
            crate::metrics::DOWNLOADS
                .failures
                .fetch_add(1, Ordering::Relaxed);
            let state = State::Failed {
                error: error.clone(),
                retry_at: None,
                attempts,
            };
            if let Some(hook) = &self.opts.on_progress {
                hook(&Status {
                    state: state.clone(),
                    ..st.clone()
                });
            }
            self.statuses.lock_recover().insert(
                st.key.clone(),
                Status {
                    state,
                    ..st.clone()
                },
            );
            self.emit(Event::Failed {
                key: st.key.clone(),
                id: st.id,
                error: error.clone(),
                retry_at: None,
            });
        }
        self.changed.notify_all();
    }
}

/// Runs when the worker thread exits, normally or by panic: marks the manager dead and, unless
/// it was stopped on purpose, fails every job it will never run (RS-04).
struct WorkerExit<'a>(&'a Shared);

impl Drop for WorkerExit<'_> {
    fn drop(&mut self) {
        let shared = self.0;
        let stopping = shared.stop.load(Ordering::SeqCst);
        if std::thread::panicking() {
            let mut reason = shared.dead_reason.lock_recover();
            reason.get_or_insert_with(|| "the worker thread panicked".to_string());
        }
        shared.dead.store(true, Ordering::SeqCst);
        if !stopping {
            tracing::error!("download manager: {}", shared.worker_gone_message());
            shared.fail_unfinished();
        } else {
            shared.changed.notify_all();
        }
    }
}

enum Msg {
    Job(Box<Job>),
    Stop,
}

/// The download manager. Cheap to share behind an `Arc`; dropping it stops the worker (an
/// unfinished download keeps its `.partial` file for the next start).
pub struct Manager {
    shared: Arc<Shared>,
    tx: Sender<Msg>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for Manager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Manager")
            .field("root", &self.shared.opts.root)
            .finish_non_exhaustive()
    }
}

/// `done` of `total` as a percent in 0..=100 (0 when `total` is 0).
pub fn percent(done: u64, total: u64) -> u8 {
    let p = (u128::from(done.min(total)) * 100)
        .checked_div(u128::from(total))
        .unwrap_or(0);
    u8::try_from(p).unwrap_or(100)
}

impl Manager {
    /// Start the worker thread.
    #[allow(
        clippy::expect_used,
        reason = "invariant: the OS can create one more thread at startup; without it the \
                  process cannot work at all, so failing to spawn is a controlled abort"
    )]
    pub fn start(opts: ManagerOptions) -> Self {
        cleanup_stale(&opts.root, SystemTime::now());
        let shared = Arc::new(Shared {
            opts,
            statuses: Mutex::new(BTreeMap::new()),
            changed: Condvar::new(),
            subscribers: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            dead_reason: Mutex::new(None),
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("downloads".into())
            .spawn(move || worker_loop(&worker, &rx))
            .expect("spawning the download thread (OS thread limit or out of memory)");
        Self {
            shared,
            tx,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// Queue `job` unless the same key is already queued, running or installed. Returns the key.
    pub fn enqueue(&self, job: Job) -> String {
        let key = job.key();
        let busy = matches!(
            self.shared.state(&key),
            Some(State::Queued | State::Downloading { .. } | State::Verifying | State::Extracting)
                | Some(State::Installed)
        );
        if !busy {
            self.shared.set(&job, State::Queued);
            let sent = self.tx.send(Msg::Job(Box::new(job.clone()))).is_ok();
            // Checked after the status is Queued: a worker dying concurrently either sees
            // the Queued status in `fail_unfinished`, or has already set `dead` (SeqCst).
            if !sent || self.shared.dead.load(Ordering::SeqCst) {
                let error = self.shared.worker_gone_message();
                tracing::warn!("download of {} not started: {error}", job.resource.id);
                self.shared.fail(&job, error, 0);
            }
        }
        key
    }

    /// The worker thread still runs (false after [`Self::shutdown`] or when it died).
    pub fn is_alive(&self) -> bool {
        !self.shared.dead.load(Ordering::SeqCst)
    }

    pub fn status(&self, key: &str) -> Option<Status> {
        let map = self.shared.statuses.lock_recover();
        map.get(key).cloned()
    }

    /// Drop the remembered status of `key` (after its files were removed), so it can be
    /// queued again. Queued or running jobs are kept.
    pub fn forget(&self, key: &str) {
        let mut map = self.shared.statuses.lock_recover();
        if map.get(key).is_some_and(|s| s.state.is_final()) {
            map.remove(key);
        }
    }

    /// Every job seen so far.
    pub fn statuses(&self) -> Vec<Status> {
        let map = self.shared.statuses.lock_recover();
        map.values().cloned().collect()
    }

    /// Terminal events from now on (installed / failed).
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut subs = self.shared.subscribers.lock_recover();
        subs.push(tx);
        rx
    }

    /// Block until every key is final ([`State::is_final`]) or `timeout` passes. Returns the
    /// statuses of `keys`.
    pub fn wait(&self, keys: &[String], timeout: Option<Duration>) -> Vec<Status> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut map = self.shared.statuses.lock_recover();
        loop {
            let done = keys
                .iter()
                .all(|k| map.get(k).is_some_and(|s| s.state.is_final()));
            if done || self.shared.dead.load(Ordering::SeqCst) {
                break;
            }
            let wait = match deadline {
                Some(d) => match d.checked_duration_since(Instant::now()) {
                    Some(left) => left,
                    None => break,
                },
                None => Duration::from_secs(1),
            };
            map = self
                .shared
                .changed
                .wait_timeout(map, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        keys.iter().filter_map(|k| map.get(k).cloned()).collect()
    }

    /// Stop the worker (the current download is abandoned, its `.partial` kept) and join it
    /// for at most `SHUTDOWN_JOIN`; a worker still busy after that (a hash or an extraction
    /// between two stop checks, a stalled read) is detached and logged.
    pub fn shutdown(&self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        let _ = self.tx.send(Msg::Stop);
        let handle = self.thread.lock_recover().take();
        if let Some(h) = handle {
            let until = Instant::now() + SHUTDOWN_JOIN;
            while !h.is_finished() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(20));
            }
            if h.is_finished() {
                let _ = h.join();
            } else {
                tracing::warn!(
                    "download worker still busy {} s after the stop request; leaving it to \
                     finish in the background (an unfinished download resumes on the next start)",
                    SHUTDOWN_JOIN.as_secs()
                );
            }
        }
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Jobs for `needs` of a resolution: runtimes into `<root>/<dest>`, models into their
/// configured directory (relative ones under `root`, the config's data root). The wanted ORT flavor is marked active
/// once installed (see [`super::resolve::Resolution::ort_activation`]).
pub fn jobs_for(
    resolution: &super::resolve::Resolution,
    needs: &[super::resolve::Need],
    root: &Path,
) -> Vec<Job> {
    needs
        .iter()
        .map(|n| {
            let mut job = match &n.model_dir {
                // Relative model dirs live under the data root (`download_dir`, else the exe dir).
                Some(dir) if dir.is_absolute() => Job::into_dir(n.resource, dir.clone()),
                Some(dir) => Job::into_dir(n.resource, root.join(dir)),
                None => Job::new(n.resource, root),
            };
            job.activate_ort =
                n.resource.flavor().is_some() && n.resource.flavor() == resolution.ort_activation();
            job
        })
        .collect()
}

/// Mark the resolution's wanted ORT flavor active when it is already installed under `root`
/// (no download needed). Returns the flavor marked.
pub fn apply_ort_activation(
    resolution: &super::resolve::Resolution,
    root: &Path,
) -> Result<Option<super::catalog::Flavor>> {
    let Some(f) = resolution.ort_activation() else {
        return Ok(None);
    };
    let dir = root.join(f.dest());
    if read_manifest(&dir.join(MANIFEST_FILE)).is_some_and(|m| m.id == f.resource_id()) {
        let ort_root = root.join(crate::backend::libs::ORT_DIR_NAME);
        crate::backend::libs::write_active_flavor(&ort_root, f.as_str())
            .with_context(|| format!("marking {f} active in {}", ort_root.display()))?;
        return Ok(Some(f));
    }
    Ok(None)
}

/// Install `jobs` now (CLI): one manager, no retries, wait for all. Errors list every failure.
pub fn install_all(opts: ManagerOptions, jobs: Vec<Job>) -> Result<Vec<Status>> {
    let manager = Manager::start(ManagerOptions {
        retry: false,
        ..opts
    });
    let keys: Vec<String> = jobs.into_iter().map(|j| manager.enqueue(j)).collect();
    let statuses = manager.wait(&keys, None);
    manager.shutdown();
    let failed: Vec<String> = statuses
        .iter()
        .filter_map(|s| match &s.state {
            State::Failed { error, .. } => Some(format!("{}: {error}", s.id)),
            _ => None,
        })
        .collect();
    if !failed.is_empty() {
        bail!(
            "{} (check the network connection and proxy, then run the command again; \
             interrupted downloads resume)",
            failed.join("; ")
        );
    }
    Ok(statuses)
}

mod progress;
mod worker;

pub use progress::*;
pub use worker::*;
