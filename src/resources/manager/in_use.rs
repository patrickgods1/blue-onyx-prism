//! Runtime directories in use (OpenVINO, an ONNX Runtime flavor, the CUDA libraries): an update is
//! never swapped in under a process that has the runtime loaded.
//!
//! A process that loads a runtime from one of our installs first calls [`before_load`], which
//! takes a shared lock on `<parent>/.<name>.in-use` and keeps it for the life of the process (the
//! OS drops it when the process exits). The download manager replaces an installed version only
//! while it can take that lock exclusively ([`place`]); otherwise the new version is kept, complete
//! and with its manifest, as `<parent>/.<name>.pending`, and the next process that loads the
//! runtime swaps it in before loading ([`before_load`]): an update of a loaded runtime is used
//! after a restart.
//!
//! Why not swap in place: on Windows the directory of a loaded DLL cannot be renamed, so the swap
//! failed and was retried for the whole backoff series; on Unix the rename works, but libraries a
//! runtime loads later (OpenVINO device plugins, ONNX Runtime providers, cuDNN sub-libraries)
//! would then come from the new version into a process running the old one. A process that does
//! not take part (an older release of this program) is caught by the failing rename on Windows,
//! which stages the update too.
//!
//! User-managed runtimes (`openvino_dir`, `onnxruntime_dir`) have no manifest of ours: they are
//! never swapped and get no lock file beside them.
//!
//! Spawned children: on Unix the lock belongs to the open file, and a child that another thread
//! is spawning holds a copy of every descriptor from its fork until its exec (close-on-exec acts
//! only then), so closing the lock file alone can leave the lock held for that moment. An
//! [`ExclusiveUse`] therefore unlocks when dropped, so neither this process's own shared lock
//! right after it nor another process's waits on it. The shared locks are held for the life of
//! the process; a process that exits while one of its threads is spawning can keep its lock for
//! that moment, and a load that starts within it applies a staged update at the next start
//! instead, so [`before_load`] does not retry the exclusive lock.

use super::worker::{put_in_place, remove_any, swap_owned};
use super::*;
use std::fs::{File, TryLockError};

/// Suffix of the lock file beside a runtime directory.
const IN_USE: &str = "in-use";
/// Suffix of a downloaded update waiting for the next start.
const PENDING: &str = "pending";
/// How long [`before_load`] waits for an install that holds the exclusive lock (a swap takes
/// seconds) before loading anyway.
const SHARED_WAIT: Duration = Duration::from_secs(30);

/// `<parent>/.<name>.pending`: the downloaded update of the runtime in `target` that waits for
/// the next start.
pub fn pending_dir(target: &Path) -> Option<PathBuf> {
    staged_sibling(target, PENDING)
}

fn lock_path(target: &Path) -> Option<PathBuf> {
    staged_sibling(target, IN_USE)
}

/// `<parent>/.<name>.<what>` beside `target` (nothing is created).
fn staged_sibling(target: &Path, what: &str) -> Option<PathBuf> {
    let name = target.file_name()?.to_string_lossy();
    Some(
        target
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(".{name}.{what}")),
    )
}

/// The version of resource `id` staged for `target` ([`pending_dir`]): its manifest names `id`
/// and every file it lists is there.
pub fn staged_version(id: &str, target: &Path) -> Option<String> {
    let pending = pending_dir(target)?;
    let path = pending.join(MANIFEST_FILE);
    read_manifest(&path)
        .filter(|m| m.id == id && install_damage(&path, &pending).is_none())
        .map(|m| m.version)
}

/// Runtimes a process loads (and so may hold in use).
fn is_runtime(kind: ResourceKind) -> bool {
    matches!(
        kind,
        ResourceKind::OpenVinoRuntime | ResourceKind::OnnxRuntime(_) | ResourceKind::CudaLibs
    )
}

fn open_lock(target: &Path) -> std::io::Result<File> {
    let path = lock_path(target).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no directory name", target.display()),
        )
    })?;
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
}

/// Exclusive use of a runtime directory: no process (this one included) has it loaded, and none
/// can load it until this is dropped.
pub(super) struct ExclusiveUse {
    file: File,
}

impl ExclusiveUse {
    /// None: some process has the runtime loaded.
    pub(super) fn try_take(target: &Path) -> std::io::Result<Option<Self>> {
        let file = open_lock(target)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(e),
        }
    }
}

impl Drop for ExclusiveUse {
    /// Unlocked, not only closed: a child that another thread is spawning holds a copy of the
    /// descriptor until it execs, and on Unix the lock lasts until every copy is closed (see the
    /// module docs).
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Where [`place`] put a finished install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Placed {
    /// At `target`, usable now.
    InPlace,
    /// In [`pending_dir`]: the runtime is loaded by a process; the next start swaps it in.
    Staged(PathBuf),
}

/// Move a finished `staging` directory of `res` to `target`, or stage it ([`Placed::Staged`]) when
/// `target` is an install of ours that a process has loaded (see the module docs).
pub(super) fn place(res: &Resource, staging: &Path, target: &Path) -> Result<Placed> {
    let ours = is_runtime(res.kind)
        && read_manifest(&target.join(MANIFEST_FILE)).is_some_and(|m| m.id == res.id);
    let Some(pending) = pending_dir(target).filter(|_| ours) else {
        put_in_place(res, staging, target)?;
        return Ok(Placed::InPlace);
    };
    let exclusive = ExclusiveUse::try_take(target);
    if let Err(e) = &exclusive {
        // No lock file possible (a read-only parent, another account's file): try the swap; a
        // loaded Windows runtime still fails the rename, which stages it below.
        tracing::debug!("{}: no in-use lock: {e}", target.display());
    }
    if !matches!(exclusive, Ok(None)) {
        match put_in_place(res, staging, target) {
            Ok(()) => {
                // An update staged earlier is superseded.
                if pending.exists() {
                    let _ = remove_any(&pending);
                }
                return Ok(Placed::InPlace);
            }
            Err(e) if staging.is_dir() && target.is_dir() => tracing::warn!(
                "{e:#}; keeping the new {} {} for the next start instead",
                res.title,
                res.version
            ),
            Err(e) => return Err(e),
        }
    }
    drop(exclusive);
    remove_any(&pending)?;
    std::fs::rename(staging, &pending)
        .with_context(|| format!("moving {} to {}", staging.display(), pending.display()))?;
    tracing::info!(
        "{} {} downloaded; {} is in use by a running process, so the new version replaces it the \
         next time Blue Onyx Prism starts (restart the service to use it now)",
        res.title,
        res.version,
        target.display()
    );
    Ok(Placed::Staged(pending))
}

/// Shared locks this process holds, by runtime directory (kept for the life of the process).
static HELD: Mutex<Vec<(PathBuf, File)>> = Mutex::new(Vec::new());

/// Call right before this process loads a runtime from `dir` (idempotent per directory): when
/// `dir` is an install of ours, swap in a staged update while no other process has it loaded,
/// then mark it in use for the life of the process. Failures are logged; loading goes on.
pub fn before_load(dir: &Path) {
    let Some(id) = read_manifest(&dir.join(MANIFEST_FILE)).map(|m| m.id) else {
        return;
    };
    let mut held = HELD.lock_recover();
    if held.iter().any(|(d, _)| d == dir) {
        return;
    }
    if let Some(pending) = pending_dir(dir).filter(|p| p.is_dir()) {
        match ExclusiveUse::try_take(dir) {
            Ok(Some(_exclusive)) => apply_staged(&id, &pending, dir),
            Ok(None) => tracing::info!(
                "a downloaded update of {id} waits in {}, but another process has {} loaded; \
                 this process uses the installed version, and the update is applied at the first \
                 start after every process using it has stopped",
                pending.display(),
                dir.display()
            ),
            Err(e) => tracing::debug!("{}: no in-use lock: {e}", dir.display()),
        }
    }
    match shared_lock(dir) {
        Ok(Some(file)) => held.push((dir.to_path_buf(), file)),
        Ok(None) => tracing::warn!(
            "{} is still being replaced by another process after {} s; loading it anyway",
            dir.display(),
            SHARED_WAIT.as_secs()
        ),
        Err(e) => tracing::debug!("{}: no in-use lock: {e}", dir.display()),
    }
}

/// A shared lock on `dir`'s lock file, waiting up to [`SHARED_WAIT`] for an install holding it
/// exclusively; None when it is still held.
fn shared_lock(dir: &Path) -> std::io::Result<Option<File>> {
    let file = open_lock(dir)?;
    let started = Instant::now();
    loop {
        match file.try_lock_shared() {
            Ok(()) => return Ok(Some(file)),
            Err(TryLockError::WouldBlock) if started.elapsed() < SHARED_WAIT => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// Swap the staged update `pending` of resource `id` into `target` (the caller holds the
/// exclusive lock). An incomplete one is deleted (the resolver then plans the update again).
fn apply_staged(id: &str, pending: &Path, target: &Path) {
    let path = pending.join(MANIFEST_FILE);
    let manifest = read_manifest(&path).filter(|m| m.id == id);
    let damage = manifest
        .as_ref()
        .map(|_| install_damage(&path, pending))
        .unwrap_or_else(|| Some(format!("no manifest of {id}")));
    let Some(m) = manifest.filter(|_| damage.is_none()) else {
        tracing::warn!(
            "discarding the downloaded update of {id} in {}: {}; it is downloaded again when \
             needed",
            pending.display(),
            damage.unwrap_or_default()
        );
        let _ = remove_any(pending);
        return;
    };
    match swap_owned(pending, target) {
        Ok(()) => tracing::info!(
            "installed the downloaded {id} {} into {} (it was staged while the previous version \
             was in use)",
            m.version,
            target.display()
        ),
        Err(e) => tracing::warn!(
            "{e:#}; staying on the installed version, the update stays in {}",
            pending.display()
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-inuse-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn openvino() -> &'static Resource {
        let p = crate::resources::catalog::Platform::current();
        crate::resources::catalog::openvino_runtime(p.os, p.arch)
            .or_else(|| crate::resources::catalog::openvino_runtime("linux", "x86_64"))
            .unwrap()
    }

    /// A complete install of `id` at `version` in `dir` with one library file holding `content`.
    fn install(dir: &Path, id: &str, version: &str, content: &[u8]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("lib.bin"), content).unwrap();
        let m = Manifest {
            id: id.into(),
            version: version.into(),
            sha256: vec!["0".repeat(64)],
            files: vec!["lib.bin".into()],
        };
        super::super::write_manifest(&dir.join(MANIFEST_FILE), &m, dir).unwrap();
    }

    fn content(dir: &Path) -> Vec<u8> {
        std::fs::read(dir.join("lib.bin")).unwrap()
    }

    /// Nobody has the runtime loaded: the new version replaces the old one at once.
    #[test]
    fn an_unused_runtime_is_replaced_in_place() {
        let root = tmp("free");
        let res = openvino();
        let target = root.join("openvino");
        install(&target, res.id, "old", b"v1");
        let staging = root.join(".openvino.staging");
        install(&staging, res.id, res.version, b"v2");
        // A leftover of an older staged update is superseded.
        install(&pending_dir(&target).unwrap(), res.id, "older", b"v0");
        assert_eq!(place(res, &staging, &target).unwrap(), Placed::InPlace);
        assert_eq!(content(&target), b"v2");
        assert!(!pending_dir(&target).unwrap().exists());
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Another process has it loaded (its shared lock): the update waits beside it, the loaded
    /// files stay, and the first load after that process stopped swaps it in.
    #[test]
    fn a_loaded_runtime_is_staged_and_applied_at_the_next_load() {
        let root = tmp("busy");
        let res = openvino();
        let target = root.join("openvino");
        install(&target, res.id, "old", b"v1");
        let other = open_lock(&target).unwrap();
        other.lock_shared().unwrap();

        let staging = root.join(".openvino.staging");
        install(&staging, res.id, res.version, b"v2");
        let pending = pending_dir(&target).unwrap();
        assert_eq!(
            place(res, &staging, &target).unwrap(),
            Placed::Staged(pending.clone())
        );
        assert_eq!(content(&target), b"v1", "the loaded files are untouched");
        assert_eq!(
            staged_version(res.id, &target).as_deref(),
            Some(res.version)
        );
        assert_eq!(staged_version("onnxruntime-cpu", &target), None);

        // Still loaded elsewhere: this process loads the installed version.
        let probe = root.join("probe");
        install(&probe, res.id, "old", b"p1");
        install(&pending_dir(&probe).unwrap(), res.id, res.version, b"p2");
        let holder = open_lock(&probe).unwrap();
        holder.lock_shared().unwrap();
        before_load(&probe);
        assert_eq!(content(&probe), b"p1");
        drop(holder);

        // The other process stopped: the next load applies the update first. Unlocked before the
        // close: other tests spawn processes, and a child forked by another thread holds a copy
        // of `other` until it execs, which a close alone leaves locked (the module docs).
        other.unlock().unwrap();
        drop(other);
        before_load(&target);
        assert_eq!(content(&target), b"v2");
        assert!(!pending.exists());
        assert_eq!(staged_version(res.id, &target), None);
        let m = read_manifest(&target.join(MANIFEST_FILE)).unwrap();
        assert_eq!(m.version, res.version);

        // Now this process holds it: a further update is staged again.
        let staging = root.join(".openvino.staging");
        install(&staging, res.id, "newer", b"v3");
        assert_eq!(
            place(res, &staging, &target).unwrap(),
            Placed::Staged(pending.clone())
        );
        assert_eq!(content(&target), b"v2");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A staged update that lost a file is discarded, not swapped in.
    #[test]
    fn a_damaged_staged_update_is_discarded() {
        let root = tmp("damaged");
        let res = openvino();
        let target = root.join("openvino");
        install(&target, res.id, "old", b"v1");
        let pending = pending_dir(&target).unwrap();
        install(&pending, res.id, res.version, b"v2");
        std::fs::remove_file(pending.join("lib.bin")).unwrap();
        assert_eq!(staged_version(res.id, &target), None);
        before_load(&target);
        assert_eq!(content(&target), b"v1");
        assert!(!pending.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A user-managed runtime (no manifest of ours) is never swapped and gets no lock file.
    #[test]
    fn a_user_managed_runtime_is_left_alone() {
        let root = tmp("user");
        let dir = root.join("openvino");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.bin"), b"mine").unwrap();
        let pending = pending_dir(&dir).unwrap();
        install(&pending, openvino().id, "x", b"v2");
        before_load(&dir);
        assert_eq!(content(&dir), b"mine");
        assert!(pending.exists());
        assert!(!lock_path(&dir).unwrap().exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Plain files (models) and a fresh install are placed directly.
    #[test]
    fn a_fresh_runtime_install_needs_no_lock() {
        let root = tmp("fresh");
        let res = openvino();
        let target = root.join("openvino");
        let staging = root.join(".openvino.staging");
        install(&staging, res.id, res.version, b"v1");
        assert_eq!(place(res, &staging, &target).unwrap(), Placed::InPlace);
        assert_eq!(content(&target), b"v1");
        assert!(!lock_path(&target).unwrap().exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
