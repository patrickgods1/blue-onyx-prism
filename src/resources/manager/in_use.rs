//! Runtime directories in use (OpenVINO, an ONNX Runtime flavor, the CUDA libraries): an update is
//! never swapped in under a process that has the runtime loaded.
//!
//! A process that loads a runtime from one of our installs first calls [`before_load`], which
//! takes a shared lock on `<parent>/.<name>.in-use` and keeps it for the life of the process (the
//! OS drops it when the process exits). The download manager replaces an installed version only
//! while it can take that lock exclusively ([`place`]); otherwise the new version is kept, complete
//! and with its manifest, as `<parent>/.<name>.pending`, and the next process that loads the
//! runtime swaps it in before loading ([`before_load`]): an update of a loaded runtime is used
//! after a restart. The service also applies it when it plans a generation that loads the runtime
//! ([`apply_pending`]); a staged update of a runtime the config no longer loads is left as it is
//! (shown with its size on the Resources card, where Remove deletes it, also when the install it
//! was meant for is gone). A swap that a crash cut between its two renames is put back before a
//! load ([`before_load`], `super::leftovers`).
//!
//! Why not swap in place: on Windows the directory of a loaded DLL cannot be renamed, so the swap
//! failed and was retried for the whole backoff series; on Unix the rename works, but libraries a
//! runtime loads later (OpenVINO device plugins, ONNX Runtime providers, cuDNN sub-libraries)
//! would then come from the new version into a process running the old one. A process that does
//! not take part (an older release of this program) is caught by the failing rename on Windows,
//! which stages the update too.
//!
//! User-managed runtimes (`openvino_dir`, `onnxruntime_dir`) have no manifest of ours: they are
//! never swapped and get no lock file beside them. An install into a user's `--dir` / `--dest`
//! that held other files has our manifest, marked merged: it is locked and staged like any
//! install of ours, but its update is merged in file by file, never swapped (QA-025,
//! [`super::exclusively_ours`]).
//!
//! Spawned children: on Unix the lock belongs to the open file, and a child that another thread
//! is spawning holds a copy of every descriptor from its fork until its exec (close-on-exec acts
//! only then), so closing the lock file alone can leave the lock held for that moment. An
//! [`ExclusiveUse`] therefore unlocks when dropped, so neither this process's own shared lock
//! right after it nor another process's waits on it. The shared locks are held for the life of
//! the process; a process that exits while one of its threads is spawning can keep its lock for
//! that moment, and a load that starts within it applies a staged update at the next start
//! instead, so [`before_load`] does not retry the exclusive lock.

use super::worker::{put_files, put_in_place, remove_any};
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

pub(super) fn lock_path(target: &Path) -> Option<PathBuf> {
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
/// `target` holds an install of ours (its manifest names `res`, also one merged into a user's
/// directory) that a process has loaded (see the module docs). Whether the directory is then
/// replaced as a whole or merged into is [`put_in_place`]'s rule, here and when the staged update
/// is applied.
pub(super) fn place(res: &Resource, staging: &Path, target: &Path) -> Result<Placed> {
    let ours = is_runtime(res.kind)
        && read_manifest(&target.join(MANIFEST_FILE)).is_some_and(|m| m.id == res.id);
    let Some(pending) = pending_dir(target).filter(|_| ours) else {
        put_in_place(res, staging, target)?;
        // A fresh install where the previous one is gone: an update staged beside that one is
        // superseded (nothing would swap it in over this version).
        if is_runtime(res.kind)
            && let Some(p) = pending_dir(target).filter(|p| p.exists())
            && let Err(e) = remove_any(&p)
        {
            tracing::warn!("{e:#}; Remove on the Resources card deletes it");
        }
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

/// This process loaded the runtime in `dir` (holds its shared lock, see [`before_load`]).
pub(super) fn loaded_here(dir: &Path) -> bool {
    HELD.lock_recover().iter().any(|(d, _)| d == dir)
}

/// Call right before this process loads a runtime from `dir` (idempotent per directory): when
/// `dir` is an install of ours, swap in a staged update while no other process has it loaded,
/// then mark it in use for the life of the process. Failures are logged; loading goes on.
pub fn before_load(dir: &Path) {
    if std::fs::symlink_metadata(dir).is_err() {
        recover_missing_runtime(dir);
    }
    let Some(id) = read_manifest(&dir.join(MANIFEST_FILE)).map(|m| m.id) else {
        return;
    };
    let mut held = HELD.lock_recover();
    if held.iter().any(|(d, _)| d == dir) {
        return;
    }
    apply_if_staged(&id, dir);
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

/// Before a generation loads the runtime in `dir` (`Provisioner::prepare`, RA resources-11): swap
/// in a staged update now, as [`before_load`] would right after, while no process has it loaded.
/// Nothing when this process loaded it already or it is no install of ours. True when applied.
pub(crate) fn apply_pending(dir: &Path) -> bool {
    let Some(id) = read_manifest(&dir.join(MANIFEST_FILE)).map(|m| m.id) else {
        return false;
    };
    let held = HELD.lock_recover();
    if held.iter().any(|(d, _)| d == dir) {
        return false;
    }
    apply_if_staged(&id, dir)
}

/// Swap the staged update of `dir` in when there is one and no process has `dir` loaded (the
/// caller holds [`HELD`]). True when applied.
fn apply_if_staged(id: &str, dir: &Path) -> bool {
    let Some(pending) = pending_dir(dir).filter(|p| p.is_dir()) else {
        return false;
    };
    match ExclusiveUse::try_take(dir) {
        Ok(Some(_exclusive)) => apply_staged(id, &pending, dir),
        Ok(None) => {
            tracing::info!(
                "a downloaded update of {id} waits in {}, but another process has {} loaded; \
                 this process uses the installed version, and the update is applied at the first \
                 start after every process using it has stopped",
                pending.display(),
                dir.display()
            );
            false
        }
        Err(e) => {
            tracing::debug!("{}: no in-use lock: {e}", dir.display());
            false
        }
    }
}

/// `dir` is missing: when a crash cut a swap of it between its two renames, its set-aside install
/// (`.<name>.old-<pid>`, whose manifest names the resource and so its download root) is put back
/// ([`super::leftovers::recover_missing`]).
fn recover_missing_runtime(dir: &Path) {
    let (Some(parent), Some(name)) = (dir.parent(), dir.file_name().and_then(|n| n.to_str()))
    else {
        return;
    };
    let Ok(rd) = std::fs::read_dir(parent) else {
        return;
    };
    let root = rd.flatten().find_map(|e| {
        let n = e.file_name();
        let (stem, _) = super::leftovers::set_aside_name(n.to_str()?)?;
        let m = read_manifest(&e.path().join(MANIFEST_FILE))?;
        (stem == name)
            .then(|| super::leftovers::root_of(dir, &m.id))
            .flatten()
    });
    if let Some(root) = root {
        super::leftovers::recover_missing(&root, dir);
    }
}

/// A shared lock on `dir`'s lock file, waiting up to [`SHARED_WAIT`] for an install holding it
/// exclusively; None when it is still held.
fn shared_lock(dir: &Path) -> std::io::Result<Option<File>> {
    let file = open_lock(dir)?;
    let waited = crate::fsx::lock_wait_with(
        &file,
        crate::fsx::LockKind::Shared,
        crate::fsx::deadline_after(SHARED_WAIT),
        crate::fsx::Pauses::fixed(Duration::from_millis(100)),
        || true,
    )?;
    Ok((waited == crate::fsx::Waited::Held).then_some(file))
}

/// Put the staged update `pending` of resource `id` into `target` (the caller holds the
/// exclusive lock): swapped in when `target` is our install alone, else merged file by file
/// (`worker::put_files`, [`super::exclusively_ours`]). An incomplete one is deleted (the resolver
/// then plans the update again). True when it is in place.
fn apply_staged(id: &str, pending: &Path, target: &Path) -> bool {
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
        return false;
    };
    // The same rule as a direct install (QA-025): only an install that is ours alone is replaced
    // as a whole; one merged into a user's directory is merged again, the user's files kept.
    let platform = crate::resources::catalog::Platform::current();
    let ort_windows =
        crate::resources::catalog::find(id, platform.os, platform.arch).and_then(|r| {
            match r.kind {
                ResourceKind::OnnxRuntime(_) => Some(r.platform.is_some_and(|p| p.os == "windows")),
                _ => None,
            }
        });
    match put_files(id, ort_windows, pending, target) {
        Ok(()) => {
            tracing::info!(
                "installed the downloaded {id} {} into {} (it was staged while the previous \
                 version was in use)",
                m.version,
                target.display()
            );
            true
        }
        Err(e) => {
            tracing::warn!(
                "{e:#}; staying on the installed version, the update stays in {}",
                pending.display()
            );
            false
        }
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

    /// QA-025: an install of ours merged into a user's folder (other files beside it) that a
    /// process had loaded is staged like any install of ours, and the next load merges the
    /// update in: the user's files stay. The same holds for a folder whose manifest says merged
    /// and nothing else.
    #[test]
    fn a_staged_update_of_a_merged_install_keeps_the_users_files() {
        let root = crate::test_dirs::TempDir::new("inuse-merged");
        let res = openvino();
        let target = root.join("shared");
        install(&target, res.id, "old", b"v1");
        std::fs::write(target.join("keep.txt"), b"mine").unwrap();
        let other = open_lock(&target).unwrap();
        other.lock_shared().unwrap();
        let staging = root.join(".shared.staging");
        install(&staging, res.id, res.version, b"v2");
        let pending = pending_dir(&target).unwrap();
        assert_eq!(
            place(res, &staging, &target).unwrap(),
            Placed::Staged(pending.clone())
        );
        other.unlock().unwrap();
        drop(other);
        before_load(&target);
        assert_eq!(content(&target), b"v2");
        assert_eq!(std::fs::read(target.join("keep.txt")).unwrap(), b"mine");
        assert!(!pending.exists());
        assert!(super::super::is_merged(&target.join(MANIFEST_FILE)));

        // Merged earlier, nothing else there: still merged, not swapped (a swap would drop the
        // file the merge left that the new version does not list).
        let dir = root.join("flagged");
        install(&dir, res.id, "old", b"v1");
        std::fs::write(dir.join("old-only.bin"), b"from v1").unwrap();
        let m = Manifest {
            id: res.id.into(),
            version: "old".into(),
            sha256: Vec::new(),
            files: vec!["lib.bin".into(), "old-only.bin".into()],
        };
        super::super::write_manifest_as(&dir.join(MANIFEST_FILE), &m, &dir, true).unwrap();
        install(&pending_dir(&dir).unwrap(), res.id, res.version, b"v2");
        assert!(apply_pending(&dir));
        assert_eq!(content(&dir), b"v2");
        assert!(dir.join("old-only.bin").is_file());
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

    /// RA resources-11: an install into a folder that is gone (deleted by hand) supersedes the
    /// update staged beside it, which nothing would ever swap in.
    #[test]
    fn a_fresh_install_supersedes_a_pending_without_its_install() {
        let root = tmp("fresh-pending");
        let res = openvino();
        let target = root.join("openvino");
        let pending = pending_dir(&target).unwrap();
        install(&pending, res.id, "older", b"v0");
        let staging = root.join(".openvino.staging");
        install(&staging, res.id, res.version, b"v1");
        assert_eq!(place(res, &staging, &target).unwrap(), Placed::InPlace);
        assert_eq!(content(&target), b"v1");
        assert!(!pending.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RA resources-11: `apply_pending` (a generation about to load the runtime) swaps a staged
    /// update in when no process has the runtime loaded, and leaves it while one has.
    #[test]
    fn apply_pending_swaps_in_only_while_nobody_loads_it() {
        let root = tmp("apply");
        let res = openvino();
        let target = root.join("openvino");
        install(&target, res.id, "old", b"v1");
        let pending = pending_dir(&target).unwrap();
        install(&pending, res.id, res.version, b"v2");
        let other = open_lock(&target).unwrap();
        other.lock_shared().unwrap();
        assert!(!apply_pending(&target));
        assert_eq!(content(&target), b"v1");
        other.unlock().unwrap();
        drop(other);
        assert!(apply_pending(&target));
        assert_eq!(content(&target), b"v2");
        assert!(!pending.exists());
        assert!(!apply_pending(&target), "nothing staged any more");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RA resources-10: a runtime missing after a swap that a crash cut between its renames is
    /// put back by the next load.
    #[test]
    fn before_load_restores_a_runtime_cut_mid_swap() {
        let root = tmp("cut");
        let res = openvino();
        let target = root.join(res.dest);
        let parent = target.parent().unwrap();
        let name = target.file_name().unwrap().to_str().unwrap();
        install(
            &parent.join(format!(".{name}.old-4242")),
            res.id,
            "1",
            b"v1",
        );
        before_load(&target);
        assert_eq!(content(&target), b"v1");
        assert!(!parent.join(format!(".{name}.old-4242")).exists());
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
