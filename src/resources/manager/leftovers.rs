//! Download leftovers (audit 2026-11, phase 2): what an interrupted install left beside the places
//! downloads write, removed, or put back, only while this process holds the download lock
//! (`<root>/.downloads.lock`), so nothing another process is installing is touched. Runs from the
//! storage sweep (startup and daily, [`sweep_stale`]) and when a download thread starts
//! ([`clean`]); skipped (debug log) while another process downloads.
//!
//! Where it looks (never the whole data root, so dataset day folders are never listed):
//! `<root>/.downloads`, the folder of every catalog destination and its parent, one level below
//! each destination (`tools/export-weights/<name>/`, `bench/<set>/`, ...), and the configured
//! model folders (`models_dir` and every model entry's folder, which may lie outside the root).
//! Every name is tested before anything is read about the entry.
//!
//! What it does, by exact name shape:
//! - `<file>.partial` files untouched for [`STALE_PARTIAL_AGE`]: a download nobody resumed.
//! - `.<name>.staging` folders untouched for [`STALE_STAGING_AGE`]: an extraction that stopped.
//!   Only that shape: `.<name>.train-staging` (a training run), `.<name>.export-staging` (a model
//!   export), `.<name>.in-use` and `.<name>.pending` (a staged update, see `in_use`) never match.
//! - `.<name>.old-<pid>` folders (an install set aside by `swap_owned`, whose rename keeps the old
//!   folder's own time, so no age test means anything here), of another process, and while no
//!   swap can run (`in_use::ExclusiveUse`, the lock `apply_staged` swaps under, is free, or this
//!   process has `<name>` loaded, whose shared lock keeps every swap out): deleted when `<name>` holds an intact install (the swap finished); when `<name>` is missing
//!   (a crash between the swap's two renames), the complete update beside it (`.<name>.pending`,
//!   else `.<name>.staging`, the catalog's version first) is moved in, else the set-aside install
//!   is renamed back. Never both copies lost (RA resources-10).
//! - Manifest temps (`.installed.json.tmp`, `.<model>.installed.json.tmp`) of a manifest write
//!   that a crash cut off, after [`STALE_STAGING_AGE`]; an exported model's temp is the export's
//!   (`export::sweep_leftovers`).
//! - In `.downloads/`, verified archives (kept after a failed extraction) untouched for
//!   [`STALE_PARTIAL_AGE`] whose resource is installed at that version or that no current catalog
//!   part names (an old pin). An archive kept on purpose (`keep_downloads`, `setup-* --keep-archive`)
//!   has a [`KEEP_SUFFIX`] marker beside it and stays.

use super::in_use::{ExclusiveUse, loaded_here, lock_path, pending_dir};
use super::*;
use crate::resources::catalog;
use crate::storage::{Removed, remove_counted};

/// `.partial` files not touched for this long are leftovers of a download nobody resumed.
pub(crate) const STALE_PARTIAL_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Staging directories (`.<name>.staging`) and manifest temps not touched for this long belong to
/// an interrupted install (a live one holds the download lock, which the cleanup takes first; the
/// age is a second guard for a process using another download root).
pub(crate) const STALE_STAGING_AGE: Duration = Duration::from_secs(60 * 60);
/// Suffix of the marker beside an archive in `.downloads/` kept on purpose (`keep_downloads`).
pub(crate) const KEEP_SUFFIX: &str = ".keep";
/// How deep below a destination the cleanup looks (`<dest>/<name>/x.partial`).
const DEST_DEPTH: usize = 1;

/// `.<stem>.staging` (non-empty stem): an install's extraction folder.
pub(super) fn staging_stem(name: &str) -> Option<&str> {
    name.strip_prefix('.')?
        .strip_suffix(".staging")
        .filter(|s| !s.is_empty())
}

/// `.<stem>.old-<pid>` (non-empty stem, all-digit pid): something an install set aside before
/// renaming its replacement in (`swap_owned` folders, the export's `InstallTx` files).
pub(crate) fn set_aside_name(name: &str) -> Option<(&str, u32)> {
    let (stem, pid) = name.strip_prefix('.')?.rsplit_once(".old-")?;
    if stem.is_empty() || pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((stem, pid.parse().ok()?))
}

/// The temp of a manifest write (`.installed.json.tmp`: Some(""), `.<model>.installed.json.tmp`:
/// Some(model)).
pub(crate) fn manifest_temp_of(name: &str) -> Option<&str> {
    let rest = name.strip_suffix(".installed.json.tmp")?;
    if rest.is_empty() {
        return Some("");
    }
    rest.strip_prefix('.').filter(|s| !s.is_empty())
}

/// The marker that keeps `archive` in `.downloads/` ([`KEEP_SUFFIX`]).
pub(super) fn keep_marker(archive: &Path) -> PathBuf {
    let mut name = archive.file_name().unwrap_or_default().to_os_string();
    name.push(KEEP_SUFFIX);
    archive.with_file_name(name)
}

/// The storage sweep's part: [`clean`] of `config`'s data root, with the model folders of the
/// config (`models_dir` and every entry's folder) as extra places to look.
pub(crate) fn sweep_stale(
    config: &crate::config::Config,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    let mut models = vec![config.data_path(&config.models_dir)];
    models.extend(
        config
            .models
            .iter()
            .filter_map(|m| config.data_path(&m.path).parent().map(Path::to_path_buf)),
    );
    clean(&config.data_root(), &models, SystemTime::now(), out, stop);
}

/// Clean the download leftovers of `root` and `models_dirs` (see the module docs), asking `stop`
/// between items. False (nothing looked at) when another process holds the download lock or the
/// root does not exist.
pub(crate) fn clean(
    root: &Path,
    models_dirs: &[PathBuf],
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) -> bool {
    if !root.is_dir() {
        return false;
    }
    let _lock = match DownloadLock::try_acquire(root) {
        Ok(Some(l)) => l,
        Ok(None) => {
            tracing::debug!(
                "download cleanup skipped: another process holds {} (it is downloading); the \
                 next sweep tries again",
                root.join(LOCK_FILE).display()
            );
            return false;
        }
        Err(e) => {
            tracing::debug!("download cleanup skipped: {e:#}");
            return false;
        }
    };
    let before = out.len();
    let export_names: Vec<&'static str> = catalog::EXPORT_MODELS
        .iter()
        .filter_map(Resource::model_name)
        .collect();
    let mut walk = Walk {
        now,
        stop,
        seen: HashSet::new(),
        export_names,
    };
    for (dir, depth) in places(root, models_dirs) {
        if stop() {
            break;
        }
        walk.dir(&dir, 0, depth, out);
    }
    if !stop() {
        superseded_archives(root, now, out, stop);
    }
    let removed = out.len().saturating_sub(before);
    if removed > 0 {
        tracing::info!(
            "download cleanup: removed {removed} leftover(s) of interrupted downloads under {}",
            root.display()
        );
    }
    true
}

/// Every resource whose download may leave something behind (the catalog of every platform, the
/// training and auto-label weights, the import datasets).
fn resources() -> impl Iterator<Item = &'static Resource> {
    catalog::all()
        .chain(catalog::TRAIN_BASE_WEIGHTS)
        .chain(catalog::AUTOLABEL_WEIGHTS)
        .chain(catalog::IMPORT_DATASETS)
        .chain(catalog::lila_night())
}

/// The folders to look in, with how deep below each (see the module docs).
fn places(root: &Path, models_dirs: &[PathBuf]) -> Vec<(PathBuf, usize)> {
    let mut v: BTreeMap<PathBuf, usize> = BTreeMap::new();
    let mut add = |p: PathBuf, depth: usize| {
        let e = v.entry(p).or_insert(depth);
        *e = (*e).max(depth);
    };
    add(root.to_path_buf(), 0);
    add(root.join(DOWNLOADS_DIR), 0);
    for r in resources() {
        let dest = root.join(r.dest);
        if let Some(parent) = dest.parent() {
            add(parent.to_path_buf(), 0);
        }
        add(dest, DEST_DEPTH);
    }
    add(
        root.join(crate::resources::export::TOOLS_DIR)
            .join("export-weights"),
        DEST_DEPTH,
    );
    for m in models_dirs {
        add(m.clone(), 0);
    }
    v.into_iter().collect()
}

struct Walk<'a> {
    now: SystemTime,
    stop: &'a dyn Fn() -> bool,
    /// Folders listed already (a destination inside another one's reach).
    seen: HashSet<PathBuf>,
    /// Exported models' names: their manifest temps are the export's.
    export_names: Vec<&'static str>,
}

impl Walk<'_> {
    fn dir(&mut self, dir: &Path, depth: usize, max: usize, out: &mut Vec<Removed>) {
        if !self.seen.insert(dir.to_path_buf()) {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for ent in rd.flatten() {
            if (self.stop)() {
                return;
            }
            let name = ent.file_name();
            let Some(name) = name.to_str() else { continue };
            // The name first: only a name of ours costs a look at the entry.
            let partial = name.len() > ".partial".len() && name.ends_with(".partial");
            let temp = manifest_temp_of(name).filter(|m| !self.export_names.contains(m));
            let staging = staging_stem(name).is_some();
            let aside = set_aside_name(name);
            let deeper = depth < max && !name.starts_with('.');
            if !(partial || temp.is_some() || staging || aside.is_some() || deeper) {
                continue;
            }
            let Ok(ft) = ent.file_type() else { continue };
            let path = ent.path();
            if ft.is_file() {
                if partial && older(&path, self.now, STALE_PARTIAL_AGE) {
                    remove_counted(&path, "partial download nobody resumed for a week", out);
                } else if temp.is_some() && older(&path, self.now, STALE_STAGING_AGE) {
                    remove_counted(&path, "manifest temp of an install a crash cut off", out);
                }
            } else if ft.is_dir() {
                if staging {
                    if older(&path, self.now, STALE_STAGING_AGE) {
                        remove_counted(&path, "extraction folder of an interrupted install", out);
                    }
                } else if let Some((stem, pid)) = aside {
                    settle_set_aside(&path, stem, pid, out);
                } else if deeper {
                    self.dir(&path, depth + 1, max, out);
                }
            }
        }
    }
}

/// Last modified at least `age` before `now` (false when unknown).
fn older(p: &Path, now: SystemTime, age: Duration) -> bool {
    std::fs::symlink_metadata(p)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|m| now.duration_since(m).ok())
        .is_some_and(|d| d >= age)
}

/// The manifest of the install in `dir`, when the files it lists are intact.
fn intact(dir: &Path) -> Option<Manifest> {
    let path = dir.join(MANIFEST_FILE);
    read_manifest(&path).filter(|_| install_damage(&path, dir).is_none())
}

/// Deal with the set-aside install `old` (`.<stem>.old-<pid>`) of `<parent>/<stem>`; the caller
/// holds the download lock (see the module docs).
fn settle_set_aside(old: &Path, stem: &str, pid: u32, out: &mut Vec<Removed>) {
    if pid == std::process::id() {
        return;
    }
    let target = old.with_file_name(stem);
    // A runtime that another process loads is swapped under its in-use lock, without the download
    // lock (`in_use::apply_staged`): that lock too. No lock file = never loaded, never swapped so.
    // Loaded by this process (its shared lock is held): no swap can run either.
    let here = loaded_here(&target);
    let _exclusive = if !here && lock_path(&target).is_some_and(|p| p.exists()) {
        match ExclusiveUse::try_take(&target) {
            Ok(Some(x)) => Some(x),
            Ok(None) => {
                tracing::debug!(
                    "{}: {} is in use by a running process; left for a later sweep",
                    old.display(),
                    target.display()
                );
                return;
            }
            Err(e) => {
                tracing::debug!("{}: no in-use lock: {e}", target.display());
                return;
            }
        }
    } else {
        None
    };
    if std::fs::symlink_metadata(&target).is_ok() {
        if intact(&target).is_some() {
            remove_counted(
                old,
                "previous install an update set aside; the update is in place",
                out,
            );
        } else {
            tracing::info!(
                "keeping {}: {} has no intact install of ours beside it, so it is not provably \
                 superseded; delete it by hand once {} works",
                old.display(),
                target.display(),
                target.display()
            );
        }
        return;
    }
    restore(old, &target, out);
}

/// `<target>` is missing and `old` holds what was there before a swap that a crash cut between
/// its two renames: move the complete update in, else `old` back (see the module docs).
fn restore(old: &Path, target: &Path, out: &mut Vec<Removed>) {
    let previous = intact(old);
    let current_version = |id: &str| resources().find(|r| r.id == id).map(|r| r.version);
    let update = [pending_dir(target), sibling_of(target, "staging")]
        .into_iter()
        .flatten()
        .filter_map(|d| intact(&d).map(|m| (d, m)))
        .filter(|(_, m)| previous.as_ref().is_none_or(|p| p.id == m.id))
        .min_by_key(|(_, m)| current_version(&m.id) != Some(m.version.as_str()));
    if let Some((dir, m)) = update {
        match std::fs::rename(&dir, target) {
            Ok(()) => {
                tracing::warn!(
                    "{} {} was being swapped in when the process stopped; finished the swap \
                     from {}",
                    m.id,
                    m.version,
                    dir.display()
                );
                remove_counted(
                    old,
                    "previous install set aside by an update that a crash cut off; the update \
                     is in place now",
                    out,
                );
            }
            Err(e) => tracing::warn!(
                "{} is missing after an interrupted update and moving {} there failed: {e}; the \
                 next sweep tries again (or rename it by hand)",
                target.display(),
                dir.display()
            ),
        }
        return;
    }
    let Some(m) = previous else {
        tracing::info!(
            "keeping {}: {} is missing, but it holds no intact install to put back; delete it by \
             hand if it is not needed",
            old.display(),
            target.display()
        );
        return;
    };
    match std::fs::rename(old, target) {
        Ok(()) => tracing::warn!(
            "{} was missing after an update that a crash cut off; restored {} {} from {}",
            target.display(),
            m.id,
            m.version,
            old.display()
        ),
        Err(e) => tracing::warn!(
            "{} is missing after an interrupted update and restoring it from {} failed: {e}; \
             the next sweep tries again (or rename it back by hand)",
            target.display(),
            old.display()
        ),
    }
}

/// `<parent>/.<name>.<what>` beside `target` (nothing created).
fn sibling_of(target: &Path, what: &str) -> Option<PathBuf> {
    let name = target.file_name()?.to_str()?;
    Some(target.with_file_name(format!(".{name}.{what}")))
}

/// Restore a runtime install that is missing after a swap a crash cut between its two renames
/// (RA resources-10; see the module docs): `target` is `<root>/<dest>` of a runtime. Takes the
/// download lock (skipped while another process holds it). True when `target` exists afterwards.
pub(crate) fn recover_missing(root: &Path, target: &Path) -> bool {
    if std::fs::symlink_metadata(target).is_ok() {
        return true;
    }
    let Some(parent) = target.parent() else {
        return false;
    };
    let Some(name) = target.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Ok(rd) = std::fs::read_dir(parent) else {
        return false;
    };
    let mut aside: Vec<(PathBuf, u32)> = rd
        .flatten()
        .filter_map(|e| {
            let n = e.file_name();
            let (stem, pid) = set_aside_name(n.to_str()?)?;
            (stem == name && e.file_type().is_ok_and(|t| t.is_dir())).then(|| (e.path(), pid))
        })
        .collect();
    if aside.is_empty() {
        return false;
    }
    if !root.is_dir() {
        return false;
    }
    let Ok(Some(_lock)) = DownloadLock::try_acquire(root) else {
        tracing::debug!(
            "{} is missing beside a set-aside install; another process downloads, left for later",
            target.display()
        );
        return false;
    };
    // Newest last: the most recent swap is the one that was cut.
    aside.sort_by_key(|(p, _)| std::fs::symlink_metadata(p).and_then(|m| m.modified()).ok());
    let mut out = Vec::new();
    for (old, pid) in aside.into_iter().rev() {
        settle_set_aside(&old, name, pid, &mut out);
        if std::fs::symlink_metadata(target).is_ok() {
            return true;
        }
    }
    false
}

/// The runtime install folders under `root` this program downloads into (OpenVINO, each ONNX
/// Runtime flavor, the CUDA libraries).
pub(crate) fn runtime_targets(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = resources()
        .filter(|r| {
            matches!(
                r.kind,
                ResourceKind::OpenVinoRuntime
                    | ResourceKind::OnnxRuntime(_)
                    | ResourceKind::CudaLibs
            )
        })
        .map(|r| root.join(r.dest))
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The download root of the runtime install `target` of resource `id` (`target` minus the
/// resource's destination), when `target` is where the catalog puts it.
pub(super) fn root_of(target: &Path, id: &str) -> Option<PathBuf> {
    let dest = Path::new(resources().find(|r| r.id == id)?.dest);
    if !target.ends_with(dest) {
        return None;
    }
    target
        .ancestors()
        .nth(dest.components().count())
        .map(Path::to_path_buf)
}

/// The verified archives in `<root>/.downloads` that nothing needs any more (see the module docs).
fn superseded_archives(
    root: &Path,
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    let dir = root.join(DOWNLOADS_DIR);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return;
    };
    for ent in rd.flatten() {
        if stop() {
            return;
        }
        let name = ent.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.ends_with(".partial") || name.starts_with('.') {
            continue;
        }
        let path = ent.path();
        if !ent.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        if let Some(archive) = name.strip_suffix(KEEP_SUFFIX) {
            // A marker whose archive went (deleted by hand): nothing left to keep.
            if !archive.is_empty() && std::fs::symlink_metadata(dir.join(archive)).is_err() {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        if keep_marker(&path).exists() || !older(&path, now, STALE_PARTIAL_AGE) {
            continue;
        }
        let owners: Vec<&'static Resource> = resources()
            .filter(|r| r.parts.iter().any(|p| p.file_name == name))
            .collect();
        if owners.is_empty() {
            remove_counted(
                &path,
                "downloaded archive of a version no current release pins",
                out,
            );
        } else if owners.iter().any(|r| is_installed(r, &root.join(r.dest))) {
            remove_counted(
                &path,
                "downloaded archive of a version that is installed already",
                out,
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp(tag: &str) -> TempDir {
        let d = std::env::temp_dir().join(format!("bop-left-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        TempDir(d)
    }

    /// Set `p`'s modification time `by` into the past. Folders only on Unix (Windows opens no
    /// folder with `File::open`); the tests that age folders are Unix-only.
    fn age(p: &Path, by: Duration) {
        let f = if p.is_dir() {
            std::fs::File::open(p).unwrap()
        } else {
            std::fs::OpenOptions::new().write(true).open(p).unwrap()
        };
        f.set_modified(SystemTime::now() - by).unwrap();
    }

    /// A complete install of `id` at `version` in `dir`.
    fn install(dir: &Path, id: &str, version: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("lib.bin"), version.as_bytes()).unwrap();
        let m = Manifest {
            id: id.into(),
            version: version.into(),
            sha256: Vec::new(),
            files: vec!["lib.bin".into()],
        };
        write_manifest(&dir.join(MANIFEST_FILE), &m, dir).unwrap();
    }

    fn version(dir: &Path) -> String {
        std::fs::read_to_string(dir.join("lib.bin")).unwrap()
    }

    fn run(root: &Path, models: &[PathBuf]) -> Vec<Removed> {
        let mut out = Vec::new();
        assert!(clean(root, models, SystemTime::now(), &mut out, &|| false));
        out
    }

    #[test]
    fn name_shapes_are_exact() {
        assert_eq!(staging_stem(".yolo.staging"), Some("yolo"));
        assert_eq!(staging_stem(".cuda-libs.staging"), Some("cuda-libs"));
        for n in [
            ".staging",
            "..staging",
            "yolo.staging",
            ".x.export-staging",
            ".cam.staging.train-staging",
            ".old-cam.train-staging",
        ] {
            assert_eq!(staging_stem(n), None, "{n}");
        }
        assert_eq!(set_aside_name(".yolo.old-123"), Some(("yolo", 123)));
        assert_eq!(set_aside_name(".a.onnx.old-7"), Some(("a.onnx", 7)));
        for n in [
            ".old-cam.train-staging",
            ".old-123",
            ".yolo.old-",
            ".yolo.old-12a",
            "yolo.old-1",
            ".cpu.in-use",
            ".openvino.pending",
            ".yolo.old-99999999999999999999",
        ] {
            assert_eq!(set_aside_name(n), None, "{n}");
        }
        assert_eq!(manifest_temp_of(".installed.json.tmp"), Some(""));
        assert_eq!(manifest_temp_of(".yolo.installed.json.tmp"), Some("yolo"));
        assert_eq!(manifest_temp_of("..installed.json.tmp"), None);
        assert_eq!(manifest_temp_of("x.installed.json.tmp"), None);
        assert_eq!(manifest_temp_of(".installed.json"), None);
    }

    /// QA-035: only the shapes installs make are removed; a live training staging folder of a
    /// model named `old-cam` or `cam.staging`, an export's staging, the in-use lock and a staged
    /// update survive, however old.
    #[cfg(unix)]
    #[test]
    fn cleanup_removes_stale_partials_and_staging_only() {
        let t = tmp("cleanup");
        let root = &t.0;
        let dl = root.join(DOWNLOADS_DIR);
        let models = root.join("models");
        std::fs::create_dir_all(&dl).unwrap();
        std::fs::create_dir_all(models.join(".yolo.staging/inner")).unwrap();
        std::fs::write(models.join(".yolo.staging/inner/f"), b"x").unwrap();
        // `.yolo.old-123` beside an intact `yolo`: the swap finished.
        install(&models.join("yolo"), "x", "1");
        std::fs::create_dir_all(models.join(".yolo.old-123")).unwrap();
        std::fs::create_dir_all(models.join(".fresh.staging")).unwrap();
        let keep = [
            ".old-cam.train-staging",
            ".cam.staging.train-staging",
            ".x.export-staging",
            ".openvino.pending",
        ];
        for d in keep {
            std::fs::create_dir_all(models.join(d)).unwrap();
        }
        std::fs::write(models.join(".cpu.in-use"), b"").unwrap();
        let old = dl.join("a.zip.partial");
        let fresh = dl.join("b.zip.partial");
        for p in [&old, &fresh] {
            std::fs::write(p, b"x").unwrap();
        }
        age(&old, STALE_PARTIAL_AGE + Duration::from_secs(60));
        for d in [".yolo.staging", ".yolo.old-123"]
            .iter()
            .chain(keep.iter())
            .chain([".cpu.in-use"].iter())
        {
            age(&models.join(d), STALE_PARTIAL_AGE * 2);
        }
        let out = run(root, &[]);
        assert_eq!(out.len(), 3, "{out:?}");
        assert!(!old.exists());
        assert!(fresh.exists());
        assert!(!models.join(".yolo.staging").exists());
        assert!(!models.join(".yolo.old-123").exists());
        assert!(models.join(".fresh.staging").exists());
        assert!(models.join("yolo").join("lib.bin").exists());
        for d in keep.iter().chain([".cpu.in-use"].iter()) {
            assert!(models.join(d).exists(), "{d} must survive");
        }
        assert!(run(root, &[]).is_empty());
    }

    /// RA resources-7: the walk stays at the download destinations; a dataset tree with names
    /// that look like leftovers is never entered, while a models folder outside the root is.
    #[test]
    fn the_walk_never_enters_a_dataset_and_reaches_outside_model_folders() {
        let t = tmp("scope");
        let root = &t.0;
        let day = root.join("dataset/day/cam1/2026-11-01");
        std::fs::create_dir_all(day.join(".x.staging")).unwrap();
        std::fs::write(day.join("f.jpg.partial"), b"x").unwrap();
        #[cfg(unix)]
        age(&day.join(".x.staging"), STALE_PARTIAL_AGE * 2);
        age(&day.join("f.jpg.partial"), STALE_PARTIAL_AGE * 2);
        let o = tmp("outside");
        let partial = o.0.join("m.onnx.partial");
        std::fs::write(&partial, b"x").unwrap();
        age(&partial, STALE_PARTIAL_AGE * 2);
        // One level below a destination: export weights.
        let w = root.join("tools/export-weights/yolo26s");
        std::fs::create_dir_all(&w).unwrap();
        std::fs::write(w.join("yolo26s.pt.partial"), b"x").unwrap();
        age(&w.join("yolo26s.pt.partial"), STALE_PARTIAL_AGE * 2);
        let out = run(root, std::slice::from_ref(&o.0));
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(!partial.exists());
        assert!(!w.join("yolo26s.pt.partial").exists());
        assert!(day.join(".x.staging").exists());
        assert!(day.join("f.jpg.partial").exists());
    }

    /// RA resources-6: another process downloading (the lock held) means nothing is touched.
    #[test]
    fn a_held_download_lock_removes_nothing() {
        let t = tmp("held");
        let root = &t.0;
        let p = root.join(DOWNLOADS_DIR).join("a.zip.partial");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"x").unwrap();
        age(&p, STALE_PARTIAL_AGE * 2);
        let held = DownloadLock::try_acquire(root).unwrap().unwrap();
        let mut out = Vec::new();
        assert!(!clean(root, &[], SystemTime::now(), &mut out, &|| false));
        assert!(out.is_empty() && p.exists());
        drop(held);
        // A child another test forks shares the lock file's open description until it execs
        // (flock), so the lock can read as held for a moment after the drop; a cleanup that finds
        // it held changes nothing, so try again.
        let removed = (0..400)
            .map(|_| {
                let n = run(root, &[]).len();
                if n == 0 {
                    std::thread::sleep(Duration::from_millis(25));
                }
                n
            })
            .find(|&n| n > 0)
            .unwrap_or(0);
        assert_eq!(removed, 1);
        assert!(!p.exists());
    }

    /// RA resources-10: a crash between `swap_owned`'s two renames leaves `.x.old-<pid>` and no
    /// `x`: the previous install is put back, never deleted.
    #[test]
    fn a_swap_cut_between_its_renames_is_restored() {
        let t = tmp("restore");
        let parent = t.0.join("onnxruntime");
        let old = parent.join(".x.old-123");
        install(&old, "onnxruntime-cpu", "1.0");
        let out = run(&t.0, &[]);
        assert!(out.is_empty(), "{out:?}");
        assert!(!old.exists());
        assert_eq!(version(&parent.join("x")), "1.0");
        // A set-aside folder without an intact install is no copy to put back: left alone.
        let broken = parent.join(".y.old-5");
        std::fs::create_dir_all(&broken).unwrap();
        assert!(run(&t.0, &[]).is_empty());
        assert!(broken.exists() && !parent.join("y").exists());
    }

    /// RA resources-10: when the complete update is still beside the cut swap, it goes in and the
    /// set-aside previous install is deleted; the staged copy of the catalog's version wins.
    #[test]
    fn a_cut_swap_with_its_update_beside_finishes_the_swap() {
        let t = tmp("finish");
        let res = resources()
            .find(|r| r.kind == ResourceKind::OpenVinoRuntime)
            .unwrap();
        let target = t.0.join(res.dest);
        let parent = target.parent().unwrap();
        let name = target.file_name().unwrap().to_str().unwrap();
        install(&parent.join(format!(".{name}.old-77")), res.id, "0.1-old");
        install(
            &parent.join(format!(".{name}.staging")),
            res.id,
            "0.2-other",
        );
        install(&pending_dir(&target).unwrap(), res.id, res.version);
        let out = run(&t.0, &[]);
        assert_eq!(version(&target), res.version);
        assert!(!parent.join(format!(".{name}.old-77")).exists());
        assert_eq!(out.len(), 1, "{out:?}");
    }

    /// A set-aside install of a runtime that a process has loaded is left alone (its swap may be
    /// running under the in-use lock).
    #[test]
    fn a_loaded_runtime_set_aside_is_left_alone() {
        let t = tmp("inuse");
        let parent = t.0.join("onnxruntime");
        install(&parent.join("x"), "id", "2");
        let old = parent.join(".x.old-9");
        install(&old, "id", "1");
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path(&parent.join("x")).unwrap())
            .unwrap();
        lock.try_lock_shared().unwrap();
        assert!(run(&t.0, &[]).is_empty());
        assert!(old.exists());
        lock.unlock().unwrap();
        assert_eq!(run(&t.0, &[]).len(), 1);
        assert!(!old.exists());
    }

    /// This process's own set-aside folder is a swap in progress: never touched.
    #[test]
    fn this_process_set_aside_is_left_alone() {
        let t = tmp("own");
        let old = t.0.join(format!(".x.old-{}", std::process::id()));
        install(&old, "id", "1");
        assert!(run(&t.0, &[]).is_empty());
        assert!(old.exists() && !t.0.join("x").exists());
    }

    /// `recover_missing` (startup, `before_load`) puts a cut swap's runtime back.
    #[test]
    fn recover_missing_restores_a_runtime() {
        let t = tmp("recover");
        let target = t.0.join("openvino");
        assert!(!recover_missing(&t.0, &target));
        install(&t.0.join(".openvino.old-4242"), "openvino-runtime", "1");
        assert!(recover_missing(&t.0, &target));
        assert_eq!(version(&target), "1");
        assert!(recover_missing(&t.0, &target));
    }

    #[test]
    fn root_of_strips_the_destination() {
        let root = Path::new("data");
        let ov = resources()
            .find(|r| r.kind == ResourceKind::OpenVinoRuntime)
            .unwrap();
        assert_eq!(
            root_of(&root.join(ov.dest), ov.id),
            Some(root.to_path_buf())
        );
        let cpu = resources()
            .find(|r| r.kind == ResourceKind::OnnxRuntime(catalog::Flavor::Cpu))
            .unwrap();
        assert_eq!(
            root_of(&root.join(cpu.dest), cpu.id),
            Some(root.to_path_buf())
        );
        assert_eq!(root_of(Path::new("elsewhere/x"), cpu.id), None);
    }

    /// RA resources-6: verified archives that nothing needs go after a week; one of an installed
    /// pin, one no pin names; a kept one (`keep_downloads`) and a fresh one stay.
    #[test]
    fn superseded_archives_go_kept_ones_stay() {
        let t = tmp("archives");
        let root = &t.0;
        let dl = root.join(DOWNLOADS_DIR);
        std::fs::create_dir_all(&dl).unwrap();
        let res = resources()
            .find(|r| {
                r.kind == ResourceKind::OpenVinoRuntime
                    && r.parts.iter().all(|p| p.archive.is_some())
            })
            .unwrap();
        let part = res.parts.first().unwrap().file_name;
        let unknown = dl.join("openvino-0.0.1-old.tgz");
        let installed = dl.join(part);
        let kept = dl.join("kept-0.0.1.zip");
        let fresh = dl.join("fresh-0.0.1.zip");
        for p in [&unknown, &installed, &kept, &fresh] {
            std::fs::write(p, b"x").unwrap();
        }
        std::fs::write(keep_marker(&kept), b"").unwrap();
        for p in [&unknown, &installed, &kept] {
            age(p, STALE_PARTIAL_AGE * 2);
        }
        // Not installed yet: the pinned archive stays (a retry extracts it without downloading).
        let out = run(root, &[]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(!unknown.exists() && installed.exists());
        let target = root.join(res.dest);
        install(&target, res.id, res.version);
        assert_eq!(run(root, &[]).len(), 1);
        assert!(!installed.exists());
        assert!(kept.exists() && keep_marker(&kept).exists() && fresh.exists());
        // A marker whose archive was deleted by hand goes with it.
        std::fs::remove_file(&kept).unwrap();
        run(root, &[]);
        assert!(!keep_marker(&kept).exists());
    }

    /// Manifest temps of a cut write go after an hour; an exported model's is the export's.
    #[test]
    fn manifest_temps_go_except_an_exports() {
        let t = tmp("temps");
        let models = t.0.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let export = catalog::EXPORT_MODELS
            .iter()
            .find_map(Resource::model_name)
            .unwrap();
        let mine = models.join(".IPcam-general.installed.json.tmp");
        let theirs = models.join(format!(".{export}.installed.json.tmp"));
        let fresh = models.join(".other.installed.json.tmp");
        for p in [&mine, &theirs, &fresh] {
            std::fs::write(p, b"{}").unwrap();
        }
        age(&mine, STALE_STAGING_AGE * 2);
        age(&theirs, STALE_STAGING_AGE * 2);
        assert_eq!(run(&t.0, &[]).len(), 1);
        assert!(!mine.exists() && theirs.exists() && fresh.exists());
    }

    /// The sweep stops between items.
    #[test]
    fn stop_ends_the_walk() {
        let t = tmp("stop");
        let p = t.0.join(DOWNLOADS_DIR).join("a.zip.partial");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"x").unwrap();
        age(&p, STALE_PARTIAL_AGE * 2);
        let mut out = Vec::new();
        clean(&t.0, &[], SystemTime::now(), &mut out, &|| true);
        assert!(out.is_empty() && p.exists());
    }
}
