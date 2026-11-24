//! Model-export leftovers swept by the storage sweep (audit 2026-11, phase 2, RA resources-2):
//! what a crashed or killed export left in the models folder and under `tools/`.
//!
//! - `.<name>.export-staging` folders: an export holds its staging folder busy
//!   ([`crate::storage::busy`]) for its whole run, so one whose marker is unlocked belongs to a
//!   crashed export and is removed. One without a marker (an export of a version before the
//!   markers, or one whose own removal failed) is removed only while this sweep holds the export
//!   lock: every export, of any version, holds that lock from before it creates its staging
//!   folder to after it removes it, so with the lock in hand no export is running.
//! - With the export lock held (`tools/.model-export.lock`, without waiting; a training
//!   environment setup holds it too, and the sweep then skips this part):
//!   - every folder in `tools/export-work/` (an export's intermediate files; the script file
//!     there is rewritten by each run and stays);
//!   - `.<file>.old-<pid>` files of another process beside an exported model's `.onnx` / `.yaml`
//!     (set aside by the install step): deleted when `<file>` is there (the install got that
//!     far), renamed back when it is missing (a crash between the two renames);
//!   - an exported model's manifest temp (`.<name>.installed.json.tmp`).
//!
//! Without `tools/` no export can run (it creates `tools/` before taking the lock), and the lock
//! file is not created by the sweep: only the busy-marker part runs then.

use super::{ExportLock, Paths};
use crate::resources::{catalog, manager};
use crate::storage::busy::{self, State};
use crate::storage::{Removed, remove_counted};
use std::path::{Path, PathBuf};

/// `.<stem>.export-staging` (non-empty stem): a model export's staging folder ([`Paths::staging`]).
pub(crate) fn export_staging_stem(name: &str) -> Option<&str> {
    name.strip_prefix('.')?
        .strip_suffix(".export-staging")
        .filter(|s| !s.is_empty())
}

/// Remove what crashed or killed model exports left behind, asking `stop` between items.
pub(crate) fn sweep_leftovers(
    config: &crate::config::Config,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    sweep_paths(&Paths::for_config(config), out, stop);
}

/// [`sweep_leftovers`] for `p`.
pub(super) fn sweep_paths(p: &Paths, out: &mut Vec<Removed>, stop: &dyn Fn() -> bool) {
    let lock = if p.tools.is_dir() {
        match ExportLock::try_acquire(&p.lock_file()) {
            Ok(l) => l,
            Err(e) => {
                tracing::debug!("{}: {e}", p.lock_file().display());
                None
            }
        }
    } else {
        None
    };
    if lock.is_none() {
        tracing::debug!(
            "model export cleanup: the export lock is held or there is no toolchain; only \
             crashed exports' staging folders are looked at"
        );
    }
    let locked = lock.is_some();
    staging(&p.models, locked, out, stop);
    if locked && !stop() {
        set_aside_and_temps(&p.models, out, stop);
    }
    if locked && !stop() {
        work(&p.work, out, stop);
    }
}

/// The export staging folders of `models` (see the module docs).
fn staging(models: &Path, locked: bool, out: &mut Vec<Removed>, stop: &dyn Fn() -> bool) {
    let Ok(rd) = std::fs::read_dir(models) else {
        return;
    };
    let mut targets: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let path = e.path();
        if export_staging_stem(&name).is_some() && e.file_type().is_ok_and(|t| t.is_dir()) {
            targets.push(path);
        } else if let Some(t) = busy::target_of(&path)
            && t.file_name()
                .and_then(|n| n.to_str())
                .and_then(export_staging_stem)
                .is_some()
        {
            // A marker whose folder is gone (a crash after the removal): the marker goes.
            targets.push(t);
        }
    }
    targets.sort();
    targets.dedup();
    for t in targets {
        if stop() {
            return;
        }
        match busy::probe(&t) {
            State::Live => {}
            State::Ended => {
                if locked && t.exists() {
                    remove_counted(
                        &t,
                        "staging folder of a model export that stopped without removing it",
                        out,
                    );
                }
            }
            State::Crashed(claim) => {
                if !t.exists()
                    || remove_counted(&t, "staging folder of a model export that crashed", out)
                {
                    claim.finish();
                }
            }
        }
    }
}

/// The set-aside output files and manifest temps of exports in `models` (the caller holds the
/// export lock).
fn set_aside_and_temps(models: &Path, out: &mut Vec<Removed>, stop: &dyn Fn() -> bool) {
    let mut outputs: Vec<String> = Vec::new();
    let mut temps: Vec<PathBuf> = Vec::new();
    for res in catalog::EXPORT_MODELS {
        if let Some((onnx, yaml)) = catalog::export_files(res) {
            outputs.push(onnx);
            outputs.push(yaml);
        }
        temps.push(manager::manifest_path(res, models).with_extension("json.tmp"));
    }
    let Ok(rd) = std::fs::read_dir(models) else {
        return;
    };
    for e in rd.flatten() {
        if stop() {
            return;
        }
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        let path = e.path();
        let is_file = || e.file_type().is_ok_and(|t| t.is_file());
        if let Some((file, pid)) = manager::set_aside_name(name)
            && pid != std::process::id()
            && outputs.iter().any(|o| o == file)
            && is_file()
        {
            let target = models.join(file);
            if std::fs::symlink_metadata(&target).is_ok() {
                remove_counted(
                    &path,
                    "previous exported model file set aside by an install a crash cut off; the \
                     new one is in place",
                    out,
                );
            } else {
                match std::fs::rename(&path, &target) {
                    Ok(()) => tracing::warn!(
                        "{} was missing after an export's install that a crash cut off; restored \
                         it from {}",
                        target.display(),
                        path.display()
                    ),
                    Err(e) => tracing::warn!(
                        "{} is missing after an interrupted export install and restoring it \
                         from {} failed: {e}; the next sweep tries again (or rename it back by \
                         hand)",
                        target.display(),
                        path.display()
                    ),
                }
            }
        } else if temps.contains(&path) && is_file() {
            remove_counted(
                &path,
                "manifest temp of a model export's install a crash cut off",
                out,
            );
        }
    }
}

/// The work folders in `work` (`tools/export-work/<name>`; the caller holds the export lock).
fn work(work: &Path, out: &mut Vec<Removed>, stop: &dyn Fn() -> bool) {
    let Ok(rd) = std::fs::read_dir(work) else {
        return;
    };
    for e in rd.flatten() {
        if stop() {
            return;
        }
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            remove_counted(
                &e.path(),
                "work folder of a model export that a crash or kill interrupted",
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

    fn paths(tag: &str) -> (TempDir, Paths) {
        let d = std::env::temp_dir().join(format!("bop-exleft-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join("tools")).unwrap();
        let p = Paths::new(&d, d.join("models"));
        std::fs::create_dir_all(&p.models).unwrap();
        (TempDir(d), p)
    }

    fn sweep(p: &Paths) -> Vec<Removed> {
        let mut out = Vec::new();
        sweep_paths(p, &mut out, &|| false);
        out
    }

    fn put(f: &Path) {
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, b"x").unwrap();
    }

    #[test]
    fn staging_names_are_exact() {
        assert_eq!(
            export_staging_stem(".yolo26n.export-staging"),
            Some("yolo26n")
        );
        for n in [
            ".export-staging",
            "yolo.export-staging",
            ".yolo.train-staging",
            ".yolo.staging",
            ".yolo.export-staging.busy",
        ] {
            assert_eq!(export_staging_stem(n), None, "{n}");
        }
    }

    /// A crashed export (unlocked marker) loses its staging folder and marker; a running one
    /// (marker held) keeps both, even while the sweep holds the export lock.
    #[test]
    fn crashed_staging_goes_live_staging_stays() {
        let (_t, p) = paths("crash");
        let crashed = p.staging("yolo26n");
        put(&crashed.join("yolo26n.onnx"));
        // What a crash leaves: the marker, unlocked.
        put(&busy::marker_of(&crashed));
        let live = p.staging("yolo26s");
        let held = crate::storage::Busy::hold(&live);
        put(&live.join("yolo26s.onnx"));
        let out = sweep(&p);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(!crashed.exists() && !busy::marker_of(&crashed).exists());
        assert!(live.join("yolo26s.onnx").exists());
        drop(held);
    }

    /// A staging folder without a marker (an older version's, or one whose removal failed) goes
    /// only while no export holds the lock.
    #[test]
    fn unmarked_staging_goes_only_under_the_export_lock() {
        let (_t, p) = paths("legacy");
        let s = p.staging("yolo26n");
        put(&s.join("yolo26n.onnx"));
        let held = ExportLock::try_acquire(&p.lock_file()).unwrap().unwrap();
        assert!(sweep(&p).is_empty());
        assert!(s.exists());
        drop(held);
        assert_eq!(sweep(&p).len(), 1);
        assert!(!s.exists());
    }

    /// With the lock: work folders, set-aside files (deleted beside their file, renamed back
    /// without it) and manifest temps go; this process's set-aside file and other models' files
    /// stay. With the lock held by an export, nothing of it is touched.
    #[test]
    fn work_set_aside_and_temps_under_the_lock() {
        let (_t, p) = paths("lock");
        let res = catalog::export_model("yolo26n").unwrap();
        let work = p.work.join("yolo26n");
        put(&work.join("tmp.onnx"));
        put(&p.work.join(super::super::SCRIPT_NAME));
        put(&p.models.join("yolo26n.onnx"));
        let aside_beside = p.models.join(".yolo26n.onnx.old-123");
        let aside_alone = p.models.join(".yolo26n.yaml.old-124");
        std::fs::write(&aside_beside, b"old onnx").unwrap();
        std::fs::write(&aside_alone, b"old yaml").unwrap();
        let own = p
            .models
            .join(format!(".yolo26n.xml.old-{}", std::process::id()));
        let mine = p
            .models
            .join(format!(".yolo26n.onnx.old-{}", std::process::id()));
        let foreign = p.models.join(".mine.onnx.old-5");
        for f in [&own, &mine, &foreign] {
            put(f);
        }
        let temp = manager::manifest_path(res, &p.models).with_extension("json.tmp");
        put(&temp);

        let held = ExportLock::try_acquire(&p.lock_file()).unwrap().unwrap();
        assert!(sweep(&p).is_empty());
        assert!(work.exists() && aside_beside.exists() && temp.exists());
        drop(held);

        // A child another test forks shares the lock file's open description until it execs
        // (flock), so the lock can read as held for a moment after the drop; a sweep that finds
        // it held changes nothing, so try again.
        let out = (0..400)
            .find_map(|_| {
                let out = sweep(&p);
                if out.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(25));
                    None
                } else {
                    Some(out)
                }
            })
            .unwrap_or_default();
        assert_eq!(out.len(), 3, "{out:?}");
        assert!(!work.exists() && !aside_beside.exists() && !temp.exists());
        assert!(p.work.join(super::super::SCRIPT_NAME).exists());
        assert!(!aside_alone.exists());
        assert_eq!(
            std::fs::read(p.models.join("yolo26n.yaml")).unwrap(),
            b"old yaml"
        );
        assert!(own.exists() && mine.exists() && foreign.exists());
    }

    /// No `tools/`: no lock file is created, and only crashed staging is looked at.
    #[test]
    fn without_a_toolchain_no_lock_file_is_made() {
        let (t, p) = paths("notools");
        std::fs::remove_dir_all(t.0.join("tools")).unwrap();
        let s = p.staging("yolo26n");
        put(&s.join("x"));
        assert!(sweep(&p).is_empty());
        assert!(s.exists() && !p.tools.exists());
    }

    /// RA resources-3: a cancel that lands after the exporter exited successfully (or any early
    /// return after the staging folder was made) leaves neither staging, its marker nor work.
    #[test]
    fn an_export_that_ends_early_leaves_no_staging_or_work() {
        use super::super::fake::Fake;
        let (t, _) = paths("early");
        let j = super::super::ExportJob {
            resource: catalog::export_model("yolo26n").unwrap(),
            paths: Paths::new(&t.0, t.0.join("models")),
            platform: catalog::Platform::MACOS_ARM64,
            timeouts: super::super::Timeouts::default(),
            system_dir: None,
            task: "export",
        };
        let fake = Fake::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // Cancel as the exporter finishes: its run returns Ok, the next check sees the cancel.
        let r = super::super::run_export(
            &j,
            &fake,
            &fake,
            &mut |pr| {
                if let super::super::Progress::Line {
                    stage: super::super::Stage::Export,
                    ..
                } = pr
                {
                    cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            },
            &cancel,
        );
        let e = r.unwrap_err();
        assert!(e.cancelled, "{}", e.message);
        let staging = j.paths.staging("yolo26n");
        assert!(!staging.exists(), "staging left behind");
        assert!(!busy::marker_of(&staging).exists());
        assert!(!j.paths.work.join("yolo26n").exists());
    }
}
