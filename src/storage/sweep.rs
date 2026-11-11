//! The storage sweep: at startup and once a day on the retention thread
//! ([`crate::retention::spawn_sweeper`]), and piece by piece from the Storage card's Clear.
//!
//! In order: cache folders unused for [`super::caches::UNUSED_DAYS`] and the CoreML orphans
//! ([`super::caches`]); `.<name>.train-staging` folders no running training holds; work folders
//! of auto-label, import, overlap and dataset-check jobs that crashed ([`super::busy`]), an
//! auto-label's finished proposals written to their frames first
//! ([`crate::training::autolabel::job::recover_crashed`],
//! the folder kept and the reason logged when that fails); then `review-queue.jsonl` compaction
//! of the dataset and review folders ([`crate::collect::layout::compact_review`]). Every removal is
//! logged with its size ([`super::remove_counted`]), and the sweep ends with one summary line.

use super::busy::{self, State};
use super::{Removed, caches, remove_counted};
use crate::config::Config;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Time between storage sweeps.
pub const INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// The work folders of the training queue's jobs: `<training dir>/<sub>/<task id>`.
pub const WORK_DIRS: [(&str, WorkKind); 4] = [
    ("autolabel", WorkKind::Autolabel),
    ("imports", WorkKind::Import),
    ("overlaps", WorkKind::Overlaps),
    ("refine", WorkKind::Refine),
];

/// Whose work folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    Autolabel,
    Import,
    Overlaps,
    /// A dataset check: its fold exports and fold models, all rebuildable.
    Refine,
}

impl WorkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkKind::Autolabel => "auto-label",
            WorkKind::Import => "import",
            WorkKind::Overlaps => "overlap check",
            WorkKind::Refine => "dataset check",
        }
    }
}

/// What a sweep looks at, resolved from the config.
#[derive(Debug, Clone)]
pub struct Settings {
    config: Box<Config>,
}

impl Settings {
    pub fn for_config(config: &Config) -> Self {
        Self {
            config: Box::new(config.clone()),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// `cache_dir`, None when caching is off.
    pub fn cache_root(&self) -> Option<PathBuf> {
        self.config.cache_dir_path()
    }

    pub fn models_dir(&self) -> PathBuf {
        self.config.data_path(&self.config.models_dir)
    }

    pub fn training_dir(&self) -> PathBuf {
        crate::training::training_dir(&self.config)
    }

    /// The model files there are now: `.onnx` files of the models folder and every configured
    /// model's path.
    pub fn model_files(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(self.models_dir())
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.extend(
            self.config
                .models
                .iter()
                .map(|m| self.config.data_path(&m.path)),
        );
        out.sort();
        out.dedup();
        out
    }

    /// The dataset folder and the folder under review (one entry when they are the same).
    pub fn dataset_dirs(&self) -> Vec<PathBuf> {
        let root = self.config.data_root();
        let mut v = vec![
            self.config.collect.resolved_dir(&root),
            self.config.collect.resolved_review_dir(&root),
        ];
        v.dedup();
        v
    }

    /// The log folder (`log_path`), None when logs go to stdout.
    pub fn log_dir(&self) -> Option<PathBuf> {
        self.config.log_path.as_deref().map(crate::resolve_path)
    }
}

/// What one sweep did.
#[derive(Debug, Default)]
pub struct Report {
    pub removed: Vec<Removed>,
    /// Folders kept although they looked like leftovers, with why.
    pub kept: Vec<String>,
    /// Review queues compacted: (folder, lines dropped).
    pub compacted: Vec<(PathBuf, usize)>,
}

/// One full sweep (see the module docs).
pub fn run(s: &Settings) -> Report {
    let now = SystemTime::now();
    let mut rep = Report::default();
    sweep_caches(s, now, &mut rep.removed);
    sweep_staging(&s.models_dir(), &mut rep.removed);
    sweep_work(s, &mut rep);
    for dir in s.dataset_dirs() {
        match crate::collect::layout::compact_review(&dir) {
            Ok(Some(c)) if c.dropped() > 0 => {
                tracing::info!(
                    "storage: review queue of {} compacted: {} duplicate and {} gone frame line(s) \
                     dropped, {} kept",
                    dir.display(),
                    c.duplicates,
                    c.gone,
                    c.kept
                );
                rep.compacted.push((dir, c.dropped()));
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(
                "storage: compacting the review queue of {}: {e:#}; it is tried again tomorrow",
                dir.display()
            ),
        }
    }
    let freed = super::total(&rep.removed);
    if !rep.removed.is_empty() || !rep.kept.is_empty() {
        tracing::info!(
            "storage sweep: {} item(s) removed, {} freed{}",
            rep.removed.len(),
            crate::resources::catalog::format_size(freed),
            if rep.kept.is_empty() {
                String::new()
            } else {
                format!("; kept: {}", rep.kept.join("; "))
            }
        );
    }
    rep
}

/// Unused cache folders and the CoreML orphans.
pub fn sweep_caches(s: &Settings, now: SystemTime, out: &mut Vec<Removed>) {
    let Some(root) = s.cache_root() else { return };
    caches::sweep_unused(&root, now, out);
    let coreml = root.join("coreml");
    if coreml.is_dir() {
        caches::sweep_coreml(&coreml, &s.model_files(), now, out);
    }
}

/// Whether `name` is a training run's staging folder (`.<name>.train-staging`).
pub fn is_staging_name(name: &str) -> bool {
    name.len() > ".train-staging".len() + 1
        && name.starts_with('.')
        && name.ends_with(".train-staging")
}

/// The staging folders under `models_dir` no running training holds (a staging folder outlives
/// its run only after a crash or a failed cleanup: nothing keeps one on purpose).
pub fn leftover_staging(models_dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| e.file_name().to_str().is_some_and(is_staging_name))
        .map(|e| e.path())
        .filter(|p| !matches!(busy::probe(p), State::Live))
        .collect();
    out.sort();
    out
}

/// Remove the leftover staging folders of `models_dir` and the markers of crashed runs.
pub fn sweep_staging(models_dir: &Path, out: &mut Vec<Removed>) {
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return;
    };
    let mut targets: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let path = e.path();
        if is_staging_name(&name) && e.file_type().is_ok_and(|t| t.is_dir()) {
            targets.push(path);
        } else if let Some(t) = busy::target_of(&path)
            && t.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_staging_name)
        {
            targets.push(t);
        }
    }
    targets.sort();
    targets.dedup();
    for t in targets {
        match busy::probe(&t) {
            State::Live => {}
            State::Ended => {
                if t.exists() {
                    remove_counted(&t, "staging folder of an interrupted training run", out);
                }
            }
            State::Crashed(claim) => {
                if !t.exists()
                    || remove_counted(&t, "staging folder of a training run that crashed", out)
                {
                    claim.finish();
                }
            }
        }
    }
}

/// A work folder of a job that crashed.
#[derive(Debug, Clone, PartialEq)]
pub struct CrashedWork {
    pub kind: WorkKind,
    pub dir: PathBuf,
}

/// Work folders that are neither running nor crashed: kept on purpose (an import's keep work, an
/// auto-label's proposals it could not write) or left by a version without busy markers.
pub fn kept_work(training_dir: &Path) -> Vec<CrashedWork> {
    let mut out = Vec::new();
    for (sub, kind) in WORK_DIRS {
        let Ok(rd) = std::fs::read_dir(training_dir.join(sub)) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if e.file_type().is_ok_and(|t| t.is_dir()) && !busy::marker_of(&p).exists() {
                out.push(CrashedWork { kind, dir: p });
            }
        }
    }
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    out
}

/// The markers of crashed jobs under `training_dir` (their folders may be gone already).
pub fn crashed_work(training_dir: &Path) -> Vec<CrashedWork> {
    let mut out = Vec::new();
    for (sub, kind) in WORK_DIRS {
        let Ok(rd) = std::fs::read_dir(training_dir.join(sub)) else {
            continue;
        };
        for e in rd.flatten() {
            let Some(t) = busy::target_of(&e.path()) else {
                continue;
            };
            if !e.file_type().is_ok_and(|t| t.is_file()) {
                continue;
            }
            // Probed without keeping the claim: the sweep claims it again.
            if matches!(busy::probe(&t), State::Crashed(_)) {
                out.push(CrashedWork { kind, dir: t });
            }
        }
    }
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    out
}

/// Deal with the crashed work folders: an auto-label's proposals are written first; then the
/// folder goes. A folder whose proposals could not all be written stays (with its marker, so the
/// next sweep tries again) and the reason goes to `rep.kept`.
pub fn sweep_work(s: &Settings, rep: &mut Report) {
    for w in crashed_work(&s.training_dir()) {
        let State::Crashed(claim) = busy::probe(&w.dir) else {
            continue;
        };
        if !w.dir.exists() {
            claim.finish();
            continue;
        }
        if w.kind == WorkKind::Autolabel
            && let Err(why) = recover_autolabel(&w.dir, s.config())
        {
            tracing::warn!("storage: kept {}: {why}", w.dir.display());
            rep.kept.push(format!("{}: {why}", w.dir.display()));
            continue;
        }
        let article = if w.kind == WorkKind::Refine {
            "a"
        } else {
            "an"
        };
        let why = format!("work folder of {article} {} that crashed", w.kind.as_str());
        if remove_counted(&w.dir, &why, &mut rep.removed) {
            claim.finish();
        }
    }
}

/// Write the finished proposals of the crashed auto-label in `work` to their frames. Err (why the
/// folder must stay) when some could not be written.
fn recover_autolabel(work: &Path, config: &Config) -> Result<(), String> {
    if !work
        .join(crate::training::autolabel::job::PROPOSALS_FILE)
        .is_file()
    {
        return Ok(());
    }
    match crate::training::autolabel::job::recover_crashed(work, config) {
        Ok(r) if r.frames_failed == 0 => {
            tracing::info!(
                "storage: {}: wrote the {} finished frame(s) of a crashed auto-label to their \
                 label files ({} queued for review)",
                work.display(),
                r.frames_saved,
                r.queued
            );
            Ok(())
        }
        Ok(r) => Err(format!(
            "{} of {} label files of a crashed auto-label could not be written (first: {}); \
             free disk space or fix the dataset folder's permissions, the next sweep tries again",
            r.frames_failed,
            r.frames_failed + r.frames_saved,
            r.first_error.as_deref().unwrap_or("unknown")
        )),
        Err(e) => Err(format!(
            "the finished frames of a crashed auto-label could not be written ({e:#}); the next \
             sweep tries again, or delete the folder by hand to drop them"
        )),
    }
}

/// Old log files under `dir`: `blue_onyx_prism.log*` files except the newest and any written in
/// the last day (the current day's file is never touched).
pub fn old_logs(dir: &Path, now: SystemTime) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(PathBuf, SystemTime)> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("blue_onyx_prism.log"))
        })
        .filter_map(|e| Some((e.path(), e.metadata().ok()?.modified().ok()?)))
        .collect();
    files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    files.pop();
    let day = Duration::from_secs(24 * 60 * 60);
    files
        .into_iter()
        .filter(|(_, m)| now.duration_since(*m).is_ok_and(|a| a >= day))
        .map(|(p, _)| p)
        .collect()
}

/// What the Storage card's Clear can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clear {
    /// One cache folder, by name.
    Cache(String),
    Staging,
    /// The crashed work folders.
    Work,
    Logs,
}

impl Clear {
    /// From the card's id: `cache:<name>`, `staging`, `work`, `logs`.
    pub fn parse(id: &str) -> Option<Self> {
        match id {
            "staging" => Some(Self::Staging),
            "work" => Some(Self::Work),
            "logs" => Some(Self::Logs),
            _ => id
                .strip_prefix("cache:")
                .filter(|n| caches::is_cache_name(n))
                .map(|n| Self::Cache(n.to_string())),
        }
    }
}

/// Run one Clear. The cache is looked up among the folders listed, never joined from the name.
pub fn clear(s: &Settings, what: &Clear) -> anyhow::Result<Report> {
    let mut rep = Report::default();
    match what {
        Clear::Cache(name) => {
            let root = s
                .cache_root()
                .ok_or_else(|| anyhow::anyhow!("caching is off (cache_dir is empty)"))?;
            let dir = caches::cache_dirs(&root)
                .into_iter()
                .find(|d| d.file_name().is_some_and(|n| n == name.as_str()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "there is no cache folder {name} in {}; reload the page",
                        root.display()
                    )
                })?;
            caches::clear(&dir, &mut rep.removed);
        }
        Clear::Staging => sweep_staging(&s.models_dir(), &mut rep.removed),
        Clear::Work => sweep_work(s, &mut rep),
        Clear::Logs => {
            if let Some(dir) = s.log_dir() {
                for p in old_logs(&dir, SystemTime::now()) {
                    remove_counted(
                        &p,
                        "old log file, cleared from the Storage card",
                        &mut rep.removed,
                    );
                }
            }
        }
    }
    Ok(rep)
}

/// Remove the review export `name` of `training_dir`'s `exports` folder (the Storage card's
/// Remove). The name must be one the card lists, never a path; an export a queued or running
/// training run reads is refused, and so is one `configured` (`training.datasets`, resolved)
/// lists, which `train run` and the Train tab would then miss.
pub fn remove_export(
    training_dir: &Path,
    name: &str,
    runs: &[crate::training::service::RunRecord],
    configured: &[std::path::PathBuf],
) -> anyhow::Result<Removed> {
    let dir = super::usage::exports(training_dir)
        .into_iter()
        .find(|p| p.file_name().is_some_and(|n| n == name))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "there is no export {name} in {}; reload the page",
                training_dir.join("exports").display()
            )
        })?;
    let same = |p: &Path| {
        p == dir
            || p.canonicalize()
                .ok()
                .zip(dir.canonicalize().ok())
                .is_some_and(|(a, b)| a == b)
    };
    if let Some(r) = runs
        .iter()
        .find(|r| r.state.is_busy() && r.request.datasets.iter().any(|d| same(d)))
    {
        anyhow::bail!(
            "training run {} ({}) uses the export {name}; wait for it to finish or cancel it, \
             then remove the export",
            r.name,
            r.id
        );
    }
    if configured.iter().any(|d| same(d)) {
        anyhow::bail!(
            "the export {name} is listed in training.datasets in the config; take it off that list \
             (Train tab or the config file) first, then remove the export"
        );
    }
    let mut out = Vec::new();
    if !remove_counted(
        &dir,
        "review export removed from the Storage card",
        &mut out,
    ) {
        anyhow::bail!(
            "{} could not be removed (a file may be open in another program); see the log",
            dir.display()
        );
    }
    out.pop()
        .ok_or_else(|| anyhow::anyhow!("internal: the removal was not recorded"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-sweep-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(p: &Path, n: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![1u8; n]).unwrap();
    }

    fn settings(root: &Path) -> Settings {
        let mut c = Config {
            download_dir: Some(root.to_path_buf()),
            cache_dir: root.join("cache").display().to_string(),
            log_path: Some(root.join("logs")),
            ..Config::default()
        };
        c.models.clear();
        Settings::for_config(&c)
    }

    #[test]
    fn staging_names() {
        assert!(is_staging_name(".m.train-staging"));
        assert!(!is_staging_name(".train-staging"));
        assert!(!is_staging_name("m.train-staging"));
        assert!(!is_staging_name(".m.train-staging.busy"));
    }

    #[test]
    fn clear_ids_parse_and_refuse_paths() {
        assert_eq!(Clear::parse("staging"), Some(Clear::Staging));
        assert_eq!(
            Clear::parse("cache:openvino-cpu-f16"),
            Some(Clear::Cache("openvino-cpu-f16".into()))
        );
        assert_eq!(Clear::parse("cache:../models"), None);
        assert_eq!(Clear::parse("cache:"), None);
        assert_eq!(Clear::parse("models"), None);
    }

    #[test]
    fn staging_goes_unless_a_run_holds_it() {
        let root = tmp("staging");
        let s = settings(&root);
        let models = s.models_dir();
        let held_dir = models.join(".live.train-staging");
        let left = models.join(".crashed.train-staging");
        let no_marker = models.join(".old.train-staging");
        let held = busy::Busy::hold(&held_dir);
        for d in [&held_dir, &left, &no_marker] {
            put(&d.join("model.onnx"), 10);
        }
        // A crash: the marker stays behind, unlocked.
        std::fs::write(busy::marker_of(&left), b"1\n").unwrap();
        // A stray marker whose folder is already gone.
        std::fs::write(busy::marker_of(&models.join(".gone.train-staging")), b"1\n").unwrap();
        put(&models.join("keep.onnx"), 5);
        let mut out = Vec::new();
        sweep_staging(&models, &mut out);
        let mut gone: Vec<&PathBuf> = out.iter().map(|r| &r.path).collect();
        gone.sort();
        assert_eq!(gone, [&left, &no_marker]);
        assert!(
            held_dir.join("model.onnx").is_file(),
            "a running run's staging stays"
        );
        assert!(!busy::marker_of(&left).exists());
        assert!(!busy::marker_of(&models.join(".gone.train-staging")).exists());
        assert!(models.join("keep.onnx").is_file());
        drop(held);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn crashed_import_and_overlap_work_goes_kept_and_running_work_stays() {
        let root = tmp("work");
        let s = settings(&root);
        let t = s.training_dir();
        let crashed = t.join("imports/20261107-1");
        let kept = t.join("imports/20261107-2");
        let running = t.join("overlaps/20261107-3");
        let crashed_ov = t.join("overlaps/20261107-4");
        for d in [&crashed, &kept, &running, &crashed_ov] {
            put(&d.join("frames/a.jpg"), 100);
        }
        std::fs::write(busy::marker_of(&crashed), b"1\n").unwrap();
        std::fs::write(busy::marker_of(&crashed_ov), b"1\n").unwrap();
        let hold = busy::Busy::hold(&running);
        // A failed training run's checkpoint: never a work folder of these jobs.
        put(&t.join("20261107-run/work/best.pt"), 50);
        assert_eq!(crashed_work(&t).len(), 2);
        let mut rep = Report::default();
        sweep_work(&s, &mut rep);
        assert_eq!(rep.removed.len(), 2, "{rep:?}");
        assert!(!crashed.exists() && !crashed_ov.exists());
        assert!(!busy::marker_of(&crashed).exists());
        assert!(kept.join("frames/a.jpg").is_file(), "kept on purpose");
        assert!(running.join("frames/a.jpg").is_file(), "running");
        assert!(t.join("20261107-run/work/best.pt").is_file());
        assert_eq!(
            kept_work(&t).iter().map(|w| &w.dir).collect::<Vec<_>>(),
            [&kept]
        );
        drop(hold);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_export_is_removed_by_name_only() {
        let root = tmp("export");
        let t = root.join("training");
        put(&t.join("exports/e1/classes.txt"), 4);
        put(&t.join("exports/e1/images/a.jpg"), 4);
        put(&t.join("exports/notes/readme.txt"), 4);
        assert!(remove_export(&t, "../training", &[], &[]).is_err());
        assert!(
            remove_export(&t, "notes", &[], &[]).is_err(),
            "not an export"
        );
        // An export the config lists stays.
        let listed = [t.join("exports/e1")];
        let e = remove_export(&t, "e1", &[], &listed)
            .unwrap_err()
            .to_string();
        assert!(e.contains("training.datasets"), "{e}");
        assert!(t.join("exports/e1/classes.txt").is_file());
        let r = remove_export(&t, "e1", &[], &[]).unwrap();
        assert_eq!(r.bytes, 8);
        assert!(!t.join("exports/e1").exists());
        assert!(t.join("exports/notes/readme.txt").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn old_logs_keep_the_newest_and_todays() {
        let root = tmp("logs");
        let old = root.join("blue_onyx_prism.log.2026-10-01");
        let newest = root.join("blue_onyx_prism.log.2026-10-02");
        let other = root.join("notes.txt");
        for p in [&old, &newest, &other] {
            put(p, 3);
        }
        let later = SystemTime::now() + Duration::from_secs(3 * 86_400);
        // The newest file always stays, whatever its age.
        let f = std::fs::File::options().write(true).open(&newest).unwrap();
        f.set_modified(SystemTime::now() + Duration::from_secs(60))
            .unwrap();
        assert_eq!(old_logs(&root, later), std::slice::from_ref(&old));
        assert!(
            old_logs(&root, SystemTime::now()).is_empty(),
            "written today"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_full_sweep_never_touches_data() {
        let root = tmp("data");
        let s = settings(&root);
        let t = s.training_dir();
        let data = [
            s.models_dir().join("m.onnx"),
            s.models_dir().join("m.yaml"),
            t.join("exports/e1/classes.txt"),
            t.join("exports/e1/images/train/a.jpg"),
            t.join("20261107-run/work/weights/best.pt"),
            t.join("runs.json"),
            t.join("tasks.json"),
            s.dataset_dirs()[0].join("night/cam/2026-11-07/a.jpg"),
            s.dataset_dirs()[0].join("night/cam/2026-11-07/a.labels.json"),
            root.join("logs/blue_onyx_prism.log.2026-01-01"),
            root.join("cfg.json"),
        ];
        for p in &data {
            put(p, 7);
        }
        // Also a cache folder unused for long: the only thing that goes.
        let cache = root.join("cache/tensorrt");
        put(&cache.join("engine"), 9);
        std::fs::write(
            cache.join(caches::LAST_USED_FILE),
            "1000000\n", // 1970
        )
        .unwrap();
        let rep = run(&s);
        assert_eq!(
            rep.removed.iter().map(|r| &r.path).collect::<Vec<_>>(),
            [&cache]
        );
        for p in &data {
            assert!(p.is_file(), "{} must stay", p.display());
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
