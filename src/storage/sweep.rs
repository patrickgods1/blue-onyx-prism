//! The storage sweep: at startup and once a day on the retention thread
//! ([`crate::retention::spawn_sweeper`]), and piece by piece from the Storage card's Clear.
//!
//! What it removes, in order, each shape matched exactly and only when its owner is provably gone:
//! - crash-left temporary files of the atomic writes ([`sweep_temp_files`]: `.<name>.<32 hex>.tmp`
//!   and the old fixed names, at least [`TEMP_MIN_AGE`] old) in [`Settings::temp_dirs`];
//! - hard-link probes (`.bop-link-probe-*`, [`LINK_PROBE_PREFIX`], at least [`TEMP_MIN_AGE`] old)
//!   in [`Settings::link_probe_dirs`];
//! - cache folders unused for [`super::caches::UNUSED_DAYS`] and the CoreML orphans
//!   ([`super::caches`]);
//! - `.<name>.train-staging` folders no running training holds, in the models folder and every
//!   `install_dir` a run record names ([`Settings::staging_dirs`]);
//! - review exports' scratch folders (`.<name>.building-<32 hex>`, `.building-<32 hex>`,
//!   [`is_building_name`]) whose export crashed ([`super::busy`] says so): in
//!   `<training dir>/exports`, in its first-level folders, and wherever a record in
//!   [`EXPORTS_PENDING_DIR`] names one; one with no marker (an older version, or a finished export
//!   whose final move failed and was kept on purpose) is only listed ([`export_scratch`]);
//! - work folders of auto-label, import, overlap and dataset-check jobs that crashed
//!   ([`super::busy`]; a dataset check's fold folders on the dataset's volume too,
//!   [`NEAR_WORK_DIRS`], [`ELSEWHERE_FILE`], and the then empty `<sub>/` and [`NEAR_DATASET_DIR`]
//!   above them), an auto-label's finished proposals written to their frames first
//!   ([`crate::training::autolabel::job::recover_crashed`], the folder kept and the reason logged
//!   when that fails);
//! - download and model-export leftovers ([`crate::resources::manager::sweep_stale`],
//!   [`crate::resources::export::sweep_leftovers`]);
//! - then `review-queue.jsonl` compaction of the dataset and review folders
//!   ([`crate::collect::layout::compact_review`]).
//!
//! Every removal is logged with its size ([`super::remove_counted`]), and the sweep ends with one
//! summary line.
//!
//! Stopping: [`run`] takes the retention thread's cancel token and checks it between items (each
//! temporary file or probe, the two cache steps, each staging or scratch folder, each crashed work
//! folder, each compaction). An item is never cut in the middle, and one not reached stays for the
//! next sweep: a crashed job's busy claim is dropped unfinished, so its marker stays. Not
//! interruptible inside one item: a large folder's removal, or an auto-label recovery writing its
//! frames ([`crate::training::autolabel::job::recover_crashed`]).

use super::busy::{self, State};
use super::{Removed, caches, remove_counted};
use crate::config::Config;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Name prefix of the hard-link probe files ([`crate::training::preflight`], the dataset check's
/// placement): made in folders the sweep owns, so one a crash left is removed after
/// [`TEMP_MIN_AGE`].
pub const LINK_PROBE_PREFIX: &str = ".bop-link-probe-";

/// The folder of the training folder that holds one record per review export being built
/// ([`write_export_record`]): `<32 hex>.txt`, naming the export's scratch folder, written before
/// the folder is made and removed once it is dealt with. A sweep after a crash finds the scratch
/// folder of an export into a folder of the user's choice through it (one under
/// `<training dir>/exports` is found by its name too).
pub const EXPORTS_PENDING_DIR: &str = "exports-pending";

/// Time between storage sweeps.
pub const INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// The work folders of the training queue's jobs: `<training dir>/<sub>/<task id>`.
pub const WORK_DIRS: [(&str, WorkKind); 4] = [
    ("autolabel", WorkKind::Autolabel),
    ("imports", WorkKind::Import),
    ("overlaps", WorkKind::Overlaps),
    ("refine", WorkKind::Refine),
];

/// The work that may live on the dataset's volume instead, in [`NEAR_DATASET_DIR`]
/// ([`near_dataset_base`]): the dataset check's fold folders, so their exports hard-link the
/// frames ([`crate::training::refine::job`]).
pub const NEAR_WORK_DIRS: [(&str, WorkKind); 1] = [("refine", WorkKind::Refine)];

/// The folder beside the folder under review that holds [`NEAR_WORK_DIRS`]:
/// `<parent>/.blue-onyx-prism-work/<sub>/<task id>`, held busy like any work folder.
pub const NEAR_DATASET_DIR: &str = ".blue-onyx-prism-work";

/// The file in a work folder naming its part elsewhere (one absolute path): written before that
/// part is made, so a sweep that removes a crashed job's work folder removes that part too
/// ([`elsewhere_of`]), wherever the configured folders point by then.
pub const ELSEWHERE_FILE: &str = "elsewhere.txt";

/// `<parent of review_dir>/.blue-onyx-prism-work`; None when `review_dir` has no parent, or that
/// folder would lie inside one of `dataset_dirs` (the dataset's walkers and exports never see work
/// folders).
pub fn near_dataset_base(review_dir: &Path, dataset_dirs: &[PathBuf]) -> Option<PathBuf> {
    let abs = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let review = abs(review_dir);
    let parent = review.parent().filter(|p| !p.as_os_str().is_empty())?;
    let base = parent.join(NEAR_DATASET_DIR);
    let inside = |d: &Path| {
        let d = abs(d);
        base.starts_with(&d)
            || d.canonicalize()
                .ok()
                .zip(parent.canonicalize().ok())
                .is_some_and(|(d, p)| p.join(NEAR_DATASET_DIR).starts_with(d))
    };
    (!dataset_dirs.iter().any(|d| inside(d))).then_some(base)
}

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
    /// The config file, when known: its folder holds the config's and the benchmark files' temps.
    config_path: Option<PathBuf>,
}

impl Settings {
    pub fn for_config(config: &Config) -> Self {
        Self {
            config: Box::new(config.clone()),
            config_path: None,
        }
    }

    /// The same, also sweeping the temporary files beside the config file `path`.
    pub fn with_config_path(mut self, path: &Path) -> Self {
        self.config_path = Some(path.to_path_buf());
        self
    }

    /// The folders whose crash-left temporary files [`sweep_temp_files`] removes (not recursive):
    /// the config's folder, the data root, `auth/` and `auth/tls/`, `tools/hf/`, the ONNX Runtime
    /// folder, the training folder and its [`EXPORTS_PENDING_DIR`], the dataset and review folders
    /// with their `spots/` and `refine/`, the review folder's day folders when it is not the
    /// dataset folder, and `save_image_path`. The dataset folder's day folders are covered by the
    /// dataset prune ([`crate::retention`]).
    pub fn temp_dirs(&self) -> Vec<PathBuf> {
        let root = self.config.data_root();
        let auth = root.join(crate::private_dir::DIR_NAME);
        let training = self.training_dir();
        let mut v = vec![
            root.clone(),
            auth.join(crate::tls::DIR_NAME),
            auth,
            root.join(crate::resources::export::TOOLS_DIR).join("hf"),
            root.join("onnxruntime"),
            training.join(EXPORTS_PENDING_DIR),
            training,
        ];
        let dataset_dirs = self.dataset_dirs();
        for d in &dataset_dirs {
            v.push(d.join(crate::collect::labels::spots::SPOTS_DIR));
            v.push(d.join(crate::training::refine::report::REPORT_DIR));
        }
        // The review folder's day folders: label saves write there, and only the dataset folder's
        // are pruned. Listed from the known layout (scene/camera/day), never deeper.
        if let [dataset, review] = dataset_dirs.as_slice()
            && dataset != review
        {
            v.extend(crate::collect::layout::day_dirs(review));
        }
        if let Some(dir) = self.config_path.as_deref().and_then(Path::parent) {
            v.push(if dir.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                dir.to_path_buf()
            });
        }
        v.extend(dataset_dirs);
        if let Some(p) = self.config.save_image_path.as_deref() {
            v.push(crate::resolve_path(p));
        }
        v.sort();
        v.dedup();
        v
    }

    /// Fixed temporary names that writers used before [`crate::fsx::write_atomic`]: a crash of
    /// an older version may have left one, and nothing writes them any more.
    fn legacy_temp_names(&self) -> Vec<String> {
        let mut v = vec![
            "runs.json.tmp".to_string(),
            "tasks.json.tmp".to_string(),
            format!("{}.tmp", crate::backend::libs::ORT_ACTIVE_FILE),
        ];
        if let Some(name) = self
            .config_path
            .as_deref()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
        {
            v.push(format!("{name}.tmp"));
        }
        v
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

    /// Where training runs stage their model (`.<name>.train-staging`): the models folder and
    /// every distinct `install_dir` the run records name. `runs.json` is only read
    /// ([`crate::training::task::read_records`]: never moved aside or written, whatever it holds;
    /// the training queue deals with damage).
    pub fn staging_dirs(&self) -> Vec<PathBuf> {
        let runs: Vec<crate::training::service::RunRecord> =
            crate::training::task::read_records(&crate::training::runs_file(&self.config));
        let mut v = vec![self.models_dir()];
        v.extend(runs.iter().filter_map(|r| {
            r.request
                .install_dir
                .as_deref()
                .filter(|d| !d.as_os_str().is_empty())
                .map(|d| self.config.data_path(d))
        }));
        v.sort();
        v.dedup();
        v
    }

    /// Where hard-link probes ([`LINK_PROBE_PREFIX`]) are made: `<training dir>/runs` (a run's
    /// preflight), each [`Settings::near_dataset_bases`] (the dataset check's placement), and the
    /// training folder itself (where versions before the probes had a folder of their own made
    /// them).
    pub fn link_probe_dirs(&self) -> Vec<PathBuf> {
        let training = self.training_dir();
        let mut v = vec![
            training.join(crate::training::preflight::LINK_PROBE_DIR),
            training,
        ];
        v.extend(self.near_dataset_bases());
        v.sort();
        v.dedup();
        v
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

    /// Where [`NEAR_WORK_DIRS`] may be: [`near_dataset_base`] of each dataset folder.
    pub fn near_dataset_bases(&self) -> Vec<PathBuf> {
        let dirs = self.dataset_dirs();
        let mut out: Vec<PathBuf> = dirs
            .iter()
            .filter_map(|d| near_dataset_base(d, &dirs))
            .collect();
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
    /// The sweep ended early for a stop; what it did not reach waits for the next one.
    pub stopped: bool,
}

/// One full sweep (see the module docs), ending between two items once `cancel` is cancelled.
pub fn run(s: &Settings, cancel: &tokio_util::sync::CancellationToken) -> Report {
    run_until(s, &|| cancel.is_cancelled())
}

/// [`run`], asking `stop` between two items.
pub fn run_until(s: &Settings, stop: &dyn Fn() -> bool) -> Report {
    let now = SystemTime::now();
    let mut rep = Report::default();
    temp_files_until(s, now, &mut rep.removed, stop);
    if !stop() {
        link_probes_until(s, now, &mut rep.removed, stop);
    }
    if !stop() {
        sweep_caches_until(s, now, &mut rep.removed, stop);
    }
    for dir in s.staging_dirs() {
        if stop() {
            break;
        }
        staging_until(&dir, &mut rep.removed, stop);
    }
    if !stop() {
        export_scratch_until(s, now, &mut rep.removed, stop);
    }
    if !stop() {
        work_until(s, &mut rep, stop);
    }
    if !stop() {
        crate::resources::manager::sweep_stale(s.config(), &mut rep.removed, stop);
    }
    if !stop() {
        crate::resources::export::sweep_leftovers(s.config(), &mut rep.removed, stop);
    }
    for dir in s.dataset_dirs() {
        if stop() {
            break;
        }
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
    rep.stopped = stop();
    let freed = super::total(&rep.removed);
    if !rep.removed.is_empty() || !rep.kept.is_empty() || rep.stopped {
        tracing::info!(
            "storage sweep: {} item(s) removed, {} freed{}{}",
            rep.removed.len(),
            crate::resources::catalog::format_size(freed),
            if rep.kept.is_empty() {
                String::new()
            } else {
                format!("; kept: {}", rep.kept.join("; "))
            },
            if rep.stopped {
                "; stopped early because the service is stopping, the next sweep goes on"
            } else {
                ""
            }
        );
    }
    rep
}

/// How old a temporary file must be before [`sweep_temp_files`] removes it: an atomic write lives
/// for milliseconds, so one this old was left by a crash, a kill or a full disk.
pub const TEMP_MIN_AGE: Duration = Duration::from_secs(60 * 60);

/// Whether `name` in one of [`Settings::temp_dirs`] is a temporary file of an atomic write:
/// the [`crate::fsx::is_temp_name`] shape, or one of the `legacy` fixed names.
pub fn is_leftover_temp(name: &str, legacy: &[String]) -> bool {
    crate::fsx::is_temp_name(name) || legacy.iter().any(|l| l == name)
}

/// Remove the temporary files of atomic writes that a crash, a kill or a full disk left in
/// [`Settings::temp_dirs`]: regular files only, of the exact shapes [`is_leftover_temp`]
/// accepts, last modified at least [`TEMP_MIN_AGE`] before `now` (a write in progress is never
/// that old).
pub fn sweep_temp_files(s: &Settings, now: SystemTime, out: &mut Vec<Removed>) {
    temp_files_until(s, now, out, &|| false);
}

/// [`sweep_temp_files`], asking `stop` before each file.
fn temp_files_until(
    s: &Settings,
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    let legacy = s.legacy_temp_names();
    for dir in s.temp_dirs() {
        let done = old_files_in(
            &dir,
            now,
            &|name| is_leftover_temp(name, &legacy),
            "temporary file of a write a crash cut off",
            out,
            stop,
        );
        if !done {
            return;
        }
    }
}

/// Whether `name` is a hard-link probe's ([`LINK_PROBE_PREFIX`] and something after it).
pub fn is_link_probe(name: &str) -> bool {
    name.len() > LINK_PROBE_PREFIX.len() && name.starts_with(LINK_PROBE_PREFIX)
}

/// Remove the hard-link probes a crash left between their link and unlink in
/// [`Settings::link_probe_dirs`]: regular files named [`is_link_probe`], at least
/// [`TEMP_MIN_AGE`] old (a probe lives for microseconds). A probe is a second name of a frame:
/// removing it frees nothing but lets the frame's own deletion free its bytes.
pub fn sweep_link_probes(s: &Settings, now: SystemTime, out: &mut Vec<Removed>) {
    link_probes_until(s, now, out, &|| false);
}

/// [`sweep_link_probes`], asking `stop` before each file.
fn link_probes_until(
    s: &Settings,
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    for dir in s.link_probe_dirs() {
        let done = old_files_in(
            &dir,
            now,
            &is_link_probe,
            "hard-link probe a crash left (a second name of a frame)",
            out,
            stop,
        );
        if !done {
            return;
        }
    }
}

/// Whether `p` was last modified at least [`TEMP_MIN_AGE`] before `now` (False when unknown).
fn old_enough(p: &Path, now: SystemTime) -> bool {
    std::fs::symlink_metadata(p)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|m| now.duration_since(m).ok())
        .is_some_and(|age| age >= TEMP_MIN_AGE)
}

/// Remove the regular files directly in `dir` (never a folder or a link's target) whose name
/// `wanted` accepts and that are at least [`TEMP_MIN_AGE`] old. False when `stop` ended it.
fn old_files_in(
    dir: &Path,
    now: SystemTime,
    wanted: &dyn Fn(&str) -> bool,
    why: &str,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return true;
    };
    for e in rd.flatten() {
        if stop() {
            return false;
        }
        if !e.file_name().to_str().is_some_and(wanted) {
            continue;
        }
        let p = e.path();
        let is_file = std::fs::symlink_metadata(&p).is_ok_and(|m| m.is_file());
        if is_file && old_enough(&p, now) {
            remove_counted(&p, why, out);
        }
    }
    true
}

/// Unused cache folders and the CoreML orphans.
pub fn sweep_caches(s: &Settings, now: SystemTime, out: &mut Vec<Removed>) {
    sweep_caches_until(s, now, out, &|| false);
}

/// [`sweep_caches`], asking `stop` between its two steps.
fn sweep_caches_until(
    s: &Settings,
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    let Some(root) = s.cache_root() else { return };
    caches::sweep_unused(&root, now, out);
    let coreml = root.join("coreml");
    if coreml.is_dir() && !stop() {
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
    staging_until(models_dir, out, &|| false);
}

/// [`sweep_staging`], asking `stop` before each folder.
fn staging_until(models_dir: &Path, out: &mut Vec<Removed>, stop: &dyn Fn() -> bool) {
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
        if stop() {
            return;
        }
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

/// [`leftover_staging`] of every [`Settings::staging_dirs`] entry.
pub fn leftover_staging_all(s: &Settings) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = s
        .staging_dirs()
        .iter()
        .flat_map(|d| leftover_staging(d))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The 32 hex digits of a review export's scratch folder name (`.building-<id>` inside an
/// existing target, `.<target name>.building-<id>` beside a new one), None for any other name.
fn building_id(name: &str) -> Option<&str> {
    let (head, id) = name.rsplit_once(".building-")?;
    let hex = id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    let head_ok = head.is_empty() || (head.len() > 1 && head.starts_with('.'));
    (hex && head_ok).then_some(id)
}

/// Whether `name` is a review export's scratch folder
/// ([`crate::collect::labels::export`]'s staging): `.building-<32 hex>` or
/// `.<name>.building-<32 hex>`, nothing else.
pub fn is_building_name(name: &str) -> bool {
    building_id(name).is_some()
}

/// Write the record of the export scratch folder `scratch` into `pending_dir`
/// ([`EXPORTS_PENDING_DIR`]) before the folder is made; the export removes it once the folder is
/// dealt with. Its path, or why it could not be written (`scratch` must have an
/// [`is_building_name`] name).
pub fn write_export_record(pending_dir: &Path, scratch: &Path) -> std::io::Result<PathBuf> {
    let id = scratch
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(building_id)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is no export scratch folder", scratch.display()),
            )
        })?;
    let record = pending_dir.join(format!("{id}.txt"));
    let abs = std::path::absolute(scratch)?;
    let opts = crate::fsx::Options {
        create_parent: true,
        ..Default::default()
    };
    crate::fsx::write_atomic_with(&record, abs.to_string_lossy().as_bytes(), &opts)
        .map_err(std::io::Error::from)?;
    Ok(record)
}

/// One record of [`EXPORTS_PENDING_DIR`]: the file, and the scratch folder it names when it names
/// one that can only be that export's (an absolute path whose name is a scratch folder's with the
/// record's id; anything else in the file is ignored, so a sweep never removes a folder a stray
/// file names).
struct ExportRecord {
    file: PathBuf,
    scratch: Option<PathBuf>,
}

fn export_records(pending_dir: &Path) -> Vec<ExportRecord> {
    let Ok(rd) = std::fs::read_dir(pending_dir) else {
        return Vec::new();
    };
    let mut out: Vec<ExportRecord> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let id = name.strip_suffix(".txt").filter(|id| {
                id.len() == 32
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })?;
            let scratch = std::fs::read_to_string(e.path())
                .ok()
                .map(|t| PathBuf::from(t.trim()))
                .filter(|p| {
                    p.is_absolute()
                        && p.file_name().and_then(|n| n.to_str()).and_then(building_id) == Some(id)
                });
            Some(ExportRecord {
                file: e.path(),
                scratch,
            })
        })
        .collect();
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

/// The scratch folders (and the markers of scratch folders) directly in `dir`.
fn scratch_in(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let path = e.path();
        let Ok(kind) = e.file_type() else { continue };
        if kind.is_dir() && e.file_name().to_str().is_some_and(is_building_name) {
            out.push(path);
        } else if kind.is_file()
            && let Some(t) = busy::target_of(&path)
            && t.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_building_name)
        {
            out.push(t);
        }
    }
}

/// The scratch folders under `<training dir>/exports`: in it and in each of its first-level
/// folders (an export into an existing folder builds inside it), never deeper or through a link.
fn scratch_under_exports(training_dir: &Path) -> Vec<PathBuf> {
    let exports = training_dir.join("exports");
    let mut out = Vec::new();
    scratch_in(&exports, &mut out);
    if let Ok(rd) = std::fs::read_dir(&exports) {
        for e in rd.flatten() {
            let named_scratch = e.file_name().to_str().is_some_and(is_building_name);
            if e.file_type().is_ok_and(|t| t.is_dir()) && !named_scratch {
                scratch_in(&e.path(), &mut out);
            }
        }
    }
    out
}

/// A review export's scratch folder no running export holds ([`export_scratch`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ExportScratch {
    pub path: PathBuf,
    /// Its export crashed (the sweep removes it); else there is no marker: an older version's,
    /// or a finished export whose final move failed, kept on purpose (only listed).
    pub crashed: bool,
}

/// The scratch folders of review exports that exist and that no running export holds: under
/// `<training dir>/exports` and those the [`EXPORTS_PENDING_DIR`] records name.
pub fn export_scratch(s: &Settings) -> Vec<ExportScratch> {
    let training = s.training_dir();
    let mut paths = scratch_under_exports(&training);
    paths.extend(
        export_records(&training.join(EXPORTS_PENDING_DIR))
            .into_iter()
            .filter_map(|r| r.scratch),
    );
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .filter(|p| p.is_dir())
        .filter_map(|path| match busy::probe(&path) {
            State::Live => None,
            State::Ended => Some(ExportScratch {
                path,
                crashed: false,
            }),
            // Probed without keeping the claim: the sweep claims it again.
            State::Crashed(_) => Some(ExportScratch {
                path,
                crashed: true,
            }),
        })
        .collect()
}

/// Remove the scratch folders of review exports that crashed (and their markers), then the
/// [`EXPORTS_PENDING_DIR`] records nothing needs any more.
pub fn sweep_export_scratch(s: &Settings, now: SystemTime, out: &mut Vec<Removed>) {
    export_scratch_until(s, now, out, &|| false);
}

/// [`sweep_export_scratch`], asking `stop` before each folder and record.
fn export_scratch_until(
    s: &Settings,
    now: SystemTime,
    out: &mut Vec<Removed>,
    stop: &dyn Fn() -> bool,
) {
    let training = s.training_dir();
    let records = export_records(&training.join(EXPORTS_PENDING_DIR));
    let mut targets = scratch_under_exports(&training);
    targets.extend(records.iter().filter_map(|r| r.scratch.clone()));
    targets.sort();
    targets.dedup();
    let mut removed: Vec<PathBuf> = Vec::new();
    for t in targets {
        if stop() {
            return;
        }
        // Only a crash is provably orphan: a folder with no marker may be a finished export
        // kept on purpose, or an older version's (listed on the Storage card, never removed).
        if let State::Crashed(claim) = busy::probe(&t)
            && (!t.exists()
                || remove_counted(&t, "unfinished review export a crash or stop cut off", out))
        {
            claim.finish();
            removed.push(t);
        }
    }
    for r in records {
        if stop() {
            return;
        }
        // A record goes once its folder is dealt with: removed now, or ended (no marker) for at
        // least TEMP_MIN_AGE, which a record made just before its export's marker never is.
        let done = match &r.scratch {
            Some(t) if removed.contains(t) => true,
            Some(t) => matches!(busy::probe(t), State::Ended) && old_enough(&r.file, now),
            None => old_enough(&r.file, now),
        };
        if done
            && let Err(e) = std::fs::remove_file(&r.file)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::debug!(
                "storage: removing the export record {}: {e}",
                r.file.display()
            );
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
    kept_in(training_dir, &WORK_DIRS)
}

/// [`kept_work`] of the training folder and of [`Settings::near_dataset_bases`].
pub fn kept_all(s: &Settings) -> Vec<CrashedWork> {
    let mut out = kept_work(&s.training_dir());
    for base in s.near_dataset_bases() {
        out.extend(kept_in(&base, &NEAR_WORK_DIRS));
    }
    out
}

fn kept_in(root: &Path, dirs: &[(&str, WorkKind)]) -> Vec<CrashedWork> {
    let mut out = Vec::new();
    for &(sub, kind) in dirs {
        let Ok(rd) = std::fs::read_dir(root.join(sub)) else {
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
    crashed_in(training_dir, &WORK_DIRS)
}

/// [`crashed_work`] of the training folder, then of [`Settings::near_dataset_bases`] (the
/// dataset check's fold folders on the dataset's volume).
pub fn crashed_all(s: &Settings) -> Vec<CrashedWork> {
    let mut out = crashed_work(&s.training_dir());
    for base in s.near_dataset_bases() {
        out.extend(crashed_in(&base, &NEAR_WORK_DIRS));
    }
    out
}

fn crashed_in(root: &Path, dirs: &[(&str, WorkKind)]) -> Vec<CrashedWork> {
    let mut out = Vec::new();
    for &(sub, kind) in dirs {
        let Ok(rd) = std::fs::read_dir(root.join(sub)) else {
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
    work_until(s, rep, &|| false);
}

/// [`sweep_work`], asking `stop` before each folder (one not reached keeps its marker).
fn work_until(s: &Settings, rep: &mut Report, stop: &dyn Fn() -> bool) {
    for w in crashed_all(s) {
        if stop() {
            return;
        }
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
        // Its part on another volume first (the folder names it, [`ELSEWHERE_FILE`]).
        if let Some(far) = elsewhere_of(&w)
            && !sweep_elsewhere(&far, &why, rep)
        {
            continue;
        }
        if remove_counted(&w.dir, &why, &mut rep.removed) {
            claim.finish();
            remove_empty_near(&w.dir);
        }
    }
}

/// After a fold folder near the dataset (`<..>/.blue-onyx-prism-work/<sub>/<id>`) is removed, its
/// `<sub>/` and the [`NEAR_DATASET_DIR`] above it when they are empty now, as the check's own end
/// does: they sit in the user's folder. Nothing for any other path (a work folder in the training
/// folder); a folder that is not empty stays (`remove_dir` refuses it).
fn remove_empty_near(removed: &Path) {
    let Some(sub) = removed.parent() else { return };
    let Some(base) = sub.parent() else { return };
    if base.file_name().is_some_and(|n| n == NEAR_DATASET_DIR) {
        let _ = std::fs::remove_dir(sub);
        let _ = std::fs::remove_dir(base);
    }
}

/// The part elsewhere of the work folder `w` ([`ELSEWHERE_FILE`]), when it names one that can only
/// be this job's: `<..>/.blue-onyx-prism-work/<w's sub>/<w's name>` (an absolute path; anything
/// else in the file is ignored, so the sweep never removes a folder a stray file names).
pub fn elsewhere_of(w: &CrashedWork) -> Option<PathBuf> {
    let text = std::fs::read_to_string(w.dir.join(ELSEWHERE_FILE)).ok()?;
    let far = PathBuf::from(text.trim());
    let sub = NEAR_WORK_DIRS.iter().find(|(_, k)| *k == w.kind)?.0;
    let name = far.file_name()?;
    let in_sub = far.parent()?;
    let base = in_sub.parent()?;
    let ok = far.is_absolute()
        && Some(name) == w.dir.file_name()
        && in_sub.file_name().is_some_and(|n| n == sub)
        && base.file_name().is_some_and(|n| n == NEAR_DATASET_DIR);
    ok.then_some(far)
}

/// Remove `far`, the part elsewhere of a crashed job's work folder, unless a running job holds
/// it. Whether it is gone (the work folder may go too).
fn sweep_elsewhere(far: &Path, why: &str, rep: &mut Report) -> bool {
    match busy::probe(far) {
        State::Live => {
            rep.kept
                .push(format!("{}: a running job holds it", far.display()));
            false
        }
        State::Crashed(claim) => {
            let gone = !far.exists() || remove_counted(far, why, &mut rep.removed);
            if gone {
                claim.finish();
                remove_empty_near(far);
            }
            gone
        }
        State::Ended => {
            let gone = !far.exists() || remove_counted(far, why, &mut rep.removed);
            if gone {
                remove_empty_near(far);
            }
            gone
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
        Clear::Staging => {
            for dir in s.staging_dirs() {
                sweep_staging(&dir, &mut rep.removed);
            }
            sweep_export_scratch(s, SystemTime::now(), &mut rep.removed);
            crate::resources::export::sweep_leftovers(s.config(), &mut rep.removed, &|| false);
        }
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

    /// Set `p`'s modification time `age` ago.
    fn age(p: &Path, age: Duration) {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        f.set_modified(SystemTime::now() - age).unwrap();
    }

    #[test]
    fn crash_left_temp_files_go_once_old_and_nothing_else_does() {
        let root = tmp("temps");
        let s = settings(&root).with_config_path(&root.join("cfg").join("prism.json"));
        let auth = root.join("auth");
        let tls = auth.join("tls");
        let cfg = root.join("cfg");
        let training = s.training_dir();
        for d in [&auth, &tls, &cfg, &training] {
            std::fs::create_dir_all(d).unwrap();
        }
        let id = "0123456789abcdef0123456789abcdef";
        let hour = Duration::from_secs(3600);
        let old_temps = [
            auth.join(format!(".users.json.{id}.tmp")),
            tls.join(format!(".cert.pem.{id}.tmp")),
            cfg.join(format!(".prism.json.{id}.tmp")),
            cfg.join("prism.json.tmp"),
            training.join("runs.json.tmp"),
            root.join(format!(".children.json.{id}.tmp")),
        ];
        let kept = [
            // Too young: a write may be in progress.
            (
                auth.join(format!(".sessions.json.{}.tmp", "f".repeat(32))),
                Duration::ZERO,
            ),
            // Not the exact shape.
            (auth.join(".users.json.tmp"), 2 * hour),
            (
                cfg.join(format!(".prism.json.{}.tmp", "g".repeat(32))),
                2 * hour,
            ),
            (cfg.join("prism.json"), 2 * hour),
            // A folder of the right name is never a temp file.
            (training.join(format!(".x.{id}.tmp")), 2 * hour),
        ];
        for p in &old_temps {
            put(p, 3);
            age(p, 2 * hour);
        }
        for (p, a) in &kept {
            if p.ends_with(format!(".x.{id}.tmp")) {
                std::fs::create_dir_all(p).unwrap();
                continue;
            }
            put(p, 3);
            age(p, *a);
        }
        let mut out = Vec::new();
        sweep_temp_files(&s, SystemTime::now(), &mut out);
        for p in &old_temps {
            assert!(!p.exists(), "{} should be gone", p.display());
        }
        for (p, _) in &kept {
            assert!(p.exists(), "{} should stay", p.display());
        }
        assert_eq!(out.len(), old_temps.len());
        std::fs::remove_dir_all(&root).ok();
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
    fn the_near_dataset_folder_lies_beside_the_review_folder_never_inside_a_dataset() {
        let root = tmp("near-base");
        let (ds, review) = (root.join("dataset"), root.join("dataset/review"));
        assert_eq!(
            near_dataset_base(&ds, std::slice::from_ref(&ds)),
            Some(root.join(NEAR_DATASET_DIR))
        );
        // A review folder inside the dataset folder: its parent is the dataset folder.
        assert_eq!(
            near_dataset_base(&review, &[ds.clone(), review.clone()]),
            None
        );
        // A dataset folder that holds the parent (a drive root): none either.
        assert_eq!(near_dataset_base(&ds, &[root.clone(), ds.clone()]), None);
        let s = settings(&root);
        let dirs = s.dataset_dirs();
        assert_eq!(
            s.near_dataset_bases(),
            dirs.iter()
                .filter_map(|d| near_dataset_base(d, &dirs))
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_crashed_dataset_checks_fold_folder_near_the_dataset_goes_like_its_work_folder() {
        let root = tmp("near-work");
        let s = settings(&root);
        let t = s.training_dir();
        let base = s.near_dataset_bases()[0].clone();
        // A crashed check: its work folder names its fold folder near the dataset.
        let work = t.join("refine/20261118-1");
        let far = base.join("refine/20261118-1");
        put(&work.join("tighten.jsonl"), 10);
        put(&far.join("fold-1/export/images/a.jpg"), 100);
        std::fs::write(work.join(ELSEWHERE_FILE), far.display().to_string()).unwrap();
        std::fs::write(busy::marker_of(&work), b"1\n").unwrap();
        std::fs::write(busy::marker_of(&far), b"1\n").unwrap();
        // A crashed fold folder whose work folder is gone already: found by its own marker.
        let lone = base.join("refine/20261118-2");
        put(&lone.join("fold-2/model/m.onnx"), 100);
        std::fs::write(busy::marker_of(&lone), b"1\n").unwrap();
        // A running check's fold folder stays.
        let running = base.join("refine/20261118-3");
        put(&running.join("fold-1/data/x.jpg"), 100);
        let hold = busy::Busy::hold(&running);
        // A work folder naming a folder that is not a near fold folder: that one is never touched.
        let other = t.join("refine/20261118-4");
        let stray = root.join("photos/20261118-4");
        put(&stray.join("keep.jpg"), 100);
        put(&other.join("x"), 1);
        std::fs::write(other.join(ELSEWHERE_FILE), stray.display().to_string()).unwrap();
        std::fs::write(busy::marker_of(&other), b"1\n").unwrap();
        let crashed: Vec<_> = crashed_all(&s).into_iter().map(|w| w.dir).collect();
        assert!(crashed.contains(&work) && crashed.contains(&far) && crashed.contains(&lone));
        assert!(!crashed.contains(&running));
        let mut rep = Report::default();
        sweep_work(&s, &mut rep);
        for gone in [&work, &far, &lone, &other] {
            assert!(!gone.exists(), "{} stayed: {rep:?}", gone.display());
            assert!(!busy::marker_of(gone).exists(), "{}", gone.display());
        }
        assert!(running.join("fold-1/data/x.jpg").is_file(), "running");
        assert!(
            stray.join("keep.jpg").is_file(),
            "never a folder a stray file names"
        );
        drop(hold);
        // Ended in order but not removed: listed as kept, like a work folder.
        put(&base.join("refine/20261118-5/fold-1/a"), 1);
        assert!(
            kept_all(&s)
                .iter()
                .any(|w| w.dir == base.join("refine/20261118-5"))
        );
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

    /// Audit 2026-11 phase 1: a stop between two items ends the sweep there; what it did not
    /// reach (a crashed job's marker included) waits for the next sweep.
    #[test]
    fn a_stopped_sweep_ends_between_items_and_the_next_one_goes_on() {
        let root = tmp("stop");
        let s = settings(&root);
        let t = s.training_dir();
        let crashed: Vec<PathBuf> = (1..=3)
            .map(|i| t.join(format!("imports/20261122-{i}")))
            .collect();
        for d in &crashed {
            put(&d.join("frames/a.jpg"), 10);
            std::fs::write(busy::marker_of(d), b"1\n").unwrap();
        }
        let id = "0123456789abcdef0123456789abcdef";
        let temp = root.join(format!(".children.json.{id}.tmp"));
        put(&temp, 3);
        age(&temp, Duration::from_secs(2 * 3600));
        // Cancelled before anything: nothing goes.
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let rep = run(&s, &cancel);
        assert!(rep.stopped && rep.removed.is_empty(), "{rep:?}");
        assert!(temp.is_file());
        // Stop once the first work folder is gone: the other two stay, markers and all.
        let first_gone = || !crashed[0].exists();
        let rep = run_until(&s, &first_gone);
        assert!(rep.stopped);
        assert_eq!(
            rep.removed.len(),
            2,
            "the temp file and one work folder: {rep:?}"
        );
        assert!(!temp.exists());
        for d in &crashed[1..] {
            assert!(d.join("frames/a.jpg").is_file(), "{}", d.display());
            assert!(busy::marker_of(d).exists(), "{}", d.display());
        }
        let rep = run(&s, &tokio_util::sync::CancellationToken::new());
        assert!(!rep.stopped);
        assert_eq!(rep.removed.len(), 2);
        assert!(crashed.iter().all(|d| !d.exists()));
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
            // Benchmark results and the labels Your cameras was scored against.
            root.join("benchmark.json"),
            root.join(crate::benchmark::cameras::snapshot::FILE),
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
        let rep = run(&s, &tokio_util::sync::CancellationToken::new());
        assert!(!rep.stopped);
        assert_eq!(
            rep.removed.iter().map(|r| &r.path).collect::<Vec<_>>(),
            [&cache]
        );
        for p in &data {
            assert!(p.is_file(), "{} must stay", p.display());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 32 lowercase hex digits, as a scratch folder's id.
    fn hex(i: u128) -> String {
        format!("{i:032x}")
    }

    /// Leave a crashed job's marker beside `target`: there, unlocked.
    fn crash(target: &Path) {
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(busy::marker_of(target), b"1\n").unwrap();
    }

    #[test]
    fn building_names_are_matched_exactly() {
        let id = hex(0xabc_def);
        assert!(is_building_name(&format!(".building-{id}")));
        assert!(is_building_name(&format!(".20261123-1200.building-{id}")));
        assert!(is_building_name(&format!(".export-20261123.building-{id}")));
        assert!(!is_building_name(&format!("x.building-{id}")));
        assert!(!is_building_name(&format!("..building-{id}")));
        assert!(!is_building_name(&format!(".building-{}", &id[1..])));
        assert!(!is_building_name(&format!(".building-{id}0")));
        assert!(!is_building_name(&format!(
            ".building-{}",
            id.to_uppercase()
        )));
        assert!(!is_building_name(&format!(".building-{id}.busy")));
        assert!(!is_building_name("building"));
    }

    /// Audit 2026-11 phase 2 (RA-055, RA-060, RA-056a): a crashed review export's scratch folder
    /// under exports/ goes (beside a new target, inside an existing one); a running export's, one
    /// without a marker (an older version's, or a complete export kept on purpose) and anything of
    /// another shape or deeper stay.
    #[test]
    fn a_crashed_exports_scratch_goes_a_running_or_kept_one_stays() {
        let root = tmp("building");
        let s = settings(&root);
        let ex = s.training_dir().join("exports");
        let crashed_new = ex.join(format!(".20261123-1200.building-{}", hex(1)));
        let crashed_inside = ex.join("chosen").join(format!(".building-{}", hex(2)));
        let crashed_beside = ex
            .join("full")
            .join(format!(".export-20261123-1200.building-{}", hex(3)));
        let running = ex.join(format!(".run.building-{}", hex(4)));
        let unmarked = ex.join(format!(".old.building-{}", hex(5)));
        let lone = ex.join(format!(".gone.building-{}", hex(6)));
        let too_deep = ex.join("a/b").join(format!(".building-{}", hex(7)));
        let other_shapes = [
            ex.join(format!("x.building-{}", hex(8))),
            ex.join(format!(".x.building-{}", &hex(9)[1..])),
        ];
        let crashed = [&crashed_new, &crashed_inside, &crashed_beside];
        for d in crashed.iter().copied().chain([&unmarked, &too_deep]) {
            put(&d.join("export/images/a.jpg"), 10);
        }
        for d in crashed.iter().copied().chain([&lone, &too_deep]) {
            crash(d);
        }
        for d in &other_shapes {
            put(&d.join("a"), 1);
            crash(d);
        }
        let hold = busy::Busy::hold(&running);
        put(&running.join("export/images/a.jpg"), 10);
        put(&ex.join("e1/classes.txt"), 3);
        put(&ex.join("full/old-export/classes.txt"), 3);

        let mut listed = export_scratch(&s);
        listed.sort_by(|a, b| a.path.cmp(&b.path));
        let mut want: Vec<ExportScratch> = crashed
            .iter()
            .map(|p| ExportScratch {
                path: (*p).clone(),
                crashed: true,
            })
            .chain([ExportScratch {
                path: unmarked.clone(),
                crashed: false,
            }])
            .collect();
        want.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(listed, want);

        let mut out = Vec::new();
        sweep_export_scratch(&s, SystemTime::now(), &mut out);
        let mut gone: Vec<&PathBuf> = out.iter().map(|r| &r.path).collect();
        gone.sort();
        let mut want: Vec<&PathBuf> = crashed.to_vec();
        want.sort();
        assert_eq!(gone, want);
        for d in crashed.iter().copied().chain([&lone]) {
            assert!(
                !d.exists() && !busy::marker_of(d).exists(),
                "{}",
                d.display()
            );
        }
        assert!(running.join("export/images/a.jpg").is_file(), "running");
        assert!(busy::marker_of(&running).is_file());
        assert!(unmarked.join("export/images/a.jpg").is_file(), "no marker");
        assert!(too_deep.join("export/images/a.jpg").is_file(), "too deep");
        for d in &other_shapes {
            assert!(d.join("a").is_file(), "{}", d.display());
        }
        assert!(ex.join("e1/classes.txt").is_file());
        assert!(ex.join("full/old-export/classes.txt").is_file());
        assert!(ex.join("chosen").is_dir(), "the user's folder stays");
        // Clear 'staging' does the same.
        let again = ex.join(format!(".again.building-{}", hex(10)));
        put(&again.join("export/a.jpg"), 4);
        crash(&again);
        let rep = clear(&s, &Clear::Staging).unwrap();
        assert_eq!(rep.removed.len(), 1, "{rep:?}");
        assert!(!again.exists());
        drop(hold);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An export into a folder of the user's choice is found through its record: a crashed one
    /// goes with its record; a running one, one about to start and one kept on purpose stay; a
    /// record naming anything but its own scratch folder never removes what it names.
    #[test]
    fn an_export_elsewhere_is_found_through_its_record() {
        let root = tmp("export-record");
        let s = settings(&root);
        let pending = s.training_dir().join(EXPORTS_PENDING_DIR);
        let user = root.join("my exports");
        let hour = Duration::from_secs(3600);

        let crashed = user.join(format!(".mine.building-{}", hex(1)));
        let rec_crashed = write_export_record(&pending, &crashed).unwrap();
        assert_eq!(rec_crashed, pending.join(format!("{}.txt", hex(1))));
        put(&crashed.join("export/images/a.jpg"), 10);
        crash(&crashed);

        let running = user.join(format!(".building-{}", hex(2)));
        let rec_running = write_export_record(&pending, &running).unwrap();
        let hold = busy::Busy::hold(&running);
        put(&running.join("a.jpg"), 10);
        age(&rec_running, 2 * hour);

        // Recorded, not yet held or made.
        let starting = user.join(format!(".s.building-{}", hex(3)));
        let rec_starting = write_export_record(&pending, &starting).unwrap();

        // A complete export whose final move failed: kept, its record goes once old.
        let kept = user.join(format!(".k.building-{}", hex(4)));
        let rec_kept = write_export_record(&pending, &kept).unwrap();
        put(&kept.join("export/classes.txt"), 3);
        age(&rec_kept, 2 * hour);

        // A record naming a folder of the user's, crashed marker and all: never removed.
        let photos = root.join("photos");
        put(&photos.join("keep.jpg"), 3);
        crash(&photos);
        assert!(write_export_record(&pending, &photos).is_err());
        let rec_stray = pending.join(format!("{}.txt", hex(5)));
        std::fs::write(&rec_stray, photos.display().to_string()).unwrap();
        age(&rec_stray, 2 * hour);
        // A record whose id is not its folder's.
        let other = user.join(format!(".o.building-{}", hex(6)));
        put(&other.join("export/a.jpg"), 3);
        crash(&other);
        let rec_other = pending.join(format!("{}.txt", hex(7)));
        std::fs::write(&rec_other, other.display().to_string()).unwrap();

        let mut out = Vec::new();
        sweep_export_scratch(&s, SystemTime::now(), &mut out);
        assert_eq!(
            out.iter().map(|r| &r.path).collect::<Vec<_>>(),
            [&crashed],
            "{out:?}"
        );
        assert!(!crashed.exists() && !busy::marker_of(&crashed).exists());
        assert!(!rec_crashed.exists(), "its record goes with it");
        assert!(running.join("a.jpg").is_file() && rec_running.is_file());
        assert!(
            rec_starting.is_file(),
            "too young to tell from a starting export"
        );
        assert!(kept.join("export/classes.txt").is_file());
        assert!(!rec_kept.exists());
        assert!(photos.join("keep.jpg").is_file() && busy::marker_of(&photos).is_file());
        assert!(
            !rec_stray.exists(),
            "an old record naming no scratch folder goes"
        );
        assert!(
            other.join("export/a.jpg").is_file(),
            "not its record's folder"
        );
        assert!(rec_other.is_file(), "young");
        drop(hold);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (RA-071, RA-056c): a crashed run's staging in an install folder a run
    /// record names goes; a running one's stays; runs.json is only read.
    #[test]
    fn staging_in_an_install_folder_a_run_names_goes_too() {
        let root = tmp("install-dir");
        let s = settings(&root);
        let custom = root.join("srv-models");
        let runs = crate::training::runs_file(s.config());
        let recipe =
            crate::training::TrainRecipe::new(crate::resources::catalog::TrainFamily::Yolo26, "n")
                .unwrap();
        let record = |id: &str, install: Option<&str>| {
            serde_json::json!({
                "id": id, "name": "m", "created": "",
                "request": {"recipe": recipe, "datasets": [], "install_dir": install},
                "device": "cpu", "state": crate::training::service::RunState::Failed,
                "stage": "train", "step": 1, "steps": 8,
                "percent": 10, "detail": "", "run_dir": root.join(id)
            })
        };
        let list = serde_json::json!([
            record("r1", Some(&custom.display().to_string())),
            record("r2", Some("rel-models")),
            record("r3", None),
        ]);
        put(&runs, 1);
        std::fs::write(&runs, list.to_string()).unwrap();
        let mut want = vec![s.models_dir(), custom.clone(), root.join("rel-models")];
        want.sort();
        assert_eq!(s.staging_dirs(), want);

        let crashed = custom.join(".a.train-staging");
        let live = custom.join(".b.train-staging");
        let rel = root.join("rel-models/.c.train-staging");
        put(&crashed.join("m.onnx"), 10);
        crash(&crashed);
        put(&rel.join("m.onnx"), 10);
        let hold = busy::Busy::hold(&live);
        put(&live.join("m.onnx"), 10);
        put(&custom.join("keep.onnx"), 10);
        put(&custom.join(".notes/x"), 1);
        let mut want = vec![crashed.clone(), rel.clone()];
        want.sort();
        assert_eq!(leftover_staging_all(&s), want);
        let rep = run(&s, &tokio_util::sync::CancellationToken::new());
        assert!(
            !crashed.exists() && !busy::marker_of(&crashed).exists(),
            "{rep:?}"
        );
        assert!(!rel.exists());
        assert!(
            live.join("m.onnx").is_file(),
            "a running run's staging stays"
        );
        assert!(custom.join("keep.onnx").is_file() && custom.join(".notes/x").is_file());
        // Clear 'staging' covers it too.
        put(&crashed.join("m.onnx"), 10);
        crash(&crashed);
        let rep = clear(&s, &Clear::Staging).unwrap();
        assert_eq!(rep.removed.len(), 1, "{rep:?}");
        drop(hold);

        // A damaged runs.json: the models folder only, and the file is left as it is.
        std::fs::write(&runs, b"[{not json").unwrap();
        assert_eq!(s.staging_dirs(), [s.models_dir()]);
        assert_eq!(std::fs::read(&runs).unwrap(), b"[{not json");
        let aside = std::fs::read_dir(runs.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("corrupt"))
            .count();
        assert_eq!(aside, 0, "never moved aside");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (RA-045): once the last crashed fold folder near the dataset is
    /// removed, the empty `refine/` and `.blue-onyx-prism-work/` go too; a base holding anything
    /// else stays, and so does a training folder's emptied `refine/`.
    #[test]
    fn the_empty_near_folders_go_with_the_last_crashed_fold_folder() {
        let root = tmp("near-empty");
        let s = settings(&root);
        let t = s.training_dir();
        let base = s.near_dataset_bases()[0].clone();
        let work = t.join("refine/20261123-1");
        let far = base.join("refine/20261123-1");
        put(&work.join("tighten.jsonl"), 1);
        put(&far.join("fold-1/export/a.jpg"), 10);
        std::fs::write(work.join(ELSEWHERE_FILE), far.display().to_string()).unwrap();
        crash(&work);
        crash(&far);
        let lone = base.join("refine/20261123-2");
        put(&lone.join("fold-2/m.onnx"), 10);
        crash(&lone);
        let mut rep = Report::default();
        sweep_work(&s, &mut rep);
        assert!(!far.exists() && !lone.exists() && !work.exists(), "{rep:?}");
        assert!(
            !base.exists(),
            "the empty program folder in the user's folder goes"
        );
        assert!(t.join("refine").is_dir(), "the training folder's own stays");

        // Something else in the base: only the emptied refine/ goes.
        let lone = base.join("refine/20261123-3");
        put(&lone.join("fold-1/a.jpg"), 1);
        crash(&lone);
        put(&base.join("notes.txt"), 1);
        let mut rep = Report::default();
        sweep_work(&s, &mut rep);
        assert!(!base.join("refine").exists());
        assert!(base.join("notes.txt").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (RA-074): hard-link probes a crash left go once old, in the folders
    /// probes are made in; the frame they name keeps its bytes.
    #[test]
    fn hard_link_probes_a_crash_left_go_once_old() {
        let root = tmp("probes");
        let s = settings(&root);
        let t = s.training_dir();
        let runs = t.join("runs");
        let base = s.near_dataset_bases()[0].clone();
        let frame = s.dataset_dirs()[0].join("night/cam/2026-11-23/a.jpg");
        put(&frame, 5);
        let hour = Duration::from_secs(3600);
        let probe = |dir: &Path, tag: &str| dir.join(format!("{LINK_PROBE_PREFIX}{tag}"));
        let old = [
            probe(&runs, "11111111-1111-1111-1111-111111111111"),
            probe(&base, "2"),
            probe(&t, "3"),
        ];
        for p in &old {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::hard_link(&frame, p).unwrap();
            age(p, 2 * hour);
        }
        // Another frame: the links of one share its modification time.
        let other_frame = s.dataset_dirs()[0].join("night/cam/2026-11-23/b.jpg");
        put(&other_frame, 5);
        let young = probe(&runs, "young");
        std::fs::hard_link(&other_frame, &young).unwrap();
        let elsewhere = probe(&s.dataset_dirs()[0], "x");
        put(&elsewhere, 1);
        age(&elsewhere, 2 * hour);
        let bare = runs.join(LINK_PROBE_PREFIX);
        put(&bare, 1);
        age(&bare, 2 * hour);
        let folder = probe(&runs, "dir");
        std::fs::create_dir_all(&folder).unwrap();
        let mut out = Vec::new();
        sweep_link_probes(&s, SystemTime::now() + Duration::from_secs(1), &mut out);
        let mut gone: Vec<&PathBuf> = out.iter().map(|r| &r.path).collect();
        gone.sort();
        let mut want: Vec<&PathBuf> = old.iter().collect();
        want.sort();
        assert_eq!(gone, want);
        assert_eq!(
            std::fs::read(&frame).unwrap(),
            vec![1u8; 5],
            "the frame stays"
        );
        for p in [&young, &elsewhere, &bare, &folder] {
            assert!(p.exists(), "{} stays", p.display());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (RA-066, RA-065): temporary files in the dataset and review folders'
    /// `spots/` and `refine/`, in a separate review folder's day folders and in the export records'
    /// folder go once old; the dataset folder's day folders are the prune's, nothing deeper is
    /// looked at.
    #[test]
    fn temps_in_spots_refine_and_a_separate_review_folders_days_go() {
        let root = tmp("temps-review");
        let mut c = settings(&root).config().clone();
        c.collect.review_dir = Some(root.join("review"));
        let s = Settings::for_config(&c);
        let dirs = s.dataset_dirs();
        assert_eq!(dirs.len(), 2);
        let (dataset, review) = (&dirs[0], &dirs[1]);
        let id = hex(0xfeed);
        let tmp_in = |d: &Path, name: &str| d.join(format!(".{name}.{id}.tmp"));
        let old = [
            tmp_in(&dataset.join("spots"), "cam.json"),
            tmp_in(&review.join("spots"), "cam.json"),
            tmp_in(&review.join("refine"), "report.json"),
            tmp_in(&review.join("night/cam/2026-11-07"), "a.labels.json"),
            tmp_in(&s.training_dir().join(EXPORTS_PENDING_DIR), "x.txt"),
        ];
        let kept = [
            // The dataset folder's day folders are the dataset prune's.
            tmp_in(&dataset.join("night/cam/2026-11-07"), "a.labels.json"),
            // Deeper than the known layout.
            tmp_in(&review.join("night/cam/2026-11-07/sub"), "a.json"),
            tmp_in(&review.join("night/cam/not-a-day"), "a.json"),
        ];
        for p in old.iter().chain(&kept) {
            put(p, 2);
            age(p, Duration::from_secs(7200));
        }
        let mut out = Vec::new();
        sweep_temp_files(&s, SystemTime::now(), &mut out);
        for p in &old {
            assert!(!p.exists(), "{} should be gone", p.display());
        }
        for p in &kept {
            assert!(p.exists(), "{} should stay", p.display());
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
