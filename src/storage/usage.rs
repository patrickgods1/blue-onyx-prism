//! The Storage card's numbers (Config page, Resources tab): what is safe to clear and what is
//! the user's data, each with its size.
//!
//! - Safe to clear: each cache folder ([`super::caches`]; an OpenVINO or TensorRT one lists its
//!   compiled models as parts, [`caches::compiled_entries`], superseded ones named), the uv
//!   package cache a failed setup left (`tools/uv-cache`), leftover staging folders (training
//!   runs' in the models folder and in every install folder a run record names, the scratch
//!   folders of review exports that crashed, and model exports' staging and work folders,
//!   [`export_leftovers`]), the work folders of crashed jobs, old log files. Each has a Clear
//!   ([`super::sweep::clear`]).
//! - Your data: the dataset and review folders, each review export (with Remove), unfinished
//!   review exports no sweep removes (no busy marker: an older version's, or a finished export
//!   whose final move failed), training runs, the checkpoints failed runs keep for a retry, run
//!   folders no record names, models, work folders kept on purpose, the model export toolchain
//!   and the training environment under `tools/` (their Remove is on the Resources card and the
//!   Train tab), and ONNX Runtime installs this config does not load (another build's folder,
//!   `cuda-libs` beside a build other than CUDA, the old flat layout; the user may switch back,
//!   so they are never superseded). Sizes only: nothing here is deleted automatically, and only
//!   exports have a Remove on this card.
//!
//! Parts (`Item::parts`) are listed under their row and already counted in its size; so are
//! the checkpoints row (in Training runs) and a review folder inside the dataset folder, or the
//! reverse (noted "included in ..."): `data_bytes` counts each byte once.
//!
//! Sizes are bytes deleting a thing frees ([`crate::training::retry::disk_use`]: an export's frames
//! are hard links to the dataset's, so they count where they are not shared). Measuring walks
//! every file, so [`measure`] runs off the request thread and its result is kept for [`FRESH`].

use super::sweep::{self, Settings};
use super::{caches, size_of};
use crate::resources::catalog::format_size;
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long a measurement is served before it is taken again.
pub const FRESH: Duration = Duration::from_secs(60);

/// One row of the card.
#[derive(Debug, Clone, Serialize)]
pub struct Item {
    /// What Clear / Remove name (`cache:coreml`, `staging`, `work`, `logs`, `export:<name>`), or
    /// a plain key for rows without an action.
    pub id: String,
    pub title: String,
    pub path: String,
    pub bytes: u64,
    pub size: String,
    pub note: String,
    /// `clear`, `remove` or None.
    pub action: Option<&'static str>,
    /// What the row holds, listed under it (report only, already counted in [`Item::bytes`]): a
    /// cache folder's compiled models, a tools row's folders, the run folders no record names.
    pub parts: Vec<Part>,
}

/// One entry listed under a row ([`Item::parts`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Part {
    pub title: String,
    pub path: String,
    pub bytes: u64,
    pub size: String,
    pub note: String,
}

impl Part {
    fn new(title: impl Into<String>, path: &Path, bytes: u64) -> Self {
        Self {
            title: title.into(),
            path: path.display().to_string(),
            bytes,
            size: format_size(bytes),
            note: String::new(),
        }
    }
}

impl Item {
    fn new(id: impl Into<String>, title: impl Into<String>, path: &Path, bytes: u64) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            path: path.display().to_string(),
            bytes,
            size: format_size(bytes),
            note: String::new(),
            action: None,
            parts: Vec::new(),
        }
    }

    fn note(mut self, n: impl Into<String>) -> Self {
        self.note = n.into();
        self
    }

    fn action(mut self, a: &'static str) -> Self {
        self.action = Some(a);
        self
    }

    fn parts(mut self, p: Vec<Part>) -> Self {
        self.parts = p;
        self
    }
}

/// The card.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub safe: Vec<Item>,
    pub data: Vec<Item>,
    pub safe_bytes: u64,
    pub data_bytes: u64,
    /// Free space under the data root.
    pub free_bytes: Option<u64>,
    /// Unix seconds of the measurement.
    pub measured_at: u64,
    /// Seconds the measurement took.
    pub seconds: f64,
}

/// "today", "yesterday", "12 days ago".
fn days_ago(secs: u64, now: SystemTime) -> String {
    let at = UNIX_EPOCH + Duration::from_secs(secs);
    let days = now.duration_since(at).unwrap_or_default().as_secs() / 86_400;
    match days {
        0 => "today".into(),
        1 => "yesterday".into(),
        d => format!("{d} days ago"),
    }
}

/// A cache folder's title: `openvino-gpu-f16` -> "OpenVINO cache (GPU, f16)".
pub fn cache_title(name: &str) -> String {
    match name {
        "coreml" => "CoreML cache (compiled models and graph rewrites)".into(),
        "tensorrt" => "TensorRT engine cache".into(),
        n => match n.strip_prefix("openvino-").and_then(|r| r.split_once('-')) {
            Some((dev, prec)) => format!("OpenVINO cache ({}, {prec})", dev.to_uppercase()),
            None => "OpenVINO cache".into(),
        },
    }
}

/// Training run folders and failed runs' checkpoints, from the run records.
fn run_items(runs: &[RunRecord], out: &mut Vec<Item>, training_dir: &Path) {
    let mut total = 0u64;
    let mut counted = 0usize;
    let mut checkpoints = 0u64;
    let mut failed = 0usize;
    for r in runs.iter().filter(|r| !r.state.is_busy()) {
        let all = crate::training::retry::disk_use(&r.run_dir);
        if all == 0 {
            continue;
        }
        total += all;
        counted += 1;
        if r.state == crate::training::service::RunState::Failed {
            let w = crate::training::retry::disk_use(&crate::training::retry::work_dir(&r.run_dir));
            if w > 0 {
                checkpoints += w;
                failed += 1;
            }
        }
    }
    out.push(
        Item::new("runs", "Training runs", training_dir, total).note(format!(
            "{counted} finished run folder(s) (datasets, logs, checkpoints); remove a run on the \
             Train area's Models tab"
        )),
    );
    if failed > 0 {
        out.push(
            Item::new(
                "checkpoints",
                "Checkpoints of failed runs",
                training_dir,
                checkpoints,
            )
            .note(format!(
                "{failed} failed run(s) keep their work folder so Retry export can make the model \
                 without training again; never deleted automatically (included in Training runs)"
            )),
        );
    }
}

/// Every review export under `<training dir>/exports`: folders with `classes.txt` or `data.yaml`.
pub fn exports(training_dir: &Path) -> Vec<PathBuf> {
    let root = training_dir.join("exports");
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .filter(|p| p.join("classes.txt").is_file() || p.join("data.yaml").is_file())
        .collect();
    v.sort();
    v
}

/// The "Leftover staging" row (Clear `staging`): training runs' staging folders, the scratch
/// folders of review exports that crashed, and model exports' leftovers ([`export_leftovers`]).
/// None when there is none of them.
fn staging_item(
    staging: &[PathBuf],
    crashed_exports: &[sweep::ExportScratch],
    model_exports: &[PathBuf],
    models: &Path,
    training: &Path,
) -> Option<Item> {
    if staging.is_empty() && crashed_exports.is_empty() && model_exports.is_empty() {
        return None;
    }
    let bytes: u64 = staging
        .iter()
        .chain(model_exports)
        .map(|p| size_of(p))
        .chain(crashed_exports.iter().map(|x| size_of(&x.path)))
        .sum();
    let mut notes = Vec::new();
    if !staging.is_empty() {
        let elsewhere: Vec<String> = staging
            .iter()
            .filter(|p| p.parent() != Some(models))
            .map(|p| p.display().to_string())
            .collect();
        let mut n = format!(
            "{} folder(s) of training runs that stopped before installing their model",
            staging.len()
        );
        if !elsewhere.is_empty() {
            n.push_str(&format!(
                " (outside the models folder: {})",
                elsewhere.join(", ")
            ));
        }
        notes.push(n);
    }
    if !crashed_exports.is_empty() {
        notes.push(format!(
            "{} unfinished review export(s) a crash or stop cut off: {}",
            crashed_exports.len(),
            crashed_exports
                .iter()
                .take(5)
                .map(|x| x.path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !model_exports.is_empty() {
        notes.push(format!(
            "{} staging or work folder(s) of model exports a crash or kill interrupted: {}",
            model_exports.len(),
            model_exports
                .iter()
                .take(5)
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let kinds = [
        !staging.is_empty(),
        !crashed_exports.is_empty(),
        !model_exports.is_empty(),
    ];
    let (title, path) = match kinds {
        [true, false, false] => ("Leftover training staging", models),
        [false, true, false] => ("Unfinished review exports", training),
        [false, false, true] => (
            "Leftover model export folders",
            // The models folder for a staging folder, else `tools/export-work`.
            model_exports
                .iter()
                .find(|p| p.parent() == Some(models))
                .or(model_exports.first())
                .and_then(|p| p.parent())
                .unwrap_or(models),
        ),
        [_, false, _] => ("Leftover staging (training runs, model exports)", models),
        [false, _, _] => ("Leftover staging (review exports, model exports)", training),
        [_, _, false] => ("Leftover staging (training runs, review exports)", training),
        _ => (
            "Leftover staging (training runs, review exports, model exports)",
            training,
        ),
    };
    Some(
        Item::new("staging", title, path, bytes)
            .note(notes.join("; "))
            .action("clear"),
    )
}

/// The report-only row of review exports' scratch folders no sweep removes (no busy marker: an
/// older version's, or a complete export whose final move failed), except those inside a folder
/// `listed` as an export (counted in its row). None when there are none.
fn unfinished_item(
    unfinished: &[sweep::ExportScratch],
    listed: &[PathBuf],
    training: &Path,
) -> Option<Item> {
    let shown: Vec<&PathBuf> = unfinished
        .iter()
        .map(|x| &x.path)
        .filter(|p| !listed.iter().any(|e| p.parent() == Some(e.as_path())))
        .collect();
    if shown.is_empty() {
        return None;
    }
    let bytes: u64 = shown.iter().map(|p| size_of(p)).sum();
    Some(
        Item::new(
            "unfinished-exports",
            "Unfinished review exports (kept)",
            training,
            bytes,
        )
        .note(format!(
            "{} hidden folder(s) of exports that never got into place: left by an older version, \
             or complete but not moved (the export said where); never deleted automatically. Move \
             one into place or delete it by hand: their frames are hard links to the dataset's, so \
             they keep the bytes of frames the dataset has since deleted. {}",
            shown.len(),
            shown
                .iter()
                .take(5)
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    )
}

/// Whether the cache folder `name` holds per-model compiled entries
/// ([`caches::compiled_entries`]): the OpenVINO and TensorRT caches.
fn has_compiled_entries(name: &str) -> bool {
    name.starts_with("openvino-") || name == "tensorrt"
}

/// The parts of an OpenVINO or TensorRT cache row: one per compiled model, superseded ones
/// first (the next sweep removes them), then by size.
fn compiled_parts(entries: &[caches::CompiledUse]) -> Vec<Part> {
    let mut sorted: Vec<&caches::CompiledUse> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        b.superseded
            .cmp(&a.superseded)
            .then(b.bytes.cmp(&a.bytes))
            .then_with(|| a.path.cmp(&b.path))
    });
    sorted
        .into_iter()
        .map(|e| {
            let title = e.model.as_deref().and_then(Path::file_name).map_or_else(
                || "compiled model (its model file is not recorded)".to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let mut part = Part::new(title, &e.path, e.bytes);
            part.note = if e.superseded {
                "superseded: its model file is gone or was compiled again; the next sweep \
                 removes it"
                    .into()
            } else {
                e.model
                    .as_ref()
                    .map(|m| format!("compiled from {}", m.display()))
                    .unwrap_or_default()
            };
            part
        })
        .collect()
}

/// "; 2 superseded compiled model(s) (1.2 GB) go at the next sweep", or nothing.
fn superseded_note(entries: &[caches::CompiledUse]) -> String {
    let gone: Vec<&caches::CompiledUse> = entries.iter().filter(|e| e.superseded).collect();
    if gone.is_empty() {
        return String::new();
    }
    format!(
        "; {} superseded compiled model(s) ({}) go at the next sweep",
        gone.len(),
        format_size(gone.iter().map(|e| e.bytes).sum())
    )
}

/// `.<stem>.export-staging` (non-empty stem): a model export's staging folder in the models
/// folder (the shape [`crate::resources::export::sweep_leftovers`] matches).
fn is_export_staging(name: &str) -> bool {
    crate::resources::export::export_staging_stem(name).is_some()
}

/// What Clear `staging` (and the daily sweep) removes of model exports
/// ([`crate::resources::export::sweep_leftovers`]): the staging folders of `models` whose export
/// crashed (busy marker unlocked), or that have no marker while the export lock is free, and
/// the folders of `tools/export-work` while the export lock is free. Looked at without waiting:
/// the export lock is tried once, only when something needs it.
fn export_leftovers(paths: &crate::resources::export::Paths) -> Vec<PathBuf> {
    let dirs_in = |dir: &Path, keep: &dyn Fn(&str) -> bool| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter(|e| e.file_name().to_str().is_some_and(keep))
            .map(|e| e.path())
            .collect();
        v.sort();
        v
    };
    let mut out = Vec::new();
    let mut unmarked = Vec::new();
    for p in dirs_in(&paths.models, &is_export_staging) {
        match super::busy::probe(&p) {
            super::busy::State::Live => {}
            super::busy::State::Ended => unmarked.push(p),
            // The claim is dropped unfinished: the marker stays for the sweep.
            super::busy::State::Crashed(_) => out.push(p),
        }
    }
    let work = dirs_in(&paths.work, &|_| true);
    if (!unmarked.is_empty() || !work.is_empty())
        && matches!(sweep::try_export_lock(paths, Duration::ZERO), Ok(Some(_)))
    {
        out.extend(unmarked);
        out.extend(work);
    }
    out
}

/// The "your data" rows of `tools/`: the model export toolchain and the training environment,
/// with their folders as parts. Their Remove is elsewhere (the Resources card's model exports,
/// the Train tab); the uv cache is a "safe to clear" row of its own and the export work folders
/// are counted with the leftover staging.
fn tools_items(config: &crate::config::Config, out: &mut Vec<Item>) {
    let train = crate::resources::export::train::TrainPaths::for_config(config);
    let e = crate::resources::export::Paths::for_config(config);
    let mut export_parts: Vec<(&str, PathBuf)> = vec![
        ("uv", e.uv_dir.clone()),
        ("Python", e.python_installs.clone()),
        ("Export environment (Python packages)", e.env.clone()),
        (
            "Export weights (also the YOLO training bases)",
            e.weights.clone(),
        ),
        ("Export settings and caches", e.yolo_config.clone()),
        ("DEIMv2 source", e.deimv2_src.clone()),
        ("Visual C++ runtime", e.vc_runtime.clone()),
    ];
    export_parts.extend(
        e.legacy_dirs()
            .into_iter()
            .map(|p| ("Files of an earlier YOLO26 export", p)),
    );
    let train_parts: Vec<(&str, PathBuf)> = vec![
        (
            "Training environment (Python packages)",
            train.export.env.clone(),
        ),
        (
            "Base weights and auto-label checkpoints",
            train.weights.clone(),
        ),
        ("Training settings and caches", train.config.clone()),
        ("Training scripts", train.scripts.clone()),
    ];
    let row = |id: &str, title: &str, parts: Vec<(&str, PathBuf)>, note: &str| -> Option<Item> {
        let parts: Vec<Part> = parts
            .into_iter()
            .filter(|(_, p)| std::fs::symlink_metadata(p).is_ok())
            .map(|(t, p)| Part::new(t, &p, size_of(&p)))
            .collect();
        if parts.is_empty() {
            return None;
        }
        let bytes = parts.iter().map(|p| p.bytes).sum();
        Some(
            Item::new(id, title, &e.tools, bytes)
                .note(note)
                .parts(parts),
        )
    };
    out.extend(row(
        "tools-export",
        "Model export toolchain",
        export_parts,
        "downloaded again when an export needs it; Remove export toolchain on the Resources \
         card (model exports group) deletes it, refused while an export, a training setup or a \
         training runs; the training environment then rebuilds on its next use",
    ));
    out.extend(row(
        "tools-train",
        "Training environment",
        train_parts,
        "built again when a training or an auto-label needs it; Remove training environment on \
         the Train area's Train tab deletes it (the auto-label checkpoints too), refused while a \
         job runs",
    ));
}

/// The report-only row of folders under `training` with a run's files that no record of `runs`
/// names ([`crate::training::service::unlisted_run_dirs`], also listed by `train list`).
fn unlisted_runs_item(training: &Path, runs: &[RunRecord]) -> Option<Item> {
    let dirs = crate::training::service::unlisted_run_dirs(training, runs);
    if dirs.is_empty() {
        return None;
    }
    let parts: Vec<Part> = dirs
        .iter()
        .map(|d| {
            let name = d
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            Part::new(name, d, crate::training::retry::disk_use(d))
        })
        .collect();
    Some(
        Item::new(
            "unlisted-runs",
            "Run folders no record names",
            training,
            parts.iter().map(|p| p.bytes).sum(),
        )
        .note(format!(
            "{} folder(s) with a run's log or history that runs.json does not list (an earlier \
             version lost their record); the Train area does not show them and nothing deletes \
             them: delete them by hand if you do not need them (`blue-onyx-prism train list` \
             lists them too)",
            dirs.len()
        ))
        .parts(parts),
    )
}

/// Whether folder `inner` is `outer` or lies inside it (as written made absolute, or once
/// resolved: a symlinked or differently spelled path).
fn dir_within(inner: &Path, outer: &Path) -> bool {
    let abs = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    abs(inner).starts_with(abs(outer))
        || inner
            .canonicalize()
            .ok()
            .zip(outer.canonicalize().ok())
            .is_some_and(|(i, o)| i.starts_with(o))
}

/// Whether `a` and `b` name the same folder (as written, or once resolved).
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b
        || a.canonicalize()
            .ok()
            .zip(b.canonicalize().ok())
            .is_some_and(|(x, y)| x == y)
}

/// The report-only rows of ONNX Runtime installs under `ort_root` that this config does not load
/// (`library`: what the lookup finds, [`crate::backend::libs::find_onnxruntime_with`]): a flavor
/// folder other than the loaded one, `cuda-libs` while a build other than CUDA loads, and the
/// files of the old flat layout. Never deleted automatically (the user may switch back); the
/// flavors and `cuda-libs` have a Remove on the Resources card.
fn runtime_items(ort_root: &Path, library: Option<&Path>) -> Vec<Item> {
    use crate::backend::libs;
    let in_use = library.and_then(Path::parent);
    let loads = in_use.map_or_else(
        || "no ONNX Runtime library is found to load".to_string(),
        |d| format!("it loads the one in {}", d.display()),
    );
    let remove = "kept so switching back needs no download; Remove it on the Resources card \
                  (runtimes), refused while a running process has it loaded";
    let mut out = Vec::new();
    for flavor in libs::ORT_FLAVOR_DIRS {
        let dir = ort_root.join(flavor);
        if !dir.is_dir() || in_use.is_some_and(|u| same_dir(u, &dir)) {
            continue;
        }
        out.push(
            Item::new(
                format!("runtime:onnxruntime-{flavor}"),
                format!("ONNX Runtime ({flavor} build), not loaded by this config"),
                &dir,
                size_of(&dir),
            )
            .note(format!("{loads}; {remove}")),
        );
    }
    // The CUDA libraries serve only the CUDA build: unused while a build known to be another
    // loads (one of our other flavor folders, or a folder whose flavor.txt names another).
    let cuda_libs = ort_root.join(libs::CUDA_LIBS_DIR_NAME);
    let other_build = in_use.is_some_and(|u| {
        libs::ORT_FLAVOR_DIRS
            .iter()
            .filter(|f| **f != "cuda")
            .any(|f| same_dir(u, &ort_root.join(f)))
            || libs::read_ort_flavor(u).is_some_and(|f| f != "cuda")
    });
    if cuda_libs.is_dir() && other_build {
        out.push(
            Item::new(
                "runtime:cuda-libs",
                "NVIDIA CUDA libraries, not used by this config",
                &cuda_libs,
                size_of(&cuda_libs),
            )
            .note(format!(
                "only the CUDA build of ONNX Runtime uses them, and {loads}; {remove}"
            )),
        );
    }
    // The flat layout of versions before the flavor folders: the libraries in the root itself.
    let flat: Vec<PathBuf> = std::fs::read_dir(ort_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| crate::resources::extract::is_ort_file(n, cfg!(windows)))
        })
        .map(|e| e.path())
        .collect();
    let flat_lib = libs::find_ort_library_in(ort_root).is_some();
    if flat_lib && !in_use.is_some_and(|u| same_dir(u, ort_root)) {
        let mut names: Vec<String> = flat
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect();
        names.sort();
        out.push(
            Item::new(
                "runtime:onnxruntime-flat",
                "ONNX Runtime files of the old flat layout, not loaded by this config",
                ort_root,
                flat.iter().map(|p| size_of(p)).sum(),
            )
            .note(format!(
                "left by a version before the per-build folders; {loads}. No Remove: delete \
                 these files in that folder by hand (never its subfolders): {}",
                names.join(", ")
            )),
        );
    }
    out
}

/// Measure the card for `s`, with the training runs `runs` (their folders are skipped while
/// busy).
pub fn measure(s: &Settings, runs: &[RunRecord]) -> Report {
    let started = Instant::now();
    let now = SystemTime::now();
    let mut safe = Vec::new();
    if let Some(root) = s.cache_root() {
        for c in caches::infos(&root) {
            let mut note = match c.last_used {
                Some(t) => format!("last used {}", days_ago(t, now)),
                None => "not used since this version started recording".into(),
            };
            if c.in_use {
                note.push_str("; in use: loaded models keep running, the next load compiles again");
            } else {
                note.push_str(&format!(
                    "; deleted automatically after {} days unused",
                    caches::UNUSED_DAYS
                ));
            }
            let entries = if has_compiled_entries(&c.name) {
                caches::compiled_entries(&c.path)
            } else {
                Vec::new()
            };
            note.push_str(&superseded_note(&entries));
            safe.push(
                Item::new(
                    format!("cache:{}", c.name),
                    cache_title(&c.name),
                    &c.path,
                    c.bytes,
                )
                .note(note)
                .action("clear")
                .parts(compiled_parts(&entries)),
            );
        }
    }
    let config = s.config();
    let export_paths = crate::resources::export::Paths::for_config(config);
    if std::fs::symlink_metadata(&export_paths.uv_cache).is_ok_and(|m| m.is_dir()) {
        safe.push(
            Item::new(
                "uv-cache",
                "uv package cache",
                &export_paths.uv_cache,
                size_of(&export_paths.uv_cache),
            )
            // "a day": sweep::UV_CACHE_MIN_AGE.
            .note(
                "Python packages a failed, timed-out or cancelled export or training setup \
                 downloaded (a finished setup deletes it); deleted automatically once a day old \
                 while no export or training setup runs, and cleared only then",
            )
            .action("clear"),
        );
    }
    let models = s.models_dir();
    let training = s.training_dir();
    let staging = sweep::leftover_staging_all(s);
    let model_exports = export_leftovers(&export_paths);
    // Only what lies in the models folder is inside the Models row below.
    let staging_bytes: u64 = staging
        .iter()
        .chain(&model_exports)
        .filter(|p| p.parent() == Some(models.as_path()))
        .map(|p| size_of(p))
        .sum();
    let (crashed_exports, unfinished_exports): (Vec<_>, Vec<_>) = sweep::export_scratch(s)
        .into_iter()
        .partition(|x| x.crashed);
    if let Some(item) = staging_item(
        &staging,
        &crashed_exports,
        &model_exports,
        &models,
        &training,
    ) {
        safe.push(item);
    }
    let crashed = sweep::crashed_all(s);
    if !crashed.is_empty() {
        let bytes: u64 = crashed.iter().map(|w| size_of(&w.dir)).sum();
        let autolabel = crashed
            .iter()
            .filter(|w| w.kind == sweep::WorkKind::Autolabel)
            .count();
        let mut note = format!("{} folder(s) of jobs that crashed", crashed.len());
        if autolabel > 0 {
            note.push_str(&format!(
                "; {autolabel} auto-label folder(s): their finished frames are written to the \
                 label files first"
            ));
        }
        safe.push(
            Item::new("work", "Work folders of crashed jobs", &training, bytes)
                .note(note)
                .action("clear"),
        );
    }
    if let Some(dir) = s.log_dir() {
        let old = sweep::old_logs(&dir, now);
        if !old.is_empty() {
            let bytes: u64 = old.iter().map(|p| size_of(p)).sum();
            safe.push(
                Item::new("logs", "Old log files", &dir, bytes)
                    .note(format!(
                        "{} file(s) before today; the newest file is kept",
                        old.len()
                    ))
                    .action("clear"),
            );
        }
    }

    let mut data = Vec::new();
    // Rows already counted in another row: left out of `data_bytes`.
    let mut counted_elsewhere: Vec<&str> = vec!["checkpoints"];
    let dataset_dirs = s.dataset_dirs();
    let dataset_title = |i: usize| {
        if i == 0 {
            ("dataset", "Dataset (collected and imported frames, labels)")
        } else {
            ("review-folder", "Folder under review")
        }
    };
    for (i, dir) in dataset_dirs.iter().enumerate() {
        let (id, title) = dataset_title(i);
        let mut item = Item::new(id, title, dir, crate::training::retry::disk_use(dir));
        // A review folder inside the dataset folder (or the reverse) is in both walks (QA-079).
        if let Some(outer) = dataset_dirs
            .iter()
            .enumerate()
            .position(|(j, o)| j != i && dir_within(dir, o) && (j < i || !dir_within(o, dir)))
        {
            item = item.note(format!("included in {}", dataset_title(outer).1));
            counted_elsewhere.push(id);
        }
        data.push(item);
    }
    let listed = exports(&training);
    if let Some(item) = unfinished_item(&unfinished_exports, &listed, &training) {
        data.push(item);
    }
    for e in listed {
        let name = e
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        data.push(
            Item::new(
                format!("export:{name}"),
                // A frame-stamp name (a time) is shown as one, in the configured zone.
                format!(
                    "Export {}",
                    crate::clock::parse_frame_stamp(&name)
                        .map_or_else(|| name.clone(), crate::clock::display)
                ),
                &e,
                crate::training::retry::disk_use(&e),
            )
            .note("frames shared with the dataset as hard links count where they are not shared")
            .action("remove"),
        );
    }
    run_items(runs, &mut data, &training);
    data.extend(unlisted_runs_item(&training, runs));
    data.push(Item::new(
        "models",
        "Models",
        &models,
        crate::training::retry::disk_use(&models).saturating_sub(staging_bytes),
    ));
    tools_items(config, &mut data);
    let ort_root = config.data_root().join(crate::backend::libs::ORT_DIR_NAME);
    let library =
        crate::backend::libs::find_onnxruntime_with(config.onnxruntime_dir.as_deref(), &ort_root)
            .library;
    data.extend(runtime_items(&ort_root, library.as_deref()));
    let kept = sweep::kept_all(s);
    if !kept.is_empty() {
        let bytes: u64 = kept.iter().map(|w| size_of(&w.dir)).sum();
        data.push(
            Item::new("kept-work", "Kept job work folders", &training, bytes).note(format!(
                "{} folder(s) kept on purpose (an import's keep work, auto-label proposals that \
                 could not be saved) or left by an older version: {}",
                kept.len(),
                kept.iter()
                    .take(5)
                    .map(|w| w.dir.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        );
    }
    Report {
        safe_bytes: safe.iter().map(|i| i.bytes).sum(),
        data_bytes: data
            .iter()
            .filter(|i| !counted_elsewhere.contains(&i.id.as_str()))
            .map(|i| i.bytes)
            .sum(),
        safe,
        data,
        free_bytes: crate::retention::free_bytes(&s.config().data_root()),
        measured_at: now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        seconds: (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }
}

/// One finished measurement: when, of which folders ([`key_of`]), and the report.
struct Measured {
    at: Instant,
    key: Key,
    report: Arc<Report>,
}

/// The folders a measurement reads ([`key_of`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Key {
    data_root: PathBuf,
    models_dir: PathBuf,
    training_dir: PathBuf,
    cache_root: Option<PathBuf>,
    dataset_dirs: Vec<PathBuf>,
    log_dir: Option<PathBuf>,
}

/// The last measurement of each set of folders (newest first, at most [`KEPT`]) and whether one
/// is running. One slot per set: with a single slot shared by all, two callers of different
/// folders (tests with their own data roots; a generation restart that moved `download_dir`)
/// overwrite each other's report before the other looks, and each measures again, round after
/// round, until its wait runs out (seen on Linux CI, where measuring an empty folder takes less
/// than one 25 ms look).
///
/// `epoch` counts [`invalidate`] calls: a measurement stores its report only when no Clear or
/// Remove happened since it started (QA-076: one that walked a cache before its Clear would
/// otherwise show the cleared sizes again, as fresh, for [`FRESH`]).
#[derive(Default)]
struct Cache {
    last: Vec<Measured>,
    running: bool,
    epoch: u64,
}

/// How many sets of folders [`Cache`] keeps a report of (the service has one; tests have more).
const KEPT: usize = 8;

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    last: Vec::new(),
    running: false,
    epoch: 0,
});

/// Clears [`Cache::running`] when the measuring thread ends, also by a panic in [`measure`] (which
/// would otherwise leave every later look waiting on a measurement that never ends).
struct Measuring;

impl Drop for Measuring {
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!(
                "storage: measuring the folders panicked; the Storage card measures again on the \
                 next look (please report this with the log)"
            );
        }
        CACHE.lock_recover().running = false;
    }
}

/// Tests: [`measure`] panics for settings of this data root (and says it did).
#[cfg(test)]
static PANIC_FOR: Mutex<Option<PathBuf>> = Mutex::new(None);
#[cfg(test)]
static PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tests: the measuring thread for settings of this data root says it started, waits to be
/// released before it measures, and says when it has stored (or dropped) its report.
#[cfg(test)]
struct Hold {
    root: PathBuf,
    entered: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
    done: std::sync::mpsc::Sender<()>,
}

#[cfg(test)]
static HOLD_FOR: Mutex<Option<Hold>> = Mutex::new(None);

/// The measurement to show: a fresh one of the same folders ([`key_of`]) is returned at once;
/// otherwise one is started on its own thread (after a measurement of other folders has ended,
/// and again when a Clear forgot it before it was read) and awaited for at most `wait`. Returns the report of these folders (None before the first
/// one ends) and whether a measurement is still running.
pub fn current(s: Settings, runs: Vec<RunRecord>, wait: Duration) -> (Option<Arc<Report>>, bool) {
    let key = key_of(&s);
    let until = Instant::now() + wait;
    loop {
        {
            let mut c = CACHE.lock_recover();
            let last = c
                .last
                .iter()
                .find(|l| l.key == key)
                .map(|l| (l.at, l.report.clone()));
            if let Some((at, r)) = &last
                && at.elapsed() < FRESH
            {
                return (Some(r.clone()), false);
            }
            // Nothing fresh of these folders and nothing running (none yet, another folder's
            // ended, or a Clear forgot ours): measure them.
            if !c.running {
                let (s, runs) = (s.clone(), runs.clone());
                c.running = true;
                let k = key.clone();
                let epoch = c.epoch;
                let spawned = std::thread::Builder::new()
                    .name("storage-usage".into())
                    .spawn(move || {
                        let _measuring = Measuring;
                        crate::backend::priority::lower_this_thread(
                            crate::backend::priority::ThreadRole::Background,
                        );
                        #[cfg(test)]
                        if PANIC_FOR.lock_recover().as_deref() == Some(&s.config().data_root()) {
                            PANICKED.store(true, std::sync::atomic::Ordering::SeqCst);
                            panic!("test: measuring panics");
                        }
                        #[cfg(test)]
                        let hold = {
                            let mut h = HOLD_FOR.lock_recover();
                            if h.as_ref().map(|h| &h.root) == Some(&s.config().data_root()) {
                                h.take()
                            } else {
                                None
                            }
                        };
                        #[cfg(test)]
                        if let Some(h) = &hold {
                            let _ = h.entered.send(());
                            let _ = h.release.recv_timeout(Duration::from_secs(60));
                        }
                        let report = Arc::new(measure(&s, &runs));
                        let mut c = CACHE.lock_recover();
                        // A Clear or Remove since the start: what was measured may be gone.
                        if c.epoch == epoch {
                            c.last.retain(|l| l.key != k);
                            c.last.insert(
                                0,
                                Measured {
                                    at: Instant::now(),
                                    key: k,
                                    report,
                                },
                            );
                            c.last.truncate(KEPT);
                        }
                        drop(c);
                        #[cfg(test)]
                        if let Some(h) = hold {
                            let _ = h.done.send(());
                        }
                    });
                if let Err(e) = spawned {
                    c.running = false;
                    tracing::warn!("storage: could not start measuring ({e})");
                    return (last.map(|(_, r)| r), false);
                }
            }
            if Instant::now() >= until {
                return (last.map(|(_, r)| r), c.running);
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// The folders a measurement covers: a report counts only for the same ones (a generation
/// restart that moves `download_dir`, or tests with their own data roots, measure others).
/// The key holds every folder [`measure`] reads, so a change of any of them (`collect.dir`,
/// `collect.review_dir`, `log_path` included) measures again.
fn key_of(s: &Settings) -> Key {
    Key {
        data_root: s.config().data_root(),
        models_dir: s.models_dir(),
        training_dir: s.training_dir(),
        cache_root: s.cache_root(),
        dataset_dirs: s.dataset_dirs(),
        log_dir: s.log_dir(),
    }
}

/// Forget the last measurement (after a Clear or Remove), so the next look measures again; a
/// measurement running now stores nothing (`Cache::epoch`).
pub fn invalidate() {
    let mut c = CACHE.lock_recover();
    c.last.clear();
    c.epoch = c.epoch.wrapping_add(1);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cache_titles() {
        assert_eq!(cache_title("openvino-gpu-f16"), "OpenVINO cache (GPU, f16)");
        assert_eq!(cache_title("openvino-cpu-f32"), "OpenVINO cache (CPU, f32)");
        assert!(cache_title("coreml").starts_with("CoreML"));
        assert!(cache_title("tensorrt").starts_with("TensorRT"));
    }

    #[test]
    fn ages_read_as_days() {
        let now = UNIX_EPOCH + Duration::from_secs(100 * 86_400);
        assert_eq!(days_ago(100 * 86_400 - 5, now), "today");
        assert_eq!(days_ago(99 * 86_400 - 5, now), "yesterday");
        assert_eq!(days_ago(88 * 86_400, now), "12 days ago");
        assert_eq!(days_ago(200 * 86_400, now), "today", "a future time");
    }

    #[test]
    fn a_report_counts_only_for_the_folders_it_measured() {
        let settings = |tag: &str| {
            let root =
                std::env::temp_dir().join(format!("bop-usage-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let mut c = crate::config::Config {
                download_dir: Some(root.clone()),
                cache_dir: root.join("cache").display().to_string(),
                ..Default::default()
            };
            c.models.clear();
            (root, Settings::for_config(&c))
        };
        let (ra, a) = settings("a");
        let (rb, b) = settings("b");
        assert_ne!(key_of(&a), key_of(&b));
        let long = Duration::from_secs(60);
        // Other tests share the process-wide cache: wait for each answer to be complete.
        let (first, running) = current(a.clone(), Vec::new(), long);
        assert!(!running);
        let first = first.unwrap();
        let (other, running) = current(b, Vec::new(), long);
        assert!(!running);
        let other = other.unwrap();
        assert!(!Arc::ptr_eq(&first, &other), "b got a's report");
        // a measures again rather than reusing b's report.
        let (again, _) = current(a, Vec::new(), long);
        assert!(!Arc::ptr_eq(&again.unwrap(), &other));
        for r in [ra, rb] {
            std::fs::remove_dir_all(r).ok();
        }
    }

    /// A caller that keeps measuring other folders never starves ours: each set of folders keeps
    /// its own report (with one shared slot this failed 10 of 10 runs; on Linux CI two tests of
    /// this module starved each other that way until their 60 s waits ran out).
    #[test]
    fn a_look_gets_its_report_while_others_measure() {
        let settings = |tag: &str| {
            let root =
                std::env::temp_dir().join(format!("bop-usage-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let mut c = crate::config::Config {
                download_dir: Some(root.clone()),
                cache_dir: root.join("cache").display().to_string(),
                ..Default::default()
            };
            c.models.clear();
            (root, Settings::for_config(&c))
        };
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (other_root, other) = settings("other");
        let st = stop.clone();
        let t = std::thread::spawn(move || {
            while !st.load(std::sync::atomic::Ordering::SeqCst) {
                let _ = current(other.clone(), Vec::new(), Duration::ZERO);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        let (mine_root, mine) = settings("mine");
        let (r, running) = current(mine, Vec::new(), Duration::from_secs(30));
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        t.join().unwrap();
        assert!(
            r.is_some() && !running,
            "starved (still running: {running})"
        );
        for root in [other_root, mine_root] {
            std::fs::remove_dir_all(root).ok();
        }
    }

    /// Audit 2026-11 phase 1: a panic while measuring must not leave the card "measuring" for
    /// good; the next look measures again.
    #[test]
    fn a_panic_while_measuring_lets_the_next_look_measure_again() {
        let root = std::env::temp_dir().join(format!("bop-usage-panic-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        *PANIC_FOR.lock().unwrap() = Some(c.data_root());
        // Other tests share the cache and may be measuring their own folders: wait for ours.
        let deadline = Instant::now() + Duration::from_secs(60);
        while !PANICKED.load(std::sync::atomic::Ordering::SeqCst) {
            let (r, _) = current(s.clone(), Vec::new(), Duration::ZERO);
            assert!(r.is_none(), "nothing measured yet");
            assert!(Instant::now() < deadline, "the measurement never started");
            std::thread::sleep(Duration::from_millis(10));
        }
        while CACHE.lock_recover().running {
            assert!(
                Instant::now() < deadline,
                "still running long after the measuring thread panicked"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        *PANIC_FOR.lock().unwrap() = None;
        let (r, running) = current(s, Vec::new(), Duration::from_secs(60));
        assert!(r.is_some() && !running);
        std::fs::remove_dir_all(&root).ok();
    }

    /// QA-076: a measurement that was running when a Clear forgot the cache stores nothing, so
    /// the next look measures again rather than showing the cleared sizes as fresh.
    #[test]
    fn a_measurement_running_across_a_clear_is_not_kept() {
        let root = std::env::temp_dir().join(format!("bop-usage-epoch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let s = settings_at(&root);
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel();
        let (done_tx, done) = std::sync::mpsc::channel();
        *HOLD_FOR.lock_recover() = Some(Hold {
            root: s.config().data_root(),
            entered: entered_tx,
            release: release_rx,
            done: done_tx,
        });
        // Other tests share the cache and may be measuring their own folders: look until ours
        // starts (and is held before it measures).
        let step = Duration::from_secs(60);
        let deadline = Instant::now() + step;
        loop {
            let _ = current(s.clone(), Vec::new(), Duration::ZERO);
            if entered.recv_timeout(Duration::from_millis(10)).is_ok() {
                break;
            }
            assert!(Instant::now() < deadline, "the measurement never started");
        }
        invalidate();
        release.send(()).unwrap();
        done.recv_timeout(step).expect("the measurement ended");
        let key = key_of(&s);
        assert!(
            CACHE.lock_recover().last.iter().all(|l| l.key != key),
            "a report measured before the Clear was kept"
        );
        // The next look measures again.
        let (r, running) = current(s, Vec::new(), step);
        assert!(r.is_some() && !running);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_card_splits_safe_from_data() {
        let root = std::env::temp_dir().join(format!("bop-usage-{}", uuid::Uuid::new_v4()));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        put(&root.join("cache/openvino-cpu-f16/blob"), 1000);
        put(&s.models_dir().join("m.onnx"), 300);
        put(&s.models_dir().join(".m.train-staging/x.onnx"), 50);
        let t = s.training_dir();
        put(&t.join("exports/e1/classes.txt"), 10);
        put(&t.join("imports/20261107-1/frames/a.jpg"), 70);
        std::fs::write(
            super::super::busy::marker_of(&t.join("imports/20261107-1")),
            b"1",
        )
        .unwrap();
        let r = measure(&s, &[]);
        let ids = |v: &[Item]| v.iter().map(|i| i.id.clone()).collect::<Vec<_>>();
        assert_eq!(
            ids(&r.safe),
            ["cache:openvino-cpu-f16", "staging", "work"],
            "{r:?}"
        );
        assert!(ids(&r.data).contains(&"export:e1".to_string()));
        let models = r.data.iter().find(|i| i.id == "models").unwrap();
        assert_eq!(
            models.bytes, 300,
            "staging is counted under safe, not twice"
        );
        let e = r.data.iter().find(|i| i.id == "export:e1").unwrap();
        assert_eq!((e.bytes, e.action), (10, Some("remove")));
        assert!(r.safe.iter().all(|i| i.action == Some("clear")));
        assert!(
            r.data
                .iter()
                .all(|i| i.action.is_none() || i.id.starts_with("export:"))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// QA-079: a review folder inside the dataset folder (or the reverse) keeps its row, noted
    /// as included in the other, and its bytes count once in `data_bytes`.
    #[test]
    fn a_nested_review_folder_is_counted_once() {
        let root = std::env::temp_dir().join(format!("bop-usage-nest-{}", uuid::Uuid::new_v4()));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        put(&root.join("ds/day/cam/a.jpg"), 1000);
        put(&root.join("ds/review/day/cam/b.jpg"), 300);
        let total = |r: &Report, id: &str| r.data.iter().find(|i| i.id == id).unwrap().clone();
        for (dir, review, inner, outer_title) in [
            ("ds", "ds/review", "review-folder", "Dataset"),
            ("ds/review", "ds", "dataset", "Folder under review"),
        ] {
            c.collect.dir = Some(root.join(dir));
            c.collect.review_dir = Some(root.join(review));
            let s = Settings::for_config(&c);
            assert_eq!(s.dataset_dirs().len(), 2);
            let r = measure(&s, &[]);
            let outer_id = if inner == "dataset" {
                "review-folder"
            } else {
                "dataset"
            };
            let (inner_row, outer_row) = (total(&r, inner), total(&r, outer_id));
            assert_eq!((outer_row.bytes, inner_row.bytes), (1300, 300), "{r:?}");
            assert!(
                inner_row
                    .note
                    .starts_with(&format!("included in {outer_title}")),
                "{}",
                inner_row.note
            );
            assert!(outer_row.note.is_empty(), "{}", outer_row.note);
            let rest: u64 = r
                .data
                .iter()
                .filter(|i| i.id != "dataset" && i.id != "review-folder")
                .map(|i| i.bytes)
                .sum();
            assert_eq!(r.data_bytes, 1300 + rest, "the review folder counts once");
        }
        // Side by side: both count.
        c.collect.dir = Some(root.join("ds/day"));
        c.collect.review_dir = Some(root.join("ds/review"));
        let r = measure(&Settings::for_config(&c), &[]);
        assert!(r.data.iter().all(|i| !i.note.starts_with("included in")));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2: a crashed review export's scratch folder is safe to clear (in the
    /// staging row); one without a marker is the user's to look at, listed with its size.
    #[test]
    fn export_scratch_rows_split_crashed_from_kept() {
        let root = std::env::temp_dir().join(format!("bop-usage-scratch-{}", uuid::Uuid::new_v4()));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: String::new(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        let ex = s.training_dir().join("exports");
        let id = |i: u128| format!("{i:032x}");
        let crashed = ex.join(format!(".a.building-{}", id(1)));
        let kept = ex.join(format!(".b.building-{}", id(2)));
        put(&crashed.join("export/images/a.jpg"), 40);
        std::fs::write(super::super::busy::marker_of(&crashed), b"1").unwrap();
        put(&kept.join("export/images/a.jpg"), 25);
        put(&ex.join("e1/classes.txt"), 10);
        let r = measure(&s, &[]);
        let staging = r.safe.iter().find(|i| i.id == "staging").unwrap();
        assert_eq!(
            (staging.bytes, staging.title.as_str()),
            (40, "Unfinished review exports")
        );
        assert!(
            staging.note.contains(&*crashed.to_string_lossy()),
            "{}",
            staging.note
        );
        let unfinished = r
            .data
            .iter()
            .find(|i| i.id == "unfinished-exports")
            .unwrap();
        assert_eq!(unfinished.bytes, 25);
        assert_eq!(unfinished.action, None, "never removed from the card");
        assert!(unfinished.note.contains(&*kept.to_string_lossy()));
        assert_eq!(r.data_bytes, r.data.iter().map(|i| i.bytes).sum::<u64>());
        let _ = std::fs::remove_dir_all(&root);
    }

    fn put(p: &Path, n: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![0u8; n]).unwrap();
    }

    fn settings_at(root: &Path) -> Settings {
        let mut c = crate::config::Config {
            download_dir: Some(root.to_path_buf()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        Settings::for_config(&c)
    }

    /// Audit 2026-11 phase 4: the uv cache is a "safe to clear" row of its own, model export
    /// leftovers are counted in the staging row (and not again under Models), and the tools/
    /// folders are "your data" rows with their folders as parts, the uv cache and the export
    /// work folders in neither.
    #[test]
    fn tools_rows_and_export_leftovers_are_counted_once() {
        let root = std::env::temp_dir().join(format!("bop-usage-tools-{}", uuid::Uuid::new_v4()));
        let s = settings_at(&root);
        let e = crate::resources::export::Paths::for_config(s.config());
        let t = crate::resources::export::train::TrainPaths::for_config(s.config());
        put(&e.uv_cache.join("wheels-v5/a.whl"), 100);
        put(&e.env.join("lib/x.py"), 20);
        put(&e.python_installs.join("bin/python"), 30);
        put(&t.export.env.join("lib/y.py"), 40);
        put(&t.weights.join("sam3/w.pt"), 50);
        put(&e.work.join("yolo26s/tmp.onnx"), 7);
        let crashed = e.models.join(".yolo26s.export-staging");
        put(&crashed.join("yolo26s.onnx"), 9);
        // Empty: the Models row counts the few bytes of a real marker beside the folder.
        std::fs::write(super::super::busy::marker_of(&crashed), b"").unwrap();
        put(&e.models.join("m.onnx"), 300);
        let r = measure(&s, &[]);
        let uv = r.safe.iter().find(|i| i.id == "uv-cache").unwrap();
        assert_eq!((uv.bytes, uv.action), (100, Some("clear")));
        let staging = r.safe.iter().find(|i| i.id == "staging").unwrap();
        assert_eq!(staging.bytes, 16, "the crashed staging and the work folder");
        assert_eq!(staging.title, "Leftover model export folders");
        let models = r.data.iter().find(|i| i.id == "models").unwrap();
        assert_eq!(
            models.bytes, 300,
            "export staging is counted under safe only"
        );
        let export = r.data.iter().find(|i| i.id == "tools-export").unwrap();
        assert_eq!(
            export.bytes, 50,
            "env + python, not the uv cache or the work folders"
        );
        assert_eq!(export.action, None);
        assert!(
            export.note.contains("Remove export toolchain"),
            "{}",
            export.note
        );
        assert_eq!(export.parts.len(), 2, "{:?}", export.parts);
        let train = r.data.iter().find(|i| i.id == "tools-train").unwrap();
        assert_eq!((train.bytes, train.action), (90, None));
        assert!(train.note.contains("Remove training environment"));
        assert!(train.parts.iter().all(|p| !p.path.contains("uv-cache")));
        assert_eq!(r.safe_bytes, r.safe.iter().map(|i| i.bytes).sum::<u64>());

        // A running export (its staging held, the export lock held) is no leftover.
        let held = super::super::busy::Busy::hold(&crashed);
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(e.lock_file())
            .unwrap();
        fs4::FileExt::try_lock(&lock).unwrap();
        assert!(export_leftovers(&e).is_empty());
        fs4::FileExt::unlock(&lock).unwrap();
        drop(held);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Run folders no record names are listed (report only) with their sizes; a recorded run is
    /// not.
    #[test]
    fn run_folders_no_record_names_are_listed() {
        let root =
            std::env::temp_dir().join(format!("bop-usage-unlisted-{}", uuid::Uuid::new_v4()));
        let s = settings_at(&root);
        let training = s.training_dir();
        put(&training.join("20261101-lost/train.log"), 11);
        put(&training.join("20261101-lost/work/last.pt"), 22);
        put(&training.join("exports/e1/classes.txt"), 1);
        let r = measure(&s, &[]);
        let row = r.data.iter().find(|i| i.id == "unlisted-runs").unwrap();
        assert_eq!((row.bytes, row.action), (33, None));
        assert_eq!(row.parts.len(), 1);
        assert_eq!(row.parts[0].title, "20261101-lost");
        assert!(row.note.contains("train list"), "{}", row.note);
        assert!(unlisted_runs_item(&root.join("nope"), &[]).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// ONNX Runtime builds this config does not load are listed, report only, and the loaded one
    /// is not; `cuda-libs` only while another build loads; the old flat layout's files only
    /// when the library is not loaded from there.
    #[test]
    fn runtimes_this_config_does_not_load_are_listed() {
        let root = std::env::temp_dir().join(format!("bop-usage-ort-{}", uuid::Uuid::new_v4()));
        let ort = root.join("onnxruntime");
        let lib = crate::backend::libs::ort_library_file_name();
        put(&ort.join("cpu").join(lib), 10);
        put(&ort.join("cuda").join(lib), 20);
        put(&ort.join("cuda-libs/libcudart.so.12"), 30);
        let ids = |v: &[Item]| v.iter().map(|i| i.id.clone()).collect::<Vec<_>>();

        // The CPU build loads: CUDA and its libraries are unused.
        let items = runtime_items(&ort, Some(&ort.join("cpu").join(lib)));
        assert_eq!(
            ids(&items),
            ["runtime:onnxruntime-cuda", "runtime:cuda-libs"],
            "{items:?}"
        );
        assert_eq!(items[0].bytes, 20);
        assert_eq!(items[1].bytes, 30);
        assert!(items.iter().all(|i| i.action.is_none()));
        assert!(
            items[0].note.contains("Resources card"),
            "{}",
            items[0].note
        );

        // The CUDA build loads: the CPU one is unused, the CUDA libraries are not.
        let items = runtime_items(&ort, Some(&ort.join("cuda").join(lib)));
        assert_eq!(ids(&items), ["runtime:onnxruntime-cpu"]);

        // A library of the old flat layout in the root, not loaded: listed with its files.
        put(&ort.join(lib), 5);
        put(&ort.join("flavor.txt"), 4);
        put(&ort.join("active.txt"), 4);
        let items = runtime_items(&ort, Some(&ort.join("cuda").join(lib)));
        let flat = items
            .iter()
            .find(|i| i.id == "runtime:onnxruntime-flat")
            .unwrap();
        assert_eq!(flat.bytes, 9, "the library and flavor.txt, not active.txt");
        // Loaded from the root itself: nothing of the flat layout is listed.
        let items = runtime_items(&ort, Some(&ort.join(lib)));
        assert!(!ids(&items).contains(&"runtime:onnxruntime-flat".to_string()));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The compiled models of an OpenVINO or TensorRT cache are parts of its row (from
    /// [`caches::compiled_entries`]; a hand-built list here), superseded ones first and named in
    /// the row's note.
    #[test]
    fn compiled_entries_are_listed_under_their_cache() {
        assert!(has_compiled_entries("openvino-gpu-f16"));
        assert!(has_compiled_entries("tensorrt"));
        assert!(!has_compiled_entries("coreml"));
        let entries = vec![
            caches::CompiledUse {
                path: PathBuf::from("cache").join("openvino-gpu-f16").join("a"),
                model: Some(PathBuf::from("models").join("yolo.onnx")),
                bytes: 2_000,
                superseded: false,
            },
            caches::CompiledUse {
                path: PathBuf::from("cache").join("openvino-gpu-f16").join("b"),
                model: Some(PathBuf::from("models").join("old.onnx")),
                bytes: 1_000,
                superseded: true,
            },
            caches::CompiledUse {
                path: PathBuf::from("cache").join("openvino-gpu-f16").join("c"),
                model: None,
                bytes: 3_000,
                superseded: false,
            },
        ];
        let parts = compiled_parts(&entries);
        let titles: Vec<&str> = parts.iter().map(|p| p.title.as_str()).collect();
        assert_eq!(
            titles,
            [
                "old.onnx",
                "compiled model (its model file is not recorded)",
                "yolo.onnx"
            ]
        );
        assert!(parts[0].note.starts_with("superseded"), "{}", parts[0].note);
        assert_eq!(parts[0].size, format_size(1_000));
        assert!(parts[2].note.contains("yolo.onnx"));
        let note = superseded_note(&entries);
        assert!(note.contains("1 superseded compiled model(s)"), "{note}");
        assert_eq!(superseded_note(&entries[..1]), "");
        assert!(compiled_parts(&[]).is_empty());
    }
}
