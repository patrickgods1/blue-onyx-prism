//! The model caches under `cache_dir`: OpenVINO's compiled blobs (`openvino-<device>-<prec>/`),
//! ONNX Runtime's CoreML models and graph rewrites (`coreml/`) and TensorRT engines (`tensorrt/`).
//! Everything here is rebuilt by the next load (slower once), so the sweep may delete it.
//!
//! - [`touch`]: a backend loading from a cache folder records the use in `<folder>/.last-used`
//!   (unix seconds), written at most once a day per folder, and the folder counts as in use by this
//!   process until it exits ([`in_use`]).
//! - [`sweep_unused`]: a folder not used for [`UNUSED_DAYS`] goes. A folder with no marker yet
//!   (made before markers existed) gets one, starting its clock: nothing is deleted on a guess.
//! - Per-model subfolders ([`model_cache_dir`]): OpenVINO and TensorRT compile each model into
//!   `openvino-<device>-<prec>/<stem>-<key16>/` and `tensorrt/<stem>-<key16>/`, with a
//!   [`MODEL_FILE`] naming the model's canonical path (as ONNX Runtime does in its CoreML
//!   folders). `key16` ([`model_key`]) is the first 16 hex digits of SHA-256 over the canonical
//!   path, the file size and its modification time in nanoseconds: cheap, and new on every
//!   re-export or retrain at the same path, so a TensorRT engine (keyed by ONNX Runtime on the
//!   file name and graph names, not the weights) of an earlier file is never reused. Each
//!   subfolder has its own `.last-used`; the folder above keeps its marker too.
//! - [`sweep_compiled`]: in every `openvino-*` folder and `tensorrt/`, even while the folder is in
//!   use, removes a subfolder whose model file is gone, whose model file changed since (its key is
//!   not the file's key now), or that was not used for [`UNUSED_DAYS`] by its own marker. Never one
//!   this process compiled into ([`in_use`]: Windows keeps OpenVINO's blob mapped), never one
//!   without a [`MODEL_FILE`] (no guess) except an empty one a crash left (an hour old), and a
//!   model file that is not configured but still on disk (a benchmark candidate, a run's model)
//!   is not gone. Compiled files these folders held directly before the subfolders (OpenVINO
//!   `<digits>.blob` / `.cl_cache`, TensorRT `TensorrtExecutionProvider_*.engine` / `.profile` /
//!   `.timing`) are removed by exact name once a day old (nothing writes there any more; the age
//!   spares an older binary still running). [`compiled_entries`] reports the same per subfolder.
//! - [`sweep_coreml`]: entries of `coreml/` whose model is gone: a rewrite
//!   `<stem>-<sha16>-v<N>.onnx` of an older rewriter or of no model file there is now, and
//!   ONNX Runtime's compiled model folders whose `model.txt` names a file that no longer exists or
//!   that a newer folder of the same file replaced (a re-export gets a new `COREML_CACHE_KEY`).

use super::Removed;
use crate::sync::LockExt;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The last-use marker of a cache folder.
pub const LAST_USED_FILE: &str = ".last-used";
/// A cache folder unused this long is deleted.
pub const UNUSED_DAYS: u64 = 60;
const DAY: Duration = Duration::from_secs(24 * 60 * 60);
/// A `.tmp` file of an interrupted rewrite older than this is a leftover (also an empty
/// per-model subfolder and its `model.txt` temporary file).
const TMP_AGE: Duration = Duration::from_secs(60 * 60);
/// A compiled file of the flat layout (before per-model subfolders) untouched this long goes: an
/// older binary still running may write one for a while.
const LEGACY_AGE: Duration = DAY;
/// The file in a per-model cache subfolder naming the model file it was compiled from (the name
/// ONNX Runtime's CoreML cache uses).
pub const MODEL_FILE: &str = "model.txt";
/// The longest model-name part of a per-model subfolder name (the key follows it).
const MAX_STEM: usize = 40;

/// Cache folders this process loaded from, with when their marker was last written or found fresh.
static TOUCHED: Mutex<BTreeMap<PathBuf, SystemTime>> = Mutex::new(BTreeMap::new());

/// The key of `dir` in [`TOUCHED`]: canonical when it exists, so `cache/x` and `./cache/x` agree.
fn key(dir: &Path) -> PathBuf {
    dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf())
}

/// Record that a backend loads from the cache folder `dir` now. Cheap: at most one marker write a
/// day per folder, nothing else on disk; a failed write is logged at debug and changes nothing.
pub fn touch(dir: &Path) {
    let now = SystemTime::now();
    let k = key(dir);
    let mut touched = TOUCHED.lock_recover();
    if touched.get(&k).is_some_and(|at| age(now, *at) < DAY) {
        return;
    }
    let at = match last_used(dir).filter(|at| age(now, *at) < DAY) {
        Some(fresh) => fresh,
        None => {
            if let Err(e) = write_marker(dir, now) {
                tracing::debug!("cache {}: recording its use: {e}", dir.display());
            }
            now
        }
    };
    touched.insert(k, at);
}

/// Whether this process loaded from `dir` (see [`touch`]).
pub fn in_use(dir: &Path) -> bool {
    TOUCHED.lock_recover().contains_key(&key(dir))
}

fn write_marker(dir: &Path, at: SystemTime) -> std::io::Result<()> {
    let secs = at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    std::fs::write(dir.join(LAST_USED_FILE), format!("{secs}\n"))
}

/// When `dir` was last used: its marker's time, else the marker file's modification time; None
/// without a marker.
pub fn last_used(dir: &Path) -> Option<SystemTime> {
    let path = dir.join(LAST_USED_FILE);
    let text = std::fs::read_to_string(&path).ok()?;
    match text.trim().parse::<u64>() {
        Ok(secs) => UNIX_EPOCH.checked_add(Duration::from_secs(secs)),
        Err(_) => std::fs::metadata(&path).and_then(|m| m.modified()).ok(),
    }
}

/// `now - at`, zero when `at` is in the future (a clock change).
fn age(now: SystemTime, at: SystemTime) -> Duration {
    now.duration_since(at).unwrap_or_default()
}

/// Whether `name` is a cache folder this program makes (anything else under `cache_dir` is left
/// alone: the setting may name a folder that holds other files).
pub fn is_cache_name(name: &str) -> bool {
    name == "coreml" || name == "tensorrt" || name.starts_with("openvino-")
}

/// The cache folders under `root`, by name.
pub fn cache_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| e.file_name().to_str().is_some_and(is_cache_name))
        .map(|e| e.path())
        .collect();
    out.sort();
    out
}

/// Whether a folder last used at `last` (None: no marker) is unused at `now`.
pub fn is_unused(last: Option<SystemTime>, now: SystemTime) -> bool {
    last.is_some_and(|at| age(now, at) >= DAY * u32::try_from(UNUSED_DAYS).unwrap_or(60))
}

/// Delete the cache folders under `root` not used for [`UNUSED_DAYS`]. A folder this process uses
/// has its marker refreshed instead; one with no marker gets one.
pub fn sweep_unused(root: &Path, now: SystemTime, out: &mut Vec<Removed>) {
    for dir in cache_dirs(root) {
        if in_use(&dir) {
            touch(&dir);
            continue;
        }
        let last = last_used(&dir);
        if last.is_none() {
            if let Err(e) = write_marker(&dir, now) {
                tracing::debug!("cache {}: starting its clock: {e}", dir.display());
            }
            continue;
        }
        if is_unused(last, now) {
            let days = last.map_or(0, |at| age(now, at).as_secs() / DAY.as_secs());
            super::remove_counted(&dir, &format!("cache not used for {days} days"), out);
        }
    }
}

/// A rewrite file name `<stem>-<sha16>-v<N>.onnx`: (stem, sha16, N).
pub fn parse_rewrite_name(name: &str) -> Option<(&str, &str, u32)> {
    let base = name.strip_suffix(".onnx")?;
    let (rest, ver) = base.rsplit_once("-v")?;
    let ver: u32 = ver.parse().ok()?;
    let (stem, sha) = rest.rsplit_once('-')?;
    let hex = sha.len() == 16 && sha.bytes().all(|b| b.is_ascii_hexdigit());
    (hex && !stem.is_empty()).then_some((stem, sha, ver))
}

/// Why the rewrite `name` is an orphan, or None to keep it. `sources` are the model files there
/// are now; `sha16` gives the first 16 hex digits of a file's SHA-256 (None: unreadable, then the
/// rewrite is kept).
pub fn rewrite_orphan(
    name: &str,
    sources: &[PathBuf],
    version: u32,
    sha16: &mut dyn FnMut(&Path) -> Option<String>,
) -> Option<String> {
    let (stem, sha, ver) = parse_rewrite_name(name)?;
    if ver != version {
        return Some(format!(
            "CoreML rewrite of an older rewriter (v{ver}, now v{version})"
        ));
    }
    let same_stem: Vec<&PathBuf> = sources
        .iter()
        .filter(|p| p.file_stem().and_then(|s| s.to_str()) == Some(stem))
        .collect();
    if same_stem.is_empty() {
        return Some(format!("CoreML rewrite of {stem}.onnx, which is gone"));
    }
    let mut unreadable = false;
    for p in same_stem {
        match sha16(p) {
            Some(h) if h.eq_ignore_ascii_case(sha) => return None,
            Some(_) => {}
            None => unreadable = true,
        }
    }
    (!unreadable).then(|| format!("CoreML rewrite of an earlier {stem}.onnx (the file changed)"))
}

/// An ONNX Runtime CoreML cache folder: its `model.txt` names the model it was compiled from.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledEntry {
    pub dir: PathBuf,
    pub model: PathBuf,
    pub modified: SystemTime,
}

/// Which compiled entries to delete, with why: those whose model file is gone (`exists`), and of
/// several for one file all but the newest. `kind` names them in the reason ("CoreML").
pub fn compiled_orphans(
    entries: &[CompiledEntry],
    exists: &dyn Fn(&Path) -> bool,
    kind: &str,
) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let mut by_model: BTreeMap<&Path, Vec<&CompiledEntry>> = BTreeMap::new();
    for e in entries {
        if exists(&e.model) {
            by_model.entry(&e.model).or_default().push(e);
        } else {
            out.push((
                e.dir.clone(),
                format!(
                    "compiled {kind} model of {}, which is gone",
                    e.model.display()
                ),
            ));
        }
    }
    for (model, mut list) in by_model {
        list.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.dir.cmp(&b.dir)));
        for old in list.iter().skip(1) {
            out.push((
                old.dir.clone(),
                format!(
                    "compiled {kind} model of an earlier {} (a newer one replaced it)",
                    model.display()
                ),
            ));
        }
    }
    out
}

/// The first 16 hex digits of `path`'s SHA-256, None when it cannot be read or is larger than the
/// rewriter reads.
fn file_sha16(path: &Path) -> Option<String> {
    use sha2::Digest;
    use std::io::Read;
    let f = std::fs::File::open(path).ok()?;
    if f.metadata().ok()?.len() > crate::backend::onnx_rewrite::MAX_REWRITE_BYTES {
        return None;
    }
    let mut r = std::io::BufReader::new(f);
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = r.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(buf.get(..n)?);
    }
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    hex.get(..16).map(str::to_string)
}

/// Delete the orphans of the `coreml` cache folder `dir` (see the module docs). `sources` are the
/// model files there are now (the models folder and the config's model paths).
pub fn sweep_coreml(dir: &Path, sources: &[PathBuf], now: SystemTime, out: &mut Vec<Removed>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut hashes: BTreeMap<PathBuf, Option<String>> = BTreeMap::new();
    let mut sha16 = |p: &Path| {
        hashes
            .entry(p.to_path_buf())
            .or_insert_with(|| file_sha16(p))
            .clone()
    };
    let mut compiled = Vec::new();
    let mut rewrites_removed = BTreeSet::new();
    for e in rd.flatten() {
        let path = e.path();
        let Some(name) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Ok(md) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if md.is_file() {
            if name.starts_with('.') && name.contains(".tmp-") {
                if md.modified().is_ok_and(|m| age(now, m) >= TMP_AGE) {
                    super::remove_counted(&path, "unfinished CoreML rewrite", out);
                }
                continue;
            }
            let version = crate::backend::onnx_rewrite::REWRITER_VERSION;
            if let Some(why) = rewrite_orphan(&name, sources, version, &mut sha16) {
                super::remove_counted(&path, &why, out);
                rewrites_removed.insert(path);
            }
        } else if md.is_dir()
            && let Ok(text) = std::fs::read_to_string(path.join("model.txt"))
        {
            let model = PathBuf::from(text.lines().next().unwrap_or_default().trim());
            if model.is_absolute() {
                compiled.push(CompiledEntry {
                    dir: path,
                    model,
                    modified: md.modified().unwrap_or(UNIX_EPOCH),
                });
            }
        }
    }
    // A compiled model of a rewrite deleted just now goes with it.
    let exists = |p: &Path| !rewrites_removed.contains(p) && std::fs::symlink_metadata(p).is_ok();
    for (path, why) in compiled_orphans(&compiled, &exists, "CoreML") {
        super::remove_counted(&path, &why, out);
    }
}

/// What a cache folder holds, for the Storage card.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheInfo {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// Unix seconds of the last use, when recorded.
    pub last_used: Option<u64>,
    /// Loaded from by this process.
    pub in_use: bool,
}

/// The cache folders under `root` with their sizes.
pub fn infos(root: &Path) -> Vec<CacheInfo> {
    cache_dirs(root)
        .into_iter()
        .map(|dir| CacheInfo {
            name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes: crate::training::retry::disk_use(&dir),
            last_used: last_used(&dir)
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs()),
            in_use: in_use(&dir),
            path: dir,
        })
        .collect()
}

/// One compiled model's entry in a per-model cache subfolder (OpenVINO `openvino-<dev>-<prec>/`,
/// TensorRT `tensorrt/`; also ONNX Runtime's CoreML folders in `coreml/`): the Storage card
/// lists them under their folder, report-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledUse {
    /// The entry (a subfolder of the cache folder).
    pub path: PathBuf,
    /// The model file it was compiled from, when recorded.
    pub model: Option<PathBuf>,
    pub bytes: u64,
    /// The model file is gone or a newer entry of the same file replaced it (the sweep removes it).
    pub superseded: bool,
}

// ---- per-model subfolders ---------------------------------------------------------------------

/// The key of a model file: the first 16 hex digits of SHA-256 over its canonical path, its size
/// and its modification time (nanoseconds since 1970; 0 before). Pure. A re-export or retrain at
/// the same path changes the time (and mostly the size), so it gets a new key.
pub fn model_key(canonical: &Path, size: u64, modified: SystemTime) -> String {
    use sha2::Digest;
    let nanos = modified
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut h = sha2::Sha256::new();
    h.update(canonical.as_os_str().as_encoded_bytes());
    h.update([0u8]);
    h.update(size.to_le_bytes());
    h.update(nanos.to_le_bytes());
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    hex.get(..16).map(str::to_string).unwrap_or(hex)
}

/// The per-model subfolder name `<stem>-<key>` for `model` (pure): the file stem with every
/// character but ASCII letters, digits, `-` and `_` replaced by `_` (no dot, so no Windows
/// device name such as `CON.x`), cut to [`MAX_STEM`]; `model` when nothing is left.
pub fn model_dir_name(model: &Path, key: &str) -> String {
    let stem: String = model
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(MAX_STEM)
        .collect();
    let stem = if stem.is_empty() {
        "model".into()
    } else {
        stem
    };
    format!("{stem}-{key}")
}

/// A per-model subfolder name `<stem>-<key16>` (16 lowercase hex digits): (stem, key).
pub fn parse_model_dir_name(name: &str) -> Option<(&str, &str)> {
    let (stem, key) = name.rsplit_once('-')?;
    let hex = key.len() == 16
        && key
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    (hex && !stem.is_empty()).then_some((stem, key))
}

/// The model file at `model` as it is now: its canonical path and [`model_key`]. None when it
/// cannot be read or its path is not valid UTF-8 (the caches take UTF-8 paths only).
pub fn model_identity(model: &Path) -> Option<(PathBuf, String)> {
    let canonical = model.canonicalize().ok()?;
    canonical.to_str()?;
    let md = std::fs::metadata(&canonical).ok()?;
    if !md.is_file() {
        return None;
    }
    let key = model_key(&canonical, md.len(), md.modified().ok()?);
    Some((canonical, key))
}

/// The per-model subfolder of the cache folder `folder` for the model file `model`
/// (`<folder>/<stem>-<key16>`, see the module docs): created, its [`MODEL_FILE`] written when it
/// does not already name the model (atomically), and both it and `folder` recorded as used
/// ([`touch`]). An error when the model cannot be read or the folder written: the caller then
/// compiles without a cache.
pub fn model_cache_dir(folder: &Path, model: &Path) -> std::io::Result<PathBuf> {
    let (canonical, key) = model_identity(model).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "model file {} cannot be read or its path is not UTF-8",
                model.display()
            ),
        )
    })?;
    let sub = folder.join(model_dir_name(model, &key));
    std::fs::create_dir_all(&sub)?;
    let text = format!("{}\n", canonical.display());
    let record = sub.join(MODEL_FILE);
    if std::fs::read_to_string(&record).ok().as_deref() != Some(text.as_str()) {
        crate::fsx::write_atomic(&record, text.as_bytes())?;
    }
    touch(&sub);
    touch(folder);
    Ok(sub)
}

/// Whether the cache folder named `name` holds per-model subfolders keyed by [`model_key`].
fn is_keyed_folder(name: &str) -> bool {
    name == "tensorrt" || name.starts_with("openvino-")
}

/// Whether `file` is a compiled file the cache folder named `folder` held directly before the
/// per-model subfolders (exact shapes only): OpenVINO `<digits>.blob` / `<digits>.cl_cache`,
/// ONNX Runtime TensorRT `TensorrtExecutionProvider_*.engine` / `.profile` / `.timing`.
pub fn is_flat_compiled(folder: &str, file: &str) -> bool {
    if folder.starts_with("openvino-") {
        let digits = |h: &str| !h.is_empty() && h.bytes().all(|b| b.is_ascii_digit());
        file.strip_suffix(".blob").is_some_and(digits)
            || file.strip_suffix(".cl_cache").is_some_and(digits)
    } else if folder == "tensorrt" {
        file.strip_prefix("TensorrtExecutionProvider_")
            .is_some_and(|rest| {
                [".engine", ".profile", ".timing"]
                    .iter()
                    .any(|ext| rest.len() > ext.len() && rest.ends_with(ext))
            })
    } else {
        false
    }
}

/// What the sweep does with one entry of a cache folder.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Fate {
    Keep,
    /// Its model file is gone or changed: provably never read again.
    Superseded(String),
    /// Unused for [`UNUSED_DAYS`].
    Expired(String),
    /// A crash leftover or a flat-layout file.
    Leftover(String),
}

/// One entry of a cache folder, judged.
#[derive(Debug, Clone)]
struct Judged {
    path: PathBuf,
    /// A subfolder (else a file).
    dir: bool,
    model: Option<PathBuf>,
    /// The subfolder has a [`MODEL_FILE`] but no `.last-used` yet.
    unmarked: bool,
    fate: Fate,
}

/// The model a compiled subfolder names in its [`MODEL_FILE`] (first line, absolute only).
fn recorded_model(dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(dir.join(MODEL_FILE)).ok()?;
    let model = PathBuf::from(text.lines().next().unwrap_or_default().trim());
    model.is_absolute().then_some(model)
}

/// Whether the subfolder `dir` holds nothing but [`crate::fsx`] temporary files (a crash between
/// creating it and writing its [`MODEL_FILE`]).
fn holds_nothing(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.flatten().all(|e| {
            e.file_type().is_ok_and(|t| t.is_file())
                && e.file_name().to_str().is_some_and(crate::fsx::is_temp_name)
        })
    })
}

/// Judge every entry of the cache folder `folder` at `now` (see the module docs). Reads only.
fn judge(folder: &Path, now: SystemTime) -> Vec<Judged> {
    let name = folder
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    let keyed = is_keyed_folder(&name);
    let Ok(rd) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let path = e.path();
        let Some(entry) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Ok(md) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let modified = md.modified().unwrap_or(now);
        if md.is_file() {
            if keyed && is_flat_compiled(&name, &entry) && age(now, modified) >= LEGACY_AGE {
                out.push(Judged {
                    path,
                    dir: false,
                    model: None,
                    unmarked: false,
                    fate: Fate::Leftover(
                        "compiled model of the flat cache layout (models now compile into \
                         per-model folders)"
                            .into(),
                    ),
                });
            }
            continue;
        }
        if !md.is_dir() {
            continue;
        }
        let model = recorded_model(&path);
        let mut judged = Judged {
            path,
            dir: true,
            model: model.clone(),
            unmarked: false,
            fate: Fate::Keep,
        };
        match model {
            None => {
                if keyed && age(now, modified) >= TMP_AGE && holds_nothing(&judged.path) {
                    judged.fate =
                        Fate::Leftover("empty per-model cache folder a crash left".into());
                }
            }
            Some(model) if keyed => {
                let last = last_used(&judged.path);
                judged.unmarked = last.is_none();
                if std::fs::symlink_metadata(&model).is_err() {
                    judged.fate = Fate::Superseded(format!(
                        "compiled model of {}, which is gone",
                        model.display()
                    ));
                } else if let Some((_, key)) = parse_model_dir_name(&entry)
                    && let Some((_, now_key)) = model_identity(&model)
                    && now_key != key
                {
                    judged.fate = Fate::Superseded(format!(
                        "compiled model of an earlier {} (the file changed since)",
                        model.display()
                    ));
                } else if is_unused(last, now) && age(now, modified) >= DAY {
                    let days = last.map_or(0, |at| age(now, at).as_secs() / DAY.as_secs());
                    judged.fate = Fate::Expired(format!(
                        "compiled model of {} not used for {days} days",
                        model.display()
                    ));
                }
            }
            // ONNX Runtime's CoreML folders: `compiled_orphans` below.
            Some(_) => {}
        }
        out.push(judged);
    }
    if !keyed {
        let entries: Vec<CompiledEntry> = out
            .iter()
            .filter_map(|j| {
                let model = j.model.clone()?;
                let modified = std::fs::symlink_metadata(&j.path)
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH);
                Some(CompiledEntry {
                    dir: j.path.clone(),
                    model,
                    modified,
                })
            })
            .collect();
        let exists = |p: &Path| std::fs::symlink_metadata(p).is_ok();
        for (path, why) in compiled_orphans(&entries, &exists, "CoreML") {
            if let Some(j) = out.iter_mut().find(|j| j.path == path) {
                j.fate = Fate::Superseded(why);
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// The per-model entries of the cache folder `dir` (OpenVINO `openvino-*`, `tensorrt`, or
/// `coreml`'s compiled folders), by path, with their size and whether the sweep removes them as
/// superseded (model file gone or changed; in `coreml/`, gone or replaced by a newer folder).
/// Empty for a folder without subfolders. Reads only.
pub fn compiled_entries(dir: &Path) -> Vec<CompiledUse> {
    judge(dir, SystemTime::now())
        .into_iter()
        .filter(|j| j.dir)
        .map(|j| CompiledUse {
            bytes: crate::training::retry::disk_use(&j.path),
            superseded: matches!(j.fate, Fate::Superseded(_)),
            model: j.model,
            path: j.path,
        })
        .collect()
}

/// Remove what is provably superseded or unused in the OpenVINO and TensorRT cache folders under
/// `root`, even while a folder is in use (see the module docs): per-model subfolders whose model
/// file is gone or changed, or unused for [`UNUSED_DAYS`] by their own marker; empty ones a crash
/// left; flat-layout compiled files a day old. A subfolder this process compiled into is never
/// removed (its marker is refreshed instead); one with a [`MODEL_FILE`] but no marker gets one.
///
/// `sources` (the configured model files) is not needed: a model file that exists is never
/// "gone", configured or not (benchmark candidates and run models compile too).
pub fn sweep_compiled(root: &Path, sources: &[PathBuf], now: SystemTime, out: &mut Vec<Removed>) {
    let _ = sources;
    for folder in cache_dirs(root) {
        let keyed = folder
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(is_keyed_folder);
        if !keyed {
            continue;
        }
        for j in judge(&folder, now) {
            if j.dir && in_use(&j.path) {
                touch(&j.path);
                continue;
            }
            match &j.fate {
                Fate::Keep => {
                    if j.unmarked
                        && let Err(e) = write_marker(&j.path, now)
                    {
                        tracing::debug!("cache {}: starting its clock: {e}", j.path.display());
                    }
                }
                Fate::Superseded(why) | Fate::Expired(why) | Fate::Leftover(why) => {
                    super::remove_counted(&j.path, why, out);
                }
            }
        }
    }
}

/// Empty the cache folder `dir` (it stays, with its marker): every entry is rebuilt by the next
/// load. Loaded models keep running.
pub fn clear(dir: &Path, out: &mut Vec<Removed>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        if e.file_name() == LAST_USED_FILE {
            continue;
        }
        super::remove_counted(&e.path(), "cleared from the Storage card", out);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-caches-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn rewrite_names_parse_from_the_right() {
        assert_eq!(
            parse_rewrite_name("IPcam-general-3b338be21f271a29-v1.onnx"),
            Some(("IPcam-general", "3b338be21f271a29", 1))
        );
        assert_eq!(
            parse_rewrite_name("a-0123456789abcdef-v12.onnx").unwrap().2,
            12
        );
        for bad in [
            "model.onnx",
            "x-0123-v1.onnx",
            "-0123456789abcdef-v1.onnx",
            "x-0123456789abcdeg-v1.onnx",
            "x-0123456789abcdef-v.onnx",
            "x-0123456789abcdef-v1.bin",
        ] {
            assert_eq!(parse_rewrite_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_rewrite_is_an_orphan_only_when_no_model_file_matches() {
        let sources = vec![
            PathBuf::from("/m/IPcam-general.onnx"),
            PathBuf::from("/m/other.onnx"),
        ];
        let mut hash = |p: &Path| -> Option<String> {
            (p.ends_with("IPcam-general.onnx")).then(|| "3B338BE21F271A29".to_string())
        };
        let name = "IPcam-general-3b338be21f271a29-v1.onnx";
        assert_eq!(rewrite_orphan(name, &sources, 1, &mut hash), None);
        assert!(
            rewrite_orphan(name, &sources, 2, &mut hash)
                .unwrap()
                .contains("older rewriter")
        );
        let changed = "IPcam-general-0000000000000000-v1.onnx";
        assert!(
            rewrite_orphan(changed, &sources, 1, &mut hash)
                .unwrap()
                .contains("the file changed")
        );
        let gone = "deleted-0000000000000000-v1.onnx";
        assert!(
            rewrite_orphan(gone, &sources, 1, &mut hash)
                .unwrap()
                .contains("gone")
        );
        // A source that cannot be read keeps the rewrite: nothing is deleted on a guess.
        let mut unreadable = |_: &Path| -> Option<String> { None };
        assert_eq!(rewrite_orphan(changed, &sources, 1, &mut unreadable), None);
        // Not a rewrite name at all: never touched.
        assert_eq!(rewrite_orphan("notes.onnx", &[], 1, &mut hash), None);
    }

    #[test]
    fn compiled_models_go_when_their_file_is_gone_or_replaced() {
        let t = |s: u64| UNIX_EPOCH + Duration::from_secs(s);
        let entry = |dir: &str, model: &str, at: u64| CompiledEntry {
            dir: PathBuf::from(dir),
            model: PathBuf::from(model),
            modified: t(at),
        };
        let entries = vec![
            entry("/c/old", "/m/a.onnx", 10),
            entry("/c/new", "/m/a.onnx", 20),
            entry("/c/gone", "/m/deleted.onnx", 30),
            entry("/c/b", "/m/b.onnx", 5),
        ];
        let exists = |p: &Path| p != Path::new("/m/deleted.onnx");
        let mut out: Vec<PathBuf> = compiled_orphans(&entries, &exists, "CoreML")
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        out.sort();
        assert_eq!(out, [PathBuf::from("/c/gone"), PathBuf::from("/c/old")]);
    }

    #[test]
    fn unused_needs_a_marker_and_sixty_days() {
        let now = UNIX_EPOCH + Duration::from_secs(400 * 86_400);
        assert!(!is_unused(None, now), "no marker: never deleted on a guess");
        assert!(!is_unused(Some(now - DAY * 59), now));
        assert!(is_unused(Some(now - DAY * 60), now));
        assert!(!is_unused(Some(now + DAY), now), "a future time is not old");
    }

    #[test]
    fn the_sweep_deletes_only_old_cache_folders_it_made() {
        let root = tmp("sweep");
        let now = SystemTime::now();
        let old = root.join("openvino-gpu-f16");
        let fresh = root.join("openvino-cpu-f32");
        let unmarked = root.join("tensorrt");
        let foreign = root.join("photos");
        for d in [&old, &fresh, &unmarked, &foreign] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("blob"), vec![0u8; 100]).unwrap();
        }
        write_marker(&old, now - DAY * 61).unwrap();
        write_marker(&fresh, now - DAY * 2).unwrap();
        write_marker(&foreign, now - DAY * 900).unwrap();
        let old_bytes = super::super::size_of(&old);
        let mut out = Vec::new();
        sweep_unused(&root, now, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].path, old);
        assert_eq!(out[0].bytes, old_bytes);
        assert!(!old.exists());
        assert!(fresh.join("blob").is_file());
        assert!(foreign.join("blob").is_file(), "not a cache folder of ours");
        assert!(
            last_used(&unmarked).is_some(),
            "an unmarked folder gets a marker, starting its clock"
        );
        assert!(unmarked.join("blob").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_folder_this_process_uses_is_never_swept() {
        let root = tmp("inuse");
        let dir = root.join("coreml");
        std::fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, SystemTime::now() - DAY * 90).unwrap();
        touch(&dir);
        assert!(in_use(&dir));
        assert!(
            last_used(&dir).is_some_and(|t| age(SystemTime::now(), t) < DAY),
            "touch rewrote the old marker"
        );
        // Even when the marker says 90 days again (a clock jump), the sweep keeps it.
        write_marker(&dir, SystemTime::now() - DAY * 90).unwrap();
        let mut out = Vec::new();
        sweep_unused(&root, SystemTime::now(), &mut out);
        assert!(out.is_empty(), "{out:?}");
        assert!(dir.is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn coreml_orphans_are_swept_and_live_entries_kept() {
        let root = tmp("coreml");
        let models = root.join("models");
        let cache = root.join("cache").join("coreml");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        let live = models.join("live.onnx");
        std::fs::write(&live, b"live model").unwrap();
        let sha = file_sha16(&live).unwrap();
        let v = crate::backend::onnx_rewrite::REWRITER_VERSION;
        let rewrite_live = cache.join(format!("live-{sha}-v{v}.onnx"));
        let rewrite_gone = cache.join(format!("gone-0123456789abcdef-v{v}.onnx"));
        std::fs::write(&rewrite_live, b"r").unwrap();
        std::fs::write(&rewrite_gone, b"r").unwrap();
        // Compiled models: of the live file, of the deleted rewrite, of a deleted model.
        let mk = |key: &str, model: &Path| {
            let d = cache.join(key);
            std::fs::create_dir_all(d.join("0_dynamic_mlprogram")).unwrap();
            std::fs::write(d.join("model.txt"), model.display().to_string()).unwrap();
            d
        };
        let c_live = mk("1111", &live);
        let c_rewrite = mk("2222", &rewrite_gone);
        let c_gone = mk("3333", &models.join("deleted.onnx"));
        let unknown = cache.join("4444");
        std::fs::create_dir_all(&unknown).unwrap();
        let mut out = Vec::new();
        sweep_coreml(
            &cache,
            std::slice::from_ref(&live),
            SystemTime::now(),
            &mut out,
        );
        let mut gone: Vec<&PathBuf> = out.iter().map(|r| &r.path).collect();
        gone.sort();
        let mut want = vec![&rewrite_gone, &c_rewrite, &c_gone];
        want.sort();
        assert_eq!(gone, want);
        assert!(rewrite_live.is_file() && c_live.is_dir());
        assert!(unknown.is_dir(), "no model.txt: not judged");
        assert!(live.is_file(), "a model file is never touched");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_model_key_follows_path_size_and_time() {
        let t = UNIX_EPOCH + Duration::from_nanos(1_700_000_000_123_456_789);
        let p = Path::new("/m/a.onnx");
        let k = model_key(p, 100, t);
        assert_eq!(k.len(), 16);
        assert!(
            k.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );
        assert_eq!(model_key(p, 100, t), k, "deterministic");
        assert_ne!(model_key(p, 101, t), k, "size");
        assert_ne!(
            model_key(p, 100, t + Duration::from_nanos(1)),
            k,
            "mtime nanos"
        );
        assert_ne!(model_key(Path::new("/m/b.onnx"), 100, t), k, "path");
        // Before 1970 counts as 0, never a panic.
        let _ = model_key(p, 0, UNIX_EPOCH - Duration::from_secs(5));
    }

    #[test]
    fn subfolder_names_are_safe_and_parse_back() {
        let key = "0123456789abcdef";
        assert_eq!(
            model_dir_name(Path::new("/m/IPcam-general.onnx"), key),
            "IPcam-general-0123456789abcdef"
        );
        assert_eq!(
            model_dir_name(Path::new("/m/CON.x.onnx"), key),
            "CON_x-0123456789abcdef",
            "no dot: never a Windows device name"
        );
        assert_eq!(
            model_dir_name(Path::new("/m/a b/é.xml"), key),
            "_-0123456789abcdef"
        );
        assert_eq!(model_dir_name(Path::new(""), key), "model-0123456789abcdef");
        let long = model_dir_name(Path::new(&format!("/m/{}.onnx", "x".repeat(99))), key);
        assert_eq!(long.len(), MAX_STEM + 17);
        assert_eq!(
            parse_model_dir_name("IPcam-general-0123456789abcdef"),
            Some(("IPcam-general", key))
        );
        for bad in [
            "x-0123456789ABCDEF",
            "x-0123456789abcde",
            "-0123456789abcdef",
            "0123456789abcdef",
            "x-0123456789abcdeg",
        ] {
            assert_eq!(parse_model_dir_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn only_exact_flat_shapes_are_legacy() {
        for (folder, file) in [
            ("openvino-gpu-f16", "4574847747418676496.blob"),
            ("openvino-cpu-f32", "123.cl_cache"),
            (
                "tensorrt",
                "TensorrtExecutionProvider_TRTKernel_graph_main_1_0_0_sm86.engine",
            ),
            (
                "tensorrt",
                "TensorrtExecutionProvider_TRTKernel_graph_main_1_0_0_sm86.profile",
            ),
            ("tensorrt", "TensorrtExecutionProvider_cache_sm86.timing"),
        ] {
            assert!(is_flat_compiled(folder, file), "{folder}/{file}");
        }
        for (folder, file) in [
            ("openvino-gpu-f16", "model.blob"),
            ("openvino-gpu-f16", ".blob"),
            ("openvino-gpu-f16", "123.blob.tmp"),
            ("openvino-gpu-f16", LAST_USED_FILE),
            ("tensorrt", "123.blob"),
            ("tensorrt", "TensorrtExecutionProvider_.engine"),
            ("tensorrt", "mine.engine"),
            ("tensorrt", "TensorrtExecutionProvider_x.onnx"),
            ("coreml", "123.blob"),
            ("photos", "TensorrtExecutionProvider_x.engine"),
        ] {
            assert!(!is_flat_compiled(folder, file), "{folder}/{file}");
        }
    }

    #[test]
    fn a_model_gets_its_own_subfolder_with_its_path() {
        let root = tmp("subdir");
        let folder = root.join("openvino-cpu-f32");
        let model = root.join("m.onnx");
        std::fs::write(&model, b"model v1").unwrap();
        let sub = model_cache_dir(&folder, &model).unwrap();
        let (canonical, key) = model_identity(&model).unwrap();
        assert_eq!(sub, folder.join(format!("m-{key}")));
        assert_eq!(
            std::fs::read_to_string(sub.join(MODEL_FILE)).unwrap(),
            format!("{}\n", canonical.display())
        );
        assert!(in_use(&sub) && in_use(&folder), "both are this process's");
        assert!(last_used(&sub).is_some() && last_used(&folder).is_some());
        assert_eq!(model_cache_dir(&folder, &model).unwrap(), sub, "stable");
        // A retrain at the same path: a new subfolder, so an old engine is never found.
        std::fs::write(&model, b"model v2, longer").unwrap();
        let again = model_cache_dir(&folder, &model).unwrap();
        assert_ne!(again, sub);
        assert!(model_cache_dir(&folder, &root.join("missing.onnx")).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A subfolder as an older process left it: `model.txt` naming `model`, a blob, a fresh
    /// marker (not touched by this process, so not in use).
    fn compiled(folder: &Path, name: &str, model: &Path) -> PathBuf {
        let d = folder.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(MODEL_FILE), format!("{}\n", model.display())).unwrap();
        std::fs::write(d.join("123.blob"), vec![0u8; 64]).unwrap();
        write_marker(&d, SystemTime::now()).unwrap();
        d
    }

    fn removed(out: &[Removed]) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = out.iter().map(|r| r.path.clone()).collect();
        v.sort();
        v
    }

    #[test]
    fn the_compiled_sweep_removes_what_is_superseded_inside_folders_in_use() {
        let root = tmp("compiled");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let live = models.join("live.onnx");
        let other = models.join("unconfigured.onnx");
        let retrained = models.join("retrained.onnx");
        for m in [&live, &other, &retrained] {
            std::fs::write(m, b"weights").unwrap();
        }
        let key_of = |m: &Path| model_identity(m).unwrap();
        let (live_c, live_k) = key_of(&live);
        let (other_c, other_k) = key_of(&other);
        let (retrained_c, retrained_k) = key_of(&retrained);
        let gpu = root.join("openvino-gpu-f16");
        let trt = root.join("tensorrt");
        // The folder is in use (the daily model loads from it): its entries are judged anyway.
        std::fs::create_dir_all(&gpu).unwrap();
        touch(&gpu);
        let c_live = compiled(&gpu, &format!("live-{live_k}"), &live_c);
        let c_other = compiled(&trt, &format!("unconfigured-{other_k}"), &other_c);
        let c_retrained_now = compiled(&gpu, &format!("retrained-{retrained_k}"), &retrained_c);
        let c_retrained_old = compiled(&gpu, "retrained-0000000000000000", &retrained_c);
        let c_gone = compiled(
            &trt,
            "deleted-1111111111111111",
            &models.join("deleted.onnx"),
        );
        let unknown = gpu.join("somebody-else");
        std::fs::create_dir_all(&unknown).unwrap();
        std::fs::write(unknown.join("x.blob"), b"x").unwrap();
        let empty = gpu.join("empty-2222222222222222");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::write(
            empty.join(format!(".{MODEL_FILE}.{}.tmp", "a".repeat(32))),
            b"",
        )
        .unwrap();
        // This process compiled into it: kept even with its model gone (a mapped blob).
        let c_mine = compiled(&gpu, "mine-3333333333333333", &models.join("gone-too.onnx"));
        touch(&c_mine);
        // An unmarked live entry gets a marker.
        let c_unmarked = compiled(&trt, &format!("live-{live_k}"), &live_c);
        std::fs::remove_file(c_unmarked.join(LAST_USED_FILE)).unwrap();
        // Flat-layout files: fresh ones stay (an older binary may be writing them).
        std::fs::write(gpu.join("4574847747418676496.blob"), b"old").unwrap();
        std::fs::write(
            trt.join("TensorrtExecutionProvider_cache_sm86.timing"),
            b"t",
        )
        .unwrap();
        std::fs::write(gpu.join("notes.blob"), b"keep").unwrap();
        // Not a keyed folder: never judged here.
        let coreml = root.join("coreml");
        compiled(&coreml, "k", &models.join("deleted.onnx"));

        let sources = vec![live.clone()];
        let mut out = Vec::new();
        sweep_compiled(&root, &sources, SystemTime::now(), &mut out);
        let mut want = vec![c_retrained_old.clone(), c_gone.clone()];
        want.sort();
        assert_eq!(removed(&out), want, "{out:?}");
        for keep in [
            &c_live,
            &c_other,
            &c_retrained_now,
            &unknown,
            &empty,
            &c_mine,
        ] {
            assert!(keep.is_dir(), "{} removed", keep.display());
        }
        assert!(coreml.join("k").is_dir());
        assert!(
            last_used(&c_unmarked).is_some(),
            "the unmarked entry's clock started"
        );
        assert!(gpu.join("4574847747418676496.blob").is_file());
        for m in [&live, &other, &retrained] {
            assert!(m.is_file(), "a model file is never touched");
        }

        // Two hours on: the crash-made empty subfolder goes; the flat files are not a day old.
        let mut out = Vec::new();
        sweep_compiled(&root, &sources, SystemTime::now() + 2 * TMP_AGE, &mut out);
        assert_eq!(removed(&out), std::slice::from_ref(&empty), "{out:?}");
        // Two days on: the flat files go, `notes.blob` (not the exact shape) stays.
        let mut out = Vec::new();
        sweep_compiled(&root, &sources, SystemTime::now() + 2 * DAY, &mut out);
        let mut want = vec![
            gpu.join("4574847747418676496.blob"),
            trt.join("TensorrtExecutionProvider_cache_sm86.timing"),
        ];
        want.sort();
        assert_eq!(removed(&out), want, "{out:?}");
        assert!(gpu.join("notes.blob").is_file());
        // 61 days on, every entry not used since by its own marker goes, in use or not around it;
        // this process's own entry stays.
        let mut out = Vec::new();
        sweep_compiled(&root, &sources, SystemTime::now() + 61 * DAY, &mut out);
        let mut want = vec![
            c_live.clone(),
            c_other.clone(),
            c_retrained_now.clone(),
            c_unmarked,
        ];
        want.sort();
        assert_eq!(removed(&out), want, "{out:?}");
        assert!(c_mine.is_dir() && unknown.is_dir() && gpu.is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn compiled_entries_report_each_model_and_what_is_superseded() {
        let root = tmp("entries");
        let model = root.join("a.onnx");
        std::fs::write(&model, b"weights").unwrap();
        let (canonical, key) = model_identity(&model).unwrap();
        let folder = root.join("tensorrt");
        let now = compiled(&folder, &format!("a-{key}"), &canonical);
        let old = compiled(&folder, "a-0000000000000000", &canonical);
        let gone = compiled(&folder, "b-1111111111111111", &root.join("b.onnx"));
        let bare = folder.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(folder.join("TensorrtExecutionProvider_x.engine"), b"e").unwrap();
        let got = compiled_entries(&folder);
        let row = |p: &Path| got.iter().find(|u| u.path == p).unwrap().clone();
        assert_eq!(got.len(), 4, "subfolders only: {got:?}");
        assert!(!row(&now).superseded && row(&now).model.as_deref() == Some(canonical.as_path()));
        assert!(row(&now).bytes >= 64);
        assert!(row(&old).superseded && row(&gone).superseded);
        assert!(!row(&bare).superseded && row(&bare).model.is_none());
        // CoreML's compiled folders: gone or replaced by a newer folder of the same file.
        let coreml = root.join("coreml");
        let c_gone = compiled(&coreml, "k1", &root.join("b.onnx"));
        let c_live = compiled(&coreml, "k2", &canonical);
        let got = compiled_entries(&coreml);
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|u| u.path == c_gone && u.superseded));
        assert!(got.iter().any(|u| u.path == c_live && !u.superseded));
        assert!(compiled_entries(&root.join("none")).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
