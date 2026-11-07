//! The model caches under `cache_dir`: OpenVINO's compiled blobs (`openvino-<device>-<prec>/`),
//! ONNX Runtime's CoreML models and graph rewrites (`coreml/`) and TensorRT engines (`tensorrt/`).
//! Everything here is rebuilt by the next load (slower once), so the sweep may delete it.
//!
//! - [`touch`]: a backend loading from a cache folder records the use in `<folder>/.last-used`
//!   (unix seconds), written at most once a day per folder, and the folder counts as in use by this
//!   process until it exits ([`in_use`]).
//! - [`sweep_unused`]: a folder not used for [`UNUSED_DAYS`] goes. A folder with no marker yet
//!   (made before markers existed) gets one, starting its clock: nothing is deleted on a guess.
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
/// A `.tmp` file of an interrupted rewrite older than this is a leftover.
const TMP_AGE: Duration = Duration::from_secs(60 * 60);

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
/// several for one file all but the newest.
pub fn compiled_orphans(
    entries: &[CompiledEntry],
    exists: &dyn Fn(&Path) -> bool,
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
                    "compiled CoreML model of {}, which is gone",
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
                    "compiled CoreML model of an earlier {} (a newer one replaced it)",
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
    for (path, why) in compiled_orphans(&compiled, &exists) {
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
        let mut out: Vec<PathBuf> = compiled_orphans(&entries, &exists)
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
}
