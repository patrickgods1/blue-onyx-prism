//! On-disk layout of the dataset folder (`docs/DATASET.md`):
//! `<dir>/<scene>/<camera>/<YYYY-MM-DD>/<ts>_<request_id>.{jpg,json}`, one `index.jsonl` per day
//! folder, and `<dir>/review-queue.jsonl`. Names, the relative-path check every served or
//! queued path goes through ([`parse_rel`], [`resolve`], and [`Walker`] for a walk over many
//! frames), the summary rebuilt from the index files ([`Summary::scan`]) and the review queue
//! ([`append_review`]).
//!
//! The JSONL files survive a crash or a full disk: [`append_line`] ends a torn last line before
//! writing, and [`read_lines`] costs a damaged line only that line, and counts it.

use super::record::{Bucket, IndexLine, Record};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Index file of a day folder.
pub const INDEX_FILE: &str = "index.jsonl";
/// The review queue the review page consumes.
pub const REVIEW_QUEUE_FILE: &str = "review-queue.jsonl";
/// Version of a review-queue line.
pub const REVIEW_VERSION: u32 = 1;
/// Scene folders.
pub const SCENES: [&str; 3] = ["day", "night", "ir"];
/// Newest frames per bucket kept in the [`Summary`] (thumbnails).
pub const NEWEST_PER_BUCKET: usize = 8;
/// Key of [`Summary::by_label`] for a frame whose reply had no detection.
pub const NO_DETECTION: &str = "(none)";
/// Longest camera folder name.
const MAX_CAMERA_LEN: usize = 64;
/// Longest request id in a file name.
const MAX_ID_LEN: usize = 32;

/// The camera folder of a request: the camera part of `camera_key` (`<model>|cam:<camera>`),
/// else `frame-<w>x<h>`; lowercase ASCII letters, digits, `-` and `_` only.
pub fn camera_dir(camera_key: &str, w: u32, h: u32) -> String {
    let raw = match camera_key.split_once("|cam:") {
        Some((_, cam)) => cam.to_string(),
        None => format!("frame-{w}x{h}"),
    };
    let mut s: String = raw
        .chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(MAX_CAMERA_LEN)
        .collect();
    if s.trim_matches('_').is_empty() {
        s = format!("frame-{w}x{h}");
    }
    s
}

/// A request id safe for a file name (`[A-Za-z0-9_-]`, at most 32 characters).
pub fn safe_id(id: &str) -> String {
    let s: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(MAX_ID_LEN)
        .collect();
    if s.is_empty() { "req".into() } else { s }
}

/// A frame's path relative to the dataset folder, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rel {
    pub scene: String,
    pub camera: String,
    pub date: String,
    /// File stem (`<ts>_<request_id>`).
    pub stem: String,
}

impl Rel {
    pub fn day_dir(&self) -> String {
        format!("{}/{}/{}", self.scene, self.camera, self.date)
    }
    pub fn jpg(&self) -> String {
        format!("{}/{}.jpg", self.day_dir(), self.stem)
    }
    pub fn json(&self) -> String {
        format!("{}/{}.json", self.day_dir(), self.stem)
    }
    pub fn to_path(&self, dir: &Path, ext: &str) -> PathBuf {
        dir.join(&self.scene)
            .join(&self.camera)
            .join(&self.date)
            .join(format!("{}.{ext}", self.stem))
    }
}

fn name_ok(s: &str, max: usize) -> bool {
    !s.is_empty()
        && s.len() <= max
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Whether `s` can be a camera folder ([`camera_dir`]'s alphabet and length).
pub fn camera_ok(s: &str) -> bool {
    name_ok(s, MAX_CAMERA_LEN)
}

fn date_ok(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        })
}

fn stem_ok(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Parse `<scene>/<camera>/<YYYY-MM-DD>/<stem>.jpg` (or `.json`): exactly four components of
/// the allowed characters, a known scene, no `.`/`..`, no separators other than `/`. Anything
/// else is None, so a path from a client can never leave the dataset folder.
pub fn parse_rel(rel: &str) -> Option<Rel> {
    if rel.contains('\\') || rel.contains('\0') {
        return None;
    }
    let parts: Vec<&str> = rel.split('/').collect();
    let [scene, camera, date, file] = parts.as_slice() else {
        return None;
    };
    let stem = file
        .strip_suffix(".jpg")
        .or_else(|| file.strip_suffix(".json"))?;
    (SCENES.contains(scene) && name_ok(camera, MAX_CAMERA_LEN) && date_ok(date) && stem_ok(stem))
        .then(|| Rel {
            scene: (*scene).to_string(),
            camera: (*camera).to_string(),
            date: (*date).to_string(),
            stem: stem.to_string(),
        })
}

/// The file `rel` names under `dir` (`ext` = `jpg` or `json`), only when it parses, exists as
/// a regular file (not a symlink) and its canonical path is inside the canonical `dir`.
pub fn resolve(dir: &Path, rel: &str, ext: &str) -> Option<PathBuf> {
    let r = parse_rel(rel)?;
    let p = r.to_path(dir, ext);
    let meta = std::fs::symlink_metadata(&p).ok()?;
    if !meta.file_type().is_file() {
        return None;
    }
    let (root, real) = (dir.canonicalize().ok()?, p.canonicalize().ok()?);
    real.starts_with(&root).then_some(real)
}

/// [`resolve`] for one walk over many frames: the same answers, with the root canonicalized once
/// and each `<scene>/<camera>/<date>` folder once, then per file only a look at the leaf
/// (`symlink_metadata`: a regular file, no symlink). The real path is the canonical day folder
/// plus the leaf's name, which is what canonicalizing a regular file's path gives.
///
/// A day folder is checked once: one that is missing, or that a symlink takes outside the root,
/// refuses all its frames for one failed call; one a symlink keeps inside the root works, as in
/// [`resolve`]. Scope it to a walk, not longer: the answers are as of the first look. [`Clone`] it
/// for the threads of a walk, so the root is not looked at again.
#[derive(Debug, Clone)]
pub struct Walker<'a> {
    dir: &'a Path,
    /// The canonical root, None when `dir` has none (then nothing resolves).
    root: Option<PathBuf>,
    /// `<scene>/<camera>/<date>` -> its canonical folder, None when it refuses.
    days: HashMap<String, Option<PathBuf>>,
    /// Day folders looked at on disk.
    #[cfg(test)]
    day_lookups: usize,
}

impl<'a> Walker<'a> {
    pub fn new(dir: &'a Path) -> Self {
        Self {
            dir,
            root: dir.canonicalize().ok(),
            days: HashMap::new(),
            #[cfg(test)]
            day_lookups: 0,
        }
    }

    /// The canonical folder of `r`'s day, when it is inside the root.
    fn day(&mut self, r: &Rel) -> Option<&Path> {
        let root = self.root.as_ref()?;
        let dir = self.dir;
        #[cfg(test)]
        let lookups = &mut self.day_lookups;
        self.days
            .entry(r.day_dir())
            .or_insert_with(|| {
                #[cfg(test)]
                {
                    *lookups += 1;
                }
                let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
                let real = day.canonicalize().ok()?;
                real.starts_with(root).then_some(real)
            })
            .as_deref()
    }

    /// The file of `r` with extension `ext`: [`resolve`] of an already parsed path.
    pub fn file(&mut self, r: &Rel, ext: &str) -> Option<PathBuf> {
        let p = self.day(r)?.join(format!("{}.{ext}", r.stem));
        let meta = std::fs::symlink_metadata(&p).ok()?;
        meta.file_type().is_file().then_some(p)
    }

    /// [`resolve`].
    pub fn resolve(&mut self, rel: &str, ext: &str) -> Option<PathBuf> {
        self.file(&parse_rel(rel)?, ext)
    }
}

/// A frame the Collect tab shows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameRef {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub rel: String,
    pub timestamp: String,
    pub scene: String,
    pub camera: String,
    pub model: String,
    pub labels: Vec<String>,
    pub buckets: Vec<Bucket>,
}

impl FrameRef {
    pub fn of(line: &IndexLine, date: &str) -> Self {
        Self {
            rel: format!("{}/{}/{date}/{}.jpg", line.scene, line.camera, line.id),
            timestamp: line.timestamp.clone(),
            scene: line.scene.clone(),
            camera: line.camera.clone(),
            model: line.model.clone(),
            labels: line.labels.clone(),
            buckets: line.buckets.clone(),
        }
    }
}

/// Per-camera totals.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CameraTotals {
    pub frames: u64,
    pub bytes: u64,
    /// Timestamp of the newest frame.
    pub newest: String,
}

/// What the dataset folder holds (from the index files; the collector keeps it current).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Summary {
    pub frames: u64,
    /// Image + sidecar bytes of every indexed frame.
    pub bytes: u64,
    /// scene -> bucket -> frames (a frame counts once in each of its buckets).
    pub by_scene: BTreeMap<String, BTreeMap<Bucket, u64>>,
    pub by_camera: BTreeMap<String, CameraTotals>,
    /// label -> scene -> frames: the reply's prediction labels at collection time (a frame
    /// counts once per distinct label; a frame with none under [`NO_DETECTION`]).
    pub by_label: BTreeMap<String, BTreeMap<String, u64>>,
    /// Newest frames per bucket, newest first.
    pub newest: BTreeMap<Bucket, VecDeque<FrameRef>>,
    /// Index lines a scan could not read ([`read_lines`]): frames it does not count.
    #[serde(skip_serializing_if = "is_zero")]
    pub damaged_lines: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl Summary {
    /// Count one indexed frame of day folder `date`.
    pub fn add(&mut self, line: &IndexLine, date: &str) {
        self.frames += 1;
        self.bytes = self.bytes.saturating_add(line.bytes);
        let scene = self.by_scene.entry(line.scene.clone()).or_default();
        for b in &line.buckets {
            *scene.entry(*b).or_default() += 1;
        }
        let cam = self.by_camera.entry(line.camera.clone()).or_default();
        cam.frames += 1;
        cam.bytes = cam.bytes.saturating_add(line.bytes);
        if line.timestamp > cam.newest {
            cam.newest = line.timestamp.clone();
        }
        let labels: BTreeSet<&str> = line.labels.iter().map(String::as_str).collect();
        let labels = if labels.is_empty() {
            BTreeSet::from([NO_DETECTION])
        } else {
            labels
        };
        for l in labels {
            *self
                .by_label
                .entry(l.to_string())
                .or_default()
                .entry(line.scene.clone())
                .or_default() += 1;
        }
        let f = FrameRef::of(line, date);
        for b in &line.buckets {
            let list = self.newest.entry(*b).or_default();
            let at = list
                .iter()
                .position(|x| x.timestamp < f.timestamp)
                .unwrap_or(list.len());
            if at < NEWEST_PER_BUCKET {
                list.insert(at, f.clone());
                list.truncate(NEWEST_PER_BUCKET);
            }
        }
    }

    /// Frames of `scene` in `bucket`.
    pub fn count(&self, scene: &str, bucket: Bucket) -> u64 {
        self.by_scene
            .get(scene)
            .and_then(|m| m.get(&bucket))
            .copied()
            .unwrap_or(0)
    }

    /// Rebuild from every `<scene>/<camera>/<date>/index.jsonl` under `dir` (a missing folder
    /// is an empty dataset). Lines that do not parse are skipped and counted
    /// ([`Self::damaged_lines`]); the newest frames listed must still have their image.
    pub fn scan(dir: &Path) -> Self {
        let mut s = Self::default();
        for day in day_dirs(dir) {
            let date = day
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let index = read_lines::<IndexLine>(&day.join(INDEX_FILE));
            s.damaged_lines += index.damaged as u64;
            for line in index.lines {
                s.add(&line, &date);
            }
        }
        for list in s.newest.values_mut() {
            list.retain(|f| resolve(dir, &f.rel, "jpg").is_some());
        }
        if s.damaged_lines > 0 {
            tracing::warn!(
                "{}: {} index line(s) could not be read (a crash or a full disk while writing?) \
                 and are skipped; those frames stay on disk but are not counted",
                dir.display(),
                s.damaged_lines
            );
        }
        s
    }
}

/// Subfolders of `p` that are real directories (symlinks skipped), sorted; `p` itself in
/// `unreadable` when it exists and cannot be listed.
fn subdirs(p: &Path, unreadable: &mut Vec<PathBuf>) -> Vec<PathBuf> {
    let rd = match std::fs::read_dir(p) {
        Ok(rd) => rd,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                unreadable.push(p.to_path_buf());
            }
            return Vec::new();
        }
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .collect();
    out.sort();
    out
}

/// Every `<dir>/<scene>/<camera>/<date>` folder with valid names, oldest date first.
pub fn day_dirs(dir: &Path) -> Vec<PathBuf> {
    day_dirs_checked(dir).0
}

/// [`day_dirs`], and the scene and camera folders it could not list, so a caller that must not
/// shrink silently can say so.
pub fn day_dirs_checked(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    days_of(dir, None)
}

/// [`day_dirs_checked`] of one camera: what filtering its answer by camera folder gives, without
/// listing the other cameras. The camera folders are listed, never joined from `camera`: a name
/// from a client reaches no path, and a symlinked camera folder is skipped like any other.
pub fn camera_day_dirs(dir: &Path, camera: &str) -> (Vec<PathBuf>, Vec<PathBuf>) {
    days_of(dir, Some(camera))
}

fn days_of(dir: &Path, only: Option<&str>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut out = Vec::new();
    let mut unreadable = Vec::new();
    for scene in SCENES {
        for cam in subdirs(&dir.join(scene), &mut unreadable) {
            let cam_ok = cam
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| name_ok(n, MAX_CAMERA_LEN) && only.is_none_or(|c| c == n));
            if !cam_ok {
                continue;
            }
            for day in subdirs(&cam, &mut unreadable) {
                if day
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(date_ok)
                {
                    out.push(day);
                }
            }
        }
    }
    out.sort_by(|a, b| a.file_name().cmp(&b.file_name()).then_with(|| a.cmp(b)));
    (out, unreadable)
}

/// What a JSONL file holds: the lines that parse, and how many did not.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonLines<T> {
    pub lines: Vec<T>,
    /// Lines that do not parse (torn by a crash, a damaged byte, an unknown format).
    pub damaged: usize,
    /// Why the file could not be read at all (a missing file is no failure: nothing in it).
    pub failed: Option<String>,
}

/// The lines of the JSONL file `path`. Bytes are split on newlines and decoded lossily, so a
/// damaged byte costs its own line, not the file; blank lines are not lines.
pub fn read_lines<T: DeserializeOwned>(path: &Path) -> JsonLines<T> {
    let mut out = JsonLines {
        lines: Vec::new(),
        damaged: 0,
        failed: None,
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                out.failed = Some(format!("{}: {e}", path.display()));
            }
            return out;
        }
    };
    for line in bytes.split(|b| *b == b'\n') {
        let text = String::from_utf8_lossy(line);
        if text.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&text) {
            Ok(v) => out.lines.push(v),
            Err(_) => out.damaged += 1,
        }
    }
    out
}

/// The parsable lines of an index file (none when it is missing or cannot be read).
pub fn read_index(path: &Path) -> Vec<IndexLine> {
    read_lines(path).lines
}

/// Rewrite `day`'s index without the frames whose image is gone (after retention), atomically.
/// Returns the lines kept.
pub fn rewrite_index(day: &Path) -> Result<usize> {
    let path = day.join(INDEX_FILE);
    let index = read_lines::<IndexLine>(&path);
    // Rewriting from an index that could not be read would drop every frame's line.
    if let Some(e) = index.failed {
        bail!("reading {e}");
    }
    let lines = index.lines;
    let kept: Vec<&IndexLine> = lines
        .iter()
        .filter(|l| day.join(format!("{}.jpg", l.id)).is_file())
        .collect();
    if kept.len() == lines.len() && path.exists() {
        return Ok(kept.len());
    }
    if kept.is_empty() {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("removing {}", path.display())),
        }
        return Ok(0);
    }
    let mut text = String::new();
    for l in &kept {
        text.push_str(&serde_json::to_string(l)?);
        text.push('\n');
    }
    write_atomic(&path, text.as_bytes())?;
    Ok(kept.len())
}

/// Write `bytes` to `path` through a temporary file in the same folder and a rename, so a
/// reader never sees a half-written file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
        drop(f);
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Whether the file `f` is empty or ends in a newline (a crash or a full disk mid-append leaves
/// it without one).
fn ends_a_line(f: &mut std::fs::File) -> std::io::Result<bool> {
    if f.metadata()?.len() == 0 {
        return Ok(true);
    }
    f.seek(SeekFrom::End(-1))?;
    let mut last = [0u8; 1];
    f.read_exact(&mut last)?;
    Ok(last == *b"\n")
}

/// Append one line (with its newline) to `path` in a single write, so concurrent appenders do not
/// interleave. After a torn last line the write starts with a newline: the torn line stays a bad
/// line of its own and this one is whole.
pub fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    let mut buf = Vec::with_capacity(line.len() + 2);
    if !ends_a_line(&mut f)? {
        buf.push(b'\n');
    }
    buf.extend_from_slice(line.as_bytes());
    buf.push(b'\n');
    f.write_all(&buf)
}

/// One line of `review-queue.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct ReviewLine {
    pub version: u32,
    /// The frame's image, relative to the dataset folder.
    pub frame: String,
    /// Its sidecar.
    pub sidecar: String,
    pub id: String,
    pub buckets: Vec<Bucket>,
    pub scene: String,
    pub camera: String,
    /// When it was queued (UTC, RFC 3339).
    pub queued_at: String,
}

/// Serializes the appends of [`append_review`] within the process.
static REVIEW_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Queue the frames `rels` (image paths relative to `dir`) for review: every one must parse
/// and exist with its sidecar; frames already queued are skipped. Returns how many were added.
pub fn append_review(dir: &Path, rels: &[String]) -> Result<usize> {
    let mut lines = Vec::new();
    for rel in rels {
        let Some(r) = parse_rel(rel).filter(|_| rel.ends_with(".jpg")) else {
            bail!("'{rel}' is not a frame of the dataset");
        };
        if resolve(dir, &r.jpg(), "jpg").is_none() {
            bail!("{}: the frame is gone (deleted by retention?)", r.jpg());
        }
        let Some(sidecar) = resolve(dir, &r.json(), "json") else {
            bail!("{}: the sidecar is missing", r.json());
        };
        let rec: Record = serde_json::from_slice(&std::fs::read(&sidecar)?)
            .with_context(|| format!("reading {}", sidecar.display()))?;
        lines.push(ReviewLine {
            version: REVIEW_VERSION,
            frame: r.jpg(),
            sidecar: r.json(),
            id: rec.id,
            buckets: rec.buckets,
            scene: r.scene.clone(),
            camera: r.camera.clone(),
            queued_at: super::utc_rfc3339(std::time::SystemTime::now()),
        });
    }
    let _held = REVIEW_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = dir.join(REVIEW_QUEUE_FILE);
    let queued: BTreeSet<String> = read_review(&path).into_iter().map(|l| l.frame).collect();
    let mut added = 0;
    let mut seen = BTreeSet::new();
    for l in lines {
        if queued.contains(&l.frame) || !seen.insert(l.frame.clone()) {
            continue;
        }
        append_line(&path, &serde_json::to_string(&l)?)
            .with_context(|| format!("writing {}", path.display()))?;
        added += 1;
    }
    Ok(added)
}

/// The parsable lines of the review queue.
pub fn read_review(path: &Path) -> Vec<ReviewLine> {
    read_lines(path).lines
}

/// Lines of `dir`'s review queue that could not be read (with `all_days`, of its index files too):
/// frames the queue cannot list.
pub fn damaged_lines(dir: &Path, all_days: bool) -> u64 {
    let queue = read_lines::<ReviewLine>(&dir.join(REVIEW_QUEUE_FILE)).damaged as u64;
    if !all_days {
        return queue;
    }
    day_dirs(dir).iter().fold(queue, |n, day| {
        n + read_lines::<IndexLine>(&day.join(INDEX_FILE)).damaged as u64
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn camera_folder_names() {
        assert_eq!(camera_dir("m|cam:driveway", 10, 10), "driveway");
        assert_eq!(camera_dir("m|cam:Back Yard/../x", 10, 10), "back_yard____x");
        assert_eq!(camera_dir("m|640x480", 640, 480), "frame-640x480");
        assert_eq!(camera_dir("m|cam:...", 4, 3), "frame-4x3");
        assert_eq!(
            camera_dir(&format!("m|cam:{}", "a".repeat(100)), 1, 1).len(),
            64
        );
        assert_eq!(safe_id("ab/../c d"), "abcd");
        assert_eq!(safe_id("../"), "req");
    }

    #[test]
    fn relative_paths_are_strict() {
        let ok = "ir/driveway/2026-10-22/20261022T010203004Z_abcd1234.jpg";
        let r = parse_rel(ok).unwrap();
        assert_eq!(r.jpg(), ok);
        assert_eq!(r.json(), ok.replace(".jpg", ".json"));
        for bad in [
            "",
            "ir/driveway/2026-10-22",
            "dusk/driveway/2026-10-22/a.jpg",
            "ir/../2026-10-22/a.jpg",
            "ir/driveway/../a.jpg",
            "ir/driveway/2026-10-22/../../../etc/passwd",
            "ir/drive way/2026-10-22/a.jpg",
            "ir/Driveway/2026-10-22/a.jpg",
            "ir/driveway/2026-1-22/a.jpg",
            "ir/driveway/2026-10-22/a.png",
            "ir/driveway/2026-10-22/.jpg",
            "ir/driveway/2026-10-22/a.b.jpg",
            "/ir/driveway/2026-10-22/a.jpg",
            "ir\\driveway\\2026-10-22\\a.jpg",
            "ir/driveway/2026-10-22/sub/a.jpg",
            "ir/driveway/2026-10-22/a.jpg\0",
        ] {
            assert!(parse_rel(bad).is_none(), "{bad:?} accepted");
        }
    }

    #[test]
    fn index_lines_round_trip_through_the_summary() {
        let line = |id: &str, ts: &str, b: Vec<Bucket>| IndexLine {
            id: id.into(),
            timestamp: ts.into(),
            scene: "ir".into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: b,
            labels: vec!["person".into()],
            bytes: 100,
        };
        let mut s = Summary::default();
        for i in 0..10 {
            s.add(
                &line(
                    &format!("f{i}"),
                    &format!("2026-10-22T00:00:0{i}Z"),
                    vec![Bucket::Positive, Bucket::Suppressed],
                ),
                "2026-10-22",
            );
        }
        assert_eq!(s.frames, 10);
        assert_eq!(s.bytes, 1000);
        assert_eq!(s.count("ir", Bucket::Positive), 10);
        assert_eq!(s.count("day", Bucket::Positive), 0);
        let newest = &s.newest[&Bucket::Suppressed];
        assert_eq!(newest.len(), NEWEST_PER_BUCKET);
        assert_eq!(newest[0].rel, "ir/cam/2026-10-22/f9.jpg");
        assert_eq!(s.by_camera["cam"].newest, "2026-10-22T00:00:09Z");
        assert_eq!(s.by_label["person"]["ir"], 10);
    }

    fn index_line(id: &str, scene: &str, labels: &[&str]) -> IndexLine {
        IndexLine {
            id: id.into(),
            timestamp: "2026-10-22T00:00:00Z".into(),
            scene: scene.into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: vec![Bucket::Positive],
            labels: labels.iter().map(|l| l.to_string()).collect(),
            bytes: 10,
        }
    }

    #[test]
    fn labels_count_once_per_frame_and_empty_frames_count_as_none() {
        let mut s = Summary::default();
        s.add(
            &index_line("a", "day", &["person", "person", "car"]),
            "2026-10-22",
        );
        s.add(&index_line("b", "night", &["person"]), "2026-10-22");
        s.add(&index_line("c", "ir", &[]), "2026-10-22");
        assert_eq!(
            s.by_label["person"]["day"], 1,
            "deduplicated within a frame"
        );
        assert_eq!(s.by_label["person"]["night"], 1);
        assert_eq!(s.by_label["car"]["day"], 1);
        assert_eq!(s.by_label[NO_DETECTION]["ir"], 1);
        assert_eq!(s.by_label.len(), 3);
    }

    #[test]
    fn scan_rebuilds_the_label_counts_from_the_index() {
        let dir = std::env::temp_dir().join(format!("bop-layout-{}", uuid::Uuid::new_v4()));
        let day = dir.join("night/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        for l in [
            index_line("a", "night", &["dog", "dog"]),
            index_line("b", "night", &[]),
        ] {
            append_line(&day.join(INDEX_FILE), &serde_json::to_string(&l).unwrap()).unwrap();
        }
        let s = Summary::scan(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(s.frames, 2);
        assert_eq!(s.by_label["dog"]["night"], 1);
        assert_eq!(s.by_label[NO_DETECTION]["night"], 1);
    }

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bop-layout-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn review_line(frame: &str) -> ReviewLine {
        ReviewLine {
            version: REVIEW_VERSION,
            frame: frame.into(),
            sidecar: frame.replace(".jpg", ".json"),
            id: "x".into(),
            buckets: vec![Bucket::Positive],
            scene: "ir".into(),
            camera: "cam".into(),
            queued_at: "2026-10-22T00:00:00Z".into(),
        }
    }

    #[test]
    fn a_torn_last_line_does_not_swallow_the_next_good_one() {
        let dir = tmp("torn");
        // An index: crash or ENOSPC mid-append left half a line and no newline.
        let index = dir.join(INDEX_FILE);
        let good = serde_json::to_string(&index_line("a", "ir", &[])).unwrap();
        std::fs::write(&index, format!("{good}\n{{\"id\":\"tor")).unwrap();
        let next = serde_json::to_string(&index_line("b", "ir", &[])).unwrap();
        append_line(&index, &next).unwrap();
        let ids: Vec<String> = read_index(&index).into_iter().map(|l| l.id).collect();
        assert_eq!(ids, ["a", "b"]);
        // The review queue the same way.
        let queue = dir.join(REVIEW_QUEUE_FILE);
        let good = serde_json::to_string(&review_line("ir/c/2026-10-22/a.jpg")).unwrap();
        std::fs::write(&queue, format!("{good}\n{{\"version\":1,\"fr")).unwrap();
        let next = serde_json::to_string(&review_line("ir/c/2026-10-22/b.jpg")).unwrap();
        append_line(&queue, &next).unwrap();
        let frames: Vec<String> = read_review(&queue).into_iter().map(|l| l.frame).collect();
        assert_eq!(frames, ["ir/c/2026-10-22/a.jpg", "ir/c/2026-10-22/b.jpg"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_invalid_byte_costs_only_its_line() {
        let dir = tmp("utf8");
        let index = dir.join(INDEX_FILE);
        let mut bytes = Vec::new();
        for id in ["a", "b"] {
            bytes.extend_from_slice(
                serde_json::to_string(&index_line(id, "ir", &[]))
                    .unwrap()
                    .as_bytes(),
            );
            bytes.push(b'\n');
            bytes.extend_from_slice(b"\xff\xfe not json\n");
        }
        std::fs::write(&index, &bytes).unwrap();
        assert_eq!(read_index(&index).len(), 2);
        let read = read_lines::<IndexLine>(&index);
        assert_eq!((read.lines.len(), read.damaged), (2, 2));
        assert_eq!(read.failed, None);
        let queue = dir.join(REVIEW_QUEUE_FILE);
        let mut bytes = serde_json::to_vec(&review_line("ir/c/2026-10-22/a.jpg")).unwrap();
        bytes.extend_from_slice(b"\n\xff\n");
        std::fs::write(&queue, &bytes).unwrap();
        assert_eq!(read_review(&queue).len(), 1);
        assert_eq!(read_lines::<ReviewLine>(&queue).damaged, 1);
        // Damage inside a string is kept, replaced: the line still names its frame.
        let line = serde_json::to_string(&review_line("ir/c/2026-10-22/a.jpg")).unwrap();
        let bytes = line.replace("\"x\"", "\"\u{1}\"").into_bytes();
        let at = bytes.iter().position(|b| *b == 1).unwrap();
        let mut damaged = bytes.clone();
        damaged[at] = 0xff;
        std::fs::write(&queue, &damaged).unwrap();
        let read = read_lines::<ReviewLine>(&queue);
        assert_eq!((read.lines.len(), read.damaged), (1, 0));
        assert_eq!(read.lines[0].id, "\u{fffd}");
        // A missing file is empty, with nothing to report.
        let none = read_lines::<ReviewLine>(&dir.join("none.jsonl"));
        assert_eq!((none.lines.len(), none.damaged, none.failed), (0, 0, None));
        // A file that cannot be read is said to be so.
        let folder = read_lines::<ReviewLine>(&dir);
        assert!(folder.failed.is_some());
        assert!(
            rewrite_index(&dir.join("x")).is_ok(),
            "no index, nothing to do"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_scan_counts_the_lines_it_could_not_read() {
        let dir = tmp("scan");
        let day = dir.join("night/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        let mut bytes = serde_json::to_vec(&index_line("a", "night", &[])).unwrap();
        bytes.extend_from_slice(b"\n\xff\xfe\n{\"id\":\"torn");
        std::fs::write(day.join(INDEX_FILE), &bytes).unwrap();
        let s = Summary::scan(&dir);
        assert_eq!((s.frames, s.damaged_lines), (1, 2));
        assert_eq!(
            serde_json::to_value(&s).unwrap()["damaged_lines"],
            serde_json::json!(2)
        );
        assert!(
            serde_json::to_value(Summary::default()).unwrap()["damaged_lines"].is_null(),
            "an intact dataset's summary is unchanged"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_index_is_not_rewritten_as_empty() {
        use std::os::unix::fs::PermissionsExt;
        let day = tmp("eacces");
        let index = day.join(INDEX_FILE);
        append_line(
            &index,
            &serde_json::to_string(&index_line("a", "ir", &[])).unwrap(),
        )
        .unwrap();
        std::fs::set_permissions(&index, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read(&index).is_err() {
            assert!(rewrite_index(&day).is_err());
            assert!(index.exists(), "kept");
        }
        let _ = std::fs::remove_dir_all(&day);
    }

    #[test]
    fn appending_to_an_empty_file_or_after_a_newline_adds_no_blank_line() {
        let dir = tmp("plain");
        let p = dir.join("x.jsonl");
        append_line(&p, "1").unwrap();
        append_line(&p, "2").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"1\n2\n");
        // A torn tail gets its newline in the same write.
        std::fs::write(&p, b"1\n2").unwrap();
        append_line(&p, "3").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"1\n2\n3\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_index_is_not_deleted_as_empty() {
        let day = tmp("rewrite");
        std::fs::write(day.join("a.jpg"), b"x").unwrap();
        let mut bytes = serde_json::to_vec(&index_line("a", "ir", &[])).unwrap();
        bytes.extend_from_slice(b"\n\xff\n");
        std::fs::write(day.join(INDEX_FILE), &bytes).unwrap();
        assert_eq!(rewrite_index(&day).unwrap(), 1);
        assert_eq!(read_index(&day.join(INDEX_FILE)).len(), 1);
        let _ = std::fs::remove_dir_all(&day);
    }

    #[test]
    fn a_walker_looks_at_a_day_folder_once() {
        let dir = tmp("walker-days");
        let rel = |date: &str, stem: &str| format!("ir/cam/{date}/{stem}.jpg");
        let day = dir.join("ir/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        for stem in ["a", "b"] {
            std::fs::write(day.join(format!("{stem}.jpg")), b"x").unwrap();
        }
        let mut w = Walker::new(&dir);
        assert!(w.resolve(&rel("2026-10-22", "a"), "jpg").is_some());
        assert!(w.resolve(&rel("2026-10-22", "b"), "jpg").is_some());
        assert!(w.resolve(&rel("2026-10-22", "gone"), "jpg").is_none());
        assert_eq!(w.day_lookups, 1, "a folder that is there");
        // A folder that is not there costs one failed look, not one per frame.
        let mut w = Walker::new(&dir);
        for i in 0..50 {
            assert!(
                w.resolve(&rel("2026-10-23", &format!("f{i}")), "jpg")
                    .is_none()
            );
        }
        assert_eq!(w.day_lookups, 1, "a folder that is gone");
        // The threads of a walk share what the root cost; each keeps its own days.
        let clone = w.clone();
        assert_eq!(clone.root, w.root);
        assert!(w.resolve(&rel("2026-10-22", "a"), "jpg").is_some());
        assert_eq!(w.day_lookups, 2, "another folder");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
