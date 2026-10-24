//! On-disk layout of the dataset folder (`docs/DATASET.md`):
//! `<dir>/<scene>/<camera>/<YYYY-MM-DD>/<ts>_<request_id>.{jpg,json}`, one `index.jsonl` per day
//! folder, and `<dir>/review-queue.jsonl`. Names, the relative-path check every served or
//! queued path goes through ([`parse_rel`], [`resolve`]), the summary rebuilt from the index
//! files ([`Summary::scan`]) and the review queue ([`append_review`]).

use super::record::{Bucket, IndexLine, Record};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write;
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

/// A frame the Dataset page shows.
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
    /// is an empty dataset). Lines that do not parse are skipped; the newest frames listed must
    /// still have their image.
    pub fn scan(dir: &Path) -> Self {
        let mut s = Self::default();
        for day in day_dirs(dir) {
            let date = day
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            for line in read_index(&day.join(INDEX_FILE)) {
                s.add(&line, &date);
            }
        }
        for list in s.newest.values_mut() {
            list.retain(|f| resolve(dir, &f.rel, "jpg").is_some());
        }
        s
    }
}

/// Subfolders of `p` that are real directories (symlinks skipped), sorted.
fn subdirs(p: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(p) else {
        return Vec::new();
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
    let mut out = Vec::new();
    for scene in SCENES {
        for cam in subdirs(&dir.join(scene)) {
            let cam_ok = cam
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| name_ok(n, MAX_CAMERA_LEN));
            if !cam_ok {
                continue;
            }
            for day in subdirs(&cam) {
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
    out
}

/// The parsable lines of an index file (none when it is missing).
pub fn read_index(path: &Path) -> Vec<IndexLine> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Rewrite `day`'s index without the frames whose image is gone (after retention), atomically.
/// Returns the lines kept.
pub fn rewrite_index(day: &Path) -> Result<usize> {
    let path = day.join(INDEX_FILE);
    let lines = read_index(&path);
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

/// Append one line (with its newline) to `path` in a single write.
pub fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut buf = Vec::with_capacity(line.len() + 1);
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
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
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
}
