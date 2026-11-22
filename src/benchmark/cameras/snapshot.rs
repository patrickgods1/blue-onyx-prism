//! The ground-truth snapshot of "Your cameras": `benchmark-gt.json` beside `benchmark.json`, the
//! labels a run scored the set against, per frame: its file name (the export's flat name), its
//! identity (`<scene>/<camera>/<date>/<stem>.jpg`), its size, its boxes (class, `ignore`) and its
//! label file's revision; plus the revision of every reviewed frame at the time, so a frame
//! reviewed later counts as new.
//!
//! The labels of the set live here only: `benchmark.json` keeps the set's images (names, tags,
//! sizes) without their boxes ([`strip`], on every save), and every load fills them in from this
//! file ([`overlay`]). So everything that re-scores saved results (Compare, the confidence search,
//! a regrade, threshold advice) scores against the snapshot, and results stay put while labels
//! change. [`changes`] counts what changed since; [`super::rescore`] scores the stored
//! predictions against the current labels and writes a new snapshot.
//!
//! It is user data (it holds labels): written through a temporary file and a rename, never
//! deleted by the storage sweeper, never in a diagnostics bundle, git-ignored.

use super::super::images::ImageSet;
use super::super::metrics::GtBox;
use super::super::report::BenchmarkResults;
use super::{ID, Source};
use crate::collect::labels::SizeFloor;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The file name, beside `benchmark.json`.
pub const FILE: &str = "benchmark-gt.json";
/// The format written; a later one is refused.
pub const VERSION: u32 = 1;

/// One scored frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GtFrame {
    /// The set's image name (the export's flat name).
    pub file: String,
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub frame: String,
    pub width: u32,
    pub height: u32,
    /// The label file's revision ([`crate::collect::labels::revision`]) the boxes are of.
    pub revision: String,
    pub boxes: Vec<GtBox>,
}

/// `benchmark-gt.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GtSnapshot {
    pub version: u32,
    /// The set it is of ([`ID`]).
    pub set: String,
    /// The folder under review.
    pub dir: PathBuf,
    /// When it was taken (UTC, RFC 3339).
    pub created: String,
    /// The scored classes (`collect.classes`).
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
    /// The frames a run scored, in the set's order.
    pub frames: Vec<GtFrame>,
    /// Every reviewed frame of the selection then: file name -> label revision.
    #[serde(default)]
    pub reviewed: BTreeMap<String, String>,
}

/// The snapshot beside the results file `results`.
pub fn path_beside(results: &Path) -> PathBuf {
    results.with_file_name(FILE)
}

fn now() -> String {
    crate::collect::utc_rfc3339(std::time::SystemTime::now())
}

impl GtSnapshot {
    /// The snapshot of a run's `set` (after its image limit), the revisions of its frames being
    /// those of `reviewed` (the whole selection).
    pub fn of(src: &Source, set: &ImageSet, reviewed: &BTreeMap<String, String>) -> Self {
        Self {
            version: VERSION,
            set: ID.to_string(),
            dir: src.dir.clone(),
            created: now(),
            classes: src.classes.clone(),
            size_floor: src.size_floor,
            frames: set
                .images
                .iter()
                .map(|i| GtFrame {
                    file: i.file.clone(),
                    frame: crate::training::autolabel::frame_of_file(&i.file).unwrap_or_default(),
                    width: i.width,
                    height: i.height,
                    revision: reviewed.get(&i.file).cloned().unwrap_or_default(),
                    boxes: i.gt.clone().unwrap_or_default(),
                })
                .collect(),
            reviewed: reviewed.clone(),
        }
    }

    /// The file at `path`; None when there is none. A later format, or a file that does not
    /// parse, is an error (never moved aside: it is user data).
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        #[derive(Deserialize)]
        struct Head {
            version: u32,
        }
        let head: Head = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if head.version > VERSION {
            bail!(
                "{} is of a newer format ({}) than this build reads ({VERSION}); update Blue \
                 Onyx Prism, or run the benchmark on {} again",
                path.display(),
                head.version,
                super::TITLE
            );
        }
        serde_json::from_slice(&bytes)
            .map(Some)
            .with_context(|| format!("parsing {}", path.display()))
    }

    /// Write it through a temporary file in the same folder, flushed, then renamed over `path`:
    /// a reader sees the old snapshot or the new one, never part of one.
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        let text = serde_json::to_vec_pretty(self)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        let write = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&text)?;
            f.sync_all()
        };
        if let Err(e) = write() {
            std::fs::remove_file(&tmp).ok();
            return Err(e).with_context(|| format!("writing {}", tmp.display()));
        }
        std::fs::rename(&tmp, path).map_err(|e| {
            std::fs::remove_file(&tmp).ok();
            anyhow::anyhow!("writing {}: {e}", path.display())
        })
    }

    /// The scored frames by file name.
    pub fn by_file(&self) -> HashMap<&str, &GtFrame> {
        self.frames.iter().map(|f| (f.file.as_str(), f)).collect()
    }
}

/// `results` as `benchmark.json` stores it: the boxes of the cameras set's images left out (they
/// are in the snapshot). Borrowed when there is no such set.
pub fn strip(results: &BenchmarkResults) -> std::borrow::Cow<'_, BenchmarkResults> {
    if !results.image_sets.iter().any(|s| s.id == ID) {
        return std::borrow::Cow::Borrowed(results);
    }
    let mut r = results.clone();
    for s in r.image_sets.iter_mut().filter(|s| s.id == ID) {
        for i in &mut s.images {
            i.objects = None;
        }
    }
    std::borrow::Cow::Owned(r)
}

/// Fill the boxes of the cameras set of `results` (loaded from `results_path`) from the snapshot
/// beside it; an image the snapshot does not hold has none (it is not scored). A warning when the
/// snapshot is missing or does not read: the set is then not scored until the next run or
/// re-score.
pub fn overlay(results: &mut BenchmarkResults, results_path: &Path) -> Option<String> {
    let set = results.image_sets.iter_mut().find(|s| s.id == ID)?;
    let path = path_beside(results_path);
    let (snap, warning) = match GtSnapshot::load(&path) {
        Ok(Some(s)) => (Some(s), None),
        Ok(None) => (
            None,
            Some(format!(
                "{} is missing, so {} is not scored until it is benchmarked again",
                path.display(),
                super::TITLE
            )),
        ),
        Err(e) => (
            None,
            Some(format!(
                "{e:#}; {} is not scored until it is benchmarked again",
                super::TITLE
            )),
        ),
    };
    let frames = snap.as_ref().map(GtSnapshot::by_file).unwrap_or_default();
    for i in &mut set.images {
        i.objects = frames.get(i.file.as_str()).map(|f| f.boxes.clone());
    }
    warning
}

/// What changed in the labels since a snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Changes {
    /// `changed + left + joined`.
    pub frames: usize,
    /// Scored frames whose label file has another revision now.
    pub changed: usize,
    /// Scored frames no longer in the reviewed set (reopened, skipped, deleted).
    pub left: usize,
    /// Reviewed frames the snapshot did not know (reviewed since): a run scores them, a re-score
    /// cannot (no predictions).
    pub joined: usize,
}

/// What changed between `snap` and `current` (file name -> revision of the reviewed frames now).
pub fn changes(snap: &GtSnapshot, current: &BTreeMap<String, String>) -> Changes {
    let mut c = Changes::default();
    for f in &snap.frames {
        match current.get(&f.file) {
            None => c.left += 1,
            Some(r) if *r != f.revision => c.changed += 1,
            Some(_) => {}
        }
    }
    c.joined = current
        .keys()
        .filter(|k| !snap.reviewed.contains_key(*k))
        .count();
    c.frames = c.changed + c.left + c.joined;
    c
}
