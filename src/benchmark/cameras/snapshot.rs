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
//! **One unit with the results.** Every writer saves the snapshot first, then `benchmark.json`
//! with the snapshot's identity in the cameras set ([`SnapshotId`]: its `created` and the
//! SHA-256 of its bytes), all under the results lock ([`crate::benchmark::results_io::lock`]).
//! A load whose snapshot is not that one (a crash between the two writes, a hand edit) warns and
//! marks the set stale ([`crate::benchmark::report::ImageSetInfo::labels_stale`]; the page
//! offers a Re-score) instead of scoring silently. A run's snapshot keeps the frames of the
//! earlier runs still in the results ([`merged_with`]), so a later run of fewer frames does not
//! take their labels away.
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

/// Which snapshot a results file was saved with: the snapshot's `created` and the SHA-256 of
/// its file's bytes ([`ImageSetInfo::snapshot`](crate::benchmark::report::ImageSetInfo)).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotId {
    pub created: String,
    pub sha256: String,
}

impl SnapshotId {
    fn of(snap: &GtSnapshot, bytes: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let sha256 = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Self {
            created: snap.created.clone(),
            sha256,
        }
    }
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
    /// parse, is an error, and so is one that cannot be read (an
    /// `report::Unreadable` in the chain, after
    /// [`crate::json_aside::read_retrying`]'s tries). A load never moves the file; a run's save
    /// sets a damaged or later-format one aside first and refuses to write over an unreadable one
    /// ([`super::write_snapshot`], QA-051).
    pub fn load(path: &Path) -> Result<Option<Self>> {
        Ok(Self::load_with_id(path)?.map(|(s, _)| s))
    }

    /// [`Self::load`] with the file's identity.
    pub fn load_with_id(path: &Path) -> Result<Option<(Self, SnapshotId)>> {
        let Some(bytes) = crate::json_aside::read_retrying(path)
            .map_err(|e| crate::benchmark::report::unreadable(path, e))?
        else {
            return Ok(None);
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
        let snap: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        let id = SnapshotId::of(&snap, &bytes);
        Ok(Some((snap, id)))
    }

    /// Write it through a temporary file of its own in the same folder, flushed, then renamed
    /// over `path` ([`crate::benchmark::results_io::write_atomic`]): a reader sees the old
    /// snapshot or the new one, never part of one. Returns its identity, for the results saved
    /// with it.
    pub fn save(&self, path: &Path) -> Result<SnapshotId> {
        let text = serde_json::to_vec_pretty(self)?;
        crate::benchmark::results_io::write_atomic(path, &text)?;
        Ok(SnapshotId::of(self, &text))
    }

    /// The scored frames by file name.
    pub fn by_file(&self) -> HashMap<&str, &GtFrame> {
        self.frames.iter().map(|f| (f.file.as_str(), f)).collect()
    }

    /// This snapshot (a run's) with the frames of `old` (the snapshot on disk) that a run of
    /// `prev` (the results the run's save merges into) still scores and this one does not hold,
    /// so a later run of fewer frames leaves the earlier runs their labels. Nothing is added
    /// from a snapshot of another folder or class list (its runs are of another set).
    #[must_use]
    pub fn merged_with(
        mut self,
        old: Option<&GtSnapshot>,
        prev: Option<&BenchmarkResults>,
    ) -> Self {
        let (Some(old), Some(prev)) = (old, prev) else {
            return self;
        };
        if old.dir != self.dir || old.classes != self.classes {
            return self;
        }
        let scored: std::collections::HashSet<&str> = prev
            .models
            .iter()
            .flat_map(|m| m.devices.iter())
            .filter_map(|d| d.run.as_ref())
            .flat_map(|r| r.per_image.iter())
            .filter(|i| i.set == ID)
            .map(|i| i.file.as_str())
            .collect();
        let have: std::collections::HashSet<String> =
            self.frames.iter().map(|f| f.file.clone()).collect();
        for f in &old.frames {
            if scored.contains(f.file.as_str()) && !have.contains(&f.file) {
                self.frames.push(f.clone());
            }
        }
        self
    }
}

/// Record `id` (the snapshot just written beside it) in the cameras set of `results`, which is
/// then no longer stale.
pub fn stamp(results: &mut BenchmarkResults, id: &SnapshotId) {
    for s in results.image_sets.iter_mut().filter(|s| s.id == ID) {
        s.snapshot = Some(id.clone());
        s.labels_stale = None;
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

/// Why the snapshot `found` is not the one `recorded` in the results (None: it is, or the
/// results predate the identity).
pub fn mismatch(recorded: Option<&SnapshotId>, found: &SnapshotId) -> Option<String> {
    let recorded = recorded?;
    (recorded != found).then(|| {
        format!(
            "the labels of {} ({FILE}, taken {}) are not the ones these results were saved with \
             (taken {}): the results were saved before or after another snapshot (an interrupted \
             save?). Re-score {} against the current labels, or run the benchmark on it again",
            super::TITLE,
            found.created,
            recorded.created,
            super::TITLE
        )
    })
}

/// Fill the boxes of the cameras set of `results` (loaded from `results_path`) from the snapshot
/// beside it; an image the snapshot does not hold has none (it is not scored). A warning when the
/// snapshot is missing or does not read (the set is then not scored until the next run or
/// re-score), and when it is not the one the results were saved with ([`mismatch`]: the set is
/// marked [`crate::benchmark::report::ImageSetInfo::labels_stale`], scored against the file
/// found until it is re-scored).
pub fn overlay(results: &mut BenchmarkResults, results_path: &Path) -> Option<String> {
    let set = results.image_sets.iter_mut().find(|s| s.id == ID)?;
    let path = path_beside(results_path);
    let (snap, warning) = match GtSnapshot::load_with_id(&path) {
        Ok(Some((s, id))) => {
            let stale = mismatch(set.snapshot.as_ref(), &id);
            set.labels_stale.clone_from(&stale);
            (Some(s), stale)
        }
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
    /// Scored frames the walk could not read this time (changed while read, a label file that
    /// does not parse, a lost image or sidecar, an unreadable folder): neither changed nor
    /// left, as far as anyone can tell; a re-score keeps them as they were.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unknown: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// What changed between `snap` and `current` (file name -> revision of the reviewed frames now).
pub fn changes(snap: &GtSnapshot, current: &BTreeMap<String, String>) -> Changes {
    changes_with(snap, current, &|_| false)
}

/// [`changes`], with `unknown(frame)` true for a scored frame the walk could not read (see
/// [`Changes::unknown`]).
pub fn changes_with(
    snap: &GtSnapshot,
    current: &BTreeMap<String, String>,
    unknown: &dyn Fn(&GtFrame) -> bool,
) -> Changes {
    let mut c = Changes::default();
    for f in &snap.frames {
        match current.get(&f.file) {
            None if unknown(f) => c.unknown += 1,
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
