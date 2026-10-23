//! Review labels of a collected frame: `<stem>.labels.json` beside its sidecar (schema
//! [`LABELS_VERSION`], `docs/DATASET.md` "Review labels"). Pure parts: the pre-fill from the
//! sidecar ([`prefill`]), the save-time normalisation ([`normalize`]: clamping, pending boxes
//! resolved, class check) and the merge of later proposals such as auto-labeler boxes
//! ([`merge_proposals`]). IO: [`Frame::resolve`] (every frame reference goes through
//! [`super::layout::parse_rel`] / [`super::layout::resolve`]), [`read`] and [`write`]
//! (atomic). The review queue is in [`queue`], the YOLO / COCO / benchmark exporters in
//! [`export`] and the X-AnyLabeling interop in [`xany`].

pub mod export;
pub mod queue;
pub mod xany;

use super::layout::{self, Rel};
use super::record::Record;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Version of the label file ([`LabelFile::version`]). Files of a newer version are skipped.
pub const LABELS_VERSION: u32 = 1;
/// Suffix of a label file after the frame's stem (`<stem>.labels.json`).
pub const LABELS_SUFFIX: &str = ".labels.json";
/// Most boxes in one label file.
pub const MAX_BOXES: usize = 500;
/// Largest label file read (bytes).
pub const MAX_LABEL_FILE_BYTES: u64 = 512 * 1024;
/// Largest sidecar read (bytes).
pub const MAX_SIDECAR_BYTES: u64 = 4 * 1024 * 1024;
/// Longest box label.
pub const MAX_LABEL_LEN: usize = 64;
/// Smallest box side (pixels) kept at save.
pub const MIN_BOX_PX: f32 = 1.0;
/// IoU at or above which a proposal matches an existing box ([`merge_proposals`]).
pub const MATCH_IOU: f32 = 0.5;
/// IoU at or above which two pre-filled boxes of one label are the same box ([`prefill`]).
pub const DUP_IOU: f32 = 0.95;

/// Where a frame is in the review.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Not reviewed yet (pre-filled, or proposals merged after a review).
    #[default]
    Unreviewed,
    /// Every box decided; the frame's labels are complete (exported).
    Reviewed,
    /// The reviewer skipped it (never exported).
    Skipped,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Unreviewed => "unreviewed",
            Status::Reviewed => "reviewed",
            Status::Skipped => "skipped",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "unreviewed" => Some(Status::Unreviewed),
            "reviewed" => Some(Status::Reviewed),
            "skipped" => Some(Status::Skipped),
            _ => None,
        }
    }
}

/// Who proposed a box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// The serving model (or its verifier), from the sidecar.
    Model,
    /// The offline auto-labeler (training env), merged through [`merge_proposals`].
    Auto,
    /// Drawn by the reviewer.
    Human,
}

/// The reviewer's decision on a box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxState {
    /// Not decided yet (pre-fill, proposals). A reviewed file has none: see [`resolve_pending`].
    Pending,
    /// A real object (model or auto box).
    Accepted,
    /// Wrong: kept as hard-negative evidence, never a training label.
    Rejected,
    /// Drawn by the reviewer (a missed object).
    Added,
}

/// Which part of the sidecar a model box came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The reply's predictions.
    Prediction,
    /// Removed by a rule or the verifier (`reason` says which).
    Suppressed,
    /// Below the request's threshold, within the near-threshold band.
    BelowThreshold,
    /// Found by the frame-mode verifier, matching no other box.
    VerifyFound,
}

/// A box of a label file. `bbox` is `[x0, y0, x1, y1]` in pixels of the original frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelBox {
    pub label: String,
    pub bbox: [f32; 4],
    pub source: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    pub state: BoxState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
    /// Suppression reason (`night_threshold`, `verify`, ...) or other provenance note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The label the model or auto-labeler gave, when the reviewer relabeled the box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_label: Option<String>,
}

impl LabelBox {
    /// A training label: accepted or added.
    pub fn is_training(&self) -> bool {
        matches!(self.state, BoxState::Accepted | BoxState::Added)
    }

    fn proposal(d: &super::record::Det, origin: Origin, reason: Option<String>) -> Self {
        Self {
            label: d.label.clone(),
            bbox: d.bbox,
            source: Source::Model,
            score: Some(d.score),
            state: BoxState::Pending,
            origin: Some(origin),
            reason,
            model_label: None,
        }
    }
}

/// `<stem>.labels.json` (see the module doc).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelFile {
    #[serde(default = "labels_version")]
    pub version: u32,
    #[serde(default)]
    pub status: Status,
    /// The reviewer confirmed the frame holds no object of any class (a hard negative).
    #[serde(default)]
    pub empty_confirmed: bool,
    #[serde(default)]
    pub boxes: Vec<LabelBox>,
    /// When the frame was reviewed or skipped (UTC, RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reviewed_at: Option<String>,
    /// [`classes_version`] of the class list the review used.
    #[serde(default)]
    pub classes_version: String,
}

fn labels_version() -> u32 {
    LABELS_VERSION
}

impl Default for LabelFile {
    fn default() -> Self {
        Self {
            version: LABELS_VERSION,
            status: Status::Unreviewed,
            empty_confirmed: false,
            boxes: Vec::new(),
            reviewed_at: None,
            classes_version: String::new(),
        }
    }
}

impl LabelFile {
    /// The training labels (accepted and added boxes).
    pub fn training(&self) -> impl Iterator<Item = &LabelBox> {
        self.boxes.iter().filter(|b| b.is_training())
    }

    /// Rejected model / auto boxes (hard-negative evidence).
    pub fn rejected(&self) -> impl Iterator<Item = &LabelBox> {
        self.boxes
            .iter()
            .filter(|b| b.state == BoxState::Rejected && b.source != Source::Human)
    }
}

/// Short fingerprint of a class list (first 16 hex digits of the SHA-256 of the names joined
/// by `\n`): a label file reviewed with another list shows it.
pub fn classes_version(classes: &[String]) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(classes.join("\n").as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Intersection over union of two `[x0, y0, x1, y1]` boxes (0 for empty boxes).
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// `b` ordered (`x0 <= x1`, `y0 <= y1`) and clamped to the `w` x `h` frame.
pub fn clamp_box(b: [f32; 4], w: u32, h: u32) -> [f32; 4] {
    let (w, h) = (w as f32, h as f32);
    let (x0, x1) = (b[0].min(b[2]), b[0].max(b[2]));
    let (y0, y1) = (b[1].min(b[3]), b[1].max(b[3]));
    [
        x0.clamp(0.0, w),
        y0.clamp(0.0, h),
        x1.clamp(0.0, w),
        y1.clamp(0.0, h),
    ]
}

fn push_unique(out: &mut Vec<LabelBox>, b: LabelBox, dup_iou: f32) {
    let dup = out
        .iter()
        .any(|o| o.label == b.label && iou(&o.bbox, &b.bbox) >= dup_iou);
    if !dup {
        out.push(b);
    }
}

/// The unreviewed label file a frame starts with: every box of the sidecar as a pending model
/// box, in this order: the reply's predictions (accepted unless rejected), the suppressed boxes
/// with their reason (rejected unless restored: a wrongly suppressed real object is a miss),
/// the below-threshold boxes and the frame-mode verifier's own finds (both rejected unless
/// accepted). A box of the same label at IoU >= [`DUP_IOU`] of an earlier one is left out (a
/// verifier find at >= [`MATCH_IOU`]).
pub fn prefill(rec: &Record) -> LabelFile {
    let mut boxes = Vec::new();
    for d in &rec.predictions {
        push_unique(
            &mut boxes,
            LabelBox::proposal(d, Origin::Prediction, None),
            DUP_IOU,
        );
    }
    for s in &rec.suppressed {
        let d = super::record::Det {
            label: s.label.clone(),
            score: s.score,
            bbox: s.bbox,
        };
        push_unique(
            &mut boxes,
            LabelBox::proposal(&d, Origin::Suppressed, Some(s.reason.clone())),
            DUP_IOU,
        );
    }
    for d in &rec.below_threshold {
        push_unique(
            &mut boxes,
            LabelBox::proposal(d, Origin::BelowThreshold, None),
            DUP_IOU,
        );
    }
    if let Some(v) = &rec.verify {
        for d in &v.found {
            let reason = Some(format!("verifier {}", v.verifier));
            push_unique(
                &mut boxes,
                LabelBox::proposal(d, Origin::VerifyFound, reason),
                MATCH_IOU,
            );
        }
    }
    for b in &mut boxes {
        b.bbox = clamp_box(b.bbox, rec.width, rec.height);
    }
    LabelFile {
        boxes,
        ..LabelFile::default()
    }
}

/// What a pending box becomes when its frame is saved as reviewed: a human box is added; a
/// prediction or an auto-labeler box is accepted (the reviewer rejects false boxes rather than
/// confirming each true one); a suppressed, below-threshold or verifier-found box is rejected
/// (it was not reported, so restoring it is an explicit decision).
pub fn resolve_pending(b: &LabelBox) -> BoxState {
    match (b.state, b.source, b.origin) {
        (BoxState::Pending, Source::Human, _) => BoxState::Added,
        (BoxState::Pending, Source::Auto, _) => BoxState::Accepted,
        (BoxState::Pending, Source::Model, Some(Origin::Prediction)) => BoxState::Accepted,
        (BoxState::Pending, Source::Model, _) => BoxState::Rejected,
        (s, _, _) => s,
    }
}

fn label_ok(l: &str) -> bool {
    !l.trim().is_empty() && l.chars().count() <= MAX_LABEL_LEN && !l.chars().any(char::is_control)
}

/// Check and complete a label file a client sends for a `w` x `h` frame, at time `now`
/// (RFC 3339):
///
/// - at most [`MAX_BOXES`] boxes; finite coordinates, ordered and clamped to the frame, both
///   sides at least [`MIN_BOX_PX`]; labels of 1 to [`MAX_LABEL_LEN`] characters; scores in 0..=1;
/// - a human box is `added` (a rejected one is dropped: it is simply deleted);
/// - `reviewed`: pending boxes resolved ([`resolve_pending`]), every training label must be in
///   `classes`, and a frame confirmed empty must have none; `reviewed_at` = `now`;
/// - `skipped`: `reviewed_at` = `now`; `unreviewed`: no `reviewed_at`;
/// - `version` and `classes_version` set.
pub fn normalize(
    mut f: LabelFile,
    classes: &[String],
    w: u32,
    h: u32,
    now: &str,
) -> Result<LabelFile> {
    if f.version > LABELS_VERSION {
        bail!(
            "label file version {} is newer than this build understands ({LABELS_VERSION}); \
             update Blue Onyx Prism",
            f.version
        );
    }
    if f.boxes.len() > MAX_BOXES {
        bail!("{} boxes; at most {MAX_BOXES} per frame", f.boxes.len());
    }
    let mut boxes = Vec::with_capacity(f.boxes.len());
    for (i, mut b) in std::mem::take(&mut f.boxes).into_iter().enumerate() {
        let n = i + 1;
        if !b.bbox.iter().all(|v| v.is_finite()) {
            bail!("box {n}: the coordinates are not finite numbers");
        }
        if !label_ok(&b.label) {
            bail!(
                "box {n}: '{}' is not a valid label (1 to {MAX_LABEL_LEN} characters)",
                b.label.escape_debug()
            );
        }
        if let Some(s) = b.score
            && !(s.is_finite() && (0.0..=1.0).contains(&s))
        {
            bail!("box {n}: score {s} is not between 0 and 1");
        }
        if b.reason.as_ref().is_some_and(|r| r.chars().count() > 200) {
            bail!("box {n}: the reason is longer than 200 characters");
        }
        b.bbox = clamp_box(b.bbox, w, h);
        if b.bbox[2] - b.bbox[0] < MIN_BOX_PX || b.bbox[3] - b.bbox[1] < MIN_BOX_PX {
            bail!(
                "box {n} ({}): smaller than {MIN_BOX_PX} px inside the {w}x{h} frame",
                b.label
            );
        }
        if b.source == Source::Human {
            match b.state {
                BoxState::Rejected => continue,
                _ => b.state = BoxState::Added,
            }
        } else if b.state == BoxState::Added {
            b.state = BoxState::Accepted;
        }
        if b.model_label.as_deref() == Some(b.label.as_str()) {
            b.model_label = None;
        }
        boxes.push(b);
    }
    f.boxes = boxes;
    f.version = LABELS_VERSION;
    f.classes_version = classes_version(classes);
    match f.status {
        Status::Reviewed => {
            for b in &mut f.boxes {
                b.state = resolve_pending(b);
            }
            let unknown: Vec<&str> = f
                .training()
                .filter(|b| !classes.contains(&b.label))
                .map(|b| b.label.as_str())
                .collect();
            if !unknown.is_empty() {
                bail!(
                    "label(s) {} are not in the class list ({}); relabel or reject those boxes, \
                     or add the class to collect.classes",
                    unknown.join(", "),
                    classes.join(", ")
                );
            }
            let n = f.training().count();
            if f.empty_confirmed && n > 0 {
                bail!(
                    "the frame is marked confirmed empty but has {n} accepted box(es); reject \
                     them or clear 'confirmed empty'"
                );
            }
            f.reviewed_at = Some(now.to_string());
        }
        Status::Skipped => f.reviewed_at = Some(now.to_string()),
        Status::Unreviewed => f.reviewed_at = None,
    }
    Ok(f)
}

/// What [`merge_proposals`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeReport {
    /// Proposals added as pending boxes.
    pub added: usize,
    /// Proposals already covered by a box.
    pub dropped: usize,
    /// The file went back to `unreviewed`.
    pub reopened: bool,
}

/// Merge boxes proposed after the pre-fill (the auto-labeler's `source: auto` boxes, a newer
/// model's detections) into `f`. Rule, per proposal, against the existing box of highest IoU
/// at or above [`MATCH_IOU`] (any label):
///
/// - no such box: the proposal is added as `pending`;
/// - same label: dropped; the existing box and any decision on it win (a rejected box stays
///   rejected, a pending one keeps its state);
/// - another label on a decided box (accepted, added or rejected): added as `pending`, a
///   disagreement for the reviewer; on a pending box: dropped.
///
/// A reviewed (or skipped) file that gains a box goes back to `unreviewed`, keeping its other
/// decisions. Proposals are stored with state `pending` whatever they carry.
pub fn merge_proposals(f: &mut LabelFile, proposals: Vec<LabelBox>) -> MergeReport {
    let mut r = MergeReport::default();
    let existing = f.boxes.len();
    for mut p in proposals {
        p.state = BoxState::Pending;
        let best = f
            .boxes
            .iter()
            .take(existing)
            .map(|b| (iou(&b.bbox, &p.bbox), b))
            .filter(|(v, _)| *v >= MATCH_IOU)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, b)| b);
        let add = match best {
            None => true,
            Some(b) if b.label == p.label => false,
            Some(b) => b.state != BoxState::Pending,
        };
        let dup_of_added = f
            .boxes
            .iter()
            .skip(existing)
            .any(|b| b.label == p.label && iou(&b.bbox, &p.bbox) >= MATCH_IOU);
        if add && !dup_of_added {
            f.boxes.push(p);
            r.added += 1;
        } else {
            r.dropped += 1;
        }
    }
    if r.added > 0 && f.status != Status::Unreviewed {
        f.status = Status::Unreviewed;
        r.reopened = true;
    }
    r
}

/// A frame of the dataset, resolved: every path is inside the dataset folder.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub rel: Rel,
    /// Canonical image path.
    pub jpg: PathBuf,
    /// Canonical sidecar path.
    pub sidecar: PathBuf,
    /// `<stem>.labels.json` in the same folder (may not exist).
    pub labels: PathBuf,
}

impl Frame {
    /// The frame `rel` (`<scene>/<camera>/<date>/<stem>.jpg`) names under `dir`, when it parses,
    /// its image and sidecar exist as regular files inside `dir` ([`layout::resolve`]) and its
    /// label file, if present, is a regular file (no symlink).
    pub fn resolve(dir: &Path, rel: &str) -> Option<Self> {
        let r = layout::parse_rel(rel).filter(|_| rel.ends_with(".jpg"))?;
        let jpg = layout::resolve(dir, &r.jpg(), "jpg")?;
        let sidecar = layout::resolve(dir, &r.json(), "json")?;
        let labels = jpg.parent()?.join(format!("{}{LABELS_SUFFIX}", r.stem));
        if let Ok(m) = std::fs::symlink_metadata(&labels)
            && !m.file_type().is_file()
        {
            return None;
        }
        Some(Self {
            rel: r,
            jpg,
            sidecar,
            labels,
        })
    }

    /// The sidecar.
    pub fn record(&self) -> Result<Record> {
        let bytes = read_capped(&self.sidecar, MAX_SIDECAR_BYTES)?;
        let rec: Record = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", self.sidecar.display()))?;
        if rec.version > super::record::RECORD_VERSION {
            bail!(
                "{}: sidecar version {} is newer than this build understands",
                self.sidecar.display(),
                rec.version
            );
        }
        Ok(rec)
    }

    /// The saved label file, if any.
    pub fn labels(&self) -> Result<Option<LabelFile>> {
        read(&self.labels)
    }
}

/// Read at most `max` bytes of `path` (an error when it is larger).
pub fn read_capped(path: &Path, max: u64) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut buf = Vec::new();
    f.take(max.saturating_add(1))
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {}", path.display()))?;
    if buf.len() as u64 > max {
        bail!("{} is larger than {max} bytes", path.display());
    }
    Ok(buf)
}

/// The label file at `path`: None when it does not exist; an error when it does not parse or
/// is of a newer version.
pub fn read(path: &Path) -> Result<Option<LabelFile>> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        Ok(m) if !m.file_type().is_file() => bail!("{} is not a regular file", path.display()),
        Ok(_) => {}
    }
    let bytes = read_capped(path, MAX_LABEL_FILE_BYTES)?;
    let f: LabelFile =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if f.version > LABELS_VERSION {
        bail!(
            "{}: label file version {} is newer than this build understands",
            path.display(),
            f.version
        );
    }
    Ok(Some(f))
}

/// Serializes label writes within the process (a save and an import of the same frame).
static WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Write `f` to `path` atomically (temporary file + rename).
pub fn write(path: &Path, f: &LabelFile) -> Result<()> {
    let mut text = serde_json::to_vec_pretty(f)?;
    text.push(b'\n');
    let _held = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    layout::write_atomic(path, &text).with_context(|| format!("writing {}", path.display()))
}

/// Every label file under `dir` with its frame (`<scene>/<camera>/<date>/<stem>.jpg`), in day
/// folder order then by name. Files whose frame image is gone are left out.
pub fn label_files(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for day in layout::day_dirs(dir) {
        let Ok(rd) = std::fs::read_dir(&day) else {
            continue;
        };
        let mut names: Vec<String> = rd
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .filter(|n| n.ends_with(LABELS_SUFFIX))
            .collect();
        names.sort();
        let Ok(rel_day) = day.strip_prefix(dir) else {
            continue;
        };
        let rel_day: Vec<String> = rel_day
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        for n in names {
            let Some(stem) = n.strip_suffix(LABELS_SUFFIX) else {
                continue;
            };
            let rel = format!("{}/{stem}.jpg", rel_day.join("/"));
            if layout::parse_rel(&rel).is_some() && day.join(format!("{stem}.jpg")).is_file() {
                out.push((rel, day.join(n)));
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::record::{Det, Suppressed, VerifyInfo, empty_record};
    use super::*;

    fn det(label: &str, score: f32, b: [f32; 4]) -> Det {
        Det {
            label: label.into(),
            score,
            bbox: b,
        }
    }

    pub(super) fn record() -> Record {
        let mut r = empty_record();
        r.id = "20261022T010203004Z_abcd".into();
        r.width = 640;
        r.height = 480;
        r.threshold = 0.5;
        r.predictions = vec![det("person", 0.8, [10.0, 10.0, 50.0, 90.0])];
        r.raw = vec![
            det("person", 0.8, [10.0, 10.0, 50.0, 90.0]),
            det("dog", 0.55, [100.0, 100.0, 150.0, 140.0]),
        ];
        r.suppressed = vec![Suppressed {
            label: "dog".into(),
            score: 0.55,
            bbox: [100.0, 100.0, 150.0, 140.0],
            reason: "night_threshold".into(),
        }];
        r.below_threshold = vec![
            det("cat", 0.45, [300.0, 300.0, 700.0, 520.0]),
            // The same box as the prediction: a duplicate.
            det("person", 0.42, [10.0, 10.0, 50.0, 90.5]),
        ];
        r.verify = Some(VerifyInfo {
            verifier: "big".into(),
            mode: "frame".into(),
            found: vec![
                det("person", 0.7, [11.0, 11.0, 51.0, 91.0]),
                det("car", 0.6, [400.0, 10.0, 600.0, 120.0]),
            ],
            ..VerifyInfo::default()
        });
        r
    }

    fn classes() -> Vec<String> {
        super::super::config::default_classes()
    }

    #[test]
    fn prefill_lists_every_sidecar_box_once_and_clamps() {
        let f = prefill(&record());
        assert_eq!(f.status, Status::Unreviewed);
        let summary: Vec<(&str, Option<Origin>)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.origin))
            .collect();
        assert_eq!(
            summary,
            [
                ("person", Some(Origin::Prediction)),
                ("dog", Some(Origin::Suppressed)),
                ("cat", Some(Origin::BelowThreshold)),
                ("car", Some(Origin::VerifyFound)),
            ]
        );
        assert_eq!(f.boxes[1].reason.as_deref(), Some("night_threshold"));
        assert_eq!(f.boxes[2].bbox, [300.0, 300.0, 640.0, 480.0], "clamped");
        assert!(f.boxes.iter().all(|b| b.state == BoxState::Pending));
    }

    #[test]
    fn reviewed_save_resolves_pending_and_checks_classes() {
        let now = "2026-10-22T12:00:00.000Z";
        let mut f = prefill(&record());
        f.status = Status::Reviewed;
        // Restore the wrongly suppressed dog (a miss of the rule).
        f.boxes[1].state = BoxState::Accepted;
        f.boxes.push(LabelBox {
            label: "raccoon".into(),
            bbox: [500.0, 400.0, 480.0, 300.0],
            source: Source::Human,
            score: None,
            state: BoxState::Pending,
            origin: None,
            reason: None,
            model_label: None,
        });
        let n = normalize(f.clone(), &classes(), 640, 480, now).unwrap();
        let states: Vec<BoxState> = n.boxes.iter().map(|b| b.state).collect();
        assert_eq!(
            states,
            [
                BoxState::Accepted,
                BoxState::Accepted,
                BoxState::Rejected,
                BoxState::Rejected,
                BoxState::Added
            ]
        );
        assert_eq!(n.boxes[4].bbox, [480.0, 300.0, 500.0, 400.0], "ordered");
        assert_eq!(n.reviewed_at.as_deref(), Some(now));
        assert_eq!(n.classes_version, classes_version(&classes()));
        assert_eq!(n.training().count(), 3);
        assert_eq!(n.rejected().count(), 2);

        let mut bad = f.clone();
        bad.boxes[0].label = "unicorn".into();
        let e = normalize(bad, &classes(), 640, 480, now).unwrap_err();
        assert!(e.to_string().contains("unicorn"), "{e}");
        let mut empty = f.clone();
        empty.empty_confirmed = true;
        let e = normalize(empty, &classes(), 640, 480, now).unwrap_err();
        assert!(e.to_string().contains("confirmed empty"), "{e}");
        let mut tiny = f.clone();
        tiny.boxes[0].bbox = [700.0, 10.0, 800.0, 20.0];
        assert!(normalize(tiny, &classes(), 640, 480, now).is_err());
        let mut nan = f;
        nan.boxes[0].bbox[0] = f32::NAN;
        assert!(normalize(nan, &classes(), 640, 480, now).is_err());
    }

    #[test]
    fn unreviewed_save_keeps_pending_and_drops_rejected_human_boxes() {
        let mut f = prefill(&record());
        f.boxes[0].label = "unicorn".into();
        f.boxes.push(LabelBox {
            label: "x".into(),
            bbox: [1.0, 1.0, 9.0, 9.0],
            source: Source::Human,
            score: None,
            state: BoxState::Rejected,
            origin: None,
            reason: None,
            model_label: None,
        });
        let n = normalize(f, &classes(), 640, 480, "now").unwrap();
        assert_eq!(n.boxes.len(), 4);
        assert!(n.boxes.iter().all(|b| b.state == BoxState::Pending));
        assert_eq!(n.reviewed_at, None);
    }

    #[test]
    fn label_file_round_trips_and_atomic_write() {
        let dir = std::env::temp_dir().join(format!("bop-labels-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.labels.json");
        assert_eq!(read(&p).unwrap(), None);
        let mut f = normalize(
            prefill(&record()),
            &classes(),
            640,
            480,
            "2026-10-22T12:00:00.000Z",
        )
        .unwrap();
        f.boxes[0].model_label = Some("pedestrian".into());
        write(&p, &f).unwrap();
        let back = read(&p).unwrap().unwrap();
        assert_eq!(back, f);
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"status\": \"unreviewed\""), "{text}");
        assert!(text.contains("\"origin\": \"suppressed\""), "{text}");
        assert!(
            !std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".tmp")),
            "no temporary file left"
        );
        std::fs::write(&p, r#"{"version": 2, "status": "reviewed"}"#).unwrap();
        assert!(read(&p).unwrap_err().to_string().contains("newer"));
        // A minimal hand-written file parses with defaults.
        std::fs::write(&p, r#"{"status": "skipped"}"#).unwrap();
        let m = read(&p).unwrap().unwrap();
        assert_eq!(
            (m.version, m.status, m.boxes.len()),
            (1, Status::Skipped, 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn auto(label: &str, b: [f32; 4]) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: b,
            source: Source::Auto,
            score: Some(0.7),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
        }
    }

    #[test]
    fn merge_keeps_decisions_and_reopens_on_new_boxes() {
        let mut f = normalize(prefill(&record()), &classes(), 640, 480, "t").unwrap();
        f.status = Status::Reviewed;
        f.boxes[1].state = BoxState::Rejected; // the suppressed dog: wrong
        let r = merge_proposals(
            &mut f,
            vec![
                auto("person", [10.0, 10.0, 50.0, 88.0]), // same label: dropped
                auto("dog", [100.0, 100.0, 150.0, 141.0]), // rejected before: dropped
                auto("cat", [100.0, 100.0, 150.0, 140.0]), // another label on a decided box
                auto("deer", [200.0, 200.0, 260.0, 260.0]), // new
                auto("deer", [201.0, 200.0, 260.0, 260.0]), // duplicate of the new one
            ],
        );
        assert_eq!(
            r,
            MergeReport {
                added: 2,
                dropped: 3,
                reopened: true
            }
        );
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(f.boxes[1].state, BoxState::Rejected, "decision kept");
        let added: Vec<&str> = f.boxes[4..].iter().map(|b| b.label.as_str()).collect();
        assert_eq!(added, ["cat", "deer"]);
        assert!(f.boxes[4..].iter().all(|b| b.state == BoxState::Pending));

        let mut g = prefill(&record());
        let r = merge_proposals(&mut g, vec![auto("cat", [10.0, 10.0, 50.0, 90.0])]);
        assert_eq!(r.added, 0, "another label on a pending box is dropped");
        assert!(!r.reopened);
    }

    #[test]
    fn iou_and_clamp() {
        assert_eq!(iou(&[0.0, 0.0, 2.0, 2.0], &[1.0, 1.0, 3.0, 3.0]), 1.0 / 7.0);
        assert_eq!(iou(&[0.0, 0.0, 1.0, 1.0], &[2.0, 2.0, 3.0, 3.0]), 0.0);
        assert_eq!(iou(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]), 0.0);
        assert_eq!(
            clamp_box([-5.0, 50.0, 20.0, -1.0], 10, 10),
            [0.0, 0.0, 10.0, 10.0]
        );
        assert_eq!(classes_version(&["a".into()]).len(), 16);
        assert_ne!(
            classes_version(&["a".into(), "b".into()]),
            classes_version(&["b".into(), "a".into()])
        );
    }

    #[test]
    fn frames_resolve_only_inside_the_dataset() {
        let dir = std::env::temp_dir().join(format!("bop-frame-{}", uuid::Uuid::new_v4()));
        let day = dir.join("ir/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        let stem = "20261022T010203004Z_abcd";
        std::fs::write(day.join(format!("{stem}.jpg")), b"jpg").unwrap();
        let rel = format!("ir/cam/2026-10-22/{stem}.jpg");
        assert!(Frame::resolve(&dir, &rel).is_none(), "no sidecar");
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&record()).unwrap(),
        )
        .unwrap();
        let f = Frame::resolve(&dir, &rel).unwrap();
        assert!(f.labels.ends_with(format!("{stem}.labels.json")));
        assert_eq!(f.record().unwrap().width, 640);
        for bad in [
            format!("ir/cam/2026-10-22/{stem}.json"),
            format!("ir/cam/2026-10-22/{stem}.labels.json"),
            "../x.jpg".to_string(),
            format!("ir/cam/2026-10-22/../../../{stem}.jpg"),
        ] {
            assert!(Frame::resolve(&dir, &bad).is_none(), "{bad}");
        }
        assert!(label_files(&dir).is_empty());
        write(&f.labels, &LabelFile::default()).unwrap();
        let listed = label_files(&dir);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0, rel);
        assert!(listed[0].1.ends_with(format!("{stem}.labels.json")));
        #[cfg(unix)]
        {
            std::fs::remove_file(&f.labels).unwrap();
            std::os::unix::fs::symlink("/etc/hosts", &f.labels).unwrap();
            assert!(Frame::resolve(&dir, &rel).is_none(), "symlinked label file");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
