//! Review labels of a collected frame: `<stem>.labels.json` beside its sidecar (schema
//! [`LABELS_VERSION`], `docs/DATASET.md` "Review labels"). Pure parts: the pre-fill from the
//! sidecar ([`prefill`]), the save-time normalisation ([`normalize`]: clamping, pending boxes
//! resolved, class check) and the merge of later proposals such as auto-labeler boxes
//! ([`merge_proposals`]), and whether the auto-labeler's triage alone settles a frame
//! ([`auto_accept_eligible`]). IO: [`Frame::resolve`] (every frame reference goes through
//! [`super::layout::parse_rel`] / [`super::layout::resolve`]), [`read`], [`revision`] and the
//! atomic writes under one process-wide lock ([`write`], [`update`], [`write_checked`]). The
//! review queue is in [`queue`], the bulk accept / low-score reject in [`bulk`], the YOLO /
//! COCO / benchmark exporters in [`export`] and the X-AnyLabeling interop in [`xany`].

pub mod bulk;
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
    /// The label the model or auto-labeler gave, when the reviewer relabeled the box, or the
    /// model's own spelling when an auto-label run mapped it onto the class list (`motorbike` for
    /// a `motorcycle` box; the Review page shows "was motorbike").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_label: Option<String>,
    /// The proposed box before the reviewer moved or resized it (set by the Review page).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orig_bbox: Option<[f32; 4]>,
    /// What the auto-labeler found out about the box (`agree`, `triage`, `vlm`, ...); empty for
    /// other boxes. Flattened: the keys sit on the box itself.
    #[serde(flatten)]
    pub auto: AutoInfo,
}

/// The auto-labeler's triage of a box ([`crate::training::autolabel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Triage {
    /// Both labelers agree at a high score and no check objects.
    Accept,
    /// A reviewer should look at it (single labeler, low score, a night person or animal, a VLM
    /// or SigLIP objection).
    Review,
}

/// The VLM crop check of a box.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VlmCheck {
    /// The checkpoint (`qwen3-vl-2b`).
    pub model: String,
    /// Its one-word answer, normalized (`person`, `vehicle`, `animal`, `statue`, `rock`, `plant`,
    /// `nothing`, or what it said).
    pub answer: String,
    /// The answer that agrees with the box's label (`person`, `vehicle` or `animal`).
    pub expected: String,
    /// `answer == expected`.
    pub ok: bool,
}

/// The auto-labeler's evidence on a box (all optional; empty for model and human boxes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AutoInfo {
    /// Both labelers (MM-Grounding-DINO and RF-DETR) found the box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agree: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triage: Option<Triage>,
    /// Why the box goes to review (empty for `accept`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub why: Vec<String>,
    /// Score of each labeler that found it (`gdino`, `rfdetr`, or a trained model's name).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub labelers: std::collections::BTreeMap<String, f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm: Option<VlmCheck>,
    /// SigLIP's zero-shot probabilities of the animal classes on the crop (sum 1).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub species_scores: std::collections::BTreeMap<String, f32>,
    /// A class suggestion: the class the auto-labelers give this box, which the box itself
    /// labels otherwise (set on an existing pending box; triage is then `review`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_label: Option<String>,
    /// The auto-labelers' fused score of `auto_label`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_score: Option<f32>,
}

impl AutoInfo {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
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
            orig_bbox: None,
            auto: AutoInfo::default(),
        }
    }
}

/// The last auto-label run (labelers mode) that looked at a frame, boxes found or not: a later
/// run with the same labelers skips the frame unless asked to label it again, or unless it asks
/// for a class this run did not look for ([`AutoRun::covers`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoRun {
    /// When (UTC, RFC 3339).
    pub at: String,
    /// The labelers (`mm-gdino-tiny + rfdetr-large + ...`).
    pub labeler: String,
    /// The classes the run looked for; empty = the whole class list of the time (files written
    /// before runs could pick classes).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
}

impl AutoRun {
    /// Whether this run looked for every class of `classes` (case-insensitive; a run without
    /// a class list covers everything).
    pub fn covers(&self, classes: &[String]) -> bool {
        self.classes.is_empty()
            || classes
                .iter()
                .all(|c| self.classes.iter().any(|r| r.eq_ignore_ascii_case(c)))
    }
}

/// Longest [`AutoRun`] field (each class name of `classes` too).
pub const MAX_AUTO_RUN_LEN: usize = 200;
/// Most classes in [`AutoRun::classes`].
pub const MAX_AUTO_RUN_CLASSES: usize = super::config::MAX_CLASSES;

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
    /// Why the frame went back to review after a re-label with a trained model (its boxes
    /// disagree with the reviewed labels); the Review queue lists these frames first. Cleared
    /// when the frame is reviewed or skipped again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disagreement: Option<String>,
    /// The last auto-label run over the frame (round-tripped by the Review page).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_run: Option<AutoRun>,
    /// Saved by the bulk accept of auto-labels (`POST /v1/review/accept-auto`), not by a
    /// reviewer looking at the frame. [`normalize`] clears it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bulk_accepted: bool,
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
            disagreement: None,
            auto_run: None,
            bulk_accepted: false,
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
/// confirming each true one); a suppressed, below-threshold or verifier-found box is accepted
/// when the auto-labelers confirmed it (`triage: accept`, no class suggestion), else rejected
/// (it was not reported, so restoring it is an explicit decision). The Review page mirrors
/// this rule.
pub fn resolve_pending(b: &LabelBox) -> BoxState {
    match (b.state, b.source, b.origin) {
        (BoxState::Pending, Source::Human, _) => BoxState::Added,
        (BoxState::Pending, Source::Auto, _) => BoxState::Accepted,
        (BoxState::Pending, Source::Model, Some(Origin::Prediction)) => BoxState::Accepted,
        (BoxState::Pending, Source::Model, _) if auto_accepted(b) => BoxState::Accepted,
        (BoxState::Pending, Source::Model, _) => BoxState::Rejected,
        (s, _, _) => s,
    }
}

/// The auto-labelers accept the box as labelled (`triage: accept`, no class suggestion).
fn auto_accepted(b: &LabelBox) -> bool {
    b.auto.triage == Some(Triage::Accept) && b.auto.auto_label.is_none()
}

/// Whether saving `f` as reviewed without looking would only confirm what the auto-labelers
/// already settled: an unreviewed frame, not confirmed empty, without a re-label
/// disagreement, with at least one pending box triaged `accept` (no class suggestion) and every
/// other pending box a model hint the labelers did not confirm (suppressed, below threshold or
/// verifier-found, no auto evidence: rejected on save). Decided boxes do not matter. Used by the
/// queue's `auto=accept` filter and the bulk accept.
pub fn auto_accept_eligible(f: &LabelFile) -> bool {
    if f.status != Status::Unreviewed || f.empty_confirmed || f.disagreement.is_some() {
        return false;
    }
    let mut any = false;
    for b in f.boxes.iter().filter(|b| b.state == BoxState::Pending) {
        if auto_accepted(b) {
            any = true;
        } else if !(b.source == Source::Model
            && b.origin != Some(Origin::Prediction)
            && b.auto.is_empty())
        {
            return false;
        }
    }
    any
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
/// - `orig_bbox` checked and clamped like `bbox`, dropped when equal to it; `auto_run` kept
///   (bounded), `bulk_accepted` cleared;
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
        check_auto(&b.auto).with_context(|| format!("box {n}"))?;
        b.bbox = clamp_box(b.bbox, w, h);
        if let Some(o) = b.orig_bbox {
            if !o.iter().all(|v| v.is_finite()) {
                bail!("box {n}: the original coordinates are not finite numbers");
            }
            let o = clamp_box(o, w, h);
            b.orig_bbox = (o != b.bbox).then_some(o);
        }
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
    f.bulk_accepted = false;
    if let Some(r) = &f.auto_run {
        if [&r.at, &r.labeler]
            .into_iter()
            .chain(&r.classes)
            .any(|s| s.chars().count() > MAX_AUTO_RUN_LEN || s.chars().any(char::is_control))
        {
            bail!(
                "auto_run: a field is longer than {MAX_AUTO_RUN_LEN} characters or holds control \
                 characters"
            );
        }
        if r.classes.len() > MAX_AUTO_RUN_CLASSES {
            bail!(
                "auto_run: {} classes; at most {MAX_AUTO_RUN_CLASSES}",
                r.classes.len()
            );
        }
    }
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
    if f.status != Status::Unreviewed {
        f.disagreement = None;
    }
    if f.disagreement
        .as_ref()
        .is_some_and(|d| d.chars().count() > 500)
    {
        bail!("the disagreement note is longer than 500 characters");
    }
    Ok(f)
}

/// Bounds of the auto-labeler fields a client sends back (they are only round-tripped).
fn check_auto(a: &AutoInfo) -> Result<()> {
    let short = |s: &str| s.chars().count() <= MAX_LABEL_LEN && !s.chars().any(char::is_control);
    let score_ok = |s: f32| s.is_finite() && (0.0..=1.0).contains(&s);
    if a.why.len() > 16 || a.why.iter().any(|w| w.chars().count() > 200) {
        bail!("too many or too long triage reasons");
    }
    if a.labelers.len() > 16 || a.species_scores.len() > super::config::MAX_CLASSES {
        bail!("too many labeler or species scores");
    }
    for (k, v) in a.labelers.iter().chain(&a.species_scores) {
        if !short(k) || !score_ok(*v) {
            bail!(
                "'{}': not a name with a score between 0 and 1",
                k.escape_debug()
            );
        }
    }
    if let Some(v) = &a.vlm
        && !(short(&v.model) && short(&v.answer) && short(&v.expected))
    {
        bail!("the VLM check has a field longer than {MAX_LABEL_LEN} characters");
    }
    if let Some(l) = &a.auto_label
        && !label_ok(l)
    {
        bail!(
            "auto_label '{}' is not a valid label (1 to {MAX_LABEL_LEN} characters)",
            l.escape_debug()
        );
    }
    if let Some(s) = a.auto_score
        && !score_ok(s)
    {
        bail!("auto_score {s} is not between 0 and 1");
    }
    Ok(())
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

/// The bytes of the label file at `path` (None when it does not exist; never through a
/// symlink).
fn read_bytes(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        Ok(m) if !m.file_type().is_file() => bail!("{} is not a regular file", path.display()),
        Ok(_) => {}
    }
    match read_capped(path, MAX_LABEL_FILE_BYTES) {
        Err(e)
            if e.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(None)
        }
        r => r.map(Some),
    }
}

/// The revision of file bytes (`""` = no file): the first 16 hex digits of their SHA-256.
fn revision_of(bytes: Option<&[u8]>) -> String {
    use sha2::Digest;
    bytes.map_or_else(String::new, |b| {
        sha2::Sha256::digest(b)
            .iter()
            .take(8)
            .map(|x| format!("{x:02x}"))
            .collect()
    })
}

/// The revision of the label file at `path`: `""` when it does not exist, else the first 16
/// hex digits of the SHA-256 of its bytes. A client sends it back with a save
/// ([`write_checked`]) so a frame changed meanwhile is not overwritten.
pub fn revision(path: &Path) -> Result<String> {
    Ok(revision_of(read_bytes(path)?.as_deref()))
}

/// The label file at `path` and its [`revision`], from one read.
pub fn read_with_revision(path: &Path) -> Result<(Option<LabelFile>, String)> {
    let bytes = read_bytes(path)?;
    let rev = revision_of(bytes.as_deref());
    match bytes {
        None => Ok((None, rev)),
        Some(b) => Ok((Some(parse(path, &b)?), rev)),
    }
}

/// The label file at `path`: None when it does not exist; an error when it does not parse or
/// is of a newer version.
pub fn read(path: &Path) -> Result<Option<LabelFile>> {
    match read_bytes(path)? {
        None => Ok(None),
        Some(b) => parse(path, &b).map(Some),
    }
}

fn parse(path: &Path, bytes: &[u8]) -> Result<LabelFile> {
    let f: LabelFile =
        serde_json::from_slice(bytes).with_context(|| format!("parsing {}", path.display()))?;
    if f.version > LABELS_VERSION {
        bail!(
            "{}: label file version {} is newer than this build understands",
            path.display(),
            f.version
        );
    }
    Ok(f)
}

/// Serializes label writes within the process (a save, an import and an auto-label run of the
/// same frame). Not reentrant: only [`write`], [`update`] and [`write_checked`] take it.
static WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn write_lock() -> std::sync::MutexGuard<'static, ()> {
    WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write `f` to `path` atomically; the caller holds [`WRITE_LOCK`].
fn write_locked(path: &Path, f: &LabelFile) -> Result<()> {
    let mut text = serde_json::to_vec_pretty(f)?;
    text.push(b'\n');
    layout::write_atomic(path, &text).with_context(|| format!("writing {}", path.display()))
}

/// Write `f` to `path` atomically (temporary file + rename).
pub fn write(path: &Path, f: &LabelFile) -> Result<()> {
    let _held = write_lock();
    write_locked(path, f)
}

/// Read-modify-write of the label file at `path` under the write lock: `f` gets the current
/// file (None when there is none) and returns the file to write (None = leave it). Returns
/// whether it wrote.
pub fn update(
    path: &Path,
    f: impl FnOnce(Option<LabelFile>) -> Result<Option<LabelFile>>,
) -> Result<bool> {
    let _held = write_lock();
    match f(read(path)?)? {
        Some(new) => write_locked(path, &new).map(|()| true),
        None => Ok(false),
    }
}

/// Write `f` to `path` only when the file's [`revision`] is still `expected` (the one the
/// client read; `""` = it did not exist). `Ok(Err(current))` when it changed meanwhile:
/// nothing is written.
pub fn write_checked(path: &Path, f: &LabelFile, expected: &str) -> Result<Result<(), String>> {
    let _held = write_lock();
    let current = revision(path)?;
    if current != expected {
        return Ok(Err(current));
    }
    write_locked(path, f).map(Ok)
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

/// Test helpers shared by the label modules' tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// A pending auto box of `label` at a fixed place.
    pub(crate) fn auto_box(label: &str) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: [10.0, 10.0, 50.0, 50.0],
            source: Source::Auto,
            score: Some(0.5),
            state: BoxState::Pending,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            auto: AutoInfo::default(),
        }
    }
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
            orig_bbox: None,
            auto: Default::default(),
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
            orig_bbox: None,
            auto: Default::default(),
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
            orig_bbox: None,
            auto: Default::default(),
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
    fn auto_info_round_trips_flat_and_disagreement_clears_on_review() {
        let mut b = tests_support::auto_box("person");
        b.auto = AutoInfo {
            agree: Some(true),
            triage: Some(Triage::Review),
            why: vec!["night person".into()],
            labelers: [("gdino".to_string(), 0.41), ("rfdetr".to_string(), 0.77)].into(),
            vlm: Some(VlmCheck {
                model: "qwen3-vl-2b".into(),
                answer: "rock".into(),
                expected: "person".into(),
                ok: false,
            }),
            species_scores: Default::default(),
            auto_label: None,
            auto_score: None,
        };
        let v = serde_json::to_value(&b).unwrap();
        assert_eq!(v["agree"], true);
        assert_eq!(v["triage"], "review");
        assert_eq!(v["vlm"]["answer"], "rock");
        assert_eq!(v["labelers"]["rfdetr"], serde_json::json!(0.77f32));
        assert!(v.get("species_scores").is_none(), "empty maps are left out");
        let back: LabelBox = serde_json::from_value(v).unwrap();
        assert_eq!(back, b);
        // A plain box has none of the keys.
        let plain = serde_json::to_value(LabelBox {
            auto: AutoInfo::default(),
            ..b.clone()
        })
        .unwrap();
        assert!(plain.get("agree").is_none() && plain.get("triage").is_none());

        let f = LabelFile {
            boxes: vec![b.clone()],
            disagreement: Some("model x: 1 class change".into()),
            ..LabelFile::default()
        };
        let un = normalize(f.clone(), &classes(), 640, 480, "t").unwrap();
        assert!(un.disagreement.is_some(), "kept while unreviewed");
        let rev = normalize(
            LabelFile {
                status: Status::Reviewed,
                ..f.clone()
            },
            &classes(),
            640,
            480,
            "t",
        )
        .unwrap();
        assert_eq!(rev.disagreement, None);
        assert_eq!(rev.boxes[0].state, BoxState::Accepted);
        let mut bad = f;
        bad.boxes[0].auto.labelers.insert("x".into(), 2.0);
        assert!(normalize(bad, &classes(), 640, 480, "t").is_err());
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

    /// A pending model box of `origin` with `triage` (and a class suggestion when given).
    fn hint(origin: Origin, triage: Option<Triage>, suggest: Option<&str>) -> LabelBox {
        LabelBox {
            label: "dog".into(),
            bbox: [100.0, 100.0, 150.0, 140.0],
            source: Source::Model,
            score: Some(0.4),
            state: BoxState::Pending,
            origin: Some(origin),
            reason: None,
            model_label: None,
            orig_bbox: None,
            auto: AutoInfo {
                triage,
                auto_label: suggest.map(str::to_string),
                auto_score: suggest.map(|_| 0.9),
                ..AutoInfo::default()
            },
        }
    }

    #[test]
    fn a_hint_the_auto_labelers_accept_is_accepted_on_save() {
        let accept = hint(Origin::Suppressed, Some(Triage::Accept), None);
        assert_eq!(resolve_pending(&accept), BoxState::Accepted);
        for o in [Origin::BelowThreshold, Origin::VerifyFound] {
            assert_eq!(
                resolve_pending(&hint(o, Some(Triage::Accept), None)),
                BoxState::Accepted
            );
        }
        // Triaged review, a class suggestion, or no evidence: still a hard negative.
        for b in [
            hint(Origin::Suppressed, Some(Triage::Review), None),
            hint(Origin::Suppressed, Some(Triage::Accept), Some("raccoon")),
            hint(Origin::Suppressed, None, None),
            LabelBox {
                origin: None,
                ..hint(Origin::Suppressed, None, None)
            },
        ] {
            assert_eq!(resolve_pending(&b), BoxState::Rejected, "{b:?}");
        }
        // Decided boxes keep their state.
        let mut decided = accept;
        decided.state = BoxState::Rejected;
        assert_eq!(resolve_pending(&decided), BoxState::Rejected);
        let mut f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![
                hint(Origin::Suppressed, Some(Triage::Accept), None),
                hint(Origin::BelowThreshold, Some(Triage::Review), None),
            ],
            ..LabelFile::default()
        };
        f.boxes[1].bbox = [10.0, 10.0, 40.0, 40.0];
        let n = normalize(f, &classes(), 640, 480, "t").unwrap();
        assert_eq!(
            n.boxes.iter().map(|b| b.state).collect::<Vec<_>>(),
            [BoxState::Accepted, BoxState::Rejected]
        );
    }

    #[test]
    fn auto_accept_needs_accepted_boxes_and_only_unconfirmed_hints_besides() {
        let mut acc = tests_support::auto_box("car");
        acc.auto.triage = Some(Triage::Accept);
        let mut rev = tests_support::auto_box("dog");
        rev.auto.triage = Some(Triage::Review);
        let plain_hint = hint(Origin::Suppressed, None, None);
        let file = |boxes: Vec<LabelBox>| LabelFile {
            boxes,
            ..LabelFile::default()
        };
        assert!(auto_accept_eligible(&file(vec![acc.clone()])));
        assert!(auto_accept_eligible(&file(vec![
            acc.clone(),
            plain_hint.clone()
        ])));
        // A confirmed hint counts as accepted.
        assert!(auto_accept_eligible(&file(vec![hint(
            Origin::VerifyFound,
            Some(Triage::Accept),
            None
        )])));
        assert!(!auto_accept_eligible(&file(vec![])), "nothing to accept");
        assert!(!auto_accept_eligible(&file(vec![plain_hint.clone()])));
        assert!(!auto_accept_eligible(&file(vec![acc.clone(), rev.clone()])));
        // A model prediction without evidence would be accepted unseen: not eligible.
        let pred = LabelBox {
            auto: AutoInfo::default(),
            ..hint(Origin::Prediction, None, None)
        };
        assert!(!auto_accept_eligible(&file(vec![acc.clone(), pred])));
        // A class suggestion is never settled.
        assert!(!auto_accept_eligible(&file(vec![
            acc.clone(),
            hint(Origin::Suppressed, Some(Triage::Accept), Some("cat"))
        ])));
        // Decided boxes are ignored.
        let mut decided = rev;
        decided.state = BoxState::Rejected;
        assert!(auto_accept_eligible(&file(vec![acc.clone(), decided])));
        // File-level conditions.
        let mut f = file(vec![acc]);
        f.empty_confirmed = true;
        assert!(!auto_accept_eligible(&f));
        f.empty_confirmed = false;
        f.disagreement = Some("x".into());
        assert!(!auto_accept_eligible(&f));
        f.disagreement = None;
        f.status = Status::Skipped;
        assert!(!auto_accept_eligible(&f));
    }

    #[test]
    fn orig_bbox_auto_run_and_bulk_flag_on_save() {
        let mut b = tests_support::auto_box("car");
        b.orig_bbox = Some([-5.0, 10.0, 50.0, 50.0]);
        b.auto.auto_label = Some("truck".into());
        b.auto.auto_score = Some(0.8);
        let mut f = LabelFile {
            boxes: vec![b],
            auto_run: Some(AutoRun {
                at: "2026-10-22T00:00:00Z".into(),
                labeler: "mm-gdino-tiny + rfdetr-large".into(),
                classes: vec!["person".into(), "car".into()],
            }),
            bulk_accepted: true,
            ..LabelFile::default()
        };
        let n = normalize(f.clone(), &classes(), 640, 480, "t").unwrap();
        assert_eq!(
            n.boxes[0].orig_bbox,
            Some([0.0, 10.0, 50.0, 50.0]),
            "clamped"
        );
        assert_eq!(n.auto_run, f.auto_run, "kept as sent");
        assert!(!n.bulk_accepted, "only the bulk accept sets it");
        let v = serde_json::to_value(&n).unwrap();
        assert_eq!(v["boxes"][0]["auto_label"], "truck");
        assert_eq!(v["auto_run"]["labeler"], "mm-gdino-tiny + rfdetr-large");
        assert_eq!(
            v["auto_run"]["classes"],
            serde_json::json!(["person", "car"])
        );
        assert!(v.get("bulk_accepted").is_none());
        assert_eq!(serde_json::from_value::<LabelFile>(v).unwrap(), n);
        // Equal to the box (after clamping): dropped.
        f.boxes[0].orig_bbox = Some([10.0, 10.0, 50.0, 50.0]);
        let n = normalize(f.clone(), &classes(), 640, 480, "t").unwrap();
        assert_eq!(n.boxes[0].orig_bbox, None);
        f.boxes[0].orig_bbox = Some([f32::INFINITY, 0.0, 1.0, 1.0]);
        assert!(normalize(f.clone(), &classes(), 640, 480, "t").is_err());
        f.boxes[0].orig_bbox = None;
        f.boxes[0].auto.auto_score = Some(1.5);
        assert!(normalize(f.clone(), &classes(), 640, 480, "t").is_err());
        f.boxes[0].auto.auto_score = None;
        f.boxes[0].auto.auto_label = Some(String::new());
        assert!(normalize(f.clone(), &classes(), 640, 480, "t").is_err());
        f.boxes[0].auto.auto_label = None;
        f.auto_run.as_mut().unwrap().classes.push("a\nb".into());
        let e = normalize(f.clone(), &classes(), 640, 480, "t").unwrap_err();
        assert!(e.to_string().contains("auto_run"), "{e}");
        f.auto_run.as_mut().unwrap().classes = vec!["x".repeat(201)];
        assert!(normalize(f.clone(), &classes(), 640, 480, "t").is_err());
        f.auto_run.as_mut().unwrap().classes = vec!["c".into(); MAX_AUTO_RUN_CLASSES + 1];
        let e = normalize(f.clone(), &classes(), 640, 480, "t").unwrap_err();
        assert!(e.to_string().contains("classes"), "{e}");
        f.auto_run.as_mut().unwrap().classes.clear();
        f.auto_run.as_mut().unwrap().labeler = "x".repeat(201);
        let e = normalize(f, &classes(), 640, 480, "t").unwrap_err();
        assert!(e.to_string().contains("auto_run"), "{e}");
    }

    #[test]
    fn auto_run_classes_cover_a_later_run() {
        let r = |c: &[&str]| AutoRun {
            at: "t".into(),
            labeler: "l".into(),
            classes: c.iter().map(|s| (*s).to_string()).collect(),
        };
        let want = |c: &[&str]| c.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert!(
            r(&[]).covers(&want(&["person", "anything"])),
            "old files: everything"
        );
        assert!(r(&["Person", "car"]).covers(&want(&["person"])));
        assert!(r(&["person", "car"]).covers(&want(&["CAR", "person"])));
        assert!(!r(&["person"]).covers(&want(&["person", "dog"])));
        assert!(r(&["person"]).covers(&[]));
        // An old file without the field reads as an empty list and writes none.
        let old: AutoRun = serde_json::from_str(r#"{"at":"t","labeler":"l"}"#).unwrap();
        assert!(old.classes.is_empty());
        assert!(!serde_json::to_string(&old).unwrap().contains("classes"));
    }

    #[test]
    fn revisions_update_and_checked_writes() {
        let dir = std::env::temp_dir().join(format!("bop-labels-rev-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.labels.json");
        assert_eq!(revision(&p).unwrap(), "");
        assert_eq!(read_with_revision(&p).unwrap(), (None, String::new()));
        // A checked write against "no file yet".
        let f = LabelFile::default();
        assert_eq!(write_checked(&p, &f, "").unwrap(), Ok(()));
        let r1 = revision(&p).unwrap();
        assert_eq!(r1.len(), 16);
        assert_eq!(
            read_with_revision(&p).unwrap(),
            (Some(f.clone()), r1.clone())
        );
        // A stale revision writes nothing and names the current one.
        let g = LabelFile {
            status: Status::Skipped,
            ..LabelFile::default()
        };
        assert_eq!(write_checked(&p, &g, "").unwrap(), Err(r1.clone()));
        assert_eq!(read(&p).unwrap(), Some(f));
        // update: None leaves the file, Some writes it; the revision follows the bytes.
        assert!(
            !update(&p, |cur| {
                assert!(cur.is_some());
                Ok(None)
            })
            .unwrap()
        );
        assert_eq!(revision(&p).unwrap(), r1);
        assert!(update(&p, |_| Ok(Some(g.clone()))).unwrap());
        let r2 = revision(&p).unwrap();
        assert_ne!(r1, r2);
        assert_eq!(
            write_checked(&p, &LabelFile::default(), &r2).unwrap(),
            Ok(())
        );
        // An error inside update writes nothing.
        assert!(update(&p, |_| -> Result<Option<LabelFile>> { bail!("no") }).is_err());
        assert_eq!(revision(&p).unwrap(), r1);
        let missing = dir.join("b.labels.json");
        assert!(
            update(&missing, |cur| {
                assert!(cur.is_none());
                Ok(Some(g.clone()))
            })
            .unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
