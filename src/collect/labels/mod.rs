//! Review labels of a collected frame: `<stem>.labels.json` beside its sidecar (schema
//! [`LABELS_VERSION`], `docs/DATASET.md` "Review labels"). Pure parts: the pre-fill from the
//! sidecar ([`prefill`]; boxes of unwanted classes, [`wanted_class`], left out), the save-time
//! normalisation ([`normalize`]: clamping, pending boxes
//! resolved, class check) and the merge of later proposals such as auto-labeler boxes
//! ([`merge_proposals`]), whether the auto-labeler's triage alone settles a frame
//! ([`auto_accept_eligible`]), and which boxes a labelers run set aside
//! ([`LabelBox::is_dismissed`]). IO: [`Frame::resolve`], [`Frame::resolve_in`] on a walk (every frame reference goes through
//! [`super::layout::parse_rel`] / [`super::layout::resolve`]), [`read`], [`revision`] and the
//! atomic writes under one process-wide lock ([`write`], [`update`], [`write_checked`]). The
//! review queue is in [`queue`], the bulk accept and the low-score and unwanted-class rejects in
//! [`bulk`], the YOLO / COCO / benchmark exporters in [`export`], the X-AnyLabeling interop in
//! [`xany`], the repeat detections of fixed cameras (spot matching, per-camera history) in
//! [`repeats`], the rejected and parked spots a labelers run remembers in [`spots`], the duplicate
//! boxes on one object (stacks, overlaps) in [`stacks`] and the stored suggestions for overlapping
//! boxes in [`overlaps`].

pub mod bulk;
pub mod export;
pub mod overlaps;
pub mod queue;
pub mod repeats;
pub mod spots;
pub mod stacks;
pub mod xany;

use super::layout::{self, Rel};
use super::record::Record;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Version of the label file ([`LabelFile::version`]). Files of a newer version are skipped.
pub const LABELS_VERSION: u32 = 1;
/// Suffix of a label file after the frame's stem (`<stem>.labels.json`).
pub const LABELS_SUFFIX: &str = ".labels.json";
/// [`LabelBox::reason`] of a box rejected by the page's "Reject the rest". It counts for the repeats
/// history and spot memory: in a time-ordered simulation on one camera, leaving it out kept 1.96
/// boxes per frame visible against 1.43, and counting it hid 1 real object of 346.
pub const REST_REASON: &str = "rest_rejected";
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
/// Longest [`LabelFile::disagreement`] note, in characters ([`normalize`] refuses a longer one).
pub const MAX_DISAGREEMENT_LEN: usize = 500;
/// Most entries of [`AutoInfo::labelers`] ([`normalize`] refuses more).
pub const MAX_LABELERS: usize = 16;

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

/// Why the reviewer rejected a box (the Review tab's reason chips): hard-negative mining keys,
/// kept in `hard_negatives.json`. Separate from [`LabelBox::reason`] (provenance). A value this
/// build does not know (a newer build's reason) loads as [`RejectReason::Other`] and is written
/// back verbatim: a label file never fails to load over it, and a re-save does not lose it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Nothing there, or not an object of the label.
    FalsePositive,
    /// A second box on an object another box already covers.
    Duplicate,
    /// An insect or a spider web, usually near the IR lights.
    InsectWeb,
    RainSnow,
    /// IR bloom, glare, headlights.
    IrGlare,
    /// A reflection, a poster, a picture of the object.
    Reflection,
    /// A fixed part of the scene (railing, plant, bin).
    StaticObject,
    /// A real object of a class the list does not have.
    OtherObject,
    /// Too small or too far to tell: a real object not worth a label, so an ignore region
    /// ([`LabelBox::is_ignore`]), not a hard negative.
    TooSmall,
    /// A reason this build does not know (kept as written, checked by [`normalize`]).
    Other(String),
}

impl RejectReason {
    const KNOWN: [(&'static str, RejectReason); 9] = [
        ("false_positive", RejectReason::FalsePositive),
        ("duplicate", RejectReason::Duplicate),
        ("insect_web", RejectReason::InsectWeb),
        ("rain_snow", RejectReason::RainSnow),
        ("ir_glare", RejectReason::IrGlare),
        ("reflection", RejectReason::Reflection),
        ("static_object", RejectReason::StaticObject),
        ("other_object", RejectReason::OtherObject),
        ("too_small", RejectReason::TooSmall),
    ];

    pub fn as_str(&self) -> &str {
        match self {
            RejectReason::Other(s) => s,
            known => Self::KNOWN
                .iter()
                .find(|(_, r)| r == known)
                .map_or("", |(s, _)| s),
        }
    }

    pub fn parse(s: &str) -> Self {
        Self::KNOWN
            .iter()
            .find(|(k, _)| *k == s)
            .map_or_else(|| RejectReason::Other(s.to_string()), |(_, r)| r.clone())
    }
}

impl Serialize for RejectReason {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RejectReason {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(Self::parse(&String::deserialize(d)?))
    }
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
    /// a `motorcycle` box; the Review tab shows "was motorbike").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_label: Option<String>,
    /// The proposed box before the reviewer moved or resized it (set by the Review tab).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orig_bbox: Option<[f32; 4]>,
    /// The reviewer could not tell what the object is (PASCAL VOC `difficult`, Review tab `d`):
    /// an ignore region ([`LabelBox::is_ignore`]), an attribute in COCO. [`normalize`] clears it on
    /// a rejected box.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub difficult: bool,
    /// Why the reviewer rejected the box (optional; absent = a plain false positive).
    /// [`normalize`] clears it unless the box is rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<RejectReason>,
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

/// The auto-labeler's evidence on a box (all optional; empty for model and human boxes, unless a
/// labelers run found one again: then `labelers` alone, a vote for the overlap check's witnesses).
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
    /// Likely real ([`crate::training::autolabel::triage::tier`]): a labelers run's box with a high
    /// enough score and a VLM yes. Shown first on the Review tab; `triage` stays `review` (that
    /// enum is strict, a new value would break older builds).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub likely: bool,
    /// A sample of the boxes a run would have dismissed, left pending so that a reviewer's verdict
    /// checks the filter ([`Counts`](queue::Counts) `audit_accepted` / `audit_rejected`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub audit: bool,
    /// A box dismissed at a rejected spot ([`bulk::SPOT_DISMISSED_REASON`]): the frames the reviewer
    /// rejected that spot on. The Review tab's "you rejected this spot N times".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot_rejections: Option<u32>,
    /// A box a labelers run accepted at a parked spot ([`spots::PARKED_REASON`]): which spot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parked: Option<ParkedMark>,
}

/// The parked spot a box was accepted at ([`AutoInfo::parked`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParkedMark {
    /// [`spots::Parked::from`]: `YYYY-MM-DD`.
    pub since: String,
    /// The spot's own box: the boxes accepted at it differ a little.
    pub bbox: [f32; 4],
}

impl VlmCheck {
    /// A clear no to the yes/no question. A seven-way answer of an older run is neither a yes nor a
    /// no here, and a check that did not run leaves no [`VlmCheck`] at all.
    pub fn says_no(&self) -> bool {
        self.expected == "yes" && !self.ok
    }

    /// A clear yes to the yes/no question.
    pub fn says_yes(&self) -> bool {
        self.expected == "yes" && self.ok
    }
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

    /// A real object not to train on, and no negative either: an unsure box, or one rejected as
    /// too small. Its frame leaves training and it is `ignore` in a benchmark set. Read from the
    /// existing fields, so frames reviewed before this rule count without a rewrite.
    pub fn is_ignore(&self) -> bool {
        self.difficult
            || (self.state == BoxState::Rejected
                && self.reject_reason == Some(RejectReason::TooSmall))
    }

    /// Set aside by a labelers run, for its score or for its spot ([`bulk::is_dismissal`]):
    /// rejected, so no label and no live box, but nobody judged it, so no negative and no decision
    /// either. A reviewer's own verdict clears the reason ([`normalize`]).
    pub fn is_dismissed(&self) -> bool {
        self.state == BoxState::Rejected && bulk::is_dismissal(self.reason.as_deref())
    }

    /// The box shows a person's hand on an unreviewed frame: drawn, moved, relabelled, marked
    /// unsure, or accepted or rejected by anything but a machine rule (a labelers run's dismissal
    /// or parked-spot accept, a bulk reject by score, class or as a stacked duplicate). A spot edit
    /// counts: the reviewer chose that spot.
    fn has_human_decision(&self) -> bool {
        if self.source == Source::Human
            || self.orig_bbox.is_some()
            || self.difficult
            || (self.source == Source::Model && self.model_label.is_some())
        {
            return true;
        }
        match self.state {
            BoxState::Pending => false,
            BoxState::Added => true,
            BoxState::Accepted => self.reason.as_deref() != Some(spots::PARKED_REASON),
            BoxState::Rejected => {
                !self.is_dismissed()
                    && self.reject_reason != Some(RejectReason::Duplicate)
                    && !matches!(
                        self.reason.as_deref(),
                        Some(bulk::LOW_SCORE_REASON | bulk::OTHER_CLASS_REASON)
                    )
            }
        }
    }

    /// A reviewer's verdict on an [`AutoInfo::audit`] box: `Some(true)` real, `Some(false)` not,
    /// None while undecided or when the decision is no verdict on the box: a bulk reject by score
    /// or class, a duplicate (another box stays on the object), too small (real, but no label).
    /// Reject the rest counts as a no: the box was on screen and `Shift+R` is the way to clear
    /// unlikely boxes, so leaving it out would leave the check nearly all accepts.
    pub fn audit_verdict(&self) -> Option<bool> {
        if !self.auto.audit {
            return None;
        }
        match self.state {
            BoxState::Accepted => Some(true),
            BoxState::Rejected
                if !matches!(
                    self.reason.as_deref(),
                    Some(bulk::LOW_SCORE_REASON | bulk::OTHER_CLASS_REASON)
                ) && !matches!(
                    self.reject_reason,
                    Some(RejectReason::Duplicate | RejectReason::TooSmall)
                ) =>
            {
                Some(false)
            }
            _ => None,
        }
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
            difficult: false,
            reject_reason: None,
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
    /// Why the frame went back to review after a re-label (a trained model's boxes disagree with
    /// the reviewed labels, or a labeler found an object they lack); the Review queue lists these
    /// frames first. Cleared when the frame is reviewed or skipped again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disagreement: Option<String>,
    /// The last auto-label run over the frame (round-tripped by the Review tab).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_run: Option<AutoRun>,
    /// Saved by the bulk accept of auto-labels (`POST /v1/review/accept-auto`), not by a
    /// reviewer looking at the frame. [`normalize`] clears it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bulk_accepted: bool,
    /// A spot edit ("reject / accept this spot everywhere", [`repeats`]) changed boxes of this
    /// unreviewed frame: with no pending box left it is a repeat-only frame
    /// ([`repeats::repeats_only`]). [`normalize`] clears it once the frame is reviewed or skipped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub repeat_spot: bool,
    /// The reviewer saved the frame with stacked or overlapping boxes still on it: they looked and
    /// kept them (two real nested cars). `auto=overlaps` passes the frame over and the bulk reject
    /// leaves it alone ([`stacks`]). The Review tab sets it on a reviewed save; [`merge_proposals`]
    /// clears it when a box is added (the look did not cover that box).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub overlaps_checked: bool,
    /// A rerun's new boxes took [`Self::overlaps_checked`] off this frame
    /// (`autolabel::triage::apply_missing`). [`bulk::restore_reopened`] gives it back once those
    /// boxes are all rejected: what is left is what the reviewer checked. A spot accept of a new box
    /// ends it, and so do a restore and a save ([`normalize`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub checked_before_reopen: bool,
    /// Suggestions for overlapping pairs from the overlap check ([`overlaps`], `train overlaps`), for
    /// a reviewer to confirm: they change no box. Round-tripped by the Review tab and kept by every
    /// edit; one whose boxes moved or went is stale and ignored ([`overlaps::live`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlap_suggestions: Vec<overlaps::Suggestion>,
}

fn labels_version() -> u32 {
    LABELS_VERSION
}

impl LabelFile {
    /// Boxes wait for a reviewer, so retention keeps the frame: an auto-label run takes hours, and
    /// pruning its results before anyone saw them costs more than a folder over its cap (the disk
    /// guard still protects the volume). A file with nothing left to look at ages out.
    pub fn awaits_review(&self) -> bool {
        self.status == Status::Unreviewed && self.boxes.iter().any(|b| b.state == BoxState::Pending)
    }

    /// A person's work is on this file, so retention keeps the frame: it is reviewed or skipped, it
    /// was reviewed before a re-label sent it back, the reviewer confirmed it empty or checked its
    /// overlaps, or a box carries a reviewer's decision ([`LabelBox::has_human_decision`]). What an
    /// auto-label run alone writes (its `auto_run` marker, pending boxes, its dismissals) is not.
    pub fn has_human_decision(&self) -> bool {
        self.status != Status::Unreviewed
            || self.reviewed_at.is_some()
            || self.empty_confirmed
            || self.overlaps_checked
            || self.checked_before_reopen
            || self.boxes.iter().any(LabelBox::has_human_decision)
    }
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
            repeat_spot: false,
            overlaps_checked: false,
            checked_before_reopen: false,
            overlap_suggestions: Vec::new(),
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

    /// [`Self::rejected`] without the boxes that are no false positive to learn from: duplicates (a
    /// second box on an object), ignore regions (too small) and dismissed boxes
    /// ([`LabelBox::is_dismissed`]: nobody judged them).
    pub fn hard_negatives(&self) -> impl Iterator<Item = &LabelBox> {
        self.rejected().filter(|b| {
            b.reject_reason != Some(RejectReason::Duplicate) && !b.is_ignore() && !b.is_dismissed()
        })
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

/// How much of the smaller of two `[x0, y0, x1, y1]` boxes lies inside the other: their
/// intersection over the smaller area (0 for an empty box). 1 = one is wholly inside the other.
pub fn inside_share(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let smaller = area(a).min(area(b));
    if smaller > 0.0 {
        iw * ih / smaller
    } else {
        0.0
    }
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

/// `b` with its label given the class list's spelling ([`map_class`]; the model's own label
/// kept in `model_label` unless only the case differs), or None when it is no wanted class of
/// the list ([`wanted_class`]).
fn on_class_list(mut b: LabelBox, classes: &[String], hidden: &[String]) -> Option<LabelBox> {
    let class = wanted_class(&b.label, classes, hidden)?;
    rename_to_class(&mut b, class);
    Some(b)
}

/// Give `b` the label `class` (the class list's spelling of its label, [`map_class`]), keeping
/// the old label in `model_label` (unless it holds one already, or only the case differs).
pub(crate) fn rename_to_class(b: &mut LabelBox, class: &str) {
    if class == b.label {
        return;
    }
    if b.model_label.is_none() && !class.eq_ignore_ascii_case(b.label.trim()) {
        b.model_label = Some(b.label.chars().take(MAX_LABEL_LEN).collect());
    }
    b.label = class.to_string();
}

/// The class of `classes` a detection `label` is (exact, case-insensitive, then the COCO / VOC
/// aliases): [`crate::training::autolabel::triage::map_label`].
pub fn map_class<'a>(label: &str, classes: &'a [String]) -> Option<&'a str> {
    crate::training::autolabel::triage::map_label(label, classes)
}

/// Whether `class` is one of `hidden` (`collect.hidden_classes`; case-insensitive).
pub fn is_hidden(class: &str, hidden: &[String]) -> bool {
    hidden.iter().any(|h| h.eq_ignore_ascii_case(class.trim()))
}

/// The class of `classes` a box `label` is ([`map_class`], against the whole list) unless that
/// class is one of `hidden`. None = an unwanted label: of no class of the list, or of a class the
/// reviewer does not want (`collect.hidden_classes`: hidden from Review, its pending boxes
/// rejected).
pub fn wanted_class<'a>(label: &str, classes: &'a [String], hidden: &[String]) -> Option<&'a str> {
    map_class(label, classes).filter(|c| !is_hidden(c, hidden))
}

/// The unreviewed label file a frame starts with: every box of the sidecar of a wanted class of
/// `classes` (the collector's class list without `hidden`, [`wanted_class`]) as a pending model
/// box, in this order: the reply's predictions (accepted unless rejected), the suppressed boxes
/// with their reason (rejected unless restored: a wrongly suppressed real object is a miss), the
/// below-threshold boxes and the frame-mode verifier's own finds (both rejected unless
/// accepted). Labels are mapped onto the list's spelling ([`map_class`]: a VOC-spelled model's
/// `pottedplant` is `potted plant`, the model's label kept in `model_label`); a box of no class
/// of the list or of a hidden class is left out (the sidecar keeps it). A box of the same
/// (mapped) label at IoU >= [`DUP_IOU`] of an earlier one is left out (a verifier find at >=
/// [`MATCH_IOU`]).
pub fn prefill(rec: &Record, classes: &[String], hidden: &[String]) -> LabelFile {
    let mut boxes = Vec::new();
    let mut push = |b: LabelBox, dup_iou: f32| {
        if let Some(b) = on_class_list(b, classes, hidden) {
            push_unique(&mut boxes, b, dup_iou);
        }
    };
    for d in &rec.predictions {
        push(LabelBox::proposal(d, Origin::Prediction, None), DUP_IOU);
    }
    for s in &rec.suppressed {
        let d = super::record::Det {
            label: s.label.clone(),
            score: s.score,
            bbox: s.bbox,
        };
        push(
            LabelBox::proposal(&d, Origin::Suppressed, Some(s.reason.clone())),
            DUP_IOU,
        );
    }
    for d in &rec.below_threshold {
        push(LabelBox::proposal(d, Origin::BelowThreshold, None), DUP_IOU);
    }
    if let Some(v) = &rec.verify {
        for d in &v.found {
            let reason = Some(format!("verifier {}", v.verifier));
            push(
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
/// (it was not reported, so restoring it is an explicit decision). The Review tab mirrors
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

/// A pending box a save must not settle: the auto-labelers flagged it (`triage: review`, or they
/// name another class) and [`resolve_pending`] would make it a label. A hint it would reject is no
/// label, so it needs no decision. Twin of the page's `mustDecide`.
pub fn needs_decision(b: &LabelBox) -> bool {
    b.state == BoxState::Pending
        && resolve_pending(b) == BoxState::Accepted
        && b.source != Source::Human
        && (b.auto.triage == Some(Triage::Review)
            || b.auto.auto_label.as_ref().is_some_and(|l| *l != b.label))
}

/// The auto-labelers accept the box as labelled (`triage: accept`, no class suggestion).
fn auto_accepted(b: &LabelBox) -> bool {
    b.auto.triage == Some(Triage::Accept) && b.auto.auto_label.is_none()
}

/// Whether saving `f` as reviewed without looking would only confirm what the auto-labelers
/// already settled: an unreviewed frame, not confirmed empty, without a re-label
/// disagreement, with at least one pending box triaged `accept` (no class suggestion) and every
/// other pending box a model hint the labelers did not confirm (suppressed, below threshold or
/// verifier-found, no auto evidence: rejected on save). Decided boxes do not matter, but a
/// dismissed box ([`LabelBox::is_dismissed`]) keeps the frame out: a saved frame claims it is
/// complete, and 5% of those boxes were real on public sets, so a real object nobody looked at
/// would train as background. Used by the queue's `auto=accept` filter and the bulk accept.
pub fn auto_accept_eligible(f: &LabelFile) -> bool {
    if f.status != Status::Unreviewed
        || f.empty_confirmed
        || f.disagreement.is_some()
        || f.boxes.iter().any(LabelBox::is_dismissed)
    {
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
/// - `reject_reason` kept on rejected boxes only, `difficult` on the others;
/// - a dismissal ([`bulk::is_dismissal`]) cleared unless the box is rejected with no reason of a
///   reviewer's; [`spots::PARKED_REASON`] and its mark cleared unless the box is accepted;
/// - `orig_bbox` checked and clamped like `bbox`, dropped when equal to it; `auto_run` kept
///   (bounded), `overlaps_checked` and `overlap_suggestions` (bounded, [`overlaps::check`]) kept,
///   `bulk_accepted` and `checked_before_reopen` cleared;
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
        if let Some(RejectReason::Other(r)) = &b.reject_reason
            && (r.is_empty()
                || r.chars().count() > MAX_LABEL_LEN
                || r.chars().any(char::is_control))
        {
            bail!(
                "box {n}: the reject reason is empty, longer than {MAX_LABEL_LEN} characters or holds control characters"
            );
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
    f.checked_before_reopen = false;
    overlaps::check(&f.overlap_suggestions)?;
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
            let undecided = f.boxes.iter().filter(|b| needs_decision(b)).count();
            if undecided > 0 {
                bail!(
                    "{undecided} box(es) the auto-labelers flagged still need a decision: accept \
                     or reject them, then save"
                );
            }
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
    // A reject reason belongs to a rejected box only; a rejected box is not "unsure". A reviewer's
    // decision ends a dismissal: then the box is a label, or a negative like any other.
    for b in &mut f.boxes {
        let dismissed = bulk::is_dismissal(b.reason.as_deref());
        // A parked box a reviewer took back is a look like any other: the rule's mark goes.
        if b.state != BoxState::Accepted && b.reason.as_deref() == Some(spots::PARKED_REASON) {
            b.reason = None;
            b.auto.parked = None;
        }
        if b.state == BoxState::Rejected {
            b.difficult = false;
            if dismissed && b.reject_reason.is_some() {
                b.reason = None;
            }
        } else {
            b.reject_reason = None;
            if dismissed || b.reason.as_deref() == Some(REST_REASON) {
                b.reason = None;
            }
        }
    }
    if f.status != Status::Unreviewed {
        f.disagreement = None;
        f.repeat_spot = false;
    }
    if f.disagreement
        .as_ref()
        .is_some_and(|d| d.chars().count() > MAX_DISAGREEMENT_LEN)
    {
        bail!("the disagreement note is longer than {MAX_DISAGREEMENT_LEN} characters");
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
    if a.labelers.len() > MAX_LABELERS || a.species_scores.len() > super::config::MAX_CLASSES {
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
    if let Some(m) = &a.parked
        && !(short(&m.since) && m.bbox.iter().all(|v| v.is_finite()))
    {
        bail!("the parked spot is not a date and a box");
    }
    if a.likely && a.audit {
        bail!(
            "marked both likely and audit: an audit sample is a box scoring too low to be likely; \
             reload the frame and save again"
        );
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
/// decisions, and a gained box ends [`LabelFile::overlaps_checked`] (a stored suggestion stays: its
/// boxes are still there). Proposals are stored with state `pending` whatever they carry.
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
    if r.added > 0 {
        f.overlaps_checked = false;
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
        Self::with_labels(r, jpg, sidecar)
    }

    /// [`Self::resolve`] on a walk over many frames ([`layout::Walker`]): the same answers.
    pub fn resolve_in(walker: &mut layout::Walker<'_>, rel: &str) -> Option<Self> {
        let r = layout::parse_rel(rel).filter(|_| rel.ends_with(".jpg"))?;
        let jpg = walker.file(&r, "jpg")?;
        let sidecar = walker.file(&r, "json")?;
        Self::with_labels(r, jpg, sidecar)
    }

    /// The frame of `jpg` and `sidecar`, resolved, when its label file is no symlink.
    fn with_labels(r: Rel, jpg: PathBuf, sidecar: PathBuf) -> Option<Self> {
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
/// same frame). Not reentrant: only [`write_lock`] takes it.
static WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The folder lock of a dataset, a zero-byte file never deleted: at the dataset's root, so day
/// folders keep emptying for retention.
const LOCK_FILE: &str = ".labels.lock";
/// Longest wait for the folder lock. A write holds it for milliseconds, but a bulk job in another
/// process takes it again at once, and a flush can take tens of milliseconds, so a waiter can miss
/// many turns. Running out means a stopped process: that one write goes on, saving winning over
/// waiting, and the next one waits again.
const LOCK_WAIT: Duration = Duration::from_secs(30);
/// How long a folder whose lock cannot work (it will not open or lock) goes without one before it
/// is tried again. A lock that is merely held is never given up on: the next write would then
/// race the holder, with its work lost.
const LOCK_RETRY: Duration = Duration::from_secs(60);
/// Wait for a lock that outlasted [`LOCK_WAIT`] until it is next taken: writes still respect it
/// the moment it frees, without each costing the long wait while a process stays stopped.
const HELD_WAIT: Duration = Duration::from_secs(1);
/// Longest pause between tries of a held lock: short, to catch the moment a busy holder lets go.
const LOCK_POLL: Duration = Duration::from_millis(1);

/// Locks that outlasted a wait, until they are next taken: see [`HELD_WAIT`].
static LONG_HELD: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Folders whose lock failed, with when: see [`LOCK_RETRY`]. A folder is listed from its first
/// failure until a lock works there again, so it warns once.
static NO_FOLDER_LOCK: std::sync::Mutex<BTreeMap<PathBuf, Instant>> =
    std::sync::Mutex::new(BTreeMap::new());

/// What [`write_lock`] holds, released on drop.
struct WriteGuard {
    /// Before `_process`: the folder lock goes first.
    folder: Option<std::fs::File>,
    _process: std::sync::MutexGuard<'static, ()>,
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        // Explicitly: a process spawning a child right now shares this open file until the
        // child execs, and closing alone would leave the lock held for that moment
        // (`TrainLock`'s history). An unlock acts on the shared open file.
        if let Some(f) = &self.folder
            && let Err(e) = fs4::FileExt::unlock(f)
        {
            tracing::debug!("labels: releasing the folder lock: {e}");
        }
    }
}

/// The dataset folder's [`LOCK_FILE`] for the label file at `path`; beside it when `path` is not
/// in the layout (`<dir>/<scene>/<camera>/<date>/<file>`).
fn lock_path(path: &Path) -> PathBuf {
    let mut up = path.ancestors().skip(3);
    let root = up
        .next()
        .and_then(Path::file_name)
        .and_then(|n| n.to_str())
        .filter(|n| layout::SCENES.contains(n))
        .and_then(|_| up.next())
        .or_else(|| path.parent())
        .unwrap_or(Path::new("."));
    root.join(LOCK_FILE)
}

/// Remember that `lock` failed; true the first time since a lock last worked there.
fn folder_lock_failed(lock: &Path) -> bool {
    let mut failed = NO_FOLDER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    failed.insert(lock.to_path_buf(), Instant::now()).is_none()
}

/// Going on without the lock of `lock`: one warning per folder.
fn without_folder_lock(lock: &Path, why: &str) {
    if folder_lock_failed(lock) {
        tracing::warn!(
            "label files: {why}, so {} locks nothing for now (a network share or read-only \
             folder, or a stopped process). Saves go on under this process's own lock; a \
             `blue-onyx-prism train` command writing this folder at the same moment could \
             overwrite one. Avoid running both, or move the folder to a local disk.",
            lock.display()
        );
    }
}

fn open_lock(lock: &Path) -> std::io::Result<std::fs::File> {
    let rw = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock);
    match rw {
        // Another account's lock file: a read handle locks as well.
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            std::fs::File::open(lock).map_err(|_| e)
        }
        r => r,
    }
}

/// The lock of the folder of the label file at `path`, waiting at most `wait`, or None to go on
/// without it: another process writes the same files (`train autolabel` from the command line),
/// but a lock that fails or never comes must not stop a save.
fn folder_lock(path: &Path, wait: Duration) -> Option<std::fs::File> {
    let lock = lock_path(path);
    {
        let failed = NO_FOLDER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if failed.get(&lock).is_some_and(|t| t.elapsed() < LOCK_RETRY) {
            return None;
        }
    }
    let file = match open_lock(&lock) {
        Ok(f) => f,
        // No folder yet: the write itself says so.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            without_folder_lock(&lock, &format!("it cannot be opened ({e})"));
            return None;
        }
    };
    let long_held = || LONG_HELD.lock().unwrap_or_else(|e| e.into_inner());
    let wait = if long_held().contains(&lock) {
        wait.min(HELD_WAIT)
    } else {
        wait
    };
    let start = Instant::now();
    let mut pause = Duration::from_micros(100);
    loop {
        match fs4::FileExt::try_lock(&file) {
            Ok(()) => break,
            Err(fs4::TryLockError::WouldBlock) if start.elapsed() < wait => {
                std::thread::sleep(pause);
                pause = (pause * 2).min(LOCK_POLL);
            }
            Err(fs4::TryLockError::WouldBlock) => {
                long_held().insert(lock.clone());
                tracing::warn!(
                    "{} stayed held for {wait:?}, so this one label write goes on without it: \
                     another process may have stopped while writing. If it is running, its \
                     change to the same frame could be overwritten",
                    lock.display()
                );
                return None;
            }
            Err(fs4::TryLockError::Error(e)) => {
                without_folder_lock(&lock, &format!("it cannot be locked ({e})"));
                return None;
            }
        }
    }
    long_held().remove(&lock);
    NO_FOLDER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&lock);
    Some(file)
}

/// The label write lock for the file at `path`: [`WRITE_LOCK`], then the lock of its dataset
/// folder. Held around one read-modify-rename only, never across a job.
fn write_lock(path: &Path) -> WriteGuard {
    let process = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    WriteGuard {
        folder: folder_lock(path, LOCK_WAIT),
        _process: process,
    }
}

/// Write `f` to `path` atomically; the caller holds [`write_lock`].
fn write_locked(path: &Path, f: &LabelFile) -> Result<()> {
    let mut text = serde_json::to_vec_pretty(f)?;
    text.push(b'\n');
    let r =
        layout::write_atomic(path, &text).with_context(|| format!("writing {}", path.display()));
    repeats::written(path);
    r
}

/// Write `f` to `path` atomically (temporary file + rename).
pub fn write(path: &Path, f: &LabelFile) -> Result<()> {
    let _held = write_lock(path);
    write_locked(path, f)
}

/// Read-modify-write of the label file at `path` under the write lock: `f` gets the current
/// file (None when there is none) and returns the file to write (None = leave it). Returns
/// whether it wrote. `f` holds up every writer of the folder, other processes too: keep it short.
pub fn update(
    path: &Path,
    f: impl FnOnce(Option<LabelFile>) -> Result<Option<LabelFile>>,
) -> Result<bool> {
    let _held = write_lock(path);
    match f(read(path)?)? {
        Some(new) => write_locked(path, &new).map(|()| true),
        None => Ok(false),
    }
}

/// Write `f` to `path` only when the file's [`revision`] is still `expected` (the one the
/// client read; `""` = it did not exist). `Ok(Err(current))` when it changed meanwhile:
/// nothing is written.
pub fn write_checked(path: &Path, f: &LabelFile, expected: &str) -> Result<Result<(), String>> {
    let _held = write_lock(path);
    let current = revision(path)?;
    if current != expected {
        return Ok(Err(current));
    }
    write_locked(path, f).map(Ok)
}

/// The label file of the frame image `jpg` (`<stem>.jpg` -> `<stem>.labels.json`).
pub fn path_of_frame(jpg: &Path) -> PathBuf {
    jpg.with_extension(LABELS_SUFFIX.trim_start_matches('.'))
}

/// True when retention must keep the frame of the label file at `path`: it holds a human decision
/// ([`LabelFile::has_human_decision`]), boxes await a reviewer ([`LabelFile::awaits_review`]), or it
/// cannot be read (it may hold either).
pub fn is_protected(path: &Path) -> bool {
    match read(path) {
        Ok(None) => false,
        Ok(Some(f)) => f.has_human_decision() || f.awaits_review(),
        Err(e) => {
            tracing::debug!("retention: keeping the frame of an unreadable label file: {e:#}");
            true
        }
    }
}

/// Run `delete` under the write lock unless the label file at `path` is protected
/// ([`is_protected`]; None then), so a save cannot land between the check and the delete.
pub fn unless_protected<T>(path: &Path, delete: impl FnOnce() -> T) -> Option<T> {
    let _held = write_lock(path);
    if is_protected(path) {
        return None;
    }
    let out = delete();
    repeats::written(path);
    Some(out)
}

/// Every label file under `dir` with its frame (`<scene>/<camera>/<date>/<stem>.jpg`), in day
/// folder order then by name. Files whose frame image is gone are left out.
pub fn label_files(dir: &Path) -> Vec<(String, PathBuf)> {
    scan_labels(dir).files
}

/// What [`scan_labels`] found.
#[derive(Debug, Default)]
pub struct LabelScan {
    /// Label files with their frame, as [`label_files`] lists them.
    pub files: Vec<(String, PathBuf)>,
    /// Label files whose frame image is gone, the same way.
    pub orphans: Vec<(String, PathBuf)>,
    /// Scene, camera and day folders that exist but could not be listed: their frames are in
    /// neither list.
    pub unreadable: Vec<PathBuf>,
}

/// The label files under `dir`, and what a caller that must not shrink silently (an export) has
/// to tell about the ones it cannot use.
pub fn scan_labels(dir: &Path) -> LabelScan {
    scan_days(dir, layout::day_dirs_checked(dir))
}

/// [`label_files`] of one camera, without listing the other cameras' folders.
pub fn camera_label_files(dir: &Path, camera: &str) -> Vec<(String, PathBuf)> {
    scan_days(dir, layout::camera_day_dirs(dir, camera)).files
}

/// [`scan_labels`] of the `(day folders, folders not listed)` of [`layout::day_dirs_checked`].
fn scan_days(dir: &Path, (days, unreadable): (Vec<PathBuf>, Vec<PathBuf>)) -> LabelScan {
    let mut out = LabelScan {
        unreadable,
        ..LabelScan::default()
    };
    for day in days {
        let rd = match std::fs::read_dir(&day) {
            Ok(rd) => rd,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    out.unreadable.push(day);
                }
                continue;
            }
        };
        // The listing says what is a regular file, so a frame costs no look of its own.
        let (mut names, mut images) = (Vec::new(), HashSet::new());
        for name in rd
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
        {
            if name.ends_with(LABELS_SUFFIX) {
                names.push(name);
            } else if name.ends_with(".jpg") {
                images.insert(name);
            }
        }
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
            if layout::parse_rel(&rel).is_none() {
                continue;
            }
            // Not listed as a regular file: a symlink, or another case on a blind file system.
            let image = format!("{stem}.jpg");
            if images.contains(&image) || day.join(image).is_file() {
                out.files.push((rel, day.join(n)));
            } else {
                out.orphans.push((rel, day.join(n)));
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
            difficult: false,
            reject_reason: None,
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
        let f = prefill(&record(), &classes(), &[]);
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
    fn prefill_keeps_only_classes_of_the_list_in_its_spelling() {
        let mut r = empty_record();
        r.width = 640;
        r.height = 480;
        r.predictions = vec![
            det("pottedplant", 0.8, [10.0, 10.0, 50.0, 90.0]),
            det("parking meter", 0.7, [100.0, 100.0, 150.0, 140.0]),
            det("Person", 0.9, [200.0, 10.0, 260.0, 200.0]),
        ];
        r.below_threshold = vec![
            // The same box as the prediction once mapped: a duplicate.
            det("potted plant", 0.4, [10.0, 10.0, 50.0, 90.5]),
            det("motorbike", 0.4, [300.0, 300.0, 400.0, 400.0]),
            det("clock", 0.4, [500.0, 10.0, 520.0, 30.0]),
        ];
        let classes: Vec<String> = ["person", "potted plant", "motorcycle"]
            .map(String::from)
            .to_vec();
        let f = prefill(&r, &classes, &[]);
        let summary: Vec<(&str, Option<&str>, Option<Origin>)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.model_label.as_deref(), b.origin))
            .collect();
        assert_eq!(
            summary,
            [
                (
                    "potted plant",
                    Some("pottedplant"),
                    Some(Origin::Prediction)
                ),
                ("person", None, Some(Origin::Prediction)),
                (
                    "motorcycle",
                    Some("motorbike"),
                    Some(Origin::BelowThreshold)
                ),
            ]
        );
        assert!(f.boxes.iter().all(|b| b.state == BoxState::Pending));
        // Nothing of the list: no box.
        assert!(prefill(&r, &["deer".to_string()], &[]).boxes.is_empty());
        // A hidden class: left out in every spelling; the other classes kept.
        let hidden = ["Potted Plant".to_string()];
        let f = prefill(&r, &classes, &hidden);
        let labels: Vec<&str> = f.boxes.iter().map(|b| b.label.as_str()).collect();
        assert_eq!(labels, ["person", "motorcycle"]);
    }

    #[test]
    fn wanted_class_maps_against_the_whole_list_then_drops_hidden() {
        let classes: Vec<String> = ["person", "potted plant", "motorcycle"]
            .map(String::from)
            .to_vec();
        let hidden = ["potted plant".to_string()];
        assert_eq!(wanted_class("Person", &classes, &hidden), Some("person"));
        assert_eq!(
            wanted_class("motorbike", &classes, &hidden),
            Some("motorcycle")
        );
        assert_eq!(wanted_class("pottedplant", &classes, &hidden), None);
        assert_eq!(wanted_class("potted plant", &classes, &hidden), None);
        assert_eq!(wanted_class("clock", &classes, &hidden), None);
        assert!(is_hidden(" Potted Plant", &hidden));
        assert!(!is_hidden("person", &hidden));
    }

    #[test]
    fn reviewed_save_resolves_pending_and_checks_classes() {
        let now = "2026-10-22T12:00:00.000Z";
        let mut f = prefill(&record(), &classes(), &[]);
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
            difficult: false,
            reject_reason: None,
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
        let mut f = prefill(&record(), &classes(), &[]);
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
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        });
        let n = normalize(f, &classes(), 640, 480, "now").unwrap();
        assert_eq!(n.boxes.len(), 4);
        assert!(n.boxes.iter().all(|b| b.state == BoxState::Pending));
        assert_eq!(n.reviewed_at, None);
    }

    #[test]
    fn unsure_and_reject_reason_round_trip_and_normalize_keeps_them_where_they_belong() {
        let mut f = prefill(&record(), &classes(), &[]);
        f.status = Status::Reviewed;
        // Accepted + unsure; rejected + reason; accepted with a stray reason; rejected + unsure.
        f.boxes[0].state = BoxState::Accepted;
        f.boxes[0].difficult = true;
        f.boxes[1].state = BoxState::Rejected;
        f.boxes[1].reject_reason = Some(RejectReason::IrGlare);
        f.boxes[2].state = BoxState::Accepted;
        f.boxes[2].reject_reason = Some(RejectReason::Duplicate);
        f.boxes[3].state = BoxState::Rejected;
        f.boxes[3].difficult = true;
        let n = normalize(f, &classes(), 640, 480, "now").unwrap();
        let got: Vec<(bool, Option<RejectReason>)> = n
            .boxes
            .iter()
            .map(|b| (b.difficult, b.reject_reason.clone()))
            .collect();
        assert_eq!(
            got,
            [
                (true, None),
                (false, Some(RejectReason::IrGlare)),
                (false, None),
                (false, None),
            ]
        );
        let text = serde_json::to_string(&n).unwrap();
        assert!(
            text.contains(r#""difficult":true"#) && text.contains(r#""reject_reason":"ir_glare""#)
        );
        assert_eq!(text.matches("difficult").count(), 1, "false is not written");
        let back: LabelFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back, n);
        // A file written before the fields existed loads with them off.
        let old: LabelBox = serde_json::from_str(
            r#"{"label": "dog", "bbox": [1, 2, 30, 40], "source": "model", "state": "rejected"}"#,
        )
        .unwrap();
        assert!(!old.difficult && old.reject_reason.is_none());
        // An unknown reason (a newer build's) loads and is written back as it was.
        let newer: LabelBox = serde_json::from_str(
            r#"{"label": "dog", "bbox": [1, 2, 30, 40], "source": "model", "state": "rejected",
               "reject_reason": "aliens"}"#,
        )
        .unwrap();
        assert_eq!(
            newer.reject_reason,
            Some(RejectReason::Other("aliens".into()))
        );
        assert!(
            serde_json::to_string(&newer)
                .unwrap()
                .contains(r#""reject_reason":"aliens""#)
        );
        assert_eq!(RejectReason::parse("ir_glare"), RejectReason::IrGlare);
        // normalize keeps a short one and refuses a control character.
        let mut f = prefill(&record(), &classes(), &[]);
        f.boxes[0].state = BoxState::Rejected;
        f.boxes[0].reject_reason = Some(RejectReason::Other("aliens".into()));
        let n = normalize(f.clone(), &classes(), 640, 480, "now").unwrap();
        assert_eq!(
            n.boxes[0].reject_reason,
            Some(RejectReason::Other("aliens".into()))
        );
        f.boxes[0].reject_reason = Some(RejectReason::Other("a\u{7}".into()));
        assert!(normalize(f, &classes(), 640, 480, "now").is_err());
    }

    #[test]
    fn label_file_round_trips_and_atomic_write() {
        let dir = std::env::temp_dir().join(format!("bop-labels-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.labels.json");
        assert_eq!(read(&p).unwrap(), None);
        let mut f = normalize(
            prefill(&record(), &classes(), &[]),
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
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    #[test]
    fn merge_keeps_decisions_and_reopens_on_new_boxes() {
        let mut f = normalize(
            prefill(&record(), &classes(), &[]),
            &classes(),
            640,
            480,
            "t",
        )
        .unwrap();
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

        let mut g = prefill(&record(), &classes(), &[]);
        let r = merge_proposals(&mut g, vec![auto("cat", [10.0, 10.0, 50.0, 90.0])]);
        assert_eq!(r.added, 0, "another label on a pending box is dropped");
        assert!(!r.reopened);
    }

    #[test]
    fn a_save_never_turns_a_flagged_box_into_a_label() {
        let reviewed = |boxes: Vec<LabelBox>| LabelFile {
            status: Status::Reviewed,
            boxes,
            ..LabelFile::default()
        };
        let save = |f| normalize(f, &classes(), 640, 480, "now");
        let mut flagged = tests_support::auto_box("person");
        flagged.auto.triage = Some(Triage::Review);
        assert!(needs_decision(&flagged));
        let e = save(reviewed(vec![flagged.clone()])).unwrap_err();
        assert!(e.to_string().contains("1 box(es)"), "{e:#}");
        // Decided either way, it saves.
        for state in [BoxState::Accepted, BoxState::Rejected] {
            let mut b = flagged.clone();
            b.state = state;
            assert_eq!(save(reviewed(vec![b])).unwrap().boxes[0].state, state);
        }
        // Another status does not settle it, so it stays allowed.
        let skipped = LabelFile {
            status: Status::Skipped,
            ..reviewed(vec![flagged.clone()])
        };
        assert_eq!(save(skipped).unwrap().boxes[0].state, BoxState::Pending);
        // A class suggestion on a model box is flagged too; the same label is not.
        let mut suggestion = hint(Origin::Prediction, None, Some("cat"));
        assert!(needs_decision(&suggestion));
        suggestion.auto.auto_label = Some(suggestion.label.clone());
        assert!(!needs_decision(&suggestion));
        // An unflagged proposal is accepted as before, and a flagged hint is rejected by default:
        // neither is a label nobody chose.
        let plain = tests_support::auto_box("person");
        assert!(!needs_decision(&plain));
        let flagged_hint = hint(Origin::Suppressed, Some(Triage::Review), None);
        assert!(!needs_decision(&flagged_hint));
        let f = save(reviewed(vec![plain, flagged_hint])).unwrap();
        assert_eq!(f.boxes[0].state, BoxState::Accepted);
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
    }

    #[test]
    fn the_reject_the_rest_mark_stays_only_on_a_rejected_box() {
        let mut b = tests_support::auto_box("person");
        b.reason = Some(REST_REASON.into());
        b.state = BoxState::Rejected;
        let mut f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![b],
            ..LabelFile::default()
        };
        let n = normalize(f.clone(), &classes(), 640, 480, "now").unwrap();
        assert_eq!(n.boxes[0].reason.as_deref(), Some(REST_REASON));
        f.boxes[0].state = BoxState::Accepted;
        let n = normalize(f, &classes(), 640, 480, "now").unwrap();
        assert_eq!(n.boxes[0].reason, None);
    }

    #[test]
    fn frames_waiting_for_review_are_not_pruned() {
        let dir = std::env::temp_dir().join(format!("bop-awaits-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.labels.json");
        let pending = tests_support::auto_box("person");
        let mut dismissed = pending.clone();
        dismissed.state = BoxState::Rejected;
        dismissed.reason = Some(bulk::DISMISSED_REASON.into());
        let waiting = LabelFile {
            boxes: vec![pending, dismissed.clone()],
            ..LabelFile::default()
        };
        write(&path, &waiting).unwrap();
        assert!(waiting.awaits_review() && !waiting.has_human_decision());
        assert!(is_protected(&path));
        // Nothing left to look at: it ages out with its frame.
        let done = LabelFile {
            boxes: vec![dismissed],
            ..LabelFile::default()
        };
        write(&path, &done).unwrap();
        assert!(!is_protected(&path));
        std::fs::remove_dir_all(&dir).unwrap();
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
            likely: false,
            audit: false,
            spot_rejections: None,
            parked: None,
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
        let mut f = f;
        f.boxes[0].state = BoxState::Accepted;
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
    fn inside_share_is_the_part_of_the_smaller_box_inside_the_other() {
        let big = [0.0, 0.0, 100.0, 100.0];
        assert_eq!(inside_share(&big, &[10.0, 10.0, 20.0, 20.0]), 1.0);
        assert_eq!(
            inside_share(&[10.0, 10.0, 20.0, 20.0], &big),
            1.0,
            "either order"
        );
        // Half of the small box sticks out on the right.
        assert_eq!(inside_share(&big, &[95.0, 0.0, 105.0, 10.0]), 0.5);
        assert_eq!(inside_share(&big, &[200.0, 0.0, 210.0, 10.0]), 0.0);
        assert_eq!(inside_share(&big, &[5.0, 5.0, 5.0, 50.0]), 0.0, "empty box");
    }

    #[test]
    fn overlaps_checked_round_trips_old_files_load_and_a_new_box_ends_it() {
        let mut f = prefill(&record(), &classes(), &[]);
        f.status = Status::Reviewed;
        f.overlaps_checked = true;
        let n = normalize(f, &classes(), 640, 480, "t").unwrap();
        assert!(n.overlaps_checked, "normalize keeps it");
        let v = serde_json::to_value(&n).unwrap();
        assert_eq!(v["overlaps_checked"], true);
        let back: LabelFile = serde_json::from_value(v).unwrap();
        assert_eq!(back, n);
        // Off: not written, and a file from before the field loads with it off.
        let off = serde_json::to_string(&LabelFile::default()).unwrap();
        assert!(!off.contains("overlaps_checked"), "{off}");
        let old: LabelFile = serde_json::from_str(r#"{"status": "reviewed"}"#).unwrap();
        assert!(!old.overlaps_checked);
        // A proposal dropped changes nothing the reviewer saw; a box added does.
        let mut m = n.clone();
        let r = merge_proposals(&mut m, vec![auto("person", [10.0, 10.0, 50.0, 88.0])]);
        assert_eq!((r.added, m.overlaps_checked), (0, true));
        let r = merge_proposals(&mut m, vec![auto("deer", [200.0, 200.0, 260.0, 260.0])]);
        assert_eq!((r.added, m.overlaps_checked), (1, false));
    }

    #[test]
    fn checked_before_reopen_round_trips_old_files_load_and_a_save_ends_it() {
        let old: LabelFile = serde_json::from_str(r#"{"status":"reviewed","boxes":[]}"#).unwrap();
        assert!(!old.checked_before_reopen);
        let off = serde_json::to_string(&old).unwrap();
        assert!(!off.contains("checked_before_reopen"), "{off}");

        // The Review tab sends back what it loaded, on any status.
        let f = LabelFile {
            disagreement: Some("re-label by sam3: 1 new box".into()),
            reviewed_at: Some("t".into()),
            checked_before_reopen: true,
            ..LabelFile::default()
        };
        let body = serde_json::to_string(&f).unwrap();
        assert!(body.contains(r#""checked_before_reopen":true"#), "{body}");
        for status in [Status::Reviewed, Status::Unreviewed, Status::Skipped] {
            let mut sent: LabelFile = serde_json::from_str(&body).unwrap();
            assert!(sent.checked_before_reopen, "the file keeps it on disk");
            sent.status = status;
            let n = normalize(sent, &["car".to_string()], 640, 480, "now").unwrap();
            assert!(!n.checked_before_reopen, "{status:?}");
        }
    }

    #[test]
    fn too_small_rejects_are_ignore_regions_not_hard_negatives() {
        let mut tiny = tests_support::auto_box("person");
        tiny.state = BoxState::Rejected;
        tiny.reject_reason = Some(RejectReason::TooSmall);
        let mut wrong = tiny.clone();
        wrong.reject_reason = Some(RejectReason::FalsePositive);
        let mut unsure = tests_support::auto_box("dog");
        unsure.state = BoxState::Accepted;
        unsure.difficult = true;
        let mut pending = tiny.clone();
        pending.state = BoxState::Pending;
        assert!(tiny.is_ignore() && unsure.is_ignore());
        assert!(!wrong.is_ignore());
        assert!(!pending.is_ignore(), "too small counts once rejected");
        // A file reviewed before this rule: nothing to rewrite, the same JSON now reads this way.
        let old = r#"{"label":"person","bbox":[1,1,9,9],"source":"model","state":"rejected",
            "reject_reason":"too_small"}"#;
        assert!(serde_json::from_str::<LabelBox>(old).unwrap().is_ignore());
        let f = LabelFile {
            boxes: vec![tiny, wrong, unsure],
            ..LabelFile::default()
        };
        assert_eq!(f.rejected().count(), 2);
        let neg: Vec<_> = f.hard_negatives().collect();
        assert_eq!(neg.len(), 1);
        assert_eq!(neg[0].reject_reason, Some(RejectReason::FalsePositive));
    }

    #[test]
    fn hard_negatives_leave_out_duplicates() {
        let mut dup = tests_support::auto_box("car");
        dup.state = BoxState::Rejected;
        dup.reject_reason = Some(RejectReason::Duplicate);
        let mut wrong = dup.clone();
        wrong.reject_reason = Some(RejectReason::IrGlare);
        let mut plain = dup.clone();
        plain.reject_reason = None;
        let f = LabelFile {
            boxes: vec![dup, wrong, plain],
            ..LabelFile::default()
        };
        assert_eq!(f.rejected().count(), 3);
        assert_eq!(f.hard_negatives().count(), 2);
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
            difficult: false,
            reject_reason: None,
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

    /// A box a labelers run dismissed.
    fn dismissed(label: &str) -> LabelBox {
        LabelBox {
            state: BoxState::Rejected,
            reason: Some(bulk::DISMISSED_REASON.into()),
            ..tests_support::auto_box(label)
        }
    }

    #[test]
    fn a_dismissed_box_is_rejected_evidence_of_nothing() {
        let plain = LabelBox {
            state: BoxState::Rejected,
            ..tests_support::auto_box("dog")
        };
        let f = LabelFile {
            boxes: vec![dismissed("car"), plain],
            ..LabelFile::default()
        };
        assert_eq!(f.rejected().count(), 2);
        let neg: Vec<&str> = f.hard_negatives().map(|b| b.label.as_str()).collect();
        assert_eq!(neg, ["dog"]);
        // The reason alone is not a dismissal.
        let mut restored = dismissed("car");
        restored.state = BoxState::Accepted;
        assert!(!restored.is_dismissed());
    }

    #[test]
    fn a_decision_on_a_dismissed_box_clears_the_reason() {
        let save = |b: LabelBox, status: Status| {
            let f = LabelFile {
                status,
                boxes: vec![b],
                ..LabelFile::default()
            };
            normalize(f, &classes(), 640, 480, "t")
                .unwrap()
                .boxes
                .remove(0)
        };
        // Left alone: it stays dismissed.
        assert!(save(dismissed("car"), Status::Reviewed).is_dismissed());
        // Accepted.
        let mut b = dismissed("car");
        b.state = BoxState::Accepted;
        assert_eq!(save(b, Status::Reviewed).reason, None);
        // Pending again.
        let mut b = dismissed("car");
        b.state = BoxState::Pending;
        assert_eq!(save(b, Status::Unreviewed).reason, None);
        // Rejected by a reviewer with a reason: a negative like any other.
        let mut b = dismissed("car");
        b.reject_reason = Some(RejectReason::IrGlare);
        let b = save(b, Status::Reviewed);
        assert_eq!(b.reason, None);
        assert_eq!(
            LabelFile {
                boxes: vec![b],
                ..LabelFile::default()
            }
            .hard_negatives()
            .count(),
            1
        );
    }

    #[test]
    fn a_frame_with_a_dismissed_box_is_not_settled_by_its_triage() {
        let mut sure = tests_support::auto_box("dog");
        sure.auto.triage = Some(Triage::Accept);
        let mut f = LabelFile {
            boxes: vec![sure],
            ..LabelFile::default()
        };
        assert!(auto_accept_eligible(&f));
        f.boxes.push(dismissed("car"));
        assert!(!auto_accept_eligible(&f));
        // Another kind of reject does not keep it out.
        f.boxes[1].reason = Some(bulk::LOW_SCORE_REASON.into());
        assert!(auto_accept_eligible(&f));
    }

    #[test]
    fn likely_and_audit_stay_off_the_json_when_false_and_old_files_load() {
        let b = tests_support::auto_box("car");
        let v = serde_json::to_value(&b).unwrap();
        assert!(v.get("likely").is_none() && v.get("audit").is_none());
        let back: LabelBox = serde_json::from_value(v).unwrap();
        assert!(!back.auto.likely && !back.auto.audit);
        let mut set = tests_support::auto_box("car");
        set.auto.likely = true;
        let v = serde_json::to_value(&set).unwrap();
        assert_eq!(v["likely"], true);
        assert!(
            !serde_json::from_value::<LabelBox>(v)
                .unwrap()
                .auto
                .is_empty()
        );
    }

    #[test]
    fn a_box_cannot_be_both_likely_and_an_audit_sample() {
        let mut b = tests_support::auto_box("car");
        b.auto.likely = true;
        b.auto.audit = true;
        let f = LabelFile {
            boxes: vec![b],
            ..LabelFile::default()
        };
        let e = normalize(f, &classes(), 640, 480, "t").unwrap_err();
        assert!(format!("{e:#}").contains("likely and audit"), "{e:#}");
    }

    #[test]
    fn audit_verdicts_count_only_a_look_at_the_box() {
        let audit = |state: BoxState| {
            let mut b = tests_support::auto_box("car");
            b.state = state;
            b.auto.audit = true;
            b
        };
        assert_eq!(audit(BoxState::Pending).audit_verdict(), None);
        assert_eq!(audit(BoxState::Accepted).audit_verdict(), Some(true));
        assert_eq!(audit(BoxState::Rejected).audit_verdict(), Some(false));
        let mut rest = audit(BoxState::Rejected);
        rest.reason = Some(REST_REASON.into());
        assert_eq!(rest.audit_verdict(), Some(false), "the box was on screen");
        for reason in [bulk::LOW_SCORE_REASON, bulk::OTHER_CLASS_REASON] {
            let mut b = audit(BoxState::Rejected);
            b.reason = Some(reason.into());
            assert_eq!(b.audit_verdict(), None, "{reason}");
        }
        for why in [RejectReason::Duplicate, RejectReason::TooSmall] {
            let mut b = audit(BoxState::Rejected);
            b.reject_reason = Some(why);
            assert_eq!(b.audit_verdict(), None);
        }
        let mut other = audit(BoxState::Accepted);
        other.auto.audit = false;
        assert_eq!(other.audit_verdict(), None);
    }

    #[test]
    fn retention_keeps_a_persons_work_and_nothing_else() {
        let pending = LabelBox {
            label: "car".into(),
            bbox: [1.0, 1.0, 50.0, 50.0],
            source: Source::Auto,
            score: Some(0.4),
            state: BoxState::Pending,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        };
        let file = |b: Vec<LabelBox>| LabelFile {
            boxes: b,
            ..LabelFile::default()
        };
        let edit = |e: &dyn Fn(&mut LabelBox)| {
            let mut b = pending.clone();
            e(&mut b);
            file(vec![b])
        };
        let rejected = |why: &str| {
            edit(&|b| {
                b.state = BoxState::Rejected;
                b.reason = Some(why.to_string());
            })
        };
        // What a machine wrote: an empty run, pending boxes, its dismissals and bulk rejects.
        assert!(!LabelFile::default().has_human_decision());
        assert!(
            !LabelFile {
                auto_run: Some(AutoRun {
                    at: "t".into(),
                    labeler: "gdino".into(),
                    classes: Vec::new(),
                }),
                ..LabelFile::default()
            }
            .has_human_decision()
        );
        assert!(!file(vec![pending.clone()]).has_human_decision());
        for why in [
            bulk::LOW_SCORE_REASON,
            bulk::OTHER_CLASS_REASON,
            bulk::DISMISSED_REASON,
            bulk::SPOT_DISMISSED_REASON,
            stacks::DUPLICATE_REASON,
        ] {
            let mut f = rejected(why);
            if why == stacks::DUPLICATE_REASON {
                f.boxes[0].reject_reason = Some(RejectReason::Duplicate);
            }
            assert!(!f.has_human_decision(), "{why}");
        }
        assert!(
            !edit(&|b| {
                b.state = BoxState::Accepted;
                b.reason = Some(spots::PARKED_REASON.to_string());
            })
            .has_human_decision()
        );
        // What a person did.
        for f in [
            LabelFile {
                status: Status::Reviewed,
                ..LabelFile::default()
            },
            LabelFile {
                status: Status::Skipped,
                ..LabelFile::default()
            },
            LabelFile {
                reviewed_at: Some("t".into()),
                ..LabelFile::default()
            },
            LabelFile {
                empty_confirmed: true,
                ..LabelFile::default()
            },
            LabelFile {
                overlaps_checked: true,
                ..LabelFile::default()
            },
            edit(&|b| b.state = BoxState::Accepted),
            edit(&|b| b.state = BoxState::Rejected),
            edit(&|b| b.source = Source::Human),
            edit(&|b| b.orig_bbox = Some([0.0, 0.0, 9.0, 9.0])),
            edit(&|b| b.difficult = true),
            rejected(REST_REASON),
        ] {
            assert!(f.has_human_decision(), "{f:?}");
        }
    }

    #[test]
    fn a_frame_is_never_deleted_past_a_save_and_unreadable_labels_protect_it() {
        let dir = std::env::temp_dir().join(format!("bop-prot-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.labels.json");
        assert!(!is_protected(&p), "no label file");
        assert_eq!(unless_protected(&p, || 7), Some(7));
        write(&p, &LabelFile::default()).unwrap();
        assert!(!is_protected(&p));
        write(
            &p,
            &LabelFile {
                status: Status::Reviewed,
                ..LabelFile::default()
            },
        )
        .unwrap();
        assert_eq!(unless_protected(&p, || 7), None);
        std::fs::write(&p, b"garbled").unwrap();
        assert!(is_protected(&p));
        assert_eq!(
            path_of_frame(Path::new("d/x_1.jpg")),
            Path::new("d/x_1.labels.json")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A label file in the dataset layout under a new folder: (dataset folder, its path).
    fn laid_out_label_file(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("bop-labels-{tag}-{}", uuid::Uuid::new_v4()));
        let day = dir.join("ir/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        (dir, day.join(format!("x{LABELS_SUFFIX}")))
    }

    fn add_a_box(path: &Path) {
        update(path, |cur| {
            let mut f = cur.unwrap_or_default();
            f.boxes.push(tests_support::auto_box("person"));
            Ok(Some(f))
        })
        .unwrap();
    }

    /// The other process of [`two_processes_never_lose_each_others_writes`]; idle in a normal run.
    #[test]
    fn label_lock_child() {
        if let (Ok(path), Ok(n)) = (
            std::env::var("BOP_TEST_LABEL_LOCK_FILE"),
            std::env::var("BOP_TEST_LABEL_LOCK_WRITES"),
        ) {
            for _ in 0..n.parse::<usize>().unwrap() {
                add_a_box(Path::new(&path));
            }
        }
    }

    #[test]
    fn two_processes_never_lose_each_others_writes() {
        const WRITES: usize = 150;
        let (dir, path) = laid_out_label_file("two-processes");
        let children: Vec<_> = (0..2)
            .map(|_| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["collect::labels::tests::label_lock_child", "--exact"])
                    .env("BOP_TEST_LABEL_LOCK_FILE", &path)
                    .env("BOP_TEST_LABEL_LOCK_WRITES", WRITES.to_string())
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap()
            })
            .collect();
        for mut c in children {
            assert!(c.wait().unwrap().success());
        }
        let boxes = read(&path).unwrap().unwrap().boxes.len();
        assert_eq!(boxes, 2 * WRITES, "writes of the other process were lost");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_write_waits_for_the_lock_another_handle_holds() {
        let (dir, path) = laid_out_label_file("two-handles");
        std::fs::create_dir_all(&dir).unwrap();
        let held = std::fs::File::create(dir.join(".labels.lock")).unwrap();
        fs4::FileExt::try_lock(&held).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let p = path.clone();
        std::thread::spawn(move || {
            add_a_box(&p);
            tx.send(()).unwrap();
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(300))
                .is_err(),
            "the write went past a held lock"
        );
        fs4::FileExt::unlock(&held).unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().boxes.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A lock another process holds for long is busy, not broken: the next write must wait for it
    /// again. Treating a timeout as a failure ran every write of the next minute with no lock, and
    /// lost writes on a slow disk.
    #[test]
    fn a_lock_that_stayed_held_is_not_given_up_on() {
        let (dir, path) = laid_out_label_file("busy-lock");
        std::fs::create_dir_all(&dir).unwrap();
        let held = std::fs::File::create(dir.join(".labels.lock")).unwrap();
        fs4::FileExt::try_lock(&held).unwrap();
        assert!(folder_lock(&path, Duration::from_millis(20)).is_none());
        let listed = |p: &Path| {
            NO_FOLDER_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(&lock_path(p))
        };
        assert!(!listed(&path), "a busy lock was given up on");
        fs4::FileExt::unlock(&held).unwrap();
        assert!(folder_lock(&path, Duration::from_secs(5)).is_some());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_lock_sits_at_the_dataset_root_whatever_the_day() {
        let root = Path::new("data").join("dataset");
        let label = root.join("ir/cam/2026-10-22/x.labels.json");
        assert_eq!(lock_path(&label), root.join(LOCK_FILE));
        let label = root.join("day/ir/2026-10-22/x.labels.json");
        assert_eq!(lock_path(&label), root.join(LOCK_FILE), "a camera named ir");
        // Outside the layout: beside the file.
        let flat = Path::new("some").join("where").join("x.labels.json");
        assert_eq!(
            lock_path(&flat),
            Path::new("some").join("where").join(LOCK_FILE)
        );
    }

    #[test]
    fn a_save_goes_on_where_locks_fail_and_warns_once() {
        let (dir, path) = laid_out_label_file("no-lock");
        // Something at the lock's path that cannot be opened as a file.
        std::fs::create_dir(dir.join(LOCK_FILE)).unwrap();
        add_a_box(&path);
        add_a_box(&path);
        assert_eq!(read(&path).unwrap().unwrap().boxes.len(), 2);
        // The folder is remembered, so the warning above was the only one.
        assert!(!folder_lock_failed(&lock_path(&path)));
        let revision = revision(&path).unwrap();
        write_checked(&path, &LabelFile::default(), &revision)
            .unwrap()
            .unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_dataset_root_still_saves_in_a_writable_day_folder() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, path) = laid_out_label_file("read-only-root");
        let mode = |m: u32| std::fs::Permissions::from_mode(m);
        std::fs::set_permissions(&dir, mode(0o555)).unwrap();
        let denied = std::fs::File::create(dir.join("probe")).is_err();
        add_a_box(&path);
        std::fs::set_permissions(&dir, mode(0o755)).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().boxes.len(), 1);
        // Root writes anywhere: then the lock simply worked.
        if denied {
            assert!(!folder_lock_failed(&lock_path(&path)), "remembered");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_lock_held_too_long_gives_way_to_the_save_and_comes_back() {
        let (dir, path) = laid_out_label_file("held");
        let lock = lock_path(&path);
        let held = std::fs::File::create(&lock).unwrap();
        fs4::FileExt::try_lock(&held).unwrap();
        let wait = Duration::from_millis(30);
        let start = Instant::now();
        assert!(folder_lock(&path, wait).is_none());
        assert!(start.elapsed() >= wait);
        // Only one long wait: the next write still tries the lock, for a moment.
        let start = Instant::now();
        assert!(folder_lock(&path, 2 * HELD_WAIT).is_none());
        assert!(start.elapsed() >= HELD_WAIT);
        assert!(start.elapsed() < 2 * HELD_WAIT);
        add_a_box(&path);
        // The holder gone, the lock works again, and the next long wait is a full one.
        fs4::FileExt::unlock(&held).unwrap();
        assert!(folder_lock(&path, wait).is_some());
        assert!(!LONG_HELD.lock().unwrap().contains(&lock));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `TrainLock`'s history: a child spawned at that moment shares the lock's open file until it
    /// execs, so a lock released by closing alone can look held. Run without the guard's unlock,
    /// this fails.
    #[cfg(unix)]
    #[test]
    fn a_child_spawned_at_the_wrong_moment_does_not_keep_the_lock() {
        let (dir, path) = laid_out_label_file("fork");
        let stop = std::sync::atomic::AtomicBool::new(false);
        let mut held = 0;
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let _ = std::process::Command::new("/bin/sh")
                            .args(["-c", "true"])
                            .status();
                    }
                });
            }
            for _ in 0..5000 {
                let guard = WriteGuard {
                    folder: folder_lock(&path, Duration::ZERO),
                    _process: WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
                };
                if guard.folder.is_none() {
                    held += 1;
                    // Back on the lock for the next round.
                    NO_FOLDER_LOCK.lock().unwrap().remove(&lock_path(&path));
                }
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(
            held, 0,
            "of 5000 locks taken at once, this many looked held"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// EACCES on the folder must not read as a camera with no spot memory: the next write would
    /// forget its parked spots.
    #[cfg(unix)]
    #[test]
    fn a_spots_store_that_cannot_be_reached_is_an_error_not_an_empty_store() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, _) = laid_out_label_file("spots");
        let folder = dir.join(spots::SPOTS_DIR);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("cam.json"), b"{}").unwrap();
        let mode = |m: u32| std::fs::Permissions::from_mode(m);
        std::fs::set_permissions(&folder, mode(0o000)).unwrap();
        let denied = std::fs::read_dir(&folder).is_err();
        let read = spots::read(&dir, "cam");
        std::fs::set_permissions(&folder, mode(0o755)).unwrap();
        // Root reads everything: nothing to show then.
        assert!(!denied || read.is_err(), "{read:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `<stem>.jpg` and `<stem>.json` in `day`.
    fn frame_files(day: &Path, stem: &str) {
        std::fs::create_dir_all(day).unwrap();
        std::fs::write(day.join(format!("{stem}.jpg")), b"x").unwrap();
        let rec = serde_json::to_vec(&record()).unwrap();
        std::fs::write(day.join(format!("{stem}.json")), rec).unwrap();
    }

    /// Every way of asking for `rel` agrees: a walker of its own, the shared `walker`, and the
    /// one-off calls; the paths are canonical on both sides. Returns whether it resolved.
    fn agree(dir: &Path, walker: &mut layout::Walker<'_>, rel: &str) -> bool {
        let one_off = Frame::resolve(dir, rel);
        let fresh = Frame::resolve_in(&mut layout::Walker::new(dir), rel);
        assert_eq!(one_off, fresh, "{rel}: a new walker");
        assert_eq!(
            one_off,
            Frame::resolve_in(walker, rel),
            "{rel}: a used walker"
        );
        for ext in ["jpg", "json"] {
            assert_eq!(
                layout::resolve(dir, rel, ext),
                walker.resolve(rel, ext),
                "{rel}: .{ext}"
            );
        }
        one_off.is_some()
    }

    #[test]
    fn a_walker_answers_what_resolve_answers() {
        let tmp = std::env::temp_dir().join(format!("bop-walker-{}", uuid::Uuid::new_v4()));
        let dir = tmp.join("data");
        let day = dir.join("ir/cam/2026-10-22");
        frame_files(&day, "plain");
        // A leaf missing, a leaf that is a folder, a label file that is one.
        frame_files(&day, "no-sidecar");
        std::fs::remove_file(day.join("no-sidecar.json")).unwrap();
        frame_files(&day, "no-image");
        std::fs::remove_file(day.join("no-image.jpg")).unwrap();
        frame_files(&day, "folder-jpg");
        std::fs::remove_file(day.join("folder-jpg.jpg")).unwrap();
        std::fs::create_dir(day.join("folder-jpg.jpg")).unwrap();
        frame_files(&day, "folder-json");
        std::fs::remove_file(day.join("folder-json.json")).unwrap();
        std::fs::create_dir(day.join("folder-json.json")).unwrap();
        frame_files(&day, "folder-labels");
        std::fs::create_dir(day.join("folder-labels.labels.json")).unwrap();
        // A day "folder" that is a file.
        std::fs::write(dir.join("ir/cam/2026-10-25"), b"x").unwrap();

        let mut walker = layout::Walker::new(&dir);
        let at = |d: &str, stem: &str| format!("{d}/{stem}.jpg");
        assert!(agree(&dir, &mut walker, &at("ir/cam/2026-10-22", "plain")));
        for refused in [
            at("ir/cam/2026-10-22", "gone"),
            at("ir/cam/2026-10-22", "no-sidecar"),
            at("ir/cam/2026-10-22", "no-image"),
            at("ir/cam/2026-10-22", "folder-jpg"),
            at("ir/cam/2026-10-22", "folder-json"),
            at("ir/cam/2026-10-22", "folder-labels"),
            at("ir/cam/2026-01-01", "plain"),
            at("ir/nobody/2026-10-22", "plain"),
            at("ir/cam/2026-10-25", "plain"),
            "ir/cam/2026-10-22/plain.json".to_string(),
            "ir/cam/2026-10-22/plain.labels.json".to_string(),
            "../data/ir/cam/2026-10-22/plain.jpg".to_string(),
            "ir/../ir/cam/plain.jpg".to_string(),
        ] {
            assert!(!agree(&dir, &mut walker, &refused), "{refused}");
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = tmp.join("outside");
            frame_files(&outside.join("day"), "plain");
            frame_files(&outside.join("cam/2026-10-22"), "plain");
            frame_files(&outside.join("scene/cam/2026-10-22"), "plain");
            // Folders a symlink takes outside the root.
            symlink(outside.join("day"), dir.join("ir/cam/2026-10-23")).unwrap();
            symlink(outside.join("cam"), dir.join("ir/out-cam")).unwrap();
            symlink(outside.join("scene"), dir.join("night")).unwrap();
            // Folders a symlink keeps inside: they work, with the real path.
            symlink(&day, dir.join("ir/cam/2026-10-24")).unwrap();
            symlink(dir.join("ir/cam"), dir.join("ir/in-cam")).unwrap();
            symlink(dir.join("ir"), dir.join("day")).unwrap();
            // Leaves.
            frame_files(&day, "link-jpg");
            std::fs::remove_file(day.join("link-jpg.jpg")).unwrap();
            symlink(day.join("plain.jpg"), day.join("link-jpg.jpg")).unwrap();
            frame_files(&day, "link-json");
            std::fs::remove_file(day.join("link-json.json")).unwrap();
            symlink(day.join("plain.json"), day.join("link-json.json")).unwrap();
            frame_files(&day, "link-out");
            std::fs::remove_file(day.join("link-out.jpg")).unwrap();
            symlink(outside.join("day/plain.jpg"), day.join("link-out.jpg")).unwrap();
            frame_files(&day, "dangling");
            std::fs::remove_file(day.join("dangling.jpg")).unwrap();
            symlink(day.join("nothing"), day.join("dangling.jpg")).unwrap();
            frame_files(&day, "link-labels");
            symlink("/etc/hosts", day.join("link-labels.labels.json")).unwrap();

            let mut walker = layout::Walker::new(&dir);
            for refused in [
                at("ir/cam/2026-10-23", "plain"),
                at("ir/out-cam/2026-10-22", "plain"),
                at("night/cam/2026-10-22", "plain"),
                at("ir/cam/2026-10-22", "link-jpg"),
                at("ir/cam/2026-10-22", "link-json"),
                at("ir/cam/2026-10-22", "link-out"),
                at("ir/cam/2026-10-22", "dangling"),
                at("ir/cam/2026-10-22", "link-labels"),
            ] {
                assert!(!agree(&dir, &mut walker, &refused), "{refused}");
            }
            let real = dir
                .canonicalize()
                .unwrap()
                .join("ir/cam/2026-10-22/plain.jpg");
            for inside in [
                at("ir/cam/2026-10-24", "plain"),
                at("ir/in-cam/2026-10-22", "plain"),
                at("day/cam/2026-10-22", "plain"),
            ] {
                assert!(agree(&dir, &mut walker, &inside), "{inside}");
                let frame = Frame::resolve_in(&mut walker, &inside).unwrap();
                assert_eq!(frame.jpg, real);
            }
            // The root itself behind a symlink.
            let alias = tmp.join("alias");
            symlink(&dir, &alias).unwrap();
            let mut walker = layout::Walker::new(&alias);
            assert!(agree(
                &alias,
                &mut walker,
                &at("ir/cam/2026-10-22", "plain")
            ));
            assert!(!agree(
                &alias,
                &mut walker,
                &at("night/cam/2026-10-22", "plain")
            ));
        }

        // No root: nothing resolves.
        let none = tmp.join("no-such-folder");
        let mut walker = layout::Walker::new(&none);
        assert!(!agree(
            &none,
            &mut walker,
            &at("ir/cam/2026-10-22", "plain")
        ));
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn one_cameras_label_files_are_what_filtering_all_of_them_gives() {
        let tmp = std::env::temp_dir().join(format!("bop-camera-{}", uuid::Uuid::new_v4()));
        let dir = tmp.join("data");
        for (scene, camera, date, stem) in [
            ("ir", "cam", "2026-10-21", "a"),
            ("ir", "cam", "2026-10-22", "b"),
            ("day", "cam", "2026-10-21", "c"),
            ("ir", "other", "2026-10-21", "d"),
            ("ir", "BadName", "2026-10-21", "e"),
            ("ir", "cam", "not-a-date", "f"),
        ] {
            let day = dir.join(scene).join(camera).join(date);
            frame_files(&day, stem);
            write(
                &day.join(format!("{stem}{LABELS_SUFFIX}")),
                &LabelFile::default(),
            )
            .unwrap();
        }
        #[cfg(unix)]
        {
            // A camera or day folder that is a symlink is no folder of the dataset; a scene folder
            // is listed through one (`Frame::resolve` refuses its frames later).
            use std::os::unix::fs::symlink;
            let scene = tmp.join("outside/scene");
            let day = scene.join("cam/2026-10-21");
            frame_files(&day, "x");
            let file = day.join(format!("x{LABELS_SUFFIX}"));
            write(&file, &LabelFile::default()).unwrap();
            symlink(&scene, dir.join("night")).unwrap();
            symlink(scene.join("cam"), dir.join("ir/linked")).unwrap();
            symlink(&day, dir.join("ir/cam/2026-10-23")).unwrap();
        }
        let all = label_files(&dir);
        let through_scene = usize::from(cfg!(unix));
        assert_eq!(all.len(), 4 + through_scene, "{all:?}");
        for camera in ["cam", "other", "linked", "nobody", "BadName", "bad-name"] {
            let filtered: Vec<_> = all
                .iter()
                .filter(|(rel, _)| rel.split('/').nth(1) == Some(camera))
                .cloned()
                .collect();
            assert_eq!(camera_label_files(&dir, camera), filtered, "{camera}");
        }
        assert_eq!(camera_label_files(&dir, "cam").len(), 3 + through_scene);
        // A name from a client reaches no path.
        for bad in ["..", "../..", "cam/..", "", "."] {
            assert!(camera_label_files(&dir, bad).is_empty(), "{bad}");
        }
        assert!(camera_label_files(&tmp.join("no-such-folder"), "cam").is_empty());
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn a_label_file_keeps_its_frame_by_the_listing_or_by_a_look() {
        let dir = std::env::temp_dir().join(format!("bop-listing-{}", uuid::Uuid::new_v4()));
        let day = dir.join("ir/cam/2026-10-22");
        frame_files(&day, "kept");
        frame_files(&day, "orphan");
        std::fs::remove_file(day.join("orphan.jpg")).unwrap();
        frame_files(&day, "folder");
        std::fs::remove_file(day.join("folder.jpg")).unwrap();
        std::fs::create_dir(day.join("folder.jpg")).unwrap();
        frame_files(&day, "linked");
        std::fs::remove_file(day.join("linked.jpg")).unwrap();
        for stem in ["kept", "orphan", "folder", "linked"] {
            write(
                &day.join(format!("{stem}{LABELS_SUFFIX}")),
                &LabelFile::default(),
            )
            .unwrap();
        }
        let linked = cfg!(unix);
        #[cfg(unix)]
        std::os::unix::fs::symlink(day.join("kept.jpg"), day.join("linked.jpg")).unwrap();
        let scan = scan_labels(&dir);
        let rels = |l: &[(String, PathBuf)]| -> Vec<String> {
            l.iter()
                .map(|(r, _)| r.rsplit('/').next().unwrap().to_string())
                .collect()
        };
        // A symlinked image is a file to `is_file` (Frame::resolve refuses it later).
        let (mut files, mut orphans) = (vec!["kept.jpg"], vec!["folder.jpg", "orphan.jpg"]);
        if linked {
            files.push("linked.jpg");
        } else {
            orphans.push("linked.jpg");
        }
        assert_eq!(rels(&scan.files), files);
        assert_eq!(rels(&scan.orphans), orphans);
        let _ = std::fs::remove_dir_all(dir);
    }
}
