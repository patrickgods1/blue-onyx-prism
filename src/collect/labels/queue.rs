//! The review queue: the frames of `review-queue.jsonl` (and, with [`QueueSource::All`], every
//! indexed frame), filtered by bucket, scene, camera, object class ([`frame_labels`]), review
//! status and auto-label state ([`AutoFilter`]), and sorted by
//! uncertainty ([`Uncertainty`]): frames with likely boxes first, then fewer flagged boxes, then the
//! near-threshold margin, verifier disagreement, the number of suppressed boxes, then newest. Also the counts the Review tab shows, and
//! [`labelled`], every label file those filters select whichever queue it is in.
//!
//! The walks read a day folder at a time ([`by_day`], [`over_days`], one [`Walker`] per day folder)
//! on [`parallel::walk_threads`] threads, and merge in a fixed order: the answers do not depend on
//! the threads. Nothing here writes.

use super::super::layout::{self, REVIEW_QUEUE_FILE, ReviewLine, Walker};
use super::super::record::{Bucket, IndexLine, Record};
use super::bulk::Wanted;
use super::{BoxState, Frame, LabelFile, Status, Triage};
use crate::parallel;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Which frames the queue draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueSource {
    /// `review-queue.jsonl` only (frames sent from the Collect tab).
    #[default]
    Queue,
    /// The queue plus every frame of the index files.
    All,
}

/// Status filter: one status, or any.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusFilter {
    Is(Status),
    Any,
}

impl Default for StatusFilter {
    fn default() -> Self {
        StatusFilter::Is(Status::Unreviewed)
    }
}

/// What the auto-labeler left on a frame (`auto=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AutoFilter {
    /// At least one pending box triaged `review` (auto boxes, annotated model boxes, class
    /// suggestions).
    Review,
    /// The triage alone settles the frame ([`super::auto_accept_eligible`]).
    Accept,
    /// Auto-labelled ([`LabelFile::auto_run`]) and no box carries auto evidence: the run found
    /// nothing of the classes it looked for ([`super::AutoRun::classes`]; empty = the whole
    /// class list of the time), which says nothing about the other classes.
    Nothing,
    /// Never auto-labelled: no `auto_run` and no box with auto evidence (frames without a label
    /// file too).
    #[serde(rename = "none")]
    Never,
    /// Repeat-only: spot edits settled every box ([`super::repeats::repeats_only`]).
    Repeats,
    /// Stacked or overlapping boxes no reviewer has looked at ([`super::stacks::needs_look`]), any
    /// review status. With a class list ([`Filters::wanted`]) rejected and unwanted boxes do not
    /// count; [`AutoFilter::matches`] has none, so there every label counts. A frame without a
    /// label file is not listed: it is pre-filled, and a save rejects its pending duplicates.
    Overlaps,
    /// A pending box with [`AGREED_LABELERS`] or more `labelers` ([`agreed_boxes`]): likely real and
    /// still unsettled, so a decided box never counts. With a class list ([`Filters::wanted`]) boxes
    /// of unwanted classes do not count either; [`AutoFilter::matches`] has none, so there every
    /// label counts.
    Both,
    /// At least one box a labelers run dismissed ([`LabelBox::is_dismissed`]), pending boxes or
    /// not: a frame left with only such boxes has nothing else to find it by.
    Dismissed,
}

/// How many labelers make a box agreed ([`AutoFilter::Both`]).
const AGREED_LABELERS: usize = 2;

impl AutoFilter {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "review" => Some(AutoFilter::Review),
            "accept" => Some(AutoFilter::Accept),
            "nothing" => Some(AutoFilter::Nothing),
            "none" => Some(AutoFilter::Never),
            "repeats" => Some(AutoFilter::Repeats),
            "overlaps" => Some(AutoFilter::Overlaps),
            "both" => Some(AutoFilter::Both),
            "dismissed" => Some(AutoFilter::Dismissed),
            _ => None,
        }
    }

    /// Whether a frame with `labels` (None = no label file) passes.
    pub fn matches(self, labels: Option<&LabelFile>) -> bool {
        let evidence = labels.is_some_and(|l| l.boxes.iter().any(|b| !b.auto.is_empty()));
        let ran = labels.is_some_and(|l| l.auto_run.is_some());
        match self {
            AutoFilter::Review => labels.is_some_and(has_auto_review),
            AutoFilter::Accept => labels.is_some_and(super::auto_accept_eligible),
            AutoFilter::Nothing => ran && !evidence,
            AutoFilter::Never => !ran && !evidence,
            AutoFilter::Repeats => labels.is_some_and(super::repeats::repeats_only),
            AutoFilter::Overlaps => labels.is_some_and(|l| super::stacks::needs_look(l, None)),
            AutoFilter::Both => labels.is_some_and(|l| agreed_boxes(l, None) > 0),
            AutoFilter::Dismissed => {
                labels.is_some_and(|l| l.boxes.iter().any(super::LabelBox::is_dismissed))
            }
        }
    }
}

/// A pending box triaged `review`.
fn has_auto_review(l: &LabelFile) -> bool {
    l.boxes
        .iter()
        .any(|b| b.state == BoxState::Pending && b.auto.triage == Some(Triage::Review))
}

/// The pending boxes of `l` with [`AGREED_LABELERS`] or more `labelers`, those of unwanted classes
/// left out when `wanted` is given.
fn agreed_boxes(l: &LabelFile, wanted: Option<&Wanted<'_>>) -> usize {
    l.boxes
        .iter()
        .filter(|b| b.state == BoxState::Pending && b.auto.labelers.len() >= AGREED_LABELERS)
        .filter(|b| wanted.is_none_or(|w| w.class_of(&b.label).is_some()))
        .count()
}

/// Queue filters (`GET /v1/review/next?...`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filters {
    pub bucket: Option<Bucket>,
    pub scene: Option<String>,
    pub camera: Option<String>,
    /// Only frames with any of these classes ([`frame_labels`]); empty = any frame.
    pub label: BTreeSet<String>,
    pub status: StatusFilter,
    pub source: QueueSource,
    /// Only frames in this auto-label state.
    pub auto: Option<AutoFilter>,
    /// `collect.classes` and `collect.hidden_classes`, for [`AutoFilter::Overlaps`] and
    /// [`AutoFilter::Both`] (not from the query: the caller's config). None = every label is wanted.
    pub wanted: Option<(Vec<String>, Vec<String>)>,
}

impl Filters {
    /// From query parameters (`bucket`, `scene`, `camera`, `label` = a comma list, any of
    /// them, empty items ignored, `status` = `unreviewed|reviewed|
    /// skipped|any`, `source` = `queue|all`, `auto` = `review|accept|nothing|none|repeats|
    /// overlaps|both|dismissed`); empty values mean "any". Errors name the field.
    pub fn from_query(q: &BTreeMap<String, String>) -> Result<Self, String> {
        let get = |k: &str| q.get(k).map(|v| v.trim()).filter(|v| !v.is_empty());
        let bucket = match get("bucket") {
            None => None,
            Some(b) => Some(Bucket::parse(b).ok_or_else(|| {
                format!(
                    "bucket: '{b}' is not one of {}",
                    Bucket::ALL.map(|b| b.as_str()).join(", ")
                )
            })?),
        };
        let scene = match get("scene") {
            None => None,
            Some(s) if layout::SCENES.contains(&s) => Some(s.to_string()),
            Some(s) => return Err(format!("scene: '{s}' is not day, night or ir")),
        };
        let status = match get("status") {
            None => StatusFilter::default(),
            Some("any") => StatusFilter::Any,
            Some(s) => StatusFilter::Is(Status::parse(s).ok_or_else(|| {
                format!("status: '{s}' is not unreviewed, reviewed, skipped or any")
            })?),
        };
        let mut label = BTreeSet::new();
        for l in get("label")
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            if super::super::config::validate_classes(&[l.to_string()]).is_err() {
                return Err(format!(
                    "label: '{}' is not a valid class name (1 to {} characters, no control \
                     characters); pick a class from the list",
                    l.escape_debug(),
                    super::super::config::MAX_CLASS_LEN
                ));
            }
            label.insert(l.to_string());
        }
        let source = match get("source") {
            None | Some("queue") => QueueSource::Queue,
            Some("all") => QueueSource::All,
            Some(s) => return Err(format!("source: '{s}' is not queue or all")),
        };
        let auto = match get("auto") {
            None => None,
            Some(a) => Some(AutoFilter::parse(a).ok_or_else(|| {
                format!(
                    "auto: '{a}' is not review, accept, nothing, none, repeats, overlaps, both \
                     or dismissed"
                )
            })?),
        };
        Ok(Self {
            bucket,
            scene,
            camera: get("camera").map(str::to_string),
            label,
            status,
            source,
            auto,
            wanted: None,
        })
    }

    /// [`Self::wanted`] as [`Wanted`].
    fn wanted(&self) -> Option<Wanted<'_>> {
        self.wanted
            .as_ref()
            .map(|(classes, hidden)| Wanted { classes, hidden })
    }

    /// Whether a frame with `labels` passes the auto filter.
    pub(super) fn auto_keeps(&self, labels: Option<&LabelFile>) -> bool {
        match self.auto {
            None => true,
            Some(AutoFilter::Overlaps) => {
                let wanted = self.wanted();
                labels.is_some_and(|l| super::stacks::needs_look(l, wanted.as_ref()))
            }
            Some(AutoFilter::Both) => {
                labels.is_some_and(|l| agreed_boxes(l, self.wanted().as_ref()) > 0)
            }
            Some(a) => a.matches(labels),
        }
    }

    /// The sort strength of a frame with `labels`: its strongest overlap suggestion under
    /// `auto=overlaps`, else 0 (no effect on the order).
    pub(super) fn overlap_strength(&self, labels: Option<&LabelFile>) -> u32 {
        match (self.auto, labels) {
            (Some(AutoFilter::Overlaps), Some(l)) => {
                super::overlaps::frame_strength(l, self.wanted().as_ref())
            }
            _ => 0,
        }
    }

    /// The sort count of a frame with `labels`: its agreed boxes under `auto=both`, else 0 (no
    /// effect on the order).
    pub(super) fn agreed_boxes(&self, labels: Option<&LabelFile>) -> u32 {
        match (self.auto, labels) {
            (Some(AutoFilter::Both), Some(l)) => {
                u32::try_from(agreed_boxes(l, self.wanted().as_ref())).unwrap_or(u32::MAX)
            }
            _ => 0,
        }
    }
}

/// The labels of the boxes of `l` that are not rejected.
pub(super) fn file_labels(l: &LabelFile) -> BTreeSet<String> {
    l.boxes
        .iter()
        .filter(|b| b.state != super::BoxState::Rejected)
        .map(|b| b.label.clone())
        .collect()
}

/// The object classes of a frame: with a label file, the labels of its boxes that are not
/// rejected (accepted, added and still pending ones); without one, the labels of the reply's
/// `predictions`.
pub fn frame_labels(rec: &Record, labels: Option<&LabelFile>) -> BTreeSet<String> {
    match labels {
        Some(l) => file_labels(l),
        None => rec.predictions.iter().map(|d| d.label.clone()).collect(),
    }
}

/// A frame the queue may list, before its sidecar is read.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub rel: String,
    pub scene: String,
    pub camera: String,
    pub buckets: Vec<Bucket>,
    /// Sent to review from the Collect tab.
    pub queued: bool,
}

/// The uncertainty of a frame from its sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Uncertainty {
    /// Smallest `|top score - threshold|` over the labels (top over `raw`, which holds the
    /// suppressed boxes, and `below_threshold`); None = no box at all.
    pub margin: Option<f32>,
    /// The verifier disagreed with the model (bucket `disagreement`, or a rejected candidate).
    pub disagreement: bool,
    /// Boxes a rule or the verifier removed.
    pub suppressed: usize,
    /// The label file's `disagreement`: a re-label with a trained model disagrees with the
    /// reviewed labels. These frames come first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relabel: Option<String>,
    /// Pending auto-labeler boxes triaged `review`.
    pub auto_review: usize,
    /// Pending boxes likely real ([`super::AutoInfo::likely`]). These frames come first, most such
    /// boxes first, so the queue opens on frames worth a look rather than on the most false boxes.
    #[serde(skip_serializing_if = "is_zero")]
    pub likely_boxes: u32,
    /// Spot edits settled every box ([`super::repeats::repeats_only`]): these frames come last.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub repeats_only: bool,
    /// `auto=overlaps` only: the strongest suggestion on the frame
    /// ([`super::overlaps::frame_strength`], 0 = none). These frames come first, strongest first.
    #[serde(skip_serializing_if = "is_zero")]
    pub overlap_strength: u32,
    /// `auto=both` only: the frame's pending boxes [`AGREED_LABELERS`] or more labelers found. These
    /// frames come first, most agreed boxes first.
    #[serde(skip_serializing_if = "is_zero")]
    pub agreed_boxes: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl Uncertainty {
    pub fn of(rec: &Record) -> Self {
        let mut top: BTreeMap<&str, f32> = BTreeMap::new();
        for d in rec.raw.iter().chain(&rec.below_threshold) {
            let t = top.entry(d.label.as_str()).or_insert(d.score);
            *t = t.max(d.score);
        }
        let margin = top
            .values()
            .map(|s| (s - rec.threshold).abs())
            .min_by(f32::total_cmp);
        let disagreement = rec.buckets.contains(&Bucket::Disagreement)
            || rec.matched.contains(&Bucket::Disagreement)
            || rec
                .verify
                .as_ref()
                .is_some_and(|v| v.candidates.iter().any(|c| c.outcome == "rejected"));
        Self {
            margin,
            disagreement,
            suppressed: rec.suppressed.len(),
            relabel: None,
            auto_review: 0,
            likely_boxes: 0,
            repeats_only: false,
            overlap_strength: 0,
            agreed_boxes: 0,
        }
    }

    /// Add what the frame's label file says: its re-label disagreement, the pending auto-labeler
    /// boxes triaged `review` and the likely ones among them.
    pub fn with_labels(mut self, labels: Option<&LabelFile>) -> Self {
        if let Some(l) = labels {
            self.relabel = l.disagreement.clone();
            self.auto_review = l
                .boxes
                .iter()
                .filter(|b| {
                    b.state == super::BoxState::Pending
                        && b.auto.triage == Some(super::Triage::Review)
                })
                .count();
            let likely = l
                .boxes
                .iter()
                .filter(|b| b.state == super::BoxState::Pending && b.auto.likely)
                .count();
            self.likely_boxes = u32::try_from(likely).unwrap_or(u32::MAX);
            self.repeats_only = super::repeats::repeats_only(l);
        }
        self
    }

    /// Sort key, most uncertain first: the strongest overlap suggestion (`auto=overlaps` only), the
    /// most agreed boxes (`auto=both` only), repeat-only frames last of all, then a re-label
    /// disagreement, then the most likely boxes, fewer flagged boxes (17 flagged boxes on a frame
    /// with no likely one are mostly false: an empty yard must not open the queue), then the margin
    /// in hundredths (no box last), verifier disagreement, more suppressed boxes.
    pub(super) fn key(&self) -> SortKey {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a margin is a difference of scores in 0..=1, so 0..=100 hundredths"
        )]
        let m = self
            .margin
            .map_or(u32::MAX, |m| (m.clamp(0.0, 1.0) * 100.0).round() as u32);
        (
            std::cmp::Reverse(self.overlap_strength),
            std::cmp::Reverse(self.agreed_boxes),
            self.repeats_only,
            self.relabel.is_none(),
            std::cmp::Reverse(self.likely_boxes),
            self.auto_review,
            m,
            !self.disagreement,
            std::cmp::Reverse(self.suppressed),
        )
    }
}

/// What [`Uncertainty::key`] sorts by: a lexicographic key reads best as one tuple.
pub(super) type SortKey = (
    std::cmp::Reverse<u32>,
    std::cmp::Reverse<u32>,
    bool,
    bool,
    std::cmp::Reverse<u32>,
    usize,
    u32,
    bool,
    std::cmp::Reverse<usize>,
);

/// One frame of the sorted queue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Item {
    pub frame: String,
    pub timestamp: String,
    pub scene: String,
    pub camera: String,
    pub model: String,
    pub buckets: Vec<Bucket>,
    pub status: Status,
    pub queued: bool,
    #[serde(flatten)]
    pub uncertainty: Uncertainty,
}

/// Sort most uncertain first ([`Uncertainty::key`]), then newest, then by path.
pub fn sort_items(items: &mut [Item]) {
    items.sort_by(|a, b| {
        a.uncertainty
            .key()
            .cmp(&b.uncertainty.key())
            .then_with(|| b.timestamp.cmp(&a.timestamp))
            .then_with(|| a.frame.cmp(&b.frame))
    });
}

/// Review counts over the frames matching the filters (the status and auto filters aside).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Counts {
    pub total: u64,
    pub reviewed: u64,
    pub skipped: u64,
    /// Not reviewed or skipped yet.
    pub remaining: u64,
    pub empty_confirmed: u64,
    /// Accepted and added boxes of reviewed frames, per label.
    pub per_class: BTreeMap<String, u64>,
    /// Rejected model boxes of reviewed frames (hard-negative evidence; duplicates are not).
    pub rejected_boxes: u64,
    /// Frames with a pending box triaged `review` ([`AutoFilter::Review`]).
    pub auto_review: u64,
    /// Frames the auto-labeler's triage alone settles ([`AutoFilter::Accept`]).
    pub auto_accept: u64,
    /// Repeat-only frames ([`AutoFilter::Repeats`]).
    pub repeats_only: u64,
    /// Audit samples of the dismissal filter on reviewed frames a reviewer found real
    /// ([`super::LabelBox::audit_verdict`]).
    pub audit_accepted: u64,
    /// Audit samples on reviewed frames a reviewer found not real.
    pub audit_rejected: u64,
}

impl Counts {
    /// Add the counts of another part of the walk.
    fn merge(&mut self, other: Self) {
        let Self {
            total,
            reviewed,
            skipped,
            remaining,
            empty_confirmed,
            per_class,
            rejected_boxes,
            auto_review,
            auto_accept,
            repeats_only,
            audit_accepted,
            audit_rejected,
        } = other;
        self.total += total;
        self.reviewed += reviewed;
        self.skipped += skipped;
        self.remaining += remaining;
        self.empty_confirmed += empty_confirmed;
        for (class, n) in per_class {
            *self.per_class.entry(class).or_default() += n;
        }
        self.rejected_boxes += rejected_boxes;
        self.auto_review += auto_review;
        self.auto_accept += auto_accept;
        self.repeats_only += repeats_only;
        self.audit_accepted += audit_accepted;
        self.audit_rejected += audit_rejected;
    }

    fn add(&mut self, labels: Option<&LabelFile>) {
        self.total += 1;
        match labels.map(|l| l.status) {
            Some(Status::Reviewed) => self.reviewed += 1,
            Some(Status::Skipped) => self.skipped += 1,
            _ => self.remaining += 1,
        }
        self.auto_review += u64::from(AutoFilter::Review.matches(labels));
        self.auto_accept += u64::from(AutoFilter::Accept.matches(labels));
        self.repeats_only += u64::from(AutoFilter::Repeats.matches(labels));
        if let Some(l) = labels.filter(|l| l.status == Status::Reviewed) {
            self.empty_confirmed += u64::from(l.empty_confirmed);
            for b in l.training() {
                *self.per_class.entry(b.label.clone()).or_default() += 1;
            }
            self.rejected_boxes += l.hard_negatives().count() as u64;
            for v in l.boxes.iter().filter_map(super::LabelBox::audit_verdict) {
                if v {
                    self.audit_accepted += 1;
                } else {
                    self.audit_rejected += 1;
                }
            }
        }
    }
}

/// The sorted queue and its counts.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Queue {
    pub items: Vec<Item>,
    pub counts: Counts,
    /// Cameras among the candidates (for the filter list).
    pub cameras: Vec<String>,
    /// Classes ([`frame_labels`]) of the frames the bucket, scene and camera filters keep, before
    /// the class filter (for the filter list).
    pub labels: Vec<String>,
    /// Frames left out: unreadable sidecar or label file.
    pub unreadable: u64,
}

/// Add the frames of review lines `lines` to `out`, each once with those in `seen`.
pub(super) fn review_candidates(
    lines: Vec<ReviewLine>,
    seen: &mut BTreeSet<String>,
    out: &mut Vec<Candidate>,
) {
    for l in lines {
        if layout::parse_rel(&l.frame).is_some() && seen.insert(l.frame.clone()) {
            out.push(Candidate {
                rel: l.frame,
                scene: l.scene,
                camera: l.camera,
                buckets: l.buckets,
                queued: true,
            });
        }
    }
}

/// Add the frames of `lines`, the index lines of a day folder of `date`, to `out`, each once with
/// those in `seen`.
pub(super) fn index_candidates(
    lines: Vec<IndexLine>,
    date: &str,
    seen: &mut BTreeSet<String>,
    out: &mut Vec<Candidate>,
) {
    for line in lines {
        let rel = format!("{}/{}/{date}/{}.jpg", line.scene, line.camera, line.id);
        if layout::parse_rel(&rel).is_some() && seen.insert(rel.clone()) {
            out.push(Candidate {
                rel,
                scene: line.scene,
                camera: line.camera,
                buckets: line.buckets,
                queued: false,
            });
        }
    }
}

/// The candidates of `dir`: the review queue's frames first, then (with `All`) every indexed
/// frame, each once.
pub fn candidates(dir: &Path, source: QueueSource) -> Vec<Candidate> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    review_candidates(
        layout::read_review(&dir.join(REVIEW_QUEUE_FILE)),
        &mut seen,
        &mut out,
    );
    if source == QueueSource::All {
        for day in layout::day_dirs(dir) {
            let date = day
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let lines = layout::read_index(&day.join(layout::INDEX_FILE));
            index_candidates(lines, &date, &mut seen, &mut out);
        }
    }
    out
}

/// The day folder of a frame's `rel` (from the path: a review line's own `camera` and `scene` may
/// say something else).
pub(super) fn day_of(rel: &str) -> &str {
    rel.rsplit_once('/').map_or(rel, |(day, _)| day)
}

/// `candidates` per day folder, in the order each folder first comes up.
pub(super) fn by_day(candidates: Vec<Candidate>) -> Vec<Vec<Candidate>> {
    let mut at: HashMap<String, usize> = HashMap::new();
    let mut days: Vec<Vec<Candidate>> = Vec::new();
    for c in candidates {
        let i = *at.entry(day_of(&c.rel).to_string()).or_insert_with(|| {
            days.push(Vec::new());
            days.len() - 1
        });
        if let Some(day) = days.get_mut(i) {
            day.push(c);
        }
    }
    days
}

/// What one day folder adds to a [`Queue`].
#[derive(Default)]
struct DayQueue {
    items: Vec<Item>,
    counts: Counts,
    labels: BTreeSet<String>,
    unreadable: u64,
}

/// The candidates of one day folder, read: the part of [`build`] that touches files.
fn build_day(mut walker: Walker<'_>, day: &[Candidate], f: &Filters) -> DayQueue {
    let mut q = DayQueue::default();
    for c in day {
        let Some(frame) = Frame::resolve_in(&mut walker, &c.rel) else {
            continue;
        };
        let (Ok(rec), Ok(labels)) = (frame.record(), frame.labels()) else {
            q.unreadable += 1;
            continue;
        };
        let classes = frame_labels(&rec, labels.as_ref());
        let keep = f.label.is_empty() || f.label.iter().any(|l| classes.contains(l));
        q.labels.extend(classes);
        if !keep {
            continue;
        }
        q.counts.add(labels.as_ref());
        let status = labels.as_ref().map_or(Status::Unreviewed, |l| l.status);
        if let StatusFilter::Is(s) = f.status
            && s != status
        {
            continue;
        }
        if !f.auto_keeps(labels.as_ref()) {
            continue;
        }
        let uncertainty = Uncertainty {
            overlap_strength: f.overlap_strength(labels.as_ref()),
            agreed_boxes: f.agreed_boxes(labels.as_ref()),
            ..Uncertainty::of(&rec).with_labels(labels.as_ref())
        };
        q.items.push(Item {
            frame: c.rel.clone(),
            timestamp: rec.timestamp.clone(),
            scene: c.scene.clone(),
            camera: c.camera.clone(),
            model: rec.model.clone(),
            buckets: c.buckets.clone(),
            status,
            queued: c.queued,
            uncertainty,
        });
    }
    q
}

/// Build the queue of `dir` for `f`. Frames whose image or sidecar is gone (retention) are
/// left out silently; a sidecar or label file that does not parse counts as `unreadable`.
pub fn build(dir: &Path, f: &Filters) -> Queue {
    build_on(dir, f, None)
}

/// [`build`] on `threads` threads (None: as many as a walk of that size takes).
fn build_on(dir: &Path, f: &Filters, threads: Option<usize>) -> Queue {
    let mut q = Queue::default();
    let mut cameras = BTreeSet::new();
    let mut kept = Vec::new();
    for c in candidates(dir, f.source) {
        cameras.insert(c.camera.clone());
        if f.scene.as_ref().is_some_and(|s| *s != c.scene)
            || f.camera.as_ref().is_some_and(|s| *s != c.camera)
            || f.bucket.is_some_and(|b| !c.buckets.contains(&b))
        {
            continue;
        }
        kept.push(c);
    }
    let threads = threads.unwrap_or_else(|| parallel::walk_threads_over(kept.len()));
    let days = by_day(kept);
    let walker = Walker::new(dir);
    let parts = parallel::map_all(days.len(), threads, &|i| {
        let day = days.get(i).map_or(&[][..], Vec::as_slice);
        build_day(walker.clone(), day, f)
    });
    let mut seen_labels = BTreeSet::new();
    for part in parts {
        q.items.extend(part.items);
        q.counts.merge(part.counts);
        seen_labels.extend(part.labels);
        q.unreadable += part.unreadable;
    }
    sort_items(&mut q.items);
    q.cameras = cameras.into_iter().collect();
    q.labels = seen_labels.into_iter().collect();
    q
}

/// Reviewed boxes of one class ([`Tally::coverage`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ClassCoverage {
    pub boxes: u64,
    /// scene -> boxes.
    pub by_scene: BTreeMap<String, u64>,
    /// camera -> boxes.
    pub by_camera: BTreeMap<String, u64>,
}

/// Where a folder's review stands, in one walk over its frames ([`tally`]): what the Train area's
/// status strip and the Dataset tab's coverage table show.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Tally {
    /// Every frame of the review queue and the index files that still exists.
    pub frames: u64,
    pub reviewed: u64,
    pub skipped: u64,
    /// Frames not reviewed or skipped yet.
    pub to_review: u64,
    /// Those of them in `review-queue.jsonl`, what the Review tab shows by default.
    pub queued_to_review: u64,
    /// Pending boxes on the frames to review. Boxes only a sidecar proposes (a frame with no
    /// label file) are not counted: reading every sidecar is what this walk avoids.
    pub pending_boxes: u64,
    /// Frames to review that no auto-label run has looked at ([`AutoFilter::Never`]).
    pub never_auto_labelled: u64,
    /// Training boxes of reviewed frames, per class.
    pub coverage: BTreeMap<String, ClassCoverage>,
    /// Frames left out: unreadable label file.
    pub unreadable: u64,
}

impl ClassCoverage {
    fn merge(&mut self, other: Self) {
        let Self {
            boxes,
            by_scene,
            by_camera,
        } = other;
        self.boxes += boxes;
        for (scene, n) in by_scene {
            *self.by_scene.entry(scene).or_default() += n;
        }
        for (camera, n) in by_camera {
            *self.by_camera.entry(camera).or_default() += n;
        }
    }
}

impl Tally {
    /// Add the counts of another part of the walk.
    fn merge(&mut self, other: Self) {
        let Self {
            frames,
            reviewed,
            skipped,
            to_review,
            queued_to_review,
            pending_boxes,
            never_auto_labelled,
            coverage,
            unreadable,
        } = other;
        self.frames += frames;
        self.reviewed += reviewed;
        self.skipped += skipped;
        self.to_review += to_review;
        self.queued_to_review += queued_to_review;
        self.pending_boxes += pending_boxes;
        self.never_auto_labelled += never_auto_labelled;
        for (class, row) in coverage {
            self.coverage.entry(class).or_default().merge(row);
        }
        self.unreadable += unreadable;
    }

    fn add(&mut self, c: &Candidate, labels: Option<&LabelFile>) {
        self.frames += 1;
        match labels.map(|l| l.status) {
            Some(Status::Reviewed) => self.reviewed += 1,
            Some(Status::Skipped) => self.skipped += 1,
            _ => {
                self.to_review += 1;
                self.queued_to_review += u64::from(c.queued);
                self.never_auto_labelled += u64::from(AutoFilter::Never.matches(labels));
                if let Some(l) = labels {
                    self.pending_boxes += l
                        .boxes
                        .iter()
                        .filter(|b| b.state == BoxState::Pending)
                        .count() as u64;
                }
            }
        }
        if let Some(l) = labels.filter(|l| l.status == Status::Reviewed) {
            for b in l.training() {
                let row = self.coverage.entry(b.label.clone()).or_default();
                row.boxes += 1;
                *row.by_scene.entry(c.scene.clone()).or_default() += 1;
                *row.by_camera.entry(c.camera.clone()).or_default() += 1;
            }
        }
    }
}

/// The [`Tally`] of `dir`. Reads the label files, never the sidecars, so it costs about one small
/// file per frame; a caller on a request path keeps the answer ([`crate::server`]'s Train area
/// does).
pub fn tally(dir: &Path) -> Tally {
    tally_on(dir, None)
}

/// [`tally`] on `threads` threads (None: as many as a walk of that size takes).
fn tally_on(dir: &Path, threads: Option<usize>) -> Tally {
    let all = candidates(dir, QueueSource::All);
    let threads = threads.unwrap_or_else(|| parallel::walk_threads_over(all.len()));
    let days = by_day(all);
    let walker = Walker::new(dir);
    let parts = parallel::map_all(days.len(), threads, &|i| {
        let mut walker = walker.clone();
        let mut t = Tally::default();
        for c in days.get(i).map_or(&[][..], Vec::as_slice) {
            let Some(frame) = Frame::resolve_in(&mut walker, &c.rel) else {
                continue;
            };
            match frame.labels() {
                Ok(labels) => t.add(c, labels.as_ref()),
                Err(_) => t.unreadable += 1,
            }
        }
        t
    });
    let mut t = Tally::default();
    for part in parts {
        t.merge(part);
    }
    t
}

/// A label file with its frame, as [`super::label_files`] lists it.
type Listed = (String, PathBuf);

/// `each` on the runs of one day folder in `files` (a list in [`super::label_files`]'s order), on
/// the threads a walk of that size takes; the results in `files`' order. For read walks only: label
/// writes stay on one thread.
pub(super) fn over_days<T: Send>(
    files: &[Listed],
    each: &(dyn Fn(&[Listed]) -> Vec<T> + Sync),
) -> Vec<T> {
    over_days_on(files, parallel::walk_threads_over(files.len()), each)
}

fn over_days_on<T: Send>(
    files: &[Listed],
    threads: usize,
    each: &(dyn Fn(&[Listed]) -> Vec<T> + Sync),
) -> Vec<T> {
    let days: Vec<&[Listed]> = files
        .chunk_by(|a, b| day_of(&a.0) == day_of(&b.0))
        .collect();
    let parts = parallel::map_all(days.len(), threads, &|i| {
        days.get(i).map_or_else(Vec::new, |day| each(day))
    });
    parts.into_iter().flatten().collect()
}

/// Whether [`labelled`] goes on to read the file of `rel`: what its name alone says.
fn name_passes(rel: &str, f: &Filters, keep: &dyn Fn(&str) -> bool) -> bool {
    layout::parse_rel(rel).is_some_and(|r| {
        keep(rel)
            && !f.scene.as_ref().is_some_and(|s| *s != r.scene)
            && !f.camera.as_ref().is_some_and(|c| *c != r.camera)
    })
}

/// The label file at `path` of frame `rel`, when the frame's files and the bucket and class filters
/// of `f` let it through.
fn read_passing(walker: &mut Walker<'_>, f: &Filters, rel: &str, path: &Path) -> Option<LabelFile> {
    let frame = Frame::resolve_in(walker, rel)?;
    // The buckets a review line or an index line carries: the sidecar's.
    if let Some(b) = f.bucket
        && !frame.record().ok()?.buckets.contains(&b)
    {
        return None;
    }
    let l = super::read(path).ok().flatten()?;
    if !f.label.is_empty() {
        let classes = file_labels(&l);
        if !f.label.iter().any(|c| classes.contains(c)) {
            return None;
        }
    }
    Some(l)
}

/// The label files of `dir` ([`super::label_files`], in its order) that the bucket, scene, camera
/// and class filters of `f` and `keep` select, read one at a time, so a caller may write between
/// them. For edits of already labelled frames, reviewed ones too: [`build`] only reaches what
/// `f.source` draws from (by default `review-queue.jsonl`, which a frame reviewed under "all
/// collected" is not in), so `source` is not used here, nor are `status` and `auto`. A frame whose
/// image, sidecar or label file does not read is left out.
pub fn labelled<'a>(
    dir: &'a Path,
    f: &'a Filters,
    keep: &'a dyn Fn(&str) -> bool,
) -> impl Iterator<Item = (String, LabelFile)> + 'a {
    let mut walker = Walker::new(dir);
    super::label_files(dir)
        .into_iter()
        .filter_map(move |(rel, path)| {
            if !name_passes(&rel, f, keep) {
                return None;
            }
            let l = read_passing(&mut walker, f, &rel, &path)?;
            Some((rel, l))
        })
}

/// What `each` makes of the files [`labelled`] selects, in the same order, on the threads a walk
/// of that size takes: for a walk that only reads and acts afterwards. `keep` runs on this thread.
pub fn labelled_map<T: Send>(
    dir: &Path,
    f: &Filters,
    keep: &dyn Fn(&str) -> bool,
    each: &(dyn Fn(&str, LabelFile) -> Option<T> + Sync),
) -> Vec<T> {
    let mut files = super::label_files(dir);
    files.retain(|(rel, _)| name_passes(rel, f, keep));
    let walker = Walker::new(dir);
    over_days(&files, &|day| {
        let mut walker = walker.clone();
        day.iter()
            .filter_map(|(rel, path)| each(rel, read_passing(&mut walker, f, rel, path)?))
            .collect()
    })
}

/// The item after `after` in `items` with its index: the first one when `after` is None or
/// not listed; when `after` is the last one, the first again with `wrapped` = true (even
/// `after` itself in a one-item list). None only for an empty list.
pub fn next_after<'a>(items: &'a [Item], after: Option<&str>) -> Option<(usize, &'a Item, bool)> {
    let pos = after.and_then(|a| items.iter().position(|i| i.frame == a));
    match pos {
        Some(p) if p + 1 < items.len() => items.get(p + 1).map(|i| (p + 1, i, false)),
        Some(_) => items.first().map(|i| (0, i, true)),
        None => items.first().map(|i| (0, i, false)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::super::record::{Det, VerifyCandidate, VerifyInfo, empty_record};
    use super::*;

    fn item(frame: &str, ts: &str, u: Uncertainty) -> Item {
        Item {
            frame: frame.into(),
            timestamp: ts.into(),
            scene: "ir".into(),
            camera: "cam".into(),
            model: "m".into(),
            buckets: vec![],
            status: Status::Unreviewed,
            queued: true,
            uncertainty: u,
        }
    }

    fn u(margin: Option<f32>, disagreement: bool, suppressed: usize) -> Uncertainty {
        Uncertainty {
            margin,
            disagreement,
            suppressed,
            relabel: None,
            auto_review: 0,
            likely_boxes: 0,
            repeats_only: false,
            overlap_strength: 0,
            agreed_boxes: 0,
        }
    }

    #[test]
    fn relabel_disagreements_come_first_then_fewer_flagged_boxes() {
        let relabel = Uncertainty {
            relabel: Some("model m: 1 new box".into()),
            ..u(None, false, 0)
        };
        let auto = Uncertainty {
            auto_review: 2,
            ..u(Some(0.3), false, 0)
        };
        let mut items = vec![
            item("plain", "9", u(Some(0.3), false, 0)),
            item("closest", "0", u(Some(0.0), false, 0)),
            item("auto", "0", auto),
            item("relabel", "0", relabel),
        ];
        sort_items(&mut items);
        let order: Vec<&str> = items.iter().map(|i| i.frame.as_str()).collect();
        assert_eq!(order, ["relabel", "closest", "plain", "auto"]);
        // A repeat-only frame goes last, even with a re-label disagreement.
        let repeats = Uncertainty {
            repeats_only: true,
            relabel: Some("x".into()),
            ..u(Some(0.0), true, 9)
        };
        items.push(item("repeats", "9", repeats));
        sort_items(&mut items);
        assert_eq!(items.last().unwrap().frame, "repeats");
        // From a label file.
        let mut f = LabelFile {
            disagreement: Some("x".into()),
            ..LabelFile::default()
        };
        let mut b = super::super::tests_support::auto_box("dog");
        b.auto.triage = Some(super::super::Triage::Review);
        f.boxes.push(b.clone());
        b.state = super::super::BoxState::Accepted;
        f.boxes.push(b);
        let x = u(None, false, 0).with_labels(Some(&f));
        assert_eq!((x.relabel.as_deref(), x.auto_review), (Some("x"), 1));
    }

    #[test]
    fn queue_ordering() {
        let mut items = vec![
            item("none", "9", u(None, true, 9)),
            item("far", "1", u(Some(0.4), false, 0)),
            item("near-old", "1", u(Some(0.021), false, 0)),
            item("near-new", "2", u(Some(0.019), false, 0)),
            item("near-dis", "0", u(Some(0.02), true, 0)),
            item("near-dis-sup", "0", u(Some(0.02), true, 3)),
            item("closest", "0", u(Some(0.0), false, 0)),
        ];
        sort_items(&mut items);
        let order: Vec<&str> = items.iter().map(|i| i.frame.as_str()).collect();
        assert_eq!(
            order,
            [
                "closest",
                "near-dis-sup",
                "near-dis",
                "near-new",
                "near-old",
                "far",
                "none"
            ]
        );
        assert_eq!(next_after(&items, None).unwrap().1.frame, "closest");
        assert_eq!(next_after(&items, Some("far")).unwrap().0, 6);
        let (i, it, wrapped) = next_after(&items, Some("none")).unwrap();
        assert_eq!(
            (i, it.frame.as_str(), wrapped),
            (0, "closest", true),
            "wraps"
        );
        assert_eq!(
            next_after(&items, Some("gone")).unwrap(),
            (0, &items[0], false)
        );
        let one = &items[..1];
        assert_eq!(
            next_after(one, Some("closest")).unwrap(),
            (0, &one[0], true)
        );
        assert!(next_after(&[], None).is_none());
        assert!(next_after(&[], Some("x")).is_none());
    }

    #[test]
    fn uncertainty_from_the_sidecar() {
        let mut r = empty_record();
        r.threshold = 0.5;
        let d = |l: &str, s: f32| Det {
            label: l.into(),
            score: s,
            bbox: [0.0, 0.0, 1.0, 1.0],
        };
        assert_eq!(Uncertainty::of(&r), u(None, false, 0));
        r.raw = vec![d("person", 0.9), d("person", 0.52), d("dog", 0.7)];
        r.below_threshold = vec![d("cat", 0.44)];
        let x = Uncertainty::of(&r);
        assert!((x.margin.unwrap() - 0.06).abs() < 1e-6, "{x:?}");
        r.verify = Some(VerifyInfo {
            candidates: vec![VerifyCandidate {
                label: "dog".into(),
                score: 0.7,
                bbox: [0.0; 4],
                outcome: "rejected".into(),
            }],
            ..VerifyInfo::default()
        });
        assert!(Uncertainty::of(&r).disagreement);
    }

    #[test]
    fn filters_from_the_query() {
        let q = |pairs: &[(&str, &str)]| {
            Filters::from_query(
                &pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            )
        };
        assert_eq!(q(&[]).unwrap(), Filters::default());
        let f = q(&[
            ("bucket", "near_threshold"),
            ("scene", "ir"),
            ("status", "any"),
            ("source", "all"),
            ("camera", "drive"),
        ])
        .unwrap();
        assert_eq!(f.bucket, Some(Bucket::NearThreshold));
        assert_eq!(f.status, StatusFilter::Any);
        assert_eq!(f.source, QueueSource::All);
        assert!(q(&[("bucket", "nope")]).unwrap_err().contains("bucket"));
        assert!(q(&[("scene", "dusk")]).unwrap_err().contains("scene"));
        assert!(q(&[("status", "done")]).unwrap_err().contains("status"));
        assert!(q(&[("source", "x")]).unwrap_err().contains("source"));
        assert_eq!(
            q(&[("auto", "none")]).unwrap().auto,
            Some(AutoFilter::Never)
        );
        assert_eq!(
            q(&[("auto", "accept")]).unwrap().auto,
            Some(AutoFilter::Accept)
        );
        assert_eq!(q(&[("auto", "")]).unwrap().auto, None);
        assert_eq!(
            q(&[("auto", "overlaps")]).unwrap().auto,
            Some(AutoFilter::Overlaps)
        );
        assert_eq!(q(&[("auto", "both")]).unwrap().auto, Some(AutoFilter::Both));
        assert!(q(&[("auto", "maybe")]).unwrap_err().contains("auto"));
        assert_eq!(q(&[("bucket", "")]).unwrap().bucket, None);
        let set = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
        assert_eq!(
            q(&[("label", " fire hydrant ")]).unwrap().label,
            set(&["fire hydrant"])
        );
        assert_eq!(
            q(&[("label", "dog, ,cat,dog,")]).unwrap().label,
            set(&["cat", "dog"]),
            "a comma list, empty items ignored"
        );
        assert!(q(&[("label", "")]).unwrap().label.is_empty());
        assert!(q(&[("label", " , ")]).unwrap().label.is_empty());
        assert!(q(&[("label", "dog,a\u{7}b")]).is_err());
        let e = q(&[("label", "a\u{7}b")]).unwrap_err();
        assert!(
            e.contains("label") && e.contains("not a valid class name"),
            "{e}"
        );
        assert!(q(&[("label", &"x".repeat(65))]).is_err());
    }

    fn det(l: &str) -> Det {
        Det {
            label: l.into(),
            score: 0.9,
            bbox: [0.0, 0.0, 10.0, 10.0],
        }
    }

    #[test]
    fn frame_labels_use_the_label_file_else_the_predictions() {
        let mut r = empty_record();
        r.predictions = vec![det("person"), det("person"), det("car")];
        r.raw = vec![det("dog")];
        let set = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
        assert_eq!(frame_labels(&r, None), set(&["car", "person"]));
        let mut f = LabelFile::default();
        assert!(frame_labels(&r, Some(&f)).is_empty(), "the label file wins");
        let mut b = super::super::tests_support::auto_box("dog");
        f.boxes.push(b.clone());
        b.label = "cat".into();
        b.state = super::super::BoxState::Rejected;
        f.boxes.push(b.clone());
        b.label = "deer".into();
        b.state = super::super::BoxState::Added;
        f.boxes.push(b.clone());
        b.label = "car".into();
        b.state = super::super::BoxState::Accepted;
        f.boxes.push(b);
        assert_eq!(frame_labels(&r, Some(&f)), set(&["car", "deer", "dog"]));
    }

    #[test]
    fn the_class_filter_keeps_frames_with_that_class_and_lists_every_class() {
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        let day = dir.join("night/cam/2026-10-22");
        std::fs::create_dir_all(&day).unwrap();
        let put = |stem: &str, labels: &[&str]| {
            let mut r = empty_record();
            r.id = stem.into();
            r.scene = "night".into();
            r.camera = "cam".into();
            r.buckets = vec![Bucket::Positive];
            r.timestamp = format!("2026-10-22T00:00:0{}Z", stem.len());
            r.predictions = labels.iter().map(|l| det(l)).collect();
            std::fs::write(day.join(format!("{stem}.jpg")), b"jpg").unwrap();
            std::fs::write(
                day.join(format!("{stem}.json")),
                serde_json::to_vec(&r).unwrap(),
            )
            .unwrap();
            let line = super::super::super::record::IndexLine::of(&r, 1);
            layout::append_line(
                &day.join(layout::INDEX_FILE),
                &serde_json::to_string(&line).unwrap(),
            )
            .unwrap();
        };
        put("a", &["person"]);
        put("bb", &["dog", "person"]);
        put("ccc", &[]);
        // A reviewed frame whose label file rejects the model's dog and adds a cat.
        put("dddd", &["dog"]);
        let mut f = LabelFile::default();
        let mut b = super::super::tests_support::auto_box("dog");
        b.state = super::super::BoxState::Rejected;
        f.boxes.push(b.clone());
        b.label = "cat".into();
        b.state = super::super::BoxState::Added;
        f.boxes.push(b);
        std::fs::write(
            day.join("dddd.labels.json"),
            serde_json::to_vec(&f).unwrap(),
        )
        .unwrap();
        let filters = |label: Option<&str>| Filters {
            source: QueueSource::All,
            status: StatusFilter::Any,
            label: label
                .map(|l| l.split(',').map(str::to_string).collect())
                .unwrap_or_default(),
            ..Filters::default()
        };
        let frames = |q: &Queue| {
            let mut v: Vec<String> = q
                .items
                .iter()
                .map(|i| i.frame.rsplit('/').next().unwrap().to_string())
                .collect();
            v.sort();
            v
        };
        let all = build(&dir, &filters(None));
        assert_eq!(all.items.len(), 4);
        assert_eq!(all.labels, ["cat", "dog", "person"]);
        let dog = build(&dir, &filters(Some("dog")));
        assert_eq!(frames(&dog), ["bb.jpg"], "the rejected dog does not count");
        assert_eq!(dog.counts.total, 1, "the counts follow the class filter");
        assert_eq!(dog.labels, all.labels, "the list ignores the class filter");
        assert_eq!(frames(&build(&dir, &filters(Some("cat")))), ["dddd.jpg"]);
        assert_eq!(
            frames(&build(&dir, &filters(Some("person")))),
            ["a.jpg", "bb.jpg"]
        );
        assert!(build(&dir, &filters(Some("deer"))).items.is_empty());
        assert_eq!(
            frames(&build(&dir, &filters(Some("cat,dog")))),
            ["bb.jpg", "dddd.jpg"],
            "any of the classes"
        );
        assert_eq!(build(&dir, &filters(Some("cat,dog"))).counts.total, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_overlaps_queue_lists_reviewed_frames_strongest_suggestion_first() {
        use super::super::bulk::test_support::{auto_box, put_frame};
        use super::super::overlaps::{Suggestion, Verdict};
        use super::super::write;
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        let (big, small) = ([0.0, 0.0, 40.0, 40.0], [5.0, 5.0, 25.0, 25.0]);
        let car = |bbox: [f32; 4]| super::super::LabelBox {
            bbox,
            ..auto_box("car", 0.5, super::super::Triage::Review)
        };
        let drop_b = |kept: f32, other: f32, weak: bool, low: bool| Suggestion {
            a: big,
            b: small,
            verdict: Verdict::DropB,
            why: "x".into(),
            kept_score: (!low).then_some(kept),
            other_score: (!low).then_some(other),
            low_confidence: low,
            weak,
        };
        let put = |stem: &str, status: Status, s: Vec<Suggestion>| {
            let rel = put_frame(&dir, stem, None, vec![car(big), car(small)]);
            let frame = Frame::resolve(&dir, &rel).unwrap();
            let mut l = frame.labels().unwrap().unwrap();
            l.status = status;
            l.overlap_suggestions = s;
            write(&frame.labels, &l).unwrap();
            rel
        };
        let none = put("none", Status::Unreviewed, vec![]);
        let witness = put(
            "witness",
            Status::Reviewed,
            vec![drop_b(0.0, 0.0, false, true)],
        );
        let strong = put(
            "strong",
            Status::Reviewed,
            vec![drop_b(0.9, 0.0, false, false)],
        );
        let two = put(
            "two",
            Status::Unreviewed,
            vec![Suggestion {
                verdict: Verdict::TwoObjects,
                ..drop_b(0.9, 0.8, false, false)
            }],
        );
        let weak = put(
            "weak",
            Status::Unreviewed,
            vec![drop_b(0.9, 0.4, true, false)],
        );
        // A frame whose suggestion went stale is one without.
        let stale = put(
            "stale",
            Status::Unreviewed,
            vec![Suggestion {
                b: [1.0, 1.0, 2.0, 2.0],
                ..drop_b(0.9, 0.0, false, false)
            }],
        );
        let classes: Vec<String> = ["car".to_string()].to_vec();
        let f = Filters {
            status: StatusFilter::Any,
            auto: Some(AutoFilter::Overlaps),
            wanted: Some((classes, Vec::new())),
            ..Filters::default()
        };
        let got: Vec<String> = build(&dir, &f).items.into_iter().map(|i| i.frame).collect();
        assert_eq!(got.len(), 6, "{got:?}");
        assert_eq!(got[..4], [strong, two, weak, witness]);
        let rest: BTreeSet<String> = got[4..].iter().cloned().collect();
        assert_eq!(rest, [none, stale].into_iter().collect());
        // The filter asks for it: another queue is not reordered, and shows no strength.
        let plain = Filters {
            status: StatusFilter::Any,
            ..Filters::default()
        };
        assert!(
            build(&dir, &plain)
                .items
                .iter()
                .all(|i| i.uncertainty.overlap_strength == 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_overlaps_filter_lists_reviewed_frames_until_a_reviewer_checked_them() {
        use super::super::bulk::test_support::put_frame;
        use super::super::{LabelBox, tests_support::auto_box, write};
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        let at = |label: &str, bbox: [f32; 4]| LabelBox {
            bbox,
            state: BoxState::Accepted,
            ..auto_box(label)
        };
        let reviewed = |rel: &str, checked: bool| {
            let frame = Frame::resolve(&dir, rel).unwrap();
            let mut l = frame.labels().unwrap().unwrap();
            l.status = Status::Reviewed;
            l.overlaps_checked = checked;
            write(&frame.labels, &l).unwrap();
        };
        let (a, near, small) = (
            [0.0, 0.0, 40.0, 40.0],
            [0.0, 0.0, 40.0, 38.0],
            [5.0, 5.0, 25.0, 25.0],
        );
        let stacked = put_frame(&dir, "s", None, vec![at("car", a), at("car", near)]);
        let nested = put_frame(&dir, "n", None, vec![at("car", a), at("truck", small)]);
        let kept = put_frame(&dir, "k", None, vec![at("car", a), at("car", small)]);
        let plain = put_frame(&dir, "p", None, vec![at("car", a)]);
        let hidden = put_frame(&dir, "h", None, vec![at("dog", a), at("dog", near)]);
        let open = put_frame(&dir, "o", None, vec![at("car", a), at("car", near)]);
        for rel in [&stacked, &nested, &plain, &hidden] {
            reviewed(rel, false);
        }
        reviewed(&kept, true);
        let frames = |f: &Filters| {
            let mut v: Vec<String> = build(&dir, f).items.into_iter().map(|i| i.frame).collect();
            v.sort();
            v
        };
        let mut f = Filters {
            status: StatusFilter::Is(Status::Reviewed),
            auto: Some(AutoFilter::Overlaps),
            ..Filters::default()
        };
        let mut want = vec![hidden.clone(), nested.clone(), stacked.clone()];
        want.sort();
        assert_eq!(frames(&f), want, "no class list: every label counts");
        // With the class list, a frame only an unwanted class overlaps on is not listed.
        f.wanted = Some((
            ["car", "truck", "dog"].map(String::from).to_vec(),
            vec!["dog".to_string()],
        ));
        want.retain(|r| *r != hidden);
        assert_eq!(frames(&f), want);
        f.status = StatusFilter::Is(Status::Unreviewed);
        assert_eq!(frames(&f), [open.as_str()], "any review status");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn labelled_reaches_frames_outside_the_queue_and_honours_the_filters() {
        use super::super::bulk::test_support::{put_frame, put_frame_at};
        use super::super::tests_support::auto_box;
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        let a = put_frame(&dir, "a", None, vec![auto_box("car")]);
        let b = put_frame_at(&dir, "drive", "b", (64, 48), None, vec![auto_box("dog")]);
        // Out of the review queue, reviewed: only the walk over the label files finds it.
        std::fs::remove_file(dir.join(REVIEW_QUEUE_FILE)).unwrap();
        let frame = Frame::resolve(&dir, &a).unwrap();
        let mut l = frame.labels().unwrap().unwrap();
        l.status = Status::Reviewed;
        super::super::write(&frame.labels, &l).unwrap();
        let rec = frame.sidecar.clone();
        let mut r = frame.record().unwrap();
        r.buckets = vec![Bucket::Positive];
        std::fs::write(&rec, serde_json::to_vec(&r).unwrap()).unwrap();
        assert!(build(&dir, &Filters::default()).items.is_empty());
        let all = |_: &str| true;
        let names = |f: &Filters, keep: &dyn Fn(&str) -> bool| {
            labelled(&dir, f, keep)
                .map(|(rel, _)| rel)
                .collect::<Vec<_>>()
        };
        let mut both = vec![a.clone(), b.clone()];
        both.sort();
        let mut got = names(&Filters::default(), &all);
        got.sort();
        assert_eq!(got, both, "a default filter reaches every label file");
        let camera = Filters {
            camera: Some("drive".into()),
            ..Filters::default()
        };
        assert_eq!(names(&camera, &all), [b.as_str()]);
        let scene = Filters {
            scene: Some("day".into()),
            ..Filters::default()
        };
        assert!(names(&scene, &all).is_empty());
        let bucket = Filters {
            bucket: Some(Bucket::Positive),
            ..Filters::default()
        };
        assert_eq!(names(&bucket, &all), [a.as_str()]);
        let label = Filters {
            label: ["dog".to_string()].into(),
            ..Filters::default()
        };
        assert_eq!(names(&label, &all), [b.as_str()]);
        let only_a = |r: &str| r == a;
        assert_eq!(names(&Filters::default(), &only_a), [a.as_str()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pending auto box found by `labelers`.
    fn found_by(label: &str, labelers: &[&str]) -> super::super::LabelBox {
        let mut b = super::super::tests_support::auto_box(label);
        b.auto.labelers = labelers.iter().map(|k| ((*k).to_string(), 0.5)).collect();
        b
    }

    #[test]
    fn auto_both_needs_two_labelers_on_a_pending_box() {
        let file = |boxes| LabelFile {
            boxes,
            ..LabelFile::default()
        };
        let matches = |l: &LabelFile| AutoFilter::Both.matches(Some(l));
        assert!(!AutoFilter::Both.matches(None), "no label file");
        assert!(matches(&file(vec![found_by("car", &["sam3", "rfdetr"])])));
        assert!(matches(&file(vec![
            found_by("car", &["sam3"]),
            found_by("dog", &["sam3", "gdino", "custom"]),
        ])));
        assert!(!matches(&file(vec![found_by("car", &["sam3"])])), "one");
        assert!(!matches(&file(vec![found_by("car", &[])])), "none");
        // A decided box is settled, however many found it.
        for state in [BoxState::Accepted, BoxState::Added, BoxState::Rejected] {
            let mut b = found_by("car", &["sam3", "rfdetr"]);
            b.state = state;
            assert!(!matches(&file(vec![b])), "{state:?}");
        }
    }

    #[test]
    fn auto_both_leaves_out_unwanted_classes_and_sorts_most_agreed_first() {
        use super::super::bulk::test_support::put_frame;
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        let car = || found_by("car", &["sam3", "rfdetr"]);
        let dog = || found_by("dog", &["sam3", "rfdetr"]);
        // Ties fall to the path, so these names alone would sort the fewest first.
        let one = put_frame(&dir, "aone", None, vec![car(), found_by("car", &["sam3"])]);
        let pair = put_frame(&dir, "bpair", None, vec![car(), car()]);
        let three = put_frame(&dir, "cthree", None, vec![car(), car(), car()]);
        let hidden = put_frame(&dir, "dhidden", None, vec![dog()]);
        let other = put_frame(
            &dir,
            "eother",
            None,
            vec![found_by("pottery", &["sam3", "rfdetr"])],
        );
        let lone = put_frame(&dir, "flone", None, vec![found_by("car", &["sam3"])]);
        // Unwanted boxes beside a wanted one: the frame stays, and only the wanted box counts.
        let mixed = put_frame(&dir, "gmixed", None, vec![car(), dog(), dog(), dog()]);
        let mut f = Filters {
            auto: Some(AutoFilter::Both),
            ..Filters::default()
        };
        let frames = |f: &Filters| -> Vec<String> {
            build(&dir, f).items.into_iter().map(|i| i.frame).collect()
        };
        // No class list: every label counts, as `matches` does.
        assert_eq!(
            frames(&f),
            [
                mixed.clone(),
                three.clone(),
                pair.clone(),
                one.clone(),
                hidden.clone(),
                other.clone()
            ]
        );
        f.wanted = Some((
            ["car", "dog"].map(String::from).to_vec(),
            vec!["dog".to_string()],
        ));
        assert_eq!(frames(&f), [three, pair, one, mixed], "{lone}");
        let items = build(&dir, &f).items;
        let counts: Vec<u32> = items.iter().map(|i| i.uncertainty.agreed_boxes).collect();
        assert_eq!(counts, [3, 2, 1, 1]);
        // The filter asks for it: another queue is not reordered, and shows no count.
        let plain = Filters {
            status: StatusFilter::Any,
            ..Filters::default()
        };
        assert!(
            build(&dir, &plain)
                .items
                .iter()
                .all(|i| i.uncertainty.agreed_boxes == 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejected_duplicates_are_not_counted_as_hard_negatives() {
        use super::super::{RejectReason, tests_support::auto_box};
        let mut dup = auto_box("car");
        dup.state = BoxState::Rejected;
        dup.reject_reason = Some(RejectReason::Duplicate);
        let mut wrong = dup.clone();
        wrong.reject_reason = None;
        let mut c = Counts::default();
        c.add(Some(&LabelFile {
            status: Status::Reviewed,
            boxes: vec![dup, wrong],
            ..LabelFile::default()
        }));
        assert_eq!(c.rejected_boxes, 1);
    }

    #[test]
    fn auto_filters_and_counts() {
        use super::super::{AutoRun, tests_support::auto_box};
        let mut accept = auto_box("car");
        accept.auto.triage = Some(Triage::Accept);
        let mut review = auto_box("dog");
        review.auto.triage = Some(Triage::Review);
        let file = |boxes: Vec<super::super::LabelBox>, ran: bool| LabelFile {
            boxes,
            auto_run: ran.then(|| AutoRun {
                at: "t".into(),
                labeler: "l".into(),
                classes: vec![],
            }),
            ..LabelFile::default()
        };
        let which = |l: Option<&LabelFile>| {
            [
                AutoFilter::Review,
                AutoFilter::Accept,
                AutoFilter::Nothing,
                AutoFilter::Never,
            ]
            .into_iter()
            .filter(|a| a.matches(l))
            .collect::<Vec<_>>()
        };
        assert_eq!(which(None), [AutoFilter::Never]);
        assert_eq!(which(Some(&file(vec![], false))), [AutoFilter::Never]);
        assert_eq!(which(Some(&file(vec![], true))), [AutoFilter::Nothing]);
        assert_eq!(
            which(Some(&file(vec![accept.clone()], true))),
            [AutoFilter::Accept]
        );
        assert_eq!(
            which(Some(&file(vec![accept.clone(), review.clone()], false))),
            [AutoFilter::Review]
        );
        // A plain model box carries no evidence: still "nothing" / "none".
        let mut model = auto_box("person");
        model.source = super::super::Source::Model;
        model.origin = Some(super::super::Origin::Prediction);
        assert_eq!(which(Some(&file(vec![model], true))), [AutoFilter::Nothing]);
        // A decided review box does not count as one to review.
        review.state = BoxState::Accepted;
        assert_eq!(
            which(Some(&file(vec![review], true))),
            Vec::<AutoFilter>::new()
        );
        let mut c = Counts::default();
        c.add(Some(&file(vec![accept], true)));
        c.add(None);
        assert_eq!((c.total, c.auto_accept, c.auto_review), (2, 1, 0));
    }

    #[test]
    fn frames_with_likely_boxes_open_the_queue_and_the_yard_of_false_boxes_comes_last() {
        let flagged = |n: usize, likely: u32, margin: Option<f32>| Uncertainty {
            auto_review: n,
            likely_boxes: likely,
            ..u(margin, false, 0)
        };
        let relabel = Uncertainty {
            relabel: Some("x".into()),
            ..u(None, false, 0)
        };
        let repeats = Uncertainty {
            repeats_only: true,
            likely_boxes: 9,
            ..u(None, false, 0)
        };
        let mut items = vec![
            item("yard", "9", flagged(17, 0, Some(0.0))),
            item("three", "1", flagged(3, 0, None)),
            item("one", "1", flagged(1, 0, Some(0.4))),
            item("likely-1", "1", flagged(9, 1, None)),
            item("likely-2", "1", flagged(2, 2, None)),
            item("likely-2-flagged", "1", flagged(5, 2, None)),
            item("relabel", "0", relabel),
            item("repeats", "9", repeats),
        ];
        sort_items(&mut items);
        let order: Vec<&str> = items.iter().map(|i| i.frame.as_str()).collect();
        assert_eq!(
            order,
            [
                "relabel",
                "likely-2",
                "likely-2-flagged",
                "likely-1",
                "one",
                "three",
                "yard",
                "repeats"
            ]
        );
    }

    #[test]
    fn likely_boxes_count_only_while_pending() {
        let mut b = super::super::tests_support::auto_box("dog");
        b.auto.likely = true;
        let mut f = LabelFile {
            boxes: vec![b.clone(), b.clone()],
            ..LabelFile::default()
        };
        f.boxes[1].state = BoxState::Accepted;
        assert_eq!(u(None, false, 0).with_labels(Some(&f)).likely_boxes, 1);
        // In the JSON only when set: older pages and tests see the items of today.
        let v = serde_json::to_value(u(None, false, 0)).unwrap();
        assert!(v.get("likely_boxes").is_none());
    }

    #[test]
    fn auto_dismissed_finds_a_frame_with_nothing_else_on_it() {
        assert_eq!(AutoFilter::parse("dismissed"), Some(AutoFilter::Dismissed));
        let q: BTreeMap<String, String> = [("auto".to_string(), "dismissed".to_string())].into();
        assert_eq!(
            Filters::from_query(&q).unwrap().auto,
            Some(AutoFilter::Dismissed)
        );
        let bad: BTreeMap<String, String> = [("auto".to_string(), "x".to_string())].into();
        assert!(Filters::from_query(&bad).unwrap_err().contains("dismissed"));
        let f = LabelFile {
            boxes: vec![super::super::LabelBox {
                state: BoxState::Rejected,
                reason: Some(super::super::bulk::DISMISSED_REASON.into()),
                ..super::super::tests_support::auto_box("car")
            }],
            ..LabelFile::default()
        };
        assert!(AutoFilter::Dismissed.matches(Some(&f)));
        assert!(!AutoFilter::Review.matches(Some(&f)), "nothing pending");
        assert!(
            !AutoFilter::Nothing.matches(Some(&f)),
            "the run found something"
        );
        assert!(!AutoFilter::Dismissed.matches(Some(&LabelFile::default())));
        assert!(!AutoFilter::Dismissed.matches(None));
        // A low-score reject is another thing.
        let mut low = f;
        low.boxes[0].reason = Some(super::super::bulk::LOW_SCORE_REASON.into());
        assert!(!AutoFilter::Dismissed.matches(Some(&low)));
    }

    #[test]
    fn the_counts_report_audit_verdicts_of_reviewed_frames_only() {
        let audit = |state: BoxState, reason: Option<&str>| super::super::LabelBox {
            state,
            reason: reason.map(str::to_string),
            auto: super::super::AutoInfo {
                audit: true,
                ..Default::default()
            },
            ..super::super::tests_support::auto_box("car")
        };
        let reviewed = LabelFile {
            status: Status::Reviewed,
            boxes: vec![
                audit(BoxState::Accepted, None),
                audit(BoxState::Rejected, None),
                audit(BoxState::Rejected, Some(super::super::REST_REASON)),
                audit(
                    BoxState::Rejected,
                    Some(super::super::bulk::LOW_SCORE_REASON),
                ),
            ],
            ..LabelFile::default()
        };
        let open = LabelFile {
            boxes: vec![audit(BoxState::Accepted, None)],
            ..LabelFile::default()
        };
        let mut c = Counts::default();
        c.add(Some(&reviewed));
        c.add(Some(&open));
        assert_eq!((c.audit_accepted, c.audit_rejected), (1, 2));
    }

    #[test]
    fn the_tally_counts_what_is_left_and_where_the_reviewed_boxes_are() {
        use super::super::{AutoRun, tests_support::auto_box};
        let cand = |scene: &str, camera: &str, queued: bool| Candidate {
            rel: String::new(),
            scene: scene.into(),
            camera: camera.into(),
            buckets: vec![],
            queued,
        };
        let mut good = auto_box("person");
        good.state = BoxState::Accepted;
        let reviewed = LabelFile {
            status: Status::Reviewed,
            boxes: vec![good.clone(), good, auto_box("car")],
            ..LabelFile::default()
        };
        let labelled = LabelFile {
            boxes: vec![auto_box("car"), auto_box("dog")],
            auto_run: Some(AutoRun {
                at: "t".into(),
                labeler: "l".into(),
                classes: vec![],
            }),
            ..LabelFile::default()
        };
        let skipped = LabelFile {
            status: Status::Skipped,
            ..LabelFile::default()
        };
        let mut t = Tally::default();
        t.add(&cand("day", "gate", true), Some(&reviewed));
        t.add(&cand("ir", "back", true), Some(&labelled));
        t.add(&cand("day", "back", false), None);
        t.add(&cand("day", "back", false), Some(&skipped));
        assert_eq!((t.frames, t.reviewed, t.skipped, t.to_review), (4, 1, 1, 2));
        assert_eq!(
            t.queued_to_review, 1,
            "the frame with no label file is not queued"
        );
        assert_eq!(t.pending_boxes, 2, "pending boxes of frames to review only");
        assert_eq!(t.never_auto_labelled, 1);
        let person = &t.coverage["person"];
        assert_eq!(person.boxes, 2);
        assert_eq!(person.by_scene["day"], 2);
        assert_eq!(person.by_camera["gate"], 2);
        assert!(!t.coverage.contains_key("car"), "a pending box is no label");
    }

    /// [`build`] as one loop over one frame at a time, [`Frame::resolve`] per frame: what the day
    /// by day walk must keep answering.
    fn serial_build(dir: &Path, f: &Filters) -> Queue {
        let mut q = Queue::default();
        let mut cameras = BTreeSet::new();
        let mut seen_labels = BTreeSet::new();
        for c in candidates(dir, f.source) {
            cameras.insert(c.camera.clone());
            if f.scene.as_ref().is_some_and(|s| *s != c.scene)
                || f.camera.as_ref().is_some_and(|s| *s != c.camera)
                || f.bucket.is_some_and(|b| !c.buckets.contains(&b))
            {
                continue;
            }
            let Some(frame) = Frame::resolve(dir, &c.rel) else {
                continue;
            };
            let (Ok(rec), Ok(labels)) = (frame.record(), frame.labels()) else {
                q.unreadable += 1;
                continue;
            };
            let classes = frame_labels(&rec, labels.as_ref());
            let keep = f.label.is_empty() || f.label.iter().any(|l| classes.contains(l));
            seen_labels.extend(classes);
            if !keep {
                continue;
            }
            q.counts.add(labels.as_ref());
            let status = labels.as_ref().map_or(Status::Unreviewed, |l| l.status);
            if let StatusFilter::Is(s) = f.status
                && s != status
            {
                continue;
            }
            if !f.auto_keeps(labels.as_ref()) {
                continue;
            }
            let uncertainty = Uncertainty {
                overlap_strength: f.overlap_strength(labels.as_ref()),
                agreed_boxes: f.agreed_boxes(labels.as_ref()),
                ..Uncertainty::of(&rec).with_labels(labels.as_ref())
            };
            q.items.push(Item {
                frame: c.rel,
                timestamp: rec.timestamp.clone(),
                scene: c.scene,
                camera: c.camera,
                model: rec.model.clone(),
                buckets: c.buckets,
                status,
                queued: c.queued,
                uncertainty,
            });
        }
        sort_items(&mut q.items);
        q.cameras = cameras.into_iter().collect();
        q.labels = seen_labels.into_iter().collect();
        q
    }

    /// [`tally`] likewise.
    fn serial_tally(dir: &Path) -> Tally {
        let mut t = Tally::default();
        for c in candidates(dir, QueueSource::All) {
            let Some(frame) = Frame::resolve(dir, &c.rel) else {
                continue;
            };
            match frame.labels() {
                Ok(labels) => t.add(&c, labels.as_ref()),
                Err(_) => t.unreadable += 1,
            }
        }
        t
    }

    /// A seeded folder of every kind of frame the walks meet: frames with and without a label file
    /// across scenes, cameras and days, gone frames and a gone day, an unreadable sidecar, an
    /// unreadable label file, review lines in a shuffled order, lines whose camera or scene differ
    /// from their path, a line naming a sidecar, indexed frames in and out of the queue.
    fn mixed_folder(dir: &Path) {
        use super::super::tests_support::auto_box;
        use super::super::write;
        use crate::collect::record::IndexLine;
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % n as u64).unwrap()
        };
        let classes = ["person", "car", "dog", "truck"];
        let mut queued: Vec<serde_json::Value> = Vec::new();
        let line = |rel: &str, scene: &str, camera: &str, buckets: Vec<Bucket>| {
            serde_json::json!({
                "version": 1, "frame": rel, "sidecar": rel.replace(".jpg", ".json"), "id": "x",
                "buckets": buckets, "scene": scene, "camera": camera,
                "queued_at": "2026-10-21T00:00:00Z",
            })
        };
        let mut index: BTreeMap<PathBuf, String> = BTreeMap::new();
        for i in 0..120 {
            let scene = ["day", "night", "ir"][next(3)];
            let camera = ["gate", "yard", "porch"][next(3)];
            let date = format!("2026-10-{:02}", 20 + next(4));
            let stem = format!("20261021T{i:06}000Z_f{i}");
            let day = dir.join(scene).join(camera).join(&date);
            std::fs::create_dir_all(&day).unwrap();
            let rel = format!("{scene}/{camera}/{date}/{stem}.jpg");
            let bucket = [Bucket::Positive, Bucket::Empty, Bucket::Suppressed][next(3)];
            let mut r = empty_record();
            r.id = stem.clone();
            r.scene = scene.into();
            r.camera = camera.into();
            // Ties on purpose: only the frame path orders these.
            r.timestamp = format!("2026-10-21T00:00:{:02}Z", next(5));
            r.threshold = 0.5;
            r.model = format!("m{}", next(2));
            r.buckets = vec![bucket];
            let d = Det {
                label: classes[next(4)].into(),
                score: 0.5 + next(5) as f32 / 10.0,
                bbox: [0.0, 0.0, 10.0, 10.0],
            };
            if next(5) > 0 {
                r.predictions = vec![d.clone()];
                r.raw = vec![d];
            }
            std::fs::write(day.join(format!("{stem}.jpg")), b"x").unwrap();
            std::fs::write(
                day.join(format!("{stem}.json")),
                serde_json::to_vec(&r).unwrap(),
            )
            .unwrap();
            let labels = day.join(format!("{stem}{}", super::super::LABELS_SUFFIX));
            match next(12) {
                // No label file (a third of the frames).
                0..=3 => {}
                4 => {
                    std::fs::write(&labels, b"{ not json").unwrap();
                }
                5 => {
                    std::fs::write(day.join(format!("{stem}.json")), b"{ not json").unwrap();
                }
                6 => {
                    std::fs::remove_file(day.join(format!("{stem}.jpg"))).unwrap();
                }
                7 => {
                    std::fs::remove_file(day.join(format!("{stem}.json"))).unwrap();
                }
                _ => {
                    let mut b = auto_box(classes[next(4)]);
                    b.auto.triage = Some(Triage::Review);
                    b.auto.likely = next(2) == 0;
                    let mut stacked = b.clone();
                    stacked.bbox[2] -= 1.0;
                    if next(3) > 0 {
                        b.state = BoxState::Accepted;
                        stacked.state = BoxState::Rejected;
                    }
                    let f = LabelFile {
                        status: [Status::Unreviewed, Status::Reviewed, Status::Skipped][next(3)],
                        boxes: vec![b, stacked],
                        ..LabelFile::default()
                    };
                    write(&labels, &f).unwrap();
                }
            }
            let il = IndexLine {
                id: stem.clone(),
                timestamp: r.timestamp.clone(),
                scene: scene.into(),
                camera: camera.into(),
                model: r.model.clone(),
                buckets: r.buckets.clone(),
                labels: vec![],
                bytes: 1,
            };
            let text = index.entry(day.join(layout::INDEX_FILE)).or_default();
            text.push_str(&serde_json::to_string(&il).unwrap());
            text.push('\n');
            if next(2) == 0 {
                // A review line may disagree with the path it names.
                let (s, c) = match i % 7 {
                    0 => ("day", "elsewhere"),
                    1 => (scene, "elsewhere"),
                    _ => (scene, camera),
                };
                queued.push(line(&rel, s, c, vec![bucket]));
            }
        }
        for (path, text) in index {
            std::fs::write(path, text).unwrap();
        }
        queued.push(line(
            "ir/gate/2026-01-01/20260101T000000000Z_gone.jpg",
            "ir",
            "lost-camera",
            vec![Bucket::Positive],
        ));
        queued.push(line(
            "ir/gate/2026-10-20/20261021T000001000Z_f1.json",
            "ir",
            "sidecar-line",
            vec![],
        ));
        for i in (1..queued.len()).rev() {
            queued.swap(i, next(i + 1));
        }
        let text: String = queued.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(dir.join(REVIEW_QUEUE_FILE), text).unwrap();
    }

    #[test]
    fn the_day_by_day_walks_answer_what_one_loop_answers() {
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        mixed_folder(&dir);
        let all = Filters {
            source: QueueSource::All,
            status: StatusFilter::Any,
            ..Filters::default()
        };
        let reference = serial_build(&dir, &all);
        assert!(reference.items.len() > 30, "{}", reference.items.len());
        assert!(reference.unreadable > 3, "{}", reference.unreadable);
        assert!(reference.cameras.iter().any(|c| c == "lost-camera"));
        assert!(reference.cameras.iter().any(|c| c == "elsewhere"));
        assert!(reference.counts.per_class.len() > 1);
        let classes: Vec<String> = ["person", "car"].map(String::from).to_vec();
        let filters = [
            Filters::default(),
            all.clone(),
            Filters {
                status: StatusFilter::Is(Status::Reviewed),
                ..all.clone()
            },
            Filters {
                scene: Some("day".into()),
                ..all.clone()
            },
            Filters {
                camera: Some("elsewhere".into()),
                ..all.clone()
            },
            Filters {
                bucket: Some(Bucket::Suppressed),
                ..all.clone()
            },
            Filters {
                label: BTreeSet::from(["person".to_string(), "dog".to_string()]),
                ..all.clone()
            },
            Filters {
                auto: Some(AutoFilter::Review),
                ..all.clone()
            },
            Filters {
                auto: Some(AutoFilter::Overlaps),
                wanted: Some((classes, Vec::new())),
                ..all.clone()
            },
        ];
        for f in &filters {
            let want = serial_build(&dir, f);
            let bytes = serde_json::to_vec(&want).unwrap();
            for threads in [1, 2, 4, 8] {
                let got = build_on(&dir, f, Some(threads));
                assert_eq!(got, want, "{threads} threads, {f:?}");
                assert_eq!(serde_json::to_vec(&got).unwrap(), bytes);
            }
            assert_eq!(build(&dir, f), want, "{f:?}");
        }
        let want = serial_tally(&dir);
        assert!(want.unreadable > 0 && want.frames > 30, "{want:?}");
        assert!(!want.coverage.is_empty());
        for threads in [1, 2, 4, 8] {
            assert_eq!(tally_on(&dir, Some(threads)), want, "{threads} threads");
        }
        assert_eq!(tally(&dir), want);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_labelled_map_lists_what_labelled_lists_in_its_order() {
        let dir = std::env::temp_dir().join(format!("bop-queue-{}", uuid::Uuid::new_v4()));
        mixed_folder(&dir);
        let drop_some = |rel: &str| !rel.contains("_f1");
        for f in [
            Filters::default(),
            Filters {
                scene: Some("night".into()),
                ..Filters::default()
            },
            Filters {
                bucket: Some(Bucket::Positive),
                ..Filters::default()
            },
            Filters {
                label: BTreeSet::from(["car".to_string()]),
                ..Filters::default()
            },
        ] {
            let want: Vec<(String, LabelFile)> = labelled(&dir, &f, &drop_some).collect();
            let got = labelled_map(&dir, &f, &drop_some, &|rel, l| Some((rel.to_string(), l)));
            assert_eq!(got, want, "{f:?}");
        }
        assert!(labelled(&dir, &Filters::default(), &drop_some).count() > 10);
        let files = super::super::label_files(&dir);
        let names = |day: &[(String, PathBuf)]| -> Vec<String> {
            day.iter().map(|(rel, _)| rel.clone()).collect()
        };
        let want: Vec<String> = files.iter().map(|(rel, _)| rel.clone()).collect();
        for threads in [1, 3, 8] {
            assert_eq!(over_days_on(&files, threads, &names), want);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
