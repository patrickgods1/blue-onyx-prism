//! The review queue: the frames of `review-queue.jsonl` (and, with [`QueueSource::All`], every
//! indexed frame), filtered by bucket, scene, camera, object class ([`frame_labels`]), review
//! status and auto-label state ([`AutoFilter`]), and sorted by
//! uncertainty ([`Uncertainty`]): the near-threshold margin first, then verifier disagreement,
//! then the number of suppressed boxes, then newest. Also the counts the Review page shows.

use super::super::layout::{self, REVIEW_QUEUE_FILE};
use super::super::record::{Bucket, Record};
use super::{BoxState, Frame, LabelFile, Status, Triage};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Which frames the queue draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueSource {
    /// `review-queue.jsonl` only (frames sent from the Dataset page).
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
}

impl AutoFilter {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "review" => Some(AutoFilter::Review),
            "accept" => Some(AutoFilter::Accept),
            "nothing" => Some(AutoFilter::Nothing),
            "none" => Some(AutoFilter::Never),
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
        }
    }
}

/// A pending box triaged `review`.
fn has_auto_review(l: &LabelFile) -> bool {
    l.boxes
        .iter()
        .any(|b| b.state == BoxState::Pending && b.auto.triage == Some(Triage::Review))
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
}

impl Filters {
    /// From query parameters (`bucket`, `scene`, `camera`, `label` = a comma list, any of
    /// them, empty items ignored, `status` = `unreviewed|reviewed|
    /// skipped|any`, `source` = `queue|all`, `auto` = `review|accept|nothing|none`); empty values
    /// mean "any". Errors name the field.
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
            Some(a) => Some(
                AutoFilter::parse(a)
                    .ok_or_else(|| format!("auto: '{a}' is not review, accept, nothing or none"))?,
            ),
        };
        Ok(Self {
            bucket,
            scene,
            camera: get("camera").map(str::to_string),
            label,
            status,
            source,
            auto,
        })
    }
}

/// The object classes of a frame: with a label file, the labels of its boxes that are not
/// rejected (accepted, added and still pending ones); without one, the labels of the reply's
/// `predictions`.
pub fn frame_labels(rec: &Record, labels: Option<&LabelFile>) -> BTreeSet<String> {
    match labels {
        Some(l) => l
            .boxes
            .iter()
            .filter(|b| b.state != super::BoxState::Rejected)
            .map(|b| b.label.clone())
            .collect(),
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
    /// Sent to review from the Dataset page.
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
        }
    }

    /// Add what the frame's label file says: its re-label disagreement and the pending
    /// auto-labeler boxes triaged `review`.
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
        }
        self
    }

    /// Sort key, most uncertain first: a re-label disagreement, then the margin in hundredths
    /// (no box last), verifier disagreement, more auto boxes to review, more suppressed boxes.
    #[allow(
        clippy::type_complexity,
        reason = "a lexicographic sort key reads best as one tuple"
    )]
    fn key(
        &self,
    ) -> (
        bool,
        u32,
        bool,
        std::cmp::Reverse<usize>,
        std::cmp::Reverse<usize>,
    ) {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a margin is a difference of scores in 0..=1, so 0..=100 hundredths"
        )]
        let m = self
            .margin
            .map_or(u32::MAX, |m| (m.clamp(0.0, 1.0) * 100.0).round() as u32);
        (
            self.relabel.is_none(),
            m,
            !self.disagreement,
            std::cmp::Reverse(self.auto_review),
            std::cmp::Reverse(self.suppressed),
        )
    }
}

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
    /// Rejected model boxes of reviewed frames (hard-negative evidence).
    pub rejected_boxes: u64,
    /// Frames with a pending box triaged `review` ([`AutoFilter::Review`]).
    pub auto_review: u64,
    /// Frames the auto-labeler's triage alone settles ([`AutoFilter::Accept`]).
    pub auto_accept: u64,
}

impl Counts {
    fn add(&mut self, labels: Option<&LabelFile>) {
        self.total += 1;
        match labels.map(|l| l.status) {
            Some(Status::Reviewed) => self.reviewed += 1,
            Some(Status::Skipped) => self.skipped += 1,
            _ => self.remaining += 1,
        }
        self.auto_review += u64::from(AutoFilter::Review.matches(labels));
        self.auto_accept += u64::from(AutoFilter::Accept.matches(labels));
        if let Some(l) = labels.filter(|l| l.status == Status::Reviewed) {
            self.empty_confirmed += u64::from(l.empty_confirmed);
            for b in l.training() {
                *self.per_class.entry(b.label.clone()).or_default() += 1;
            }
            self.rejected_boxes += l.rejected().count() as u64;
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

/// The candidates of `dir`: the review queue's frames first, then (with `All`) every indexed
/// frame, each once.
pub fn candidates(dir: &Path, source: QueueSource) -> Vec<Candidate> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for l in layout::read_review(&dir.join(REVIEW_QUEUE_FILE)) {
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
    if source == QueueSource::All {
        for day in layout::day_dirs(dir) {
            let date = day
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            for line in layout::read_index(&day.join(layout::INDEX_FILE)) {
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
    }
    out
}

/// Build the queue of `dir` for `f`. Frames whose image or sidecar is gone (retention) are
/// left out silently; a sidecar or label file that does not parse counts as `unreadable`.
pub fn build(dir: &Path, f: &Filters) -> Queue {
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
        if f.auto.is_some_and(|a| !a.matches(labels.as_ref())) {
            continue;
        }
        q.items.push(Item {
            frame: c.rel,
            timestamp: rec.timestamp.clone(),
            scene: c.scene,
            camera: c.camera,
            model: rec.model.clone(),
            buckets: c.buckets,
            status,
            queued: c.queued,
            uncertainty: Uncertainty::of(&rec).with_labels(labels.as_ref()),
        });
    }
    sort_items(&mut q.items);
    q.cameras = cameras.into_iter().collect();
    q.labels = seen_labels.into_iter().collect();
    q
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
        }
    }

    #[test]
    fn relabel_disagreements_come_first_then_auto_boxes_to_review() {
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
        assert_eq!(order, ["relabel", "closest", "auto", "plain"]);
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
}
