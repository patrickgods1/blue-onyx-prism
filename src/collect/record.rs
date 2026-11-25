//! The per-frame record (the sidecar JSON next to every saved frame, schema
//! [`RECORD_VERSION`], documented in `docs/DATASET.md`) and the pure bucket decision
//! [`decide`].

use crate::model::filters::SuppressReason;
use crate::verify::Outcome;
use serde::{Deserialize, Serialize};

/// Version of the sidecar schema ([`Record::version`]). Readers skip records of a newer major.
pub const RECORD_VERSION: u32 = 1;

/// Why a frame is worth keeping. A frame may match several; it is stored once and lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bucket {
    /// The reply has detections.
    Positive,
    /// The model found nothing at or above the threshold (sampled hard-negative candidates).
    Empty,
    /// A rule removed a box (night threshold, label threshold, aspect, min area, overlap,
    /// static): false-positive candidates, or misses when the rule was wrong.
    Suppressed,
    /// The verifier rejected a candidate.
    VerifyRejected,
    /// The verifier confirmed a candidate.
    VerifyKept,
    /// A label's top score is within the margin of the threshold (uncertain frames).
    NearThreshold,
    /// The verifier and the model disagree (see [`decide`]).
    Disagreement,
}

impl Bucket {
    pub const ALL: [Bucket; 7] = [
        Bucket::Positive,
        Bucket::Empty,
        Bucket::Suppressed,
        Bucket::VerifyRejected,
        Bucket::VerifyKept,
        Bucket::NearThreshold,
        Bucket::Disagreement,
    ];

    /// A frame kept for this bucket is saved even when it looks like a recent frame of its camera:
    /// these frames matter for a box or a score, and a distant person changes only a few of the 64
    /// dHash bits. `positive` and `empty` keep the dedup window. Limit:
    /// [`super::store::DEDUP_SKIPS_PER_CAMERA`].
    pub fn skips_dedup(self) -> bool {
        !matches!(self, Bucket::Positive | Bucket::Empty)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::Positive => "positive",
            Bucket::Empty => "empty",
            Bucket::Suppressed => "suppressed",
            Bucket::VerifyRejected => "verify_rejected",
            Bucket::VerifyKept => "verify_kept",
            Bucket::NearThreshold => "near_threshold",
            Bucket::Disagreement => "disagreement",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.as_str() == s)
    }

    /// Short human text for the Collect tab.
    pub fn describe(self) -> &'static str {
        match self {
            Bucket::Positive => "Detections in the reply",
            Bucket::Empty => "Nothing detected (sampled)",
            Bucket::Suppressed => "A rule removed a box",
            Bucket::VerifyRejected => "The verifier rejected a box",
            Bucket::VerifyKept => "The verifier confirmed a box",
            Bucket::NearThreshold => "A score close to the threshold",
            Bucket::Disagreement => "The verifier and the model disagree",
        }
    }
}

/// A box in original frame pixels (`x1, y1, x2, y2`, clamped to the frame, 0.1 px).
pub type BoxPx = [f32; 4];

/// One detection of the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Det {
    pub label: String,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: BoxPx,
}

/// A detection a rule removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suppressed {
    pub label: String,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: BoxPx,
    /// `night_threshold | label_threshold | aspect | min_area | overlap | static | verify`.
    pub reason: String,
}

/// One verifier candidate and what happened to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyCandidate {
    pub label: String,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: BoxPx,
    /// `kept | rejected | unverified` ([`Outcome::as_str`]; a string, so an unknown outcome a newer
    /// build wrote still reads).
    pub outcome: String,
}

impl VerifyCandidate {
    /// Whether the stored outcome is `o`.
    pub fn is(&self, o: Outcome) -> bool {
        self.outcome == o.as_str()
    }
}

/// The verifier cascade's part of a record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VerifyInfo {
    pub verifier: String,
    /// `crop | frame`.
    pub mode: String,
    /// Why candidates stayed unverified (`timeout`, `busy`, ...), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    pub candidates: Vec<VerifyCandidate>,
    /// Mode `frame`: every box the verifier reported on the whole frame (crop mode: empty,
    /// its boxes are in crop coordinates).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub found: Vec<Det>,
    /// The verifier's `min_confidence` and `iou` (what `found` is compared with).
    pub min_confidence: f32,
    pub iou: f32,
}

/// Scene statistics of the frame ([`crate::image_stats::ImageStats`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Stats {
    pub mean_luma: f32,
    pub chroma_spread: f32,
    pub tint_spread: f32,
    pub is_ir: bool,
    pub is_dark: bool,
}

impl From<crate::image_stats::ImageStats> for Stats {
    fn from(s: crate::image_stats::ImageStats) -> Self {
        Self {
            mean_luma: round2(s.mean_luma),
            chroma_spread: round2(s.chroma_spread),
            tint_spread: round2(s.tint_spread),
            is_ir: s.is_ir,
            is_dark: s.is_dark,
        }
    }
}

/// The saved image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageInfo {
    /// File name next to the sidecar.
    pub file: String,
    pub bytes: u64,
    /// The request's own bytes (false: the upload was not a JPEG and was re-encoded).
    pub original: bool,
}

/// The sidecar JSON of one saved frame (`docs/DATASET.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub version: u32,
    /// `<ts>_<request_id>`, the file stem of the image and the sidecar.
    pub id: String,
    pub request_id: String,
    /// UTC, RFC 3339 with milliseconds.
    pub timestamp: String,
    pub model: String,
    pub device: String,
    /// The model that forwarded the frame to this (night) model; None = asked directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_from: Option<String>,
    pub image_name: String,
    /// Folder name of the camera ([`super::camera_dir`]).
    pub camera: String,
    /// [`crate::static_memory::camera_key`] of the request.
    pub camera_key: String,
    /// `day | night | ir`.
    pub scene: String,
    pub stats: Stats,
    pub width: u32,
    pub height: u32,
    pub image: ImageInfo,
    /// The request's effective confidence threshold.
    pub threshold: f32,
    /// The `near_threshold` band used (None: not computed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near_margin: Option<f32>,
    /// The model's detections at or above `threshold`, after NMS, before any rule.
    pub raw: Vec<Det>,
    /// Detections in `[threshold - margin, threshold)` (only with `near_threshold` on).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub below_threshold: Vec<Det>,
    /// Every box a rule or the verifier removed, with the reason.
    pub suppressed: Vec<Suppressed>,
    /// The reply's predictions.
    pub predictions: Vec<Det>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<VerifyInfo>,
    /// The buckets the frame was kept for (it passed their sampling and rate limits).
    pub buckets: Vec<Bucket>,
    /// Every enabled bucket the frame matched ([`decide`]), a superset of `buckets`.
    #[serde(default)]
    pub matched: Vec<Bucket>,
    /// 64-bit dHash of the frame (16 hex digits).
    pub dhash: String,
    /// What the collector's dedup saw when it kept the frame ([`DedupInfo`]); absent on imported
    /// frames and on frames saved before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedup: Option<DedupInfo>,
    /// Where the frame came from when the service did not receive it from Blue Iris:
    /// `import:<id>` for imported footage ([`crate::training::import`]). None = collected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Provenance of an imported frame (file or URL, licence and its confirmation, time offset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// The dedup's view of a kept frame (`docs/DATASET.md`, "Sidecar JSON"): small, additive, for
/// tuning the rule on live data later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DedupInfo {
    /// `collect.dedup_mode` at the time: `both` or `dhash`.
    pub mode: String,
    /// Smallest dHash distance over every reference of the camera in the window (any scene);
    /// absent without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nearest_hamming: Option<u32>,
    /// One block-test verdict: its largest |z| (0.1) and the cells of its largest group. On a near
    /// duplicate saved for a skip bucket, that of the reference that confirmed it; otherwise the
    /// closest call (smallest changed group) among the dHash look-alikes it checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_max_z: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_largest_blob: Option<u32>,
    /// Why the frame was kept: [`DedupInfo::NO_REFERENCE`], [`DedupInfo::DHASH`],
    /// [`DedupInfo::BLOCKDIFF`], [`DedupInfo::UNCHECKED`] or [`DedupInfo::SKIPS_DEDUP`].
    pub decided_by: String,
}

impl DedupInfo {
    /// No reference of the camera in the window.
    pub const NO_REFERENCE: &str = "no_reference";
    /// No reference within `dedup_hamming` dHash bits.
    pub const DHASH: &str = "dhash";
    /// dHash look-alikes, and the block test found a change against each it checked.
    pub const BLOCKDIFF: &str = "blockdiff";
    /// dHash look-alikes, none the block test could check (another scene, or its thumbnail evicted).
    pub const UNCHECKED: &str = "unchecked";
    /// A near duplicate saved because its bucket skips the dedup ([`Bucket::skips_dedup`]).
    pub const SKIPS_DEDUP: &str = "skips_dedup";
}

/// Provenance of an imported frame (`docs/DATASET.md`, "Imported frames").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// The import's id (`<UTC time>-<4 hex>`).
    pub import_id: String,
    /// `video | image | url | dataset`.
    pub kind: String,
    /// The local file the frame was read from (the video or the image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The page or file URL of a downloaded video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The pinned dataset resource (`import-data:meva-sample`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The licence the user stated (or the pinned dataset's).
    pub license: String,
    /// The user confirmed they may use the footage under that licence (always true for a URL or
    /// a file import, which is refused without the confirmation).
    pub license_confirmed: bool,
    /// When the licence was confirmed (UTC, RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed_at: Option<String>,
    /// What the site reported (yt-dlp: title, uploader, licence, page URL), for URL imports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<serde_json::Value>,
    /// Seconds from the start of the video (None for an image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_offset_s: Option<f64>,
    /// Attribution to show where the frame is used (CC BY).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
}

/// A record with nothing filled in (the worker fills it, the collector completes it).
pub(crate) fn empty_record() -> Record {
    Record {
        version: RECORD_VERSION,
        id: String::new(),
        request_id: String::new(),
        timestamp: String::new(),
        model: String::new(),
        device: String::new(),
        routed_from: None,
        image_name: String::new(),
        camera: String::new(),
        camera_key: String::new(),
        scene: "day".into(),
        stats: Stats::default(),
        width: 0,
        height: 0,
        image: ImageInfo::default(),
        threshold: 0.5,
        near_margin: None,
        raw: Vec::new(),
        below_threshold: Vec::new(),
        suppressed: Vec::new(),
        predictions: Vec::new(),
        verify: None,
        buckets: Vec::new(),
        matched: Vec::new(),
        dhash: String::new(),
        dedup: None,
        source: None,
        provenance: None,
    }
}

/// One line of a day folder's `index.jsonl`: what the Collect tab and the later stages list
/// without opening every sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexLine {
    pub id: String,
    pub timestamp: String,
    pub scene: String,
    pub camera: String,
    pub model: String,
    pub buckets: Vec<Bucket>,
    /// Labels of the reply's predictions.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Image + sidecar bytes.
    pub bytes: u64,
}

impl IndexLine {
    pub fn of(r: &Record, bytes: u64) -> Self {
        Self {
            id: r.id.clone(),
            timestamp: r.timestamp.clone(),
            scene: r.scene.clone(),
            camera: r.camera.clone(),
            model: r.model.clone(),
            buckets: r.buckets.clone(),
            labels: r.predictions.iter().map(|p| p.label.clone()).collect(),
            bytes,
        }
    }

    /// The frame's image path relative to the dataset root when this line is in day folder
    /// `date`: `<scene>/<camera>/<date>/<id>.jpg` (forward slashes). The one mapping from an index
    /// line to its frame: the queue, the frame references and the strays all use it.
    pub fn rel(&self, date: &str) -> String {
        format!("{}/{}/{date}/{}.jpg", self.scene, self.camera, self.id)
    }
}

pub(crate) fn round2(x: f32) -> f32 {
    (x * 100.0).round() / 100.0
}

/// Which buckets `r` belongs to (in [`Bucket::ALL`] order), before the enabled / rate / dedup
/// checks:
///
/// - `positive`: the reply has predictions;
/// - `empty`: `raw` is empty (the model found nothing at or above the threshold; boxes only
///   below it still count as empty, and can make the frame `near_threshold` too);
/// - `suppressed`: a rule other than the verifier removed a box;
/// - `verify_rejected` / `verify_kept`: a candidate was rejected / confirmed;
/// - `near_threshold`: for some label, the top score over `raw` (suppressed boxes included)
///   and `below_threshold` is within `margin` of `threshold` (`|top - threshold| <= margin`).
///   Per label, so an uncertain dog next to a sure person still counts;
/// - `disagreement`: the verifier rejected a candidate, or (mode `frame`) reported a box at or
///   above its `min_confidence` that matches no box of the model's `raw` detections (same
///   label, IoU >= its `iou`): a miss of the model. The night model switch alone is never a
///   disagreement: the primary does not run on a forwarded frame, so there is nothing to
///   compare without spending a second inference; a forwarded frame enters this bucket only
///   through its verifier.
pub fn decide(r: &Record, margin: f32) -> Vec<Bucket> {
    let mut out = Vec::new();
    if !r.predictions.is_empty() {
        out.push(Bucket::Positive);
    }
    if r.raw.is_empty() {
        out.push(Bucket::Empty);
    }
    if r.suppressed
        .iter()
        .any(|s| s.reason != SuppressReason::Verify.as_str())
    {
        out.push(Bucket::Suppressed);
    }
    let outcome = |o: Outcome| {
        r.verify
            .as_ref()
            .is_some_and(|v| v.candidates.iter().any(|c| c.is(o)))
    };
    let rejected = outcome(Outcome::Rejected);
    if rejected {
        out.push(Bucket::VerifyRejected);
    }
    if outcome(Outcome::Kept) {
        out.push(Bucket::VerifyKept);
    }
    if near_threshold(r, margin) {
        out.push(Bucket::NearThreshold);
    }
    if rejected || verifier_found_a_miss(r) {
        out.push(Bucket::Disagreement);
    }
    out
}

fn near_threshold(r: &Record, margin: f32) -> bool {
    let mut top: std::collections::BTreeMap<String, f32> = std::collections::BTreeMap::new();
    for d in r.raw.iter().chain(&r.below_threshold) {
        let e = top.entry(d.label.to_ascii_lowercase()).or_insert(d.score);
        if d.score > *e {
            *e = d.score;
        }
    }
    // A small tolerance: scores and thresholds are f32 and the band edges are inclusive.
    top.values()
        .any(|s| (s - r.threshold).abs() <= margin + 1e-6)
}

fn verifier_found_a_miss(r: &Record) -> bool {
    let Some(v) = &r.verify else { return false };
    v.found.iter().any(|f| {
        f.score >= v.min_confidence
            && !r.raw.iter().any(|d| {
                d.label.eq_ignore_ascii_case(&f.label)
                    && crate::verify::iou(d.bbox, f.bbox) >= v.iou
            })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn det(label: &str, score: f32, b: BoxPx) -> Det {
        Det {
            label: label.into(),
            score,
            bbox: b,
        }
    }

    pub(crate) fn record() -> Record {
        Record {
            camera: "cam".into(),
            near_margin: Some(0.1),
            ..empty_record()
        }
    }

    const B: BoxPx = [10.0, 10.0, 50.0, 90.0];

    #[test]
    fn empty_frame_is_only_empty() {
        assert_eq!(decide(&record(), 0.1), vec![Bucket::Empty]);
    }

    #[test]
    fn sure_positive() {
        let mut r = record();
        r.raw = vec![det("person", 0.9, B)];
        r.predictions = r.raw.clone();
        assert_eq!(decide(&r, 0.1), vec![Bucket::Positive]);
    }

    #[test]
    fn near_threshold_band_is_inclusive_and_per_label() {
        let mut r = record();
        r.raw = vec![det("person", 0.6, B)];
        r.predictions = r.raw.clone();
        assert_eq!(
            decide(&r, 0.1),
            vec![Bucket::Positive, Bucket::NearThreshold]
        );
        assert_eq!(decide(&r, 0.05), vec![Bucket::Positive]);
        // A sure person does not hide an uncertain dog.
        r.raw = vec![det("person", 0.95, B), det("dog", 0.52, B)];
        r.predictions = r.raw.clone();
        assert!(decide(&r, 0.1).contains(&Bucket::NearThreshold));
        // Per label the top score counts: a second, weaker person box does not.
        r.raw = vec![det("person", 0.95, B), det("person", 0.52, B)];
        assert!(!decide(&r, 0.1).contains(&Bucket::NearThreshold));
    }

    #[test]
    fn below_threshold_boxes_make_an_empty_frame_near_threshold() {
        let mut r = record();
        r.below_threshold = vec![det("person", 0.42, B)];
        assert_eq!(decide(&r, 0.1), vec![Bucket::Empty, Bucket::NearThreshold]);
        r.below_threshold = vec![det("person", 0.3, B)];
        assert_eq!(decide(&r, 0.1), vec![Bucket::Empty]);
    }

    #[test]
    fn suppressed_boxes_count_and_their_scores_too() {
        let mut r = record();
        r.raw = vec![det("person", 0.55, B)];
        r.suppressed = vec![Suppressed {
            label: "person".into(),
            score: 0.55,
            bbox: B,
            reason: "night_threshold".into(),
        }];
        assert_eq!(
            decide(&r, 0.1),
            vec![Bucket::Suppressed, Bucket::NearThreshold]
        );
        // A verifier rejection alone is not `suppressed`.
        r.suppressed[0].reason = "verify".into();
        r.raw = vec![det("person", 0.9, B)];
        assert_eq!(decide(&r, 0.1), Vec::<Bucket>::new());
    }

    fn verify(outcomes: &[&str], found: Vec<Det>) -> VerifyInfo {
        VerifyInfo {
            verifier: "v".into(),
            mode: "frame".into(),
            failure: None,
            candidates: outcomes
                .iter()
                .map(|o| VerifyCandidate {
                    label: "person".into(),
                    score: 0.7,
                    bbox: B,
                    outcome: (*o).into(),
                })
                .collect(),
            found,
            min_confidence: 0.3,
            iou: 0.5,
        }
    }

    #[test]
    fn verify_outcomes_and_disagreement() {
        let mut r = record();
        r.raw = vec![det("person", 0.9, B)];
        r.predictions = r.raw.clone();
        r.verify = Some(verify(&["kept"], vec![det("person", 0.8, B)]));
        assert_eq!(decide(&r, 0.1), vec![Bucket::Positive, Bucket::VerifyKept]);
        // Rejected: verify_rejected and disagreement (the reply lost the box).
        r.predictions.clear();
        r.verify = Some(verify(&["rejected"], Vec::new()));
        assert_eq!(
            decide(&r, 0.1),
            vec![Bucket::VerifyRejected, Bucket::Disagreement]
        );
        // Unverified only: nothing from the verifier.
        r.predictions = r.raw.clone();
        r.verify = Some(verify(&["unverified"], Vec::new()));
        assert_eq!(decide(&r, 0.1), vec![Bucket::Positive]);
    }

    #[test]
    fn verifier_box_the_model_missed_is_a_disagreement() {
        let mut r = record();
        r.raw = vec![det("person", 0.9, B)];
        r.predictions = r.raw.clone();
        let car = det("car", 0.6, [60.0, 60.0, 95.0, 95.0]);
        r.verify = Some(verify(&["kept"], vec![det("person", 0.8, B), car.clone()]));
        assert!(decide(&r, 0.1).contains(&Bucket::Disagreement));
        // Below the verifier's min_confidence: not a miss.
        let mut weak = car;
        weak.score = 0.2;
        r.verify = Some(verify(&["kept"], vec![weak]));
        assert!(!decide(&r, 0.1).contains(&Bucket::Disagreement));
    }

    #[test]
    fn routing_alone_is_no_disagreement() {
        let mut r = record();
        r.routed_from = Some("primary".into());
        r.raw = vec![det("dog", 0.9, B)];
        r.predictions = r.raw.clone();
        assert_eq!(decide(&r, 0.1), vec![Bucket::Positive]);
    }

    #[test]
    fn record_round_trips() {
        let mut r = record();
        r.raw = vec![det("person", 0.9, B)];
        r.verify = Some(verify(&["kept"], Vec::new()));
        r.buckets = decide(&r, 0.1);
        let text = serde_json::to_string(&r).unwrap();
        assert!(text.contains("\"box\":[10.0,10.0,50.0,90.0]"), "{text}");
        assert!(text.contains("\"buckets\":[\"verify_kept\"]"), "{text}");
        let back: Record = serde_json::from_str(&text).unwrap();
        assert_eq!(back, r);
        assert_eq!(Bucket::parse("near_threshold"), Some(Bucket::NearThreshold));
        assert_eq!(Bucket::parse("x"), None);
    }

    // QA-183: the stored outcome strings are the verifier's own spelling, and an unknown one is
    // none of them.
    #[test]
    fn a_candidate_is_its_stored_outcome_only() {
        let c = |o: &str| VerifyCandidate {
            label: "person".into(),
            score: 0.5,
            bbox: B,
            outcome: o.into(),
        };
        assert!(c("kept").is(Outcome::Kept));
        assert!(c("rejected").is(Outcome::Rejected));
        assert!(c("unverified").is(Outcome::Unverified));
        for o in Outcome::ALL {
            assert!(c(o.as_str()).is(o));
            assert!(!c("later").is(o));
            assert!(!c("Rejected").is(o), "exact spelling");
        }
        assert_eq!(SuppressReason::Verify.as_str(), "verify");
    }

    // QA-182: the one mapping from an index line to its frame.
    #[test]
    fn an_index_line_names_its_frame() {
        let mut r = record();
        r.id = "20261021T010203004Z_a".into();
        r.scene = "night".into();
        let line = IndexLine::of(&r, 10);
        assert_eq!(
            line.rel("2026-10-21"),
            "night/cam/2026-10-21/20261021T010203004Z_a.jpg"
        );
    }
}
