//! The service's half of auto-labelling (pure): the proposals `scripts/autolabel.py` returns
//! ([`FrameProposals`]), the triage rule ([`triage`]), proposals to pending `source: auto` label
//! boxes ([`to_boxes`], labels mapped onto the class list's spelling by [`map_label`]) and their
//! merge into a label file ([`apply`]: evidence on boxes it confirms, class suggestions on
//! pending boxes it labels otherwise, then [`merge_proposals`]); and for runs with a model
//! ([`model_boxes`], [`model_triage`]: `review`, or `accept` at a pre-accept score), the
//! disagreement of its detections with the reviewed labels ([`disagreement`]); and for a
//! `missing_only` run, the objects a reviewed frame has no box for ([`apply_missing`], which
//! bypasses [`apply`] and [`merge_proposals`] on purpose). A labelers run writes through
//! [`apply_labelers`]: what a frame takes depends on its status when the file is written, a
//! reviewed frame only takes the confident new objects past its gate ([`gate_drop`]), and every
//! frame takes the labelers' votes on boxes they found again ([`vote`]). A run with a model votes
//! the same way ([`apply_relabel`]); a `missing_only` run does not. On a frame no reviewer has
//! decided, [`tier`] sorts a labelers run's new boxes into dismissed, likely and the rest, then
//! [`apply_spots`] applies what the reviewer taught about fixed places of the camera
//! ([`labels::spots`]): rejected spots hide boxes, parked spots accept them.

use super::catalog::same_class;
use super::{Category, MISSING_THRESHOLD};
use crate::collect::labels::repeats::{REPEAT_SPOT_REASON, same_object};
use crate::collect::labels::spots::{CameraMemory, PARKED_REASON};
use crate::collect::labels::stacks::{DUPLICATE_REASON, STACK_IOU};
use crate::collect::labels::{
    self, AutoInfo, BoxState, LabelBox, LabelFile, MATCH_IOU, MergeReport, Origin, ParkedMark,
    Source, Status, Triage, VlmCheck, inside_share, iou, merge_proposals,
};
use crate::resources::catalog::SAM3_KEY;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

/// Fused score at or above which a box both labelers found is accepted.
pub const ACCEPT_SCORE: f32 = 0.6;
/// `missing_only`: IoU at or above which a training or pending box of the same object covers a
/// detection ([`apply_missing`]). Lower than [`MATCH_IOU`]: a loosely drawn box still labels the
/// object.
pub const MISSING_IOU: f32 = 0.3;
/// `missing_only`: share of a detection inside a training or pending box of the same object at
/// which that box covers it ([`apply_missing`]).
pub const MISSING_INSIDE: f32 = 0.5;

// The tiers of a labelers run's boxes ([`tier`]), measured in `docs/TRAINING.md` "Measuring
// auto-label". Change them only after running `scripts/research/eval_autolabel.py`.

/// Fused score below which a box is dismissed: 5% real on public sets, 0% on a fixed camera.
pub const DISMISS_SCORE: f32 = 0.15;
/// Fused score from which a box with a VLM yes is likely real: 77% / 96% real.
pub const LIKELY_SCORE: f32 = 0.25;
/// Classes never dismissed: 10.7% of real trucks scored below [`DISMISS_SCORE`].
pub const NEVER_DISMISSED: [&str; 1] = ["truck"];
/// One box in this many that would be dismissed stays visible, so that the filter can be checked
/// (about 2%).
pub const AUDIT_ONE_IN: u32 = 50;
/// The reason an audit sample shows on the Review tab.
pub const AUDIT_WHY: &str = "audit sample: normally hidden";
/// Fused score from which a box stays visible at a rejected spot, whatever the spot's record:
/// a real object at a bad spot on an unlucky frame.
pub const SPOT_KEEP_SCORE: f32 = 0.5;
/// The reason a box at a parked spot stays flagged: the VLM said no, so the object may have left.
pub const PARKED_GONE_WHY: &str =
    "at a parked spot, but the VLM sees nothing there: it may have left";

/// The VLM check of a box, as the script reports it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VlmOut {
    #[serde(default)]
    pub model: String,
    pub answer: String,
    pub expected: String,
    pub ok: bool,
}

/// One fused box of the script.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Proposal {
    pub label: String,
    pub bbox: [f32; 4],
    pub score: f32,
    /// Best score per labeler (`gdino` or `sam3`, `rfdetr`).
    #[serde(default)]
    pub sources: BTreeMap<String, f32>,
    #[serde(default)]
    pub agree: bool,
    #[serde(default)]
    pub vlm: Option<VlmOut>,
    #[serde(default)]
    pub species_scores: BTreeMap<String, f32>,
}

/// One output line of the script.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FrameProposals {
    pub frame: String,
    #[serde(default)]
    pub boxes: Vec<Proposal>,
}

/// The triage of a proposal on a `scene` frame: `accept` only when both labelers agree, the
/// fused score is at least `accept_score`, it is not a person or an animal at night (`night` /
/// `ir`), the VLM did not object, SigLIP ranks the box's own species first and SAM 3 did not find
/// it ([`ACCEPT_SCORE`] was tuned on MM-Grounding-DINO + RF-DETR, and a bulk accept reaches the
/// exports); else `review` with the reasons.
pub fn triage(p: &Proposal, scene: &str, accept_score: f32) -> (Triage, Vec<String>) {
    let mut why = Vec::new();
    if !p.agree {
        let who: Vec<&str> = p.sources.keys().map(String::as_str).collect();
        why.push(format!(
            "found by {} only",
            if who.is_empty() {
                "one labeler".to_string()
            } else {
                who.join(", ")
            }
        ));
    }
    if p.score < accept_score {
        why.push(format!("score {:.2} below {accept_score:.2}", p.score));
    }
    if p.sources.contains_key(SAM3_KEY) {
        why.push("found by SAM 3: always reviewed".to_string());
    }
    if let Some(n) = night_review(&p.label, scene) {
        why.push(n.to_string());
    }
    if let Some(v) = &p.vlm
        && !v.ok
    {
        why.push(vlm_objection(p, v));
    }
    if let Some((top, s)) = siglip_objection(&p.label, &p.species_scores) {
        why.push(format!("SigLIP ranks {top} first ({s:.2})"));
    }
    if why.is_empty() {
        (Triage::Accept, why)
    } else {
        (Triage::Review, why)
    }
}

/// A VLM "no" as a reason. `expected: "yes"` marks the yes/no question; label files of older runs
/// hold the seven-way answer (`plant`, expected `person`), which the Review tab still shows.
fn vlm_objection(p: &Proposal, v: &VlmOut) -> String {
    let who = if v.model.is_empty() {
        "the VLM"
    } else {
        &v.model
    };
    if v.expected == "yes" {
        return format!("{who} sees no {}", p.label);
    }
    let answer = if v.answer.is_empty() {
        "nothing"
    } else {
        &v.answer
    };
    format!("{who} says {answer} (expected {})", v.expected)
}

/// The species SigLIP ranks first when that is not `label`: an objection to the box's class.
fn siglip_objection<'a>(
    label: &str,
    scores: &'a BTreeMap<String, f32>,
) -> Option<(&'a String, &'a f32)> {
    scores
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .filter(|(k, _)| !k.eq_ignore_ascii_case(label))
}

/// Why a `label` box on a `scene` frame is always reviewed: a person or an animal at night
/// (`night` / `ir`). The labelers' [`triage`] and a model's [`model_triage`] share the rule.
pub fn night_review(label: &str, scene: &str) -> Option<&'static str> {
    if !matches!(scene, "night" | "ir") {
        return None;
    }
    match Category::of(label) {
        Category::Person => Some("a person at night: always reviewed"),
        Category::Animal => Some("an animal at night: always reviewed"),
        _ => None,
    }
}

fn round(v: f32, scale: f32) -> f32 {
    (v * scale).round() / scale
}

/// The class of `classes` a detection `label` is: the same name, else the same name ignoring
/// case, else the same COCO class through the aliases (a VOC-spelled model's `motorbike`,
/// `tvmonitor`, `pottedplant`, `aeroplane`, `sofa`, `diningtable` are the list's `motorcycle`,
/// `tv`, `potted plant`, `airplane`, `couch`, `dining table`). None: not a class of the run.
pub fn map_label<'a>(label: &str, classes: &'a [String]) -> Option<&'a str> {
    let l = label.trim();
    classes
        .iter()
        .find(|c| c.as_str() == l)
        .or_else(|| classes.iter().find(|c| c.eq_ignore_ascii_case(l)))
        .or_else(|| classes.iter().find(|c| same_class(c, l)))
        .map(String::as_str)
}

/// [`to_boxes`] with what it left out and renamed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Boxes {
    pub boxes: Vec<LabelBox>,
    /// Proposals whose label is no class of the run, also after [`map_label`].
    pub other_class: usize,
    /// Labels given the class list's spelling: model label -> class (`motorbike` ->
    /// `motorcycle`); a case-only change is not listed.
    pub renamed: BTreeMap<String, String>,
}

/// The pending `source: auto` boxes of `proposals` for a `w` x `h` frame of `scene`. Each label
/// is mapped onto `classes` ([`map_label`]; the original kept in `model_label` when it is not
/// just another case): labels outside `classes` and boxes smaller than [`labels::MIN_BOX_PX`]
/// after clamping are left out; coordinates rounded to 0.1 px, scores to 4 decimals.
pub fn to_boxes(
    proposals: &[Proposal],
    classes: &[String],
    scene: &str,
    w: u32,
    h: u32,
    accept_score: f32,
) -> Vec<LabelBox> {
    to_boxes_counted(proposals, classes, scene, w, h, accept_score).boxes
}

/// [`to_boxes`], counting the proposals of other classes and the renamed labels.
pub fn to_boxes_counted(
    proposals: &[Proposal],
    classes: &[String],
    scene: &str,
    w: u32,
    h: u32,
    accept_score: f32,
) -> Boxes {
    let mut out = Boxes::default();
    for p in proposals {
        if !(p.bbox.iter().all(|v| v.is_finite()) && p.score.is_finite()) {
            continue;
        }
        let Some(class) = map_label(&p.label, classes) else {
            out.other_class += 1;
            continue;
        };
        let model_label = (!class.eq_ignore_ascii_case(p.label.trim())).then(|| {
            p.label
                .chars()
                .take(labels::MAX_LABEL_LEN)
                .collect::<String>()
        });
        let mapped;
        let p = if class == p.label {
            p
        } else {
            mapped = Proposal {
                label: class.to_string(),
                ..p.clone()
            };
            &mapped
        };
        if let Some(b) = proposal_box(p, scene, w, h, accept_score) {
            if let Some(m) = &model_label {
                out.renamed.insert(m.clone(), class.to_string());
            }
            out.boxes.push(LabelBox { model_label, ..b });
        }
    }
    out
}

/// The box of one proposal whose label is a class of the run (None: too small).
fn proposal_box(p: &Proposal, scene: &str, w: u32, h: u32, accept_score: f32) -> Option<LabelBox> {
    let b = labels::clamp_box(p.bbox, w, h).map(|v| round(v, 10.0));
    if b[2] - b[0] < labels::MIN_BOX_PX || b[3] - b[1] < labels::MIN_BOX_PX {
        return None;
    }
    let (t, why) = triage(p, scene, accept_score);
    let labelers: BTreeMap<String, f32> = p
        .sources
        .iter()
        .filter(|(_, s)| s.is_finite())
        .map(|(k, s)| (k.clone(), round(s.clamp(0.0, 1.0), 1e4)))
        .collect();
    Some(LabelBox {
        label: p.label.clone(),
        bbox: b,
        source: Source::Auto,
        score: Some(round(p.score.clamp(0.0, 1.0), 1e4)),
        state: BoxState::Pending,
        origin: None,
        reason: Some(format!(
            "auto: {}",
            labelers.keys().cloned().collect::<Vec<_>>().join(" + ")
        )),
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: AutoInfo {
            agree: Some(p.agree),
            triage: Some(t),
            why,
            labelers,
            vlm: p.vlm.as_ref().map(|v| VlmCheck {
                model: v.model.chars().take(labels::MAX_LABEL_LEN).collect(),
                answer: v.answer.chars().take(labels::MAX_LABEL_LEN).collect(),
                expected: v.expected.chars().take(labels::MAX_LABEL_LEN).collect(),
                ok: v.ok,
            }),
            species_scores: p
                .species_scores
                .iter()
                .filter(|(_, s)| s.is_finite())
                .map(|(k, s)| (k.clone(), round(s.clamp(0.0, 1.0), 1e4)))
                .collect(),
            auto_label: None,
            auto_score: None,
            likely: false,
            audit: false,
            spot_rejections: None,
            parked: None,
        },
    })
}

/// What [`apply`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub merge: MergeReport,
    /// Existing boxes of the proposal's label that got its evidence.
    pub annotated: usize,
    /// Pending boxes of another label that got the proposal's class as a suggestion.
    pub class_suggestions: usize,
}

/// Merge `boxes` into `f`. Per proposal, against the existing (pre-merge) boxes at IoU >=
/// [`MATCH_IOU`]:
///
/// - the best one of the same label without auto-labeler evidence gets the proposal's (its
///   state and decision unchanged: a model box the VLM calls a rock shows that on the Review
///   page); [`merge_proposals`] then drops the proposal;
/// - when no box of the same label matches and the best box is a pending one of another label
///   without evidence, it gets the proposal's evidence as a class suggestion (`auto_label`,
///   `auto_score`, triage `review`, a reason) and the proposal is dropped (no duplicate box);
/// - everything else goes to [`merge_proposals`] by its rule.
pub fn apply(f: &mut LabelFile, boxes: Vec<LabelBox>) -> ApplyReport {
    /// Index of the existing box (the first `existing`) of highest IoU >= [`MATCH_IOU`] with
    /// `p` among those `keep` accepts.
    fn best(
        f: &LabelFile,
        existing: usize,
        p: &LabelBox,
        keep: impl Fn(&LabelBox) -> bool,
    ) -> Option<usize> {
        f.boxes
            .iter()
            .take(existing)
            .enumerate()
            .map(|(i, b)| (i, iou(&b.bbox, &p.bbox), b))
            .filter(|(_, v, b)| *v >= MATCH_IOU && keep(b))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _, _)| i)
    }
    let existing = f.boxes.len();
    let mut r = ApplyReport::default();
    let mut rest = Vec::with_capacity(boxes.len());
    for p in boxes {
        let same_label = |b: &LabelBox| b.label == p.label;
        if let Some(i) = best(f, existing, &p, |b| same_label(b) && b.auto.is_empty())
            && let Some(b) = f.boxes.get_mut(i)
        {
            b.auto = p.auto.clone();
            r.annotated += 1;
            rest.push(p);
            continue;
        }
        let suggest = if best(f, existing, &p, same_label).is_none() {
            best(f, existing, &p, |_| true).filter(|i| {
                f.boxes
                    .get(*i)
                    .is_some_and(|b| b.state == BoxState::Pending && b.auto.is_empty())
            })
        } else {
            None
        };
        if let Some(b) = suggest.and_then(|i| f.boxes.get_mut(i)) {
            let mut auto = p.auto.clone();
            auto.triage = Some(Triage::Review);
            let score = p.score.map(|s| format!(" ({s:.2})")).unwrap_or_default();
            auto.why.push(format!(
                "the auto-labelers say {}{score}; the box says {}",
                p.label, b.label
            ));
            auto.auto_label = Some(p.label.clone());
            auto.auto_score = p.score;
            b.auto = auto;
            r.class_suggestions += 1;
            continue;
        }
        rest.push(p);
    }
    r.merge = merge_proposals(f, rest);
    r
}

/// One detection of a trained model.
#[derive(Debug, Clone, PartialEq)]
pub struct Det {
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

/// How a model's detections disagree with a reviewed label file, over the run's `classes` only:
/// each detection is mapped onto `classes` ([`map_label`]); one of them matching no training box
/// (accepted or added, any class) at IoU >= [`MATCH_IOU`] is new, one whose best match has
/// another label is a class change, a training box of one of them no detection matches is
/// missed. Detections and training boxes of other classes are no disagreement: a run for a few
/// classes does not flag what it did not look for. None when the file is not reviewed (there is
/// nothing to disagree with) or everything matches.
pub fn disagreement(
    previous: &LabelFile,
    dets: &[Det],
    model: &str,
    classes: &[String],
) -> Option<String> {
    if previous.status != Status::Reviewed {
        return None;
    }
    let in_run = |l: &str| classes.iter().any(|c| c.eq_ignore_ascii_case(l));
    let train: Vec<&LabelBox> = previous.training().collect();
    let dets: Vec<(&str, &Det)> = dets
        .iter()
        .filter_map(|d| map_label(&d.label, classes).map(|c| (c, d)))
        .collect();
    let (mut new, mut changed) = (0usize, 0usize);
    for (label, d) in &dets {
        let best = train
            .iter()
            .map(|b| (iou(&b.bbox, &d.bbox), *b))
            .filter(|(v, _)| *v >= MATCH_IOU)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            None => new += 1,
            Some((_, b)) if !same_class(&b.label, label) => changed += 1,
            Some(_) => {}
        }
    }
    let missed = train
        .iter()
        .filter(|b| in_run(&b.label))
        .filter(|b| !dets.iter().any(|(_, d)| iou(&b.bbox, &d.bbox) >= MATCH_IOU))
        .count();
    if new + changed + missed == 0 {
        return None;
    }
    let mut parts = Vec::new();
    if new > 0 {
        parts.push(format!("{new} new box{}", if new == 1 { "" } else { "es" }));
    }
    if missed > 0 {
        parts.push(format!("{missed} missed"));
    }
    if changed > 0 {
        parts.push(format!(
            "{changed} class change{}",
            if changed == 1 { "" } else { "s" }
        ));
    }
    Some(format!("re-label by {model}: {}", parts.join(", ")))
}

/// A run with a model ([`model_boxes`]).
#[derive(Debug, Clone, Copy)]
pub struct ModelRun<'a> {
    /// The model's name (the boxes' labeler, the reasons).
    pub model: &'a str,
    /// The run's classes.
    pub classes: &'a [String],
    /// The frame's scene (`day` | `night` | `ir`).
    pub scene: &'a str,
    /// Pre-accept: boxes at or above this score are triaged `accept` (None: every box `review`).
    pub accept_score: Option<f32>,
}

/// The triage of a model's `label` box at `score`: `review` ("re-label by <model>") unless the
/// run pre-accepts at `accept_score` and the score reaches it ("model <name> at 0.87 >= 0.80");
/// a person or an animal at night is always reviewed ([`night_review`], as for the labelers).
pub fn model_triage(label: &str, score: f32, run: &ModelRun<'_>) -> (Triage, Vec<String>) {
    let base = format!("re-label by {}", run.model);
    let Some(min) = run.accept_score else {
        return (Triage::Review, vec![base]);
    };
    if let Some(n) = night_review(label, run.scene) {
        return (Triage::Review, vec![base, n.to_string()]);
    }
    if score >= min {
        (
            Triage::Accept,
            vec![format!("model {} at {score:.2} >= {min:.2}", run.model)],
        )
    } else {
        (
            Triage::Review,
            vec![base, format!("score {score:.2} below {min:.2}")],
        )
    }
}

/// The pending auto boxes of a model's detections on a `w` x `h` frame: labels mapped onto the
/// run's classes ([`to_boxes_counted`]), triaged by [`model_triage`] (a model is checked against
/// the reviewer, unless the run pre-accepts its sure boxes).
pub fn model_boxes(dets: &[Det], run: &ModelRun<'_>, w: u32, h: u32) -> Boxes {
    let props: Vec<Proposal> = dets
        .iter()
        .map(|d| Proposal {
            label: d.label.clone(),
            bbox: d.bbox,
            score: d.score,
            sources: [(run.model.to_string(), d.score)].into(),
            agree: false,
            vlm: None,
            species_scores: BTreeMap::new(),
        })
        .collect();
    let mut out = to_boxes_counted(&props, run.classes, run.scene, w, h, ACCEPT_SCORE);
    for b in &mut out.boxes {
        let (t, why) = model_triage(&b.label, b.score.unwrap_or(0.0), run);
        b.auto.agree = None;
        b.auto.triage = Some(t);
        b.auto.why = why;
        b.reason = Some(format!("model {}", run.model));
    }
    out
}

/// What [`apply_relabel`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelabelReport {
    pub apply: ApplyReport,
    /// Existing boxes that gained the model as a labeler ([`vote`]). A vote changes nothing a
    /// reviewer has to look at, so it queues no frame.
    pub votes: usize,
    /// Detections left out because the frame is closed ([`LabelersPath::Closed`]).
    pub closed: usize,
    /// The note of a frame that went back to `unreviewed` ([`disagreement`]).
    pub disagreement: Option<String>,
}

/// Apply a model's detections to a frame's label file: the disagreement note (reviewed files
/// only; the file goes back to `unreviewed`), `boxes` (its [`model_boxes`]) merged by [`apply`],
/// and the model's votes on the boxes it found again ([`vote`]), so its name survives a proposal
/// the merge drops. The votes come after [`apply`], as on [`apply_labelers`]' unreviewed path: a
/// vote on a box without evidence would stop it taking the proposal's. A skipped or confirmed-empty
/// frame ([`LabelersPath::Closed`]) takes the votes and nothing else, as on [`apply_labelers`]: a
/// box there would send it back to `unreviewed`, out of the exports, and lose its hard negative.
pub fn apply_relabel(
    f: &mut LabelFile,
    dets: &[Det],
    boxes: Vec<LabelBox>,
    model: &str,
    classes: &[String],
) -> RelabelReport {
    if labelers_path(f) == LabelersPath::Closed {
        let votes = vote(f, &boxes, f.boxes.len());
        return RelabelReport {
            votes,
            closed: boxes.len(),
            ..RelabelReport::default()
        };
    }
    let why = disagreement(f, dets, model, classes);
    undismiss_found(f, &boxes);
    let existing = f.boxes.len();
    let merged = apply(f, boxes.clone());
    let votes = vote(f, &boxes, existing);
    if let Some(w) = &why {
        f.status = Status::Unreviewed;
        f.disagreement = Some(w.clone());
    }
    RelabelReport {
        apply: merged,
        votes,
        closed: 0,
        disagreement: why,
    }
}

/// Whether a `missing_only` run may look at `f`: a reviewed frame, or an unreviewed one that a
/// re-label sent back (`disagreement`, so a second model can add to the first one's proposals). A
/// frame confirmed empty is left out: accepting a box there would fail [`labels::normalize`].
pub fn missing_scope(f: &LabelFile) -> bool {
    !f.empty_confirmed
        && (f.status == Status::Reviewed
            || (f.status == Status::Unreviewed && f.disagreement.is_some()))
}

/// What [`apply_missing`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MissingReport {
    /// New pending boxes.
    pub added: usize,
    /// Pending boxes that took the place of an automatic reject.
    pub overwritten: usize,
    /// Detections a training box, a pending box or a box of this run already covers.
    pub covered: usize,
    /// Detections on a box a reviewer rejected.
    pub judged: usize,
    /// Detections with no room left in the file ([`labels::MAX_BOXES`]).
    pub full: usize,
}

impl MissingReport {
    /// The boxes the reviewer has to look at because of this run.
    pub fn candidates(&self) -> usize {
        self.added + self.overwritten
    }
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// Whether `b` covers the detection `p`: IoU >= [`MISSING_IOU`], or at least [`MISSING_INSIDE`] of
/// `p` inside `b`. Only that direction: a small box inside a large detection does not cover it.
/// [`inside_share`] measures the smaller box, which is `p` under the area check; a `p` larger than
/// `b` with that share of it inside `b` has an IoU of at least 1/3 anyway, so the check loses
/// nothing.
fn covers(b: &LabelBox, p: &LabelBox) -> bool {
    iou(&b.bbox, &p.bbox) >= MISSING_IOU
        || (area(&p.bbox) <= area(&b.bbox) && inside_share(&p.bbox, &b.bbox) >= MISSING_INSIDE)
}

/// A rejected box no reviewer judged: a [`labels::bulk::LOW_SCORE_REASON`] bulk reject, a dismissal
/// ([`labels::bulk::is_dismissal`]), or a model hint (suppressed, below threshold, verifier find)
/// the page's save default rejected. A reject
/// reason, a spot edit or a duplicate reject means somebody looked, whatever the box's origin. A
/// spot edit leaves its mark only on a box without a `reason` of its own, so a suppressed hint
/// rejected by one cannot be told from the save default.
fn unjudged_reject(b: &LabelBox) -> bool {
    let reason = b.reason.as_deref();
    b.state == BoxState::Rejected
        && b.reject_reason.is_none()
        && !matches!(reason, Some(REPEAT_SPOT_REASON | DUPLICATE_REASON))
        && (reason == Some(labels::bulk::LOW_SCORE_REASON)
            || labels::bulk::is_dismissal(reason)
            || (b.source == Source::Model && b.origin != Some(Origin::Prediction)))
}

/// The note of a frame with `f`'s pending proposals: who proposed them and how many, within
/// [`labels::MAX_DISAGREEMENT_LEN`] characters (names past that are counted, not listed).
fn missing_note(f: &LabelFile, model: &str) -> String {
    let pending: Vec<&LabelBox> = f
        .boxes
        .iter()
        .filter(|b| b.state == BoxState::Pending && !b.auto.labelers.is_empty())
        .collect();
    let mut names: BTreeSet<&str> = pending
        .iter()
        .flat_map(|b| b.auto.labelers.keys().map(String::as_str))
        .collect();
    names.insert(model);
    let n = pending.len();
    let tail = format!(": {n} new box{}", if n == 1 { "" } else { "es" });
    // Room for the prefix, the tail and " and 999 more".
    let room = labels::MAX_DISAGREEMENT_LEN
        .saturating_sub("re-label by ".len() + tail.len() + " and 999 more".len());
    let mut listed = String::new();
    let mut left = names.len();
    for name in names {
        let extra = name.chars().count() + if listed.is_empty() { 0 } else { 2 };
        if listed.chars().count() + extra > room {
            break;
        }
        if !listed.is_empty() {
            listed.push_str(", ");
        }
        listed.push_str(name);
        left -= 1;
    }
    match (listed.is_empty(), left) {
        (true, _) => format!("re-label by {left} model(s){tail}"),
        (false, 0) => format!("re-label by {listed}{tail}"),
        (false, _) => format!("re-label by {listed} and {left} more{tail}"),
    }
}

/// Propose the objects `f` has no box for: `boxes` are `model`'s [`model_boxes`], taken highest
/// score first. Does nothing outside [`missing_scope`]. Per box, against the boxes of the same
/// object ([`same_object`], so a car on a labelled truck is no new object):
///
/// - a training box, a pending box or a box added by this call that [`covers`] it: dropped;
/// - a rejected box at IoU >= [`MATCH_IOU`] that a reviewer judged (anything but
///   [`unjudged_reject`]): dropped, whatever else matches;
/// - an [`unjudged_reject`] at that IoU: the box takes its place (the closest one; any others
///   there are removed), so `hard_negatives.json` does not call a real object a false positive;
/// - else it is added.
///
/// Boxes are `pending`, triaged `review`; existing boxes get no evidence and no class suggestion.
/// A frame that gains a box goes back to `unreviewed` with a `disagreement` note naming the
/// labelers of its pending boxes, and loses [`LabelFile::overlaps_checked`] (the reviewer's look
/// did not cover this box), remembered in [`LabelFile::checked_before_reopen`] for a restore. A
/// frame that gains none is left exactly as it was.
pub fn apply_missing(f: &mut LabelFile, mut boxes: Vec<LabelBox>, model: &str) -> MissingReport {
    let mut r = MissingReport::default();
    if !missing_scope(f) {
        return r;
    }
    boxes.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
    for mut p in boxes {
        let same = |b: &&LabelBox| same_object(&b.label, &p.label);
        if f.boxes
            .iter()
            .filter(same)
            .filter(|b| b.is_training() || b.state == BoxState::Pending)
            .any(|b| covers(b, &p))
        {
            r.covered += 1;
            continue;
        }
        // The same-object rejects at this place, closest first.
        let mut rejects: Vec<(usize, f32, bool)> = f
            .boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| same(b) && b.state == BoxState::Rejected)
            .map(|(i, b)| (i, iou(&b.bbox, &p.bbox), unjudged_reject(b)))
            .filter(|x| x.1 >= MATCH_IOU)
            .collect();
        if rejects.iter().any(|x| !x.2) {
            r.judged += 1;
            continue;
        }
        rejects.sort_by(|a, b| b.1.total_cmp(&a.1));
        let closest = rejects.first().map(|x| x.0);
        if closest.is_none() && f.boxes.len() >= labels::MAX_BOXES {
            r.full += 1;
            continue;
        }
        p.state = BoxState::Pending;
        p.auto.triage = Some(Triage::Review);
        match closest.and_then(|i| f.boxes.get_mut(i)) {
            Some(slot) => {
                *slot = p;
                r.overwritten += 1;
            }
            None => {
                f.boxes.push(p);
                r.added += 1;
            }
        }
        let gone: BTreeSet<usize> = rejects.iter().skip(1).map(|x| x.0).collect();
        if !gone.is_empty() {
            let mut i = 0;
            f.boxes.retain(|_| {
                i += 1;
                !gone.contains(&(i - 1))
            });
        }
    }
    if r.candidates() > 0 {
        f.status = Status::Unreviewed;
        f.disagreement = Some(missing_note(f, model));
        f.checked_before_reopen |= f.overlaps_checked;
        f.overlaps_checked = false;
    }
    r
}

/// Why a proposal may not reopen a reviewed frame ([`gate_drop`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateDrop {
    /// The labelers do not agree and no labeler reached [`MISSING_THRESHOLD`].
    LowScore,
    /// The VLM said no, or SigLIP ranks another species first.
    Objected,
    /// It sits on a decided box of another object.
    OtherObject,
    /// It sits on a rejected box of the object that no reviewer judged ([`unjudged_reject`]).
    OnReject,
    /// One labeler found it and the VLM did not say yes (it did not run or skipped the box).
    Unchecked,
}

/// The proposals the gate of a reviewed frame left out, per reason (each one counted once, at the
/// first reason that applies, in this order).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateReport {
    /// The labelers do not agree and no labeler reached [`MISSING_THRESHOLD`].
    pub low_score: usize,
    /// The VLM said no, or SigLIP ranks another species first.
    pub objected: usize,
    /// On a decided box of another object at IoU >= [`MATCH_IOU`].
    pub other_object: usize,
    /// On a rejected box of the object no reviewer judged, at IoU >= [`MATCH_IOU`].
    pub on_reject: usize,
    /// One labeler found it and the VLM did not say yes. Like the other reasons this comes before
    /// [`apply_missing`], so it also counts a box that only repeats one already on the frame.
    pub unchecked: usize,
}

impl GateReport {
    pub fn total(&self) -> usize {
        self.low_score + self.objected + self.other_object + self.on_reject + self.unchecked
    }
}

/// What [`apply_labelers`] did. `apply` is the unreviewed path's, `gate` and `missing` the reviewed
/// path's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabelersReport {
    pub apply: ApplyReport,
    pub gate: GateReport,
    pub missing: MissingReport,
    /// Existing boxes that gained a labeler ([`vote`]).
    pub votes: usize,
    /// Proposals a skipped or confirmed-empty frame took no box for.
    pub closed: usize,
    /// New boxes dismissed ([`Tier::Dismissed`]).
    pub dismissed: usize,
    /// New boxes kept visible to check the dismissals ([`Tier::Audit`]).
    pub audit: usize,
    /// New boxes likely real ([`Tier::Likely`]).
    pub likely: usize,
    /// New boxes dismissed at a rejected spot ([`apply_spots`]).
    pub spot_dismissed: usize,
    /// Pending boxes accepted at a parked spot ([`apply_spots`]).
    pub parked_accepted: usize,
    /// Of `parked_accepted`, boxes this run added.
    pub parked_new: usize,
}

impl LabelersReport {
    /// The boxes changed for a reviewer: the frame is queued for review. A vote is no such change.
    pub fn for_review(&self) -> bool {
        self.apply.merge.added > 0
            || self.apply.annotated > 0
            || self.apply.class_suggestions > 0
            || self.missing.candidates() > 0
    }
}

/// How [`apply_labelers`] treats a frame, from its status now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelersPath {
    /// No review so far: [`apply`].
    Merge,
    /// A reviewer's labels: the gate, then [`apply_missing`].
    Gated,
    /// Decided as a whole (skipped, or confirmed empty on any status: a save refuses a training
    /// box there): no box comes in.
    Closed,
}

/// A frame a re-label sent back ([`LabelFile::disagreement`]) still holds a reviewer's decisions, so
/// it takes the reviewed path too: the default `status=unreviewed` selects it, and [`apply`] would
/// write on those decisions and add every proposal.
fn labelers_path(f: &LabelFile) -> LabelersPath {
    let reviewed = f.status == Status::Reviewed
        || (f.status == Status::Unreviewed && f.disagreement.is_some());
    if f.status == Status::Skipped || f.empty_confirmed {
        LabelersPath::Closed
    } else if reviewed {
        LabelersPath::Gated
    } else {
        LabelersPath::Merge
    }
}

/// Why the proposal `p` may not reopen the reviewed frame `f` (None: it may). It passes when the
/// labelers agree or its best source score reaches [`MISSING_THRESHOLD`] (the source's, because the
/// fused score halves a box one labeler found), no check objects, it sits on no decided box of
/// another object ([`apply_missing`] compares boxes of one object only, so a `dog` on an accepted
/// `cat` would pass it) and on no [`unjudged_reject`] (which [`apply_missing`] would overwrite: on a
/// labelers run a rejected box never changes, `--missing-only` is the run that may replace one), and
/// a box only one labeler found has a VLM yes. A check that did not run or gave no clear answer is
/// no evidence: `--vlm off`, a box under `--vlm-min-score` and an unclear reply (`vlm_skipped`, which
/// no field here reads) all leave `vlm` empty, and such a lone box cannot reopen a frame at any score.
fn gate_drop(f: &LabelFile, p: &LabelBox) -> Option<GateDrop> {
    let lone = p.auto.agree != Some(true);
    let best = p.auto.labelers.values().copied().fold(0.0, f32::max);
    if lone && best < MISSING_THRESHOLD {
        return Some(GateDrop::LowScore);
    }
    if p.auto.vlm.as_ref().is_some_and(|v| !v.ok)
        || siglip_objection(&p.label, &p.auto.species_scores).is_some()
    {
        return Some(GateDrop::Objected);
    }
    let here = |b: &LabelBox| iou(&b.bbox, &p.bbox) >= MATCH_IOU;
    let other_object = |b: &LabelBox| !same_object(&b.label, &p.label);
    // A dismissed box is no decision: nobody said what is there.
    if f.boxes
        .iter()
        .any(|b| b.state != BoxState::Pending && !b.is_dismissed() && other_object(b) && here(b))
    {
        return Some(GateDrop::OtherObject);
    }
    if f.boxes
        .iter()
        .any(|b| !other_object(b) && unjudged_reject(b) && here(b))
    {
        return Some(GateDrop::OnReject);
    }
    if lone && !p.auto.vlm.as_ref().is_some_and(|v| v.ok) {
        return Some(GateDrop::Unchecked);
    }
    None
}

/// The reason a box past the gate carries on the Review tab.
fn gate_reason(p: &LabelBox) -> String {
    let best = p.auto.labelers.iter().max_by(|a, b| a.1.total_cmp(b.1));
    match best {
        Some((k, s)) if p.auto.agree != Some(true) => {
            format!("new on a reviewed frame: {k} at {s:.2}")
        }
        _ => "new on a reviewed frame: the labelers agree".to_string(),
    }
}

/// Record the labelers of `proposals` on the box each one found again, among the first `existing`
/// boxes of `f`: its best match of the same object ([`same_object`]) at IoU >= [`STACK_IOU`], any
/// state. Only keys are added to `labelers`, a stored score stays, and nothing else of the box
/// changes. A pending box without evidence gets no vote: having none is what lets [`apply`] annotate
/// it and [`labels::auto_accept_eligible`] pass it, so a vote would change how it is triaged. Stops
/// at [`labels::MAX_LABELERS`] and skips a name over [`labels::MAX_LABEL_LEN`] (a model's file name
/// can be), past either of which a save fails. Returns the boxes that gained a labeler.
fn vote(f: &mut LabelFile, proposals: &[LabelBox], existing: usize) -> usize {
    let mut gained = 0;
    for p in proposals {
        let best = f
            .boxes
            .iter()
            .take(existing)
            .enumerate()
            .filter(|(_, b)| same_object(&b.label, &p.label))
            .map(|(i, b)| (i, iou(&b.bbox, &p.bbox)))
            .filter(|(_, v)| *v >= STACK_IOU)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some(b) = best.and_then(|(i, _)| f.boxes.get_mut(i)) else {
            continue;
        };
        if b.state == BoxState::Pending && b.auto.is_empty() {
            continue;
        }
        let before = b.auto.labelers.len();
        for (k, s) in &p.auto.labelers {
            let short = k.chars().count() <= labels::MAX_LABEL_LEN;
            if short && b.auto.labelers.len() < labels::MAX_LABELERS {
                b.auto.labelers.entry(k.clone()).or_insert(*s);
            }
        }
        gained += usize::from(b.auto.labelers.len() > before);
    }
    gained
}

/// How much a reviewer should trust a new box of a labelers run ([`tier`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Too unlikely to show: rejected with [`labels::bulk::DISMISSED_REASON`], recoverable on the
    /// Review tab.
    Dismissed,
    /// Would be dismissed, but [`audit_sample`] keeps it visible to check the filter.
    Audit,
    /// Shown first ([`AutoInfo::likely`]).
    Likely,
    /// Everything else: visible, shown after the likely boxes.
    Unlikely,
}

/// Whether a box is among the [`AUDIT_ONE_IN`] kept back from dismissal: a hash of its frame, label
/// and whole-pixel box, so a rerun gives the same box the same answer.
pub fn audit_sample(frame: &str, label: &str, bbox: &[f32; 4]) -> bool {
    use sha2::Digest;
    let [x0, y0, x1, y1] = bbox;
    let key = format!("{frame}\n{label}\n{x0:.0},{y0:.0},{x1:.0},{y1:.0}");
    let digest = sha2::Sha256::digest(key.as_bytes());
    let n = digest
        .iter()
        .take(4)
        .fold(0u32, |n, b| (n << 8) | u32::from(*b));
    n % AUDIT_ONE_IN == 0
}

/// A pending auto box with no sign of a reviewer.
fn untouched(b: &LabelBox) -> bool {
    b.state == BoxState::Pending
        && b.source == Source::Auto
        && b.orig_bbox.is_none()
        && !b.difficult
        && b.reject_reason.is_none()
}

fn found_by_sam3(b: &LabelBox) -> bool {
    b.auto.labelers.contains_key(SAM3_KEY)
}

/// [`LIKELY_SCORE`] and a yes to the yes/no question.
fn is_likely(b: &LabelBox, score: f32) -> bool {
    score >= LIKELY_SCORE && b.auto.vlm.as_ref().is_some_and(VlmCheck::says_yes)
}

/// The [`Tier`] of `b`, a box of a labelers run on `frame` (thresholds and their measured reasons
/// above [`DISMISS_SCORE`]). Only a pending auto box with no sign of a reviewer gets a tier other
/// than [`Tier::Unlikely`].
///
/// - Likely: fused score >= [`LIKELY_SCORE`] and a yes to the yes/no question. An older run's
///   seven-way VLM answer is never one.
/// - Dismissed: fused score < [`DISMISS_SCORE`], not triaged `accept`, not of a class in
///   [`NEVER_DISMISSED`], not found by SAM 3, and `sam3_run` false (the thresholds were measured on
///   MM-Grounding-DINO + RF-DETR). Only the score decides: a VLM no never hides a box, 40% of
///   those were real on public sets. A class only one labeler can report is not halved by the
///   fusion, so it stays at or above that labeler's threshold (0.2 for MM-Grounding-DINO).
pub fn tier(b: &LabelBox, frame: &str, sam3_run: bool) -> Tier {
    let Some(score) = b.score.filter(|_| untouched(b)) else {
        return Tier::Unlikely;
    };
    if is_likely(b, score) {
        return Tier::Likely;
    }
    let protected = sam3_run
        || found_by_sam3(b)
        || b.auto.triage == Some(Triage::Accept)
        || NEVER_DISMISSED
            .iter()
            .any(|c| c.eq_ignore_ascii_case(b.label.trim()));
    if score >= DISMISS_SCORE || protected {
        Tier::Unlikely
    } else if audit_sample(frame, &b.label, &b.bbox) {
        Tier::Audit
    } else {
        Tier::Dismissed
    }
}

/// Give the boxes of `f` from index `from` on their [`tier`], counted in `r`.
fn apply_tiers(f: &mut LabelFile, from: usize, frame: &str, name: &str, r: &mut LabelersReport) {
    for b in f.boxes.iter_mut().skip(from) {
        match tier(b, frame, name == SAM3_KEY) {
            Tier::Dismissed => {
                b.state = BoxState::Rejected;
                b.reason = Some(labels::bulk::DISMISSED_REASON.into());
                r.dismissed += 1;
            }
            Tier::Audit => {
                b.auto.audit = true;
                b.auto.why.push(AUDIT_WHY.to_string());
                r.audit += 1;
            }
            Tier::Likely => {
                b.auto.likely = true;
                r.likely += 1;
            }
            Tier::Unlikely => {}
        }
    }
}

/// The spots of a frame's camera for a run ([`labels::spots::Snapshot`]).
#[derive(Debug, Clone, Copy)]
pub struct FrameSpots<'a> {
    pub memory: &'a CameraMemory,
    pub width: u32,
    pub height: u32,
}

/// Whether `b`, a new box at a rejected spot, stays visible at fused score `score`. The truck
/// exception of [`tier`] is not here: that one is about the score, a spot rests on a reviewer's
/// own look.
fn spot_keeps(b: &LabelBox, score: f32) -> bool {
    score >= SPOT_KEEP_SCORE || is_likely(b, score) || b.auto.triage == Some(Triage::Accept)
}

/// What the reviewer taught about the camera's places, on the boxes of `f` after [`apply_tiers`].
/// Not on a SAM 3 run or a box SAM 3 found: its boxes never get past a reviewer unseen.
///
/// - Parked spot ([`CameraMemory::parked_at`]: the exact label, IoU >= 0.8), any pending auto box
///   with no sign of a reviewer and no class suggestion, an [`AUDIT_ONE_IN`] sample excepted:
///   accepted, [`PARKED_REASON`], so it leaves the frame's undecided count. The frame stays
///   unreviewed and is marked [`LabelFile::repeat_spot`]. A VLM no in the yes/no contract leaves it
///   pending and flagged instead: the car may have left.
/// - Rejected spot ([`CameraMemory::rejections_at`]), only the boxes this run added (from index
///   `from` on) and not near a parked spot: dismissed with
///   [`labels::bulk::SPOT_DISMISSED_REASON`], unless [`spot_keeps`]; an [`audit_sample`] stays
///   visible as [`Tier::Audit`] does.
fn apply_spots(
    f: &mut LabelFile,
    from: usize,
    frame: &str,
    name: &str,
    spots: FrameSpots<'_>,
    r: &mut LabelersReport,
) {
    if name == SAM3_KEY {
        return;
    }
    let FrameSpots {
        memory,
        width,
        height,
    } = spots;
    let before = r.parked_accepted;
    for (i, b) in f.boxes.iter_mut().enumerate() {
        if !untouched(b) || found_by_sam3(b) || b.auto.audit {
            continue;
        }
        if let Some(p) = memory.parked_at(b, width, height) {
            if b.auto.vlm.as_ref().is_some_and(VlmCheck::says_no) {
                b.auto.triage = Some(Triage::Review);
                if !b.auto.why.iter().any(|w| w == PARKED_GONE_WHY) {
                    b.auto.why.push(PARKED_GONE_WHY.to_string());
                }
            } else if b.auto.auto_label.as_ref().is_none_or(|l| *l == b.label) {
                b.state = BoxState::Accepted;
                b.reason = Some(PARKED_REASON.to_string());
                b.auto.parked = Some(ParkedMark {
                    since: p.from.clone(),
                    bbox: p.bbox,
                });
                r.parked_accepted += 1;
                r.parked_new += usize::from(i >= from);
            }
            continue;
        }
        let Some(score) = b.score.filter(|_| i >= from) else {
            continue;
        };
        let n = memory.rejections_at(b, frame, width, height);
        if n == 0 || spot_keeps(b, score) || memory.parked_near(b, width, height) {
            continue;
        }
        if audit_sample(frame, &b.label, &b.bbox) {
            b.auto.audit = true;
            b.auto.why.push(AUDIT_WHY.to_string());
            r.audit += 1;
        } else {
            b.state = BoxState::Rejected;
            b.reason = Some(labels::bulk::SPOT_DISMISSED_REASON.into());
            b.auto.spot_rejections = u32::try_from(n).ok();
            r.spot_dismissed += 1;
        }
    }
    if r.parked_accepted > before {
        f.repeat_spot = true;
        // The look that flag remembers did not cover this box.
        f.checked_before_reopen = false;
    }
}

/// Take away the dismissed boxes of `f` that one of `proposals` found again, same label at IoU >=
/// [`MATCH_IOU`]: nobody decided on them, so [`merge_proposals`] must not drop the proposal as
/// covered. It comes back as new, and a labelers run tiers it afresh: the same box scores the same
/// and is dismissed again, a stronger find or SAM 3's stays visible.
fn undismiss_found(f: &mut LabelFile, proposals: &[LabelBox]) {
    f.boxes.retain(|b| {
        !b.is_dismissed()
            || !proposals
                .iter()
                .any(|p| p.label == b.label && iou(&b.bbox, &p.bbox) >= MATCH_IOU)
    });
}

/// Merge the proposals of a labelers run into `f`, by the frame's status now (the caller holds the
/// write lock, so a frame reviewed while the run looked at it counts as reviewed):
///
/// | Frame | Result |
/// |---|---|
/// | Unreviewed, not sent back | [`apply`], then votes, then the new boxes' [`tier`], then `spots` ([`apply_spots`]) |
/// | Reviewed, or sent back by a re-label | votes, the gate ([`gate_drop`]), then [`apply_missing`] |
/// | Confirmed empty, any status | votes: a save refuses a training box on such a frame |
/// | Skipped | votes: a skipped frame must not come back and reach an export by a bulk accept |
///
/// On those last three rows accepted and rejected boxes keep everything but what a vote adds to
/// `labelers`. A box past the gate gets a reason and triage `review`; a frame that gains one goes
/// back to `unreviewed` with a `disagreement` note naming `name` (one short name, `sam3`, not the
/// run's joined list) and loses [`LabelFile::bulk_accepted`]: a reviewer has to look at it now.
/// `frame` (`<scene>/<camera>/<date>/<stem>.jpg`) seeds the [`audit_sample`]. `spots` is None for a
/// camera without a snapshot.
pub fn apply_labelers(
    f: &mut LabelFile,
    boxes: Vec<LabelBox>,
    name: &str,
    frame: &str,
    spots: Option<FrameSpots<'_>>,
) -> LabelersReport {
    let mut r = LabelersReport::default();
    let path = labelers_path(f);
    if path == LabelersPath::Merge {
        undismiss_found(f, &boxes);
    }
    let existing = f.boxes.len();
    match path {
        LabelersPath::Merge => {
            // After `apply`: a vote on a box without evidence would stop it taking the proposal's.
            r.apply = apply(f, boxes.clone());
            r.votes = vote(f, &boxes, existing);
            // Only the boxes this run added: a stored box is never tiered again.
            apply_tiers(f, existing, frame, name, &mut r);
            if let Some(spots) = spots {
                apply_spots(f, existing, frame, name, spots, &mut r);
            }
        }
        LabelersPath::Closed => {
            r.votes = vote(f, &boxes, existing);
            r.closed = boxes.len();
        }
        LabelersPath::Gated => {
            r.votes = vote(f, &boxes, existing);
            let mut passed = Vec::with_capacity(boxes.len());
            for mut p in boxes {
                match gate_drop(f, &p) {
                    Some(GateDrop::LowScore) => r.gate.low_score += 1,
                    Some(GateDrop::Objected) => r.gate.objected += 1,
                    Some(GateDrop::OtherObject) => r.gate.other_object += 1,
                    Some(GateDrop::OnReject) => r.gate.on_reject += 1,
                    Some(GateDrop::Unchecked) => r.gate.unchecked += 1,
                    None => {
                        p.auto.why.push(gate_reason(&p));
                        passed.push(p);
                    }
                }
            }
            r.missing = apply_missing(f, passed, name);
            if r.missing.candidates() > 0 {
                f.bulk_accepted = false;
            }
        }
    }
    r
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::labels::repeats::{Decision, History};
    use crate::collect::labels::spots::{CameraSpots, Parked};

    const REL: &str = "day/yard/2026-10-22/20261022T010203004Z_abcd.jpg";

    /// [`super::apply_labelers`] for a camera without a snapshot.
    fn apply_labelers(
        f: &mut LabelFile,
        boxes: Vec<LabelBox>,
        name: &str,
        frame: &str,
    ) -> LabelersReport {
        super::apply_labelers(f, boxes, name, frame, None)
    }

    fn p(label: &str, score: f32, agree: bool) -> Proposal {
        Proposal {
            label: label.into(),
            bbox: [10.0, 10.0, 60.0, 90.0],
            score,
            sources: if agree {
                [("gdino".to_string(), 0.5), ("rfdetr".to_string(), 0.9)].into()
            } else {
                [("gdino".to_string(), score * 2.0)].into()
            },
            agree,
            vlm: None,
            species_scores: BTreeMap::new(),
        }
    }

    fn classes() -> Vec<String> {
        crate::collect::config::default_classes()
    }

    #[test]
    fn only_agreed_high_daytime_boxes_without_objections_are_accepted() {
        assert_eq!(
            triage(&p("car", 0.75, true), "day", ACCEPT_SCORE).0,
            Triage::Accept
        );
        assert_eq!(
            triage(&p("car", 0.6, true), "night", ACCEPT_SCORE).0,
            Triage::Accept
        );
        let (t, why) = triage(&p("car", 0.59, true), "day", ACCEPT_SCORE);
        assert_eq!(t, Triage::Review);
        assert_eq!(why, ["score 0.59 below 0.60"]);
        let (_, why) = triage(&p("car", 0.4, false), "day", ACCEPT_SCORE);
        assert_eq!(why[0], "found by gdino only");
        // Night person / animal: always reviewed, even when sure.
        let (t, why) = triage(&p("person", 0.95, true), "ir", ACCEPT_SCORE);
        assert_eq!(t, Triage::Review);
        assert_eq!(why, ["a person at night: always reviewed"]);
        assert!(triage(&p("raccoon", 0.95, true), "night", ACCEPT_SCORE).1[0].contains("animal"));
        assert_eq!(
            triage(&p("person", 0.95, true), "day", ACCEPT_SCORE).0,
            Triage::Accept
        );
        // A VLM "no" and a species disagreement flag the box.
        let mut v = p("person", 0.9, true);
        v.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "rock".into(),
            expected: "person".into(),
            ok: false,
        });
        let (t, why) = triage(&v, "day", ACCEPT_SCORE);
        assert_eq!(t, Triage::Review);
        assert_eq!(why, ["qwen3-vl-2b says rock (expected person)"]);
        let mut s = p("cat", 0.9, true);
        s.species_scores = [("cat".to_string(), 0.2), ("raccoon".to_string(), 0.7)].into();
        assert_eq!(
            triage(&s, "day", ACCEPT_SCORE).1,
            ["SigLIP ranks raccoon first (0.70)"]
        );
        s.species_scores = [("cat".to_string(), 0.8), ("raccoon".to_string(), 0.2)].into();
        assert_eq!(triage(&s, "day", ACCEPT_SCORE).0, Triage::Accept);
    }

    #[test]
    fn a_box_sam3_found_is_never_accepted() {
        // Everything else would accept it.
        let mut b = p("car", 0.95, true);
        b.sources = [("sam3".to_string(), 0.95), ("rfdetr".to_string(), 0.95)].into();
        let (t, why) = triage(&b, "day", ACCEPT_SCORE);
        assert_eq!(t, Triage::Review);
        assert_eq!(why, ["found by SAM 3: always reviewed"]);
        // RF-DETR alone on a SAM 3 run is an ordinary one-labeler box.
        b.sources = [("rfdetr".to_string(), 0.95)].into();
        b.agree = false;
        assert_eq!(triage(&b, "day", ACCEPT_SCORE).1, ["found by rfdetr only"]);
        // Through to the box the page shows.
        b.sources = [("sam3".to_string(), 0.95)].into();
        let boxes = to_boxes(&[b], &classes(), "day", 640, 480, ACCEPT_SCORE);
        assert_eq!(boxes[0].auto.triage, Some(Triage::Review));
        assert!(
            boxes[0].auto.why.iter().any(|w| w.contains("SAM 3")),
            "{:?}",
            boxes[0].auto.why
        );
    }

    #[test]
    fn proposals_become_pending_auto_boxes_in_the_class_list() {
        let mut big = p("car", 0.81234, true);
        big.bbox = [-5.0, 10.04, 700.0, 90.0];
        let props = vec![big, p("unicorn", 0.9, true), {
            let mut tiny = p("dog", 0.9, true);
            tiny.bbox = [10.0, 10.0, 10.5, 50.0];
            tiny
        }];
        let b = to_boxes(&props, &classes(), "day", 640, 480, ACCEPT_SCORE);
        assert_eq!(b.len(), 1, "unknown label and tiny box left out");
        let b = &b[0];
        assert_eq!(b.bbox, [0.0, 10.0, 640.0, 90.0]);
        assert_eq!(b.source, Source::Auto);
        assert_eq!(b.state, BoxState::Pending);
        assert_eq!(b.score, Some(0.8123));
        assert_eq!(b.auto.triage, Some(Triage::Accept));
        assert_eq!(b.auto.agree, Some(true));
        assert_eq!(b.reason.as_deref(), Some("auto: gdino + rfdetr"));
        // Serialized on the box itself.
        let v = serde_json::to_value(b).unwrap();
        assert_eq!(v["triage"], "accept");
        assert_eq!(v["labelers"]["rfdetr"], serde_json::json!(0.9f32));
    }

    #[test]
    fn merge_keeps_decisions_and_annotates_confirmed_boxes() {
        let model = LabelBox {
            label: "person".into(),
            bbox: [10.0, 10.0, 60.0, 90.0],
            source: Source::Model,
            score: Some(0.55),
            state: BoxState::Pending,
            origin: Some(labels::Origin::Prediction),
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        };
        let mut f = LabelFile {
            boxes: vec![model],
            ..LabelFile::default()
        };
        let mut person = p("person", 0.7, true);
        person.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "statue".into(),
            expected: "person".into(),
            ok: false,
        });
        let mut dog = p("dog", 0.8, true);
        dog.bbox = [200.0, 200.0, 260.0, 260.0];
        let boxes = to_boxes(&[person, dog], &classes(), "day", 640, 480, ACCEPT_SCORE);
        let rep = apply(&mut f, boxes);
        assert_eq!(
            (rep.merge.added, rep.merge.dropped, rep.annotated),
            (1, 1, 1)
        );
        assert_eq!(rep.class_suggestions, 0);
        // The model's person box keeps its source and state, and shows the VLM's objection.
        assert_eq!(f.boxes[0].source, Source::Model);
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        assert_eq!(f.boxes[0].auto.vlm.as_ref().unwrap().answer, "statue");
        assert_eq!(f.boxes[1].label, "dog");
        assert_eq!(f.boxes[1].source, Source::Auto);
    }

    /// A pending model box (`origin`) at the proposals' place.
    fn model_box(label: &str, origin: labels::Origin, state: BoxState) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: [10.0, 10.0, 60.0, 90.0],
            source: Source::Model,
            score: Some(0.55),
            state,
            origin: Some(origin),
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        }
    }

    #[test]
    fn a_proposal_of_another_class_on_a_pending_box_becomes_a_class_suggestion() {
        let mut f = LabelFile {
            boxes: vec![model_box(
                "dog",
                labels::Origin::Prediction,
                BoxState::Pending,
            )],
            ..LabelFile::default()
        };
        let mut raccoon = p("raccoon", 0.9, true);
        raccoon.bbox = [11.0, 10.0, 60.0, 90.0];
        let boxes = to_boxes(
            &[raccoon.clone()],
            &classes(),
            "day",
            640,
            480,
            ACCEPT_SCORE,
        );
        assert_eq!(boxes[0].auto.triage, Some(Triage::Accept));
        let rep = apply(&mut f, boxes.clone());
        assert_eq!(
            (
                rep.class_suggestions,
                rep.annotated,
                rep.merge.added,
                rep.merge.dropped
            ),
            (1, 0, 0, 0)
        );
        assert_eq!(f.boxes.len(), 1, "no duplicate box");
        let b = &f.boxes[0];
        assert_eq!(
            (b.label.as_str(), b.source, b.state),
            ("dog", Source::Model, BoxState::Pending)
        );
        assert_eq!(b.auto.auto_label.as_deref(), Some("raccoon"));
        assert_eq!(b.auto.auto_score, Some(0.9));
        assert_eq!(b.auto.triage, Some(Triage::Review), "forced to review");
        assert_eq!(b.auto.agree, Some(true));
        assert_eq!(
            b.auto.why.last().map(String::as_str),
            Some("the auto-labelers say raccoon (0.90); the box says dog")
        );
        // On save the dog prediction is still accepted as a dog; the reviewer relabels it.
        assert_eq!(labels::resolve_pending(b), BoxState::Accepted);
        // A second run: the box has evidence now, the proposal is dropped as covered.
        let rep = apply(&mut f, boxes.clone());
        assert_eq!((rep.class_suggestions, rep.merge.dropped), (0, 1));
        assert_eq!(f.boxes.len(), 1);
        // A decided box of another label: a disagreement box is added, the box is untouched.
        let mut g = LabelFile {
            boxes: vec![model_box(
                "dog",
                labels::Origin::Prediction,
                BoxState::Accepted,
            )],
            ..LabelFile::default()
        };
        let rep = apply(&mut g, boxes.clone());
        assert_eq!((rep.class_suggestions, rep.merge.added), (0, 1));
        assert!(g.boxes[0].auto.is_empty());
        // A box of the proposal's own label also matches: that one is annotated instead.
        let mut h = LabelFile {
            boxes: vec![
                model_box("dog", labels::Origin::Prediction, BoxState::Pending),
                LabelBox {
                    bbox: [12.0, 10.0, 60.0, 92.0],
                    ..model_box("raccoon", labels::Origin::Suppressed, BoxState::Pending)
                },
            ],
            ..LabelFile::default()
        };
        let rep = apply(&mut h, boxes);
        assert_eq!((rep.class_suggestions, rep.annotated), (0, 1));
        assert!(h.boxes[0].auto.is_empty());
        assert_eq!(h.boxes[1].auto.triage, Some(Triage::Accept));
    }

    #[test]
    fn evidence_on_a_suppressed_box_of_the_same_class_accepts_it_on_save() {
        let mut f = LabelFile {
            boxes: vec![model_box(
                "car",
                labels::Origin::Suppressed,
                BoxState::Pending,
            )],
            ..LabelFile::default()
        };
        let boxes = to_boxes(
            &[p("car", 0.8, true)],
            &classes(),
            "day",
            640,
            480,
            ACCEPT_SCORE,
        );
        let rep = apply(&mut f, boxes);
        assert_eq!((rep.annotated, rep.merge.added), (1, 0));
        assert_eq!(f.boxes[0].auto.triage, Some(Triage::Accept));
        assert_eq!(labels::resolve_pending(&f.boxes[0]), BoxState::Accepted);
        assert!(labels::auto_accept_eligible(&f));
        // Triaged review (a single labeler): still rejected on save.
        let mut g = LabelFile {
            boxes: vec![model_box(
                "car",
                labels::Origin::BelowThreshold,
                BoxState::Pending,
            )],
            ..LabelFile::default()
        };
        apply(
            &mut g,
            to_boxes(
                &[p("car", 0.4, false)],
                &classes(),
                "day",
                640,
                480,
                ACCEPT_SCORE,
            ),
        );
        assert_eq!(g.boxes[0].auto.triage, Some(Triage::Review));
        assert_eq!(labels::resolve_pending(&g.boxes[0]), BoxState::Rejected);
    }

    fn reviewed() -> LabelFile {
        let b = |label: &str, bbox: [f32; 4], state: BoxState| LabelBox {
            label: label.into(),
            bbox,
            source: Source::Model,
            score: Some(0.7),
            state,
            origin: Some(labels::Origin::Prediction),
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        };
        LabelFile {
            status: Status::Reviewed,
            boxes: vec![
                b("person", [10.0, 10.0, 60.0, 90.0], BoxState::Accepted),
                b("dog", [200.0, 200.0, 260.0, 260.0], BoxState::Accepted),
                b("car", [400.0, 10.0, 500.0, 60.0], BoxState::Rejected),
            ],
            ..LabelFile::default()
        }
    }

    /// [`apply_relabel`] with [`model_boxes`] of a day frame, no pre-accept.
    fn relabel(
        f: &mut LabelFile,
        dets: &[Det],
        model: &str,
        classes: &[String],
    ) -> (MergeReport, Option<String>) {
        let run = ModelRun {
            model,
            classes,
            scene: "day",
            accept_score: None,
        };
        let boxes = model_boxes(dets, &run, 640, 480).boxes;
        let r = apply_relabel(f, dets, boxes, model, classes);
        (r.apply.merge, r.disagreement)
    }

    #[test]
    fn model_labels_are_mapped_onto_the_class_list() {
        let list: Vec<String> = [
            "person",
            "Motorcycle",
            "car",
            "tv",
            "potted plant",
            "raccoon",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(map_label("person", &list), Some("person"));
        assert_eq!(map_label("PERSON", &list), Some("person"));
        assert_eq!(map_label("motorbike", &list), Some("Motorcycle"));
        assert_eq!(map_label("motorcycle", &list), Some("Motorcycle"));
        assert_eq!(map_label("tvmonitor", &list), Some("tv"));
        assert_eq!(map_label("pottedplant", &list), Some("potted plant"));
        assert_eq!(map_label(" car ", &list), Some("car"));
        assert_eq!(
            map_label("aeroplane", &list),
            None,
            "airplane is not listed"
        );
        assert_eq!(map_label("dog", &list), None);
        // An exact spelling wins over a case-insensitive one.
        let two: Vec<String> = ["Car", "car"].map(String::from).to_vec();
        assert_eq!(map_label("car", &two), Some("car"));
        // Proposals: mapped, the model's spelling kept, others counted.
        let props = vec![
            p("motorbike", 0.8, true),
            p("Person", 0.7, true),
            p("aeroplane", 0.9, true),
            p("sofa", 0.9, true),
        ];
        let out = to_boxes_counted(&props, &list, "day", 640, 480, ACCEPT_SCORE);
        assert_eq!(out.other_class, 2, "aeroplane and sofa are not run classes");
        assert_eq!(out.boxes.len(), 2);
        let moto = &out.boxes[0];
        assert_eq!(moto.label, "Motorcycle");
        assert_eq!(moto.model_label.as_deref(), Some("motorbike"));
        assert_eq!(moto.auto.triage, Some(Triage::Accept));
        let person = &out.boxes[1];
        assert_eq!(person.label, "person");
        assert_eq!(
            person.model_label, None,
            "a case-only change is not a rename"
        );
        assert_eq!(
            out.renamed,
            [("motorbike".to_string(), "Motorcycle".to_string())].into()
        );
        let v = serde_json::to_value(moto).unwrap();
        assert_eq!(v["model_label"], "motorbike");
        // The night rule follows the mapped class (`people` is a person).
        let night = to_boxes(&[p("people", 0.9, true)], &list, "ir", 640, 480, 0.6);
        assert_eq!(night[0].label, "person");
        assert_eq!(night[0].auto.triage, Some(Triage::Review));
    }

    #[test]
    fn model_boxes_are_reviewed_unless_pre_accepted() {
        let cls: Vec<String> = ["person", "motorcycle", "car", "raccoon"]
            .map(String::from)
            .to_vec();
        let dets = [
            Det {
                label: "motorbike".into(),
                score: 0.87,
                bbox: [10.0, 10.0, 60.0, 90.0],
            },
            Det {
                label: "car".into(),
                score: 0.5,
                bbox: [200.0, 10.0, 300.0, 90.0],
            },
            Det {
                label: "person".into(),
                score: 0.95,
                bbox: [400.0, 10.0, 450.0, 90.0],
            },
            Det {
                label: "tvmonitor".into(),
                score: 0.95,
                bbox: [500.0, 10.0, 550.0, 90.0],
            },
        ];
        let mut run = ModelRun {
            model: "rfdetr-large",
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        let out = model_boxes(&dets, &run, 640, 480);
        assert_eq!(out.other_class, 1, "the tv is not a run class");
        assert_eq!(out.boxes.len(), 3);
        assert!(
            out.boxes
                .iter()
                .all(|b| b.auto.triage == Some(Triage::Review)
                    && b.auto.why == ["re-label by rfdetr-large"]
                    && b.auto.agree.is_none()
                    && b.auto.labelers.contains_key("rfdetr-large"))
        );
        assert_eq!(out.boxes[0].label, "motorcycle");
        assert_eq!(out.boxes[0].model_label.as_deref(), Some("motorbike"));
        // Pre-accept at 0.8: the motorcycle and the person, not the car.
        run.accept_score = Some(0.8);
        let out = model_boxes(&dets, &run, 640, 480);
        let t: Vec<Option<Triage>> = out.boxes.iter().map(|b| b.auto.triage).collect();
        assert_eq!(
            t,
            [
                Some(Triage::Accept),
                Some(Triage::Review),
                Some(Triage::Accept)
            ]
        );
        assert_eq!(
            out.boxes[0].auto.why,
            ["model rfdetr-large at 0.87 >= 0.80"]
        );
        assert_eq!(
            out.boxes[1].auto.why,
            ["re-label by rfdetr-large", "score 0.50 below 0.80"]
        );
        // At the score itself: accepted.
        assert_eq!(model_triage("car", 0.8, &run).0, Triage::Accept);
        // A frame of only pre-accepted boxes is one the bulk accept takes.
        let mut f = LabelFile::default();
        let only = model_boxes(&dets[..1], &run, 640, 480).boxes;
        apply(&mut f, only);
        assert!(labels::auto_accept_eligible(&f));
        // Night / IR: the person stays in review, the vehicles do not.
        run.scene = "ir";
        let out = model_boxes(&dets, &run, 640, 480);
        assert_eq!(out.boxes[0].auto.triage, Some(Triage::Accept));
        assert_eq!(out.boxes[2].auto.triage, Some(Triage::Review));
        assert_eq!(
            out.boxes[2].auto.why,
            [
                "re-label by rfdetr-large",
                "a person at night: always reviewed"
            ]
        );
        assert_eq!(
            model_triage("raccoon", 0.99, &run).1[1],
            "an animal at night: always reviewed"
        );
        // Without pre-accept the night rule adds nothing (every box is reviewed anyway).
        run.accept_score = None;
        assert_eq!(
            model_triage("person", 0.99, &run).1,
            ["re-label by rfdetr-large"]
        );
    }

    #[test]
    fn a_voc_spelled_model_agrees_with_reviewed_motorcycles() {
        let cls: Vec<String> = ["person", "motorcycle"].map(String::from).to_vec();
        let mut f = reviewed();
        f.boxes[0].label = "motorcycle".into();
        let dets = [d("motorbike", [11.0, 10.0, 60.0, 90.0])];
        assert_eq!(disagreement(&f, &dets, "m", &cls), None);
        // Before the mapping, the motorbike was dropped and the motorcycle missed.
        let (rep, why) = relabel(&mut f, &dets, "m", &cls);
        assert_eq!((rep.added, rep.dropped, why), (0, 1, None));
        assert!(
            f.boxes[0].auto.labelers.contains_key("m"),
            "the reviewed motorcycle gets the model's evidence"
        );
    }

    fn d(label: &str, bbox: [f32; 4]) -> Det {
        Det {
            label: label.into(),
            score: 0.8,
            bbox,
        }
    }

    #[test]
    fn self_training_flags_disagreements_with_the_reviewed_labels() {
        let f = reviewed();
        // The same boxes (the rejected car is not a training box): no disagreement.
        let same = [
            d("person", [11.0, 10.0, 60.0, 90.0]),
            d("dog", [200.0, 201.0, 260.0, 260.0]),
        ];
        assert_eq!(disagreement(&f, &same, "m", &classes()), None);
        // A class change, a miss and a new box.
        let dets = [
            d("cat", [200.0, 200.0, 261.0, 260.0]),
            d("raccoon", [300.0, 300.0, 340.0, 340.0]),
        ];
        assert_eq!(
            disagreement(&f, &dets, "night-yolo26n", &classes()).as_deref(),
            Some("re-label by night-yolo26n: 1 new box, 1 missed, 1 class change")
        );
        // A run for some classes only counts those (detections and reviewed boxes alike).
        let only = |c: &[&str]| -> Option<String> {
            let c: Vec<String> = c.iter().map(|s| (*s).to_string()).collect();
            disagreement(&f, &dets, "m", &c)
        };
        assert_eq!(
            only(&["Raccoon"]).as_deref(),
            Some("re-label by m: 1 new box")
        );
        assert_eq!(
            only(&["person"]).as_deref(),
            Some("re-label by m: 1 missed")
        );
        assert_eq!(
            only(&["dog"]).as_deref(),
            Some("re-label by m: 1 missed"),
            "the cat is not looked at; the reviewed dog is missed"
        );
        assert_eq!(
            only(&["cat"]).as_deref(),
            Some("re-label by m: 1 class change"),
            "a cat on the reviewed dog"
        );
        assert_eq!(only(&["truck"]), None);
        assert_eq!(disagreement(&f, &same, "m", &["person".to_string()]), None);
        // Unreviewed files have nothing to disagree with.
        let mut un = reviewed();
        un.status = Status::Unreviewed;
        assert_eq!(disagreement(&un, &dets, "m", &classes()), None);

        let mut g = reviewed();
        let (rep, why) = relabel(&mut g, &dets, "night-yolo26n", &classes());
        assert!(why.is_some());
        assert_eq!(g.status, Status::Unreviewed);
        assert_eq!(g.disagreement, why);
        assert_eq!(
            rep.added, 2,
            "the cat on the decided dog and the new raccoon"
        );
        assert!(
            g.boxes[3..]
                .iter()
                .all(|b| b.auto.triage == Some(Triage::Review)
                    && b.auto.labelers.contains_key("night-yolo26n"))
        );
        // Decisions are kept.
        assert_eq!(g.boxes[2].state, BoxState::Rejected);
        // Agreement: nothing changes but the evidence.
        let mut h = reviewed();
        let (rep, why) = relabel(&mut h, &same, "m", &classes());
        assert_eq!((rep.added, why), (0, None));
        assert_eq!(h.status, Status::Reviewed);
    }

    /// `boxes` as a reviewed file.
    fn frame(boxes: Vec<LabelBox>) -> LabelFile {
        LabelFile {
            status: Status::Reviewed,
            boxes,
            ..LabelFile::default()
        }
    }

    /// A box of `source`, `state` at `bbox`.
    fn lb(label: &str, bbox: [f32; 4], source: Source, state: BoxState) -> LabelBox {
        LabelBox {
            bbox,
            source,
            state,
            origin: (source == Source::Model).then_some(labels::Origin::Prediction),
            ..labels::tests_support::auto_box(label)
        }
    }

    fn accepted(label: &str, bbox: [f32; 4]) -> LabelBox {
        lb(label, bbox, Source::Model, BoxState::Accepted)
    }

    /// [`apply_missing`] of `dets` by `model` on a 640x480 day frame.
    fn missing(f: &mut LabelFile, dets: &[Det], model: &str) -> MissingReport {
        let cls = classes();
        let run = ModelRun {
            model,
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        apply_missing(f, model_boxes(dets, &run, 640, 480).boxes, model)
    }

    const CAR: [f32; 4] = [100.0, 100.0, 200.0, 200.0];

    #[test]
    fn a_new_object_reopens_the_frame_with_a_pending_review_box() {
        let mut f = frame(vec![accepted("person", [400.0, 10.0, 450.0, 90.0])]);
        f.overlaps_checked = true;
        f.reviewed_at = Some("2026-10-22T12:00:00.000Z".into());
        let r = missing(&mut f, &[d("car", CAR)], "nano-2");
        assert_eq!(r.added, 1);
        assert_eq!(r.candidates(), 1);
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by nano-2: 1 new box")
        );
        assert!(!f.overlaps_checked, "the look did not cover this box");
        let b = &f.boxes[1];
        assert_eq!((b.source, b.state), (Source::Auto, BoxState::Pending));
        assert_eq!(b.auto.triage, Some(Triage::Review));
        assert!(b.auto.labelers.contains_key("nano-2"));
        assert_eq!(f.boxes[0].state, BoxState::Accepted, "decisions are kept");
        assert!(f.boxes[0].auto.is_empty(), "no evidence on existing boxes");
    }

    #[test]
    fn a_frame_that_gains_nothing_is_unchanged() {
        let mut f = frame(vec![accepted("car", CAR)]);
        f.overlaps_checked = true;
        let before = f.clone();
        let r = missing(&mut f, &[d("car", [101.0, 100.0, 200.0, 200.0])], "m");
        assert_eq!((r.candidates(), r.covered), (0, 1));
        assert_eq!(f, before);
        assert_eq!(missing(&mut f, &[], "m"), MissingReport::default());
        assert_eq!(f, before);
    }

    #[test]
    fn a_training_box_covers_a_detection_from_iou_0_3_or_half_inside_it() {
        let wide = [0.0, 0.0, 100.0, 100.0];
        // Exactly IoU 0.3 (6000 / 20000), a third of the detection inside: covered.
        let mut f = frame(vec![accepted("car", wide)]);
        assert_eq!(
            missing(&mut f, &[d("car", [40.0, 0.0, 200.0, 100.0])], "m").covered,
            1
        );
        // IoU 0.25: a new object.
        assert_eq!(
            missing(&mut f, &[d("car", [50.0, 0.0, 200.0, 100.0])], "m").added,
            1
        );
        let big = [0.0, 0.0, 300.0, 300.0];
        // Half of a small detection inside: covered, at an IoU of 0.02.
        let mut f = frame(vec![accepted("car", big)]);
        let half = [270.0, 0.0, 330.0, 60.0];
        assert_eq!(missing(&mut f, &[d("car", half)], "m").covered, 1);
        // A third of it inside: kept.
        let third = [280.0, 0.0, 340.0, 60.0];
        assert_eq!(missing(&mut f, &[d("car", third)], "m").added, 1);
        // The other way round: a small box inside a large detection does not cover it, though
        // `inside_share` of the two is 1.
        let mut f = frame(vec![accepted("car", [100.0, 100.0, 130.0, 130.0])]);
        let large = [50.0, 50.0, 300.0, 300.0];
        assert_eq!(inside_share(&large, &f.boxes[0].bbox), 1.0);
        assert_eq!(missing(&mut f, &[d("car", large)], "m").added, 1);
        // A rejected box does not cover at IoU 0.3.
        let mut f = frame(vec![lb("car", wide, Source::Model, BoxState::Rejected)]);
        assert_eq!(
            missing(&mut f, &[d("car", [40.0, 0.0, 200.0, 100.0])], "m").added,
            1
        );
    }

    #[test]
    fn a_car_on_a_labelled_truck_neither_adds_nor_reopens() {
        let mut f = frame(vec![accepted("truck", CAR)]);
        let before = f.clone();
        let r = missing(&mut f, &[d("car", CAR)], "m");
        assert_eq!((r.candidates(), r.covered), (0, 1));
        assert_eq!(f, before);
        // Another object class on the same place is a new object.
        let r = missing(&mut f, &[d("person", CAR)], "m");
        assert_eq!(r.added, 1);
        assert_eq!(f.status, Status::Unreviewed);
    }

    #[test]
    fn pending_boxes_and_this_runs_boxes_cover_highest_score_first() {
        let mut f = frame(vec![]);
        let low = Det {
            score: 0.6,
            ..d("car", [102.0, 100.0, 200.0, 200.0])
        };
        let high = Det {
            score: 0.9,
            ..d("truck", CAR)
        };
        let r = missing(&mut f, &[low, high], "m");
        assert_eq!((r.added, r.covered), (1, 1));
        assert_eq!(f.boxes.len(), 1);
        assert_eq!(f.boxes[0].label, "truck", "the sure box stays");
        // Pending hint of the sidecar on an unreviewed frame sent back by a re-label.
        let mut g = LabelFile {
            disagreement: Some("x".into()),
            boxes: vec![lb("car", CAR, Source::Model, BoxState::Pending)],
            ..LabelFile::default()
        };
        assert_eq!(missing(&mut g, &[d("car", CAR)], "m").covered, 1);
        assert_eq!(g.disagreement.as_deref(), Some("x"), "not rebuilt");
    }

    #[test]
    fn a_judged_reject_blocks_the_detection() {
        let hint = |origin| LabelBox {
            origin: Some(origin),
            ..lb("car", CAR, Source::Model, BoxState::Rejected)
        };
        let cases = [
            // By hand: the reviewer rejected a prediction or a proposal.
            lb("car", CAR, Source::Model, BoxState::Rejected),
            LabelBox {
                reason: Some("model m".into()),
                ..lb("car", CAR, Source::Auto, BoxState::Rejected)
            },
            // With a reject reason, on a hint too: somebody chose it.
            LabelBox {
                reject_reason: Some(labels::RejectReason::IrGlare),
                ..hint(labels::Origin::BelowThreshold)
            },
            // A spot edit.
            LabelBox {
                reason: Some(labels::repeats::REPEAT_SPOT_REASON.into()),
                ..hint(labels::Origin::BelowThreshold)
            },
            // A duplicate reject, on a hint and on a box rejected low before.
            LabelBox {
                reject_reason: Some(labels::RejectReason::Duplicate),
                reason: Some(DUPLICATE_REASON.into()),
                ..hint(labels::Origin::Suppressed)
            },
            LabelBox {
                reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
                reject_reason: Some(labels::RejectReason::FalsePositive),
                ..lb("car", CAR, Source::Auto, BoxState::Rejected)
            },
        ];
        for c in cases {
            let mut f = frame(vec![c.clone()]);
            let before = f.clone();
            let r = missing(&mut f, &[d("truck", CAR)], "m");
            assert_eq!((r.candidates(), r.judged), (0, 1), "{c:?}");
            assert_eq!(f, before, "{c:?}");
            // Below IoU 0.5 it says nothing about the detection.
            let r = missing(&mut f, &[d("car", [150.0, 100.0, 250.0, 200.0])], "m");
            assert_eq!(r.added, 1, "{c:?}");
        }
    }

    #[test]
    fn an_unjudged_reject_is_overwritten_in_place() {
        let low = LabelBox {
            reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
            ..lb("truck", CAR, Source::Auto, BoxState::Rejected)
        };
        let hint = LabelBox {
            origin: Some(labels::Origin::BelowThreshold),
            ..lb(
                "car",
                [101.0, 100.0, 200.0, 200.0],
                Source::Model,
                BoxState::Rejected,
            )
        };
        let suppressed = LabelBox {
            origin: Some(labels::Origin::Suppressed),
            reason: Some("night_threshold".into()),
            ..hint.clone()
        };
        let other = accepted("person", [400.0, 10.0, 450.0, 90.0]);
        for reject in [low, hint.clone(), suppressed] {
            let mut f = frame(vec![reject, other.clone()]);
            let r = missing(&mut f, &[d("car", CAR)], "m");
            assert_eq!((r.added, r.overwritten, r.candidates()), (0, 1, 1));
            assert_eq!(f.boxes.len(), 2, "no second box");
            let b = &f.boxes[0];
            assert_eq!(
                (b.label.as_str(), b.source, b.state, b.bbox),
                ("car", Source::Auto, BoxState::Pending, CAR)
            );
            assert_eq!(b.origin, None);
            assert_eq!(b.reason.as_deref(), Some("model m"));
            assert_eq!(f.hard_negatives().count(), 0, "no contradiction");
            assert_eq!(f.status, Status::Unreviewed);
            assert_eq!(f.disagreement.as_deref(), Some("re-label by m: 1 new box"));
        }
        // Two automatic rejects on the object: the closest is taken, the other goes.
        let far = LabelBox {
            bbox: [105.0, 100.0, 200.0, 200.0],
            ..hint.clone()
        };
        let mut f = frame(vec![far, other.clone(), hint.clone()]);
        let r = missing(&mut f, &[d("car", CAR)], "m");
        assert_eq!(r.overwritten, 1);
        assert_eq!(f.boxes.len(), 2);
        assert_eq!(f.boxes[0].label, "person");
        assert_eq!(f.boxes[1].state, BoxState::Pending);
        assert_eq!(f.hard_negatives().count(), 0);
        // A judged reject beside an automatic one blocks: nothing changes.
        let judged = lb("car", CAR, Source::Model, BoxState::Rejected);
        let mut f = frame(vec![hint.clone(), judged]);
        let before = f.clone();
        assert_eq!(missing(&mut f, &[d("car", CAR)], "m").judged, 1);
        assert_eq!(f, before);
        // Of another object: not touched.
        let mut f = frame(vec![LabelBox {
            label: "person".into(),
            ..hint
        }]);
        assert_eq!(missing(&mut f, &[d("car", CAR)], "m").added, 1);
        assert_eq!(f.boxes.len(), 2);
        assert_eq!(f.boxes[0].state, BoxState::Rejected);
    }

    #[test]
    fn only_reviewed_frames_and_re_label_returns_are_in_scope() {
        assert!(missing_scope(&frame(vec![])));
        let back = LabelFile {
            disagreement: Some("x".into()),
            ..LabelFile::default()
        };
        assert!(missing_scope(&back));
        for out in [
            LabelFile::default(),
            LabelFile {
                status: Status::Skipped,
                disagreement: Some("x".into()),
                ..LabelFile::default()
            },
            LabelFile {
                empty_confirmed: true,
                ..frame(vec![])
            },
            LabelFile {
                empty_confirmed: true,
                ..back
            },
        ] {
            assert!(!missing_scope(&out), "{out:?}");
            let mut f = out.clone();
            assert_eq!(
                missing(&mut f, &[d("car", CAR)], "m"),
                MissingReport::default()
            );
            assert_eq!(f, out);
        }
    }

    #[test]
    fn two_models_in_turn_add_nothing_twice_and_share_the_note() {
        let mut f = frame(vec![]);
        assert_eq!(missing(&mut f, &[d("car", CAR)], "b-model").added, 1);
        let snapshot = f.clone();
        // The same object from the second model: nothing, the note stays.
        let r = missing(&mut f, &[d("car", [102.0, 100.0, 200.0, 200.0])], "a-model");
        assert_eq!((r.candidates(), r.covered), (0, 1));
        assert_eq!(f, snapshot);
        // A rerun of the first: nothing.
        assert_eq!(missing(&mut f, &[d("car", CAR)], "b-model").candidates(), 0);
        // A second object: the note names both, the count is the pending boxes'.
        let r = missing(&mut f, &[d("dog", [300.0, 300.0, 360.0, 360.0])], "a-model");
        assert_eq!(r.added, 1);
        assert_eq!(f.boxes.len(), 2);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by a-model, b-model: 2 new boxes")
        );
    }

    #[test]
    fn a_reopened_frame_saves_and_its_note_stays_under_the_limit() {
        let now = "2026-10-22T12:00:00.000Z";
        let cls = classes();
        // 30 models of the longest name a save accepts, each with its own box.
        let mut f = frame(vec![]);
        for i in 0..30u16 {
            let model = format!("{i:02}{}", "m".repeat(labels::MAX_LABEL_LEN - 2));
            let x = f32::from(i) * 20.0;
            let r = missing(&mut f, &[d("car", [x, 0.0, x + 10.0, 10.0])], &model);
            assert_eq!(r.added, 1);
        }
        let note = f.disagreement.clone().unwrap();
        assert!(
            note.chars().count() < labels::MAX_DISAGREEMENT_LEN,
            "{note}"
        );
        assert!(note.ends_with(" more: 30 new boxes"), "{note}");
        assert!(note.contains("00mmm"), "{note}");
        // Unreviewed save: the note is kept, and accepted.
        let saved = labels::normalize(f.clone(), &cls, 640, 480, now).unwrap();
        assert_eq!(saved.disagreement.as_deref(), Some(note.as_str()));
        // Saved as reviewed once the reviewer accepted the proposals: the note goes.
        let mut done = saved;
        done.status = Status::Reviewed;
        done.boxes
            .iter_mut()
            .for_each(|b| b.state = BoxState::Accepted);
        let done = labels::normalize(done, &cls, 640, 480, now).unwrap();
        assert_eq!(done.training().count(), 30);
        assert_eq!(done.disagreement, None);
        // One name longer than the note: the count stands in for it.
        let mut g = frame(vec![]);
        missing(&mut g, &[d("car", CAR)], &"x".repeat(900));
        let note = g.disagreement.unwrap();
        assert_eq!(note, "re-label by 1 model(s): 1 new box");
    }

    #[test]
    fn accept_this_spot_takes_a_reopened_frames_proposal() {
        use labels::repeats::{ACCEPT_SPOT_IOU, Spot, Verdict};
        let mut f = frame(vec![accepted("person", [400.0, 10.0, 450.0, 90.0])]);
        missing(&mut f, &[d("car", CAR)], "m");
        let spot = Spot {
            camera: "cam".into(),
            label: "car".into(),
            bbox: CAR,
            width: 640,
            height: 480,
            min_iou: ACCEPT_SPOT_IOU,
        };
        assert!(spot.matches(&f.boxes[1], &Verdict::Accept));
        assert_eq!(spot.apply(&mut f, &Verdict::Accept), 1);
        assert_eq!(f.boxes[1].state, BoxState::Accepted);
        // So is an overwritten reject's.
        let mut g = frame(vec![LabelBox {
            reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
            ..lb("car", CAR, Source::Auto, BoxState::Rejected)
        }]);
        missing(&mut g, &[d("car", CAR)], "m");
        assert!(spot.matches(&g.boxes[0], &Verdict::Accept));

        // Accepted with no save, so it can still be restored: the check does not come back.
        let mut h = frame(vec![accepted("person", [400.0, 10.0, 450.0, 90.0])]);
        h.overlaps_checked = true;
        h.reviewed_at = Some("t".into());
        missing(&mut h, &[d("car", CAR)], "m");
        assert!(h.checked_before_reopen);
        assert_eq!(spot.apply(&mut h, &Verdict::Accept), 1);
        assert_eq!(labels::bulk::restore_reopened(&mut h), 1);
        assert!(!h.overlaps_checked && !h.checked_before_reopen);
    }

    /// Two accepted cars a reviewer kept or not (`checked`), reviewed, the second inside the first.
    fn stacked_cars(checked: bool) -> LabelFile {
        LabelFile {
            overlaps_checked: checked,
            reviewed_at: Some("t".into()),
            ..frame(vec![
                accepted("car", CAR),
                accepted("car", [100.0, 100.0, 200.0, 196.0]),
            ])
        }
    }

    #[test]
    fn a_checked_frame_reopened_then_restored_is_checked_again() {
        let cl = classes();
        let w = labels::bulk::Wanted {
            classes: &cl,
            hidden: &[],
        };
        let reject_new = |f: &mut LabelFile| {
            for b in &mut f.boxes {
                if b.state == BoxState::Pending {
                    b.state = BoxState::Rejected;
                }
            }
        };
        // Through the labelers' gate and through `--missing-only`.
        let mut gated = stacked_cars(true);
        apply_labelers(&mut gated, boxes(&[sam("dog", OFF, 0.9)]), "sam3", REL);
        let mut direct = stacked_cars(true);
        missing(&mut direct, &[d("dog", OFF)], "m");
        for mut f in [gated, direct] {
            assert_eq!(f.status, Status::Unreviewed);
            assert!(!f.overlaps_checked && f.checked_before_reopen);
            // A second rerun on the sent-back frame must not forget it.
            let free = [10.0, 300.0, 60.0, 360.0];
            apply_labelers(&mut f, boxes(&[sam("cat", free, 0.9)]), "sam3", REL);
            assert!(f.checked_before_reopen);
            assert_eq!(labels::bulk::restore_reopened(&mut f), 0, "still pending");
            reject_new(&mut f);
            assert_eq!(labels::bulk::restore_reopened(&mut f), 1);
            assert_eq!(f.status, Status::Reviewed);
            assert!(f.overlaps_checked && !f.checked_before_reopen);
            assert_eq!(labels::stacks::reject_stacked(&mut f, &w), 0);
            assert_eq!(f.boxes[1].state, BoxState::Accepted);
        }

        // Nobody checked this one: it stays in reach.
        let mut f = stacked_cars(false);
        apply_labelers(&mut f, boxes(&[sam("dog", OFF, 0.9)]), "sam3", REL);
        assert!(!f.checked_before_reopen);
        reject_new(&mut f);
        assert_eq!(labels::bulk::restore_reopened(&mut f), 1);
        assert!(!f.overlaps_checked);
        assert_eq!(labels::stacks::reject_stacked(&mut f, &w), 1);

        // A frame that gains nothing keeps its check where it was.
        let mut f = stacked_cars(true);
        let before = f.clone();
        apply_labelers(&mut f, boxes(&[unasked("dog", OFF, 0.9)]), "sam3", REL);
        assert_eq!(f, before);
    }

    #[test]
    fn a_full_file_gains_no_box() {
        let mut f = frame(vec![
            accepted("car", [0.0, 0.0, 10.0, 10.0]);
            labels::MAX_BOXES
        ]);
        let before = f.clone();
        let r = missing(&mut f, &[d("dog", CAR)], "m");
        assert_eq!((r.candidates(), r.full), (0, 1));
        assert_eq!(f, before);
    }

    const OFF: [f32; 4] = [300.0, 300.0, 360.0, 360.0];

    /// A proposal one labeler (`sam3`) found at `source`, which the VLM said yes to; its fused score
    /// is halved, as `wbf` does.
    fn sam(label: &str, bbox: [f32; 4], source: f32) -> Proposal {
        Proposal {
            vlm: Some(VlmOut {
                model: "qwen3-vl-2b".into(),
                answer: "yes".into(),
                expected: "yes".into(),
                ok: true,
            }),
            ..unasked(label, bbox, source)
        }
    }

    /// [`sam`] with no VLM evidence.
    fn unasked(label: &str, bbox: [f32; 4], source: f32) -> Proposal {
        Proposal {
            bbox,
            sources: [("sam3".to_string(), source)].into(),
            ..p(label, source / 2.0, false)
        }
    }

    /// The boxes of `props` on a 640x480 day frame.
    fn boxes(props: &[Proposal]) -> Vec<LabelBox> {
        to_boxes(props, &classes(), "day", 640, 480, ACCEPT_SCORE)
    }

    fn json(b: &LabelBox) -> String {
        serde_json::to_string(b).unwrap()
    }

    /// `b` as it was before a vote.
    fn without_votes(b: &LabelBox) -> LabelBox {
        let mut b = b.clone();
        b.auto.labelers.clear();
        b
    }

    #[test]
    fn decided_boxes_are_byte_identical_after_a_reviewed_frame_run() {
        let human = lb(
            "person",
            [400.0, 10.0, 450.0, 90.0],
            Source::Human,
            BoxState::Added,
        );
        let judged = lb(
            "dog",
            [10.0, 10.0, 60.0, 60.0],
            Source::Model,
            BoxState::Rejected,
        );
        let low = LabelBox {
            reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
            ..lb("car", CAR, Source::Auto, BoxState::Rejected)
        };
        let mut seen = accepted("truck", [500.0, 300.0, 600.0, 400.0]);
        seen.score = Some(0.42);
        seen.auto.labelers.insert("rfdetr".into(), 0.33);
        let before = frame(vec![human, judged, low, seen]);
        let mut f = before.clone();
        let props = [
            sam("dog", [11.0, 10.0, 60.0, 60.0], 0.9), // on the judged reject
            sam("car", CAR, 0.9),                      // on the automatic reject
            sam("person", [401.0, 10.0, 450.0, 90.0], 0.9), // on the drawn box
            sam("truck", [500.0, 300.0, 600.0, 400.0], 0.9),
            sam("dog", OFF, 0.2),
        ];
        let r = apply_labelers(&mut f, boxes(&props), "sam3", REL);
        assert_eq!(r.missing.candidates(), 0, "{r:?}");
        assert_eq!(f.status, Status::Reviewed);
        assert_eq!(f.boxes.len(), before.boxes.len());
        for (new, old) in f.boxes.iter().zip(&before.boxes) {
            assert_eq!(json(&without_votes(new)), json(&without_votes(old)));
        }
        // Past the votes nothing moved either: the stored score, and the box no vote reached.
        assert_eq!(f.boxes[3].auto.labelers["rfdetr"], 0.33);
        assert_eq!(f.boxes[3].auto.labelers["sam3"], 0.9);
        let mut g = before.clone();
        apply_labelers(&mut g, boxes(&[sam("dog", OFF, 0.2)]), "sam3", REL);
        assert_eq!(g, before, "no vote, no box: the frame is as it was");
    }

    #[test]
    fn a_dog_on_an_accepted_cat_is_dropped_and_counted() {
        let cat = [100.0, 100.0, 200.0, 200.0];
        for state in [BoxState::Accepted, BoxState::Rejected] {
            let mut f = frame(vec![lb("cat", cat, Source::Model, state)]);
            let before = f.clone();
            let r = apply_labelers(&mut f, boxes(&[sam("dog", CAR, 0.9)]), "sam3", REL);
            assert_eq!(r.gate.other_object, 1, "{state:?}");
            assert_eq!(r.missing.candidates(), 0);
            assert_eq!(f, before, "no vote: another object");
        }
        let mut f = frame(vec![lb("cat", cat, Source::Human, BoxState::Added)]);
        let r = apply_labelers(&mut f, boxes(&[sam("dog", cat, 0.9)]), "sam3", REL);
        assert_eq!(r.gate.other_object, 1);
        // Below IoU 0.5 it is a new object, an accepted cat or not.
        let mut f = frame(vec![accepted("cat", cat)]);
        let beside = [160.0, 100.0, 260.0, 200.0];
        assert!(iou(&cat, &beside) < MATCH_IOU);
        let r = apply_labelers(&mut f, boxes(&[sam("dog", beside, 0.9)]), "sam3", REL);
        assert_eq!((r.gate.other_object, r.missing.added), (0, 1));
        // A pending box is no decision.
        let mut f = frame(vec![lb("cat", cat, Source::Auto, BoxState::Pending)]);
        let r = apply_labelers(&mut f, boxes(&[sam("dog", cat, 0.9)]), "sam3", REL);
        assert_eq!((r.gate.other_object, r.missing.added), (0, 1));
    }

    #[test]
    fn a_skipped_frame_is_not_reopened_and_cannot_reach_a_bulk_accept() {
        let mut f = LabelFile {
            status: Status::Skipped,
            reviewed_at: Some("2026-10-22T12:00:00.000Z".into()),
            boxes: vec![accepted("person", [400.0, 10.0, 450.0, 90.0])],
            ..LabelFile::default()
        };
        let before = f.clone();
        let mut sure = p("car", 0.9, true);
        sure.bbox = CAR;
        let r = apply_labelers(&mut f, boxes(&[sure]), "gdino", REL);
        assert_eq!(r.closed, 1);
        assert!(!r.for_review());
        assert_eq!(f, before);
        assert!(!labels::auto_accept_eligible(&f));
        // Votes still land: the frame's boxes were found again.
        let again = sam("person", [401.0, 10.0, 450.0, 90.0], 0.8);
        let r = apply_labelers(&mut f, boxes(&[again]), "sam3", REL);
        assert_eq!((r.votes, f.status), (1, Status::Skipped));
        assert_eq!(f.boxes.len(), 1);
    }

    #[test]
    fn a_low_score_does_not_reopen() {
        let mut f = frame(vec![]);
        let before = f.clone();
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.49)]), "sam3", REL);
        assert_eq!((r.gate.low_score, r.missing.candidates()), (1, 0));
        assert_eq!(f, before);
        // The source's score decides: fused 0.25 reopens at a source of exactly 0.5.
        let at = boxes(&[sam("car", CAR, MISSING_THRESHOLD)]);
        assert_eq!(at[0].score, Some(0.25));
        assert_eq!(apply_labelers(&mut f, at, "sam3", REL).missing.added, 1);
        assert_eq!(f.status, Status::Unreviewed);
    }

    #[test]
    fn a_vlm_no_reads_by_the_question_that_was_asked() {
        let mut p = sam("person", CAR, 0.9);
        p.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "no".into(),
            expected: "yes".into(),
            ok: false,
        });
        let (t, why) = triage(&p, "day", ACCEPT_SCORE);
        assert_eq!(t, Triage::Review);
        assert!(
            why.iter().any(|w| w == "qwen3-vl-2b sees no person"),
            "{why:?}"
        );
        // A label file of an older run keeps its seven-way wording.
        p.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "plant".into(),
            expected: "person".into(),
            ok: false,
        });
        let (_, why) = triage(&p, "day", ACCEPT_SCORE);
        assert!(
            why.iter()
                .any(|w| w == "qwen3-vl-2b says plant (expected person)"),
            "{why:?}"
        );
    }

    #[test]
    fn a_vlm_or_siglip_objection_does_not_reopen() {
        let mut vlm = sam("person", CAR, 0.9);
        vlm.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "rock".into(),
            expected: "person".into(),
            ok: false,
        });
        let mut agreed = p("cat", 0.9, true);
        agreed.bbox = CAR;
        agreed.species_scores = [("cat".to_string(), 0.2), ("raccoon".to_string(), 0.7)].into();
        for (o, name) in [(vlm.clone(), "vlm"), (agreed.clone(), "siglip")] {
            let mut f = frame(vec![]);
            let before = f.clone();
            let r = apply_labelers(&mut f, boxes(&[o]), "sam3", REL);
            assert_eq!((r.gate.objected, r.missing.candidates()), (1, 0), "{name}");
            assert_eq!(f, before, "{name}");
        }
        // A check that agrees is no objection.
        vlm.vlm.as_mut().unwrap().ok = true;
        agreed.species_scores = [("cat".to_string(), 0.7), ("raccoon".to_string(), 0.2)].into();
        for o in [vlm, agreed] {
            let mut f = frame(vec![]);
            assert_eq!(
                apply_labelers(&mut f, boxes(&[o]), "sam3", REL)
                    .missing
                    .added,
                1
            );
        }
    }

    #[test]
    fn a_lone_box_needs_a_vlm_yes_to_reopen_and_agreement_does_not() {
        // No VLM evidence, whatever the score and whoever found it.
        let mut gdino = p("car", 0.45, false);
        gdino.bbox = CAR;
        for lone in [unasked("car", CAR, 0.9), gdino.clone()] {
            let mut f = frame(vec![]);
            let before = f.clone();
            let r = apply_labelers(&mut f, boxes(&[lone]), "sam3", REL);
            assert_eq!((r.gate.unchecked, r.missing.candidates()), (1, 0));
            assert_eq!(r.gate.total(), 1);
            assert!(!r.for_review());
            assert_eq!(f, before);
        }

        // Skipped by `--vlm-min-score`: the script's own line, with no `vlm`.
        let line = r#"{"frame":"a.jpg","boxes":[{"label":"car","bbox":[100,100,200,200],
            "score":0.45,"sources":{"sam3":0.9},"agree":false,"vlm_skipped":"score"}]}"#;
        let skipped: FrameProposals = serde_json::from_str(line).unwrap();
        let mut f = frame(vec![]);
        let before = f.clone();
        let r = apply_labelers(&mut f, boxes(&skipped.boxes), "sam3", REL);
        assert_eq!((r.gate.unchecked, r.missing.candidates()), (1, 0));
        assert_eq!(f, before);

        // The VLM's yes opens it, for either labeler.
        let mut f = frame(vec![]);
        assert_eq!(
            apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL)
                .missing
                .added,
            1
        );
        assert_eq!(f.status, Status::Unreviewed);
        gdino.vlm = sam("car", CAR, 0.9).vlm;
        let mut f = frame(vec![]);
        let r = apply_labelers(&mut f, boxes(&[gdino]), "gdino", REL);
        assert_eq!(r.missing.added, 1);

        // A VLM yes does not lift the score rule.
        let mut f = frame(vec![]);
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.4)]), "sam3", REL);
        assert_eq!((r.gate.low_score, r.gate.unchecked), (1, 0));

        // Labelers that agree need no VLM, unless a check objected.
        let mut agreed = p("car", 0.7, true);
        agreed.bbox = CAR;
        let mut f = frame(vec![]);
        let r = apply_labelers(&mut f, boxes(std::slice::from_ref(&agreed)), "sam3", REL);
        assert_eq!((r.gate.unchecked, r.missing.added), (0, 1));
        agreed.vlm = Some(VlmOut {
            model: "qwen3-vl-2b".into(),
            answer: "rock".into(),
            expected: "vehicle".into(),
            ok: false,
        });
        let mut f = frame(vec![]);
        let r = apply_labelers(&mut f, boxes(&[agreed]), "sam3", REL);
        assert_eq!((r.gate.objected, r.missing.added), (1, 0));

        // Votes do not ask: the unchecked box still counts for the box it found again.
        let seen = [400.0, 10.0, 450.0, 90.0];
        let mut f = frame(vec![accepted("person", seen)]);
        let r = apply_labelers(&mut f, boxes(&[unasked("person", seen, 0.8)]), "sam3", REL);
        assert_eq!((r.votes, r.gate.unchecked), (1, 1));
        assert_eq!(f.status, Status::Reviewed);
        assert_eq!(f.boxes[0].auto.labelers["sam3"], 0.8);
    }

    #[test]
    fn a_frame_confirmed_empty_on_any_status_takes_only_votes() {
        let rejected = lb("car", CAR, Source::Model, BoxState::Rejected);
        let mut f = LabelFile {
            empty_confirmed: true,
            boxes: vec![rejected],
            ..LabelFile::default()
        };
        assert_eq!(
            (f.status, f.disagreement.is_some()),
            (Status::Unreviewed, false)
        );
        let before = f.clone();
        let mut sure = p("dog", 0.9, true);
        sure.bbox = OFF;
        let r = apply_labelers(&mut f, boxes(&[sure, sam("person", OFF, 0.9)]), "sam3", REL);
        assert_eq!((r.closed, r.votes), (2, 0));
        assert!(!r.for_review());
        assert_eq!(f, before);
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!((r.closed, r.votes), (1, 1));
        assert_eq!(f.boxes.len(), 1);
        // So the later save still goes through.
        f.status = Status::Reviewed;
        assert!(labels::normalize(f, &classes(), 640, 480, "t").is_ok());
    }

    /// A closed frame ([`LabelersPath::Closed`]) takes a re-label's votes and nothing else, as it
    /// does a labelers run's.
    #[test]
    fn a_relabel_leaves_a_closed_frame_as_it_is_but_for_votes() {
        let cls = classes();
        let run = ModelRun {
            model: "custom",
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        let det = |label: &str, bbox| Det {
            label: label.into(),
            score: 0.9,
            bbox,
        };
        for (status, empty_confirmed) in [
            (Status::Reviewed, true),
            (Status::Unreviewed, true),
            (Status::Skipped, false),
        ] {
            let mut rejected = lb("car", CAR, Source::Model, BoxState::Rejected);
            rejected.auto.labelers.insert("gdino".into(), 0.4);
            let mut f = LabelFile {
                status,
                empty_confirmed,
                boxes: vec![rejected],
                ..LabelFile::default()
            };
            let before = f.clone();
            let dets = [det("dog", OFF)];
            let out = model_boxes(&dets, &run, 640, 480);
            let r = apply_relabel(&mut f, &dets, out.boxes, "custom", &cls);
            assert_eq!(f, before, "{status:?}: nothing but votes changes the file");
            assert_eq!(r.disagreement, None, "{status:?}");
            assert_eq!(r.apply, ApplyReport::default(), "{status:?}");
            let dets = [det("car", CAR)];
            let out = model_boxes(&dets, &run, 640, 480);
            let r = apply_relabel(&mut f, &dets, out.boxes, "custom", &cls);
            assert_eq!(r.votes, 1, "{status:?}");
            assert_eq!(f.boxes.len(), 1, "{status:?}");
            assert_eq!(f.boxes[0].state, BoxState::Rejected, "{status:?}");
            assert_eq!(f.status, status, "{status:?}");
            assert_eq!(f.disagreement, None, "{status:?}");
            assert_eq!(f.boxes[0].auto.labelers["custom"], 0.9, "{status:?}");
        }
    }

    #[test]
    fn a_frame_confirmed_empty_gains_nothing() {
        let rejected = lb("car", CAR, Source::Model, BoxState::Rejected);
        let mut f = LabelFile {
            empty_confirmed: true,
            ..frame(vec![rejected])
        };
        let before = f.clone();
        let r = apply_labelers(&mut f, boxes(&[sam("dog", OFF, 0.9)]), "sam3", REL);
        assert_eq!((r.closed, r.votes), (1, 0));
        assert_eq!(f, before);
        // Its rejected box takes a vote, and that is all.
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!((r.closed, r.votes), (1, 1));
        assert_eq!(f.status, Status::Reviewed);
        assert_eq!(f.boxes.len(), 1);
        assert_eq!(f.boxes[0].state, BoxState::Rejected);
    }

    #[test]
    fn labelers_that_agree_pass_the_gate_with_a_reason_and_a_short_name() {
        let mut f = frame(vec![accepted("person", [400.0, 10.0, 450.0, 90.0])]);
        f.overlaps_checked = true;
        f.bulk_accepted = true;
        f.reviewed_at = Some("2026-10-22T12:00:00.000Z".into());
        // Both found it, neither reaches the threshold, and the triage says accept.
        let mut low = p("car", 0.62, true);
        low.bbox = CAR;
        low.sources = [("gdino".to_string(), 0.4), ("rfdetr".to_string(), 0.45)].into();
        let b = boxes(&[low]);
        assert_eq!(b[0].auto.triage, Some(Triage::Accept));
        assert!(b[0].auto.why.is_empty());
        let r = apply_labelers(&mut f, b, "gdino", REL);
        assert_eq!(r.missing.added, 1);
        assert!(r.for_review());
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by gdino, rfdetr: 1 new box")
        );
        assert!(!f.bulk_accepted, "a reviewer has to look at it now");
        assert!(!f.overlaps_checked);
        assert_eq!(f.reviewed_at.as_deref(), Some("2026-10-22T12:00:00.000Z"));
        let n = &f.boxes[1];
        assert_eq!(n.auto.triage, Some(Triage::Review));
        assert_eq!(n.auto.why, ["new on a reviewed frame: the labelers agree"]);
        // One labeler: its name and score, after the triage's own reasons.
        let mut g = frame(vec![]);
        apply_labelers(&mut g, boxes(&[sam("car", CAR, 0.8)]), "sam3", REL);
        assert_eq!(
            g.boxes[0].auto.why.last().map(String::as_str),
            Some("new on a reviewed frame: sam3 at 0.80")
        );
        assert_eq!(
            g.disagreement.as_deref(),
            Some("re-label by sam3: 1 new box")
        );
        // The file still saves; a reviewed save waits for the reviewer's decision.
        let saved = labels::normalize(g, &classes(), 640, 480, "t").unwrap();
        let mut done = saved;
        done.status = Status::Reviewed;
        assert!(labels::normalize(done.clone(), &classes(), 640, 480, "t").is_err());
        done.boxes[0].state = BoxState::Accepted;
        let done = labels::normalize(done, &classes(), 640, 480, "t").unwrap();
        assert_eq!(done.training().count(), 1);
        assert_eq!(done.disagreement, None);
    }

    #[test]
    fn a_rejected_box_no_reviewer_judged_stays_and_the_proposal_is_counted() {
        for reject in [
            LabelBox {
                reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
                ..lb("car", CAR, Source::Auto, BoxState::Rejected)
            },
            LabelBox {
                origin: Some(labels::Origin::BelowThreshold),
                ..lb("truck", CAR, Source::Model, BoxState::Rejected)
            },
        ] {
            let mut f = frame(vec![reject.clone()]);
            let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
            assert_eq!((r.gate.on_reject, r.missing.overwritten), (1, 0));
            assert_eq!(f.status, Status::Reviewed);
            assert_eq!(f.boxes.len(), 1);
            assert_eq!(without_votes(&f.boxes[0]), without_votes(&reject));
        }
        // Past MATCH_IOU it says nothing, as for `apply_missing`.
        let low = LabelBox {
            reason: Some(labels::bulk::LOW_SCORE_REASON.into()),
            ..lb("car", CAR, Source::Auto, BoxState::Rejected)
        };
        let mut f = frame(vec![low]);
        let beside = [160.0, 100.0, 260.0, 200.0];
        let r = apply_labelers(&mut f, boxes(&[sam("car", beside, 0.9)]), "sam3", REL);
        assert_eq!((r.gate.on_reject, r.missing.added), (0, 1));
    }

    #[test]
    fn a_frame_a_re_label_sent_back_is_still_gated() {
        let mut f = frame(vec![accepted("cat", CAR)]);
        apply_labelers(&mut f, boxes(&[sam("car", OFF, 0.9)]), "sam3", REL);
        assert_eq!(f.status, Status::Unreviewed);
        let before = f.clone();
        // What `apply` would have added, and what it would have written on the accepted cat.
        let props = [
            sam("dog", CAR, 0.9),
            sam("person", [0.0, 0.0, 40.0, 40.0], 0.3),
        ];
        let r = apply_labelers(&mut f, boxes(&props), "gdino", REL);
        assert_eq!((r.gate.other_object, r.gate.low_score), (1, 1));
        assert_eq!(f, before);
        // A confident new object joins, and the note counts both.
        let r = apply_labelers(
            &mut f,
            boxes(&[sam("dog", [0.0, 0.0, 40.0, 40.0], 0.9)]),
            "sam3",
            REL,
        );
        assert_eq!(r.missing.added, 1);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by sam3: 2 new boxes")
        );
        // Confirmed empty as well: closed.
        let mut g = before;
        g.empty_confirmed = true;
        let frozen = g.clone();
        let r = apply_labelers(
            &mut g,
            boxes(&[sam("dog", [0.0, 0.0, 40.0, 40.0], 0.9)]),
            "sam3",
            REL,
        );
        assert_eq!(r.closed, 1);
        assert_eq!(g, frozen);
    }

    #[test]
    fn an_unreviewed_frame_takes_apply_and_votes_on_what_it_keeps() {
        // The same proposals through `apply` alone, with no box to vote on: the same result.
        let props = boxes(&[sam("dog", OFF, 0.2), sam("car", CAR, 0.3)]);
        let (mut a, mut b) = (LabelFile::default(), LabelFile::default());
        apply(&mut a, props.clone());
        let r = apply_labelers(&mut b, props, "sam3", REL);
        assert_eq!(a, b);
        assert_eq!((r.apply.merge.added, r.votes), (2, 0));
        // A pending model box without evidence gets the proposal's evidence, as ever.
        let mut f = LabelFile {
            boxes: vec![lb("car", CAR, Source::Model, BoxState::Pending)],
            ..LabelFile::default()
        };
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!((r.apply.annotated, r.votes), (1, 0));
        assert_eq!(f.boxes[0].auto.agree, Some(false));
        // A pending box with evidence: dropped by `apply`, and its new labeler is recorded.
        let mut g = LabelFile::default();
        apply_labelers(
            &mut g,
            boxes(&[{
                let mut gd = sam("car", CAR, 0.8);
                gd.sources = [("gdino".to_string(), 0.8)].into();
                gd
            }]),
            "gdino",
            REL,
        );
        let triaged = (g.boxes[0].auto.triage, g.boxes[0].auto.why.clone());
        let r = apply_labelers(&mut g, boxes(&[sam("car", CAR, 0.6)]), "sam3", REL);
        assert_eq!((r.apply.merge.dropped, r.votes), (1, 1));
        assert!(!r.for_review(), "a vote queues nothing");
        assert_eq!(g.boxes.len(), 1);
        assert_eq!(
            (g.boxes[0].auto.triage, g.boxes[0].auto.why.clone()),
            triaged
        );
        assert_eq!(g.boxes[0].auto.labelers["gdino"], 0.8);
        assert_eq!(g.boxes[0].auto.labelers["sam3"], 0.6);
    }

    /// [`apply_relabel`] of one `custom` detection at 0.9 on a 640x480 day frame.
    fn found_by_custom(f: &mut LabelFile, label: &str, bbox: [f32; 4]) -> RelabelReport {
        let cls = classes();
        let dets = [Det {
            label: label.into(),
            score: 0.9,
            bbox,
        }];
        let run = ModelRun {
            model: "custom",
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        let out = model_boxes(&dets, &run, 640, 480);
        for b in &out.boxes {
            let keys: Vec<&String> = b.auto.labelers.keys().collect();
            assert_eq!(keys, ["custom"], "keyed by the model's name");
        }
        apply_relabel(f, &dets, out.boxes, "custom", &cls)
    }

    #[test]
    fn a_model_run_votes_on_a_box_it_finds_again_and_changes_nothing_else() {
        // A pending box of a labelers run, which `apply` leaves alone: it has evidence.
        let mut b = lb("car", CAR, Source::Auto, BoxState::Pending);
        b.auto.labelers.insert("gdino".into(), 0.41);
        b.auto.triage = Some(Triage::Review);
        b.auto.why = vec!["found by gdino only".into()];
        let mut f = frame(vec![b]);
        f.status = Status::Unreviewed;
        let before = f.clone();
        let r = found_by_custom(&mut f, "car", CAR);
        assert_eq!(r.votes, 1, "{r:?}");
        assert_eq!(r.disagreement, None);
        assert_eq!(
            r.apply.merge.dropped, 1,
            "the proposal is dropped as before"
        );
        assert_eq!(
            (
                r.apply.merge.added,
                r.apply.annotated,
                r.apply.class_suggestions
            ),
            (0, 0, 0)
        );
        assert_eq!(f.boxes[0].auto.labelers["gdino"], 0.41);
        assert_eq!(f.boxes[0].auto.labelers["custom"], 0.9);
        let mut g = f.clone();
        g.boxes[0].auto.labelers = before.boxes[0].auto.labelers.clone();
        assert_eq!(g, before, "only `labelers` moved");

        // A reviewed frame: its box matches, so it stays reviewed with no note.
        let mut b = accepted("car", CAR);
        b.auto.labelers.insert("gdino".into(), 0.41);
        let mut f = frame(vec![b]);
        f.status = Status::Reviewed;
        f.overlaps_checked = true;
        let before = f.clone();
        let r = found_by_custom(&mut f, "car", CAR);
        assert_eq!((r.votes, r.apply.annotated), (1, 0), "{r:?}");
        assert_eq!(r.disagreement, None);
        let mut g = f.clone();
        g.boxes[0].auto.labelers = before.boxes[0].auto.labelers.clone();
        assert_eq!(g, before, "status, note and the box stand");
    }

    #[test]
    fn a_model_vote_does_not_undo_a_disagreement_or_add_to_the_annotated_count() {
        // Vehicles are one object, so the truck takes the vote while the class differs.
        let mut b = accepted("truck", CAR);
        b.auto.labelers.insert("gdino".into(), 0.41);
        let mut f = frame(vec![b]);
        f.status = Status::Reviewed;
        let r = found_by_custom(&mut f, "car", CAR);
        assert_eq!(r.votes, 1, "{r:?}");
        assert_eq!(f.status, Status::Unreviewed, "a class change still reopens");
        assert_eq!(f.disagreement, r.disagreement);
        assert!(r.disagreement.is_some());
        assert_eq!(r.apply.merge.added, 1, "the other label is added as before");
        assert_eq!(r.apply.annotated, 0);
    }

    #[test]
    fn a_model_gives_no_vote_to_a_pending_box_without_evidence() {
        // Of the same label `apply` annotates it, as ever: evidence, no vote on top.
        let mut f = frame(vec![lb("car", CAR, Source::Model, BoxState::Pending)]);
        f.status = Status::Unreviewed;
        let r = found_by_custom(&mut f, "car", CAR);
        assert_eq!((r.apply.annotated, r.votes), (1, 0), "{r:?}");
        // Another vehicle, closest to the detection, while a car with evidence blocks both the
        // annotation and the class suggestion: nothing takes the vote.
        let near = [100.0, 100.0, 200.0, 190.0];
        assert!(iou(&near, &CAR) >= STACK_IOU);
        let hint = LabelBox {
            origin: Some(labels::Origin::BelowThreshold),
            ..lb("truck", CAR, Source::Model, BoxState::Pending)
        };
        let mut car = lb("car", near, Source::Auto, BoxState::Pending);
        car.auto.labelers.insert("gdino".into(), 0.41);
        let mut f = frame(vec![hint, car]);
        f.status = Status::Unreviewed;
        let before = f.clone();
        let r = found_by_custom(&mut f, "car", CAR);
        assert_eq!(r.votes, 0, "{r:?}");
        assert_eq!(
            f, before,
            "the best match may not take it, nor the next one"
        );
    }

    #[test]
    fn a_long_model_name_cannot_break_a_save_through_its_vote() {
        let cls = classes();
        let name = "m".repeat(labels::MAX_LABEL_LEN + 30);
        let dets = [Det {
            label: "car".into(),
            score: 0.9,
            bbox: CAR,
        }];
        let run = ModelRun {
            model: &name,
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        let boxes = model_boxes(&dets, &run, 640, 480).boxes;
        let mut b = accepted("car", CAR);
        b.auto.labelers.insert("gdino".into(), 0.41);
        let mut f = frame(vec![b]);
        f.status = Status::Reviewed;
        assert_eq!(apply_relabel(&mut f, &dets, boxes, &name, &cls).votes, 0);
        assert!(labels::normalize(f, &cls, 640, 480, "t").is_ok());
    }

    #[test]
    fn a_missing_only_run_never_votes() {
        let mut b = accepted("car", CAR);
        b.auto.labelers.insert("gdino".into(), 0.41);
        let mut f = frame(vec![b]);
        let before = f.clone();
        let cls = classes();
        let run = ModelRun {
            model: "custom",
            classes: &cls,
            scene: "day",
            accept_score: None,
        };
        let dets = [Det {
            label: "car".into(),
            score: 0.9,
            bbox: CAR,
        }];
        let boxes = model_boxes(&dets, &run, 640, 480).boxes;
        let m = apply_missing(&mut f, boxes, "custom");
        assert_eq!(m.covered, 1);
        assert_eq!(f, before, "existing boxes get no evidence");
    }

    #[test]
    fn votes_never_overwrite_or_change_anything_but_labelers() {
        let mut b = accepted("car", CAR);
        b.score = Some(0.42);
        b.auto.labelers.insert("sam3".into(), 0.33);
        let mut f = frame(vec![b]);
        let before = f.clone();
        let mut props = boxes(&[sam("car", CAR, 0.9)]);
        props[0].auto.labelers.insert("rfdetr".into(), 0.7);
        let r = apply_labelers(&mut f, props, "sam3", REL);
        assert_eq!((r.votes, r.missing.covered), (1, 1));
        assert!(!r.for_review());
        assert_eq!(f.boxes[0].auto.labelers["sam3"], 0.33, "never overwritten");
        assert_eq!(f.boxes[0].auto.labelers["rfdetr"], 0.7);
        assert_eq!(without_votes(&f.boxes[0]), without_votes(&before.boxes[0]));
        let mut g = f.clone();
        g.boxes[0].auto.labelers = before.boxes[0].auto.labelers.clone();
        assert_eq!(g, before, "only `labelers` moved");
        // A serving-model box without evidence takes `labelers` and nothing else.
        let mut h = frame(vec![accepted("car", CAR)]);
        assert!(h.boxes[0].auto.is_empty());
        apply_labelers(&mut h, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        let auto = &h.boxes[0].auto;
        assert_eq!(auto.labelers.len(), 1);
        assert_eq!((auto.agree, auto.triage), (None, None));
        assert!(auto.why.is_empty());
        assert!(labels::normalize(h.clone(), &classes(), 640, 480, "t").is_ok());
        // The run counted as evidence on a reviewed frame: not "found nothing", not "never run".
        h.auto_run = Some(labels::AutoRun {
            at: "t".into(),
            labeler: "x".into(),
            classes: Vec::new(),
        });
        use labels::queue::AutoFilter;
        assert!(!AutoFilter::Nothing.matches(Some(&h)));
        assert!(!AutoFilter::Never.matches(Some(&h)));
    }

    #[test]
    fn a_vote_goes_to_the_best_match_of_the_object_at_iou_0_7() {
        let (near, far) = ([100.0, 100.0, 200.0, 200.0], [110.0, 100.0, 200.0, 200.0]);
        let mut f = frame(vec![accepted("car", far), accepted("car", near)]);
        assert!(iou(&far, &CAR) >= STACK_IOU);
        apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert!(f.boxes[0].auto.is_empty(), "second best");
        assert!(f.boxes[1].auto.labelers.contains_key("sam3"));
        // Vehicles are one object.
        let mut f = frame(vec![accepted("truck", CAR)]);
        assert_eq!(
            apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL).votes,
            1
        );
        // Another object: none, at any IoU.
        let mut f = frame(vec![accepted("dog", CAR)]);
        assert_eq!(
            apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL).votes,
            0
        );
        // Under IoU 0.7: none (and the box covers the proposal).
        let under = [100.0, 100.0, 200.0, 160.0];
        assert!(iou(&under, &CAR) < STACK_IOU);
        let mut f = frame(vec![accepted("car", under)]);
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!((r.votes, r.missing.covered), (0, 1));
        // A best match that may not take the vote does not pass it to the next one.
        let mut f = frame(vec![
            accepted("car", far),
            lb("car", near, Source::Model, BoxState::Pending),
        ]);
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!(r.votes, 0);
        assert!(f.boxes[0].auto.is_empty());
        assert!(f.boxes[1].auto.is_empty());
    }

    #[test]
    fn a_pending_hint_without_evidence_gets_no_vote_so_its_triage_stands() {
        // A frame the bulk accept takes: one accepted-by-triage box, one model hint.
        let mut sure = lb(
            "person",
            [400.0, 10.0, 450.0, 90.0],
            Source::Auto,
            BoxState::Pending,
        );
        sure.auto.triage = Some(Triage::Accept);
        let hint = LabelBox {
            origin: Some(labels::Origin::BelowThreshold),
            ..lb("truck", CAR, Source::Model, BoxState::Pending)
        };
        let mut f = LabelFile {
            boxes: vec![sure, hint],
            ..LabelFile::default()
        };
        assert!(labels::auto_accept_eligible(&f));
        // Direct: `apply` would give it a class suggestion, so go through `vote` alone.
        let props = boxes(&[sam("car", CAR, 0.9)]);
        assert_eq!(vote(&mut f, &props, 2), 0);
        assert!(f.boxes[1].auto.is_empty());
        assert!(labels::auto_accept_eligible(&f));
    }

    #[test]
    fn votes_stop_at_the_cap_a_save_enforces() {
        let mut b = accepted("car", CAR);
        for i in 0..labels::MAX_LABELERS {
            b.auto.labelers.insert(format!("m{i:02}"), 0.5);
        }
        let mut f = frame(vec![b]);
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!(r.votes, 0);
        assert_eq!(f.boxes[0].auto.labelers.len(), labels::MAX_LABELERS);
        assert!(labels::normalize(f, &classes(), 640, 480, "t").is_ok());
    }

    #[test]
    fn votes_leave_overlap_suggestions_live() {
        use labels::overlaps::{self, Suggestion, Verdict};
        let inner = [110.0, 110.0, 170.0, 170.0];
        let cls = classes();
        let wanted = labels::bulk::Wanted {
            classes: &cls,
            hidden: &[],
        };
        let mut f = frame(vec![accepted("car", CAR), accepted("car", inner)]);
        f.overlap_suggestions = vec![Suggestion {
            a: CAR,
            b: inner,
            verdict: Verdict::DropB,
            why: "x".into(),
            kept_score: Some(0.9),
            other_score: Some(0.0),
            low_confidence: false,
            weak: false,
        }];
        assert_eq!(overlaps::live(&f, Some(&wanted)).len(), 1);
        let stored = f.overlap_suggestions.clone();
        let r = apply_labelers(&mut f, boxes(&[sam("car", CAR, 0.9)]), "sam3", REL);
        assert_eq!(r.votes, 1);
        assert_eq!(f.overlap_suggestions, stored);
        assert_eq!(overlaps::live(&f, Some(&wanted)).len(), 1);
        // The vote is there for the next check: the box has one more witness.
        assert!(overlaps::witnesses(&f.boxes, 0).contains(&"sam3".to_string()));
    }

    /// The script's own tests of its fusion code (WBF, NMS, prompt batching, phrase mapping,
    /// VLM answers): pure Python, run when a Python 3 is on PATH (skipped otherwise).
    #[test]
    fn the_autolabel_script_self_test_passes() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/autolabel.py");
        let Some(py) = ["python3", "python"].into_iter().find(|p| {
            std::process::Command::new(p)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        }) else {
            eprintln!("no Python 3 on PATH; skipping the autolabel.py self-test");
            return;
        };
        let out = std::process::Command::new(py)
            .arg("-I")
            .arg(&script)
            .arg("--self-test")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{text}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(text.contains("\"event\":\"self_test\""), "{text}");
    }

    #[test]
    fn script_lines_parse() {
        let line = r#"{"frame":"ir/cam/2026-10-22/a.jpg","width":640,"height":480,"boxes":[{"label":"person","bbox":[1,2,30,40],"score":0.42,"sources":{"gdino":0.42},"agree":false,"vlm":{"model":"qwen3-vl-2b","answer":"rock","expected":"person","ok":false,"raw":"Rock."}}],"ms":{"gdino":1200}}"#;
        let f: FrameProposals = serde_json::from_str(line).unwrap();
        assert_eq!(f.boxes[0].vlm.as_ref().unwrap().answer, "rock");
        assert!(!f.boxes[0].agree);
    }

    /// A proposal both gdino and RF-DETR found, fused to `score`, with this VLM reply.
    fn fused(label: &str, bbox: [f32; 4], score: f32, vlm: Option<(&str, bool)>) -> Proposal {
        Proposal {
            bbox,
            vlm: vlm.map(|(expected, ok)| VlmOut {
                model: "qwen3-vl-2b".into(),
                answer: if ok { expected } else { "no" }.into(),
                expected: expected.into(),
                ok,
            }),
            ..p(label, score, false)
        }
    }

    const YES: Option<(&str, bool)> = Some(("yes", true));
    const NO: Option<(&str, bool)> = Some(("yes", false));

    /// A place for a `label` box whose hash is (or is not) an audit sample on [`REL`].
    fn place(label: &str, audit: bool) -> [f32; 4] {
        (0..2000)
            .map(|i| {
                let x = i as f32;
                [x, 10.0, x + 60.0, 90.0]
            })
            .find(|b| audit_sample(REL, label, b) == audit)
            .unwrap()
    }

    /// What a labelers run makes of one proposal on an empty unreviewed frame.
    fn one(run: &str, prop: Proposal) -> (LabelBox, LabelersReport) {
        let mut f = LabelFile::default();
        let r = apply_labelers(&mut f, boxes(&[prop]), run, REL);
        assert_eq!(f.boxes.len(), 1);
        (f.boxes.remove(0), r)
    }

    #[test]
    fn the_tiers_turn_at_their_exact_scores() {
        let at = place("car", false);
        let (b, r) = one("gdino", fused("car", at, 0.1499, None));
        assert!(b.is_dismissed());
        assert_eq!(b.reason.as_deref(), Some(labels::bulk::DISMISSED_REASON));
        assert_eq!((r.dismissed, r.likely, r.audit), (1, 0, 0));
        assert!(r.for_review(), "the frame stays reachable from the queue");
        let (b, r) = one("gdino", fused("car", at, 0.15, None));
        assert_eq!(
            (b.state, b.auto.likely, r.dismissed),
            (BoxState::Pending, false, 0)
        );
        assert_ne!(b.reason.as_deref(), Some(labels::bulk::DISMISSED_REASON));
        let (b, _) = one("gdino", fused("car", at, 0.2499, YES));
        assert!(!b.auto.likely);
        let (b, r) = one("gdino", fused("car", at, 0.25, YES));
        assert_eq!(
            (b.state, b.auto.likely, r.likely),
            (BoxState::Pending, true, 1)
        );
        assert_eq!(b.auto.triage, Some(Triage::Review), "triage stays review");
    }

    #[test]
    fn a_vlm_no_never_hides_a_box() {
        let at = place("car", false);
        for score in [0.15, 0.3, 0.55] {
            let (b, _) = one("gdino", fused("car", at, score, NO));
            assert_eq!(
                (b.state, b.auto.likely),
                (BoxState::Pending, false),
                "{score}"
            );
        }
    }

    #[test]
    fn a_low_score_goes_whatever_the_vlm_said() {
        let at = place("car", false);
        assert!(one("gdino", fused("car", at, 0.1, YES)).0.is_dismissed());
        assert!(one("gdino", fused("car", at, 0.1, NO)).0.is_dismissed());
    }

    #[test]
    fn with_the_vlm_off_nothing_is_likely_and_the_score_still_dismisses() {
        let at = place("car", false);
        let (b, r) = one("gdino", fused("car", at, 0.9, None));
        assert_eq!((b.auto.likely, r.likely), (false, 0));
        assert!(one("gdino", fused("car", at, 0.1, None)).0.is_dismissed());
    }

    #[test]
    fn an_old_seven_way_vlm_answer_is_never_likely() {
        let at = place("person", false);
        let (b, _) = one("gdino", fused("person", at, 0.9, Some(("person", true))));
        assert!(!b.auto.likely);
        assert_eq!(b.state, BoxState::Pending);
    }

    #[test]
    fn trucks_are_never_dismissed_in_any_spelling() {
        let at = place("truck", false);
        let (b, r) = one("gdino", fused("truck", at, 0.05, None));
        assert_eq!((b.state, r.dismissed), (BoxState::Pending, 0));
        let mut upper = scored_box(0.05);
        upper.label = "Truck".into();
        assert_eq!(tier(&upper, REL, false), Tier::Unlikely);
    }

    /// A pending auto box scoring `score` at a place that is no audit sample.
    fn scored_box(score: f32) -> LabelBox {
        LabelBox {
            bbox: place("car", false),
            score: Some(score),
            ..labels::tests_support::auto_box("car")
        }
    }

    #[test]
    fn a_class_only_one_labeler_reports_stays_at_its_labelers_threshold() {
        // RF-DETR cannot report a raccoon, so the fusion does not halve it: its fused score is
        // gdino's own, at least the 0.2 threshold.
        let at = place("raccoon", false);
        let (b, r) = one("gdino", fused("raccoon", at, 0.2, None));
        assert_eq!((b.state, r.dismissed), (BoxState::Pending, 0));
    }

    #[test]
    fn sam3_and_its_runs_never_dismiss() {
        let at = place("car", false);
        // Found by SAM 3, whatever the run's name.
        let mut prop = unasked("car", at, 0.2);
        prop.score = 0.1;
        let (b, r) = one("gdino", prop);
        assert_eq!((b.state, r.dismissed), (BoxState::Pending, 0));
        // RF-DETR alone on a SAM 3 run: the thresholds were not measured with SAM 3 in the fusion.
        let mut alone = fused("car", at, 0.1, None);
        alone.sources = [("rfdetr".to_string(), 0.2)].into();
        let (b, r) = one(SAM3_KEY, alone.clone());
        assert_eq!((b.state, r.dismissed), (BoxState::Pending, 0));
        assert!(one("rfdetr", alone).0.is_dismissed());
    }

    #[test]
    fn a_box_triage_accepted_is_never_dismissed() {
        let mut b = scored_box(0.1);
        b.auto.triage = Some(Triage::Accept);
        assert_eq!(tier(&b, REL, false), Tier::Unlikely);
    }

    #[test]
    fn only_untouched_pending_auto_boxes_get_a_tier() {
        let touched: [fn(&mut LabelBox); 5] = [
            |b| b.source = Source::Model,
            |b| b.state = BoxState::Accepted,
            |b| b.orig_bbox = Some([0.0, 0.0, 5.0, 5.0]),
            |b| b.difficult = true,
            |b| b.score = None,
        ];
        assert_eq!(tier(&scored_box(0.1), REL, false), Tier::Dismissed);
        for t in touched {
            let mut b = scored_box(0.1);
            t(&mut b);
            assert_eq!(tier(&b, REL, false), Tier::Unlikely, "{b:?}");
        }
    }

    #[test]
    fn a_model_run_never_dismisses_or_marks_likely() {
        let dets = [Det {
            label: "car".into(),
            score: 0.05,
            bbox: place("car", false),
        }];
        let run = ModelRun {
            model: "nano",
            classes: &classes(),
            scene: "day",
            accept_score: None,
        };
        let b = model_boxes(&dets, &run, 640, 480).boxes;
        let mut f = LabelFile::default();
        apply_relabel(&mut f, &dets, b, "nano", &classes());
        assert_eq!(
            (f.boxes[0].state, f.boxes[0].auto.likely),
            (BoxState::Pending, false)
        );
        // Nor does a `missing_only` run, on the same low score.
        let mut g = LabelFile {
            status: Status::Reviewed,
            ..LabelFile::default()
        };
        let b = model_boxes(&dets, &run, 640, 480).boxes;
        assert_eq!(apply_missing(&mut g, b, "nano").added, 1);
        assert_eq!(g.boxes[0].state, BoxState::Pending);
    }

    #[test]
    fn about_one_in_fifty_stays_visible_and_the_same_one_on_a_rerun() {
        let at = place("car", true);
        let (b, r) = one("gdino", fused("car", at, 0.1, None));
        assert_eq!(
            (b.state, b.auto.audit, r.audit, r.dismissed),
            (BoxState::Pending, true, 1, 0)
        );
        assert_eq!(b.auto.why.last().map(String::as_str), Some(AUDIT_WHY));
        assert_eq!(
            b.auto.triage,
            Some(Triage::Review),
            "a save needs a decision"
        );
        assert!(labels::needs_decision(&b));
        // Sub-pixel noise does not change the answer.
        assert_eq!(
            audit_sample(REL, "car", &[10.2, 10.0, 70.1, 90.0]),
            audit_sample(REL, "car", &[10.3, 10.1, 70.0, 89.9])
        );
        // Frame, label and place all count.
        assert_ne!(
            (0..200)
                .map(|i| audit_sample(&format!("{REL}{i}"), "car", &at))
                .collect::<Vec<_>>(),
            (0..200)
                .map(|i| audit_sample(&format!("{REL}{i}"), "dog", &at))
                .collect::<Vec<_>>()
        );
        let n = 20_000;
        let hits = (0..n)
            .filter(|i| audit_sample(&format!("day/c/2026-01-01/{i}.jpg"), "car", &at))
            .count();
        assert!((300..500).contains(&hits), "{hits} of {n}");
        // A rerun of the same proposals: the audit box stays one, the dismissed box comes back
        // dismissed, nothing doubles.
        let props = boxes(&[
            fused("car", at, 0.1, None),
            fused("dog", place("dog", false), 0.1, None),
        ]);
        let mut f = LabelFile::default();
        apply_labelers(&mut f, props.clone(), "gdino", REL);
        let once = f.clone();
        apply_labelers(&mut f, props, "gdino", REL);
        assert_eq!(f.boxes.len(), 2);
        assert!(once.boxes[1].is_dismissed() && f.boxes[1].is_dismissed());
        assert_eq!(once.boxes, f.boxes);
    }

    #[test]
    fn stored_boxes_and_decided_frames_are_never_tiered() {
        // An older run's box: scoring low, a yes on file, no tier.
        let at = place("car", false);
        let mut old = scored_box(0.1);
        old.auto.triage = Some(Triage::Review);
        let mut high = scored_box(0.9);
        high.bbox = OFF;
        high.auto.labelers = [("gdino".to_string(), 0.9)].into();
        high.auto.vlm = Some(VlmCheck {
            model: "m".into(),
            answer: "yes".into(),
            expected: "yes".into(),
            ok: true,
        });
        let mut f = LabelFile {
            boxes: vec![old, high],
            ..LabelFile::default()
        };
        let before = f.clone();
        let r = apply_labelers(&mut f, Vec::new(), "gdino", REL);
        assert_eq!((r.dismissed, r.likely), (0, 0));
        assert_eq!(f, before);
        // A reviewed frame: the gate, no tier, also for a box the gate lets through at a low fused
        // score (the labelers agree).
        let mut agreed = fused("car", at, 0.1, YES);
        agreed.agree = true;
        for (status, empty) in [
            (Status::Reviewed, false),
            (Status::Skipped, false),
            (Status::Reviewed, true),
        ] {
            let mut g = LabelFile {
                status,
                empty_confirmed: empty,
                ..LabelFile::default()
            };
            let r = apply_labelers(&mut g, boxes(&[agreed.clone()]), "gdino", REL);
            assert_eq!((r.dismissed, r.likely, r.audit), (0, 0, 0));
            assert!(!g.boxes.iter().any(|b| b.is_dismissed() || b.auto.likely));
        }
    }

    /// A dismissed `label` box at `bbox`.
    fn dismissed(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            bbox,
            state: BoxState::Rejected,
            reason: Some(labels::bulk::DISMISSED_REASON.into()),
            ..labels::tests_support::auto_box(label)
        }
    }

    #[test]
    fn a_dismissal_is_no_judged_reject() {
        assert!(unjudged_reject(&dismissed("car", CAR)));
        let mut judged = dismissed("car", CAR);
        judged.reject_reason = Some(labels::RejectReason::FalsePositive);
        assert!(!unjudged_reject(&judged));
    }

    #[test]
    fn a_model_finding_a_dismissed_box_overwrites_it() {
        let mut f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![dismissed("car", CAR)],
            ..LabelFile::default()
        };
        let mut b = labels::tests_support::auto_box("car");
        b.bbox = CAR;
        let r = apply_missing(&mut f, vec![b], "nano");
        assert_eq!((r.overwritten, r.added), (1, 0));
        assert_eq!(f.boxes.len(), 1);
        assert_eq!(f.boxes[0].state, BoxState::Pending);
    }

    #[test]
    fn a_dismissed_box_blocks_no_other_object_at_the_gate() {
        let mut dog = sam("dog", CAR, 0.9);
        dog.agree = true;
        let dog = boxes(&[dog]);
        let f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![dismissed("car", CAR)],
            ..LabelFile::default()
        };
        assert_eq!(gate_drop(&f, &dog[0]), None);
        // A reviewer's reject there still does.
        let mut judged = f.clone();
        judged.boxes[0].reason = None;
        assert_eq!(gate_drop(&judged, &dog[0]), Some(GateDrop::OtherObject));
        // The same object: a labelers run leaves a rejected box alone, dismissed or not.
        let car = boxes(&[sam("car", CAR, 0.9)]);
        assert_eq!(gate_drop(&f, &car[0]), Some(GateDrop::OnReject));
    }

    #[test]
    fn a_later_run_can_bring_a_dismissed_box_back() {
        let at = place("car", false);
        let mut f = LabelFile {
            boxes: vec![dismissed("car", at)],
            ..LabelFile::default()
        };
        // SAM 3 on an unreviewed frame: the proposal is no covered box.
        let r = apply_labelers(&mut f, boxes(&[sam("car", at, 0.2)]), SAM3_KEY, REL);
        assert_eq!((r.apply.merge.added, r.apply.merge.dropped), (1, 0));
        assert_eq!(f.boxes.len(), 1);
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        // A proposal of a later run scoring higher.
        let mut g = LabelFile {
            boxes: vec![dismissed("car", at)],
            ..LabelFile::default()
        };
        apply_labelers(&mut g, boxes(&[fused("car", at, 0.4, YES)]), "gdino", REL);
        assert!(!g.boxes[0].is_dismissed() && g.boxes[0].auto.likely);
        // A dismissed box elsewhere stays as it is.
        let mut h = LabelFile {
            boxes: vec![dismissed("car", OFF)],
            ..LabelFile::default()
        };
        apply_labelers(&mut h, boxes(&[fused("car", CAR, 0.4, YES)]), "gdino", REL);
        assert!(h.boxes[0].is_dismissed());
        // A trained model's detection on one.
        let dets = [Det {
            label: "car".into(),
            score: 0.9,
            bbox: OFF,
        }];
        let run = ModelRun {
            model: "nano",
            classes: &classes(),
            scene: "day",
            accept_score: None,
        };
        let b = model_boxes(&dets, &run, 640, 480).boxes;
        apply_relabel(&mut h, &dets, b, "nano", &classes());
        assert_eq!(h.boxes.len(), 2);
        assert!(h.boxes.iter().all(|b| !b.is_dismissed()));
    }

    #[test]
    fn a_dismissed_box_stays_rejected_through_a_save() {
        let mut f = LabelFile {
            boxes: vec![dismissed("car", CAR)],
            ..LabelFile::default()
        };
        f.status = Status::Reviewed;
        let n = labels::normalize(f, &classes(), 640, 480, "2026-10-22T12:00:00.000Z").unwrap();
        assert!(n.boxes[0].is_dismissed());
        assert_eq!(n.training().count(), 0);
    }

    // ---- spots ------------------------------------------------------------------------------

    const FRAME_SIZE: (u32, u32) = (640, 480);

    /// A rejected box of camera `yard` at `at` on the frame of `day` number `n`.
    fn decision(day: &str, n: usize, label: &str, at: [f32; 4], state: BoxState) -> Decision {
        Decision {
            frame: format!("day/yard/{day}/20260101T000000000Z_{n}.jpg"),
            label: label.into(),
            bbox: at,
            state,
            reject_reason: None,
            width: FRAME_SIZE.0,
            height: FRAME_SIZE.1,
        }
    }

    fn memory(decisions: Vec<Decision>, store: CameraSpots) -> CameraMemory {
        let h = History {
            camera: "yard".into(),
            decisions,
            truncated: false,
        };
        CameraMemory::new(&h, store)
    }

    /// Rejected on two frames, a week before [`REL`].
    fn bad_spot(at: [f32; 4]) -> CameraMemory {
        let d = (0..2)
            .map(|n| decision("2026-10-15", n, "car", at, BoxState::Rejected))
            .collect();
        memory(d, CameraSpots::default())
    }

    fn parked_spot(at: [f32; 4]) -> CameraMemory {
        let mut store = CameraSpots::default();
        store.parked.insert(
            "640x480".into(),
            vec![Parked {
                label: "car".into(),
                bbox: at,
                from: "2026-10-01".into(),
                retired: None,
            }],
        );
        memory(Vec::new(), store)
    }

    /// One proposal on an empty unreviewed frame of [`REL`] with `m`.
    fn at_spot(run: &str, m: &CameraMemory, prop: Proposal) -> (LabelFile, LabelersReport) {
        let mut f = LabelFile::default();
        let spots = FrameSpots {
            memory: m,
            width: FRAME_SIZE.0,
            height: FRAME_SIZE.1,
        };
        let r = super::apply_labelers(&mut f, boxes(&[prop]), run, REL, Some(spots));
        (f, r)
    }

    #[test]
    fn a_rejected_spot_hides_a_new_box_with_its_own_reason() {
        let at = place("truck", false);
        let m = bad_spot(at);
        let (f, r) = at_spot("gdino", &m, fused("truck", at, 0.3, None));
        let b = &f.boxes[0];
        assert!(
            b.is_dismissed(),
            "a truck: the exception is the score rule's"
        );
        assert_eq!(
            b.reason.as_deref(),
            Some(labels::bulk::SPOT_DISMISSED_REASON)
        );
        assert_eq!(b.auto.spot_rejections, Some(2));
        assert_eq!((r.spot_dismissed, r.dismissed, r.audit), (1, 0, 0));
        // Elsewhere nothing happens, and a frame without a memory neither.
        let (f, _) = at_spot("gdino", &m, fused("truck", OFF, 0.3, None));
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        // A score below the tiers' line is the tier's dismissal, not the spot's.
        let at = place("car", false);
        let (f, r) = at_spot("gdino", &bad_spot(at), fused("car", at, 0.1, None));
        assert_eq!(
            f.boxes[0].reason.as_deref(),
            Some(labels::bulk::DISMISSED_REASON)
        );
        assert_eq!((r.spot_dismissed, r.dismissed), (0, 1));
    }

    #[test]
    fn a_strong_box_stays_visible_at_a_rejected_spot() {
        let at = place("car", false);
        let m = bad_spot(at);
        let state = |prop| at_spot("gdino", &m, prop).0.boxes[0].state;
        assert_eq!(state(fused("car", at, 0.4999, None)), BoxState::Rejected);
        assert_eq!(state(fused("car", at, 0.5, None)), BoxState::Pending);
        // Likely: 0.25 with a yes, and just under it the spot wins.
        assert_eq!(state(fused("car", at, 0.25, YES)), BoxState::Pending);
        assert_eq!(state(fused("car", at, 0.2499, YES)), BoxState::Rejected);
        // A VLM no is no reason either way; an old seven-way yes is no likely box.
        assert_eq!(state(fused("car", at, 0.3, NO)), BoxState::Rejected);
        let old = Some(("vehicle", true));
        assert_eq!(state(fused("car", at, 0.3, old)), BoxState::Rejected);
        // A box the triage accepts at a lowered accept score.
        let mut sure = boxes(&[fused("car", at, 0.3, None)]).remove(0);
        sure.auto.triage = Some(Triage::Accept);
        assert!(spot_keeps(&sure, 0.3));
    }

    #[test]
    fn the_audit_sample_stays_visible_at_a_rejected_spot() {
        let at = place("car", true);
        let (f, r) = at_spot("gdino", &bad_spot(at), fused("car", at, 0.3, None));
        let b = &f.boxes[0];
        assert_eq!(b.state, BoxState::Pending);
        assert!(b.auto.audit && b.auto.why.iter().any(|w| w == AUDIT_WHY));
        assert_eq!((r.audit, r.spot_dismissed), (1, 0));
        // And it is no parked accept: a verdict on it checks the filter.
        let (f, r) = at_spot("gdino", &parked_spot(at), fused("car", at, 0.1, None));
        assert_eq!(
            (f.boxes[0].state, r.parked_accepted),
            (BoxState::Pending, 0)
        );
    }

    #[test]
    fn sam3_runs_use_no_spot() {
        let at = place("car", false);
        for m in [bad_spot(at), parked_spot(at)] {
            let (f, r) = at_spot(SAM3_KEY, &m, fused("car", at, 0.3, None));
            assert_eq!(f.boxes[0].state, BoxState::Pending);
            assert_eq!((r.spot_dismissed, r.parked_accepted), (0, 0));
            // A box SAM 3 found in another run's name.
            let sam_box = Proposal {
                sources: [(SAM3_KEY.to_string(), 0.3)].into(),
                ..fused("car", at, 0.3, None)
            };
            let (f, r) = at_spot("gdino", &m, sam_box);
            assert_eq!(f.boxes[0].state, BoxState::Pending);
            assert_eq!((r.spot_dismissed, r.parked_accepted), (0, 0));
        }
    }

    #[test]
    fn a_stored_box_is_never_dismissed_for_its_spot_and_a_reviewed_frame_takes_nothing() {
        let at = place("car", false);
        let m = bad_spot(at);
        let spots = FrameSpots {
            memory: &m,
            width: FRAME_SIZE.0,
            height: FRAME_SIZE.1,
        };
        // An older run's box, found again: only this run's boxes are sorted.
        let mut f = LabelFile {
            boxes: boxes(&[fused("car", at, 0.3, None)]),
            ..LabelFile::default()
        };
        let r = super::apply_labelers(
            &mut f,
            boxes(&[fused("car", at, 0.3, None)]),
            "gdino",
            REL,
            Some(spots),
        );
        assert_eq!((f.boxes[0].state, r.spot_dismissed), (BoxState::Pending, 0));
        // Reviewed, sent-back and skipped frames follow their own paths.
        for status in [Status::Reviewed, Status::Skipped] {
            let mut f = LabelFile {
                status,
                ..LabelFile::default()
            };
            let r = super::apply_labelers(
                &mut f,
                boxes(&[fused("car", at, 0.3, None)]),
                "gdino",
                REL,
                Some(spots),
            );
            assert_eq!(r.spot_dismissed, 0, "{status:?}");
        }
    }

    #[test]
    fn a_rerun_finds_a_spot_dismissed_box_again_and_decides_afresh() {
        let at = place("car", false);
        let m = bad_spot(at);
        let spots = FrameSpots {
            memory: &m,
            width: FRAME_SIZE.0,
            height: FRAME_SIZE.1,
        };
        let (mut f, _) = at_spot("gdino", &m, fused("car", at, 0.3, None));
        let run =
            |f: &mut LabelFile, p| super::apply_labelers(f, boxes(&[p]), "gdino", REL, Some(spots));
        // Same box, same answer; nothing doubles.
        run(&mut f, fused("car", at, 0.3, None));
        assert_eq!(f.boxes.len(), 1);
        assert!(f.boxes[0].is_dismissed());
        // Stronger now: it shows.
        run(&mut f, fused("car", at, 0.6, None));
        assert_eq!((f.boxes.len(), f.boxes[0].state), (1, BoxState::Pending));
    }

    #[test]
    fn a_spot_dismissal_is_no_judged_reject_and_no_decision_at_the_gate() {
        let dismissed = LabelBox {
            state: BoxState::Rejected,
            reason: Some(labels::bulk::SPOT_DISMISSED_REASON.into()),
            bbox: CAR,
            ..labels::tests_support::auto_box("car")
        };
        assert!(unjudged_reject(&dismissed));
        let mut judged = dismissed.clone();
        judged.reject_reason = Some(labels::RejectReason::Reflection);
        assert!(!unjudged_reject(&judged));
        // On a reviewed frame: a dog on it is no box on a decided car...
        let f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![dismissed],
            ..LabelFile::default()
        };
        let dog = boxes(&[sam("dog", CAR, 0.9)]).remove(0);
        assert_eq!(gate_drop(&f, &dog), None);
        // ...and a car on it stays out, as on a score dismissal.
        let car = boxes(&[sam("car", CAR, 0.9)]).remove(0);
        assert_eq!(gate_drop(&f, &car), Some(GateDrop::OnReject));
    }

    #[test]
    fn a_parked_spot_accepts_the_exact_label_and_keeps_the_evidence() {
        let at = place("car", false);
        let m = parked_spot(at);
        let (f, r) = at_spot("gdino", &m, fused("car", at, 0.3, None));
        let b = &f.boxes[0];
        assert_eq!(b.state, BoxState::Accepted);
        assert_eq!(b.reason.as_deref(), Some(PARKED_REASON));
        let mark = b.auto.parked.as_ref().unwrap();
        assert_eq!((mark.since.as_str(), mark.bbox), ("2026-10-01", at));
        assert!(b.auto.labelers.contains_key("gdino"), "the evidence stays");
        assert_eq!((r.parked_accepted, r.parked_new), (1, 1));
        assert_eq!(f.status, Status::Unreviewed, "the frame stays unreviewed");
        assert!(f.repeat_spot);
        assert!(!labels::needs_decision(b));
        assert!(labels::repeats::repeats_only(&f));
        // Another vehicle class at the place: not this label, and not hidden either.
        let (f, r) = at_spot("gdino", &m, fused("truck", at, 0.3, None));
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        assert_eq!((r.parked_accepted, f.repeat_spot), (0, false));
        // Below IoU 0.8.
        let [x0, y0, x1, y1] = at;
        let loose = [x0, y0, x1, y0 + (y1 - y0) * 0.75];
        let (f, r) = at_spot("gdino", &m, fused("car", loose, 0.3, None));
        assert_eq!(
            (f.boxes[0].state, r.parked_accepted),
            (BoxState::Pending, 0)
        );
    }

    #[test]
    fn a_vlm_no_blocks_a_parked_accept_and_a_bulk_save() {
        let at = place("car", false);
        let m = parked_spot(at);
        let (f, r) = at_spot("gdino", &m, fused("car", at, 0.6, NO));
        let b = &f.boxes[0];
        assert_eq!(b.state, BoxState::Pending);
        assert_eq!(b.auto.triage, Some(Triage::Review));
        assert!(b.auto.why.iter().any(|w| w == PARKED_GONE_WHY));
        assert!(labels::needs_decision(b), "flagged: a save asks");
        assert_eq!(r.parked_accepted, 0);
        // Only the yes/no contract counts: a seven-way answer or no check accepts.
        let old = Some(("vehicle", false));
        for vlm in [old, None, YES] {
            let (f, _) = at_spot("gdino", &m, fused("car", at, 0.6, vlm));
            assert_eq!(f.boxes[0].state, BoxState::Accepted, "{vlm:?}");
        }
        // The frame-level save has the same rule, for a box a spot edit accepted.
        let mut g = LabelFile {
            repeat_spot: true,
            boxes: boxes(&[fused("car", at, 0.6, NO)]),
            ..LabelFile::default()
        };
        g.boxes[0].state = BoxState::Accepted;
        assert!(!labels::repeats::repeats_only(&g));
        g.boxes[0].auto.vlm = None;
        assert!(labels::repeats::repeats_only(&g));
    }

    #[test]
    fn a_parked_spot_takes_a_stored_pending_box_and_shields_its_place() {
        let at = place("car", false);
        let m = parked_spot(at);
        let spots = FrameSpots {
            memory: &m,
            width: FRAME_SIZE.0,
            height: FRAME_SIZE.1,
        };
        let mut f = LabelFile {
            boxes: boxes(&[fused("car", at, 0.3, None)]),
            ..LabelFile::default()
        };
        let r = super::apply_labelers(&mut f, Vec::new(), "gdino", REL, Some(spots));
        assert_eq!((f.boxes[0].state, r.parked_new), (BoxState::Accepted, 0));
        // Rejected on two frames as well, say before the car was parked: the reviewer's later
        // word wins, also for another vehicle at the place, which stays visible.
        let d = (0..2)
            .map(|n| decision("2026-10-15", n, "car", at, BoxState::Rejected))
            .collect();
        let both = memory(d, m.store().clone());
        let (f, r) = at_spot("gdino", &both, fused("truck", at, 0.3, None));
        assert_eq!((f.boxes[0].state, r.spot_dismissed), (BoxState::Pending, 0));
    }

    /// The test behind the feature, in small: decisions on the early dates, boxes on the later ones.
    #[test]
    fn spots_learned_early_hide_later_false_boxes_and_a_real_box_survives() {
        let rail = place("car", false);
        let plant = [300.0, 40.0, 380.0, 120.0];
        let early: Vec<Decision> = (0..6)
            .map(|n| {
                let day = format!("2026-10-0{}", n / 2 + 1);
                let at = if n % 2 == 0 { rail } else { plant };
                decision(&day, n, "car", at, BoxState::Rejected)
            })
            .collect();
        let m = memory(early, CameraSpots::default());
        let later = |date: &str, p: Proposal| {
            let rel = format!("day/yard/{date}/20261020T000000000Z_x.jpg");
            let mut f = LabelFile::default();
            let spots = FrameSpots {
                memory: &m,
                width: FRAME_SIZE.0,
                height: FRAME_SIZE.1,
            };
            super::apply_labelers(&mut f, boxes(&[p]), "gdino", &rel, Some(spots));
            f.boxes.remove(0)
        };
        let mut shown = 0;
        for day in 10..=30 {
            let day = format!("2026-10-{day}");
            let day = day.as_str();
            for (at, score) in [(rail, 0.2), (plant, 0.31)] {
                let b = later(day, fused("car", at, score, None));
                if b.state == BoxState::Pending {
                    assert!(b.auto.audit, "only an audit sample stays");
                    shown += 1;
                } else {
                    assert_eq!(
                        b.reason.as_deref(),
                        Some(labels::bulk::SPOT_DISMISSED_REASON)
                    );
                }
            }
            // A real car at the bad spot on an unlucky frame.
            let real = later(day, fused("car", rail, 0.6, None));
            assert_eq!(real.state, BoxState::Pending);
            // And a real box somewhere the reviewer never rejected.
            let elsewhere = later(day, fused("person", OFF, 0.2, None));
            assert_eq!(elsewhere.state, BoxState::Pending);
        }
        assert!(shown < 5, "{shown} of 42 false boxes shown");
        // More than 30 days on, the spot has gone stale.
        let stale = later("2026-11-15", fused("car", rail, 0.2, None));
        assert_eq!(stale.state, BoxState::Pending);
    }
}
