//! The service's half of auto-labelling (pure): the proposals `scripts/autolabel.py` returns
//! ([`FrameProposals`]), the triage rule ([`triage`]), proposals to pending `source: auto` label
//! boxes ([`to_boxes`]) and their merge into a label file ([`apply`]: evidence on boxes it
//! confirms, class suggestions on pending boxes it labels otherwise, then [`merge_proposals`]);
//! and for the self-training loop, the disagreement of a
//! trained model's detections with the reviewed labels ([`disagreement`]).

use super::Category;
use crate::collect::labels::{
    self, AutoInfo, BoxState, LabelBox, LabelFile, MATCH_IOU, MergeReport, Source, Status, Triage,
    VlmCheck, iou, merge_proposals,
};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Fused score at or above which a box both labelers found is accepted.
pub const ACCEPT_SCORE: f32 = 0.6;

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
    /// Best score per labeler (`gdino`, `rfdetr`).
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
/// `ir`), the VLM did not object and SigLIP ranks the box's own species first; else `review`
/// with the reasons.
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
    let cat = Category::of(&p.label);
    if matches!(scene, "night" | "ir") && matches!(cat, Category::Person | Category::Animal) {
        why.push(
            if cat == Category::Person {
                "a person at night: always reviewed"
            } else {
                "an animal at night: always reviewed"
            }
            .to_string(),
        );
    }
    if let Some(v) = &p.vlm
        && !v.ok
    {
        why.push(format!(
            "{} says {} (expected {})",
            if v.model.is_empty() {
                "the VLM"
            } else {
                &v.model
            },
            if v.answer.is_empty() {
                "nothing"
            } else {
                &v.answer
            },
            v.expected
        ));
    }
    if let Some((top, s)) = p
        .species_scores
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .filter(|(k, _)| !k.eq_ignore_ascii_case(&p.label))
    {
        why.push(format!("SigLIP ranks {top} first ({s:.2})"));
    }
    if why.is_empty() {
        (Triage::Accept, why)
    } else {
        (Triage::Review, why)
    }
}

fn round(v: f32, scale: f32) -> f32 {
    (v * scale).round() / scale
}

/// The pending `source: auto` boxes of `proposals` for a `w` x `h` frame of `scene`: labels
/// outside `classes` and boxes smaller than [`labels::MIN_BOX_PX`] after clamping are left out;
/// coordinates rounded to 0.1 px, scores to 4 decimals.
pub fn to_boxes(
    proposals: &[Proposal],
    classes: &[String],
    scene: &str,
    w: u32,
    h: u32,
    accept_score: f32,
) -> Vec<LabelBox> {
    proposals
        .iter()
        .filter(|p| classes.iter().any(|c| c == &p.label))
        .filter(|p| p.bbox.iter().all(|v| v.is_finite()) && p.score.is_finite())
        .filter_map(|p| {
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
                },
            })
        })
        .collect()
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

/// How a trained model's detections disagree with a reviewed label file, over the run's
/// `classes` only (case-insensitive): a detection of one of them matching no training box
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
    let dets: Vec<&Det> = dets.iter().filter(|d| in_run(&d.label)).collect();
    let (mut new, mut changed) = (0usize, 0usize);
    for d in &dets {
        let best = train
            .iter()
            .map(|b| (iou(&b.bbox, &d.bbox), *b))
            .filter(|(v, _)| *v >= MATCH_IOU)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            None => new += 1,
            Some((_, b)) if b.label != d.label => changed += 1,
            Some(_) => {}
        }
    }
    let missed = train
        .iter()
        .filter(|b| in_run(&b.label))
        .filter(|b| !dets.iter().any(|d| iou(&b.bbox, &d.bbox) >= MATCH_IOU))
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

/// The pending auto boxes of a trained model's detections (always `review`: a model is checked
/// against the reviewer, not trusted).
pub fn model_boxes(dets: &[Det], model: &str, classes: &[String], w: u32, h: u32) -> Vec<LabelBox> {
    let props: Vec<Proposal> = dets
        .iter()
        .map(|d| Proposal {
            label: d.label.clone(),
            bbox: d.bbox,
            score: d.score,
            sources: [(model.to_string(), d.score)].into(),
            agree: false,
            vlm: None,
            species_scores: BTreeMap::new(),
        })
        .collect();
    to_boxes(&props, classes, "day", w, h, ACCEPT_SCORE)
        .into_iter()
        .map(|mut b| {
            b.auto.agree = None;
            b.auto.triage = Some(Triage::Review);
            b.auto.why = vec![format!("re-label by {model}")];
            b.reason = Some(format!("model {model}"));
            b
        })
        .collect()
}

/// Apply a trained model's detections to a frame's label file: the disagreement note (reviewed
/// files only; the file goes back to `unreviewed`) and the detections as pending proposals.
/// Returns the merge report and the disagreement.
pub fn apply_relabel(
    f: &mut LabelFile,
    dets: &[Det],
    model: &str,
    classes: &[String],
    w: u32,
    h: u32,
) -> (MergeReport, Option<String>) {
    let why = disagreement(f, dets, model, classes);
    let rep = apply(f, model_boxes(dets, model, classes, w, h)).merge;
    if let Some(w) = &why {
        f.status = Status::Unreviewed;
        f.disagreement = Some(w.clone());
    }
    (rep, why)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

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
        let (rep, why) = apply_relabel(&mut g, &dets, "night-yolo26n", &classes(), 640, 480);
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
        let (rep, why) = apply_relabel(&mut h, &same, "m", &classes(), 640, 480);
        assert_eq!((rep.added, why), (0, None));
        assert_eq!(h.status, Status::Reviewed);
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
}
