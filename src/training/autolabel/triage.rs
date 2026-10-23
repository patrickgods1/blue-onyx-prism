//! The service's half of auto-labelling (pure): the proposals `scripts/autolabel.py` returns
//! ([`FrameProposals`]), the triage rule ([`triage`]), proposals to pending `source: auto` label
//! boxes ([`to_boxes`]) and their merge into a label file ([`apply`]: evidence on boxes it
//! confirms, then [`merge_proposals`]); and for the self-training loop, the disagreement of a
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
                },
            })
        })
        .collect()
}

/// Merge `boxes` into `f`: first, an existing box of the same label at IoU >= [`MATCH_IOU`]
/// without auto-labeler evidence gets the proposal's (its state and decision unchanged: a model
/// box the VLM calls a rock shows that on the Review page); then [`merge_proposals`] adds the
/// rest by its rule. Returns the merge report and how many boxes got evidence.
pub fn apply(f: &mut LabelFile, boxes: Vec<LabelBox>) -> (MergeReport, usize) {
    let mut annotated = 0;
    for p in &boxes {
        let best = f
            .boxes
            .iter_mut()
            .filter(|b| b.label == p.label && b.auto.is_empty())
            .map(|b| (iou(&b.bbox, &p.bbox), b))
            .filter(|(v, _)| *v >= MATCH_IOU)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, b)| b);
        if let Some(b) = best {
            b.auto = p.auto.clone();
            annotated += 1;
        }
    }
    (merge_proposals(f, boxes), annotated)
}

/// One detection of a trained model.
#[derive(Debug, Clone, PartialEq)]
pub struct Det {
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

/// How a trained model's detections disagree with a reviewed label file: a detection matching
/// no training box (accepted or added) at IoU >= [`MATCH_IOU`] is new, one whose best match has
/// another label is a class change, a training box no detection matches is missed. None when the
/// file is not reviewed (there is nothing to disagree with) or everything matches.
pub fn disagreement(previous: &LabelFile, dets: &[Det], model: &str) -> Option<String> {
    if previous.status != Status::Reviewed {
        return None;
    }
    let train: Vec<&LabelBox> = previous.training().collect();
    let (mut new, mut changed) = (0usize, 0usize);
    for d in dets {
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
    let why = disagreement(f, dets, model);
    let (rep, _) = apply(f, model_boxes(dets, model, classes, w, h));
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
        let (rep, annotated) = apply(&mut f, boxes);
        assert_eq!((rep.added, rep.dropped, annotated), (1, 1, 1));
        // The model's person box keeps its source and state, and shows the VLM's objection.
        assert_eq!(f.boxes[0].source, Source::Model);
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        assert_eq!(f.boxes[0].auto.vlm.as_ref().unwrap().answer, "statue");
        assert_eq!(f.boxes[1].label, "dog");
        assert_eq!(f.boxes[1].source, Source::Auto);
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
        assert_eq!(disagreement(&f, &same, "m"), None);
        // A class change, a miss and a new box.
        let dets = [
            d("cat", [200.0, 200.0, 261.0, 260.0]),
            d("raccoon", [300.0, 300.0, 340.0, 340.0]),
        ];
        assert_eq!(
            disagreement(&f, &dets, "night-yolo26n").as_deref(),
            Some("re-label by night-yolo26n: 1 new box, 1 missed, 1 class change")
        );
        // Unreviewed files have nothing to disagree with.
        let mut un = reviewed();
        un.status = Status::Unreviewed;
        assert_eq!(disagreement(&un, &dets, "m"), None);

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
