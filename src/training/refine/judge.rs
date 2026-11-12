//! The judge of the dataset check: one reviewed frame's label file against what a fold model that
//! never saw its camera day detected there ([`judge`]). Pure. It proposes suggestions only
//! ([`crate::collect::labels::refine`]) and returns what the report counts (loose labels, which
//! training boxes the model found).
//!
//! The rules, in order (every constant below with its reason):
//!
//! 1. Candidates: detections of wanted classes at [`MATCH_MIN_SCORE`] or more, and the frame's
//!    training boxes that are no ignore region.
//! 2. One-to-one matching of same-object pairs ([`same_object`]) at IoU >= [`LOOSE_IOU_MIN`], best
//!    IoU first. Without it a loose label (IoU 0.1-0.3 to its detection) would yield an `add` on top
//!    of the object plus a `verify` on the label.
//! 3. A matched pair at IoU >= [`GOOD_IOU`], or on a box a reviewer drew, moved or relabelled: fine.
//!    Below: `loose`, counted for the report only (a later phase proposes the tighter box).
//! 4. `add`: an unmatched detection at [`ADD_SCORE`] or more, at least [`FLOOR_PX`] on its short side
//!    at model input, that no box of its object that is not rejected and no ignore box covers
//!    ([`covers`]), and no judged reject of its object lies on at IoU >= [`REJECT_BLOCK_IOU`].
//! 5. `verify`: an unmatched training box no reviewer drew or moved, with no same-object detection
//!    (at [`MATCH_MIN_SCORE`] or more, as every candidate) at IoU >= [`VERIFY_IOU`], of a class
//!    with [`MIN_CLASS_BOXES`] boxes or more in what the fold model trained on, at least
//!    [`FLOOR_PX`] on its short side at model input.
//! 6. `relabel`: a training box no reviewer drew or relabelled, on which a detection of another
//!    object (not [`same_object`]) at [`RELABEL_SCORE`] or more lies at IoU >= [`RELABEL_IOU`], and
//!    no same-object detection at [`RELABEL_BLOCK_SCORE`] or more at IoU >= [`VERIFY_IOU`]. Such a
//!    box gets no `verify` (the relabel says more), and the detection no `add` of its own (it is the
//!    evidence of the relabel, not a second object). A detection matched in step 2 is never relabel
//!    evidence: its own box explains it.
//! 7. Only what [`refine::live`] shows is returned, so the judge never stores a suggestion a
//!    reviewer would not see. Only reviewed frames are judged (a confirmed-empty one can get an
//!    `add`); skipped and unreviewed frames get nothing.
//!
//! "Short side at model input" is `min(w / W, h / H) * imgsz` ([`short_px`]), the plan's measure of
//! too small (the size floor of a later phase uses it too).

use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::refine::{self, Kind, Suggestion};
use crate::collect::labels::repeats::same_object;
use crate::collect::labels::{
    BoxState, LabelBox, LabelFile, Source, Status, clamp_box, covers, iou,
};
use crate::training::autolabel::triage::Det;
use std::collections::BTreeMap;

/// Detections below this score are not used at all (no candidate, no pair, not "seen"; the fold
/// models are run with it as their threshold). Measured: under 0.15 a fold model's guesses overlap
/// nearly every label, which silenced `verify` (2 of 2,755 training boxes on the first real run, at
/// 0.05, although the fold models missed 304 at 0.3). Labels whose best same-object detection
/// scores under 0.15 were about half label problems on a by-hand check; those at 0.15-0.3 almost
/// all real, dim objects.
pub const MATCH_MIN_SCORE: f32 = 0.15;
/// IoU from which a detection and a training box of one object are a pair: a loose label still
/// overlaps its object this much.
pub const LOOSE_IOU_MIN: f32 = 0.1;
/// IoU from which a matched pair is a good label; below it the label is `loose`.
pub const GOOD_IOU: f32 = 0.75;
/// Score from which an unmatched detection is proposed as a missing box: below it fold models
/// guess too often for a reviewer's time.
pub const ADD_SCORE: f32 = 0.5;
/// Short side at model input (pixels) under which neither an `add` nor a `verify` is proposed:
/// RF-DETR cannot see below about one 10-12 px patch.
pub const FLOOR_PX: f32 = 10.0;
/// IoU from which a same-object detection counts as having found a training box (no `verify`).
pub const VERIFY_IOU: f32 = 0.3;
/// Boxes of a class the fold model trained on from which it is expected to find the class at all:
/// rarer classes would make every box a `verify` (counted per fold: a dominant camera day held out
/// can leave its fold model few boxes of a class the whole selection has many of).
pub const MIN_CLASS_BOXES: usize = 50;
/// IoU from which another object's detection lies on a training box (`relabel`).
pub const RELABEL_IOU: f32 = 0.7;
/// Score from which another object's detection is evidence for a `relabel`.
pub const RELABEL_SCORE: f32 = 0.8;
/// A same-object detection at this score on the box means the model sees the labelled class too:
/// no `relabel`.
pub const RELABEL_BLOCK_SCORE: f32 = 0.3;
/// IoU from which a judged reject of its object blocks an `add` (the place a reviewer said no to;
/// [`crate::collect::labels::MATCH_IOU`]).
pub const REJECT_BLOCK_IOU: f32 = crate::collect::labels::MATCH_IOU;
/// Score from which a detection "finds" a training box in the report's recall: a production-like
/// threshold.
pub const FOUND_SCORE: f32 = 0.3;
/// IoU from which a detection finds a training box in the report's recall (the benchmark's AP50).
pub const FOUND_IOU: f32 = 0.5;

/// One frame to judge.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput<'a> {
    pub file: &'a LabelFile,
    /// The frame's size (its sidecar).
    pub width: u32,
    pub height: u32,
    /// The fold model's detections on it.
    pub dets: &'a [Det],
}

/// What every frame of a check shares.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    pub wanted: &'a Wanted<'a>,
    /// The fold models' input size.
    pub imgsz: u32,
    /// Training boxes per class (the list's spelling) of the frames the fold model trained on
    /// (the selection without the fold's frames: [`super::class_boxes`]), ignore regions left out.
    pub class_boxes: &'a BTreeMap<String, usize>,
    /// The check's id ([`Suggestion::run`]).
    pub run: &'a str,
}

/// A training box as the report's recall counts it.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxFact {
    /// The class (the list's spelling).
    pub class: String,
    /// Short side at model input ([`short_px`]).
    pub short_px: f32,
    /// A same-object detection at [`FOUND_SCORE`] found it at IoU >= [`FOUND_IOU`] (one to one).
    pub found: bool,
}

/// What [`judge`] made of a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Judged {
    /// The suggestions, every one [`refine::live`] on the file.
    pub suggestions: Vec<Suggestion>,
    /// Matched pairs below [`GOOD_IOU`] on boxes no reviewer drew, moved or relabelled.
    pub loose: usize,
    /// The frame's training boxes of wanted classes, for the recall.
    pub boxes: Vec<BoxFact>,
}

/// Finite coordinates of a box with a positive size.
fn proper(b: &[f32; 4]) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// The short side of `b` at model input: `min(w / W, h / H) * imgsz`. Infinite for a frame of
/// unknown size (no floor applies there).
pub fn short_px(b: &[f32; 4], width: u32, height: u32, imgsz: u32) -> f32 {
    if width == 0 || height == 0 {
        return f32::INFINITY;
    }
    let w = (b[2] - b[0]).max(0.0) / width as f32;
    let h = (b[3] - b[1]).max(0.0) / height as f32;
    w.min(h) * imgsz as f32
}

/// The object key of a label: the wanted class's spelling, else the label, lower case (for
/// [`same_object`]).
fn key_of(label: &str, wanted: &Wanted<'_>) -> String {
    wanted
        .class_of(label)
        .unwrap_or(label)
        .trim()
        .to_ascii_lowercase()
}

/// A detection of a wanted class.
#[derive(Debug)]
struct D {
    class: String,
    key: String,
    bbox: [f32; 4],
    score: f32,
}

/// A training box that is no ignore region.
#[derive(Debug)]
struct T<'a> {
    b: &'a LabelBox,
    /// Its wanted class (the list's spelling); None for an unwanted label.
    class: Option<String>,
    key: String,
}

/// Per training box: its matched detection and their IoU.
type ByBox = Vec<Option<(usize, f32)>>;
/// Per detection: its matched training box.
type ByDet = Vec<Option<usize>>;

/// Greedy one-to-one matching of same-object pairs of `dets` and `train` at IoU >= `min_iou`
/// (detections at `min_score` or more), best IoU first: `(box -> (detection, IoU), detection ->
/// box)`.
fn matching(dets: &[D], train: &[T<'_>], min_iou: f32, min_score: f32) -> (ByBox, ByDet) {
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (di, d) in dets.iter().enumerate() {
        if d.score < min_score {
            continue;
        }
        for (ti, t) in train.iter().enumerate() {
            if !same_object(&d.key, &t.key) {
                continue;
            }
            let v = iou(&d.bbox, &t.b.bbox);
            if v >= min_iou {
                pairs.push((v, di, ti));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut by_box = vec![None; train.len()];
    let mut by_det = vec![None; dets.len()];
    for (v, di, ti) in pairs {
        if by_det.get(di).copied().flatten().is_some()
            || by_box.get(ti).copied().flatten().is_some()
        {
            continue;
        }
        if let (Some(d), Some(t)) = (by_det.get_mut(di), by_box.get_mut(ti)) {
            *d = Some(ti);
            *t = Some((di, v));
        }
    }
    (by_box, by_det)
}

fn suggestion(kind: Kind, run: &str, why: String) -> Suggestion {
    Suggestion {
        kind,
        target: None,
        target_label: None,
        bbox: None,
        label: None,
        score: None,
        why,
        spot: None,
        answer: None,
        run: run.to_string(),
    }
}

/// Judge one frame (see the module docs).
pub fn judge(frame: &FrameInput<'_>, ctx: &Context<'_>) -> Judged {
    let f = frame.file;
    let mut out = Judged::default();
    if f.status != Status::Reviewed {
        return out;
    }
    let wanted = ctx.wanted;
    let size = |b: &[f32; 4]| short_px(b, frame.width, frame.height, ctx.imgsz);
    // 1. Candidates: scores within 0..=1 and boxes within the frame (a model's raw output can
    // stray past either, and a suggestion must stay within what a save accepts).
    let dets: Vec<D> = frame
        .dets
        .iter()
        .filter(|d| d.score.is_finite() && d.score >= MATCH_MIN_SCORE && proper(&d.bbox))
        .filter_map(|d| {
            let class = wanted.class_of(&d.label)?;
            let bbox = if frame.width > 0 && frame.height > 0 {
                clamp_box(d.bbox, frame.width, frame.height)
            } else {
                d.bbox
            };
            proper(&bbox).then(|| D {
                class: class.to_string(),
                key: class.trim().to_ascii_lowercase(),
                bbox,
                score: d.score.clamp(0.0, 1.0),
            })
        })
        .collect();
    let train: Vec<T<'_>> = f
        .boxes
        .iter()
        .filter(|b| b.is_training() && !b.is_ignore() && proper(&b.bbox))
        .map(|b| T {
            b,
            class: wanted.class_of(&b.label).map(str::to_string),
            key: key_of(&b.label, wanted),
        })
        .collect();
    // The report's recall: its own matching at a production-like score.
    let (found, _) = matching(&dets, &train, FOUND_IOU, FOUND_SCORE);
    for (t, hit) in train.iter().zip(&found) {
        if let Some(class) = &t.class {
            out.boxes.push(BoxFact {
                class: class.clone(),
                short_px: size(&t.b.bbox),
                found: hit.is_some(),
            });
        }
    }
    // 2. Pairs.
    let (by_box, by_det) = matching(&dets, &train, LOOSE_IOU_MIN, MATCH_MIN_SCORE);
    let mut found_list: Vec<Suggestion> = Vec::new();
    let mut evidence = vec![false; dets.len()];
    let mut relabelled = vec![false; train.len()];
    // 3. Loose labels.
    for (t, m) in train.iter().zip(&by_box) {
        let reviewer =
            t.b.source == Source::Human || t.b.orig_bbox.is_some() || t.b.model_label.is_some();
        if let Some((_, v)) = m
            && *v < GOOD_IOU
            && !reviewer
        {
            out.loose += 1;
        }
    }
    // 6. Another class (before the adds: its detection is no add).
    for (ti, t) in train.iter().enumerate() {
        let Some(class) = &t.class else { continue };
        if t.b.source == Source::Human || t.b.model_label.is_some() {
            continue;
        }
        let sees_it = dets.iter().any(|d| {
            same_object(&d.key, &t.key)
                && d.score >= RELABEL_BLOCK_SCORE
                && iou(&d.bbox, &t.b.bbox) >= VERIFY_IOU
        });
        if sees_it {
            continue;
        }
        let best = dets
            .iter()
            .enumerate()
            .filter(|(di, d)| {
                by_det.get(*di).copied().flatten().is_none()
                    && !same_object(&d.key, &t.key)
                    && d.score >= RELABEL_SCORE
                    && iou(&d.bbox, &t.b.bbox) >= RELABEL_IOU
            })
            .max_by(|a, b| a.1.score.total_cmp(&b.1.score).then(b.0.cmp(&a.0)));
        if let Some((di, d)) = best {
            if let Some(e) = evidence.get_mut(di) {
                *e = true;
            }
            if let Some(r) = relabelled.get_mut(ti) {
                *r = true;
            }
            found_list.push(Suggestion {
                target: Some(t.b.bbox),
                target_label: Some(t.b.label.clone()),
                label: Some(d.class.clone()),
                score: Some(d.score),
                ..suggestion(
                    Kind::Relabel,
                    ctx.run,
                    format!(
                        "a model that never saw this camera-day sees a {} here, score {:.2}, \
                         and no {class}",
                        d.class, d.score
                    ),
                )
            });
        }
    }
    // 4. Missing boxes.
    for (di, d) in dets.iter().enumerate() {
        if by_det.get(di).copied().flatten().is_some()
            || evidence.get(di).copied().unwrap_or(false)
            || d.score < ADD_SCORE
            || size(&d.bbox) < FLOOR_PX
        {
            continue;
        }
        let covered = f.boxes.iter().any(|b| {
            (b.state != BoxState::Rejected || b.is_ignore())
                && same_object(&key_of(&b.label, wanted), &d.key)
                && covers(&b.bbox, &d.bbox)
        });
        let blocked = f.boxes.iter().any(|b| {
            b.state == BoxState::Rejected
                && !b.is_ignore()
                && !crate::collect::labels::unjudged_reject(b)
                && same_object(&key_of(&b.label, wanted), &d.key)
                && iou(&b.bbox, &d.bbox) >= REJECT_BLOCK_IOU
        });
        if covered || blocked {
            continue;
        }
        found_list.push(Suggestion {
            bbox: Some(d.bbox),
            label: Some(d.class.clone()),
            score: Some(d.score),
            ..suggestion(
                Kind::Add,
                ctx.run,
                format!(
                    "a model that never saw this camera-day found a {} here, score {:.2}",
                    d.class, d.score
                ),
            )
        });
    }
    // 5. Labels no model found.
    for (ti, t) in train.iter().enumerate() {
        let Some(class) = &t.class else { continue };
        if by_box.get(ti).copied().flatten().is_some()
            || relabelled.get(ti).copied().unwrap_or(false)
            || t.b.source == Source::Human
            || t.b.orig_bbox.is_some()
            || ctx.class_boxes.get(class).copied().unwrap_or(0) < MIN_CLASS_BOXES
            || size(&t.b.bbox) < FLOOR_PX
        {
            continue;
        }
        let seen = dets
            .iter()
            .any(|d| same_object(&d.key, &t.key) && iou(&d.bbox, &t.b.bbox) >= VERIFY_IOU);
        if seen {
            continue;
        }
        found_list.push(Suggestion {
            target: Some(t.b.bbox),
            target_label: Some(t.b.label.clone()),
            ..suggestion(
                Kind::Verify,
                ctx.run,
                format!("no model that never saw this camera-day found this {class}"),
            )
        });
    }
    // 7. Only what a reviewer would see.
    let mut probe = f.clone();
    probe.refine = found_list;
    let live: Vec<usize> = refine::live(&probe, Some(wanted))
        .iter()
        .map(|l| l.index)
        .collect();
    out.suggestions = probe
        .refine
        .into_iter()
        .enumerate()
        .filter(|(i, _)| live.contains(i))
        .map(|(_, s)| s)
        .collect();
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::labels::{AutoInfo, Origin, RejectReason, bulk};

    fn classes() -> Vec<String> {
        ["person", "car", "truck", "dog"].map(String::from).to_vec()
    }

    fn many() -> BTreeMap<String, usize> {
        classes().into_iter().map(|c| (c, 500)).collect()
    }

    fn b(label: &str, bbox: [f32; 4], state: BoxState) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox,
            source: Source::Auto,
            score: Some(0.7),
            state,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        }
    }

    fn acc(label: &str, bbox: [f32; 4]) -> LabelBox {
        b(label, bbox, BoxState::Accepted)
    }

    fn det(label: &str, bbox: [f32; 4], score: f32) -> Det {
        Det {
            label: label.into(),
            score,
            bbox,
        }
    }

    fn reviewed(boxes: Vec<LabelBox>) -> LabelFile {
        LabelFile {
            status: Status::Reviewed,
            boxes,
            ..LabelFile::default()
        }
    }

    /// Judge `f` (640 x 480, input 640) with `dets`; with `counts` per class.
    fn run_with(f: &LabelFile, dets: &[Det], counts: &BTreeMap<String, usize>) -> Judged {
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let ctx = Context {
            wanted: &w,
            imgsz: 640,
            class_boxes: counts,
            run: "r1",
        };
        let j = judge(
            &FrameInput {
                file: f,
                width: 640,
                height: 480,
                dets,
            },
            &ctx,
        );
        // Everything returned is live (rule 7), whatever the case.
        let mut probe = f.clone();
        probe.refine = j.suggestions.clone();
        assert_eq!(
            refine::live(&probe, Some(&w)).len(),
            j.suggestions.len(),
            "{j:?}"
        );
        assert!(j.suggestions.iter().all(|s| s.run == "r1"));
        j
    }

    fn run(f: &LabelFile, dets: &[Det]) -> Judged {
        run_with(f, dets, &many())
    }

    fn kinds(j: &Judged) -> Vec<Kind> {
        j.suggestions.iter().map(|s| s.kind).collect()
    }

    const CAR: [f32; 4] = [100.0, 100.0, 200.0, 180.0];
    /// IoU with [`CAR`] about 0.92.
    const NEAR: [f32; 4] = [102.0, 101.0, 201.0, 182.0];
    const FAR: [f32; 4] = [400.0, 300.0, 460.0, 400.0];
    /// IoU with [`CAR`] = 0.5 (a loose label of it).
    const LOOSE: [f32; 4] = [100.0, 100.0, 300.0, 180.0];

    #[test]
    fn a_good_match_gives_nothing_and_counts_as_found() {
        let f = reviewed(vec![acc("car", CAR)]);
        let j = run(&f, &[det("car", NEAR, 0.9)]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert_eq!(j.loose, 0);
        assert_eq!(j.boxes.len(), 1);
        assert!(j.boxes[0].found);
        // A truck detection on a car box: one object, found too.
        let j = run(&f, &[det("truck", NEAR, 0.9)]);
        assert!(j.suggestions.is_empty() && j.boxes[0].found);
        // Below the production threshold but above the noise floor: not found, but no verify
        // either (the model saw it): MATCH_MIN_SCORE <= 0.2 < FOUND_SCORE.
        let j = run(&f, &[det("car", NEAR, 0.2)]);
        assert!(j.suggestions.is_empty() && !j.boxes[0].found);
    }

    #[test]
    fn a_loose_label_is_loose_and_neither_an_add_nor_a_verify() {
        assert!((iou(&CAR, &LOOSE) - 0.5).abs() < 0.01);
        let wide = [100.0, 100.0, 350.0, 230.0];
        let v = iou(&wide, &CAR);
        assert!(v > LOOSE_IOU_MIN && v < VERIFY_IOU, "{v}");
        let f = reviewed(vec![acc("car", wide)]);
        let j = run(&f, &[det("car", CAR, 0.95)]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert_eq!(j.loose, 1);
        // Between 0.3 and 0.75 too.
        let f = reviewed(vec![acc("car", LOOSE)]);
        let j = run(&f, &[det("car", CAR, 0.95)]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert_eq!(j.loose, 1);
        // A reviewer's box is never loose.
        for edit in [
            |b: &mut LabelBox| b.source = Source::Human,
            |b: &mut LabelBox| b.orig_bbox = Some([0.0, 0.0, 9.0, 9.0]),
            |b: &mut LabelBox| b.model_label = Some("truck".into()),
        ] {
            let mut f = reviewed(vec![acc("car", LOOSE)]);
            edit(&mut f.boxes[0]);
            let j = run(&f, &[det("car", CAR, 0.95)]);
            assert_eq!((j.loose, j.suggestions.len()), (0, 0), "{f:?}");
        }
    }

    #[test]
    fn an_unmatched_confident_detection_is_an_add() {
        let f = reviewed(vec![acc("car", CAR)]);
        let j = run(&f, &[det("car", NEAR, 0.9), det("person", FAR, 0.82)]);
        assert_eq!(kinds(&j), [Kind::Add]);
        let s = &j.suggestions[0];
        assert_eq!(
            (s.bbox, s.label.as_deref(), s.score),
            (Some(FAR), Some("person"), Some(0.82))
        );
        assert_eq!(
            s.why,
            "a model that never saw this camera-day found a person here, score 0.82"
        );
        // Too weak, or of an unwanted class: nothing.
        let car = det("car", NEAR, 0.9);
        assert!(
            run(&f, &[car.clone(), det("person", FAR, 0.49)])
                .suggestions
                .is_empty()
        );
        assert!(
            run(&f, &[car, det("zebra", FAR, 0.99)])
                .suggestions
                .is_empty()
        );
        // A confirmed-empty frame can get one.
        let mut empty = reviewed(vec![]);
        empty.empty_confirmed = true;
        assert_eq!(kinds(&run(&empty, &[det("dog", FAR, 0.7)])), [Kind::Add]);
    }

    #[test]
    fn an_add_is_not_proposed_where_a_box_or_a_judged_reject_is() {
        // Covered at IoU 0.3-0.5 by a pending box of its object (no training box to match).
        let half = [100.0, 100.0, 320.0, 180.0];
        let v = iou(&half, &CAR);
        assert!((0.3..0.5).contains(&v), "{v}");
        let pending = reviewed(vec![b("car", half, BoxState::Pending)]);
        assert!(
            run(&pending, &[det("car", CAR, 0.9)])
                .suggestions
                .is_empty()
        );
        // An ignore box: too small, or unsure.
        let small = LabelBox {
            reject_reason: Some(RejectReason::TooSmall),
            ..b("car", CAR, BoxState::Rejected)
        };
        assert!(
            run(&reviewed(vec![small]), &[det("car", NEAR, 0.9)])
                .suggestions
                .is_empty()
        );
        let unsure = LabelBox {
            difficult: true,
            ..acc("car", CAR)
        };
        let j = run(&reviewed(vec![unsure]), &[det("car", NEAR, 0.9)]);
        assert!(j.suggestions.is_empty() && j.boxes.is_empty(), "{j:?}");
        // A judged reject at the place.
        let fp = LabelBox {
            reject_reason: Some(RejectReason::FalsePositive),
            ..b("car", CAR, BoxState::Rejected)
        };
        assert!(
            run(&reviewed(vec![fp]), &[det("truck", NEAR, 0.9)])
                .suggestions
                .is_empty()
        );
        // An unjudged reject does not block: the add replaces it on accept.
        let low = LabelBox {
            reason: Some(bulk::LOW_SCORE_REASON.into()),
            ..b("car", CAR, BoxState::Rejected)
        };
        assert!(crate::collect::labels::unjudged_reject(&low));
        assert_eq!(
            kinds(&run(&reviewed(vec![low]), &[det("car", NEAR, 0.9)])),
            [Kind::Add]
        );
        let hint = LabelBox {
            source: Source::Model,
            origin: Some(Origin::BelowThreshold),
            ..b("car", CAR, BoxState::Rejected)
        };
        assert_eq!(
            kinds(&run(&reviewed(vec![hint]), &[det("car", NEAR, 0.9)])),
            [Kind::Add]
        );
        // A box of another object covers nothing.
        assert_eq!(
            kinds(&run(
                &reviewed(vec![acc("dog", CAR)]),
                &[det("car", NEAR, 0.6)]
            )),
            [Kind::Add, Kind::Verify]
        );
    }

    #[test]
    fn small_detections_and_boxes_give_nothing() {
        // 640 x 480 at input 640: 9 px wide = 9 px at input.
        let tiny = [10.0, 10.0, 19.0, 60.0];
        assert!(short_px(&tiny, 640, 480, 640) < FLOOR_PX);
        let ok = [10.0, 10.0, 20.0, 60.0];
        assert!((short_px(&ok, 640, 480, 640) - 10.0).abs() < 1e-4);
        let f = reviewed(vec![]);
        assert!(run(&f, &[det("person", tiny, 0.99)]).suggestions.is_empty());
        assert_eq!(kinds(&run(&f, &[det("person", ok, 0.99)])), [Kind::Add]);
        // No detection on a tiny box: no verify, but it counts (not found) in the recall.
        let j = run(&reviewed(vec![acc("person", tiny)]), &[]);
        assert!(j.suggestions.is_empty());
        assert_eq!(j.boxes.len(), 1);
        assert!(!j.boxes[0].found);
        assert_eq!(
            kinds(&run(&reviewed(vec![acc("person", ok)]), &[])),
            [Kind::Verify]
        );
        // A frame of unknown size: no floor.
        assert_eq!(short_px(&tiny, 0, 480, 640), f32::INFINITY);
    }

    #[test]
    fn a_label_no_model_found_is_a_verify_unless_the_class_is_rare() {
        let f = reviewed(vec![acc("car", CAR)]);
        let j = run(&f, &[]);
        assert_eq!(kinds(&j), [Kind::Verify]);
        assert_eq!(j.suggestions[0].target, Some(CAR));
        assert_eq!(
            j.suggestions[0].why,
            "no model that never saw this camera-day found this car"
        );
        assert_eq!(j.suggestions[0].score, None);
        // A detection elsewhere, or of another object, does not find it.
        assert_eq!(
            kinds(&run(&f, &[det("car", FAR, 0.9)])),
            [Kind::Add, Kind::Verify]
        );
        // A faint same-object detection at IoU >= 0.3 does, from the noise floor on.
        assert!(
            run(&f, &[det("truck", NEAR, MATCH_MIN_SCORE)])
                .suggestions
                .is_empty()
        );
        // A rare class: no verify.
        let mut counts = many();
        counts.insert("car".into(), MIN_CLASS_BOXES - 1);
        assert!(run_with(&f, &[], &counts).suggestions.is_empty());
        counts.insert("car".into(), MIN_CLASS_BOXES);
        assert_eq!(kinds(&run_with(&f, &[], &counts)), [Kind::Verify]);
    }

    #[test]
    fn detections_under_the_noise_floor_do_not_silence_a_verify() {
        // A training box with only a same-object detection on it, IoU about 0.92.
        assert!(iou(&CAR, &NEAR) > 0.9);
        let f = reviewed(vec![acc("car", CAR)]);
        let cases: [(f32, &[Kind]); 5] = [
            (0.05, &[Kind::Verify]),
            (0.10, &[Kind::Verify]),
            (0.149, &[Kind::Verify]),
            (MATCH_MIN_SCORE, &[]),
            (0.20, &[]),
        ];
        for (score, want) in cases {
            let j = run(&f, &[det("car", NEAR, score)]);
            assert_eq!(kinds(&j), want, "score {score}: {j:?}");
            // Not a pair below the floor, a good one above: never loose, never found at 0.3.
            assert_eq!(j.loose, 0, "score {score}");
            assert!(!j.boxes[0].found, "score {score}");
        }
        // A loose pair at IoU 0.2 with a confident detection that the label does not cover:
        // loose, and neither an add (the detection is the label's pair) nor a verify.
        let big = [100.0, 100.0, 300.0, 300.0];
        assert!((iou(&CAR, &big) - 0.2).abs() < 1e-6);
        assert!(!covers(&CAR, &big), "only the pairing keeps the add away");
        let j = run(&f, &[det("car", big, 0.6)]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert_eq!(j.loose, 1);
        // The same detection under the floor is no pair: the label is a verify, and the
        // detection, too weak for an add, nothing.
        let j = run(&f, &[det("car", big, 0.1)]);
        assert_eq!(kinds(&j), [Kind::Verify]);
        assert_eq!(j.loose, 0);
    }

    #[test]
    fn a_reviewers_boxes_get_nothing() {
        let other = det("dog", NEAR, 0.95);
        for (edit, why) in [
            (
                (|b: &mut LabelBox| {
                    b.source = Source::Human;
                    b.state = BoxState::Added;
                }) as fn(&mut LabelBox),
                "drawn",
            ),
            (
                |b: &mut LabelBox| b.orig_bbox = Some([1.0, 1.0, 50.0, 50.0]),
                "moved",
            ),
            (
                |b: &mut LabelBox| b.model_label = Some("truck".into()),
                "relabelled",
            ),
        ] {
            let mut f = reviewed(vec![acc("car", CAR)]);
            edit(&mut f.boxes[0]);
            let none = run(&f, &[]);
            let dog = run(&f, std::slice::from_ref(&other));
            match why {
                // Drawn: no verify, no relabel.
                "drawn" => {
                    assert!(none.suggestions.is_empty(), "{why}");
                    assert_eq!(kinds(&dog), [Kind::Add], "{why}: the dog is another object");
                }
                // Moved: no verify; another class is still a question.
                "moved" => {
                    assert!(none.suggestions.is_empty(), "{why}");
                    assert_eq!(kinds(&dog), [Kind::Relabel], "{why}");
                }
                // Relabelled: no relabel; a verify is still a question.
                _ => {
                    assert_eq!(kinds(&none), [Kind::Verify], "{why}");
                    assert_eq!(kinds(&dog), [Kind::Add, Kind::Verify], "{why}");
                }
            }
        }
    }

    #[test]
    fn relabel_needs_a_confident_other_object_and_never_between_vehicles() {
        let f = reviewed(vec![acc("car", CAR)]);
        let j = run(&f, &[det("person", NEAR, 0.85)]);
        assert_eq!(kinds(&j), [Kind::Relabel], "no verify and no add beside it");
        let s = &j.suggestions[0];
        assert_eq!(
            (s.target, s.label.as_deref(), s.score),
            (Some(CAR), Some("person"), Some(0.85))
        );
        // Two vehicle classes are one object: a truck on a car is a match.
        assert!(run(&f, &[det("truck", NEAR, 0.99)]).suggestions.is_empty());
        let truck = reviewed(vec![acc("truck", CAR)]);
        assert!(
            run(&truck, &[det("car", NEAR, 0.99)])
                .suggestions
                .is_empty()
        );
        // Too weak or too far off: no relabel (then the box is a verify and the find an add).
        assert_eq!(
            kinds(&run(&f, &[det("person", NEAR, 0.79)])),
            [Kind::Add, Kind::Verify]
        );
        let off = [130.0, 100.0, 230.0, 180.0];
        assert!(iou(&off, &CAR) < RELABEL_IOU);
        assert_eq!(
            kinds(&run(&f, &[det("person", off, 0.95)])),
            [Kind::Add, Kind::Verify]
        );
        // The model also sees the car: no relabel; the person is then another object there.
        assert_eq!(
            kinds(&run(&f, &[det("person", NEAR, 0.95), det("car", CAR, 0.4)])),
            [Kind::Add]
        );
    }

    #[test]
    fn unreviewed_and_skipped_frames_are_not_judged() {
        for status in [Status::Unreviewed, Status::Skipped] {
            let f = LabelFile {
                status,
                ..reviewed(vec![acc("car", CAR)])
            };
            let j = run(&f, &[det("person", FAR, 0.9)]);
            assert_eq!(j, Judged::default(), "{status:?}");
        }
    }

    #[test]
    fn scores_and_boxes_out_of_range_are_clamped_so_a_save_accepts_them() {
        // 640 x 480: a box past the right and bottom edges, a score over 1.
        let f = reviewed(vec![]);
        let j = run(&f, &[det("person", [600.0, 400.0, 700.0, 520.0], 1.2)]);
        assert_eq!(kinds(&j), [Kind::Add]);
        let s = &j.suggestions[0];
        assert_eq!(s.bbox, Some([600.0, 400.0, 640.0, 480.0]));
        assert_eq!(s.score, Some(1.0));
        assert!(s.why.contains("score 1.00"), "{}", s.why);
        refine::check(&j.suggestions).unwrap();
        // A detection wholly outside the frame is no box once clamped: nothing.
        assert!(
            run(&f, &[det("person", [650.0, 10.0, 700.0, 90.0], 0.9)])
                .suggestions
                .is_empty()
        );
        // A frame of unknown size: the box as it came.
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let counts = many();
        let ctx = Context {
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
        };
        let dets = [det("person", [600.0, 400.0, 700.0, 520.0], 0.9)];
        let j = judge(
            &FrameInput {
                file: &f,
                width: 0,
                height: 0,
                dets: &dets,
            },
            &ctx,
        );
        assert_eq!(j.suggestions[0].bbox, Some([600.0, 400.0, 700.0, 520.0]));
    }

    #[test]
    fn two_labels_at_one_place_each_get_their_own_suggestion() {
        // A car and a dog labelled with the very same box, no model finds either.
        let f = reviewed(vec![acc("car", CAR), acc("dog", CAR)]);
        let j = run(&f, &[]);
        assert_eq!(kinds(&j), [Kind::Verify, Kind::Verify]);
        let labels: Vec<Option<&str>> = j
            .suggestions
            .iter()
            .map(|s| s.target_label.as_deref())
            .collect();
        assert_eq!(labels, [Some("car"), Some("dog")]);
        // Written, both stay and name their own box.
        let mut g = f.clone();
        let a = refine::apply(
            &mut g,
            j.suggestions,
            &[Kind::Add, Kind::Verify, Kind::Relabel],
        );
        assert_eq!(a.written, 2);
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let targets: Vec<Option<usize>> = refine::live(&g, Some(&w))
            .iter()
            .map(|l| l.target)
            .collect();
        assert_eq!(targets, [Some(0), Some(1)]);
        // A relabel names its box too: a person seen on the dog's box only.
        let f = reviewed(vec![acc("dog", CAR)]);
        let j = run(&f, &[det("person", NEAR, 0.9)]);
        assert_eq!(kinds(&j), [Kind::Relabel]);
        assert_eq!(j.suggestions[0].target_label.as_deref(), Some("dog"));
    }

    #[test]
    fn hidden_classes_are_not_judged() {
        let c = classes();
        let hidden = ["dog".to_string()];
        let w = Wanted {
            classes: &c,
            hidden: &hidden,
        };
        let counts = many();
        let ctx = Context {
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
        };
        let f = reviewed(vec![acc("dog", CAR)]);
        let dets = [det("dog", FAR, 0.9)];
        let j = judge(
            &FrameInput {
                file: &f,
                width: 640,
                height: 480,
                dets: &dets,
            },
            &ctx,
        );
        assert_eq!(j, Judged::default());
    }
}
