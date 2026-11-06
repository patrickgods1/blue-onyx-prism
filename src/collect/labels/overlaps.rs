//! Suggestions for overlapping boxes ([`stacks::overlaps`]: nested or partly overlapping pairs no
//! rule may settle). Pure. A blind human test failed the automatic rule (13 of 99 drops removed a
//! real object) while it named the right box on 86 of 99, so a suggestion is only that: nothing here
//! changes a box's state, a frame's status, `reviewed_at` or `overlaps_checked`. [`apply`] only
//! writes [`LabelFile::overlap_suggestions`]; a box changes when a reviewer accepts a suggestion on
//! the Review tab.
//!
//! The signals, ported from `scripts/research/eval_overlaps.py` (the reference that reproduced the
//! measured numbers; `tests/data/overlaps_parity.json` holds its answers on synthetic pairs):
//!
//! - [`witnesses`]: the independent detectors that found a box, counted once each;
//! - the zoom check: RF-DETR L on a crop of the pair's union, matched one-to-one to the two boxes by
//!   [`match_zoom`], in frame coordinates ([`ZOOM_MATCH_IOU`], [`ZOOM_SCORE`]).
//!
//! [`suggest`] combines them. The research table, then the fallback the assist mode adds:
//!
//! | Zoom sees | Witnesses | Suggestion |
//! |---|---|---|
//! | both boxes | any | [`Verdict::TwoObjects`] |
//! | only X | X has more, or a tie | drop the other |
//! | only X | the other has more | [`Verdict::Unclear`] |
//! | neither, or it cannot look | one box has more | drop the other, `low_confidence` |
//! | neither, or it cannot look | a tie | [`Verdict::Unclear`] |
//!
//! "It cannot look": a class outside RF-DETR's label space ([`no_zoom`]) or an unreadable frame.
//! [`Verdict::Unclear`] records that a pair was looked at, so a rerun resumes after it; the page
//! shows nothing for it.
//!
//! **Guards** ([`guarded`]): a drop never takes a drawn box, a box a reviewer moved (`orig_bbox`) or
//! one a reviewer relabelled; such a pair is [`Verdict::Unclear`]. `model_label` alone is not a
//! reviewer's mark: an auto-label run and the pre-fill also set it when they map a spelling onto the
//! class list (`motorbike` on a `motorcycle` box, the guard the research report found firing wrongly).
//! A label that names the box's own class under another spelling ([`super::map_class`]) therefore does
//! not guard; one of another class (`truck` on a `car` box), or of no class of the list, does.
//!
//! **Keys.** A suggestion names its two boxes by `bbox`. A suggestion whose boxes moved or went, or
//! whose pair stopped being open, is stale and ignored ([`live`]), never an error. A frame holds at
//! most [`MAX_SUGGESTIONS`], strongest first ([`strength`]).
//!
//! Twin of the Review tab's `liveSuggestions` and `acceptOverlap` (`assets/review.js`), which
//! [`live`] and [`accept`] state as Rust: keep them in step. The page cannot call [`super::map_class`],
//! so `GET /v1/review/labels` names the live suggestions (`overlap_live`) and the page only adds what
//! changes while the frame is open (a move, a relabel, a rejected box).

use super::bulk::Wanted;
use super::repeats::same_object;
use super::stacks::{self, STACK_IOU};
use super::{BoxState, LabelBox, LabelFile, Source, Status};
use crate::training::autolabel::catalog::coco_class;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The zoom's detection must overlap a box this much to have seen it (the measured operating point).
pub const ZOOM_MATCH_IOU: f64 = 0.6;
/// Lowest zoom detection score that counts as seeing a box (the measured operating point).
pub const ZOOM_SCORE: f64 = 0.5;
/// Lowest detection score the zoom check keeps: under [`ZOOM_SCORE`] a detection cannot see a box,
/// but it still fills `other_score`.
pub const ZOOM_FLOOR: f64 = 0.1;
/// Crop padding, a share of the union's size on every side.
pub const ZOOM_PAD: f64 = 0.2;
/// An `other_score` at or above this is the weak band: 6 of 17 such drops removed a real object.
pub const WEAK_OTHER_SCORE: f64 = 0.3;
/// Lower edges of the `other_score` bands of a strong drop, weakest detection first (judged failures:
/// none 2 of 31, under 0.2 3 of 35, under 0.3 2 of 17).
const OTHER_BANDS: [f64; 2] = [0.1, 0.2];
/// Most suggestions of one frame (the most open pairs measured on one frame is 29).
pub const MAX_SUGGESTIONS: usize = 64;
/// Longest [`Suggestion::why`], in characters.
pub const MAX_WHY_LEN: usize = 300;

/// What a suggestion proposes for its pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Reject box `a` as a duplicate of box `b`.
    DropA,
    /// Reject box `b` as a duplicate of box `a`.
    DropB,
    /// One real object each: keep both.
    TwoObjects,
    /// Looked at, no suggestion: the reviewer's call.
    Unclear,
    /// A verdict this build does not know (a newer build's): ignored, the file still loads.
    #[serde(other)]
    Unknown,
}

/// A suggestion for one open pair (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    /// The two boxes, lower bbox first: the key.
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub verdict: Verdict,
    /// Why, in plain words.
    #[serde(default)]
    pub why: String,
    /// A drop: the zoom's score on the kept box. Two objects: its score on box `a`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kept_score: Option<f32>,
    /// A drop: the best other detection on the dropped box, 0 if none. Two objects: the zoom's score
    /// on box `b`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_score: Option<f32>,
    /// The witnesses alone: the zoom gave no answer.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub low_confidence: bool,
    /// A drop with `other_score` at or above [`WEAK_OTHER_SCORE`]: the page warns.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub weak: bool,
}

/// One box of a pair as [`suggest`] sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Side {
    pub bbox: [f32; 4],
    /// The list's spelling of its label.
    pub class: String,
    /// A drop may not take it ([`guarded`]).
    pub guarded: bool,
}

/// An open pair, lower bbox first (`a`).
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub a: Side,
    pub b: Side,
}

impl Pair {
    /// The box the zoom crops around.
    pub fn union(&self) -> [f32; 4] {
        let (a, b) = (&self.a.bbox, &self.b.bbox);
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }

    fn is(&self, s: &Suggestion) -> bool {
        self.a.bbox == s.a && self.b.bbox == s.b
    }
}

/// An open pair of a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Open {
    pub pair: Pair,
    /// [`witnesses`] of `a` and `b`.
    pub witnesses: [Vec<String>; 2],
}

/// A detection of the zoom check, in frame coordinates.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Det {
    pub label: String,
    pub bbox: [f64; 4],
    pub score: f64,
}

/// What the zoom check has for a pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Zoom<'a> {
    /// It could not look: a class outside its label space, an unreadable frame.
    Unavailable,
    /// Its detections on the pair's crop.
    Looked(&'a [Det]),
}

/// A box coordinate as the label file holds it: the decimal text f32 shortest formatting writes, read
/// as f64. That is what the script and the research reference compare, so a threshold falls alike.
fn wide(b: &[f32; 4]) -> [f64; 4] {
    b.map(|v| v.to_string().parse().unwrap_or_else(|_| f64::from(v)))
}

/// Intersection over union in f64 (the script's `iou`): 0 when the boxes do not overlap.
fn iou64(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let iw = a[2].min(b[2]) - a[0].max(b[0]);
    let ih = a[3].min(b[3]) - a[1].max(b[1]);
    if iw <= 0.0 || ih <= 0.0 {
        return 0.0;
    }
    let inter = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The independent detectors that found box `i`: over the frame's boxes in any state, box `i` and
/// those of one object at IoU >= [`STACK_IOU`], the keys of `labelers`, plus `serving` for a
/// `source: model` box. Sorted, each once.
pub fn witnesses(boxes: &[LabelBox], i: usize) -> Vec<String> {
    let Some(b) = boxes.get(i) else {
        return Vec::new();
    };
    let mut out = BTreeSet::new();
    for (k, o) in boxes.iter().enumerate() {
        if k == i || (same_object(&b.label, &o.label) && super::iou(&b.bbox, &o.bbox) >= STACK_IOU)
        {
            out.extend(o.auto.labelers.keys().cloned());
            if o.source == Source::Model {
                out.insert("serving".to_string());
            }
        }
    }
    out.into_iter().collect()
}

/// A reviewer changed the class of `b`: `model_label` names another class than the box's own, or no
/// class of `classes` (see the module docs on a mapped spelling).
fn reviewer_relabelled(b: &LabelBox, classes: &[String]) -> bool {
    b.model_label.as_deref().is_some_and(|m| {
        !super::map_class(m, classes).is_some_and(|c| c.eq_ignore_ascii_case(b.label.trim()))
    })
}

/// Whether a drop may not take `b`: drawn, moved by a reviewer, or relabelled by one.
pub fn guarded(b: &LabelBox, classes: &[String]) -> bool {
    b.source == Source::Human || b.orig_bbox.is_some() || reviewer_relabelled(b, classes)
}

/// `true` when `x` comes before `y` by bbox, so a pair has one order whatever the file's.
fn before(x: &LabelBox, y: &LabelBox) -> bool {
    let first = x
        .bbox
        .iter()
        .zip(&y.bbox)
        .map(|(p, q)| p.total_cmp(q))
        .find(|o| o.is_ne());
    first.is_none_or(std::cmp::Ordering::is_lt)
}

/// The open pairs of `f` ([`stacks::overlaps`]), each lower bbox first with its witnesses.
pub fn open_pairs(f: &LabelFile, wanted: &Wanted<'_>) -> Vec<Open> {
    let side = |b: &LabelBox| Side {
        bbox: b.bbox,
        class: wanted.class_of(&b.label).unwrap_or(&b.label).to_string(),
        guarded: guarded(b, wanted.classes),
    };
    stacks::overlaps(&f.boxes, wanted)
        .into_iter()
        .filter_map(|(i, j)| {
            let (x, y) = (f.boxes.get(i)?, f.boxes.get(j)?);
            let (ia, ib) = if before(x, y) { (i, j) } else { (j, i) };
            Some(Open {
                pair: Pair {
                    a: side(f.boxes.get(ia)?),
                    b: side(f.boxes.get(ib)?),
                },
                witnesses: [witnesses(&f.boxes, ia), witnesses(&f.boxes, ib)],
            })
        })
        .collect()
}

/// Whether RF-DETR's label space lacks one of the pair's classes: it cannot look.
pub fn no_zoom(pair: &Pair) -> bool {
    coco_class(&pair.a.class).is_none() || coco_class(&pair.b.class).is_none()
}

/// Which detection of `dets` saw each box of the pair: `[a, b]`, indices into `dets`. One detection
/// per box, of one object, score at least `threshold`, IoU at least `min_iou`, best IoU first; a tie of
/// IoU goes to the higher score, then the earlier detection, then box A.
pub fn match_zoom(dets: &[Det], pair: &Pair, min_iou: f64, threshold: f64) -> [Option<usize>; 2] {
    let boxes = [wide(&pair.a.bbox), wide(&pair.b.bbox)];
    let classes = [&pair.a.class, &pair.b.class];
    let mut cand = Vec::new();
    for (i, d) in dets.iter().enumerate() {
        if d.score < threshold {
            continue;
        }
        for side in 0..2 {
            let (Some(class), Some(bbox)) = (classes.get(side), boxes.get(side)) else {
                continue;
            };
            if same_object(&d.label, class) {
                let v = iou64(&d.bbox, bbox);
                if v >= min_iou {
                    cand.push((v, d.score, i, side));
                }
            }
        }
    }
    cand.sort_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then(y.1.total_cmp(&x.1))
            .then(x.2.cmp(&y.2))
            .then(x.3.cmp(&y.3))
    });
    let mut got = [None, None];
    let mut used = Vec::new();
    for (_, _, i, side) in cand {
        if let Some(slot) = got.get_mut(side)
            && slot.is_none()
            && !used.contains(&i)
        {
            *slot = Some(i);
            used.push(i);
        }
    }
    got
}

/// The best saved detection on the dropped box other than the kept box's, 0 if none: it splits the
/// drops into "the zoom saw nothing there" and "the zoom saw something weak there".
pub fn other_score(dets: &[Det], kept: usize, dropped: &Side, min_iou: f64) -> f64 {
    let bbox = wide(&dropped.bbox);
    dets.iter()
        .enumerate()
        .filter(|(i, d)| {
            *i != kept && same_object(&d.label, &dropped.class) && iou64(&d.bbox, &bbox) >= min_iou
        })
        .map(|(_, d)| d.score)
        .fold(0.0, f64::max)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "a score in 0..=1 stored as f32"
)]
fn score32(v: f64) -> f32 {
    v.clamp(0.0, 1.0) as f32
}

/// "3 detectors (gdino, rfdetr, serving)".
fn found_by(w: &[String]) -> String {
    match w.len() {
        0 => "no detector".to_string(),
        1 => format!("1 detector ({})", w.join(", ")),
        n => format!("{n} detectors ({})", w.join(", ")),
    }
}

/// The suggestion for `pair` (see the module docs' table). `witnesses`: of `a` and `b`.
pub fn suggest(pair: &Pair, witnesses: &[Vec<String>; 2], zoom: &Zoom<'_>) -> Suggestion {
    let [wa, wb] = witnesses;
    let base = |verdict: Verdict, why: String| Suggestion {
        a: pair.a.bbox,
        b: pair.b.bbox,
        verdict,
        why,
        kept_score: None,
        other_score: None,
        low_confidence: false,
        weak: false,
    };
    let drop_of = |drop_a: bool| {
        if drop_a {
            Verdict::DropA
        } else {
            Verdict::DropB
        }
    };
    let from_witnesses = |no_look: &str| {
        if wa.len() == wb.len() {
            return base(
                Verdict::Unclear,
                format!("{no_look}, and {} found each box", found_by(wa)),
            );
        }
        let drop_a = wa.len() < wb.len();
        let (kept, dropped, side) = if drop_a {
            (wb, wa, &pair.a)
        } else {
            (wa, wb, &pair.b)
        };
        if side.guarded {
            return base(
                Verdict::Unclear,
                format!("{no_look}, and the box with fewer detectors was set by a reviewer"),
            );
        }
        Suggestion {
            low_confidence: true,
            ..base(
                drop_of(drop_a),
                format!(
                    "{no_look}: {} found the kept box, {} found the dropped one",
                    found_by(kept),
                    found_by(dropped)
                ),
            )
        }
    };
    let dets = match zoom {
        Zoom::Unavailable => return from_witnesses("The closer look could not run on this pair"),
        Zoom::Looked(d) => *d,
    };
    let score_of = |i: usize| dets.get(i).map_or(0.0, |d| d.score);
    match match_zoom(dets, pair, ZOOM_MATCH_IOU, ZOOM_SCORE) {
        [Some(x), Some(y)] => {
            let (sx, sy) = (score_of(x), score_of(y));
            Suggestion {
                kept_score: Some(score32(sx)),
                other_score: Some(score32(sy)),
                ..base(
                    Verdict::TwoObjects,
                    format!("A closer look found each box on its own ({sx:.2} and {sy:.2})"),
                )
            }
        }
        [None, None] => from_witnesses("A closer look found neither box"),
        [a_seen, b_seen] => {
            let drop_a = a_seen.is_none();
            let (kept, dropped, side) = if drop_a {
                (wb, wa, &pair.a)
            } else {
                (wa, wb, &pair.b)
            };
            if dropped.len() > kept.len() {
                return base(
                    Verdict::Unclear,
                    format!(
                        "A closer look found only one box, but {} found the other against {}",
                        found_by(dropped),
                        found_by(kept)
                    ),
                );
            }
            if side.guarded {
                return base(
                    Verdict::Unclear,
                    "A closer look found only one box, but a reviewer set the other".to_string(),
                );
            }
            let kept_det = a_seen.or(b_seen).unwrap_or_default();
            let other = other_score(dets, kept_det, side, ZOOM_MATCH_IOU);
            let score = score_of(kept_det);
            let mut why = format!(
                "A closer look found only the kept box ({score:.2}): {} found it, {} found the \
                 dropped one",
                found_by(kept),
                found_by(dropped)
            );
            if other > 0.0 {
                why.push_str(&format!(
                    "; something weak ({other:.2}) was seen on the dropped box"
                ));
            }
            Suggestion {
                kept_score: Some(score32(score)),
                other_score: Some(score32(other)),
                weak: other >= WEAK_OTHER_SCORE,
                ..base(drop_of(drop_a), why)
            }
        }
    }
}

/// How strong a suggestion is, higher first; 0 for one that proposes nothing. Bands from the judged
/// sample: strong drops by their `other_score`, then two objects (9 of 45 wrong), then the weak band
/// (6 of 17 wrong), then the witnesses alone (not judged); within a band the higher zoom score.
pub fn strength(s: &Suggestion) -> u32 {
    let other = f64::from(s.other_score.unwrap_or(0.0));
    let band = match s.verdict {
        Verdict::Unclear | Verdict::Unknown => return 0,
        Verdict::TwoObjects => 4,
        Verdict::DropA | Verdict::DropB if s.low_confidence => 2,
        Verdict::DropA | Verdict::DropB if s.weak => 3,
        Verdict::DropA | Verdict::DropB => 7 - OTHER_BANDS.iter().filter(|b| other >= **b).count(),
    };
    let score = (f64::from(s.kept_score.unwrap_or(0.0)).clamp(0.0, 1.0) * 999.0).round();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a band under 8 and a whole number in 0..=999"
    )]
    let v = band as u32 * 1000 + score as u32;
    v
}

/// A suggestion that still applies, with its boxes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Live<'a> {
    /// Its index in [`LabelFile::overlap_suggestions`].
    pub index: usize,
    pub suggestion: &'a Suggestion,
    /// The boxes `a` and `b` name, indices into the frame's boxes.
    pub a: usize,
    pub b: usize,
}

/// The suggestions of `f` a reviewer can act on: of a frame not skipped and not checked by a reviewer,
/// proposing something, whose pair is still open under its two bboxes, and whose dropped box is not
/// [`guarded`] (it may have been edited since the job). Two boxes at one position (a car and a truck)
/// make one key: the first pair that qualifies. `wanted` None: every label wanted, no class list to
/// tell a mapped spelling from a relabel, so every `model_label` guards.
pub fn live<'a>(f: &'a LabelFile, wanted: Option<&Wanted<'_>>) -> Vec<Live<'a>> {
    if f.status == Status::Skipped || f.overlaps_checked || f.overlap_suggestions.is_empty() {
        return Vec::new();
    }
    let classes = wanted.map_or(&[][..], |w| w.classes);
    let pairs = stacks::overlap_pairs(&f.boxes, wanted);
    let mut out = Vec::new();
    for (index, s) in f.overlap_suggestions.iter().enumerate() {
        if strength(s) == 0 {
            continue;
        }
        let found = pairs.iter().find_map(|&(i, j)| {
            let (x, y) = (f.boxes.get(i)?, f.boxes.get(j)?);
            let (a, b) = if x.bbox == s.a && y.bbox == s.b {
                (i, j)
            } else if y.bbox == s.a && x.bbox == s.b {
                (j, i)
            } else {
                return None;
            };
            let dropped = match s.verdict {
                Verdict::DropA => Some(f.boxes.get(a)?),
                Verdict::DropB => Some(f.boxes.get(b)?),
                _ => None,
            };
            (!dropped.is_some_and(|d| guarded(d, classes))).then_some((a, b))
        });
        if let Some((a, b)) = found {
            out.push(Live {
                index,
                suggestion: s,
                a,
                b,
            });
        }
    }
    out
}

/// The strength of the strongest [`live`] suggestion of `f`, 0 for none: `auto=overlaps` lists
/// the frames in this order.
pub fn frame_strength(f: &LabelFile, wanted: Option<&Wanted<'_>>) -> u32 {
    live(f, wanted)
        .iter()
        .map(|l| strength(l.suggestion))
        .max()
        .unwrap_or(0)
}

/// Whether `f` already holds a suggestion, of any verdict, for `pair`: the job leaves it alone.
pub fn answered(f: &LabelFile, pair: &Pair) -> bool {
    f.overlap_suggestions.iter().any(|s| pair.is(s))
}

/// What [`apply`] did to a frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Applied {
    /// Suggestions proposing a drop, written now.
    pub drops: usize,
    /// Of those, in the weak band.
    pub weak: usize,
    /// Of those, from the witnesses alone.
    pub low_confidence: usize,
    /// Two-object suggestions written now.
    pub two_objects: usize,
    /// Pairs recorded as unclear now.
    pub unclear: usize,
    /// Open pairs with no zoom result: selected but not finished, or moved since.
    pub no_result: usize,
}

impl Applied {
    /// Whether the frame gained or changed a suggestion: only then is it written.
    pub fn changed(&self) -> bool {
        self.drops + self.two_objects + self.unclear > 0
    }
}

/// Write the suggestions of `f`'s open pairs as they stand now (run under the label write lock: a
/// reviewer may be working on the frame), and nothing else: boxes, status, `reviewed_at` and
/// `overlaps_checked` are not touched. `zoom_of` gives the job's answer for a pair of exactly these
/// two boxes, None when there is none: the pair is passed over (a box moved under the job, or its
/// crop did not finish), as is one that stopped being open. A skipped frame and one a reviewer
/// checked are left alone. A pair already answered stays unless `again`.
/// When something new is written, stale suggestions go too and the list is sorted strongest first,
/// at most [`MAX_SUGGESTIONS`].
pub fn apply<'z>(
    f: &mut LabelFile,
    wanted: &Wanted<'_>,
    again: bool,
    zoom_of: &dyn Fn(&Pair) -> Option<Zoom<'z>>,
) -> Applied {
    let mut done = Applied::default();
    if f.status == Status::Skipped || f.overlaps_checked {
        return done;
    }
    let opens = open_pairs(f, wanted);
    let mut fresh: Vec<Suggestion> = Vec::new();
    let mut list: Vec<Suggestion> = f
        .overlap_suggestions
        .iter()
        .filter(|s| opens.iter().any(|o| o.pair.is(s)))
        .cloned()
        .collect();
    for o in &opens {
        if fresh.iter().any(|s| o.pair.is(s)) || (!again && list.iter().any(|s| o.pair.is(s))) {
            continue;
        }
        let Some(zoom) = zoom_of(&o.pair) else {
            done.no_result += 1;
            continue;
        };
        let s = suggest(&o.pair, &o.witnesses, &zoom);
        if !list.contains(&s) {
            fresh.push(s);
        }
    }
    if fresh.is_empty() {
        return done;
    }
    for s in &fresh {
        match s.verdict {
            Verdict::DropA | Verdict::DropB => {
                done.drops += 1;
                done.weak += usize::from(s.weak);
                done.low_confidence += usize::from(s.low_confidence);
            }
            Verdict::TwoObjects => done.two_objects += 1,
            Verdict::Unclear | Verdict::Unknown => done.unclear += 1,
        }
    }
    list.retain(|s| !fresh.iter().any(|n| n.a == s.a && n.b == s.b));
    list.extend(fresh);
    list.sort_by_key(|s| std::cmp::Reverse(strength(s)));
    list.truncate(MAX_SUGGESTIONS);
    f.overlap_suggestions = list;
    done
}

/// A pending hint [`super::resolve_pending`] would reject, accepted: a save cannot take it.
fn promote(b: &mut LabelBox) {
    if b.state == BoxState::Pending && super::resolve_pending(b) == BoxState::Rejected {
        b.state = BoxState::Accepted;
    }
}

/// What the reviewer's key does (the page's `acceptOverlap`, which is the code that runs: this is
/// its statement in Rust, tested against [`super::resolve_pending`] and [`super::normalize`]): a drop
/// rejects the box as [`super::RejectReason::Duplicate`] and promotes a hint keeper, which a save would
/// otherwise reject, leaving the object with no box; two objects promotes both. Returns false, changing
/// nothing, when `s` is not [`live`].
pub fn accept(f: &mut LabelFile, s: &Suggestion, wanted: &Wanted<'_>) -> bool {
    let Some((a, b)) = live(f, Some(wanted))
        .iter()
        .find(|l| l.suggestion == s)
        .map(|l| (l.a, l.b))
    else {
        return false;
    };
    let (keep, drop) = match s.verdict {
        Verdict::DropA => (vec![b], Some(a)),
        Verdict::DropB => (vec![a], Some(b)),
        _ => (vec![a, b], None),
    };
    for k in keep {
        if let Some(x) = f.boxes.get_mut(k) {
            promote(x);
        }
    }
    if let Some(x) = drop.and_then(|d| f.boxes.get_mut(d)) {
        stacks::mark(x);
    }
    true
}

/// Bounds of the suggestions a client sends back (they are only round-tripped): [`super::normalize`].
pub fn check(list: &[Suggestion]) -> Result<()> {
    if list.len() > MAX_SUGGESTIONS {
        bail!(
            "{} overlap suggestions; at most {MAX_SUGGESTIONS} per frame",
            list.len()
        );
    }
    let score_ok = |s: Option<f32>| s.is_none_or(|v| v.is_finite() && (0.0..=1.0).contains(&v));
    for (i, s) in list.iter().enumerate() {
        let n = i + 1;
        if !s.a.iter().chain(&s.b).all(|v| v.is_finite()) {
            bail!("overlap suggestion {n}: the coordinates are not finite numbers");
        }
        if s.why.chars().count() > MAX_WHY_LEN || s.why.chars().any(char::is_control) {
            bail!(
                "overlap suggestion {n}: the reason is longer than {MAX_WHY_LEN} characters or \
                 holds control characters"
            );
        }
        if !score_ok(s.kept_score) || !score_ok(s.other_score) {
            bail!("overlap suggestion {n}: a score is not between 0 and 1");
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::tests_support::auto_box;
    use super::super::{Origin, RejectReason, Triage, normalize, resolve_pending};
    use super::*;

    fn classes() -> Vec<String> {
        ["person", "car", "truck", "motorcycle", "raccoon"]
            .map(String::from)
            .to_vec()
    }

    fn wanted(c: &[String]) -> Wanted<'_> {
        Wanted {
            classes: c,
            hidden: &[],
        }
    }

    const BIG: [f32; 4] = [0.0, 0.0, 100.0, 100.0];
    const SMALL: [f32; 4] = [10.0, 10.0, 60.0, 60.0];
    const FAR: [f32; 4] = [300.0, 300.0, 400.0, 400.0];

    fn side(bbox: [f32; 4]) -> Side {
        Side {
            bbox,
            class: "car".into(),
            guarded: false,
        }
    }

    fn pair() -> Pair {
        Pair {
            a: side(BIG),
            b: side(SMALL),
        }
    }

    fn det(label: &str, bbox: [f32; 4], score: f64) -> Det {
        Det {
            label: label.into(),
            bbox: bbox.map(f64::from),
            score,
        }
    }

    fn w(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    fn at(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            bbox,
            ..auto_box(label)
        }
    }

    fn hint(bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            bbox,
            source: Source::Model,
            origin: Some(Origin::BelowThreshold),
            score: Some(0.31),
            ..auto_box("car")
        }
    }

    fn accepted(b: LabelBox) -> LabelBox {
        LabelBox {
            state: BoxState::Accepted,
            ..b
        }
    }

    fn drop_b_suggestion() -> Suggestion {
        suggest(
            &pair(),
            &[w(&["gdino"]), w(&["gdino"])],
            &Zoom::Looked(&[det("car", BIG, 0.8)]),
        )
    }

    fn file(boxes: Vec<LabelBox>, s: Vec<Suggestion>) -> LabelFile {
        LabelFile {
            boxes,
            overlap_suggestions: s,
            ..LabelFile::default()
        }
    }

    #[test]
    fn the_table_of_zoom_and_witnesses() {
        let two = [w(&["gdino", "rfdetr"]), w(&["serving"])];
        let tie = [w(&["serving"]), w(&["gdino"])];
        let rev = [w(&["serving"]), w(&["gdino", "rfdetr"])];
        let both = [det("car", BIG, 0.8), det("car", SMALL, 0.6)];
        let only_a = [det("car", BIG, 0.8)];
        let only_b = [det("truck", SMALL, 0.7)];
        let far = [det("car", FAR, 0.9)];
        let weak_a = [det("car", BIG, 0.49)];
        // (witnesses, detections, verdict, low confidence)
        type Row<'a> = (&'a [Vec<String>; 2], Option<&'a [Det]>, Verdict, bool);
        let table: [Row<'_>; 12] = [
            (&two, Some(&both), Verdict::TwoObjects, false),
            (&rev, Some(&both), Verdict::TwoObjects, false),
            (&two, Some(&only_a), Verdict::DropB, false),
            (&tie, Some(&only_a), Verdict::DropB, false),
            (&rev, Some(&only_a), Verdict::Unclear, false),
            (&tie, Some(&only_b), Verdict::DropA, false),
            (&two, Some(&only_b), Verdict::Unclear, false),
            (&tie, Some(&far), Verdict::Unclear, false),
            (&two, Some(&far), Verdict::DropB, true),
            (&rev, Some(&weak_a), Verdict::DropA, true),
            (&two, None, Verdict::DropB, true),
            (&tie, None, Verdict::Unclear, false),
        ];
        for (i, (wit, dets, verdict, low)) in table.iter().enumerate() {
            let zoom = dets.map_or(Zoom::Unavailable, Zoom::Looked);
            let s = suggest(&pair(), wit, &zoom);
            assert_eq!(s.verdict, *verdict, "row {i}: {s:?}");
            assert_eq!(s.low_confidence, *low, "row {i}: {s:?}");
            assert_eq!((s.a, s.b), (BIG, SMALL));
            assert!(!s.why.is_empty() && s.why.chars().count() <= MAX_WHY_LEN);
            assert!(!s.why.contains(char::is_control), "{}", s.why);
        }
    }

    #[test]
    fn scores_and_the_weak_band() {
        let wit = [w(&["gdino"]), w(&["serving"])];
        let s = suggest(
            &pair(),
            &wit,
            &Zoom::Looked(&[
                det("car", BIG, 0.85),
                det("truck", SMALL, 0.3),
                det("person", SMALL, 0.45),
            ]),
        );
        assert_eq!(s.verdict, Verdict::DropB);
        assert_eq!(s.kept_score, Some(0.85));
        assert_eq!(s.other_score, Some(0.3));
        assert!(s.weak, "0.3 is in the weak band: {s:?}");
        let s = suggest(
            &pair(),
            &wit,
            &Zoom::Looked(&[det("car", BIG, 0.85), det("car", SMALL, 0.29)]),
        );
        assert!(!s.weak);
        // Nothing else there: 0, a strong drop, no note about it.
        let s = suggest(&pair(), &wit, &Zoom::Looked(&[det("car", BIG, 0.85)]));
        assert_eq!(s.other_score, Some(0.0));
        assert!(!s.why.contains("weak"), "{}", s.why);
        // The zoom's one box matches neither side when it is far from both boxes' IoU.
        let wit = [w(&["gdino"]), w(&["gdino"])];
        let s = suggest(&pair(), &wit, &Zoom::Looked(&[det("car", BIG, 0.5)]));
        assert_eq!((s.verdict, s.kept_score), (Verdict::DropB, Some(0.5)));
        assert!(!s.low_confidence);
    }

    #[test]
    fn one_detection_sees_only_one_box_the_best_iou_first() {
        // A detection on both boxes goes to the one it fits best.
        let mid = [12.0, 12.0, 58.0, 62.0];
        let p = Pair {
            a: side(SMALL),
            b: side([10.0, 10.0, 60.0, 80.0]),
        };
        assert_eq!(match_zoom(&[det("car", mid, 0.9)], &p, 0.5, 0.5)[1], None);
        let two = [
            det("car", mid, 0.9),
            det("car", [10.0, 10.0, 60.0, 79.0], 0.6),
        ];
        assert_eq!(match_zoom(&two, &p, 0.5, 0.5), [Some(0), Some(1)]);
        // A tie of IoU: the higher score.
        let same = [det("car", BIG, 0.6), det("car", BIG, 0.9)];
        assert_eq!(match_zoom(&same, &pair(), 0.6, 0.5), [Some(1), None]);
        // Then the earlier detection.
        let same = [det("car", BIG, 0.9), det("car", BIG, 0.9)];
        assert_eq!(match_zoom(&same, &pair(), 0.6, 0.5), [Some(0), None]);
        // Another object, a low score: no match.
        assert_eq!(
            match_zoom(&[det("person", BIG, 0.9)], &pair(), 0.6, 0.5),
            [None, None]
        );
        // Car and truck are one object.
        assert_eq!(
            match_zoom(&[det("truck", BIG, 0.9)], &pair(), 0.6, 0.5),
            [Some(0), None]
        );
    }

    #[test]
    fn a_threshold_reads_the_files_decimal_not_the_f32_value() {
        // 816.2 is 816.2000122 as f32: compared as the file holds it, this IoU is exactly the
        // reference's.
        assert_eq!(wide(&[816.2, 0.1, 2592.6, 1518.9])[0], 816.2);
        assert_eq!(wide(&[816.2, 0.1, 2592.6, 1518.9])[3], 1518.9);
    }

    #[test]
    fn guards_block_a_drop_and_a_mapped_spelling_does_not() {
        let c = classes();
        // A drawn box, a moved box, a relabelled box.
        let drawn = LabelBox {
            source: Source::Human,
            state: BoxState::Added,
            ..at("car", SMALL)
        };
        let moved = LabelBox {
            orig_bbox: Some([1.0, 2.0, 3.0, 4.0]),
            ..at("car", SMALL)
        };
        let relabelled = LabelBox {
            model_label: Some("truck".into()),
            ..at("car", SMALL)
        };
        let off_list = LabelBox {
            model_label: Some("refrigerator".into()),
            ..at("truck", SMALL)
        };
        for b in [&drawn, &moved, &relabelled, &off_list] {
            assert!(guarded(b, &c), "{b:?}");
        }
        // An auto-label run mapped the spelling: the class is the box's own, no reviewer here.
        let mapped = LabelBox {
            model_label: Some("motorbike".into()),
            ..at("motorcycle", SMALL)
        };
        assert!(!guarded(&mapped, &c));
        assert!(!guarded(&at("car", SMALL), &c));
        // The kept box may be guarded; the dropped one may not.
        let mut p = pair();
        p.b.guarded = true;
        let tie = [w(&["a"]), w(&["b"])];
        let only_a = [det("car", BIG, 0.8)];
        let s = suggest(&p, &tie, &Zoom::Looked(&only_a));
        assert_eq!(s.verdict, Verdict::Unclear);
        let only_b = [det("car", SMALL, 0.8)];
        let s = suggest(&p, &tie, &Zoom::Looked(&only_b));
        assert_eq!(s.verdict, Verdict::DropA);
        // The witnesses' fallback obeys them too.
        let more_a = [w(&["a", "b"]), w(&["c"])];
        assert_eq!(
            suggest(&p, &more_a, &Zoom::Unavailable).verdict,
            Verdict::Unclear
        );
        // Two objects drops nothing, so a guard does not matter.
        let both = [det("car", BIG, 0.8), det("car", SMALL, 0.8)];
        assert_eq!(
            suggest(&p, &tie, &Zoom::Looked(&both)).verdict,
            Verdict::TwoObjects
        );
    }

    #[test]
    fn witnesses_count_detectors_once_in_any_state_and_of_one_object() {
        let mut a = at("car", BIG);
        a.auto.labelers.insert("gdino".into(), 0.5);
        let mut rejected = at("truck", [0.0, 0.0, 100.0, 96.0]);
        rejected.state = BoxState::Rejected;
        rejected.auto.labelers.insert("rfdetr".into(), 0.5);
        rejected.auto.labelers.insert("gdino".into(), 0.4);
        let model = LabelBox {
            source: Source::Model,
            ..at("car", [0.0, 0.0, 96.0, 100.0])
        };
        // Near but another object, and of one object but not near.
        let mut person = at("person", BIG);
        person.auto.labelers.insert("night-x".into(), 0.9);
        let mut small = at("car", SMALL);
        small.auto.labelers.insert("other".into(), 0.9);
        let boxes = vec![a, rejected, model, person, small];
        assert_eq!(witnesses(&boxes, 0), ["gdino", "rfdetr", "serving"]);
        assert_eq!(witnesses(&boxes, 4), ["other"]);
        assert!(witnesses(&boxes, 99).is_empty());
    }

    #[test]
    fn pairs_read_in_one_order_and_the_class_is_the_lists() {
        let c = classes();
        let f = file(
            vec![at("car", SMALL), at("car", BIG), at("car", FAR)],
            vec![],
        );
        let o = open_pairs(&f, &wanted(&c));
        assert_eq!(o.len(), 1);
        assert_eq!((o[0].pair.a.bbox, o[0].pair.b.bbox), (BIG, SMALL));
        assert_eq!(o[0].pair.union(), BIG);
        let p = Pair {
            a: Side {
                class: "raccoon".into(),
                ..side(BIG)
            },
            b: side(SMALL),
        };
        assert!(no_zoom(&p));
        assert!(!no_zoom(&pair()));
    }

    fn write(f: &mut LabelFile, c: &[String], dets: &[Det]) -> Applied {
        apply(f, &wanted(c), false, &|_| Some(Zoom::Looked(dets)))
    }

    #[test]
    fn apply_writes_only_suggestions_and_keeps_an_answered_pair() {
        let c = classes();
        let mut f = LabelFile {
            status: Status::Reviewed,
            reviewed_at: Some("t".into()),
            boxes: vec![accepted(at("car", BIG)), accepted(at("car", SMALL))],
            ..LabelFile::default()
        };
        let before = f.clone();
        let a = write(&mut f, &c, &[det("car", BIG, 0.8)]);
        assert_eq!((a.drops, a.changed()), (1, true));
        assert_eq!(f.overlap_suggestions[0].verdict, Verdict::DropB);
        let mut back = f.clone();
        back.overlap_suggestions.clear();
        assert_eq!(back, before, "nothing but the suggestions");
        // Run again on other detections: answered, no change.
        let kept = f.clone();
        let a = write(&mut f, &c, &[det("car", SMALL, 0.8)]);
        assert!(!a.changed());
        assert_eq!(f, kept);
        // With `again` the pair is looked at afresh, and replaced, not doubled.
        let d = [det("car", SMALL, 0.8)];
        let a = apply(&mut f, &wanted(&c), true, &|_| Some(Zoom::Looked(&d)));
        assert_eq!(a.drops, 1);
        assert_eq!(f.overlap_suggestions.len(), 1);
        assert_eq!(f.overlap_suggestions[0].verdict, Verdict::DropA);
    }

    #[test]
    fn apply_leaves_skipped_and_checked_frames_and_pairs_with_no_answer() {
        let c = classes();
        let boxes = vec![at("car", BIG), at("car", SMALL)];
        let d = [det("car", BIG, 0.8)];
        let mut skipped = LabelFile {
            status: Status::Skipped,
            ..file(boxes.clone(), vec![])
        };
        assert!(!write(&mut skipped, &c, &d).changed());
        assert!(skipped.overlap_suggestions.is_empty());
        let mut checked = LabelFile {
            overlaps_checked: true,
            ..file(boxes.clone(), vec![])
        };
        assert!(!write(&mut checked, &c, &d).changed());
        // No result for the pair (a box moved under the job, or a cancel).
        let mut f = file(boxes, vec![]);
        let a = apply(&mut f, &wanted(&c), false, &|_| None::<Zoom<'_>>);
        assert_eq!((a.no_result, a.changed()), (1, false));
        // An answer of "could not look": the witnesses speak.
        let mut boxes = vec![at("raccoon", BIG), at("raccoon", SMALL)];
        boxes[0].auto.labelers.insert("gdino".into(), 0.5);
        let mut f = file(boxes, vec![]);
        let a = apply(&mut f, &wanted(&c), false, &|_| Some(Zoom::Unavailable));
        assert_eq!((a.drops, a.low_confidence), (1, 1));
        assert_eq!(f.overlap_suggestions[0].verdict, Verdict::DropB);
    }

    #[test]
    fn stale_suggestions_are_ignored_and_go_at_the_next_write() {
        let c = classes();
        let w = wanted(&c);
        let s = drop_b_suggestion();
        let mut f = file(
            vec![
                at("car", BIG),
                at("car", SMALL),
                at("car", [0.0, 0.0, 50.0, 60.0]),
            ],
            vec![s.clone()],
        );
        assert_eq!(live(&f, Some(&w)).len(), 1);
        assert_eq!(frame_strength(&f, Some(&w)), strength(&s));
        // A box moved: the key no longer names it.
        let mut moved = f.clone();
        moved.boxes[1].bbox = [11.0, 10.0, 60.0, 60.0];
        assert!(live(&moved, Some(&w)).is_empty());
        // A box gone.
        let mut gone = f.clone();
        gone.boxes.remove(1);
        assert!(live(&gone, Some(&w)).is_empty());
        // Rejected: the pair is settled.
        let mut rejected = f.clone();
        rejected.boxes[0].state = BoxState::Rejected;
        assert!(live(&rejected, Some(&w)).is_empty());
        // A reviewer looked, or skipped the frame.
        let checked = LabelFile {
            overlaps_checked: true,
            ..f.clone()
        };
        assert!(live(&checked, Some(&w)).is_empty());
        assert_eq!(frame_strength(&checked, Some(&w)), 0);
        let skipped = LabelFile {
            status: Status::Skipped,
            ..f.clone()
        };
        assert!(live(&skipped, Some(&w)).is_empty());
        // Relabelled since the job: the guard holds at read time too.
        let mut relabelled = f.clone();
        relabelled.boxes[1].model_label = Some("truck".into());
        assert!(live(&relabelled, Some(&w)).is_empty());
        // Unclear never shows, but counts as answered.
        let mut unclear = f.clone();
        unclear.overlap_suggestions[0].verdict = Verdict::Unclear;
        assert!(live(&unclear, Some(&w)).is_empty());
        assert!(answered(&unclear, &open_pairs(&unclear, &w)[0].pair));
        // A write drops what went stale: the key of a moved box.
        f.overlap_suggestions[0].a = [9.0, 9.0, 10.0, 10.0];
        let a = write(&mut f, &c, &[det("car", BIG, 0.8)]);
        assert!(a.drops >= 1);
        assert!(
            f.overlap_suggestions
                .iter()
                .all(|s| s.a != [9.0, 9.0, 10.0, 10.0])
        );
    }

    #[test]
    fn suggestions_are_bounded_and_strongest_first() {
        let c = classes();
        // Nested boxes, each too much larger than the last to stack: every pair overlaps.
        let boxes: Vec<LabelBox> = (0..13)
            .map(|i| {
                let side = 30.0 * 1.4_f32.powi(i);
                let mut b = at("car", [0.0, 0.0, side, side]);
                b.auto.labelers.insert(format!("d{i}"), 0.5);
                b
            })
            .collect();
        let mut f = file(boxes, vec![]);
        let open = open_pairs(&f, &wanted(&c)).len();
        assert!(open > MAX_SUGGESTIONS, "{open}");
        write(&mut f, &c, &[]);
        assert_eq!(f.overlap_suggestions.len(), MAX_SUGGESTIONS);
        check(&f.overlap_suggestions).unwrap();
        let s: Vec<u32> = f.overlap_suggestions.iter().map(strength).collect();
        assert!(s.windows(2).all(|p| p[0] >= p[1]), "{s:?}");
    }

    #[test]
    fn strength_orders_the_bands_by_what_the_sample_measured() {
        let s = |verdict, kept, other, low, weak| Suggestion {
            a: BIG,
            b: SMALL,
            verdict,
            why: String::new(),
            kept_score: kept,
            other_score: other,
            low_confidence: low,
            weak,
        };
        let order = [
            s(Verdict::DropA, Some(0.9), Some(0.0), false, false),
            s(Verdict::DropB, Some(0.6), Some(0.0), false, false),
            s(Verdict::DropA, Some(0.9), Some(0.15), false, false),
            s(Verdict::DropA, Some(0.9), Some(0.25), false, false),
            s(Verdict::TwoObjects, Some(0.9), Some(0.9), false, false),
            s(Verdict::DropA, Some(0.9), Some(0.4), false, true),
            s(Verdict::DropA, None, None, true, false),
        ];
        let got: Vec<u32> = order.iter().map(strength).collect();
        assert!(got.windows(2).all(|p| p[0] > p[1]), "{got:?}");
        assert_eq!(strength(&s(Verdict::Unclear, None, None, false, false)), 0);
        assert_eq!(strength(&s(Verdict::Unknown, None, None, false, false)), 0);
    }

    #[test]
    fn a_winning_hint_survives_the_save_and_the_object_keeps_a_box() {
        let c = classes();
        let w = wanted(&c);
        // The whole car is a hint, the fragment a prediction: without the promotion the save would
        // reject the hint, and rejecting the fragment would leave the object with no box.
        let keeper = hint(BIG);
        assert_eq!(resolve_pending(&keeper), BoxState::Rejected);
        let fragment = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            ..at("car", SMALL)
        };
        let s = drop_b_suggestion();
        let mut f = file(vec![keeper, fragment], vec![s.clone()]);
        assert!(accept(&mut f, &s, &w));
        assert_eq!(f.boxes[0].state, BoxState::Accepted);
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
        assert_eq!(f.boxes[1].reject_reason, Some(RejectReason::Duplicate));
        assert_eq!(f.boxes[1].reason.as_deref(), Some(stacks::DUPLICATE_REASON));
        f.status = Status::Reviewed;
        let n = normalize(f, &c, 640, 480, "now").unwrap();
        assert_eq!(n.training().count(), 1, "the object keeps its box");
        assert_eq!(n.training().next().unwrap().bbox, BIG);
        // A pending prediction keeper needs no promotion: the save accepts it, pending it stays.
        let plain = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            ..at("car", BIG)
        };
        let mut f = file(vec![plain, at("car", SMALL)], vec![s.clone()]);
        assert!(accept(&mut f, &s, &w));
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        // A confirmed hint likewise.
        let mut confirmed = hint(BIG);
        confirmed.auto.triage = Some(Triage::Accept);
        let mut f = file(vec![confirmed, at("car", SMALL)], vec![s.clone()]);
        assert!(accept(&mut f, &s, &w));
        assert_eq!(f.boxes[0].state, BoxState::Pending);
        // Not live: nothing changes.
        let mut f = file(vec![at("car", BIG)], vec![s.clone()]);
        let before = f.clone();
        assert!(!accept(&mut f, &s, &w));
        assert_eq!(f, before);
    }

    #[test]
    fn accepting_two_objects_keeps_both_even_as_hints() {
        let c = classes();
        let w = wanted(&c);
        let s = suggest(
            &pair(),
            &[w_of("a"), w_of("b")],
            &Zoom::Looked(&[det("car", BIG, 0.8), det("car", SMALL, 0.7)]),
        );
        assert_eq!(s.verdict, Verdict::TwoObjects);
        let mut f = file(vec![hint(BIG), hint(SMALL)], vec![s.clone()]);
        assert!(accept(&mut f, &s, &w));
        f.status = Status::Reviewed;
        let n = normalize(f, &c, 640, 480, "now").unwrap();
        assert_eq!(n.training().count(), 2);
    }

    fn w_of(name: &str) -> Vec<String> {
        vec![name.to_string()]
    }

    #[test]
    fn check_bounds_what_a_client_sends() {
        let good = drop_b_suggestion();
        check(std::slice::from_ref(&good)).unwrap();
        let mut s = good.clone();
        s.a[0] = f32::NAN;
        assert!(check(&[s]).is_err());
        let mut s = good.clone();
        s.why = "x".repeat(MAX_WHY_LEN + 1);
        assert!(check(&[s]).is_err());
        let mut s = good.clone();
        s.why = "a\nb".into();
        assert!(check(&[s]).is_err());
        let mut s = good.clone();
        s.kept_score = Some(1.5);
        assert!(check(&[s]).is_err());
        assert!(check(&vec![good; MAX_SUGGESTIONS + 1]).is_err());
    }

    #[test]
    fn an_unknown_verdict_loads_and_is_ignored() {
        let json = r#"{"a":[0,0,1,1],"b":[0,0,2,2],"verdict":"three_objects","why":"x"}"#;
        let s: Suggestion = serde_json::from_str(json).unwrap();
        assert_eq!(s.verdict, Verdict::Unknown);
        assert_eq!(strength(&s), 0);
        let s: Suggestion =
            serde_json::from_str(r#"{"a":[0,0,1,1],"b":[0,0,2,2],"verdict":"drop_a"}"#).unwrap();
        assert_eq!(s.verdict, Verdict::DropA);
        assert!(!serde_json::to_string(&s).unwrap().contains("weak"));
    }

    #[test]
    fn suggestions_survive_a_save_and_a_file_without_them_reads_as_before() {
        let c = classes();
        let s = drop_b_suggestion();
        let f = file(vec![at("car", BIG), at("car", SMALL)], vec![s.clone()]);
        for status in [Status::Unreviewed, Status::Reviewed, Status::Skipped] {
            let n = normalize(
                LabelFile {
                    status,
                    ..f.clone()
                },
                &c,
                640,
                480,
                "now",
            )
            .unwrap();
            assert_eq!(
                n.overlap_suggestions,
                std::slice::from_ref(&s),
                "{status:?}"
            );
        }
        // The file's JSON round trip, exact floats included.
        let text = serde_json::to_string_pretty(&f).unwrap();
        assert!(text.contains("\"overlap_suggestions\""));
        assert_eq!(serde_json::from_str::<LabelFile>(&text).unwrap(), f);
        // Unbounded input is refused like any other field.
        let mut bad = f.clone();
        bad.overlap_suggestions[0].why = "x".repeat(MAX_WHY_LEN + 1);
        let e = normalize(bad, &c, 640, 480, "now").unwrap_err();
        assert!(e.to_string().contains("overlap suggestion 1"), "{e}");
        // Off, the field is not written, and an old file loads without it.
        let off = file(vec![at("car", BIG)], vec![]);
        let text = serde_json::to_string_pretty(&off).unwrap();
        assert!(!text.contains("overlap_suggestions"), "{text}");
        let old: LabelFile = serde_json::from_str(r#"{"status":"reviewed"}"#).unwrap();
        assert!(old.overlap_suggestions.is_empty());
    }

    #[test]
    fn suggestions_survive_merges_triage_and_an_import() {
        let s = drop_b_suggestion();
        let base = LabelFile {
            status: Status::Reviewed,
            ..file(
                vec![accepted(at("car", BIG)), accepted(at("car", SMALL))],
                vec![s.clone()],
            )
        };
        // merge_proposals: the gained box reopens the frame, the two boxes are still there.
        let mut f = base.clone();
        let r = super::super::merge_proposals(&mut f, vec![at("person", FAR)]);
        assert_eq!((r.added, f.status), (1, Status::Unreviewed));
        assert_eq!(f.overlap_suggestions, std::slice::from_ref(&s));
        // Auto-label triage and a missing-label run.
        use crate::training::autolabel::triage;
        let mut f = base.clone();
        triage::apply(&mut f, vec![at("person", FAR)]);
        assert_eq!(f.overlap_suggestions, std::slice::from_ref(&s));
        let mut f = base.clone();
        let m = triage::apply_missing(&mut f, vec![at("person", FAR)], "night");
        assert_eq!(m.added, 1);
        assert_eq!(f.overlap_suggestions, std::slice::from_ref(&s));
        // An X-AnyLabeling import rebuilds the file from its rectangles.
        let rects = [
            ("car".to_string(), BIG, None, false),
            ("car".to_string(), SMALL, None, false),
        ];
        let f = super::super::xany::apply_import(base, &rects);
        assert_eq!(f.overlap_suggestions, [s]);
    }

    #[test]
    fn suggestions_survive_every_bulk_edit() {
        use super::super::bulk::test_support::{labels_of, put_frame};
        use super::super::bulk::{self, Scope};
        use super::super::queue::Filters;
        use super::super::repeats::{self, Spot};
        use super::super::{Frame, write};
        let dir = std::env::temp_dir().join(format!("bop-sug-bulk-{}", uuid::Uuid::new_v4()));
        let c = classes();
        let hidden = ["truck".to_string()];
        let w = Wanted {
            classes: &c,
            hidden: &hidden,
        };
        let s = drop_b_suggestion();
        let filters = Filters::default();
        let (box_a, box_b) = ([0.0, 0.0, 30.0, 30.0], [2.0, 2.0, 20.0, 20.0]);
        let pair = |extra: Vec<LabelBox>| {
            let mut v = vec![at("car", box_a), at("car", box_b)];
            v.extend(extra);
            v
        };
        // Every frame carries the suggestion; each edit below must change its frame and keep it.
        let put = |stem: &str, boxes: Vec<LabelBox>, status: Status| {
            let rel = put_frame(&dir, stem, None, boxes);
            let frame = Frame::resolve(&dir, &rel).unwrap();
            let mut l = frame.labels().unwrap().unwrap();
            l.status = status;
            l.overlap_suggestions = vec![s.clone()];
            write(&frame.labels, &l).unwrap();
            rel
        };
        let low = LabelBox {
            score: Some(0.1),
            ..at("dog", [40.0, 30.0, 60.0, 45.0])
        };
        let check_kept = |rel: &str, what: &str, changed: bool| {
            assert!(changed, "{what} changed nothing");
            assert_eq!(
                labels_of(&dir, rel).overlap_suggestions,
                std::slice::from_ref(&s),
                "{what}"
            );
        };
        let rel = put("low", pair(vec![low]), Status::Unreviewed);
        let o = bulk::bulk_reject_below(&dir, &filters, 0.3, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-low", o.boxes == 1);
        let clock = at("clock", [40.0, 30.0, 60.0, 45.0]);
        let rel = put("other", pair(vec![clock.clone()]), Status::Unreviewed);
        let o = bulk::bulk_reject_other_class(&dir, &filters, &w, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-other-class", o.boxes == 1);
        let rel = put("unwanted", pair(vec![clock.clone()]), Status::Unreviewed);
        let o = bulk::bulk_reject_unwanted(&dir, &w, Scope::Unreviewed, 500, false);
        check_kept(&rel, "unwanted on class change", o.boxes >= 1);
        let truck = accepted(at("truck", [40.0, 30.0, 60.0, 45.0]));
        let rel = put("reviewed", vec![truck], Status::Reviewed);
        let o = bulk::bulk_reject_unwanted(&dir, &w, Scope::Reviewed, 500, false);
        check_kept(&rel, "unwanted on reviewed frames", o.boxes == 1);
        let near = [0.0, 0.0, 30.0, 29.0];
        let rel = put("stacked", pair(vec![at("car", near)]), Status::Unreviewed);
        let o = bulk::bulk_reject_stacked(&dir, &filters, &w, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-stacked", o.outcome.boxes == 1);
        let sure = LabelBox {
            bbox: box_a,
            ..super::super::bulk::test_support::auto_box("car", 0.9, Triage::Accept)
        };
        let rel = put("accept", vec![sure], Status::Unreviewed);
        let ok = bulk::accept_one(&dir, &rel, &c, "now").unwrap();
        check_kept(&rel, "bulk accept", ok);
        let spot = Spot {
            camera: "cam".into(),
            label: "dog".into(),
            bbox: [40.0, 30.0, 60.0, 45.0],
            width: 64,
            height: 48,
            min_iou: repeats::REJECT_SPOT_IOU,
        };
        let v = repeats::Verdict::Reject(None);
        let dog = at("dog", spot.bbox);
        let rel = put("spot", vec![dog], Status::Unreviewed);
        let o = bulk::bulk_spot(&dir, &filters, &c, &[], &spot, &v, 500, false, &|r| {
            r == rel
        });
        check_kept(&rel, "reject-spot", o.boxes == 1);
        let o = bulk::bulk_accept_repeats(&dir, &filters, &c, 500, "now", false, &|r| r == rel);
        check_kept(&rel, "accept-repeats", o.done_count == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One case of `tests/data/overlaps_parity.json`: what `eval_overlaps.py` answers.
    #[derive(Deserialize)]
    struct Case {
        name: String,
        a: CaseSide,
        b: CaseSide,
        dets: Option<Vec<Det>>,
        matched: Option<[Option<usize>; 2]>,
        verdict: String,
        drop: Option<String>,
        reason: String,
        kept_score: Option<f64>,
        other_score: Option<f64>,
        scores: Option<[f64; 2]>,
    }

    #[derive(Deserialize)]
    struct CaseSide {
        cls: String,
        bbox: [f32; 4],
        witnesses: Vec<String>,
        guarded: bool,
    }

    #[test]
    fn suggestions_agree_with_the_research_reference() {
        let text = include_str!("../../../tests/data/overlaps_parity.json");
        let doc: serde_json::Value = serde_json::from_str(text).unwrap();
        let point = &doc["point"];
        assert_eq!(point["match_iou"].as_f64(), Some(ZOOM_MATCH_IOU));
        assert_eq!(point["score"].as_f64(), Some(ZOOM_SCORE));
        assert_eq!(point["floor"].as_f64(), Some(ZOOM_FLOOR));
        assert_eq!(point["pad"].as_f64(), Some(ZOOM_PAD));
        let cases: Vec<Case> = serde_json::from_value(doc["cases"].clone()).unwrap();
        assert!(cases.len() >= 12, "{}", cases.len());
        check_cases(cases);
    }

    /// The same check on every open pair of a research snapshot: a JSON file of the fixture's
    /// `cases`, written from `eval_overlaps.py`'s `pairs.json` and `zoom.jsonl`.
    /// `BOP_OVERLAPS_PARITY=<file> cargo test --lib real_pairs -- --ignored`.
    #[test]
    #[ignore = "needs a research snapshot"]
    fn real_pairs_agree_with_the_research_reference() {
        // CI runs every ignored test: without a snapshot there is nothing to check.
        let Ok(path) = std::env::var("BOP_OVERLAPS_PARITY") else {
            println!("skipped: BOP_OVERLAPS_PARITY is not set");
            return;
        };
        let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let cases: Vec<Case> = serde_json::from_value(doc["cases"].clone()).unwrap();
        println!("{} pairs", cases.len());
        assert!(!cases.is_empty());
        check_cases(cases);
    }

    fn check_cases(cases: Vec<Case>) {
        for c in cases {
            let side = |s: &CaseSide| Side {
                bbox: s.bbox,
                class: s.cls.clone(),
                guarded: s.guarded,
            };
            let p = Pair {
                a: side(&c.a),
                b: side(&c.b),
            };
            let wit = [c.a.witnesses.clone(), c.b.witnesses.clone()];
            let zoom = c.dets.as_deref().map_or(Zoom::Unavailable, Zoom::Looked);
            if let (Some(d), Some(m)) = (&c.dets, c.matched) {
                assert_eq!(
                    match_zoom(d, &p, ZOOM_MATCH_IOU, ZOOM_SCORE),
                    m,
                    "{}",
                    c.name
                );
            }
            let s = suggest(&p, &wit, &zoom);
            let want = match (c.verdict.as_str(), c.drop.as_deref()) {
                ("two_objects", _) => Verdict::TwoObjects,
                ("drop_other", Some("a")) => Verdict::DropA,
                ("drop_other", Some("b")) => Verdict::DropB,
                // The reference leaves these with a reviewer; the fallback answers from witnesses.
                ("reviewer", _) if matches!(c.reason.as_str(), "neither" | "no_zoom") => {
                    let (n, m) = (c.a.witnesses.len(), c.b.witnesses.len());
                    let blocked = (n < m && c.a.guarded) || (m < n && c.b.guarded);
                    if n == m || blocked {
                        Verdict::Unclear
                    } else if n < m {
                        Verdict::DropA
                    } else {
                        Verdict::DropB
                    }
                }
                ("reviewer", _) => Verdict::Unclear,
                other => panic!("{}: {other:?}", c.name),
            };
            assert_eq!(s.verdict, want, "{}: {s:?}", c.name);
            if c.verdict == "drop_other" {
                assert_eq!(
                    s.kept_score,
                    c.kept_score.map(score32),
                    "kept score of {}",
                    c.name
                );
                assert_eq!(
                    s.other_score,
                    c.other_score.map(score32),
                    "other score of {}",
                    c.name
                );
                assert_eq!(
                    s.weak,
                    c.other_score.is_some_and(|o| o >= WEAK_OTHER_SCORE),
                    "{}",
                    c.name
                );
            }
            if let Some([x, y]) = c.scores {
                assert_eq!(
                    (s.kept_score, s.other_score),
                    (Some(score32(x)), Some(score32(y))),
                    "{}",
                    c.name
                );
            }
        }
    }
}
