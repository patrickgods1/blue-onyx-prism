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
//!    Below: `loose`, counted for the report. A loose pair whose detection scores
//!    [`TIGHTEN_FOLD_SCORE`] or more, on a label of a wanted class RF-DETR L can see
//!    ([`coco_class`]) that is not too small (the floor of rule 4), is a candidate for a closer
//!    look ([`Judged::candidates`], [`Loose`]): the zoom stage of the check
//!    ([`super::tighten`]) runs RF-DETR L on a crop of the pair, and [`tighten`] decides from
//!    both looks whether a better-fitting box is proposed (`tighten`). A fold model's box alone
//!    never overrules a label: fold models are short-trained at a small input size.
//! 4. `add`: an unmatched detection at [`ADD_SCORE`] or more, at least [`FLOOR_PX`] on its short side
//!    at model input, that no box of its object that is not rejected and no ignore box covers
//!    ([`covers`]), and no judged reject of its object lies on at IoU >= [`REJECT_BLOCK_IOU`].
//!    **At a static spot** ([`spot_of`]: the detection lies at a spot of the check's table,
//!    [`statics::Table::spot_at`], that is still a spot without the frame's camera-day,
//!    [`statics::Spot::holds_without`]; people and animals form no spot,
//!    [`statics::may_be_static`]) the score may be as low as [`ADD_SCORE_AT_SPOT`]; such an
//!    add carries the spot's id ([`Suggestion::spot`], the check's evidence: the Static objects
//!    card places it by geometry), its own class from [`ADD_SCORE`] up and the spot's most common
//!    label (through the class list) below, and says where its evidence comes from. Measured on kept fold detections of 851
//!    frames: an uncovered detection at a place labelled on 10 or more other frames of its camera
//!    is real about 90% of the time down to 0.3, while elsewhere detections at 0.3-0.5 are almost
//!    all false. The evidence leaves out the frame's camera-day, not the frame's whole fold: it is
//!    the reviewer's labels, which the fold model's blindness does not concern (it never saw
//!    those frames either way), and other days of the fold are other looks at the place; only
//!    frames of the same day are near copies of this one.
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
//! too small. With `collect.size_floor` set ([`Context::floor`]) that rule applies too
//! ([`under_floor`], at the floor's own input size, as the export and the Review page apply it):
//! "at least [`FLOOR_PX`]" above reads "at least [`FLOOR_PX`] and not under the floor", and a
//! training box under the floor is an ignore region of the export, so the check treats it as one:
//! no candidate in steps 2-6 (no verify, no relabel, no loose count, out of the recall but for its
//! size table, where it counts with [`BoxFact::under_floor`]), while it still covers detections of
//! its object in step 4 (no add lands on it). The fold models themselves train on every label
//! (their exports apply no floor), so the size table measures what the recipe can find.
//!
//! **A quick check** ([`Context::model`], [`super::quick`]) applies the same rules with a model the
//! user trained instead of a fold model, on frames it never trained or validated on: `imgsz` is its
//! input size, `class_boxes` what its run trained on, and a wanted class it does not have
//! ([`ModelJudge::classes`]) gets no `verify` (step 5), its labels are no `relabel` target (step 6)
//! and stay out of the recall (its detections of other classes cannot say the label is wrong; its
//! silence is no miss), and such a label covers a detection of any class in step 4 at IoU >=
//! [`UNKNOWN_CLASS_COVER_IOU`] (the model names the object it cannot name by another class: no
//! second box on it; a smaller object inside or beside the box is still an add). Every `why`
//! names the model instead of "a model that never saw this camera-day": it may well have trained
//! on other frames of that camera-day.

use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::overlaps::Det as ZoomDet;
use crate::collect::labels::refine::{self, Kind, Suggestion};
use crate::collect::labels::repeats::same_object;
use crate::collect::labels::statics::{self, Spot};
use crate::collect::labels::{
    BoxState, LabelBox, LabelFile, SizeFloor, Source, Status, clamp_box, covers, inside_share, iou,
    stacks, under_floor,
};
use crate::model::coco::coco_class;
use crate::training::autolabel::triage::Det;
use crate::training::overlaps::zoom::CropPlace;
use serde::{Deserialize, Serialize};
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
/// IoU from which a label of a wanted class a quick check's model does not have explains a
/// detection of another class (rule 4): the model names that box's object by a class it has, so the
/// detection is about the same box; a person standing in front of a labelled truck is not.
pub const UNKNOWN_CLASS_COVER_IOU: f32 = 0.5;
/// Score from which an unmatched detection at a static spot is proposed as a missing box (rule 4):
/// labels at the place on other days make a weaker find worth a look.
pub const ADD_SCORE_AT_SPOT: f32 = 0.3;
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
/// Score from which a fold model's detection of a loose label makes the pair a candidate for the
/// closer look ([`Loose`]): weaker finds are too often a short-trained model's guess to spend a
/// crop on (the same cut-off as an `add`, [`ADD_SCORE`]).
pub const TIGHTEN_FOLD_SCORE: f32 = 0.5;
/// Score from which a detection of the closer look counts as a box at all: the overlap check's
/// measured operating point ([`crate::collect::labels::overlaps::ZOOM_SCORE`]), same model, same
/// crops.
pub const TIGHTEN_ZOOM_SCORE: f32 = 0.5;
/// IoU from which the closer look's box agrees with the fold model's: the overlap check's match IoU
/// ([`crate::collect::labels::overlaps::ZOOM_MATCH_IOU`]). Two independent looks that agree with
/// each other and not with the label outvote it; one look alone does not.
pub const TIGHTEN_AGREE_IOU: f32 = 0.6;
/// Share of a second detection of the closer look (of the label's object, not `Z` again) inside the
/// label from which the label spans that object too ([`NoTighten::SpansTwo`]): the overlap check's
/// share for a box nested in another ([`stacks::NESTED_SHARE`], 0.8). Moving such a label onto `Z`
/// would leave the other object unlabelled.
pub const TIGHTEN_SPAN_INSIDE: f32 = stacks::NESTED_SHARE;
/// Pixels within which an edge of `Z` lies on an edge of its crop ([`NoTighten::CropEdge`]): the
/// script's boxes are kept to 0.1 px and its crops are whole pixels, so an object cut by the crop
/// ends there exactly, give or take the model's last pixel.
pub const TIGHTEN_EDGE_PX: f32 = 1.0;

/// One frame to judge.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput<'a> {
    pub file: &'a LabelFile,
    /// The frame's size (its sidecar).
    pub width: u32,
    pub height: u32,
    /// The fold model's detections on it.
    pub dets: &'a [Det],
    /// Its camera and day (`YYYY-MM-DD`): the static spots it may use ([`spot_of`]).
    pub camera: &'a str,
    pub day: &'a str,
}

/// The model of a quick check ([`super::quick`]) as the judge needs it: its name for the `why` of
/// its suggestions and the wanted classes it has.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelJudge {
    /// The configured model's name.
    pub name: String,
    /// Its classes in the class list's spelling ([`Wanted::class_of`] of each), hidden ones too.
    /// A wanted class it does not have: its labels get no `verify` and no `relabel` (the model
    /// cannot see the class, so its silence or another class's detection on such a label says
    /// nothing about it), and are not in the recall.
    pub classes: std::collections::BTreeSet<String>,
}

/// What every frame of a check shares.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    pub wanted: &'a Wanted<'a>,
    /// The fold models' input size; a quick check's model's own.
    pub imgsz: u32,
    /// Training boxes per class (the list's spelling) of the frames the fold model trained on
    /// (the selection without the fold's frames: [`super::class_boxes`]), ignore regions left out;
    /// for a quick check, what its model's run trained on (its `dataset.json`).
    pub class_boxes: &'a BTreeMap<String, usize>,
    /// The check's id ([`Suggestion::run`]).
    pub run: &'a str,
    /// The static spots of the check's selection ([`super::Selected::spots`]); None: no spot rule.
    pub spots: Option<&'a statics::Table>,
    /// `collect.size_floor`: when set, a training box under it ([`under_floor`], as the export and
    /// the Review page apply it) is an ignore region here, and nothing under it is suggested; the
    /// noise rule of [`FLOOR_PX`] at `imgsz` holds as well (see [`judge`]).
    pub floor: Option<SizeFloor>,
    /// A quick check's model; None: the fold models (each never saw the frame's camera-day).
    pub model: Option<&'a ModelJudge>,
}

impl Context<'_> {
    /// Who looked, as the `why` of a suggestion starts: `a model that never saw this camera-day`,
    /// or `<model> (never trained on this frame)`.
    fn who(&self) -> String {
        match self.model {
            None => "a model that never saw this camera-day".into(),
            Some(m) => format!("{} (never trained on this frame)", m.name),
        }
    }

    /// Whether the judging model has `class` (a wanted class): fold models have every one.
    fn knows(&self, class: &str) -> bool {
        self.model.is_none_or(|m| m.classes.contains(class))
    }
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
    /// It lies at a static spot that holds without the frame's camera-day ([`spot_of`]).
    pub at_spot: bool,
    /// Under the configured size floor ([`Context::floor`]): an ignore region for the check, so
    /// only the recall by size counts it (that table is what the floor is chosen from).
    pub under_floor: bool,
}

/// The static spot of `ctx` a box of `label` at `bbox` (pixels of `frame`) lies at, when that spot
/// is still one without the frame's camera-day ([`statics::Spot::holds_without`]: one day's
/// frames are near copies, so a spot that day alone makes is no evidence for its frames). None
/// without a spot table, for a frame of unknown size, or at no such spot.
pub fn spot_of<'c>(
    ctx: &Context<'c>,
    frame: &FrameInput<'_>,
    label: &str,
    bbox: &[f32; 4],
) -> Option<&'c Spot> {
    let table = ctx.spots?;
    let nb = statics::normalised(bbox, frame.width, frame.height)?;
    table
        .spot_at(frame.camera, label, &nb)
        .filter(|s| s.holds_without(frame.day))
}

/// What [`judge`] made of a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Judged {
    /// The suggestions, every one [`refine::live`] on the file.
    pub suggestions: Vec<Suggestion>,
    /// Matched pairs below [`GOOD_IOU`] on boxes no reviewer drew, moved or relabelled.
    pub loose: usize,
    /// Of those, the candidates for the closer look (rule 3).
    pub candidates: Vec<Loose>,
    /// The frame's training boxes of wanted classes, for the recall.
    pub boxes: Vec<BoxFact>,
}

/// A loose label worth a closer look (rule 3): the label box `L` and the fold model's detection
/// `D` of the same object. The zoom stage crops their union ([`Loose::union`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loose {
    /// The label box `L`, exactly as in the file (the key of a `tighten`).
    pub target: [f32; 4],
    /// Its label as written ([`Suggestion::target_label`]).
    pub target_label: String,
    /// Its class (the list's spelling).
    pub class: String,
    /// The fold model's detection `D`, clamped to the frame.
    pub det: [f32; 4],
    /// `D`'s score.
    pub det_score: f32,
}

impl Loose {
    /// The union of `L` and `D`: the crop the closer look pads and runs on.
    pub fn union(&self) -> [f32; 4] {
        let (l, d) = (self.target, self.det);
        [
            l[0].min(d[0]),
            l[1].min(d[1]),
            l[2].max(d[2]),
            l[3].max(d[3]),
        ]
    }

    /// Whether `other` is this pair (exact boxes, the label alike): a candidate of a judgement
    /// made again under the label write lock finds the closer look made for it.
    pub fn same_pair(&self, other: &Loose) -> bool {
        let bits = |b: &[f32; 4]| b.map(f32::to_bits);
        bits(&self.target) == bits(&other.target)
            && bits(&self.det) == bits(&other.det)
            && self
                .target_label
                .trim()
                .eq_ignore_ascii_case(other.target_label.trim())
    }
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

/// A box a reviewer drew, moved or relabelled: never loose, never refitted.
fn reviewers(b: &LabelBox) -> bool {
    b.source == Source::Human || b.orig_bbox.is_some() || b.model_label.is_some()
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
    // A training box under the configured size floor (the export's rule) is an ignore region of
    // the export: not judged.
    let floor = ctx.floor.filter(SizeFloor::is_on);
    let under = |b: &[f32; 4]| under_floor(b, frame.width, frame.height, floor);
    let ignored = |b: &LabelBox| under(&b.bbox);
    // Too small to suggest anything about: under that floor, or under FLOOR_PX at the fold
    // models' input, whichever is larger (a permissive floor does not let through what the fold
    // models cannot see).
    let small = |b: &[f32; 4]| under(b) || size(b) < FLOOR_PX;
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
    let all_train: Vec<T<'_>> = f
        .boxes
        .iter()
        .filter(|b| b.is_training() && !b.is_ignore() && proper(&b.bbox))
        .map(|b| T {
            b,
            class: wanted.class_of(&b.label).map(str::to_string),
            key: key_of(&b.label, wanted),
        })
        .collect();
    // The report's recall: its own matching at a production-like score, over every training box
    // (the recall by size is what a floor is chosen from; the others leave out those under it).
    let (found, _) = matching(&dets, &all_train, FOUND_IOU, FOUND_SCORE);
    for (t, hit) in all_train.iter().zip(&found) {
        // A class the quick check's model does not have is no miss of it.
        if let Some(class) = t.class.as_ref().filter(|c| ctx.knows(c)) {
            out.boxes.push(BoxFact {
                class: class.clone(),
                short_px: size(&t.b.bbox),
                found: hit.is_some(),
                at_spot: spot_of(ctx, frame, class, &t.b.bbox).is_some(),
                under_floor: ignored(t.b),
            });
        }
    }
    let train: Vec<T<'_>> = all_train.into_iter().filter(|t| !ignored(t.b)).collect();
    // 2. Pairs.
    let (by_box, by_det) = matching(&dets, &train, LOOSE_IOU_MIN, MATCH_MIN_SCORE);
    let mut found_list: Vec<Suggestion> = Vec::new();
    let mut evidence = vec![false; dets.len()];
    let mut relabelled = vec![false; train.len()];
    // 3. Loose labels, and the candidates for a closer look among them.
    for (t, m) in train.iter().zip(&by_box) {
        if let Some((di, v)) = m
            && *v < GOOD_IOU
            && !reviewers(t.b)
        {
            out.loose += 1;
            if let (Some(class), Some(d)) = (&t.class, dets.get(*di))
                && d.score >= TIGHTEN_FOLD_SCORE
                && !small(&t.b.bbox)
                && coco_class(class).is_some()
            {
                out.candidates.push(Loose {
                    target: t.b.bbox,
                    target_label: t.b.label.clone(),
                    class: class.clone(),
                    det: d.bbox,
                    det_score: d.score,
                });
            }
        }
    }
    // 6. Another class (before the adds: its detection is no add).
    for (ti, t) in train.iter().enumerate() {
        let Some(class) = &t.class else { continue };
        // A model without the label's class cannot say it is another one.
        if t.b.source == Source::Human || t.b.model_label.is_some() || !ctx.knows(class) {
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
                        "{} sees a {} here, score {:.2}, and no {class}",
                        ctx.who(),
                        d.class,
                        d.score
                    ),
                )
            });
        }
    }
    // 4. Missing boxes (from a lower score at a static spot).
    for (di, d) in dets.iter().enumerate() {
        if by_det.get(di).copied().flatten().is_some()
            || evidence.get(di).copied().unwrap_or(false)
            || d.score < ADD_SCORE_AT_SPOT.min(ADD_SCORE)
            || small(&d.bbox)
        {
            continue;
        }
        let spot = spot_of(ctx, frame, &d.class, &d.bbox);
        if spot.is_none() && d.score < ADD_SCORE {
            continue;
        }
        let covered = f.boxes.iter().any(|b| {
            (b.state != BoxState::Rejected || b.is_ignore())
                && ((same_object(&key_of(&b.label, wanted), &d.key) && covers(&b.bbox, &d.bbox))
                    // A quick check's model calls an object of a class it does not have by
                    // another name: a label of such a class explains a detection on that same
                    // box, never a smaller object in or beside it.
                    || (wanted.class_of(&b.label).is_some_and(|c| !ctx.knows(c))
                        && iou(&b.bbox, &d.bbox) >= UNKNOWN_CLASS_COVER_IOU))
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
        let (label, why, spot_id) = match spot {
            Some(s) => {
                // A weak find takes the spot's own label (a truck found at 0.4 where a car is
                // labelled is that car); a confident one keeps its class (a clear truck at a
                // car's place is a truck: another vehicle parked there).
                let at_spot = wanted.class_of(&s.label).unwrap_or(&d.class);
                let label = if d.score >= ADD_SCORE {
                    d.class.clone()
                } else {
                    at_spot.to_string()
                };
                let found = if label == at_spot {
                    "one".to_string()
                } else {
                    format!("a {label}")
                };
                let why = format!(
                    "a {at_spot} is labelled at this place on {} other frames of this camera; {} \
                     found {found} here, score {:.2}",
                    s.frames_without(frame.day),
                    ctx.who(),
                    d.score
                );
                (label, why, Some(s.id.clone()))
            }
            None => (
                d.class.clone(),
                format!(
                    "{} found a {} here, score {:.2}",
                    ctx.who(),
                    d.class,
                    d.score
                ),
                None,
            ),
        };
        found_list.push(Suggestion {
            bbox: Some(d.bbox),
            label: Some(label),
            score: Some(d.score),
            spot: spot_id,
            ..suggestion(Kind::Add, ctx.run, why)
        });
    }
    // 5. Labels no model found.
    for (ti, t) in train.iter().enumerate() {
        let Some(class) = &t.class else { continue };
        if by_box.get(ti).copied().flatten().is_some()
            || relabelled.get(ti).copied().unwrap_or(false)
            || t.b.source == Source::Human
            || t.b.orig_bbox.is_some()
            || !ctx.knows(class)
            || ctx.class_boxes.get(class).copied().unwrap_or(0) < MIN_CLASS_BOXES
            || small(&t.b.bbox)
        {
            continue;
        }
        let seen = dets
            .iter()
            .any(|d| same_object(&d.key, &t.key) && iou(&d.bbox, &t.b.bbox) >= VERIFY_IOU);
        if seen {
            continue;
        }
        let why = match ctx.model {
            None => format!("no model that never saw this camera-day found this {class}"),
            Some(_) => format!("{} did not find this {class}", ctx.who()),
        };
        found_list.push(Suggestion {
            target: Some(t.b.bbox),
            target_label: Some(t.b.label.clone()),
            ..suggestion(Kind::Verify, ctx.run, why)
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

/// The closer look's answer for one candidate: RF-DETR L's detections on the padded crop of the
/// pair, in frame coordinates, and where that crop lay (the script's own line; None when it does
/// not say, which [`NoTighten::CropEdge`] then cannot judge).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Look {
    pub dets: Vec<ZoomDet>,
    pub place: Option<CropPlace>,
}

/// Why the closer look's answer for a candidate proposes nothing ([`tighten_verdict`]), in the
/// order the rules are asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NoTighten {
    /// No box of the object agrees with the fold model's (none at all, or (a) fails).
    NoAgree,
    /// (b) `Z` fits the label: the fold model was imprecise, the label is fine.
    LabelFine,
    /// (e) Another box of the closer look fits the label: it found the label's object as labelled.
    LabelFound,
    /// (f) The label spans a second object the closer look tells apart.
    SpansTwo,
    /// (c) `Z` is too small to propose.
    Small,
    /// (g) `Z` ends on an edge of its crop inside the frame: the object goes on past it.
    CropEdge,
    /// (d) `Z` is the neighbour's box.
    Neighbour,
    /// The label is no candidate any more (changed, a reviewer's, too small), or the proposal
    /// would not be live.
    Other,
}

impl NoTighten {
    /// One word for it (the replay's tally).
    pub const fn as_str(self) -> &'static str {
        match self {
            NoTighten::NoAgree => "no_agree",
            NoTighten::LabelFine => "label_fine",
            NoTighten::LabelFound => "label_found",
            NoTighten::SpansTwo => "spans_two",
            NoTighten::Small => "small",
            NoTighten::CropEdge => "crop_edge",
            NoTighten::Neighbour => "neighbour",
            NoTighten::Other => "other",
        }
    }
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// Whether `z` ends within [`TIGHTEN_EDGE_PX`] of an edge of the crop at `place` that is not the
/// image's own edge: the object was cut there. False without a place.
fn cut_by_crop(z: &[f32; 4], place: Option<CropPlace>) -> bool {
    let Some(p) = place else {
        return false;
    };
    #[allow(clippy::cast_possible_truncation, reason = "pixels")]
    let c = p.crop.map(|v| v as f32);
    #[allow(clippy::cast_precision_loss, reason = "pixels")]
    let (w, h) = (p.size[0] as f32, p.size[1] as f32);
    let e = TIGHTEN_EDGE_PX;
    (c[0] > 0.0 && z[0] <= c[0] + e)
        || (c[1] > 0.0 && z[1] <= c[1] + e)
        || (c[2] < w && z[2] >= c[2] - e)
        || (c[3] < h && z[3] >= c[3] - e)
}

/// The `tighten` the closer look supports for candidate `c` ([`tighten_verdict`]), or None.
pub fn tighten(
    frame: &FrameInput<'_>,
    c: &Loose,
    look: &Look,
    ctx: &Context<'_>,
) -> Option<Suggestion> {
    tighten_verdict(frame, c, look, ctx).ok()
}

/// The `tighten` the closer look supports for candidate `c` (a [`Loose`] pair of `frame`, whose
/// file is the label file as it is now), from `look` ([`Look`]), or why none. Pure. The closer
/// look's boxes of the object are its detections of the same object ([`same_object`]) at
/// [`TIGHTEN_ZOOM_SCORE`] or more, clamped to the frame; `Z` is the one with the best IoU to the
/// fold model's `D` (ties: the higher score, then the earlier detection). A `tighten` to `Z` (score
/// `Z`'s) only when:
///
/// - (a) `Z` agrees with the fold model: IoU(`Z`, `D`) >= [`TIGHTEN_AGREE_IOU`];
/// - (b) the label is loose against `Z` too: IoU(`L`, `Z`) < [`GOOD_IOU`] (else the label was fine
///   and the fold model imprecise);
/// - (e) no box of the closer look fits the label: none at IoU(`L`, box) >= [`GOOD_IOU`] (it found
///   the label's object as labelled; `D` and `Z` are then another object, say a car parked behind
///   the labelled one that the fold model missed);
/// - (f) the label spans no second object the closer look tells apart: no other box of the closer
///   look, not `Z` again (IoU with `Z` under [`TIGHTEN_AGREE_IOU`] and not nested in it or around
///   it, [`TIGHTEN_SPAN_INSIDE`]), lies [`TIGHTEN_SPAN_INSIDE`] or more inside `L` (and is no larger
///   than `L`) without another box of the object on the frame covering it ([`covers`]): moving
///   the label onto `Z` would leave that object unlabelled;
/// - (c) `Z` is a proper box on the frame, not under the size floor nor [`FLOOR_PX`] at the fold
///   models' input (rule 4's floor);
/// - (g) `Z` was not cut by its crop: no edge of `Z` within [`TIGHTEN_EDGE_PX`] of an edge of the
///   crop that is not the image's edge (the object goes on past the padded union; the script
///   keeps its boxes to the crop);
/// - (d) `Z` is not another object's box: no other training box or ignore region of the object on
///   the frame overlaps `Z` more than `L` does (the two looks found the neighbour);
///
/// and the label is still a candidate: a training box of the frame, reviewed, no reviewer's box,
/// not under the floor, and the suggestion is [`refine::live`] ([`NoTighten::Other`] otherwise).
pub fn tighten_verdict(
    frame: &FrameInput<'_>,
    c: &Loose,
    look: &Look,
    ctx: &Context<'_>,
) -> Result<Suggestion, NoTighten> {
    let f = frame.file;
    if f.status != Status::Reviewed || !proper(&c.target) || !proper(&c.det) {
        return Err(NoTighten::Other);
    }
    let wanted = ctx.wanted;
    let (w, h) = (frame.width, frame.height);
    let floor = ctx.floor.filter(SizeFloor::is_on);
    let small =
        |b: &[f32; 4]| under_floor(b, w, h, floor) || short_px(b, w, h, ctx.imgsz) < FLOOR_PX;
    let key = key_of(&c.class, wanted);
    // The label: still a training box no reviewer touched, and big enough to judge.
    let at = f
        .boxes
        .iter()
        .position(|b| {
            b.bbox == c.target
                && b.label.trim().eq_ignore_ascii_case(c.target_label.trim())
                && b.is_training()
                && !b.is_ignore()
        })
        .ok_or(NoTighten::Other)?;
    let l = f.boxes.get(at).ok_or(NoTighten::Other)?;
    if reviewers(l) || small(&l.bbox) {
        return Err(NoTighten::Other);
    }
    let fit = |b: [f32; 4]| {
        if w > 0 && h > 0 {
            clamp_box(b, w, h)
        } else {
            b
        }
    };
    // The closer look's boxes of the object.
    let seen: Vec<(usize, [f32; 4], f32)> = look
        .dets
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            #[allow(clippy::cast_possible_truncation, reason = "a score and pixels")]
            let (score, bbox) = (d.score as f32, d.bbox.map(|v| v as f32));
            if !(score.is_finite() && score >= TIGHTEN_ZOOM_SCORE && proper(&bbox)) {
                return None;
            }
            if !same_object(&key_of(&d.label, wanted), &key) {
                return None;
            }
            let b = fit(bbox);
            proper(&b).then_some((i, b, score.min(1.0)))
        })
        .collect();
    let (zi, z, z_score) = seen
        .iter()
        .copied()
        .max_by(|a, b| {
            iou(&a.1, &c.det)
                .total_cmp(&iou(&b.1, &c.det))
                .then(a.2.total_cmp(&b.2))
                .then(b.0.cmp(&a.0))
        })
        .ok_or(NoTighten::NoAgree)?;
    // (a) Two looks agree ...
    if iou(&z, &c.det) < TIGHTEN_AGREE_IOU {
        return Err(NoTighten::NoAgree);
    }
    // (b) ... and not with the label.
    let lz = iou(&c.target, &z);
    if lz >= GOOD_IOU {
        return Err(NoTighten::LabelFine);
    }
    // (e) The closer look found the label's object as labelled: D and Z are another object.
    if seen.iter().any(|(_, b, _)| iou(&c.target, b) >= GOOD_IOU) {
        return Err(NoTighten::LabelFound);
    }
    // (f) The label spans a second object that no other box labels.
    let labelled_elsewhere = |b: &[f32; 4]| {
        f.boxes.iter().enumerate().any(|(i, x)| {
            i != at
                && (x.is_training() || x.is_ignore())
                && same_object(&key_of(&x.label, wanted), &key)
                && covers(&x.bbox, b)
        })
    };
    let second = seen.iter().any(|(i, b, _)| {
        *i != zi
            && iou(b, &z) < TIGHTEN_AGREE_IOU
            && inside_share(b, &z) < TIGHTEN_SPAN_INSIDE
            && area(b) <= area(&c.target)
            && inside_share(b, &c.target) >= TIGHTEN_SPAN_INSIDE
            && !labelled_elsewhere(b)
    });
    if second {
        return Err(NoTighten::SpansTwo);
    }
    // (c) A box worth proposing.
    if small(&z) {
        return Err(NoTighten::Small);
    }
    // (g) Not an object cut by its crop.
    if cut_by_crop(&z, look.place) {
        return Err(NoTighten::CropEdge);
    }
    // (d) Not the neighbour's box.
    let neighbour = f.boxes.iter().enumerate().any(|(i, b)| {
        i != at
            && (b.is_training() || b.is_ignore())
            && same_object(&key_of(&b.label, wanted), &key)
            && iou(&b.bbox, &z) > lz
    });
    if neighbour {
        return Err(NoTighten::Neighbour);
    }
    let s = Suggestion {
        target: Some(c.target),
        target_label: Some(c.target_label.clone()),
        bbox: Some(z),
        score: Some(z_score),
        ..suggestion(
            Kind::Tighten,
            ctx.run,
            format!(
                "{} (score {:.2}) and a closer look by a larger model (score {z_score:.2}) both \
                 put this {} here; the label overlaps that box at IoU {lz:.2}",
                match ctx.model {
                    None => "a model that never saw this camera-day".to_string(),
                    Some(m) => format!("{}, which never trained on this frame", m.name),
                },
                c.det_score,
                c.class
            ),
        )
    };
    let mut probe = f.clone();
    probe.refine = vec![s.clone()];
    if refine::live(&probe, Some(wanted)).is_empty() {
        return Err(NoTighten::Other);
    }
    Ok(s)
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
        run_at(f, dets, counts, None)
    }

    /// [`run_with`] with the size floor `floor`.
    fn run_at(
        f: &LabelFile,
        dets: &[Det],
        counts: &BTreeMap<String, usize>,
        floor: Option<SizeFloor>,
    ) -> Judged {
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let ctx = Context {
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: counts,
            run: "r1",
            spots: None,
            floor,
        };
        let j = judge(
            &FrameInput {
                file: f,
                width: 640,
                height: 480,
                dets,
                camera: "cam",
                day: "2026-10-01",
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

    /// Judge `f` with `dets` as a quick check's model `mine` that has `has` of [`classes`].
    fn run_model(f: &LabelFile, dets: &[Det], has: &[&str]) -> Judged {
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let m = ModelJudge {
            name: "mine".into(),
            classes: has.iter().map(|s| s.to_string()).collect(),
        };
        let counts = many();
        let ctx = Context {
            model: Some(&m),
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "q1",
            spots: None,
            floor: None,
        };
        judge(
            &FrameInput {
                file: f,
                width: 640,
                height: 480,
                dets,
                camera: "cam",
                day: "2026-10-01",
            },
            &ctx,
        )
    }

    #[test]
    fn a_quick_checks_model_names_itself_and_says_nothing_of_classes_it_lacks() {
        // With every class: the same findings as a fold model, in the model's words.
        let f = reviewed(vec![acc("car", CAR), acc("person", FAR)]);
        let all = ["person", "car", "truck", "dog"];
        let j = run_model(&f, &[det("person", [10.0, 10.0, 60.0, 120.0], 0.9)], &all);
        assert_eq!(kinds(&j), [Kind::Add, Kind::Verify, Kind::Verify], "{j:?}");
        assert_eq!(
            j.suggestions[0].why,
            "mine (never trained on this frame) found a person here, score 0.90"
        );
        assert_eq!(
            j.suggestions[1].why,
            "mine (never trained on this frame) did not find this car"
        );
        assert!(j.suggestions.iter().all(|s| s.run == "q1"));
        assert_eq!(j.boxes.len(), 2);
        // A dog label the fold models would relabel as a person ...
        let dog = reviewed(vec![acc("dog", CAR)]);
        let person_on_it = [det("person", NEAR, 0.95)];
        let folds = run(&dog, &person_on_it);
        assert_eq!(kinds(&folds), [Kind::Relabel], "{folds:?}");
        // ... is no relabel for a model without dogs, nor an add of a person on top of it, nor a
        // verify (its silence on a dog is no miss), and not in the recall.
        let j = run_model(&dog, &person_on_it, &["person", "car", "truck"]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert!(j.boxes.is_empty(), "{j:?}");
        let j = run_model(&dog, &[], &["person", "car", "truck"]);
        assert!(j.suggestions.is_empty(), "{j:?}");
        // Elsewhere on the frame its person is still an add.
        let j = run_model(
            &dog,
            &[det("person", FAR, 0.9)],
            &["person", "car", "truck"],
        );
        assert_eq!(kinds(&j), [Kind::Add], "{j:?}");
        // A model with dogs but no persons: the relabel needs the label's class only.
        let j = run_model(&dog, &person_on_it, &["dog"]);
        assert_eq!(kinds(&j), [Kind::Relabel], "{j:?}");
        assert!(
            j.suggestions[0]
                .why
                .starts_with("mine (never trained on this frame) sees a person here"),
            "{j:?}"
        );
    }

    #[test]
    fn a_label_of_a_class_the_model_lacks_hides_only_a_detection_of_its_own_box() {
        // A model of person and car; the list also wants truck and dog.
        let has = ["person", "car"];
        let truck = [100.0, 50.0, 500.0, 400.0];
        // A person standing in front of the truck: wholly inside its box, IoU small.
        let person = [150.0, 150.0, 200.0, 300.0];
        assert!(iou(&truck, &person) < 0.1 && inside_share(&person, &truck) > 0.99);
        let f = reviewed(vec![acc("truck", truck)]);
        let j = run_model(&f, &[det("person", person, 0.9)], &has);
        assert_eq!(kinds(&j), [Kind::Add], "{j:?}");
        assert_eq!(j.suggestions[0].label.as_deref(), Some("person"));
        // The same for an unsure (ignore) region of such a class.
        let mut unsure = acc("truck", truck);
        unsure.difficult = true;
        let j = run_model(&reviewed(vec![unsure]), &[det("person", person, 0.9)], &has);
        assert_eq!(kinds(&j), [Kind::Add], "{j:?}");
        // A detection about a dog label's own box (the model, without dogs, calls it a person):
        // no add, on a label or an unsure region alike.
        let dog = [100.0, 50.0, 500.0, 400.0];
        let on_it = [110.0, 60.0, 500.0, 400.0];
        assert!(iou(&dog, &on_it) >= 0.9);
        let d = reviewed(vec![acc("dog", dog)]);
        let j = run_model(&d, &[det("person", on_it, 0.9)], &has);
        assert!(j.suggestions.is_empty(), "{j:?}");
        let mut unsure_dog = acc("dog", dog);
        unsure_dog.difficult = true;
        let j = run_model(
            &reviewed(vec![unsure_dog]),
            &[det("person", on_it, 0.9)],
            &has,
        );
        assert!(j.suggestions.is_empty(), "{j:?}");
        // Under the cut (although half of it and more lies inside the dog's box): another object.
        let part = [100.0, 50.0, 290.0, 400.0];
        assert!(iou(&dog, &part) < UNKNOWN_CLASS_COVER_IOU && covers(&dog, &part));
        let j = run_model(&d, &[det("person", part, 0.9)], &has);
        assert_eq!(kinds(&j), [Kind::Add], "{j:?}");
    }

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
    fn the_configured_size_floor_is_the_one_rule_and_its_boxes_are_ignore_regions() {
        // At 384 a 14 px wide box of a 640 frame is 8.4 px: under a 10 px floor, though above
        // FLOOR_PX at the fold models' 640.
        let floor = Some(SizeFloor {
            px: 10.0,
            input: 384,
        });
        let small = [100.0, 100.0, 114.0, 200.0];
        let big = [300.0, 100.0, 340.0, 200.0];
        assert!(short_px(&small, 640, 480, 640) >= FLOOR_PX);
        assert!(under_floor(&small, 640, 480, floor));
        let many = many();
        // Without the floor: a verify on the small label, as before.
        let f = reviewed(vec![acc("person", small), acc("person", big)]);
        let j = run_with(&f, &[], &many);
        assert_eq!(kinds(&j), [Kind::Verify, Kind::Verify]);
        assert!(j.boxes.iter().all(|b| !b.under_floor));
        // With it: an ignore region, no verify; in the facts only for the size table.
        let j = run_at(&f, &[], &many, floor);
        assert_eq!(kinds(&j), [Kind::Verify]);
        assert_eq!(j.suggestions[0].target, Some(big));
        assert_eq!(
            j.boxes.iter().map(|b| b.under_floor).collect::<Vec<_>>(),
            [true, false]
        );
        // A detection under it is no add; one above it is.
        let empty = reviewed(vec![]);
        assert!(
            run_at(&empty, &[det("person", small, 0.99)], &many, floor)
                .suggestions
                .is_empty()
        );
        assert_eq!(
            kinds(&run_at(&empty, &[det("person", big, 0.99)], &many, floor)),
            [Kind::Add]
        );
        // The small label still covers its object: a larger detection on it adds nothing. At IoU
        // 0.58 that pair would be a loose label; an ignore region is never one.
        let on_it = [100.0, 100.0, 124.0, 200.0];
        assert!(!under_floor(&on_it, 640, 480, floor));
        let tiny_label = reviewed(vec![acc("person", small)]);
        let j = run_at(&tiny_label, &[det("person", on_it, 0.99)], &many, floor);
        assert!(j.suggestions.is_empty(), "{:?}", j.suggestions);
        assert_eq!(j.loose, 0, "an ignore region is never a loose label");
        assert_eq!(
            run_with(&tiny_label, &[det("person", on_it, 0.99)], &many).loose,
            1,
            "without the floor it is one"
        );
        // Nor does another object's detection on it relabel it.
        let car = run_at(&tiny_label, &[det("car", small, 0.95)], &many, floor);
        assert!(car.suggestions.is_empty(), "{:?}", car.suggestions);
        assert_eq!(
            kinds(&run_with(&tiny_label, &[det("car", small, 0.95)], &many)),
            [Kind::Relabel]
        );
        // A floor that is off is no floor: FLOOR_PX at the input again.
        let off = Some(SizeFloor {
            px: 0.0,
            input: 384,
        });
        assert_eq!(
            kinds(&run_at(&f, &[], &many, off)),
            [Kind::Verify, Kind::Verify]
        );
    }

    #[test]
    fn a_permissive_floor_keeps_the_noise_rule_of_the_fold_models_input() {
        // 9 px wide on a 640 frame: 9 px at the fold models' 640, under FLOOR_PX, but over a
        // permissive floor of 8 px at 640.
        let permissive = Some(SizeFloor {
            px: 8.0,
            input: 640,
        });
        let thin = [100.0, 100.0, 109.0, 300.0];
        assert!(short_px(&thin, 640, 480, 640) < FLOOR_PX);
        assert!(!under_floor(&thin, 640, 480, permissive));
        let many = many();
        // A label no model found: no verify, with or without the floor.
        let f = reviewed(vec![acc("person", thin)]);
        assert!(run_with(&f, &[], &many).suggestions.is_empty());
        let j = run_at(&f, &[], &many, permissive);
        assert!(j.suggestions.is_empty(), "{:?}", j.suggestions);
        assert_eq!(
            j.boxes.iter().map(|b| b.under_floor).collect::<Vec<_>>(),
            [false],
            "not an ignore region: the floor alone decides that"
        );
        // A detection that thin: no add either.
        let empty = reviewed(vec![]);
        assert!(
            run_at(&empty, &[det("person", thin, 0.99)], &many, permissive)
                .suggestions
                .is_empty()
        );
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
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: None,
            floor: None,
        };
        let dets = [det("person", [600.0, 400.0, 700.0, 520.0], 0.9)];
        let j = judge(
            &FrameInput {
                file: &f,
                width: 0,
                height: 0,
                dets: &dets,
                camera: "cam",
                day: "2026-10-01",
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
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: None,
            floor: None,
        };
        let f = reviewed(vec![acc("dog", CAR)]);
        let dets = [det("dog", FAR, 0.9)];
        let j = judge(
            &FrameInput {
                file: &f,
                width: 640,
                height: 480,
                dets: &dets,
                camera: "cam",
                day: "2026-10-01",
            },
            &ctx,
        );
        assert_eq!(j, Judged::default());
    }

    /// A spot table with a car labelled at [`CAR`] (of a 640 x 480 frame) on `frames` frames of
    /// camera `cam`: `on_day` of them on 2026-10-01 (the judged frames' day), the others spread
    /// over 2026-10-02 to -04.
    fn car_spot(frames: usize, on_day: usize) -> statics::Table {
        let nb = statics::normalised(&CAR, 640, 480).unwrap();
        statics::Table::build((0..frames).map(|i| statics::Obs {
            camera: "cam".into(),
            frame: format!("cam/{i}"),
            day: if i < on_day {
                "2026-10-01".to_string()
            } else {
                format!("2026-10-0{}", i % 3 + 2)
            },
            scene: "day".into(),
            label: if i % 4 == 0 { "truck" } else { "car" }.into(),
            bbox: nb,
        }))
    }

    /// Judge `f` (640 x 480, camera `cam`, 2026-10-01) with `dets` and the spot table `spots`.
    fn run_spots(f: &LabelFile, dets: &[Det], spots: &statics::Table) -> Judged {
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let counts = many();
        let ctx = Context {
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: Some(spots),
            floor: None,
        };
        let j = judge(
            &FrameInput {
                file: f,
                width: 640,
                height: 480,
                dets,
                camera: "cam",
                day: "2026-10-01",
            },
            &ctx,
        );
        let mut probe = f.clone();
        probe.refine = j.suggestions.clone();
        assert_eq!(refine::live(&probe, Some(&w)).len(), j.suggestions.len());
        refine::check(&j.suggestions).unwrap();
        j
    }

    #[test]
    fn a_detection_at_a_static_spot_is_an_add_from_a_lower_score() {
        let table = car_spot(12, 0);
        let spot = &table.camera("cam")[0];
        assert!(spot.holds_without("2026-10-01"));
        let f = reviewed(vec![]);
        // 0.35 at the spot: an add with the spot's id and its own words.
        let j = run_spots(&f, &[det("car", NEAR, 0.35)], &table);
        assert_eq!(kinds(&j), [Kind::Add], "{j:?}");
        let s = &j.suggestions[0];
        assert_eq!(s.spot.as_deref(), Some(spot.id.as_str()));
        assert_eq!((s.label.as_deref(), s.score), (Some("car"), Some(0.35)));
        assert_eq!(
            s.why,
            "a car is labelled at this place on 12 other frames of this camera; a model that \
             never saw this camera-day found one here, score 0.35"
        );
        // Without the table, or under 0.3: nothing.
        assert!(run(&f, &[det("car", NEAR, 0.35)]).suggestions.is_empty());
        assert!(
            run_spots(&f, &[det("car", NEAR, ADD_SCORE_AT_SPOT - 0.01)], &table)
                .suggestions
                .is_empty()
        );
        // Elsewhere the cut-off stays 0.5.
        let elsewhere = [300.0, 300.0, 380.0, 360.0];
        assert!(
            run_spots(&f, &[det("car", elsewhere, 0.45)], &table)
                .suggestions
                .is_empty()
        );
        let j = run_spots(&f, &[det("car", elsewhere, 0.55)], &table);
        assert_eq!(kinds(&j), [Kind::Add]);
        assert_eq!(j.suggestions[0].spot, None);
        // A weak truck where the car is labelled: the spot's label.
        let j = run_spots(&f, &[det("truck", NEAR, 0.4)], &table);
        assert_eq!(j.suggestions[0].label.as_deref(), Some("car"));
        // A confident truck there is a truck (another vehicle parked at the car's place), and
        // says so; at exactly ADD_SCORE it is confident.
        for score in [ADD_SCORE, 0.8] {
            let j = run_spots(&f, &[det("truck", NEAR, score)], &table);
            let s = &j.suggestions[0];
            assert_eq!(s.label.as_deref(), Some("truck"), "{score}");
            assert_eq!(s.spot.as_deref(), Some(spot.id.as_str()));
            assert!(
                s.why.starts_with(
                    "a car is labelled at this place on 12 other frames of this camera; a model \
                     that never saw this camera-day found a truck here, score"
                ),
                "{}",
                s.why
            );
        }
        // A strong find at the spot says so too, at the same strength as one elsewhere.
        let j = run_spots(&f, &[det("car", NEAR, 0.8)], &table);
        assert_eq!(j.suggestions[0].spot.as_deref(), Some(spot.id.as_str()));
        assert!(
            j.suggestions[0]
                .why
                .starts_with("a car is labelled at this place")
        );
        let plain = run(&f, &[det("car", NEAR, 0.8)]);
        assert_eq!(
            refine::strength(&j.suggestions[0]),
            refine::strength(&plain.suggestions[0])
        );
        // A person at the car's place is another object: no spot for it.
        assert!(
            run_spots(&reviewed(vec![]), &[det("person", NEAR, 0.4)], &table)
                .suggestions
                .is_empty()
        );
    }

    #[test]
    fn a_person_labelled_at_one_place_lowers_no_cut_off() {
        // A person labelled at CAR's place on 12 frames of 3 days: no spot (people never park).
        let nb = statics::normalised(&CAR, 640, 480).unwrap();
        let persons = statics::Table::build((0..12).map(|i| statics::Obs {
            camera: "cam".into(),
            frame: format!("cam/{i}"),
            day: format!("2026-10-0{}", i % 3 + 2),
            scene: "day".into(),
            label: "person".into(),
            bbox: nb,
        }));
        assert!(persons.is_empty(), "{persons:?}");
        let f = reviewed(vec![]);
        assert!(
            run_spots(&f, &[det("person", NEAR, 0.35)], &persons)
                .suggestions
                .is_empty()
        );
        // From 0.5 it is an ordinary add, with no spot.
        let j = run_spots(&f, &[det("person", NEAR, 0.6)], &persons);
        assert_eq!(kinds(&j), [Kind::Add]);
        assert_eq!(j.suggestions[0].spot, None);
        // A person's training box is never counted at a spot, the car's still is.
        let j = run_spots(&reviewed(vec![acc("person", CAR)]), &[], &car_spot(12, 0));
        assert!(!j.boxes[0].at_spot);
        let j = run_spots(&reviewed(vec![acc("car", CAR)]), &[], &car_spot(12, 0));
        assert!(j.boxes[0].at_spot);
    }

    #[test]
    fn the_add_rules_still_hold_at_a_spot() {
        let table = car_spot(12, 0);
        // Labelled: a match, no add.
        let f = reviewed(vec![acc("car", CAR)]);
        let j = run_spots(&f, &[det("car", NEAR, 0.35)], &table);
        assert!(j.suggestions.is_empty(), "{j:?}");
        assert!(j.boxes[0].at_spot);
        // A judged reject at the place blocks it.
        let fp = LabelBox {
            reject_reason: Some(RejectReason::FalsePositive),
            ..b("car", CAR, BoxState::Rejected)
        };
        assert!(
            run_spots(&reviewed(vec![fp]), &[det("car", NEAR, 0.35)], &table)
                .suggestions
                .is_empty()
        );
        // Too small at model input: nothing, wherever.
        let tiny = [100.0, 100.0, 108.0, 160.0];
        assert!(
            run_spots(&reviewed(vec![]), &[det("car", tiny, 0.9)], &table)
                .suggestions
                .is_empty()
        );
        // A box elsewhere is not at a spot.
        let j = run_spots(&reviewed(vec![acc("car", FAR)]), &[], &table);
        assert!(!j.boxes[0].at_spot);
    }

    #[test]
    fn a_spot_made_by_the_frames_own_day_is_no_evidence() {
        // 12 frames, 9 of them of the judged frame's day: without that day 3 frames are left.
        let table = car_spot(12, 9);
        let spot = &table.camera("cam")[0];
        assert!(!spot.holds_without("2026-10-01"));
        let f = reviewed(vec![]);
        assert!(
            run_spots(&f, &[det("car", NEAR, 0.35)], &table)
                .suggestions
                .is_empty()
        );
        // A strong find is an ordinary add.
        let j = run_spots(&f, &[det("car", NEAR, 0.6)], &table);
        assert_eq!(kinds(&j), [Kind::Add]);
        assert_eq!(j.suggestions[0].spot, None);
        assert!(j.suggestions[0].why.starts_with("a model that never saw"));
        // Its training box there does not count as at a spot either.
        let j = run_spots(&reviewed(vec![acc("car", CAR)]), &[], &table);
        assert!(!j.boxes[0].at_spot);
        // Another camera has no spot.
        let c = classes();
        let w = Wanted {
            classes: &c,
            hidden: &[],
        };
        let counts = many();
        let ok = car_spot(12, 0);
        let ctx = Context {
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: Some(&ok),
            floor: None,
        };
        let dets = [det("car", NEAR, 0.35)];
        let j = judge(
            &FrameInput {
                file: &f,
                width: 640,
                height: 480,
                dets: &dets,
                camera: "drive",
                day: "2026-10-01",
            },
            &ctx,
        );
        assert!(j.suggestions.is_empty());
    }

    // ---- the closer look (`tighten`)

    fn zd(label: &str, bbox: [f32; 4], score: f64) -> ZoomDet {
        ZoomDet {
            label: label.into(),
            bbox: bbox.map(f64::from),
            score,
        }
    }

    /// [`tighten`] on `f` (640 x 480, input 640) for `c` with `zoom` (no crop place), under `floor`.
    fn refit(
        f: &LabelFile,
        c: &Loose,
        zoom: &[ZoomDet],
        floor: Option<SizeFloor>,
    ) -> Option<Suggestion> {
        verdict(f, c, zoom, None, floor).ok()
    }

    /// [`tighten_verdict`] on `f` (640 x 480, input 640) for `c` with `zoom` cut from `place`.
    fn verdict(
        f: &LabelFile,
        c: &Loose,
        zoom: &[ZoomDet],
        place: Option<CropPlace>,
        floor: Option<SizeFloor>,
    ) -> Result<Suggestion, NoTighten> {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let counts = many();
        let ctx = Context {
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: None,
            floor,
        };
        let s = tighten_verdict(
            &FrameInput {
                file: f,
                width: 640,
                height: 480,
                dets: &[],
                camera: "cam",
                day: "2026-10-01",
            },
            c,
            &Look {
                dets: zoom.to_vec(),
                place,
            },
            &ctx,
        );
        if let Ok(s) = &s {
            refine::check(std::slice::from_ref(s)).unwrap();
            assert_eq!(s.run, "r1");
        }
        s
    }

    /// The loose label [`LOOSE`] (a car) and the fold model's [`CAR`] at 0.9: IoU 0.5.
    fn loose_car() -> (LabelFile, Loose) {
        let f = reviewed(vec![acc("car", LOOSE)]);
        let j = run(&f, &[det("car", CAR, 0.9)]);
        assert_eq!(j.loose, 1);
        assert_eq!(j.candidates.len(), 1, "{j:?}");
        (f, j.candidates[0].clone())
    }

    #[test]
    fn a_loose_pair_with_a_confident_detection_is_a_candidate() {
        let (_, c) = loose_car();
        assert_eq!(
            c,
            Loose {
                target: LOOSE,
                target_label: "car".into(),
                class: "car".into(),
                det: CAR,
                det_score: 0.9,
            }
        );
        assert_eq!(c.union(), [100.0, 100.0, 300.0, 180.0]);
        // The fold model's score decides: at TIGHTEN_FOLD_SCORE yes, under it loose only.
        let f = reviewed(vec![acc("car", LOOSE)]);
        let j = run(&f, &[det("car", CAR, TIGHTEN_FOLD_SCORE)]);
        assert_eq!((j.loose, j.candidates.len()), (1, 1));
        let j = run(&f, &[det("car", CAR, 0.49)]);
        assert_eq!((j.loose, j.candidates.len()), (1, 0), "low fold score");
        // A truck detection of a car label is one object: its pair, kept with the label's class.
        let j = run(&f, &[det("truck", CAR, 0.9)]);
        assert_eq!(j.candidates[0].class, "car");
        // A good label is no candidate.
        let j = run(&reviewed(vec![acc("car", CAR)]), &[det("car", NEAR, 0.9)]);
        assert!(j.candidates.is_empty());
        // A reviewer's box is never one.
        for edit in [
            |b: &mut LabelBox| b.source = Source::Human,
            |b: &mut LabelBox| b.orig_bbox = Some([0.0, 0.0, 9.0, 9.0]),
            |b: &mut LabelBox| b.model_label = Some("truck".into()),
        ] {
            let mut f = reviewed(vec![acc("car", LOOSE)]);
            edit(&mut f.boxes[0]);
            let j = run(&f, &[det("car", CAR, 0.95)]);
            assert!(j.candidates.is_empty(), "{f:?}");
        }
        // An unreviewed frame has none.
        let mut f = reviewed(vec![acc("car", LOOSE)]);
        f.status = Status::Unreviewed;
        assert!(run(&f, &[det("car", CAR, 0.9)]).candidates.is_empty());
    }

    #[test]
    fn a_small_label_an_ignore_region_or_a_class_rf_detr_cannot_see_is_no_candidate() {
        // Under FLOOR_PX at the fold models' input: loose, no candidate.
        let label = [100.0, 100.0, 109.0, 200.0];
        let d = [100.0, 100.0, 109.0, 150.0];
        assert!(short_px(&label, 640, 480, 640) < FLOOR_PX);
        let j = run(
            &reviewed(vec![acc("person", label)]),
            &[det("person", d, 0.9)],
        );
        assert_eq!((j.loose, j.candidates.len()), (1, 0));
        // Under the configured floor: an ignore region, not even loose.
        let floor = Some(SizeFloor {
            px: 10.0,
            input: 384,
        });
        let label = [100.0, 100.0, 114.0, 200.0];
        let d = [100.0, 100.0, 124.0, 200.0];
        let f = reviewed(vec![acc("person", label)]);
        let j = run_at(&f, &[det("person", d, 0.99)], &many(), floor);
        assert_eq!((j.loose, j.candidates.len()), (0, 0));
        assert_eq!(
            run_with(&f, &[det("person", d, 0.99)], &many())
                .candidates
                .len(),
            1
        );
        // A class outside RF-DETR L's label space: the closer look cannot see it.
        let cl: Vec<String> = ["van", "car"].map(String::from).to_vec();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let counts = many();
        let ctx = Context {
            model: None,
            wanted: &w,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: None,
            floor: None,
        };
        let at = |label: &str| {
            let f = reviewed(vec![acc(label, LOOSE)]);
            let dets = [det(label, CAR, 0.9)];
            judge(
                &FrameInput {
                    file: &f,
                    width: 640,
                    height: 480,
                    dets: &dets,
                    camera: "cam",
                    day: "2026-10-01",
                },
                &ctx,
            )
        };
        assert!(coco_class("van").is_none());
        let van = at("van");
        assert_eq!((van.loose, van.candidates.len()), (1, 0));
        assert_eq!(at("car").candidates.len(), 1);
    }

    #[test]
    fn two_looks_that_agree_propose_the_closer_looks_box() {
        let (f, c) = loose_car();
        let s = refit(&f, &c, &[zd("car", NEAR, 0.83)], None).unwrap();
        assert_eq!(s.kind, Kind::Tighten);
        assert_eq!(s.target, Some(LOOSE));
        assert_eq!(s.target_label.as_deref(), Some("car"));
        assert_eq!(s.bbox, Some(NEAR));
        assert_eq!(s.score, Some(0.83));
        assert_eq!(s.label, None);
        assert_eq!(
            s.why,
            "a model that never saw this camera-day (score 0.90) and a closer look by a larger \
             model (score 0.83) both put this car here; the label overlaps that box at IoU 0.48"
        );
        // A truck of the closer look is the car's box too; the best IoU to D wins, then the
        // higher score; an unwanted label of another object is no box of it. (`off` half outside
        // the label: one mostly inside it would be a second object the label spans.)
        let off = [140.0, 60.0, 240.0, 140.0];
        assert!(inside_share(&off, &LOOSE) < TIGHTEN_SPAN_INSIDE);
        let s = refit(
            &f,
            &c,
            &[
                zd("truck", off, 0.99),
                zd("person", CAR, 0.99),
                zd("car", NEAR, 0.6),
                zd("truck", NEAR, 0.7),
            ],
            None,
        )
        .unwrap();
        assert_eq!((s.bbox, s.score), (Some(NEAR), Some(0.7)));
        // A box past the frame's edge is clamped onto it.
        let (l2, d2) = ([0.0, 100.0, 300.0, 180.0], [0.0, 100.0, 190.0, 180.0]);
        let (wide, clamped) = ([-10.0, 100.0, 195.0, 180.0], [0.0, 100.0, 195.0, 180.0]);
        assert!(iou(&clamped, &d2) >= TIGHTEN_AGREE_IOU && iou(&clamped, &l2) < GOOD_IOU);
        let f2 = reviewed(vec![acc("car", l2)]);
        let c2 = Loose {
            target: l2,
            det: d2,
            ..c.clone()
        };
        let s = refit(&f2, &c2, &[zd("car", wide, 0.9)], None).unwrap();
        assert_eq!(s.bbox, Some(clamped));
    }

    #[test]
    fn no_proposal_without_a_closer_look_that_agrees_with_the_fold_model() {
        let (f, c) = loose_car();
        // No detection, too weak, of another object.
        assert!(refit(&f, &c, &[], None).is_none());
        assert!(refit(&f, &c, &[zd("car", NEAR, 0.49)], None).is_none());
        assert!(refit(&f, &c, &[zd("person", NEAR, 0.95)], None).is_none());
        // One that disagrees with D: IoU under TIGHTEN_AGREE_IOU.
        let half = [150.0, 100.0, 250.0, 180.0];
        assert!(iou(&half, &CAR) < TIGHTEN_AGREE_IOU);
        assert!(refit(&f, &c, &[zd("car", half, 0.95)], None).is_none());
        // The constants are the overlap check's, the same model on the same crops.
        use crate::collect::labels::overlaps::{ZOOM_MATCH_IOU, ZOOM_SCORE};
        assert_eq!(f64::from(TIGHTEN_ZOOM_SCORE), ZOOM_SCORE);
        assert!((f64::from(TIGHTEN_AGREE_IOU) - ZOOM_MATCH_IOU).abs() < 1e-6);
    }

    #[test]
    fn a_closer_look_that_agrees_with_the_label_says_the_label_was_fine() {
        let (f, c) = loose_car();
        // IoU 0.667 with D (agrees with the fold model) and 0.75 with L: the label is fine.
        let z = [100.0, 100.0, 250.0, 180.0];
        assert!(iou(&z, &CAR) >= TIGHTEN_AGREE_IOU);
        assert!((iou(&z, &LOOSE) - GOOD_IOU).abs() < 1e-6);
        assert!(refit(&f, &c, &[zd("car", z, 0.9)], None).is_none());
        // Just under GOOD_IOU: proposed.
        let z = [100.0, 100.0, 249.0, 180.0];
        assert!(iou(&z, &LOOSE) < GOOD_IOU);
        assert!(refit(&f, &c, &[zd("car", z, 0.9)], None).is_some());
    }

    #[test]
    fn the_neighbours_box_is_not_proposed() {
        // A second car labelled where the closer look's box is: it fits that box better than the
        // loose label does, so the two looks found the neighbour.
        let neighbour = [101.0, 100.0, 203.0, 182.0];
        let (_, c) = loose_car();
        let f = reviewed(vec![acc("car", LOOSE), acc("car", neighbour)]);
        assert!(iou(&neighbour, &NEAR) > iou(&LOOSE, &NEAR));
        assert!(refit(&f, &c, &[zd("car", NEAR, 0.9)], None).is_none());
        // An unsure box there counts as the neighbour too; a rejected one or another object's
        // does not.
        let unsure = LabelBox {
            difficult: true,
            ..acc("truck", neighbour)
        };
        let f = reviewed(vec![acc("car", LOOSE), unsure]);
        assert!(refit(&f, &c, &[zd("car", NEAR, 0.9)], None).is_none());
        for other in [
            b("car", neighbour, BoxState::Rejected),
            acc("person", neighbour),
        ] {
            let f = reviewed(vec![acc("car", LOOSE), other]);
            assert!(refit(&f, &c, &[zd("car", NEAR, 0.9)], None).is_some());
        }
    }

    /// The candidate of a car label `l` the fold model found at `d` (score 0.9).
    fn pair_of(l: [f32; 4], d: [f32; 4]) -> Loose {
        Loose {
            target: l,
            target_label: "car".into(),
            class: "car".into(),
            det: d,
            det_score: 0.9,
        }
    }

    #[test]
    fn a_closer_look_that_finds_the_label_as_labelled_moves_no_label_to_another_object() {
        // Two overlapping parked cars, only car 1 labelled (correctly). The fold model misses car
        // 1 and finds car 2, which the one-to-one matching pairs with car 1's label as "loose".
        let car1 = CAR;
        let car2 = [150.0, 100.0, 250.0, 180.0];
        assert!((iou(&car1, &car2) - 1.0 / 3.0).abs() < 1e-4);
        let f = reviewed(vec![acc("car", car1)]);
        let j = run(&f, &[det("car", car2, 0.9)]);
        assert_eq!((j.loose, j.candidates.len()), (1, 1), "{j:?}");
        assert!(
            j.suggestions.is_empty(),
            "car 2's detection is the label's pair: no add"
        );
        let c = j.candidates[0].clone();
        assert_eq!(c, pair_of(car1, car2));
        // The closer look sees both: Z1 on the label, Z2 on car 2 (the best IoU to D, so `Z`).
        let z1 = [101.0, 100.0, 201.0, 181.0];
        let z2 = [151.0, 100.0, 251.0, 180.0];
        assert!(iou(&z1, &car1) >= GOOD_IOU && iou(&z2, &car2) >= TIGHTEN_AGREE_IOU);
        assert!(iou(&z2, &car1) < GOOD_IOU);
        let both = [zd("car", z2, 0.9), zd("car", z1, 0.8)];
        assert_eq!(
            verdict(&f, &c, &both, None, None).unwrap_err(),
            NoTighten::LabelFound
        );
        // A truck of the closer look on the label is the car found as labelled too; one under
        // TIGHTEN_ZOOM_SCORE is no box.
        let truck = [zd("car", z2, 0.9), zd("truck", z1, 0.6)];
        assert_eq!(
            verdict(&f, &c, &truck, None, None).unwrap_err(),
            NoTighten::LabelFound
        );
        let faint = [zd("car", z2, 0.9), zd("car", z1, 0.49)];
        assert!(verdict(&f, &c, &faint, None, None).is_ok());
        // Seeing car 2 alone, the closer look cannot tell (the limit of one crop's evidence).
        assert!(verdict(&f, &c, &both[..1], None, None).is_ok());
    }

    #[test]
    fn a_label_around_two_objects_the_closer_look_tells_apart_is_not_moved_onto_one() {
        // One label drawn around two cars; the fold model finds car 2, the closer look both.
        let l = LOOSE;
        let car2 = [200.0, 100.0, 300.0, 180.0];
        assert!((iou(&l, &car2) - 0.5).abs() < 1e-6);
        let f = reviewed(vec![acc("car", l)]);
        let c = pair_of(l, car2);
        let z2 = [201.0, 100.0, 300.0, 181.0];
        let z1 = [100.0, 102.0, 195.0, 178.0];
        assert!(inside_share(&z1, &l) >= TIGHTEN_SPAN_INSIDE);
        assert!(iou(&z1, &l) < GOOD_IOU, "not the label found as labelled");
        let zoom = [zd("car", z2, 0.9), zd("car", z1, 0.85)];
        assert_eq!(
            verdict(&f, &c, &zoom, None, None).unwrap_err(),
            NoTighten::SpansTwo
        );
        // Car 1 mostly outside the label: not spanned.
        let out = [40.0, 102.0, 135.0, 178.0];
        assert!(inside_share(&out, &l) < TIGHTEN_SPAN_INSIDE);
        let s = verdict(
            &f,
            &c,
            &[zd("car", z2, 0.9), zd("car", out, 0.85)],
            None,
            None,
        )
        .unwrap();
        assert_eq!(s.bbox, Some(z2));
        // Car 1 labelled by a box of its own: the move leaves nothing unlabelled.
        let own = [100.0, 100.0, 196.0, 180.0];
        let f2 = reviewed(vec![acc("car", l), acc("car", own)]);
        assert!(verdict(&f2, &c, &zoom, None, None).is_ok());
        // A second box of Z's own object (a truck on the same car, or a part of it) is no second
        // object.
        for copy in [[200.0, 100.0, 299.0, 180.0], [230.0, 110.0, 290.0, 170.0]] {
            let zoom = [zd("car", z2, 0.9), zd("truck", copy, 0.8)];
            assert!(verdict(&f, &c, &zoom, None, None).is_ok(), "{copy:?}");
        }
        // Nor is one under TIGHTEN_ZOOM_SCORE, or of another object.
        for other in [zd("car", z1, 0.4), zd("person", z1, 0.95)] {
            let zoom = [zd("car", z2, 0.9), other];
            assert!(verdict(&f, &c, &zoom, None, None).is_ok());
        }
        // The share is the overlap check's nested share.
        assert_eq!(TIGHTEN_SPAN_INSIDE, stacks::NESTED_SHARE);
    }

    #[test]
    fn a_box_the_crop_cut_off_is_not_proposed() {
        let (f, c) = loose_car();
        let place = |crop: [f64; 4], size: [u32; 2]| Some(CropPlace { crop, size });
        let zoom = [zd("car", NEAR, 0.9)];
        // The script's crop: the union [100, 100, 300, 180] padded by 20%.
        let ok = place([60.0, 84.0, 340.0, 196.0], [640, 480]);
        assert!(verdict(&f, &c, &zoom, ok, None).is_ok());
        // NEAR = [102, 101, 201, 182]: an edge within a pixel of a crop edge inside the frame is a
        // cut; a pixel and more away is not.
        let cases: [([f64; 4], bool); 6] = [
            ([101.0, 84.0, 340.0, 196.0], true),
            ([100.5, 84.0, 340.0, 196.0], false),
            ([60.0, 100.0, 340.0, 196.0], true),
            ([60.0, 84.0, 202.0, 196.0], true),
            ([60.0, 84.0, 340.0, 183.0], true),
            ([60.0, 84.0, 340.0, 183.5], false),
        ];
        for (crop, cut) in cases {
            let v = verdict(&f, &c, &zoom, place(crop, [640, 480]), None);
            assert_eq!(v.err(), cut.then_some(NoTighten::CropEdge), "{crop:?}");
        }
        // The crop's edge is the image's: no cut. A car at the frame's left edge, the crop from 0.
        let (l2, d2) = ([0.0, 100.0, 300.0, 180.0], [0.0, 100.0, 190.0, 180.0]);
        let f2 = reviewed(vec![acc("car", l2)]);
        let c2 = pair_of(l2, d2);
        let z = [zd("car", [0.0, 100.0, 195.0, 180.0], 0.9)];
        assert!(
            verdict(
                &f2,
                &c2,
                &z,
                place([0.0, 84.0, 360.0, 196.0], [640, 480]),
                None
            )
            .is_ok()
        );
        // Its right edge on the crop's: cut, unless the image ends there.
        let right = [0.0, 84.0, 195.0, 196.0];
        assert_eq!(
            verdict(&f2, &c2, &z, place(right, [640, 480]), None).unwrap_err(),
            NoTighten::CropEdge
        );
        assert!(verdict(&f2, &c2, &z, place(right, [195, 480]), None).is_ok());
    }

    #[test]
    fn each_rule_says_why_it_proposes_nothing() {
        let (f, c) = loose_car();
        let why = |zoom: &[ZoomDet], f: &LabelFile| verdict(f, &c, zoom, None, None).unwrap_err();
        assert_eq!(why(&[], &f), NoTighten::NoAgree);
        assert_eq!(
            why(&[zd("car", [150.0, 100.0, 250.0, 180.0], 0.9)], &f),
            NoTighten::NoAgree
        );
        assert_eq!(
            why(&[zd("car", [100.0, 100.0, 250.0, 180.0], 0.9)], &f),
            NoTighten::LabelFine
        );
        let neighbour = reviewed(vec![
            acc("car", LOOSE),
            acc("car", [101.0, 100.0, 203.0, 182.0]),
        ]);
        assert_eq!(
            why(&[zd("car", NEAR, 0.9)], &neighbour),
            NoTighten::Neighbour
        );
        let mut drawn = f.clone();
        drawn.boxes[0].source = Source::Human;
        assert_eq!(why(&[zd("car", NEAR, 0.9)], &drawn), NoTighten::Other);
        let words: Vec<&str> = [
            NoTighten::NoAgree,
            NoTighten::LabelFine,
            NoTighten::LabelFound,
            NoTighten::SpansTwo,
            NoTighten::Small,
            NoTighten::CropEdge,
            NoTighten::Neighbour,
            NoTighten::Other,
        ]
        .map(NoTighten::as_str)
        .to_vec();
        assert_eq!(
            words,
            [
                "no_agree",
                "label_fine",
                "label_found",
                "spans_two",
                "small",
                "crop_edge",
                "neighbour",
                "other"
            ]
        );
    }

    #[test]
    fn a_reviewers_box_a_changed_label_or_a_small_box_gets_no_proposal() {
        let (_, c) = loose_car();
        let z = [zd("car", NEAR, 0.9)];
        for edit in [
            |b: &mut LabelBox| b.source = Source::Human,
            |b: &mut LabelBox| b.orig_bbox = Some([0.0, 0.0, 9.0, 9.0]),
            |b: &mut LabelBox| b.model_label = Some("truck".into()),
            |b: &mut LabelBox| b.bbox = CAR,
            |b: &mut LabelBox| b.label = "dog".into(),
            |b: &mut LabelBox| b.difficult = true,
            |b: &mut LabelBox| b.state = BoxState::Rejected,
        ] {
            let mut f = reviewed(vec![acc("car", LOOSE)]);
            edit(&mut f.boxes[0]);
            assert!(refit(&f, &c, &z, None).is_none(), "{f:?}");
        }
        let mut f = reviewed(vec![acc("car", LOOSE)]);
        f.status = Status::Skipped;
        assert!(refit(&f, &c, &z, None).is_none());
        // The closer look's box under the floor: nothing. Here a thin person label, 30 px wide,
        // whose closer look is 14 px wide (8.4 px at 384, under a 10 px floor).
        let label = [100.0, 100.0, 130.0, 300.0];
        let d = [100.0, 100.0, 116.0, 300.0];
        let thin = [100.0, 100.0, 114.0, 300.0];
        let f = reviewed(vec![acc("person", label)]);
        let j = run(&f, &[det("person", d, 0.9)]);
        let c = j.candidates[0].clone();
        let z = [zd("person", thin, 0.9)];
        let floor = Some(SizeFloor {
            px: 10.0,
            input: 384,
        });
        assert!(refit(&f, &c, &z, None).is_some());
        assert!(refit(&f, &c, &z, floor).is_none());
        // A label that became too small under a floor set since is no candidate either.
        let small = [100.0, 100.0, 113.0, 300.0];
        let f = reviewed(vec![acc("person", small)]);
        let c = Loose { target: small, ..c };
        assert!(
            refit(
                &f,
                &c,
                &[zd("person", [100.0, 100.0, 140.0, 300.0], 0.9)],
                floor
            )
            .is_none()
        );
    }
}
