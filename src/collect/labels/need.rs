//! The recall need of a frame: what the Recommended order ([`super::recommend`]) sorts by, so the
//! frames reviewed next most improve each camera's recall on every class it sees, from diverse
//! examples. Pure; the walk ([`super::queue::build`]) and the review index call the same functions
//! ([`of_file`], [`of_reply`], [`Need::add_reviewed`], [`day`], [`Scorer::score`]) on the same
//! inputs in the same order, so their scores are equal floats.
//!
//! **Candidates** ([`CellBox`], 12 bytes): the boxes a frame may add to a cell (camera x class x
//! scene) once reviewed, of the wanted classes only (`collect.classes` without
//! `collect.hidden_classes`; without a class list no box is a candidate and every frame weighs as
//! an empty one). [`of_file`]: on a reviewed frame its training boxes, else every box that is not
//! rejected and no ignore region, plus a second candidate for a box's `auto_label` (the
//! auto-labelers' class suggestion: a raccoon the camera model called a cat keeps only this),
//! flagged [`MISSED`]. [`of_reply`] (no label file): the reply's predictions, plus the sidecar's
//! below-threshold and suppressed boxes, flagged [`MISSED`]. A box is [`MISSED`] ([`missed`]) when
//! it is pending, has evidence the camera model did not alert on (an auto-labeler found it, or the
//! camera model scored it below its threshold or a rule suppressed it), and no prediction of the
//! same object ([`super::repeats::same_object`]) lies on it at IoU >= [`MISSED_IOU`].
//!
//! **Evidence** ([`evidence`]) of a pending box: [`STRONG`] (1) with two or more labelers, `agree`,
//! `likely` or a VLM yes; else `sqrt(score)` x [`W_LONE`]. A VLM no multiplies either by
//! [`W_VLM_NO`] and makes the box weak (two labelers agreeing on what the VLM calls no is no strong
//! box): it ranks lower, never hides. Here a VLM check counts by its `ok` in either of its formats:
//! the yes/no question, or an older run's category answer (`plant` where `animal` was expected is
//! a no; [`super::VlmCheck::says_no`] leaves those out for the rules that decide boxes, the order
//! only ranks). SigLIP's `species_scores` make no box strong: they sum to 1 over the wanted animals
//! whatever the crop shows (with cat and dog wanted, one of them is always at 0.5 or more), so they
//! name an animal and do not show there is one; on a real folder lone SAM 3 boxes split 0.52 / 0.48
//! between cat and dog were "strong" that way and escaped the camera's precision prior κ, which is
//! what lets the order learn from rejects. A decided box (accepted, added) is 1.
//!
//! **Tally** ([`CellTally`], integers, so the order a walk adds frames in changes nothing): reviewed
//! training boxes in quarter-boxes (4 a box, [`SPOT_QUARTERS`] at a known parked spot of the
//! camera, [`Steer::spots`]) per (class, scene) and per (camera, class, scene), per (camera, class,
//! region of a 3 x 3 grid of the frame), the days per (camera, class, scene), empty reviewed frames
//! per scene, reviewed frames and those with an animal per (camera, scene), and the camera's kept
//! share of the classes models proposed ([`super::queue::Kept`]).
//!
//! **Score** of frame f (camera m, scene s, day d), per wanted class k of its candidates b:
//!
//! ```text
//! a_b = e_b · (W_SPOT at a spot of m) · (W_MISSED if MISSED or GAP)
//!           · (1 + W_REGION · T_R/(T_R + n_{m,k,region(b)}))/2 · (κ_mk unless STRONG)
//! E_k = min(Σ a_b, BOX_CAP)
//! v_k = ½·T/(T + n_{k,s}) + ½·T_CAM/(T_CAM + n_{m,k,s})
//! D_k = 1 + W_DAY if d is a new day of (m, k, s), else 1
//! F_k = RARE_GAIN if class k has fewer than CLASS_FLOOR reviewed boxes, else 1
//! r_k = 1 + RECALL_GAIN · m̂   (m̂: the shrunk held-out miss rate, recall::Measured::miss)
//! U_k = CATCHUP_GAIN if k is camera m's lowest-recall class (within CATCHUP_TOL), else 1
//! κ_mk = (kept + 1)/(proposed + 1) of the camera's proposals of k
//! g_k = 1 (held-out steering comes later)
//! V = Σ_k E_k · v_k · D_k · F_k · r_k · U_k · g_k
//! empty frame: V = W_EMPTY · T/(T + empty_s) · (1 + ANIMAL_GAIN · p_{m,s})
//! score = round(1000 · V)
//! ```
//!
//! `n` are reviewed boxes (quarter-boxes / 4). Classes are summed in class-list order, boxes in the
//! order of their candidates. The recall factor: a class the trained model lacks
//! ([`super::recall::InRun::Absent`]) needs most (`r = 1 + RECALL_GAIN`, and catch-up applies); a
//! class the run dropped on purpose ([`super::recall::InRun::Dropped`]), a scene without a measured
//! run and no measured run at all are `r = U = 1`. Catch-up is water-filling within a camera: its
//! lowest-recall class among those measured on at least [`super::recall::PRIOR_GT`] held-out
//! objects of the camera ([`super::recall::CameraRecall::well_supported`]) is boosted, ties and
//! classes within [`CATCHUP_TOL`] of it too, until it reaches the next one.
//!
//! **Visits** ([`day`]): per event (one camera's activity, [`super::recommend::chain`]) the classes
//! of its decided training boxes ([`DayOut::visit_classes`]): an event holding a box of a class
//! under the floor or of the camera's catch-up class divides by `1 + rank · VISIT_P` rather than `1
//! + rank` ([`Scorer::expands`]: more poses of a rare visit). And the gaps: a class tracked
//! (decided training boxes and pending [`STRONG`] ones) in an earlier and a later frame of the
//! event, each within [`GAP_S`] of the frame, with no box of the class here at IoU >= [`GAP_IOU`]
//! to the box interpolated between them, adds a synthetic [`GAP`] candidate of evidence
//! [`GAP_EVIDENCE`]: something the camera and the labelers lost for a frame.
//!
//! Constants, and why:
//! - [`T`] = 100, [`T_CAM`] = 30: reviewed boxes at which a cell's need halves, over every camera
//!   and on one camera. A class at a hundred boxes in a scene trains well; one camera's view of it
//!   needs about a third of that (its backgrounds and angles), so a camera's own weak class keeps
//!   pulling after the scene is well covered. [`T_R`] = 10: a region of a camera's view is new
//!   until about ten boxes there (persons and animals included, which never form spots).
//! - [`BOX_CAP`] = 3: a crowd counts as three boxes; past that a frame's boxes are near copies.
//! - [`CLASS_FLOOR`] = 20 and [`RARE_GAIN`] = 3: under 20 boxes the benchmark scores no class
//!   ([`crate::benchmark::metrics::MIN_CLASS_GT`]), so the first 20 come first.
//! - [`RECALL_GAIN`] = 2: a class missed half the time weighs twice one never missed.
//! - [`CATCHUP_GAIN`] = 2, [`CATCHUP_TOL`] = 0.05: a camera's lowest class weighs double until it
//!   is within five points of the next; one held-out object of twenty moves recall by five points.
//! - [`W_SPOT`] = 1/4: a parked car at its spot teaches little; it counts a quarter in the tally
//!   ([`SPOT_QUARTERS`]) and in a frame's need.
//! - [`W_MISSED`] = 2: a missed object is the recall the camera lacks today.
//! - [`W_DAY`] = 0.5: a new day of a cell brings other light and weather.
//! - [`W_LONE`] = 0.5, [`W_VLM_NO`] = 0.25: a lone weak box is often false; one the VLM rejects
//!   mostly is.
//! - [`W_EMPTY`] = 0.3, [`ANIMAL_GAIN`] = 3: an empty frame is a negative (worth a third of a
//!   fresh box), more so on a camera whose reviewed frames held animals (`p_{m,s}`, their share):
//!   an "empty" frame there may hold the animal nobody proposed.
//! - [`MISSED_IOU`] = 0.5: one object; [`GAP_IOU`] = 0.3 and [`GAP_S`] = 20 s: an object moves, so
//!   the interpolated box is loose, and over 20 s it may have left.
//! - [`GAP_EVIDENCE`] = 0.5: as a lone weak box of score 1.

use super::bulk::Wanted;
use super::queue::Kept;
use super::recall::{CameraRecall, InRun, Measured, Miss};
use super::repeats::same_object;
use super::{BoxState, LabelBox, LabelFile, Origin, Source, Status, iou, statics};
use crate::collect::record::{Det, Record};
use crate::training::autolabel::Category;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

/// A box at a known parked spot of its camera weighs this much ([`SPOT_QUARTERS`] in the tally).
pub const W_SPOT: f64 = 0.25;
/// Quarter-boxes a reviewed box adds to the tally, and one at a parked spot.
pub const BOX_QUARTERS: u64 = 4;
pub const SPOT_QUARTERS: u64 = 1;
/// A [`MISSED`] or [`GAP`] candidate weighs this much more.
pub const W_MISSED: f64 = 2.0;
/// Weight of a box's region novelty within its camera.
pub const W_REGION: f64 = 1.0;
/// Most boxes one class of a frame counts as.
pub const BOX_CAP: f64 = 3.0;
/// Reviewed boxes at which a class's need in a scene halves, over every camera.
pub const T: f64 = 100.0;
/// The same on one camera.
pub const T_CAM: f64 = 30.0;
/// Reviewed boxes at which a region of a camera's view stops being new (halfway).
pub const T_R: f64 = 10.0;
/// What a new day of a cell adds.
pub const W_DAY: f64 = 0.5;
/// A class under [`CLASS_FLOOR`] reviewed boxes weighs this much more.
pub const RARE_GAIN: f64 = 3.0;
/// Reviewed boxes (over every camera and scene) under which a class is rare.
pub const CLASS_FLOOR: u64 = 20;
/// How much a held-out miss rate raises a class's need (`r = 1 + RECALL_GAIN · m̂`).
pub const RECALL_GAIN: f64 = 2.0;
/// A camera's catch-up class weighs this much more.
pub const CATCHUP_GAIN: f64 = 2.0;
/// Recall within this of a camera's lowest is caught up too.
pub const CATCHUP_TOL: f64 = 0.05;
/// What an empty frame weighs, against a fresh box.
pub const W_EMPTY: f64 = 0.3;
/// How much a camera's share of reviewed frames with an animal raises its empty frames.
pub const ANIMAL_GAIN: f64 = 3.0;
/// A pending box without strong evidence weighs this much of `sqrt(score)`.
pub const W_LONE: f64 = 0.5;
/// And this much more when the VLM said no.
pub const W_VLM_NO: f64 = 0.25;
/// IoU from which a prediction is on the same object as a box (no miss).
pub const MISSED_IOU: f32 = 0.5;
/// Most seconds from a frame to the earlier and the later frame a gap is interpolated between.
pub const GAP_S: i64 = 20;
/// IoU from which a box of the frame covers the interpolated box (no gap).
pub const GAP_IOU: f32 = 0.3;
/// Evidence of a synthetic [`GAP`] candidate.
pub const GAP_EVIDENCE: f64 = 0.5;
/// Scores in one band compare equal in the order ([`super::recommend::key`]).
pub const NEED_BAND: u32 = 100;
/// The visit divisor's step in quarters: `1 + rank` (4) or, expanded, `1 + rank / 4` (1).
pub const VISIT_QUARTERS: u32 = 4;
pub const VISIT_P_QUARTERS: u32 = 1;

/// A decided training box (accepted or added).
pub const TRAINING: u8 = 1;
/// Evidence the camera model did not alert on ([`missed`]), or a class suggestion.
pub const MISSED: u8 = 2;
/// Strong evidence ([`evidence`]), or decided.
pub const STRONG: u8 = 4;
/// Synthetic: a gap in a tracked visit ([`day`]).
pub const GAP: u8 = 8;

/// The scenes of the tally; any other scene is the fourth.
const SCENES: [&str; 4] = ["day", "night", "ir", "other"];
/// The region of a box of a frame of unknown size.
const NO_REGION: u8 = 9;

/// A candidate box of a frame (see the module docs): its class (index in the class list), its box
/// in whole pixels, its evidence in hundredths and its flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord)]
pub struct CellBox {
    pub class: u16,
    pub bbox: [u16; 4],
    pub evidence: u8,
    pub flags: u8,
}

const _: () = assert!(size_of::<CellBox>() == 12);

impl CellBox {
    pub fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    /// Evidence, 0..=1.
    pub fn e(&self) -> f64 {
        f64::from(self.evidence) / 100.0
    }

    fn rect(&self) -> [f32; 4] {
        self.bbox.map(f32::from)
    }

    /// Tracked through a visit ([`day`]): decided, or pending with strong evidence.
    fn tracked(&self) -> bool {
        !self.has(GAP) && (self.has(TRAINING) || self.has(STRONG))
    }
}

/// A pixel coordinate as a whole pixel, 0..=65535 (NaN 0).
fn px(v: f32) -> u16 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the u16 range first"
    )]
    let p = if v.is_finite() {
        v.round().clamp(0.0, f32::from(u16::MAX)) as u16
    } else {
        0
    };
    p
}

fn pixels(b: &[f32; 4]) -> [u16; 4] {
    b.map(px)
}

/// A share 0..=1 in hundredths.
fn hundredths(e: f64) -> u8 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=100 first"
    )]
    let h = (e.clamp(0.0, 1.0) * 100.0).round() as u8;
    h
}

/// The index of `label`'s wanted class in the class list (aliases mapped, hidden classes out).
pub fn class_index(w: &Wanted<'_>, label: &str) -> Option<u16> {
    let class = w.class_of(label)?;
    w.classes
        .iter()
        .position(|c| c == class)
        .and_then(|i| u16::try_from(i).ok())
}

/// The wanted classes of `labels` (proposed labels, [`super::queue::proposed_labels`]), sorted,
/// each once.
pub fn proposed_classes<'a>(
    labels: impl IntoIterator<Item = &'a str>,
    w: Option<&Wanted<'_>>,
) -> Vec<u16> {
    let Some(w) = w else { return Vec::new() };
    let set: BTreeSet<u16> = labels
        .into_iter()
        .filter_map(|l| class_index(w, l))
        .collect();
    set.into_iter().collect()
}

/// The date of a day folder `<scene>/<camera>/<date>` (its last part).
pub fn date_of(day: &str) -> &str {
    day.rsplit_once('/').map_or(day, |(_, d)| d)
}

/// The evidence of a pending box of score `score` with the auto-labelers' findings `auto` (None:
/// no auto-labeler looked), in hundredths, and whether it is [`STRONG`].
pub fn evidence(score: Option<f32>, auto: Option<&super::AutoInfo>) -> (u8, bool) {
    let strong = auto.is_some_and(|a| {
        a.labelers.len() >= 2
            || a.agree == Some(true)
            || a.likely
            || a.vlm.as_ref().is_some_and(|v| v.ok)
    });
    let e = if strong {
        1.0
    } else {
        f64::from(score.unwrap_or(1.0).clamp(0.0, 1.0)).sqrt() * W_LONE
    };
    if auto.is_some_and(|a| a.vlm.as_ref().is_some_and(|v| !v.ok)) {
        return (hundredths(e * W_VLM_NO), false);
    }
    (hundredths(e), strong)
}

/// A prediction of the same object lies on `bbox` at IoU >= [`MISSED_IOU`]: the camera model
/// alerted on it.
fn alerted<'a>(
    predictions: impl IntoIterator<Item = (&'a str, &'a [f32; 4])>,
    label: &str,
    bbox: &[f32; 4],
) -> bool {
    predictions
        .into_iter()
        .any(|(l, b)| same_object(l, label) && iou(b, bbox) >= MISSED_IOU)
}

/// A box of `l` the camera model missed (see the module docs).
pub fn missed(b: &LabelBox, l: &LabelFile) -> bool {
    if b.state != BoxState::Pending {
        return false;
    }
    let auto = b.source == Source::Auto || !b.auto.labelers.is_empty();
    let skipped = b.source == Source::Model
        && matches!(b.origin, Some(Origin::BelowThreshold | Origin::Suppressed));
    (auto || skipped)
        && !alerted(
            l.boxes
                .iter()
                .filter(|o| o.source == Source::Model && o.origin == Some(Origin::Prediction))
                .map(|o| (o.label.as_str(), &o.bbox)),
            &b.label,
            &b.bbox,
        )
}

/// The candidates of a frame with label file `l` (see the module docs), in the file's order.
pub fn of_file(l: &LabelFile, w: Option<&Wanted<'_>>) -> Vec<CellBox> {
    let Some(w) = w else { return Vec::new() };
    let reviewed = l.status == Status::Reviewed;
    let mut out = Vec::new();
    for b in &l.boxes {
        if b.is_ignore() || b.state == BoxState::Rejected || (reviewed && !b.is_training()) {
            continue;
        }
        let suggested = (b.state == BoxState::Pending)
            .then_some(b.auto.auto_label.as_deref())
            .flatten()
            .and_then(|a| class_index(w, a));
        let own = class_index(w, &b.label);
        let suggested = suggested.filter(|k| Some(*k) != own);
        if let Some(k) = own {
            let (evidence, flags) = if b.is_training() {
                (100, TRAINING | STRONG)
            } else {
                // A class suggestion's findings are about the suggested class, not this one.
                let auto = (suggested.is_none() && !b.auto.is_empty()).then_some(&b.auto);
                let (e, strong) = evidence(b.score, auto);
                let flags = if strong { STRONG } else { 0 } | if missed(b, l) { MISSED } else { 0 };
                (e, flags)
            };
            out.push(CellBox {
                class: k,
                bbox: pixels(&b.bbox),
                evidence,
                flags,
            });
        }
        if let Some(k) = suggested {
            let (e, strong) = evidence(b.auto.auto_score.or(b.score), Some(&b.auto));
            out.push(CellBox {
                class: k,
                bbox: pixels(&b.bbox),
                evidence: e,
                flags: MISSED | if strong { STRONG } else { 0 },
            });
        }
    }
    out
}

/// The candidates of a frame without a label file, from its sidecar `rec`: the predictions, then
/// the below-threshold and suppressed boxes ([`MISSED`] unless a prediction is on them).
pub fn of_reply(rec: &Record, w: Option<&Wanted<'_>>) -> Vec<CellBox> {
    let Some(w) = w else { return Vec::new() };
    let mut out = Vec::new();
    let predictions = || rec.predictions.iter().map(|d| (d.label.as_str(), &d.bbox));
    let mut push = |label: &str, score: f32, bbox: &[f32; 4], skipped: bool| {
        let Some(k) = class_index(w, label) else {
            return;
        };
        let (e, _) = evidence(Some(score), None);
        let missed = skipped && !alerted(predictions(), label, bbox);
        out.push(CellBox {
            class: k,
            bbox: pixels(bbox),
            evidence: e,
            flags: if missed { MISSED } else { 0 },
        });
    };
    for d in &rec.predictions {
        push(&d.label, d.score, &d.bbox, false);
    }
    for d in &rec.below_threshold {
        let Det { label, score, bbox } = d;
        push(label, *score, bbox, true);
    }
    for s in &rec.suppressed {
        push(&s.label, s.score, &s.bbox, true);
    }
    out
}

/// What steers the order from outside the folder (the server's `review::steer`), one snapshot per
/// query, handed to the walk and the index alike ([`super::queue::Filters::steer`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Steer {
    /// The measured recall of the newest trained models; None = nothing measured.
    pub recall: Option<Arc<Measured>>,
    /// The folder's parked spots (empty until the first build: neutral).
    pub spots: Arc<statics::Table>,
    /// Camera days listed in `heldout-days.json` (`<camera>/<date>`), for held-out steering
    /// (not read yet: `g_k` is 1).
    pub heldout_listed: Arc<BTreeSet<String>>,
}

/// The index of a scene in the tally.
fn scene_ix(scene: &str) -> u8 {
    match scene {
        "day" => 0,
        "night" => 1,
        "ir" => 2,
        _ => 3,
    }
}

/// The region of `b` in a `w` x `h` frame: its centre on a 3 x 3 grid, row by row;
/// [`NO_REGION`] for a frame of unknown size.
fn region(b: &CellBox, w: u32, h: u32) -> u8 {
    if w == 0 || h == 0 {
        return NO_REGION;
    }
    let cell = |a: u16, z: u16, size: u32| -> u8 {
        let c = (f64::from(a) + f64::from(z)) / 2.0 / f64::from(size);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=2 first"
        )]
        let i = (c * 3.0).floor().clamp(0.0, 2.0) as u8;
        i
    };
    cell(b.bbox[1], b.bbox[3], h) * 3 + cell(b.bbox[0], b.bbox[2], w)
}

/// One camera's part of the tally.
#[derive(Debug, Clone, Default, PartialEq)]
struct CameraCells {
    /// (class, scene) -> quarter-boxes.
    cells: BTreeMap<(u16, u8), u64>,
    /// (class, region) -> quarter-boxes.
    regions: BTreeMap<(u16, u8), u64>,
    /// (class, scene) -> the days with a box.
    days: BTreeMap<(u16, u8), BTreeSet<String>>,
    /// Reviewed frames per scene, and those with an animal's training box.
    frames: [u64; 4],
    animal_frames: [u64; 4],
    /// Per class: reviewed frames a model or auto-labeler proposed it on, and those that kept it.
    precision: BTreeMap<u16, Kept>,
}

/// Reviewed boxes per cell (see the module docs). Integers and sets only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellTally {
    /// (class, scene) -> quarter-boxes over every camera.
    class_scene: BTreeMap<(u16, u8), u64>,
    /// Reviewed frames with no wanted training box, per scene.
    empty: [u64; 4],
    cameras: BTreeMap<String, CameraCells>,
}

impl CellTally {
    /// Add another part of a walk.
    pub fn merge(&mut self, other: Self) {
        let Self {
            class_scene,
            empty,
            cameras,
        } = other;
        for (k, n) in class_scene {
            *self.class_scene.entry(k).or_default() += n;
        }
        for (a, b) in self.empty.iter_mut().zip(empty) {
            *a += b;
        }
        for (camera, c) in cameras {
            let mine = self.cameras.entry(camera).or_default();
            for (k, n) in c.cells {
                *mine.cells.entry(k).or_default() += n;
            }
            for (k, n) in c.regions {
                *mine.regions.entry(k).or_default() += n;
            }
            for (k, days) in c.days {
                mine.days.entry(k).or_default().extend(days);
            }
            for (a, b) in mine.frames.iter_mut().zip(c.frames) {
                *a += b;
            }
            for (a, b) in mine.animal_frames.iter_mut().zip(c.animal_frames) {
                *a += b;
            }
            for (k, p) in c.precision {
                let m = mine.precision.entry(k).or_default();
                m.kept += p.kept;
                m.proposed += p.proposed;
            }
        }
    }

    /// Quarter-boxes of class `k` over every scene.
    fn class_quarters(&self, k: u16) -> u64 {
        (0..4u8)
            .map(|s| self.class_scene.get(&(k, s)).copied().unwrap_or(0))
            .sum()
    }

    /// What the answer shows (`need`): reviewed boxes per class and scene, empty frames per scene.
    pub fn counts(&self, classes: &[String]) -> NeedCounts {
        let mut boxes: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for ((k, s), q) in &self.class_scene {
            let (Some(class), Some(scene)) =
                (classes.get(usize::from(*k)), SCENES.get(usize::from(*s)))
            else {
                continue;
            };
            boxes
                .entry(class.clone())
                .or_default()
                .insert((*scene).to_string(), *q as f64 / BOX_QUARTERS as f64);
        }
        NeedCounts {
            boxes,
            empty: SCENES
                .iter()
                .zip(self.empty)
                .filter(|(_, n)| *n > 0)
                .map(|(s, n)| ((*s).to_string(), n))
                .collect(),
            floor: CLASS_FLOOR,
            target: T,
            camera_target: T_CAM,
        }
    }
}

/// The tally as the answer serializes it ([`CellTally::counts`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct NeedCounts {
    /// Class -> scene -> reviewed boxes (a box at a parked spot counts a quarter).
    pub boxes: BTreeMap<String, BTreeMap<String, f64>>,
    /// Scene -> reviewed frames with no wanted training box.
    pub empty: BTreeMap<String, u64>,
    pub floor: u64,
    pub target: f64,
    pub camera_target: f64,
}

/// A reviewed frame, for [`Need::add_reviewed`].
#[derive(Debug, Clone, Copy)]
pub struct Reviewed<'a> {
    pub camera: &'a str,
    pub scene: &'a str,
    /// Its date folder.
    pub day: &'a str,
    pub width: u32,
    pub height: u32,
    /// Its candidates ([`of_file`] of a reviewed frame: its training boxes).
    pub boxes: &'a [CellBox],
    /// The wanted classes models and auto-labelers proposed on it and a reviewer judged
    /// ([`super::queue::proposed_labels`]), sorted, each once.
    pub proposed: &'a [u16],
}

/// A frame to weigh, for [`Scorer::score`].
#[derive(Debug, Clone, Copy)]
pub struct Facts<'a> {
    pub camera: &'a str,
    pub scene: &'a str,
    pub day: &'a str,
    pub width: u32,
    pub height: u32,
}

/// What weighs frames of one query: the class list and the steer.
#[derive(Debug)]
pub struct Need<'a> {
    wanted: Option<Wanted<'a>>,
    steer: Option<&'a Steer>,
    /// Per class: an animal ([`Category::Animal`]), and may form a parked spot.
    animal: Vec<bool>,
    may_park: Vec<bool>,
}

impl<'a> Need<'a> {
    pub fn new(wanted: Option<Wanted<'a>>, steer: Option<&'a Steer>) -> Self {
        let classes = wanted.map_or(&[][..], |w| w.classes);
        Self {
            animal: classes
                .iter()
                .map(|c| Category::of(c) == Category::Animal)
                .collect(),
            may_park: classes.iter().map(|c| statics::may_be_static(c)).collect(),
            wanted,
            steer,
        }
    }

    pub fn wanted(&self) -> Option<&Wanted<'a>> {
        self.wanted.as_ref()
    }

    /// The class list's name of class `k`.
    pub fn name(&self, k: u16) -> &'a str {
        self.wanted
            .and_then(|w| w.classes.get(usize::from(k)))
            .map_or("", String::as_str)
    }

    /// `b`, a box of a `w` x `h` frame of `camera`, lies at one of its parked spots.
    fn at_spot(&self, camera: &str, b: &CellBox, w: u32, h: u32) -> bool {
        let Some(steer) = self.steer else {
            return false;
        };
        if !self
            .may_park
            .get(usize::from(b.class))
            .copied()
            .unwrap_or(false)
            || steer.spots.camera(camera).is_empty()
        {
            return false;
        }
        statics::normalised(&b.rect(), w, h).is_some_and(|nb| {
            steer
                .spots
                .spot_at(camera, self.name(b.class), &nb)
                .is_some()
        })
    }

    /// Add reviewed frame `r` to `t`: the one way both walks count.
    pub fn add_reviewed(&self, t: &mut CellTally, r: &Reviewed<'_>) {
        let s = scene_ix(r.scene);
        if !t.cameras.contains_key(r.camera) {
            t.cameras
                .insert(r.camera.to_string(), CameraCells::default());
        }
        let Some(cam) = t.cameras.get_mut(r.camera) else {
            return;
        };
        let si = usize::from(s);
        if let Some(n) = cam.frames.get_mut(si) {
            *n += 1;
        }
        let mut trained: Vec<u16> = Vec::new();
        let mut animal = false;
        for b in r.boxes.iter().filter(|b| b.has(TRAINING)) {
            let q = if self.at_spot(r.camera, b, r.width, r.height) {
                SPOT_QUARTERS
            } else {
                BOX_QUARTERS
            };
            *t.class_scene.entry((b.class, s)).or_default() += q;
            *cam.cells.entry((b.class, s)).or_default() += q;
            *cam.regions
                .entry((b.class, region(b, r.width, r.height)))
                .or_default() += q;
            let days = cam.days.entry((b.class, s)).or_default();
            if !days.contains(r.day) {
                days.insert(r.day.to_string());
            }
            trained.push(b.class);
            animal |= self
                .animal
                .get(usize::from(b.class))
                .copied()
                .unwrap_or(false);
        }
        if trained.is_empty()
            && let Some(n) = t.empty.get_mut(si)
        {
            *n += 1;
        }
        if animal && let Some(n) = cam.animal_frames.get_mut(si) {
            *n += 1;
        }
        for k in r.proposed {
            let p = cam.precision.entry(*k).or_default();
            p.proposed += 1;
            if trained.contains(k) {
                p.kept += 1;
            }
        }
    }
}

/// A counted frame of one day folder, for [`day`].
#[derive(Debug, Clone, Copy)]
pub struct DayFrame<'a> {
    /// [`super::recommend::event_ms`] of its time.
    pub ms: Option<i64>,
    /// An event of its own ([`super::recommend::own_event`]).
    pub alone: bool,
    pub boxes: &'a [CellBox],
    /// The queue lists it: [`day`] works out its gaps and visit.
    pub listed: bool,
}

/// What [`day`] finds for a listed frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DayOut {
    /// Synthetic [`GAP`] candidates, by class then box.
    pub gaps: Vec<CellBox>,
    /// The classes of the decided training boxes of its event, sorted, each once.
    pub visit_classes: Vec<u16>,
}

const GAP_MS: i64 = GAP_S * 1000;

/// The gaps and visits of one day folder's counted frames (see the module docs), per frame in
/// their order (empty for those not listed). The order they come in changes nothing.
pub fn day(frames: &[DayFrame<'_>]) -> Vec<DayOut> {
    let times: Vec<Option<i64>> = frames.iter().map(|f| f.ms.filter(|_| !f.alone)).collect();
    let events = super::recommend::chain(&times);
    let count = events.iter().copied().max().map_or(0, |m| m + 1);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (i, e) in events.iter().enumerate() {
        if let Some(m) = members.get_mut(*e) {
            m.push(i);
        }
    }
    let classes: Vec<Vec<u16>> = members
        .iter()
        .map(|m| {
            let set: BTreeSet<u16> = m
                .iter()
                .filter_map(|i| frames.get(*i))
                .flat_map(|f| f.boxes.iter().filter(|b| b.has(TRAINING)).map(|b| b.class))
                .collect();
            set.into_iter().collect()
        })
        .collect();
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if !f.listed {
                return DayOut::default();
            }
            let e = events.get(i).copied().unwrap_or(0);
            let mates = members.get(e).map_or(&[][..], Vec::as_slice);
            DayOut {
                gaps: gaps_at(i, frames, &times, mates),
                visit_classes: classes.get(e).cloned().unwrap_or_default(),
            }
        })
        .collect()
}

/// The tracked boxes of class `k` of the frames of `mates` at time `t`, sorted.
fn tracked_at(
    frames: &[DayFrame<'_>],
    times: &[Option<i64>],
    mates: &[usize],
    t: i64,
    k: u16,
) -> Vec<CellBox> {
    let mut out: Vec<CellBox> = mates
        .iter()
        .filter(|j| times.get(**j).copied().flatten() == Some(t))
        .filter_map(|j| frames.get(*j))
        .flat_map(|f| f.boxes.iter().copied())
        .filter(|b| b.class == k && b.tracked())
        .collect();
    out.sort_unstable();
    out
}

/// The [`GAP`] candidates of frame `i` among its event's frames `mates`.
fn gaps_at(
    i: usize,
    frames: &[DayFrame<'_>],
    times: &[Option<i64>],
    mates: &[usize],
) -> Vec<CellBox> {
    let (Some(t), Some(here)) = (times.get(i).copied().flatten(), frames.get(i)) else {
        return Vec::new();
    };
    let others: Vec<(i64, &DayFrame<'_>)> = mates
        .iter()
        .filter(|j| **j != i)
        .filter_map(|j| Some((times.get(*j).copied().flatten()?, frames.get(*j)?)))
        .collect();
    let classes: BTreeSet<u16> = others
        .iter()
        .flat_map(|(_, f)| f.boxes.iter().filter(|b| b.tracked()).map(|b| b.class))
        .collect();
    let mut out: Vec<CellBox> = Vec::new();
    for k in classes {
        let has = |f: &DayFrame<'_>| f.boxes.iter().any(|b| b.class == k && b.tracked());
        let before = others
            .iter()
            .filter(|(ms, f)| *ms < t && t - *ms <= GAP_MS && has(f))
            .map(|(ms, _)| *ms)
            .max();
        let after = others
            .iter()
            .filter(|(ms, f)| *ms > t && *ms - t <= GAP_MS && has(f))
            .map(|(ms, _)| *ms)
            .min();
        let (Some(t0), Some(t1)) = (before, after) else {
            continue;
        };
        let (a, b) = (
            tracked_at(frames, times, mates, t0, k),
            tracked_at(frames, times, mates, t1, k),
        );
        let frac = (t - t0) as f64 / (t1 - t0) as f64;
        for x in &a {
            // The later box most like this one; the first of a tie (they are sorted).
            let Some(y) = b.iter().fold(None::<(f32, &CellBox)>, |best, y| {
                let v = iou(&x.rect(), &y.rect());
                match best {
                    Some((bv, _)) if bv >= v => best,
                    _ => Some((v, y)),
                }
            }) else {
                continue;
            };
            let y = y.1;
            let mut bbox = [0u16; 4];
            for (c, (p, q)) in bbox.iter_mut().zip(x.bbox.iter().zip(y.bbox)) {
                let v = f64::from(*p) + (f64::from(q) - f64::from(*p)) * frac;
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "between two u16 values, so within f32 and u16"
                )]
                let v = v as f32;
                *c = px(v);
            }
            let gap = CellBox {
                class: k,
                bbox,
                evidence: hundredths(GAP_EVIDENCE),
                flags: GAP,
            };
            let covered = |o: &CellBox| o.class == k && iou(&o.rect(), &gap.rect()) >= GAP_IOU;
            if here.boxes.iter().filter(|o| !o.has(GAP)).any(covered) || out.iter().any(covered) {
                continue;
            }
            out.push(gap);
        }
    }
    out.sort_unstable();
    out
}

/// The recall steering of one camera ([`Scorer`]).
#[derive(Debug, Default)]
struct CameraSteer {
    /// Per scene (day, night, ir), per class: the shrunk miss rate.
    miss: [Vec<Option<Miss>>; 3],
    /// Per class: catch-up applies.
    catch_up: Vec<bool>,
}

/// The catch-up classes of a camera over `recall` (per class; see the module docs): a class the
/// model lacks, and the lowest-recall well-supported classes within [`CATCHUP_TOL`] of the lowest.
fn catch_up_of(recall: &[Option<CameraRecall>]) -> Vec<bool> {
    let low = recall
        .iter()
        .flatten()
        .filter(|r| r.well_supported())
        .filter_map(|r| r.recall)
        .fold(None::<f64>, |m, r| Some(m.map_or(r, |m| m.min(r))));
    recall
        .iter()
        .map(|r| match r {
            Some(r) if r.class == InRun::Absent => true,
            Some(r) if r.well_supported() => {
                matches!((r.recall, low), (Some(x), Some(l)) if x <= l + CATCHUP_TOL)
            }
            _ => false,
        })
        .collect()
}

/// The recall factor of a class from its miss rate (see the module docs).
fn recall_factor(m: Option<&Miss>) -> f64 {
    match m {
        Some(m) => match m.class {
            InRun::Class { .. } => 1.0 + RECALL_GAIN * m.rate.unwrap_or(0.0),
            InRun::Absent => 1.0 + RECALL_GAIN,
            InRun::Dropped => 1.0,
        },
        None => 1.0,
    }
}

/// What [`Scorer::score`] found out about a frame, numbers only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detail {
    pub score: u32,
    /// The class with the largest term (None: an empty frame).
    pub top: Option<u16>,
    /// The top class's E_k, n_{m,k,s}, n_{k,s} and reviewed boxes over every scene.
    pub boxes: f64,
    pub camera_reviewed: f64,
    pub reviewed: f64,
    pub class_reviewed: f64,
    pub below_floor: bool,
    pub catch_up: bool,
    pub new_day: bool,
    pub new_region: bool,
    pub spot: bool,
    pub missed: bool,
    pub gap: bool,
    pub strong: bool,
    /// The best evidence of the top class's boxes.
    pub evidence: f64,
    /// The camera's proposals of the top class when some were not kept.
    pub kept: Option<Kept>,
    /// An empty frame: the scene's reviewed empty frames and the camera's animal share there.
    pub empty: Option<(u64, f64)>,
}

/// Weighs the frames of one query against a finished tally.
#[derive(Debug)]
pub struct Scorer<'a> {
    need: &'a Need<'a>,
    tally: &'a CellTally,
    /// Per camera, made when first asked.
    cameras: RefCell<HashMap<String, Arc<CameraSteer>>>,
}

impl<'a> Scorer<'a> {
    pub fn new(need: &'a Need<'a>, tally: &'a CellTally) -> Self {
        Self {
            need,
            tally,
            cameras: RefCell::new(HashMap::new()),
        }
    }

    fn measured(&self) -> Option<&Measured> {
        self.need.steer.and_then(|s| s.recall.as_deref())
    }

    /// The recall steering of `camera`.
    fn camera(&self, camera: &str) -> Arc<CameraSteer> {
        if let Some(c) = self.cameras.borrow().get(camera) {
            return c.clone();
        }
        let classes = self.need.wanted.map_or(&[][..], |w| w.classes);
        let c = Arc::new(match self.measured() {
            None => CameraSteer::default(),
            Some(m) => {
                let per_scene = |s: &str| classes.iter().map(|k| m.miss(camera, s, k)).collect();
                let recall: Vec<Option<CameraRecall>> = classes
                    .iter()
                    .map(|k| m.class_recall_on_camera(camera, k))
                    .collect();
                CameraSteer {
                    miss: [per_scene("day"), per_scene("night"), per_scene("ir")],
                    catch_up: catch_up_of(&recall),
                }
            }
        });
        self.cameras
            .borrow_mut()
            .insert(camera.to_string(), c.clone());
        c
    }

    /// The shrunk miss rate of class `k` on `camera` in `scene` (None: not measured).
    pub fn miss(&self, camera: &str, scene: &str, k: u16) -> Option<Miss> {
        let c = self.camera(camera);
        c.miss
            .get(usize::from(scene_ix(scene)))
            .and_then(|m| m.get(usize::from(k)))
            .cloned()
            .flatten()
    }

    fn catch_up(&self, cam: &CameraSteer, k: u16) -> bool {
        cam.catch_up.get(usize::from(k)).copied().unwrap_or(false)
    }

    fn below_floor(&self, k: u16) -> bool {
        self.tally.class_quarters(k) < CLASS_FLOOR * BOX_QUARTERS
    }

    /// The visit of a frame of `camera` whose event holds decided training boxes of
    /// `visit_classes` expands ([`VISIT_P_QUARTERS`]): one of them is under the floor or the
    /// camera's catch-up class.
    pub fn expands(&self, camera: &str, visit_classes: &[u16]) -> bool {
        if visit_classes.is_empty() {
            return false;
        }
        let cam = self.camera(camera);
        visit_classes
            .iter()
            .any(|k| self.below_floor(*k) || self.catch_up(&cam, *k))
    }

    /// The need of frame `f` with candidates `boxes` and gaps `gaps` (see the module docs).
    pub fn score(&self, f: &Facts<'_>, boxes: &[CellBox], gaps: &[CellBox]) -> Detail {
        let s = scene_ix(f.scene);
        let cam_cells = self.tally.cameras.get(f.camera);
        let all = || boxes.iter().chain(gaps);
        let mut classes: Vec<u16> = all().map(|b| b.class).collect();
        classes.sort_unstable();
        classes.dedup();
        if classes.is_empty() {
            let empty = self.tally.empty.get(usize::from(s)).copied().unwrap_or(0);
            let share = cam_cells.map_or(0.0, |c| {
                let n = c.frames.get(usize::from(s)).copied().unwrap_or(0);
                let a = c.animal_frames.get(usize::from(s)).copied().unwrap_or(0);
                if n == 0 { 0.0 } else { a as f64 / n as f64 }
            });
            let v = W_EMPTY * T / (T + empty as f64) * (1.0 + ANIMAL_GAIN * share);
            return Detail {
                score: score_of(v),
                empty: Some((empty, share)),
                ..Detail::default()
            };
        }
        let cam = self.camera(f.camera);
        let quarters = |m: Option<&BTreeMap<(u16, u8), u64>>, key: (u16, u8)| {
            m.and_then(|m| m.get(&key)).copied().unwrap_or(0) as f64 / BOX_QUARTERS as f64
        };
        let mut v = 0.0;
        let mut best: Option<(f64, Detail)> = None;
        for k in classes {
            let kept = cam_cells
                .and_then(|c| c.precision.get(&k))
                .copied()
                .unwrap_or_default();
            let kappa = (kept.kept as f64 + 1.0) / (kept.proposed as f64 + 1.0);
            let mut sum = 0.0;
            let mut d = Detail::default();
            for b in all().filter(|b| b.class == k) {
                let spot = self.need.at_spot(f.camera, b, f.width, f.height);
                let n_region = quarters(
                    cam_cells.map(|c| &c.regions),
                    (k, region(b, f.width, f.height)),
                );
                let strong = b.has(STRONG);
                let missed = b.has(MISSED) || b.has(GAP);
                let mut a = b.e();
                if spot {
                    a *= W_SPOT;
                }
                if missed {
                    a *= W_MISSED;
                }
                a *= (1.0 + W_REGION * T_R / (T_R + n_region)) / 2.0;
                if !strong {
                    a *= kappa;
                }
                sum += a;
                d.spot |= spot;
                d.missed |= b.has(MISSED);
                d.gap |= b.has(GAP);
                d.strong |= strong;
                d.new_region |= n_region == 0.0;
                d.evidence = d.evidence.max(b.e());
            }
            let e_k = sum.min(BOX_CAP);
            let n_ks = quarters(Some(&self.tally.class_scene), (k, s));
            let n_mks = quarters(cam_cells.map(|c| &c.cells), (k, s));
            let v_k = 0.5 * T / (T + n_ks) + 0.5 * T_CAM / (T_CAM + n_mks);
            let new_day = !cam_cells
                .and_then(|c| c.days.get(&(k, s)))
                .is_some_and(|days| days.contains(f.day));
            let d_k = if new_day { 1.0 + W_DAY } else { 1.0 };
            let floor = self.below_floor(k);
            let f_k = if floor { RARE_GAIN } else { 1.0 };
            let miss = cam
                .miss
                .get(usize::from(s))
                .and_then(|m| m.get(usize::from(k)));
            let r_k = recall_factor(miss.and_then(Option::as_ref));
            let catch_up = self.catch_up(&cam, k);
            let u_k = if catch_up { CATCHUP_GAIN } else { 1.0 };
            let g_k = self.heldout_gain(f, k);
            let term = e_k * v_k * d_k * f_k * r_k * u_k * g_k;
            v += term;
            if best.as_ref().is_none_or(|(t, _)| term > *t) {
                d.boxes = e_k;
                d.camera_reviewed = n_mks;
                d.reviewed = n_ks;
                d.class_reviewed = self.tally.class_quarters(k) as f64 / BOX_QUARTERS as f64;
                d.below_floor = floor;
                d.catch_up = catch_up;
                d.new_day = new_day;
                d.kept = (kept.kept < kept.proposed).then_some(kept);
                d.top = Some(k);
                best = Some((term, d));
            }
        }
        let mut d = best.map(|(_, d)| d).unwrap_or_default();
        d.score = score_of(v);
        d
    }

    /// `g_k`: held-out steering, not applied yet (1).
    fn heldout_gain(&self, _f: &Facts<'_>, _k: u16) -> f64 {
        1.0
    }

    /// The page's explanation of a frame of `camera` in `scene` ([`Self::score`]'s `d`), its
    /// visit expanded or not. Shares are rounded to [`four`] decimals: the page shows no more, and a
    /// short decimal reads back as the same float wherever the JSON is parsed.
    pub fn explain(&self, d: &Detail, camera: &str, scene: &str, expanded: bool) -> RecallNeed {
        let miss = d.top.and_then(|k| self.miss(camera, scene, k));
        let lacks = miss.as_ref().is_some_and(|m| m.class == InRun::Absent);
        RecallNeed {
            score: d.score,
            visit_score: d.score,
            class: d.top.map(|k| self.need.name(k).to_string()),
            boxes: four(d.boxes),
            camera_reviewed: d.camera_reviewed,
            camera_target: T_CAM,
            reviewed: d.reviewed,
            target: T,
            class_reviewed: d.class_reviewed,
            floor: CLASS_FLOOR,
            recall: miss.as_ref().and_then(|m| m.rate).map(|r| four(1.0 - r)),
            heldout_gt: miss
                .as_ref()
                .filter(|m| m.rate.is_some())
                .map(|m| m.cell.gt),
            run: miss.as_ref().map(|m| m.run.clone()),
            model_lacks: lacks,
            below_floor: d.below_floor,
            catch_up: d.catch_up,
            missed: d.missed,
            gap: d.gap,
            spot: d.spot,
            new_region: d.new_region && d.top.is_some(),
            new_day: d.new_day,
            strong: d.strong,
            evidence: four(d.evidence),
            kept: d.kept.map(|k| k.kept),
            proposed: d.kept.map(|k| k.proposed),
            expanded,
            empty_reviewed: d.empty.map(|e| e.0),
            animal_share: d.empty.map(|e| four(e.1)).filter(|s| *s > 0.0),
        }
    }
}

/// `x` rounded to four decimals.
fn four(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// `round(1000 · v)` as a score.
fn score_of(v: f64) -> u32 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the u32 range first"
    )]
    let s = (1000.0 * v).round().clamp(0.0, f64::from(u32::MAX)) as u32;
    s
}

/// The visit divisor: `score / (1 + rank · P)`, P a quarter of [`VISIT_QUARTERS`] when `expanded`.
/// Integer math, rounded down once.
pub fn visit_score(score: u32, rank: u32, expanded: bool) -> u32 {
    let p = if expanded {
        VISIT_P_QUARTERS
    } else {
        VISIT_QUARTERS
    };
    let den = u64::from(VISIT_QUARTERS) + u64::from(rank) * u64::from(p);
    u32::try_from(u64::from(score) * u64::from(VISIT_QUARTERS) / den).unwrap_or(u32::MAX)
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Why the Recommended order places a frame where it does (item field `need`): its score, the
/// score after the visit divisor and, for its top class, what the score came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RecallNeed {
    /// `round(1000 · V)`.
    pub score: u32,
    /// [`visit_score`] at the frame's event rank: what the order bands ([`NEED_BAND`]).
    pub visit_score: u32,
    /// The class with the largest term; absent: an empty frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Its boxes on the frame, weighed (E_k).
    pub boxes: f64,
    /// Reviewed boxes of the class in the frame's scene, on its camera and on every camera, and
    /// the targets at which the need halves.
    pub camera_reviewed: f64,
    pub camera_target: f64,
    pub reviewed: f64,
    pub target: f64,
    /// Reviewed boxes of the class, every camera and scene, and the floor.
    pub class_reviewed: f64,
    pub floor: u64,
    /// The held-out recall of the class on the camera in the scene (shrunk), on `heldout_gt`
    /// objects of the camera there, of run `run`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heldout_gt: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    /// The trained model has no such class.
    #[serde(skip_serializing_if = "is_false")]
    pub model_lacks: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub below_floor: bool,
    /// The camera's lowest-recall class.
    #[serde(skip_serializing_if = "is_false")]
    pub catch_up: bool,
    /// The camera model missed a box of it.
    #[serde(skip_serializing_if = "is_false")]
    pub missed: bool,
    /// A gap in its visit.
    #[serde(skip_serializing_if = "is_false")]
    pub gap: bool,
    /// A box at a parked spot.
    #[serde(skip_serializing_if = "is_false")]
    pub spot: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub new_region: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub new_day: bool,
    /// Strong evidence on a box of it.
    #[serde(skip_serializing_if = "is_false")]
    pub strong: bool,
    /// The best evidence of its boxes, 0..=1.
    pub evidence: f64,
    /// The camera's reviewed proposals of the class, when some were not kept.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kept: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed: Option<u64>,
    /// The visit expands: a rare or catch-up class was decided in it.
    #[serde(skip_serializing_if = "is_false")]
    pub expanded: bool,
    /// An empty frame: the scene's reviewed empty frames, and the camera's share of reviewed
    /// frames with an animal there (when any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty_reviewed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub animal_share: Option<f64>,
}

#[cfg(test)]
#[path = "need/tests.rs"]
mod tests;
