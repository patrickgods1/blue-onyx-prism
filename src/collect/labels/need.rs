//! The recall need of a frame: what the Recommended order ([`super::recommend`]) sorts by, so the
//! frames reviewed next most improve each camera's recall on every class it sees, from diverse
//! examples, and now and then teach it a likely false box. Pure; the walk
//! ([`super::queue::build`]) and the review index call the same functions ([`of_file`],
//! [`of_reply`], [`Need::add_reviewed`], [`day`], [`Scorer::score`]) on the same inputs in the
//! same order, so their scores are equal floats.
//!
//! **Candidates** ([`CellBox`], 12 bytes): the boxes a frame may add to a cell (camera x class x
//! scene) once reviewed, of the wanted classes only (`collect.classes` without
//! `collect.hidden_classes`; without a class list no box is a candidate and every frame weighs as
//! an empty one). [`of_file`] of an unreviewed or skipped frame: every box that is not rejected and
//! no ignore region, plus a second candidate for a box's `auto_label` (the auto-labelers' class
//! suggestion: a raccoon the camera model called a cat keeps only this), flagged [`MISSED`].
//! [`of_reply`] (no label file): the reply's predictions, plus the sidecar's below-threshold and
//! suppressed boxes. A box is [`MISSED`] ([`missed`]) when it is pending, has evidence the camera
//! model did not alert on (an auto-labeler found it, or the camera model scored it below its
//! threshold or a rule suppressed it), and no prediction of the same object
//! ([`super::repeats::same_object`]) lies on it at IoU >= [`MISSED_IOU`]. At most
//! [`MAX_PER_CLASS`] pending candidates of a class are kept (the strongest; ties by order): a storm
//! of weak boxes is no crowd, and the index holds less. Candidates come sorted by class, in their
//! order within a class. A pending candidate of evidence under [`MIN_EVIDENCE`] ([`faint`]) adds
//! no recall need: [`Scorer::score`] leaves it out unless it is likely false (a VLM no, or κ under
//! [`LIKELY_FALSE_KAPPA`]), and then it counts for its hard-negative share alone, so a frame
//! holding nothing else scores its `H` ([`Detail::hard_only`]), not as an empty frame.
//!
//! A reviewed frame's [`of_file`] is what the tally reads instead: its training boxes
//! ([`TRAINING`]), its hard negatives ([`REJECTED`]: rejected machine boxes a reviewer judged, no
//! duplicates, ignore regions or dismissals, [`LabelFile::hard_negatives`]) and one marker per
//! class a model or auto-labeler proposed there and a reviewer judged ([`PROPOSED`],
//! [`super::queue::proposed_labels`]), plus an [`UNSPOTTED`] marker when an export leaves the frame
//! out (a pending box, a training label not in the class list): [`spot_obs`], the one source of
//! parked-spot observations for the walk and the index, then takes none of its boxes. The markers
//! are no candidates.
//!
//! **Evidence** ([`evidence`]) of a pending box: [`STRONG`] (1) with two or more labelers, `agree`,
//! `likely` or a VLM yes; else `sqrt(score)` x [`W_LONE`]. A VLM no multiplies either by
//! [`W_VLM_NO`], makes the box weak and marks it [`VLM_NO`] (two labelers agreeing on what the VLM
//! calls no is no strong box): it ranks lower, never hides. Here a VLM check counts by its `ok` in
//! either of its formats: the yes/no question, or an older run's category answer (`plant` where
//! `animal` was expected is a no; [`super::VlmCheck::says_no`] leaves those out for the rules that
//! decide boxes, the order only ranks). SigLIP's `species_scores` make no box strong: they sum to 1
//! over the wanted animals whatever the crop shows, so they name an animal without showing there is
//! one. A decided box (accepted, added) is 1.
//!
//! **Tally** ([`CellTally`], integers, so the order a walk adds frames in changes nothing), over
//! the reviewed frames the view counts (its scene, camera, bucket, class and source filters: a
//! camera's view weighs within that camera): training boxes in quarter-boxes (4 a box,
//! [`SPOT_QUARTERS`] at a known parked spot of the camera, [`Steer::spots`]) per (class, scene) and
//! per (camera, class, scene), per (camera, class, region of a 3 x 3 grid of the frame), whole
//! training boxes per class (the floor counts whole boxes, parked or not), the days per (camera,
//! class, scene), empty reviewed frames per scene, reviewed frames and those with an animal per
//! (camera, scene), per camera and class the reviewed frames a class was proposed on and kept on
//! ([`super::queue::Kept`]), and the hard negatives.
//!
//! **Score** of frame f (camera m, scene s, day d), per wanted class k of its candidates b:
//!
//! ```text
//! κ_b  = (kept + K_WEAK)/(proposed + K_WEAK) of m's proposals of k; (kept + K_STRONG)/(proposed + K_STRONG)
//!        for a strong box; 1 for a decided one
//! a_b  = e_b · (W_SPOT at a spot of m) · (W_MISSED if MISSED or GAP, e_b >= MISSED_MIN_E, κ_b >= MISSED_MIN_KAPPA)
//!            · (1 + W_REGION · T_R/(T_R + n_{m,k,region(b)}))/2 · κ_b
//! objects: candidates of other classes confused with each other (two animals, two vehicles),
//!          areas within a factor OBJECT_AREA, at IoU >= OBJECT_IOU or intersection over the smaller
//!          box >= OBJECT_INSIDE, are one object, never joining two decided boxes; only its best
//!          class (a decided box's, else the largest Σ a_b · w_k) keeps its boxes
//! E_k  = crowd(Σ a_b) = min(S, 1) + MORE_WEIGHT · min(max(S − 1, 0), MORE_BOXES)
//!                                 w_k = v_k · D_k · F_k · r_k · U_k · g_k
//! v_k  = ½·T/(T + n_{k,s}) + ½·T_CAM/(T_CAM + n_{m,k,s})
//! D_k  = 1 + W_DAY if d is a new day of (m, k, s), else 1
//! F_k  = RARE_GAIN under CLASS_FLOOR whole reviewed boxes, else 1; when the class is mostly false
//!        (FALSE_MIN_PROPOSED or more judged proposals over every camera, under FALSE_KEPT_SHARE of
//!        them kept) only a frame with a strong or decided box of it (after objects) keeps the gain
//! r_k  = 1 + RECALL_GAIN · m̂   (m̂: the shrunk held-out miss rate, recall::Measured::miss)
//! U_k  = CATCHUP_GAIN if k is camera m's lowest-recall class (within CATCHUP_TOL), else 1
//! g_k  = HELDOUT_GAIN on the split the cell (m, k, s) needs: d held out while h_{m,k,s} < HELDOUT_FLOOR,
//!        d a training day once h_{m,k,s} >= HELDOUT_FLOOR; else 1   (h: held-out boxes)
//! V    = Σ_k E_k · w_k
//! H    = W_HARDNEG · crowd(Σ_b f_b · T_H/(T_H + h_{m,k}) · v_k · region_b)
//!        f_b = max(1 − κ_b, VLM_NO_FALSE if a VLM no) for a likely false box (κ_b < LIKELY_FALSE_KAPPA
//!        or a VLM no), else 0; h_{m,k}: m's hard negatives of k
//!        H = 0 on a held-out day (a held-out frame never trains), and faint boxes are no candidates there
//! empty frame: V = W_EMPTY · T/(T + empty_s) · (1 + ANIMAL_GAIN · p_{m,s}), H = 0
//! score = round(1000 · (V + H))
//! ```
//!
//! `n` are reviewed boxes (quarter-boxes / 4). Classes are summed in class-list order, boxes in the
//! order of their candidates. The recall factor: a class the trained model lacks
//! ([`super::recall::InRun::Absent`]) needs most (`r = 1 + RECALL_GAIN`, and catch-up applies); a
//! class the run dropped on purpose ([`super::recall::InRun::Dropped`]), a scene without a measured
//! run and no measured run at all are `r = U = 1`. Catch-up is water-filling within a camera: its
//! lowest-recall class among those measured on at least [`super::recall::PRIOR_GT`] held-out
//! objects of the camera ([`super::recall::CameraRecall::well_supported`]) is boosted, ties and
//! classes within [`CATCHUP_TOL`] of it too, until it reaches the next one. `H` is the
//! hard-negative share: a box the camera's rejects say is likely false still teaches the model once
//! rejected, until the camera has about [`T_H`] hard negatives of the class; it is weighed like a
//! box by its class's diminishing returns and its region, never by the floor, recall or catch-up
//! gains (a false box adds no recall), so a real object outranks a likely false one of the same
//! class at equal counts.
//!
//! **Held-out steering** ([`Steer::heldout_listed`], on with a steer that carries the list, as the
//! server's always does): a camera day is held out ([`Need::heldout_day`]) when
//! `heldout-days.json` lists it or the export's hash split holds it out at the default fraction
//! ([`super::export::hash_heldout`]): what the next export of the default split, and a training run
//! from it, hold out (see [`Need::heldout_day`] for where they can differ). The tally counts the
//! training boxes of held-out days again, apart, per (camera, class, scene) in quarter-boxes (`h`).
//! A cell short of [`HELDOUT_FLOOR`] held-out boxes pulls its held-out days up; one at the floor
//! pulls its training days. On a held-out day [`W_MISSED`], the [`GAP`] candidates and `H` are off:
//! they would fill the held-out set with the hard cases the camera model missed or falsely
//! proposed, its figures would read low, and a rejected box there teaches nothing. Per-cell recall stays exact over the cell's own images; the random picks
//! ([`super::recommend::random_slots`]), scored apart as `heldout:random`, stay the unbiased figure.
//!
//! **Visits** ([`day`]): per event (one camera's activity, [`super::recommend::chain`]) the classes
//! of its decided training boxes ([`DayOut::visit_classes`]): an event holding a box of a class
//! under the floor or of the camera's catch-up class divides by `1 + rank · VISIT_P` rather than `1
//! + rank` ([`Scorer::expands`] names that class: more poses of a rare visit). And the gaps: a class
//! tracked (decided training boxes and pending [`STRONG`] ones) in an earlier and a later frame of
//! the event, each within [`GAP_S`] of the frame, with no box of the class here at IoU >=
//! [`GAP_IOU`] to the box interpolated between them, adds a synthetic [`GAP`] candidate of evidence
//! [`GAP_EVIDENCE`]: something the camera and the labelers lost for a frame.
//!
//! Constants, and why:
//! - [`T`] = 100, [`T_CAM`] = 30: reviewed boxes at which a cell's need halves, over every camera
//!   and on one camera. A class at a hundred boxes in a scene trains well; one camera's view of it
//!   needs about a third of that (its backgrounds and angles), so a camera's own weak class keeps
//!   pulling after the scene is well covered. [`T_R`] = 10: a region of a camera's view is new
//!   until about ten boxes there (persons and animals included, which never form spots).
//! - [`crowd`] ([`MORE_WEIGHT`] = 0.5, [`MORE_BOXES`] = 2): the first box of a class on a frame is
//!   worth the most, each further one half as much, and three full boxes (2) are the most a class
//!   of a frame counts: a frame of five cars teaches little more than one of two, and a storm of
//!   weak boxes never reaches a single strong one's worth by number.
//! - [`MIN_EVIDENCE`] = 0.1, [`MAX_PER_CLASS`] = 8: a box under a tenth of evidence (a VLM no on a
//!   lone low score) adds nothing a reviewer would keep, but one likely false still teaches once
//!   rejected; eight of a class are more than the cap counts.
//! - [`CLASS_FLOOR`] = 20 and [`RARE_GAIN`] = 3: under 20 boxes the benchmark scores no class
//!   ([`crate::benchmark::metrics::MIN_CLASS_GT`]), so the first 20 come first. A parked box counts
//!   whole here: the floor is about how many boxes exist. [`FALSE_MIN_PROPOSED`] = 20 and
//!   [`FALSE_KEPT_SHARE`] = 0.2: a class proposed on 20 reviewed frames and kept on fewer than one
//!   in five is mostly a labeler's false positive (lone SAM 3 cats): its weak boxes leave the floor,
//!   while a frame with a strong box of it keeps the gain (a real cat is still rare), lowered per
//!   camera by κ as that camera's rejects accumulate.
//! - [`RECALL_GAIN`] = 2: a class missed half the time weighs twice one never missed.
//! - [`CATCHUP_GAIN`] = 2, [`CATCHUP_TOL`] = 0.05: a camera's lowest class weighs double until it
//!   is within five points of the next; one held-out object of twenty moves recall by five points.
//! - [`W_SPOT`] = 1/4: a parked car at its spot teaches little; it counts a quarter in the tally
//!   ([`SPOT_QUARTERS`]) and in a frame's need.
//! - [`W_MISSED`] = 2: a missed object is the recall the camera lacks today; only for a box of
//!   evidence [`MISSED_MIN_E`] = 0.5 or more whose camera keeps at least [`MISSED_MIN_KAPPA`] = 0.5
//!   of its class's proposals, so a labeler's false-positive pattern never gets the maximum.
//! - [`K_WEAK`] = 1, [`K_STRONG`] = 5: the camera's kept share of a class's proposals, with a
//!   prior worth one kept proposal for a weak box and five for a strong one: rejects lower strong
//!   boxes too, more gently (30 rejected strong proposals bring them to a seventh).
//! - [`OBJECT_IOU`] = 0.5, [`OBJECT_INSIDE`] = 0.8, [`OBJECT_AREA`] = 2: one animal proposed as
//!   cat, dog and raccoon is one object, worth its best class once. Only classes labelers confuse
//!   join (an animal with an animal, a vehicle with a vehicle), of similar sizes: a person on a
//!   motorcycle, a person by a car, a cat on a car stay two; two decided boxes are two objects a
//!   reviewer drew.
//! - [`W_DAY`] = 0.5: a new day of a cell brings other light and weather.
//! - [`W_LONE`] = 0.5, [`W_VLM_NO`] = 0.25: a lone weak box is often false; one the VLM rejects
//!   mostly is.
//! - [`W_EMPTY`] = 0.3, [`ANIMAL_GAIN`] = 3: an empty frame is a negative (worth a third of a
//!   fresh box), more so on a camera whose reviewed frames held animals (`p_{m,s}`, their share):
//!   an "empty" frame there may hold the animal nobody proposed.
//! - [`W_HARDNEG`] = 0.5, [`T_H`] = 20, [`LIKELY_FALSE_KAPPA`] = 0.5, [`VLM_NO_FALSE`] = 0.5: a
//!   likely false box is worth at most half a fresh box of its class; about twenty hard negatives
//!   of a camera's class teach what its false boxes look like, after which the share halves; a box
//!   is likely false when its camera keeps under half of the class's proposals, or the VLM said no
//!   (then at least an even chance).
//! - [`CUE_SHARE`] = 1/4: the page names a cue (missed, a new place, a parked spot, a gap) only of
//!   boxes that carry at least a quarter of their class's weight.
//! - [`MISSED_IOU`] = 0.5: one object; [`GAP_IOU`] = 0.3 and [`GAP_S`] = 20 s: an object moves, so
//!   the interpolated box is loose, and over 20 s it may have left.
//! - [`GAP_EVIDENCE`] = 0.5: as a lone weak box of score 1.
//! - [`HELDOUT_FLOOR`] = [`crate::benchmark::metrics::MIN_CLASS_GT`] (20): the benchmark scores a
//!   class from 20 held-out objects, so a cell's held-out boxes are wanted up to that, then its
//!   training boxes. [`HELDOUT_GAIN`] = 2: the split a cell needs weighs double; less than the
//!   floor's [`RARE_GAIN`], so a rare class's frames still come first on either split.

use super::bulk::Wanted;
use super::queue::Kept;
use super::recall::{CameraRecall, InRun, Measured, Miss};
use super::repeats::same_object;
use super::{
    BoxState, LabelBox, LabelFile, Origin, Source, Status, area, inside_share, iou, statics,
};
use crate::collect::record::{Det, Record};
use crate::training::autolabel::Category;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

/// A box at a known parked spot of its camera weighs this much ([`SPOT_QUARTERS`] in the tally).
pub const W_SPOT: f64 = 0.25;
/// Quarter-boxes a reviewed box adds to the tally, and one at a parked spot.
pub const BOX_QUARTERS: u64 = 4;
pub const SPOT_QUARTERS: u64 = 1;
/// A [`MISSED`] or [`GAP`] candidate weighs this much more (see [`MISSED_MIN_E`]).
pub const W_MISSED: f64 = 2.0;
/// Least evidence, and least kept share of its camera's proposals, of a box [`W_MISSED`] raises.
pub const MISSED_MIN_E: f64 = 0.5;
pub const MISSED_MIN_KAPPA: f64 = 0.5;
/// Weight of a box's region novelty within its camera.
pub const W_REGION: f64 = 1.0;
/// [`crowd`]: what each box of a class past the first is worth, and how many such boxes count.
pub const MORE_WEIGHT: f64 = 0.5;
pub const MORE_BOXES: f64 = 2.0;

/// What `s` (the summed weight of one class's boxes on a frame, or of its likely false boxes)
/// counts as: the first box whole, further ones at [`MORE_WEIGHT`], up to [`MORE_BOXES`] more.
pub fn crowd(s: f64) -> f64 {
    s.min(1.0) + MORE_WEIGHT * (s - 1.0).clamp(0.0, MORE_BOXES)
}
/// Pending candidates under this evidence are left out; at most this many of a class are kept.
pub const MIN_EVIDENCE: f64 = 0.1;
pub const MAX_PER_CLASS: usize = 8;
/// Reviewed boxes at which a class's need in a scene halves, over every camera.
pub const T: f64 = 100.0;
/// The same on one camera.
pub const T_CAM: f64 = 30.0;
/// Reviewed boxes at which a region of a camera's view stops being new (halfway).
pub const T_R: f64 = 10.0;
/// What a new day of a cell adds.
pub const W_DAY: f64 = 0.5;
/// A class under [`CLASS_FLOOR`] whole reviewed boxes weighs this much more.
pub const RARE_GAIN: f64 = 3.0;
/// Whole reviewed training boxes (every camera and scene, parked ones too) under which a class is
/// rare.
pub const CLASS_FLOOR: u64 = 20;
/// A class with this many judged proposals (reviewed frames, every camera) of which fewer than
/// [`FALSE_KEPT_SHARE`] were kept is mostly false: it leaves the floor.
pub const FALSE_MIN_PROPOSED: u64 = 20;
pub const FALSE_KEPT_SHARE: f64 = 0.2;
/// How much a held-out miss rate raises a class's need (`r = 1 + RECALL_GAIN · m̂`).
pub const RECALL_GAIN: f64 = 2.0;
/// A camera's catch-up class weighs this much more.
pub const CATCHUP_GAIN: f64 = 2.0;
/// Recall within this of a camera's lowest is caught up too.
pub const CATCHUP_TOL: f64 = 0.05;
/// The kept share's prior, in kept proposals, for a weak and for a strong box.
pub const K_WEAK: f64 = 1.0;
pub const K_STRONG: f64 = 5.0;
/// Candidates of other classes this close, of areas within a factor [`OBJECT_AREA`], are one
/// object.
pub const OBJECT_IOU: f32 = 0.5;
pub const OBJECT_INSIDE: f32 = 0.8;
pub const OBJECT_AREA: f32 = 2.0;
/// What an empty frame weighs, against a fresh box.
pub const W_EMPTY: f64 = 0.3;
/// How much a camera's share of reviewed frames with an animal raises its empty frames.
pub const ANIMAL_GAIN: f64 = 3.0;
/// A pending box without strong evidence weighs this much of `sqrt(score)`.
pub const W_LONE: f64 = 0.5;
/// And this much more when the VLM said no.
pub const W_VLM_NO: f64 = 0.25;
/// The hard-negative share: its weight, the camera's hard negatives of a class at which it halves,
/// the kept share under which a box is likely false, and the falseness of a VLM no.
pub const W_HARDNEG: f64 = 0.5;
pub const T_H: f64 = 20.0;
pub const LIKELY_FALSE_KAPPA: f64 = 0.5;
pub const VLM_NO_FALSE: f64 = 0.5;
/// Share of its class's weight from which a box's cues are named.
pub const CUE_SHARE: f64 = 0.25;
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
/// Held-out boxes of a cell (camera, class, scene) wanted before its training days pull instead.
pub const HELDOUT_FLOOR: u64 = crate::benchmark::metrics::MIN_CLASS_GT as u64;
/// The split a cell needs weighs this much more (`g_k`).
pub const HELDOUT_GAIN: f64 = 2.0;
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
/// A reviewed frame's hard negative (no candidate).
pub const REJECTED: u8 = 16;
/// A reviewed frame's marker of a class proposed there and judged (no candidate, no box).
pub const PROPOSED: u8 = 32;
/// The VLM said no.
pub const VLM_NO: u8 = 64;
/// A reviewed frame's marker: an export leaves the frame out (a pending box, or a training label
/// off the class list), so it is no parked-spot evidence ([`spot_obs`]).
pub const UNSPOTTED: u8 = 128;

/// The scenes of the tally; any other scene is the fourth.
const SCENES: [&str; 4] = ["day", "night", "ir", "other"];
/// The region of a box of a frame of unknown size; regions in all.
const NO_REGION: u8 = 9;
const REGIONS: usize = 10;

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

    /// A candidate of the need: no hard negative or proposal marker.
    pub fn is_candidate(&self) -> bool {
        self.flags & (REJECTED | PROPOSED | UNSPOTTED) == 0
    }

    /// Tracked through a visit ([`day`]): decided, or pending with strong evidence.
    fn tracked(&self) -> bool {
        self.is_candidate() && !self.has(GAP) && (self.has(TRAINING) || self.has(STRONG))
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

/// The date of a day folder `<scene>/<camera>/<date>` (its last part).
pub fn date_of(day: &str) -> &str {
    day.rsplit_once('/').map_or(day, |(_, d)| d)
}

/// The evidence of a pending box of score `score` with the auto-labelers' findings `auto` (None:
/// no auto-labeler looked), in hundredths, whether it is [`STRONG`], and whether the VLM said no.
pub fn evidence(score: Option<f32>, auto: Option<&super::AutoInfo>) -> (u8, bool, bool) {
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
        return (hundredths(e * W_VLM_NO), false, true);
    }
    (hundredths(e), strong, false)
}

/// The flags of evidence `(strong, vlm_no)`.
fn evidence_flags(strong: bool, vlm_no: bool) -> u8 {
    (if strong { STRONG } else { 0 }) | if vlm_no { VLM_NO } else { 0 }
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

/// A pending candidate: what [`MIN_EVIDENCE`] and [`MAX_PER_CLASS`] thin out.
fn thinned(b: &CellBox) -> bool {
    b.is_candidate() && !b.has(TRAINING)
}

/// `c` sorted by class (stable), at most [`MAX_PER_CLASS`] pending candidates of a class kept (the
/// strongest; ties by order). Decided boxes and markers all stay. Candidates under
/// [`MIN_EVIDENCE`] stay too: whether one counts depends on the camera ([`Scorer::score`] keeps a
/// [`faint`] one only when it is likely false).
fn finish(c: Vec<CellBox>) -> Vec<CellBox> {
    let mut kept: Vec<(usize, CellBox)> = c.into_iter().enumerate().collect();
    // Strongest first within a class, to keep the first few.
    kept.sort_by_key(|(i, b)| (b.class, thinned(b), std::cmp::Reverse(b.evidence), *i));
    let mut out: Vec<(usize, CellBox)> = Vec::with_capacity(kept.len());
    for run in kept.chunk_by(|a, b| a.1.class == b.1.class) {
        let mut pending = 0usize;
        for (i, b) in run {
            if thinned(b) {
                pending += 1;
                if pending > MAX_PER_CLASS {
                    continue;
                }
            }
            out.push((*i, *b));
        }
    }
    out.sort_by_key(|(i, b)| (b.class, *i));
    out.into_iter().map(|(_, b)| b).collect()
}

/// A pending candidate under [`MIN_EVIDENCE`]: it adds no recall need; it counts only when it is
/// likely false ([`Scorer::likely_false`]), for its hard-negative share alone.
fn faint(b: &CellBox) -> bool {
    thinned(b) && b.e() < MIN_EVIDENCE
}

/// The parked-spot observations of a reviewed frame of `boxes` ([`of_file`]: its training boxes),
/// by the export's rules, the one way the walk and the review index both make the spot table:
/// none when [`UNSPOTTED`] (pending boxes, a label off the class list), else each training box of
/// a class of `classes`, its whole-pixel box normalised to the `width` x `height` frame. `rel`'s
/// path names the camera, date and scene.
pub fn spot_obs(
    rel: &str,
    width: u32,
    height: u32,
    boxes: &[CellBox],
    classes: &[String],
) -> Vec<statics::Obs> {
    let mut parts = rel.split('/');
    let (Some(scene), Some(camera), Some(date)) = (parts.next(), parts.next(), parts.next()) else {
        return Vec::new();
    };
    if boxes.iter().any(|b| b.has(UNSPOTTED)) {
        return Vec::new();
    }
    boxes
        .iter()
        .filter(|b| b.has(TRAINING))
        .filter_map(|b| {
            Some(statics::Obs {
                camera: camera.to_string(),
                frame: rel.to_string(),
                day: date.to_string(),
                scene: scene.to_string(),
                label: classes.get(usize::from(b.class))?.clone(),
                bbox: statics::normalised(&b.rect(), width, height)?,
            })
        })
        .collect()
}

/// The candidates of a frame with label file `l` (see the module docs); of a reviewed frame, what
/// the tally reads.
pub fn of_file(l: &LabelFile, w: Option<&Wanted<'_>>) -> Vec<CellBox> {
    let Some(w) = w else { return Vec::new() };
    let mut out = Vec::new();
    if l.status == Status::Reviewed {
        // As the export: a frame with a pending box or a label off the list is left out.
        if l.boxes.iter().any(|b| b.state == BoxState::Pending)
            || l.training().any(|b| !w.classes.contains(&b.label))
        {
            out.push(CellBox {
                flags: UNSPOTTED,
                ..CellBox::default()
            });
        }
        for b in l.training().filter(|b| !b.is_ignore()) {
            if let Some(k) = class_index(w, &b.label) {
                out.push(CellBox {
                    class: k,
                    bbox: pixels(&b.bbox),
                    evidence: 100,
                    flags: TRAINING | STRONG,
                });
            }
        }
        for b in l.hard_negatives() {
            if let Some(k) = class_index(w, &b.label) {
                out.push(CellBox {
                    class: k,
                    bbox: pixels(&b.bbox),
                    evidence: 0,
                    flags: REJECTED,
                });
            }
        }
        let proposed: BTreeSet<u16> = super::queue::proposed_labels(l)
            .into_iter()
            .filter_map(|p| class_index(w, p))
            .collect();
        out.extend(proposed.into_iter().map(|k| CellBox {
            class: k,
            flags: PROPOSED,
            ..CellBox::default()
        }));
        return finish(out);
    }
    for b in &l.boxes {
        if b.is_ignore() || b.state == BoxState::Rejected {
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
                let (e, strong, no) = evidence(b.score, auto);
                let missed = if missed(b, l) { MISSED } else { 0 };
                (e, evidence_flags(strong, no) | missed)
            };
            out.push(CellBox {
                class: k,
                bbox: pixels(&b.bbox),
                evidence,
                flags,
            });
        }
        if let Some(k) = suggested {
            let (e, strong, no) = evidence(b.auto.auto_score.or(b.score), Some(&b.auto));
            out.push(CellBox {
                class: k,
                bbox: pixels(&b.bbox),
                evidence: e,
                flags: MISSED | evidence_flags(strong, no),
            });
        }
    }
    finish(out)
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
        let (e, _, _) = evidence(Some(score), None);
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
    finish(out)
}

/// What steers the order from outside the folder (the server's `review::steer`), one snapshot per
/// query, handed to the walk and the index alike ([`super::queue::Filters::steer`]). Compared by
/// the identity of its parts (a snapshot equals another made of the same tables), and shown as a
/// summary.
#[derive(Clone, Default)]
pub struct Steer {
    /// The measured recall of the newest trained models; None = nothing measured.
    pub recall: Option<Arc<Measured>>,
    /// The folder's parked spots (empty until the first build: neutral).
    pub spots: Arc<statics::Table>,
    /// Held-out steering (see the module docs): the camera days `heldout-days.json` lists
    /// (`<camera>/<date>`; empty when it is missing, newer or damaged), with which the hash split's
    /// days are held out too. None = no held-out steering (the default: no day is held out).
    pub heldout_listed: Option<Arc<BTreeSet<String>>>,
}

/// Two optional tables are the same one.
fn same_arc<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl PartialEq for Steer {
    fn eq(&self, other: &Self) -> bool {
        same_arc(self.recall.as_ref(), other.recall.as_ref())
            && Arc::ptr_eq(&self.spots, &other.spots)
            && same_arc(self.heldout_listed.as_ref(), other.heldout_listed.as_ref())
    }
}

impl std::fmt::Debug for Steer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Steer")
            .field(
                "recall_slices",
                &self
                    .recall
                    .as_ref()
                    .map(|m| m.scenes.len() + usize::from(m.overall.is_some())),
            )
            .field("spots", &self.spots.len())
            .field(
                "heldout_listed",
                &self.heldout_listed.as_ref().map(|l| l.len()),
            )
            .finish()
    }
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
    /// (class, scene) -> quarter-boxes of held-out days (counted in `cells` too).
    heldout: BTreeMap<(u16, u8), u64>,
    /// (class, region) -> quarter-boxes.
    regions: BTreeMap<(u16, u8), u64>,
    /// (class, scene) -> the days with a box.
    days: BTreeMap<(u16, u8), BTreeSet<String>>,
    /// Reviewed frames per scene, and those with an animal's training box.
    frames: [u64; 4],
    animal_frames: [u64; 4],
    /// Per class: reviewed frames a model or auto-labeler proposed it on, and those that kept it.
    precision: BTreeMap<u16, Kept>,
    /// Per class: hard negatives.
    hard: BTreeMap<u16, u64>,
}

impl CameraCells {
    fn merge(&mut self, c: &CameraCells) {
        for (k, n) in &c.cells {
            *self.cells.entry(*k).or_default() += n;
        }
        for (k, n) in &c.heldout {
            *self.heldout.entry(*k).or_default() += n;
        }
        for (k, n) in &c.regions {
            *self.regions.entry(*k).or_default() += n;
        }
        for (k, days) in &c.days {
            let mine = self.days.entry(*k).or_default();
            for d in days {
                if !mine.contains(d) {
                    mine.insert(d.clone());
                }
            }
        }
        for (a, b) in self.frames.iter_mut().zip(c.frames) {
            *a += b;
        }
        for (a, b) in self.animal_frames.iter_mut().zip(c.animal_frames) {
            *a += b;
        }
        for (k, p) in &c.precision {
            let m = self.precision.entry(*k).or_default();
            m.kept += p.kept;
            m.proposed += p.proposed;
        }
        for (k, n) in &c.hard {
            *self.hard.entry(*k).or_default() += n;
        }
    }
}

/// Reviewed boxes per cell (see the module docs). Integers and sets only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellTally {
    /// (class, scene) -> quarter-boxes over every camera.
    class_scene: BTreeMap<(u16, u8), u64>,
    /// Class -> whole training boxes over every camera and scene.
    class_boxes: BTreeMap<u16, u64>,
    /// Reviewed frames with no wanted training box, per scene.
    empty: [u64; 4],
    cameras: BTreeMap<String, CameraCells>,
}

/// About the heap bytes of `m`'s entries (a B-tree node's share each), for [`CellTally::bytes`].
fn map_bytes<K, V>(m: &BTreeMap<K, V>) -> usize {
    m.len() * (size_of::<(K, V)>() + 2 * size_of::<usize>())
}

impl CellTally {
    /// About its bytes (the review index's ceiling counts the parts it keeps).
    pub fn bytes(&self) -> usize {
        let cameras: usize = self
            .cameras
            .iter()
            .map(|(name, c)| {
                let days: usize = c
                    .days
                    .values()
                    .map(|d| {
                        d.iter().map(String::capacity).sum::<usize>()
                            + d.len() * (size_of::<String>() + 2 * size_of::<usize>())
                    })
                    .sum();
                name.capacity()
                    + map_bytes(&c.cells)
                    + map_bytes(&c.heldout)
                    + map_bytes(&c.regions)
                    + map_bytes(&c.days)
                    + days
                    + map_bytes(&c.precision)
                    + map_bytes(&c.hard)
            })
            .sum();
        size_of::<Self>()
            + map_bytes(&self.class_scene)
            + map_bytes(&self.class_boxes)
            + map_bytes(&self.cameras)
            + cameras
    }

    /// Add another part of a walk.
    pub fn merge(&mut self, other: Self) {
        self.merge_ref(&other);
    }

    /// [`Self::merge`] of a part kept elsewhere (the index's per-shard parts).
    pub fn merge_ref(&mut self, other: &Self) {
        for (k, n) in &other.class_scene {
            *self.class_scene.entry(*k).or_default() += n;
        }
        for (k, n) in &other.class_boxes {
            *self.class_boxes.entry(*k).or_default() += n;
        }
        for (a, b) in self.empty.iter_mut().zip(other.empty) {
            *a += b;
        }
        for (camera, c) in &other.cameras {
            match self.cameras.get_mut(camera) {
                Some(mine) => mine.merge(c),
                None => {
                    self.cameras.insert(camera.clone(), c.clone());
                }
            }
        }
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
    /// [`of_file`] of the reviewed frame: training boxes, hard negatives, proposal markers.
    pub boxes: &'a [CellBox],
    /// Its day is held out ([`Need::heldout_day`]).
    pub heldout: bool,
}

/// A frame to weigh, for [`Scorer::score`].
#[derive(Debug, Clone, Copy)]
pub struct Facts<'a> {
    pub camera: &'a str,
    pub scene: &'a str,
    pub day: &'a str,
    pub width: u32,
    pub height: u32,
    /// Its day is held out ([`Need::heldout_day`]).
    pub heldout: bool,
}

/// What weighs frames of one query: the class list and the steer.
#[derive(Debug)]
pub struct Need<'a> {
    wanted: Option<Wanted<'a>>,
    steer: Option<&'a Steer>,
    /// Per class: an animal ([`Category::Animal`]), may form a parked spot, and its category (what
    /// a class is confused with: one object, [`objects`]).
    animal: Vec<bool>,
    may_park: Vec<bool>,
    category: Vec<Category>,
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
            category: classes.iter().map(|c| Category::of(c)).collect(),
            wanted,
            steer,
        }
    }

    pub fn wanted(&self) -> Option<&Wanted<'a>> {
        self.wanted.as_ref()
    }

    /// The parked spots this weighs by (None without a steer): what the index's cached tally parts
    /// are kept with, and compared to by identity.
    pub fn spots(&self) -> Option<&'a Arc<statics::Table>> {
        self.steer.map(|s| &s.spots)
    }

    /// The listed held-out days when held-out steering is on (see the module docs).
    fn listed(&self) -> Option<&'a BTreeSet<String>> {
        self.steer.and_then(|s| s.heldout_listed.as_deref())
    }

    /// Held-out steering is on: a steer with the listed days.
    pub fn steers_heldout(&self) -> bool {
        self.listed().is_some()
    }

    /// Whether the day folder `day` (`<scene>/<camera>/<date>`) is a held-out day: with held-out
    /// steering on, its camera day listed in `heldout-days.json` or held out by the export's hash
    /// split at [`super::export::DEFAULT_HOLDOUT_FRACTION`] ([`super::export::hash_heldout`]);
    /// never without. The one test the walk and the index ask, per day folder. This is what the
    /// next export of the default split and a training run from it hold out, but for: the export's
    /// "one group on each side" guard (only a folder with every day on one side), the days an
    /// export adds for a camera under its minimum of held-out days (listed from that export on),
    /// an export of other split settings, and a training run whose exports had no held-out frame
    /// (it splits by its own share).
    pub fn heldout_day(&self, day: &str) -> bool {
        let Some(listed) = self.listed() else {
            return false;
        };
        let mut parts = day.split('/');
        let (Some(_scene), Some(camera), Some(date), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return false;
        };
        listed.contains(&format!("{camera}/{date}"))
            || super::export::hash_heldout(
                camera,
                date,
                f64::from(super::export::DEFAULT_HOLDOUT_FRACTION),
            )
    }

    /// The class list's name of class `k`.
    pub fn name(&self, k: u16) -> &'a str {
        self.wanted
            .and_then(|w| w.classes.get(usize::from(k)))
            .map_or("", String::as_str)
    }

    /// The parked spots of `camera` (none without a steer).
    fn spots_of(&self, camera: &str) -> &'a [statics::Spot] {
        self.steer.map_or(&[][..], |s| s.spots.camera(camera))
    }

    /// `b`, a box of a `w` x `h` frame, lies at one of `spots` (its camera's).
    fn at_spot(&self, spots: &[statics::Spot], b: &CellBox, w: u32, h: u32) -> bool {
        if spots.is_empty()
            || !self
                .may_park
                .get(usize::from(b.class))
                .copied()
                .unwrap_or(false)
        {
            return false;
        }
        let t = self.steer.map_or(statics::SPOT_IOU, |s| s.spots.iou);
        statics::normalised(&b.rect(), w, h)
            .is_some_and(|nb| statics::spot_at_iou(spots, self.name(b.class), &nb, t).is_some())
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
        let spots = self.spots_of(r.camera);
        let mut trained: Vec<u16> = Vec::new();
        let mut animal = false;
        for b in r.boxes {
            if b.has(REJECTED) {
                *cam.hard.entry(b.class).or_default() += 1;
                continue;
            }
            if !b.has(TRAINING) {
                continue;
            }
            let q = if self.at_spot(spots, b, r.width, r.height) {
                SPOT_QUARTERS
            } else {
                BOX_QUARTERS
            };
            *t.class_scene.entry((b.class, s)).or_default() += q;
            *t.class_boxes.entry(b.class).or_default() += 1;
            *cam.cells.entry((b.class, s)).or_default() += q;
            if r.heldout {
                *cam.heldout.entry((b.class, s)).or_default() += q;
            }
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
        for b in r.boxes.iter().filter(|b| b.has(PROPOSED)) {
            let p = cam.precision.entry(b.class).or_default();
            p.proposed += 1;
            if trained.contains(&b.class) {
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
}

/// What [`day`] finds for a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DayOut {
    /// Synthetic [`GAP`] candidates, by class then box.
    pub gaps: Vec<CellBox>,
    /// The classes of the decided training boxes of its event, sorted, each once.
    pub visit_classes: Vec<u16>,
}

const GAP_MS: i64 = GAP_S * 1000;

/// A set of classes (the class list holds at most 256: [`crate::collect::config::MAX_CLASSES`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ClassSet([u64; 4]);

const _: () = assert!(crate::collect::config::MAX_CLASSES <= 256);

impl ClassSet {
    fn insert(&mut self, k: u16) {
        if let Some(w) = self.0.get_mut(usize::from(k / 64)) {
            *w |= 1u64 << (k % 64);
        }
    }

    fn contains(&self, k: u16) -> bool {
        self.0
            .get(usize::from(k / 64))
            .is_some_and(|w| w & (1u64 << (k % 64)) != 0)
    }

    fn union(&mut self, o: &Self) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a |= b;
        }
    }

    fn intersect(&mut self, o: &Self) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a &= b;
        }
    }

    fn iter(self) -> impl Iterator<Item = u16> {
        (0..256u16).filter(move |k| self.contains(*k))
    }

    fn of_tracked(boxes: &[CellBox]) -> Self {
        let mut s = Self::default();
        for b in boxes.iter().filter(|b| b.tracked()) {
            s.insert(b.class);
        }
        s
    }
}

/// The gaps and visits of one day folder's counted frames (see the module docs), per frame in
/// their order. The order they come in changes nothing. Each event's frames are sorted by time
/// once, and a frame looks only at the frames within [`GAP_S`] of it (by binary search), through
/// each frame's set of tracked classes.
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
    let tracked: Vec<ClassSet> = frames
        .iter()
        .map(|f| ClassSet::of_tracked(f.boxes))
        .collect();
    let mut out: Vec<DayOut> = vec![DayOut::default(); frames.len()];
    for m in &members {
        let mut classes = BTreeSet::new();
        for f in m.iter().filter_map(|i| frames.get(*i)) {
            classes.extend(f.boxes.iter().filter(|b| b.has(TRAINING)).map(|b| b.class));
        }
        let visit: Vec<u16> = classes.into_iter().collect();
        // The event's timed frames, by time.
        let mut timed: Vec<(i64, usize)> = m
            .iter()
            .filter_map(|i| Some((times.get(*i).copied().flatten()?, *i)))
            .collect();
        timed.sort_unstable();
        for i in m {
            let gaps = times
                .get(*i)
                .copied()
                .flatten()
                .map(|t| gaps_at(*i, t, frames, &timed, &tracked))
                .unwrap_or_default();
            if let Some(o) = out.get_mut(*i) {
                o.visit_classes.clone_from(&visit);
                o.gaps = gaps;
            }
        }
    }
    out
}

/// The tracked boxes of class `k` of the frames of `timed` at time `t`, sorted.
fn tracked_at(frames: &[DayFrame<'_>], timed: &[(i64, usize)], t: i64, k: u16) -> Vec<CellBox> {
    let from = timed.partition_point(|(ms, _)| *ms < t);
    let to = timed.partition_point(|(ms, _)| *ms <= t);
    let mut out: Vec<CellBox> = timed
        .get(from..to)
        .unwrap_or_default()
        .iter()
        .filter_map(|(_, j)| frames.get(*j))
        .flat_map(|f| f.boxes.iter().copied())
        .filter(|b| b.class == k && b.tracked())
        .collect();
    out.sort_unstable();
    out
}

/// The [`GAP`] candidates of frame `i` at time `t` among its event's frames `timed` (by time).
fn gaps_at(
    i: usize,
    t: i64,
    frames: &[DayFrame<'_>],
    timed: &[(i64, usize)],
    tracked: &[ClassSet],
) -> Vec<CellBox> {
    let Some(here) = frames.get(i) else {
        return Vec::new();
    };
    let lo = timed.partition_point(|(ms, _)| *ms < t.saturating_sub(GAP_MS));
    let hi = timed.partition_point(|(ms, _)| *ms <= t.saturating_add(GAP_MS));
    let window = timed.get(lo..hi).unwrap_or_default();
    let (mut both, mut after) = (ClassSet::default(), ClassSet::default());
    for (ms, j) in window {
        let set = tracked.get(*j).copied().unwrap_or_default();
        if *ms < t {
            both.union(&set);
        } else if *ms > t {
            after.union(&set);
        }
    }
    both.intersect(&after);
    let mut out: Vec<CellBox> = Vec::new();
    for k in both.iter() {
        let has = |j: usize| tracked.get(j).is_some_and(|s| s.contains(k));
        let t0 = window
            .iter()
            .rev()
            .find(|(ms, j)| *ms < t && has(*j))
            .map(|(ms, _)| *ms);
        let t1 = window
            .iter()
            .find(|(ms, j)| *ms > t && has(*j))
            .map(|(ms, _)| *ms);
        let (Some(t0), Some(t1)) = (t0, t1) else {
            continue;
        };
        let (a, b) = (
            tracked_at(frames, timed, t0, k),
            tracked_at(frames, timed, t1, k),
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
            if here
                .boxes
                .iter()
                .filter(|o| o.is_candidate() && !o.has(GAP))
                .any(covered)
                || out.iter().any(covered)
            {
                continue;
            }
            out.push(gap);
        }
    }
    out.sort_unstable();
    out
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
    /// The top class's E_k, n_{m,k,s}, n_{k,s} and whole reviewed boxes over every scene.
    pub boxes: f64,
    pub camera_reviewed: f64,
    pub reviewed: f64,
    pub class_reviewed: f64,
    pub below_floor: bool,
    pub catch_up: bool,
    pub new_day: bool,
    /// Cues of the top class's boxes that carry at least [`CUE_SHARE`] of its weight.
    pub new_region: bool,
    pub spot: bool,
    pub missed: bool,
    pub gap: bool,
    pub strong: bool,
    /// The best evidence of the top class's boxes.
    pub evidence: f64,
    /// The camera's proposals of the top class when their kept share lowered one of its boxes.
    pub kept: Option<Kept>,
    /// The hard-negative share, and the class of its largest part.
    pub hard: f64,
    pub hard_class: Option<u16>,
    /// The frame adds no recall need, only likely false boxes: its score is their hard-negative
    /// share.
    pub hard_only: bool,
    /// An empty frame: the scene's reviewed empty frames and the camera's animal share there.
    pub empty: Option<(u64, f64)>,
    /// The frame's day is held out (held-out steering on).
    pub heldout_day: bool,
    /// With held-out steering on, the top class's held-out boxes on the camera in the scene, and
    /// whether `g_k` raised it.
    pub heldout_boxes: Option<f64>,
    pub heldout_gain: bool,
}

/// What one camera's frames are weighed by, per class (made when the camera is first asked).
#[derive(Debug)]
struct Cam<'a> {
    cells: Option<&'a CameraCells>,
    spots: &'a [statics::Spot],
    /// Per class, per scene: v and r; per class, per measured scene: the shrunk miss rate.
    v: Vec<[f64; 4]>,
    /// Per class, per scene: held-out boxes.
    heldout: Vec<[f64; 4]>,
    r: Vec<[f64; 4]>,
    miss: Vec<[Option<Miss>; 3]>,
    catch_up: Vec<bool>,
    /// Per class: the kept share for a weak and a strong box, the proposals, and the
    /// hard-negative factor `T_H/(T_H + h)`.
    kappa_weak: Vec<f64>,
    kappa_strong: Vec<f64>,
    kept: Vec<Kept>,
    hard: Vec<f64>,
    /// Per class, per region: reviewed boxes.
    regions: Vec<[f64; REGIONS]>,
    /// Per scene: the share of reviewed frames with an animal.
    animal: [f64; 4],
}

/// Weighs the frames of one query against a finished tally. What a class or a camera weighs by is
/// worked out once (per class at [`Scorer::new`], per camera when first asked), with the same
/// floating-point operations whichever path asks.
#[derive(Debug)]
pub struct Scorer<'a> {
    need: &'a Need<'a>,
    tally: &'a CellTally,
    /// Per class: under the floor (whole boxes), mostly false (its weak boxes leave the floor),
    /// and whole reviewed boxes.
    rare: Vec<bool>,
    mostly_false: Vec<bool>,
    boxes: Vec<f64>,
    /// Per class, per scene: reviewed boxes over every camera.
    n_ks: Vec<[f64; 4]>,
    /// The cameras made so far, by name.
    cams: RefCell<HashMap<String, Rc<Cam<'a>>>>,
}

/// `i` as a class index.
fn k16(i: usize) -> u16 {
    u16::try_from(i).unwrap_or(u16::MAX)
}

/// Quarter-boxes of `key` in `m`, as boxes.
fn quarters(m: Option<&BTreeMap<(u16, u8), u64>>, key: (u16, u8)) -> f64 {
    m.and_then(|m| m.get(&key)).copied().unwrap_or(0) as f64 / BOX_QUARTERS as f64
}

impl<'a> Scorer<'a> {
    pub fn new(need: &'a Need<'a>, tally: &'a CellTally) -> Self {
        let classes = need.wanted.map_or(0, |w| w.classes.len());
        let mut proposed = vec![Kept::default(); classes];
        for c in tally.cameras.values() {
            for (k, p) in &c.precision {
                if let Some(x) = proposed.get_mut(usize::from(*k)) {
                    x.kept += p.kept;
                    x.proposed += p.proposed;
                }
            }
        }
        let boxes: Vec<u64> = (0..classes)
            .map(|i| tally.class_boxes.get(&k16(i)).copied().unwrap_or(0))
            .collect();
        let rare = boxes.iter().map(|n| *n < CLASS_FLOOR).collect();
        let mostly_false = proposed
            .iter()
            .map(|p| {
                p.proposed >= FALSE_MIN_PROPOSED
                    && (p.kept as f64) < FALSE_KEPT_SHARE * p.proposed as f64
            })
            .collect();
        let n_ks = (0..classes)
            .map(|i| {
                let mut a = [0.0; 4];
                for (s, x) in a.iter_mut().enumerate() {
                    *x = quarters(
                        Some(&tally.class_scene),
                        (k16(i), u8::try_from(s).unwrap_or(3)),
                    );
                }
                a
            })
            .collect();
        Self {
            need,
            tally,
            rare,
            mostly_false,
            boxes: boxes.iter().map(|n| *n as f64).collect(),
            n_ks,
            cams: RefCell::new(HashMap::new()),
        }
    }

    fn measured(&self) -> Option<&'a Measured> {
        self.need.steer.and_then(|s| s.recall.as_deref())
    }

    /// What `camera`'s frames are weighed by.
    fn cam(&self, camera: &str) -> Rc<Cam<'a>> {
        if let Some(c) = self.cams.borrow().get(camera) {
            return c.clone();
        }
        let c = Rc::new(self.make_cam(camera));
        self.cams.borrow_mut().insert(camera.to_string(), c.clone());
        c
    }

    fn make_cam(&self, camera: &str) -> Cam<'a> {
        let classes = self.need.wanted.map_or(&[][..], |w| w.classes);
        let cells = self.tally.cameras.get(camera);
        let (miss, catch_up): (Vec<[Option<Miss>; 3]>, Vec<bool>) = match self.measured() {
            None => (
                vec![[None, None, None]; classes.len()],
                vec![false; classes.len()],
            ),
            Some(m) => {
                let miss = classes
                    .iter()
                    .map(|k| {
                        [
                            m.miss(camera, "day", k),
                            m.miss(camera, "night", k),
                            m.miss(camera, "ir", k),
                        ]
                    })
                    .collect();
                let recall: Vec<Option<CameraRecall>> = classes
                    .iter()
                    .map(|k| m.class_recall_on_camera(camera, k))
                    .collect();
                (miss, catch_up_of(&recall))
            }
        };
        let n = classes.len();
        let mut c = Cam {
            cells,
            spots: self.need.spots_of(camera),
            v: Vec::with_capacity(n),
            heldout: Vec::with_capacity(n),
            r: Vec::with_capacity(n),
            miss,
            catch_up,
            kappa_weak: Vec::with_capacity(n),
            kappa_strong: Vec::with_capacity(n),
            kept: Vec::with_capacity(n),
            hard: Vec::with_capacity(n),
            regions: Vec::with_capacity(n),
            animal: [0.0; 4],
        };
        for i in 0..n {
            let k = k16(i);
            let n_ks = self.n_ks.get(i).copied().unwrap_or_default();
            let mut vk = [0.0; 4];
            let mut rk = [1.0; 4];
            let mut hk = [0.0; 4];
            for (s, ((v, r), h)) in vk
                .iter_mut()
                .zip(rk.iter_mut())
                .zip(hk.iter_mut())
                .enumerate()
            {
                let key = (k, u8::try_from(s).unwrap_or(3));
                let n_mks = quarters(cells.map(|c| &c.cells), key);
                *h = quarters(cells.map(|c| &c.heldout), key);
                let n = n_ks.get(s).copied().unwrap_or(0.0);
                *v = 0.5 * T / (T + n) + 0.5 * T_CAM / (T_CAM + n_mks);
                *r = recall_factor(
                    c.miss
                        .get(i)
                        .and_then(|m| m.get(s))
                        .and_then(Option::as_ref),
                );
            }
            c.v.push(vk);
            c.heldout.push(hk);
            c.r.push(rk);
            let kept = cells
                .and_then(|c| c.precision.get(&k))
                .copied()
                .unwrap_or_default();
            c.kappa_weak
                .push((kept.kept as f64 + K_WEAK) / (kept.proposed as f64 + K_WEAK));
            c.kappa_strong
                .push((kept.kept as f64 + K_STRONG) / (kept.proposed as f64 + K_STRONG));
            c.kept.push(kept);
            let h = cells.and_then(|c| c.hard.get(&k)).copied().unwrap_or(0) as f64;
            c.hard.push(T_H / (T_H + h));
            let mut rg = [0.0; REGIONS];
            for (g, x) in rg.iter_mut().enumerate() {
                *x = quarters(
                    cells.map(|c| &c.regions),
                    (k, u8::try_from(g).unwrap_or(NO_REGION)),
                );
            }
            c.regions.push(rg);
        }
        if let Some(cells) = cells {
            for (s, x) in c.animal.iter_mut().enumerate() {
                let n = cells.frames.get(s).copied().unwrap_or(0);
                let a = cells.animal_frames.get(s).copied().unwrap_or(0);
                *x = if n == 0 { 0.0 } else { a as f64 / n as f64 };
            }
        }
        c
    }

    /// The shrunk miss rate of class `k` on `camera` in `scene` (None: not measured).
    pub fn miss(&self, camera: &str, scene: &str, k: u16) -> Option<Miss> {
        self.cam(camera)
            .miss
            .get(usize::from(k))
            .and_then(|m| m.get(usize::from(scene_ix(scene))))
            .cloned()
            .flatten()
    }

    /// Class `k` is under the floor of whole boxes.
    fn below_floor(&self, k: u16) -> bool {
        self.rare.get(usize::from(k)).copied().unwrap_or(false)
    }

    /// Class `k` is mostly false (see the module docs).
    fn mostly_false(&self, k: u16) -> bool {
        self.mostly_false
            .get(usize::from(k))
            .copied()
            .unwrap_or(false)
    }

    /// The floor's gain applies to a frame's class `k`, `strong` when it has a strong or decided
    /// box of it: under the floor, and for a mostly false class only with a strong box.
    fn floor_gain(&self, k: u16, strong: bool) -> bool {
        self.below_floor(k) && (strong || !self.mostly_false(k))
    }

    /// The class that makes the visit of a frame of `camera` expand, of the classes of its event's
    /// decided training boxes `visit_classes` ([`VISIT_P_QUARTERS`]): the first under the floor or
    /// the camera's catch-up class.
    pub fn expands(&self, camera: &str, visit_classes: &[u16]) -> Option<u16> {
        if visit_classes.is_empty() {
            return None;
        }
        let cam = self.cam(camera);
        visit_classes.iter().copied().find(|k| {
            self.below_floor(*k) || cam.catch_up.get(usize::from(*k)).copied().unwrap_or(false)
        })
    }

    /// The need of frame `f` with candidates `boxes` and gaps `gaps` (see the module docs).
    pub fn score(&self, f: &Facts<'_>, boxes: &[CellBox], gaps: &[CellBox]) -> Detail {
        let s8 = scene_ix(f.scene);
        let s = usize::from(s8);
        let cam = self.cam(f.camera);
        // No gap on a held-out day: a hard case for the held-out set. No hard-negative share
        // there either (a held-out frame never trains), so a faint likely false box is no
        // candidate on it.
        let gaps = if f.heldout { &[][..] } else { gaps };
        let cands: Vec<&CellBox> = boxes
            .iter()
            .chain(gaps)
            .filter(|b| {
                b.is_candidate() && (!faint(b) || (!f.heldout && self.likely_false(&cam, b)))
            })
            .collect();
        if cands.is_empty() {
            let empty = self.tally.empty.get(s).copied().unwrap_or(0);
            let share = cam.animal.get(s).copied().unwrap_or(0.0);
            let v = W_EMPTY * T / (T + empty as f64) * (1.0 + ANIMAL_GAIN * share);
            return Detail {
                score: score_of(v),
                empty: Some((empty, share)),
                heldout_day: f.heldout,
                ..Detail::default()
            };
        }
        // The classes present, ascending, with their weights w_k.
        let mut classes: Vec<u16> = cands.iter().map(|b| b.class).collect();
        classes.sort_unstable();
        classes.dedup();
        // A strong or decided box of class `k` among `boxes` (with `keep`, the kept ones only).
        let strong_of = |k: u16, keep: Option<&[bool]>| {
            cands.iter().enumerate().any(|(i, b)| {
                b.class == k
                    && (b.has(STRONG) || b.has(TRAINING))
                    && keep.is_none_or(|kept| kept.get(i).copied().unwrap_or(false))
            })
        };
        let weigh = |k: u16, strong: bool| {
            let ki = usize::from(k);
            let v_k = cam.v.get(ki).and_then(|v| v.get(s)).copied().unwrap_or(1.0);
            let new_day = !cam
                .cells
                .and_then(|c| c.days.get(&(k, s8)))
                .is_some_and(|days| days.contains(f.day));
            let d_k = if new_day { 1.0 + W_DAY } else { 1.0 };
            let floor = self.floor_gain(k, strong);
            let f_k = if floor { RARE_GAIN } else { 1.0 };
            let r_k = cam.r.get(ki).and_then(|r| r.get(s)).copied().unwrap_or(1.0);
            let catch_up = cam.catch_up.get(ki).copied().unwrap_or(false);
            let u_k = if catch_up { CATCHUP_GAIN } else { 1.0 };
            let heldout_gain = self.heldout_gain(&cam, f, k);
            let g_k = if heldout_gain { HELDOUT_GAIN } else { 1.0 };
            ClassWeight {
                k,
                v: v_k,
                w: v_k * d_k * f_k * r_k * u_k * g_k,
                new_day,
                floor,
                catch_up,
                heldout_gain,
            }
        };
        // Objects are found with every candidate's strength; the floor then follows the boxes
        // that survive them.
        let first: Vec<ClassWeight> = classes
            .iter()
            .map(|&k| weigh(k, strong_of(k, None)))
            .collect();
        let weight_of = |k: u16| first.iter().find(|w| w.k == k).map_or(0.0, |w| w.w);
        let vals: Vec<BoxVal> = cands.iter().map(|b| self.box_val(&cam, b, f)).collect();
        // One class: nothing to join (boxes of one class never do), every box kept.
        let keep = if classes.len() < 2 {
            vec![true; cands.len()]
        } else {
            objects(&cands, &vals, &weight_of, &|a, b| self.confusable(a, b))
        };
        let mut weights = first;
        if keep.contains(&false) {
            for w in &mut weights {
                let kept = strong_of(w.k, Some(&keep));
                if kept != strong_of(w.k, None) {
                    *w = weigh(w.k, kept);
                }
            }
        }
        let mut v = 0.0;
        let mut hard = 0.0;
        let (mut hard_best, mut hard_class) = (0.0, None);
        let mut best: Option<(f64, Detail)> = None;
        for cw in &weights {
            let k = cw.k;
            let of_k = || {
                cands
                    .iter()
                    .zip(&vals)
                    .zip(&keep)
                    .filter(move |((b, _), kept)| b.class == k && **kept)
                    .map(|((b, x), _)| (*b, x))
            };
            let (mut sum, mut h_k) = (0.0, 0.0);
            for (_, x) in of_k() {
                sum += x.a;
                h_k += x.h;
            }
            let e_k = crowd(sum);
            let term = e_k * cw.w;
            v += term;
            let h_k = h_k * cw.v;
            hard += h_k;
            if h_k > hard_best {
                (hard_best, hard_class) = (h_k, Some(k));
            }
            if best.as_ref().is_none_or(|(t, _)| term > *t) {
                best = Some((term, self.detail_of(&cam, cw, of_k(), e_k, s8)));
            }
        }
        let hard = if f.heldout {
            0.0
        } else {
            W_HARDNEG * crowd(hard)
        };
        let mut d = best.map(|(_, d)| d).unwrap_or_default();
        d.hard = hard;
        d.hard_class = hard_class.filter(|_| hard > 0.0);
        // No recall need, only likely false boxes (faint ones, or of a κ of 0): the frame is worth
        // its hard-negative share alone, no class named.
        if v == 0.0 && hard > 0.0 {
            d = Detail {
                hard,
                hard_class: d.hard_class,
                hard_only: true,
                ..Detail::default()
            };
        }
        d.score = score_of(v + hard);
        d.heldout_day = f.heldout;
        d
    }

    /// The page's explanation of the top class `cw.k` of a frame of `cam` in scene `s8`: its
    /// candidates `of_k` (the kept ones, with their values), whose crowd is `e_k`, give the cues
    /// and the evidence; the rest is the class's counts and why it weighs what it does. Reads
    /// only: the score's sums are [`Self::score`]'s.
    fn detail_of<'b>(
        &self,
        cam: &Cam<'_>,
        cw: &ClassWeight,
        of_k: impl Iterator<Item = (&'b CellBox, &'b BoxVal)>,
        e_k: f64,
        s8: u8,
    ) -> Detail {
        let (k, s) = (cw.k, usize::from(s8));
        let mut d = Detail::default();
        for (b, x) in of_k {
            d.evidence = d.evidence.max(b.e());
            if x.kappa.is_some_and(|kp| kp < 1.0) {
                d.kept = cam.kept.get(usize::from(k)).copied();
            }
            if x.a >= CUE_SHARE * e_k {
                d.spot |= x.spot;
                d.missed |= x.missed;
                d.gap |= b.has(GAP);
                d.strong |= x.strong;
                d.new_region |= x.new_region;
            }
        }
        let ki = usize::from(k);
        d.boxes = e_k;
        d.camera_reviewed = quarters(cam.cells.map(|c| &c.cells), (k, s8));
        d.reviewed = self
            .n_ks
            .get(ki)
            .and_then(|n| n.get(s))
            .copied()
            .unwrap_or(0.0);
        d.class_reviewed = self.boxes.get(ki).copied().unwrap_or(0.0);
        d.below_floor = cw.floor;
        d.catch_up = cw.catch_up;
        d.new_day = cw.new_day;
        d.heldout_gain = cw.heldout_gain;
        d.heldout_boxes = self
            .need
            .steers_heldout()
            .then(|| cam.heldout.get(ki).and_then(|h| h.get(s)).copied())
            .flatten();
        d.top = Some(k);
        d
    }

    /// Classes `a` and `b` can be one object proposed twice: an animal for another or a vehicle
    /// for another, never a person or any other class.
    fn confusable(&self, a: u16, b: u16) -> bool {
        let of = |k: u16| self.need.category.get(usize::from(k)).copied();
        match (of(a), of(b)) {
            (Some(x), Some(y)) => x == y && matches!(x, Category::Vehicle | Category::Animal),
            _ => false,
        }
    }

    /// What one candidate `b` of a frame at `f` adds (see the module docs).
    /// κ of box `b` on `cam`: None for a decided box, else the camera's kept share of the class
    /// for boxes of its strength.
    fn kappa(&self, cam: &Cam<'_>, b: &CellBox) -> Option<f64> {
        let ki = usize::from(b.class);
        if b.has(TRAINING) {
            None
        } else if b.has(STRONG) {
            Some(cam.kappa_strong.get(ki).copied().unwrap_or(1.0))
        } else {
            Some(cam.kappa_weak.get(ki).copied().unwrap_or(1.0))
        }
    }

    /// Box `b` is likely false on `cam`: pending, no gap, and a VLM no or κ under
    /// [`LIKELY_FALSE_KAPPA`].
    fn likely_false(&self, cam: &Cam<'_>, b: &CellBox) -> bool {
        let kp = self.kappa(cam, b).unwrap_or(1.0);
        !b.has(TRAINING) && !b.has(GAP) && (kp < LIKELY_FALSE_KAPPA || b.has(VLM_NO))
    }

    fn box_val(&self, cam: &Cam<'_>, b: &CellBox, f: &Facts<'_>) -> BoxVal {
        let ki = usize::from(b.class);
        let decided = b.has(TRAINING);
        let strong = decided || b.has(STRONG);
        let spot = self.need.at_spot(cam.spots, b, f.width, f.height);
        let g = usize::from(region(b, f.width, f.height));
        let n_region = cam
            .regions
            .get(ki)
            .and_then(|r| r.get(g))
            .copied()
            .unwrap_or(0.0);
        let kappa = self.kappa(cam, b);
        let kp = kappa.unwrap_or(1.0);
        // Never on a held-out day: a hard case for the held-out set.
        let missed = !f.heldout
            && (b.has(MISSED) || b.has(GAP))
            && b.e() >= MISSED_MIN_E
            && kp >= MISSED_MIN_KAPPA;
        let rf = (1.0 + W_REGION * T_R / (T_R + n_region)) / 2.0;
        // A faint box adds no recall need, only its hard-negative share.
        let mut a = if faint(b) { 0.0 } else { b.e() };
        if spot {
            a *= W_SPOT;
        }
        if missed {
            a *= W_MISSED;
        }
        a *= rf;
        a *= kp;
        let vlm_no = b.has(VLM_NO);
        let likely_false = self.likely_false(cam, b);
        let falseness = if likely_false {
            (1.0 - kp).max(if vlm_no { VLM_NO_FALSE } else { 0.0 })
        } else {
            0.0
        };
        let h = falseness * cam.hard.get(ki).copied().unwrap_or(1.0) * rf;
        BoxVal {
            a,
            h,
            kappa,
            spot,
            missed,
            strong,
            new_region: n_region == 0.0,
        }
    }

    /// Whether `g_k` is [`HELDOUT_GAIN`] for class `k` of frame `f` on `cam` (see the module
    /// docs): with held-out steering on, its day is held out and the cell has fewer than
    /// [`HELDOUT_FLOOR`] held-out boxes, or a training day and the cell has the floor.
    fn heldout_gain(&self, cam: &Cam<'_>, f: &Facts<'_>, k: u16) -> bool {
        if !self.need.steers_heldout() {
            return false;
        }
        let h = cam
            .heldout
            .get(usize::from(k))
            .and_then(|h| h.get(usize::from(scene_ix(f.scene))))
            .copied()
            .unwrap_or(0.0);
        let short = h < HELDOUT_FLOOR as f64;
        f.heldout == short
    }

    /// The page's explanation of a frame of `camera` in `scene` ([`Self::score`]'s `d`), its visit
    /// expanded by class `visit` or not. Shares are rounded to [`four`] decimals: the page shows
    /// no more, and a short decimal reads back as the same float wherever the JSON is parsed.
    pub fn explain(&self, d: &Detail, camera: &str, scene: &str, visit: Option<u16>) -> RecallNeed {
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
            expanded: visit.is_some(),
            visit_class: visit.map(|k| self.need.name(k).to_string()),
            hard_negative: four(d.hard),
            hard_class: d.hard_class.map(|k| self.need.name(k).to_string()),
            hard_only: d.hard_only,
            empty_reviewed: d.empty.map(|e| e.0),
            animal_share: d.empty.map(|e| four(e.1)).filter(|s| *s > 0.0),
            heldout_day: d.heldout_day,
            heldout_boxes: d.heldout_boxes,
            heldout_floor: d.heldout_boxes.map(|_| HELDOUT_FLOOR),
            heldout_gain: d.heldout_gain,
        }
    }
}

/// One present class's weight in [`Scorer::score`].
#[derive(Debug, Clone, Copy)]
struct ClassWeight {
    k: u16,
    v: f64,
    w: f64,
    new_day: bool,
    floor: bool,
    catch_up: bool,
    heldout_gain: bool,
}

/// What one candidate adds ([`Scorer::box_val`]).
#[derive(Debug, Clone, Copy)]
struct BoxVal {
    /// a_b.
    a: f64,
    /// Its hard-negative part, before its class's v_k.
    h: f64,
    /// The kept share that weighed it (None: decided).
    kappa: Option<f64>,
    spot: bool,
    missed: bool,
    strong: bool,
    new_region: bool,
}

/// The root of `i` in the union-find `p`.
fn root(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while let Some(&up) = p.get(r) {
        if up == r {
            break;
        }
        r = up;
    }
    // Path compression.
    let mut j = i;
    while let Some(&up) = p.get(j) {
        if up == j {
            break;
        }
        if let Some(x) = p.get_mut(j) {
            *x = r;
        }
        j = up;
    }
    r
}

/// Which of `cands` count (see the module docs, "objects"): candidates of other classes that
/// `confusable` allows (two animals, two vehicles), of areas within [`OBJECT_AREA`] of each other
/// and at IoU >= [`OBJECT_IOU`] or [`OBJECT_INSIDE`] inside one another, form one object, and only
/// its best class keeps its boxes: a decided box's class when it holds one, else the largest
/// `Σ a_b · w_k` (ties to the first class). Boxes of one class never join directly, and two
/// objects that each hold a decided box never join. The one rule for both paths.
fn objects(
    cands: &[&CellBox],
    vals: &[BoxVal],
    weight: &dyn Fn(u16) -> f64,
    confusable: &dyn Fn(u16, u16) -> bool,
) -> Vec<bool> {
    let n = cands.len();
    let mut parent: Vec<usize> = (0..n).collect();
    // Per root: the class of a decided box in it.
    let mut decided: Vec<Option<u16>> = cands
        .iter()
        .map(|b| b.has(TRAINING).then_some(b.class))
        .collect();
    for (i, a) in cands.iter().enumerate() {
        for (j, b) in cands.iter().enumerate().skip(i + 1) {
            if a.class == b.class || !confusable(a.class, b.class) {
                continue;
            }
            let (ra, rb) = (a.rect(), b.rect());
            let (aa, ab) = (area(&ra), area(&rb));
            let similar = aa > 0.0 && ab > 0.0 && {
                let r = aa / ab;
                (1.0 / OBJECT_AREA..=OBJECT_AREA).contains(&r)
            };
            if !similar || !(iou(&ra, &rb) >= OBJECT_IOU || inside_share(&ra, &rb) >= OBJECT_INSIDE)
            {
                continue;
            }
            let (x, y) = (root(&mut parent, i), root(&mut parent, j));
            let (dx, dy) = (
                decided.get(x).copied().flatten(),
                decided.get(y).copied().flatten(),
            );
            if x == y || (dx.is_some() && dy.is_some()) {
                continue;
            }
            let (lo, hi) = (x.min(y), x.max(y));
            if let Some(p) = parent.get_mut(hi) {
                *p = lo;
            }
            if let Some(d) = decided.get_mut(lo) {
                *d = dx.or(dy);
            }
        }
    }
    let roots: Vec<usize> = (0..n).map(|i| root(&mut parent, i)).collect();
    let mut sums: BTreeMap<(usize, u16), f64> = BTreeMap::new();
    for ((b, x), r) in cands.iter().zip(vals).zip(&roots) {
        *sums.entry((*r, b.class)).or_default() += x.a;
    }
    // Per object, its best class (a decided one, else in class order the first of a tie).
    let mut best: BTreeMap<usize, (u16, f64)> = BTreeMap::new();
    for ((r, k), sum) in sums {
        if let Some(d) = decided.get(r).copied().flatten() {
            best.insert(r, (d, f64::INFINITY));
            continue;
        }
        let v = sum * weight(k);
        let e = best.entry(r).or_insert((k, v));
        if v > e.1 {
            *e = (k, v);
        }
    }
    cands
        .iter()
        .zip(&roots)
        .map(|(b, r)| best.get(r).is_none_or(|(k, _)| *k == b.class))
        .collect()
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

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// Why the Recommended order places a frame where it does (item field `need`): its score, the
/// score after the visit divisor and, for its top class, what the score came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RecallNeed {
    /// `round(1000 · (V + H))`.
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
    /// Whole reviewed boxes of the class, every camera and scene, and the floor.
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
    /// The camera model missed a box of it (cues come only of boxes carrying at least
    /// [`CUE_SHARE`] of the class's weight).
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
    /// The camera's reviewed proposals of the class, when their kept share lowered a box of it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kept: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed: Option<u64>,
    /// The visit expands, and the class that expanded it (decided in the visit, rare or catch-up).
    #[serde(skip_serializing_if = "is_false")]
    pub expanded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visit_class: Option<String>,
    /// The hard-negative share (`H`) and the class of its largest part: a likely false box that
    /// teaches the model once rejected.
    #[serde(skip_serializing_if = "is_zero")]
    pub hard_negative: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hard_class: Option<String>,
    /// The frame adds no recall need, only likely false boxes (no class named): it is here for its
    /// hard-negative share alone.
    #[serde(skip_serializing_if = "is_false")]
    pub hard_only: bool,
    /// An empty frame: the scene's reviewed empty frames, and the camera's share of reviewed
    /// frames with an animal there (when any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty_reviewed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub animal_share: Option<f64>,
    /// Held-out steering: the frame's day is held out (listed in `heldout-days.json`, or by the
    /// export's hash split), ...
    #[serde(skip_serializing_if = "is_false")]
    pub heldout_day: bool,
    /// ... the class's held-out boxes on the camera in the scene (a box at a parked spot counts a
    /// quarter) and the floor they are wanted up to (absent without held-out steering or a class),
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heldout_boxes: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heldout_floor: Option<u64>,
    /// ... and whether the split the cell needs raised the class (`g_k`): a held-out day of a cell
    /// short of its held-out boxes, or a training day of one that has them.
    #[serde(skip_serializing_if = "is_false")]
    pub heldout_gain: bool,
}

#[cfg(test)]
#[path = "need/tests.rs"]
mod tests;
