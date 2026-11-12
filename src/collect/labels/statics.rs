//! Static spots of fixed cameras: the places where the same object (a parked car, a boat on its
//! trailer) is labelled frame after frame, and stays. Pure. A spot is a cluster of training boxes
//! (accepted or added, no ignore region) of one object ([`object_of`]: the vehicle classes are one
//! object, like [`super::repeats::same_object`]) at one place of one camera, in coordinates
//! normalised to the frame (frames of a camera can differ in size), on at least [`MIN_FRAMES`]
//! distinct frames of at least [`MIN_DAYS`] days on which it stayed ([`MIN_DAY_SHARE`];
//! [`Table::build`]).
//!
//! **Only things that park** ([`may_be_static`]): a vehicle, or another known inanimate COCO class
//! (a bench, a potted plant). People and animals ([`is_animate`]) never form a spot: they are never
//! parked, their frames are the scarce data a dataset needs most, and the dataset check's lower
//! cut-off was measured on vehicles. A class this build cannot place (a custom name outside the
//! person, vehicle, animal and COCO lists, such as `groundhog`, or a merged `vehicle`) forms no spot
//! either: a missed cap or a missed low-score `add` costs little, a capped-away animal frame costs
//! the class.
//!
//! **Only things that stay** (the dwell rule): a day counts toward [`MIN_DAYS`] only when the spot
//! is labelled on at least [`MIN_DAY_SHARE`] of that camera's frames of that day that have any
//! training box (of any class: what the reviewer labelled that day; empty frames are left out,
//! their number follows the collector's rate limits rather than the place). Visitors' cars
//! stopping at a gate now and then, among the day's other traffic, are on a small share of its
//! frames; a parked car is on most of them.
//!
//! Two readers: the training plan caps the frames that show nothing but such objects
//! (`training::dataset`, `static_cap`), and the dataset check takes a detection at a spot as an
//! `add` from a lower score than elsewhere (`training::refine::judge`). The Review side lists a
//! folder's spots with [`overview`]. Each builds its table from the frames it has (the plan: its
//! train / valid pool after the scene filter; the check: its selection; the card: every reviewed
//! frame of the folder), so their spots can differ; a suggestion is matched to the card's spots by
//! geometry ([`Table::spot_of_add`]), never by the id the check stored.
//!
//! **Finding the spots** (per camera and object, [`cluster`]), on a bounded sample: a camera with
//! more than [`MAX_OBS_PER_CAMERA`] boxes of classes that may be static is clustered on the boxes of
//! a fixed sample of its frames (the frames with the lowest [`fnv1a`] hash of their key until the
//! bound; the dwell rule then uses the sample's own frames per day). The observations are put in a
//! canonical order first (box coordinates, then frame, day, scene, label), so the result depends on
//! the set of observations only, never on the order a walk found them in. Then, repeatedly: the
//! unassigned observation with the most support (distinct frames with an unassigned box of its
//! object at IoU >= the spot IoU with it; ties to the first in the canonical order) seeds a
//! cluster; its members are, per frame, the unassigned box closest to the representative at an
//! IoU of the spot IoU or more; the representative becomes the coordinate-wise (lower) median of
//! the members, and members and representative are taken again until they do not change (at most
//! [`MAX_ROUNDS`] times; the last is kept, so it ends however the data look). A cluster that meets
//! the rule (frames, days it stayed) is a candidate and its members leave the pool; one that does
//! not leaves its members in the pool but none of them seeds again. It stops when no observation
//! left has enough support. Every member is within the spot IoU of the one representative box, so
//! two neighbouring parking places do not chain into one the way single linkage would: a box
//! between them joins the one it overlaps more, and a place counts as a spot only on its own.
//!
//! **Counting the spots** ([`settle`]), on every box: each box of a class that may be static lies at
//! the candidate of its object with the best IoU at the spot IoU or more (ties to the first id,
//! [`Table::spot_at`]); a spot's `frames`, `days`, `by_day`, `scenes` and label are those boxes'
//! (one per frame, the closest), so they are exact whether the camera was sampled or not, and they
//! are what [`Table::spot_at`] finds later. A candidate that no longer meets the rule on these
//! counts (a neighbour took its boxes) is dropped and the rest are counted again (dropping one only
//! gives the others boxes, so this ends after one more pass in practice, and after at most one pass
//! per candidate in any case).
//!
//! **Cost**: no neighbour list is ever stored. Support is counted on demand by a sweep over the
//! boxes sorted by `x0` within the window an IoU of the spot IoU allows (`x0` within
//! `[x0 - (1-t)/t w, x0 + (1-t) w]` of a box of width `w`), and kept as an upper bound that is only
//! recounted when it reaches the top of a max-heap (lazy greedy: supports only fall as boxes are
//! assigned, so the seed is the exact one of the eager rule). With `S` <= [`MAX_OBS_PER_CAMERA`]
//! boxes clustered, a camera's clustering takes O(S x W) IoUs (W the boxes in a window, W <= S, so
//! O(S^2) for one parked car in every frame) and O(S) memory, whatever the camera holds; counting
//! takes O(N x K) IoUs (N boxes, K candidates of their object) and O(N) memory. One car in 20,000
//! frames: at most 16 million IoUs to cluster its sample, 20,000 to count, where storing every
//! pair took 200 million IoUs and 4 GB.
//!
//! **Ids** ([`Spot::id`]): `<camera>/<object>/<x0>-<y0>-<x1>-<y1>`, the representative box in
//! steps of 1/[`ID_STEPS`] of the frame (2%). A rerun on the same labels gives the same ids; a rerun
//! after more labels keeps an id unless the median moves across a step (a pixel of jitter does so
//! for a coordinate that sits on a step's edge: an id is a convenience for grouping, never a key
//! that must survive). Two spots of one camera and object whose boxes round alike get `~2`, `~3`
//! in the canonical order.

use super::bulk::Wanted;
use super::export::ExportFrame;
use super::iou;
use super::refine::{Kind, Suggestion};
use super::repeats::VEHICLES;
use crate::model::coco::coco_class;
use crate::training::autolabel::Category;
use serde::Serialize;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// IoU (normalised boxes) from which a box lies at a spot: with the spot's representative box,
/// and between two boxes of a cluster's seed. Measured on 851 reviewed frames (docs/TRAINING.md,
/// "Static spots").
pub const SPOT_IOU: f32 = 0.6;
/// Fewest distinct frames with a box at a place that make it a spot.
pub const MIN_FRAMES: usize = 10;
/// Fewest days it stayed ([`MIN_DAY_SHARE`]): one afternoon of a car parked is no static object.
pub const MIN_DAYS: usize = 2;
/// Share of a camera's frames of a day with any training box on which a spot must be labelled for
/// that day to count toward [`MIN_DAYS`] (the dwell rule, see the module docs). Measured on the 851
/// reviewed frames of one installation (docs/TRAINING.md, "Static spots").
pub const MIN_DAY_SHARE: f32 = 0.3;
/// Most boxes of classes that may be static clustered per camera (a sample of its frames, see the
/// module docs). A spot needs [`MIN_FRAMES`] frames on [`MIN_DAYS`] days, so a place labelled on
/// 1 in 400 of a camera's boxes still shows up in a sample of 4,000; clustering it costs at most
/// about 16 million IoUs whatever the camera holds (the largest camera of the measured set has
/// 1,746 such boxes, so it is clustered whole).
pub const MAX_OBS_PER_CAMERA: usize = 4_000;
/// Most times a cluster's members and representative are taken again.
pub const MAX_ROUNDS: usize = 20;
/// Steps across the frame of the box in a spot's id.
pub const ID_STEPS: f32 = 50.0;
/// The object key of every vehicle class ([`VEHICLES`]).
pub const VEHICLE: &str = "vehicle";

/// The object a label names: `vehicle` for every vehicle class, else the label (trimmed, lower
/// case). Two labels of one object ([`super::repeats::same_object`]) have the same key.
pub fn object_of(label: &str) -> String {
    let l = label.trim().to_ascii_lowercase();
    if VEHICLES.contains(&l.as_str()) {
        VEHICLE.to_string()
    } else {
        l
    }
}

/// A person or an animal: `person` and every animal class of the auto-labeler's list
/// ([`Category::of`]: the 33 animal names its VLM check and SigLIP use, a superset of the
/// benchmark's COCO `animal` group). The one place this module decides what is animate.
pub fn is_animate(label: &str) -> bool {
    matches!(Category::of(label), Category::Person | Category::Animal)
}

/// Whether boxes of `label` may form a static spot (see the module docs): a vehicle class
/// ([`Category::Vehicle`]), or a COCO class (aliases applied, [`coco_class`]) that is neither a
/// person nor an animal. Everything else, animate or unknown, never does.
pub fn may_be_static(label: &str) -> bool {
    match Category::of(label) {
        Category::Vehicle => true,
        Category::Person | Category::Animal => false,
        Category::Object => coco_class(label.trim()).is_some_and(|c| !is_animate(c)),
    }
}

/// One training box of a camera's frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Obs {
    pub camera: String,
    /// What identifies the frame (a path); two boxes of one frame have the same key.
    pub frame: String,
    /// `YYYY-MM-DD` (or any day key).
    pub day: String,
    /// `day`, `night`, `ir` or `unknown`.
    pub scene: String,
    /// The class (the list's spelling).
    pub label: String,
    /// `[x0, y0, x1, y1]` normalised to the frame (0..1).
    pub bbox: [f32; 4],
}

/// A static spot (see the module docs). The counts are over every box the table was built from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Spot {
    /// `<camera>/<object>/<x0>-<y0>-<x1>-<y1>` (see the module docs).
    pub id: String,
    pub camera: String,
    /// [`object_of`] its labels.
    pub object: String,
    /// The most common label of its boxes (one per frame; ties: the first by name).
    pub label: String,
    /// The representative box: the coordinate-wise median of the cluster, normalised.
    pub bbox: [f32; 4],
    /// Distinct frames with a box at the spot ([`Table::spot_at`]).
    pub frames: usize,
    /// Distinct days.
    pub days: usize,
    /// Of `days`, those on which it stayed ([`MIN_DAY_SHARE`]): the days [`MIN_DAYS`] counts.
    pub stay_days: usize,
    /// Frames per scene.
    pub scenes: BTreeMap<String, usize>,
    /// Frames per day.
    pub by_day: BTreeMap<String, usize>,
    /// Per day of `by_day`: the camera's frames with any training box that day (the dwell rule's
    /// denominator).
    #[serde(skip)]
    pub day_frames: BTreeMap<String, usize>,
    /// The days of `stay_days`.
    #[serde(skip)]
    stays: BTreeSet<String>,
}

impl Spot {
    /// Whether a box of `label` at `bbox` (normalised) lies at this spot at IoU `min_iou`: its
    /// IoU with the representative box, None when it does not lie there, is of another object, or
    /// is of a class that is never static ([`may_be_static`]).
    pub fn iou_with(&self, label: &str, bbox: &[f32; 4], min_iou: f32) -> Option<f32> {
        if !may_be_static(label) || object_of(label) != self.object || !proper(bbox) {
            return None;
        }
        let v = iou(&self.bbox, bbox);
        (v >= min_iou).then_some(v)
    }

    /// Frames of the spot not of `day`.
    pub fn frames_without(&self, day: &str) -> usize {
        self.frames - self.by_day.get(day).copied().unwrap_or(0)
    }

    /// Days of the spot other than `day`.
    pub fn days_without(&self, day: &str) -> usize {
        self.days - usize::from(self.by_day.contains_key(day))
    }

    /// Whether the object stayed on `day` ([`MIN_DAY_SHARE`]).
    pub fn stayed_on(&self, day: &str) -> bool {
        self.stays.contains(day)
    }

    /// Days it stayed other than `day`.
    pub fn stay_days_without(&self, day: &str) -> usize {
        self.stay_days - usize::from(self.stays.contains(day))
    }

    /// The share of the camera's frames with a training box on which the spot is labelled, per day.
    pub fn day_shares(&self) -> BTreeMap<&str, f64> {
        self.by_day
            .iter()
            .map(|(d, n)| {
                let total = self.day_frames.get(d).copied().unwrap_or(0).max(*n).max(1);
                (d.as_str(), *n as f64 / total as f64)
            })
            .collect()
    }

    /// Whether the spot still is one without the frames of `day` ([`MIN_FRAMES`] frames, the
    /// object staying on [`MIN_DAYS`] days): the evidence a frame of `day` may use (one day's
    /// frames are near copies of each other, so a spot only that day makes is no evidence for a
    /// frame of it). The counts are the spot's minus that day's; the cluster is not made again
    /// without it.
    pub fn holds_without(&self, day: &str) -> bool {
        self.frames_without(day) >= MIN_FRAMES && self.stay_days_without(day) >= MIN_DAYS
    }
}

/// Finite coordinates of a box with a positive size.
fn proper(b: &[f32; 4]) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// FNV-1a, 64 bits: a fixed hash for the frame sample (std's hasher may change between releases).
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn cmp_box(a: &[f32; 4], b: &[f32; 4]) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

/// The canonical order of observations: the box, then frame, day, scene, label.
fn canonical(a: &Obs, b: &Obs) -> Ordering {
    cmp_box(&a.bbox, &b.bbox)
        .then_with(|| a.frame.cmp(&b.frame))
        .then_with(|| a.day.cmp(&b.day))
        .then_with(|| a.scene.cmp(&b.scene))
        .then_with(|| a.label.cmp(&b.label))
}

/// The lower median of `v` (sorted here); `v` is not empty.
fn lower_median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get((v.len().saturating_sub(1)) / 2)
        .copied()
        .unwrap_or(0.0)
}

/// Whether `count` frames of a day's `total` frames with a box reach `share` (exactly: a relative
/// slack of 1e-6 covers the rounding of `share` as an f32, so 3 of 10 reach 0.3).
fn reaches(count: usize, total: usize, share: f32) -> bool {
    count as f64 >= f64::from(share) * total as f64 * (1.0 - 1e-6)
}

/// The days of `per_day` (frames at a place per day) that reach `share` of `totals` (frames with
/// any box per day).
fn stay_days<'a>(
    per_day: impl IntoIterator<Item = (&'a str, usize)>,
    totals: &BTreeMap<&str, usize>,
    share: f32,
) -> BTreeSet<&'a str> {
    per_day
        .into_iter()
        .filter(|(d, n)| reaches(*n, totals.get(d).copied().unwrap_or(0).max(*n), share))
        .map(|(d, _)| d)
        .collect()
}

/// Distinct frames per day of `obs` (every box: any class).
fn day_totals<'a>(obs: impl IntoIterator<Item = &'a Obs>) -> BTreeMap<&'a str, usize> {
    let mut frames: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for o in obs {
        frames
            .entry(o.day.as_str())
            .or_default()
            .insert(o.frame.as_str());
    }
    frames.into_iter().map(|(d, f)| (d, f.len())).collect()
}

/// What a build computed, counted (a test bounds it).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Work {
    /// IoUs computed while finding the spots (on the sample).
    pub cluster_ious: u64,
    /// IoUs computed while counting every box against the candidates.
    pub count_ious: u64,
    /// Boxes clustered (the sample), over every camera and object.
    pub clustered: u64,
}

/// The index range of `x0s` (sorted) that can hold a box at IoU >= `t` with `b`: from `IoU >= t`
/// follows an intersection width of at least `t` times each width, so a box starting right of `b`
/// starts within `(1 - t) w` of it and one starting left within `(1 - t) / t w` (its width is at
/// most `w / t`). A margin of 1e-4 keeps every box whose computed IoU reaches `t` inside.
fn window(x0s: &[f32], b: &[f32; 4], t: f32) -> (usize, usize) {
    if t.is_nan() || t <= 0.0 {
        return (0, x0s.len());
    }
    const MARGIN: f32 = 1e-4;
    let w = b[2] - b[0];
    let lo_x = b[0] - (1.0 - t) / t * w - MARGIN;
    let hi_x = b[0] + (1.0 - t) * w + MARGIN;
    let lo = x0s.partition_point(|&x| x < lo_x);
    let hi = x0s.partition_point(|&x| x <= hi_x);
    (lo, hi.max(lo))
}

/// The representatives of the spots of one camera's boxes of one object (see the module docs);
/// `obs` in canonical order, `t` the spot IoU, `share` the dwell rule's, `totals` the frames with
/// any box per day of the same frames `obs` come from.
#[allow(
    clippy::indexing_slicing,
    reason = "every index is below n (obs, x0s, assigned, no_seed, support, frame_of) or a frame \
              index below the frame count (seen); window() returns a range within 0..=n"
)]
fn cluster(
    obs: &[&Obs],
    t: f32,
    share: f32,
    totals: &BTreeMap<&str, usize>,
    work: &mut Work,
) -> Vec<[f32; 4]> {
    let n = obs.len();
    work.clustered += n as u64;
    let mut frame_ids: BTreeMap<&str, usize> = BTreeMap::new();
    for o in obs {
        let next = frame_ids.len();
        frame_ids.entry(o.frame.as_str()).or_insert(next);
    }
    let frame_of: Vec<usize> = obs
        .iter()
        .map(|o| frame_ids.get(o.frame.as_str()).copied().unwrap_or(0))
        .collect();
    // The canonical order sorts by x0 first.
    let x0s: Vec<f32> = obs.iter().map(|o| o.bbox[0]).collect();
    let mut assigned = vec![false; n];
    let mut no_seed = vec![false; n];
    // Distinct frames marker, by generation (no clearing between counts).
    let mut seen = vec![0u32; frame_ids.len()];
    let mut stamp = 0u32;
    // Distinct frames with an unassigned box at IoU >= t with box i (its own frame included).
    let mut support_of = |i: usize, assigned: &[bool], work: &mut Work| -> usize {
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            seen.iter_mut().for_each(|s| *s = 0);
            stamp = 1;
        }
        let a = &obs[i].bbox;
        seen[frame_of[i]] = stamp;
        let mut k = 1;
        let (lo, hi) = window(&x0s, a, t);
        for j in lo..hi {
            if assigned[j] || seen[frame_of[j]] == stamp {
                continue;
            }
            work.cluster_ious += 1;
            if iou(a, &obs[j].bbox) >= t {
                seen[frame_of[j]] = stamp;
                k += 1;
            }
        }
        k
    };
    // The unassigned box per frame closest to `rep` at IoU >= t (ties: the first), sorted.
    let gather = |rep: &[f32; 4], assigned: &[bool], work: &mut Work| -> Vec<usize> {
        let mut best: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
        let (lo, hi) = window(&x0s, rep, t);
        for i in lo..hi {
            if assigned[i] {
                continue;
            }
            work.cluster_ious += 1;
            let v = iou(rep, &obs[i].bbox);
            if v < t {
                continue;
            }
            match best.get(&frame_of[i]) {
                Some((bv, _)) if *bv >= v => {}
                _ => {
                    best.insert(frame_of[i], (v, i));
                }
            }
        }
        let mut m: Vec<usize> = best.into_values().map(|(_, i)| i).collect();
        m.sort_unstable();
        m
    };
    let median = |members: &[usize]| -> [f32; 4] {
        let at = |k: usize| lower_median(members.iter().map(|&i| obs[i].bbox[k]).collect());
        [at(0), at(1), at(2), at(3)]
    };
    // Supports are upper bounds once boxes are assigned (they only fall): recounted when on top.
    let mut support: Vec<usize> = (0..n).map(|i| support_of(i, &assigned, work)).collect();
    let mut heap: BinaryHeap<(usize, Reverse<usize>)> = support
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, Reverse(i)))
        .collect();
    let mut out = Vec::new();
    while let Some((s, Reverse(seed))) = heap.pop() {
        if s < MIN_FRAMES {
            break;
        }
        if assigned[seed] || no_seed[seed] || support[seed] != s {
            continue;
        }
        let now = support_of(seed, &assigned, work);
        if now != s {
            support[seed] = now;
            heap.push((now, Reverse(seed)));
            continue;
        }
        let mut rep = obs[seed].bbox;
        let mut members = gather(&rep, &assigned, work);
        for _ in 0..MAX_ROUNDS {
            if members.is_empty() {
                break;
            }
            let next_rep = median(&members);
            let next = gather(&next_rep, &assigned, work);
            let same = next == members;
            rep = next_rep;
            members = next;
            if same {
                break;
            }
        }
        let mut per_day: BTreeMap<&str, usize> = BTreeMap::new();
        for &i in &members {
            *per_day.entry(obs[i].day.as_str()).or_default() += 1;
        }
        let stayed = stay_days(per_day, totals, share).len();
        if members.len() < MIN_FRAMES || stayed < MIN_DAYS || !proper(&rep) {
            no_seed[seed] = true;
            for &i in &members {
                no_seed[i] = true;
            }
            continue;
        }
        for &i in &members {
            assigned[i] = true;
        }
        out.push(rep);
    }
    out
}

/// `cands` (object, representative) of `camera` as spots with ids, counts empty: ids in a fixed
/// order, object then box.
fn named(camera: &str, mut cands: Vec<(String, [f32; 4])>) -> Vec<Spot> {
    cands.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| cmp_box(&a.1, &b.1)));
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    cands
        .into_iter()
        .map(|(object, bbox)| {
            let base = spot_id(camera, &object, &bbox);
            let n = used.entry(base.clone()).or_default();
            *n += 1;
            let id = if *n == 1 { base } else { format!("{base}~{n}") };
            Spot {
                id,
                camera: camera.to_string(),
                object,
                label: String::new(),
                bbox,
                frames: 0,
                days: 0,
                stay_days: 0,
                scenes: BTreeMap::new(),
                by_day: BTreeMap::new(),
                day_frames: BTreeMap::new(),
                stays: BTreeSet::new(),
            }
        })
        .collect()
}

/// Count every box of `statics` (one camera's boxes of classes that may be static, `objects` their
/// [`object_of`]) into `spots`: each box at the spot of its object with the best IoU at `t` or more
/// (ties to the first id, as [`spot_at_iou`]), one box per frame and spot (the closest; ties to the
/// first in the canonical order).
#[allow(
    clippy::indexing_slicing,
    reason = "k comes from by_object, built from the indices of spots; members has spots.len() \
              entries"
)]
fn count(
    spots: &mut [Spot],
    statics: &[&Obs],
    objects: &[String],
    totals: &BTreeMap<&str, usize>,
    t: f32,
    share: f32,
    work: &mut Work,
) {
    let mut by_object: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (k, s) in spots.iter().enumerate() {
        by_object.entry(s.object.as_str()).or_default().push(k);
    }
    let mut members: Vec<BTreeMap<&str, (f32, &Obs)>> = vec![BTreeMap::new(); spots.len()];
    for (o, object) in statics.iter().zip(objects) {
        let Some(idx) = by_object.get(object.as_str()) else {
            continue;
        };
        let mut best: Option<(f32, usize)> = None;
        for &k in idx {
            work.count_ious += 1;
            let v = iou(&spots[k].bbox, &o.bbox);
            if v < t {
                continue;
            }
            best = match best {
                Some((bv, bk)) if bv > v || (bv == v && spots[bk].id <= spots[k].id) => {
                    Some((bv, bk))
                }
                _ => Some((v, k)),
            };
        }
        let Some((v, k)) = best else { continue };
        let slot = members[k].entry(o.frame.as_str()).or_insert((v, o));
        if v > slot.0 || (v == slot.0 && canonical(o, slot.1) == Ordering::Less) {
            *slot = (v, o);
        }
    }
    for (s, m) in spots.iter_mut().zip(members) {
        let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
        let mut scenes: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_day: BTreeMap<&str, usize> = BTreeMap::new();
        for (_, o) in m.values() {
            *labels.entry(o.label.as_str()).or_default() += 1;
            *scenes.entry(o.scene.clone()).or_default() += 1;
            *by_day.entry(o.day.as_str()).or_default() += 1;
        }
        // Most common, ties to the first by name (the map's order; `max_by` keeps the last max).
        s.label = labels
            .iter()
            .rev()
            .max_by_key(|(_, k)| **k)
            .map(|(l, _)| (*l).to_string())
            .unwrap_or_default();
        s.frames = m.len();
        s.days = by_day.len();
        s.stays = stay_days(by_day.iter().map(|(d, n)| (*d, *n)), totals, share)
            .into_iter()
            .map(str::to_string)
            .collect();
        s.stay_days = s.stays.len();
        s.day_frames = by_day
            .keys()
            .map(|d| ((*d).to_string(), totals.get(d).copied().unwrap_or(0)))
            .collect();
        s.by_day = by_day
            .into_iter()
            .map(|(d, n)| (d.to_string(), n))
            .collect();
        s.scenes = scenes;
    }
}

/// The spots of one camera from the candidates `cands`, counted over every box of `statics` (see
/// the module docs): candidates that fail the rule on the counts are dropped and the rest counted
/// again.
fn settle(
    camera: &str,
    mut cands: Vec<(String, [f32; 4])>,
    statics: &[&Obs],
    totals: &BTreeMap<&str, usize>,
    t: f32,
    share: f32,
    work: &mut Work,
) -> Vec<Spot> {
    let objects: Vec<String> = statics.iter().map(|o| object_of(&o.label)).collect();
    loop {
        let mut spots = named(camera, cands);
        count(&mut spots, statics, &objects, totals, t, share, work);
        let before = spots.len();
        spots.retain(|s| s.frames >= MIN_FRAMES && s.stay_days >= MIN_DAYS);
        if spots.len() == before {
            return spots;
        }
        cands = spots.into_iter().map(|s| (s.object, s.bbox)).collect();
    }
}

/// The frames of a camera's `list` clustered when it holds more than [`MAX_OBS_PER_CAMERA`] boxes
/// of classes that may be static: by [`fnv1a`] of their key (then the key), whole frames, until
/// the next would pass the bound. Frames without such a box come along (the dwell rule's
/// denominator of the sample).
fn sample_frames(list: &[Obs]) -> BTreeSet<&str> {
    let mut per_frame: BTreeMap<&str, usize> = BTreeMap::new();
    for o in list {
        let k = per_frame.entry(o.frame.as_str()).or_default();
        if proper(&o.bbox) && may_be_static(&o.label) {
            *k += 1;
        }
    }
    let mut frames: Vec<(&str, usize)> = per_frame.into_iter().collect();
    frames.sort_by(|a, b| fnv1a(a.0).cmp(&fnv1a(b.0)).then(a.0.cmp(b.0)));
    let mut keep = BTreeSet::new();
    let mut total = 0;
    for (f, k) in frames {
        if total + k > MAX_OBS_PER_CAMERA {
            break;
        }
        total += k;
        keep.insert(f);
    }
    keep
}

/// The spots of every camera in `obs`, by camera (see the module docs).
fn build(
    obs: impl IntoIterator<Item = Obs>,
    t: f32,
    share: f32,
    work: &mut Work,
) -> BTreeMap<String, Vec<Spot>> {
    let mut by_camera: BTreeMap<String, Vec<Obs>> = BTreeMap::new();
    for o in obs {
        if !o.camera.is_empty() {
            by_camera.entry(o.camera.clone()).or_default().push(o);
        }
    }
    let mut out = BTreeMap::new();
    for (camera, list) in &by_camera {
        // People, animals and classes this build cannot place are not clustered at all; they
        // count only in the frames per day.
        let statics: Vec<&Obs> = list
            .iter()
            .filter(|o| proper(&o.bbox) && may_be_static(&o.label))
            .collect();
        if statics.is_empty() {
            continue;
        }
        let totals = day_totals(list);
        let sample = (statics.len() > MAX_OBS_PER_CAMERA).then(|| sample_frames(list));
        let in_sample = |o: &Obs| sample.as_ref().is_none_or(|s| s.contains(o.frame.as_str()));
        let sample_totals = match &sample {
            Some(_) => day_totals(list.iter().filter(|o| in_sample(o))),
            None => totals.clone(),
        };
        let mut by_object: BTreeMap<String, Vec<&Obs>> = BTreeMap::new();
        for o in statics.iter().copied().filter(|o| in_sample(o)) {
            by_object.entry(object_of(&o.label)).or_default().push(o);
        }
        let mut cands: Vec<(String, [f32; 4])> = Vec::new();
        for (object, mut v) in by_object {
            v.sort_by(|a, b| canonical(a, b));
            for rep in cluster(&v, t, share, &sample_totals, work) {
                cands.push((object.clone(), rep));
            }
        }
        let spots = settle(camera, cands, &statics, &totals, t, share, work);
        if !spots.is_empty() {
            out.insert(camera.clone(), spots);
        }
    }
    out
}

/// `<camera>/<object>/<x0>-<y0>-<x1>-<y1>` in steps of 1/[`ID_STEPS`].
fn spot_id(camera: &str, object: &str, b: &[f32; 4]) -> String {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a coordinate within 0..=1 times ID_STEPS"
    )]
    let step = |v: f32| (v.clamp(0.0, 1.0) * ID_STEPS).round() as u32;
    format!(
        "{camera}/{object}/{}-{}-{}-{}",
        step(b[0]),
        step(b[1]),
        step(b[2]),
        step(b[3])
    )
}

/// A frame's camera and size: what places a suggestion's pixels at a spot.
#[derive(Debug, Clone, Copy)]
pub struct FrameAt<'a> {
    pub camera: &'a str,
    pub width: u32,
    pub height: u32,
}

/// The spots of a set of observations, per camera.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Table {
    /// The IoU spots were made and are matched at.
    pub iou: f32,
    /// The dwell rule's share ([`MIN_DAY_SHARE`]).
    pub min_day_share: f32,
    /// Spots per camera, by object then box.
    pub cameras: BTreeMap<String, Vec<Spot>>,
}

impl Table {
    /// The spots of `obs` at [`SPOT_IOU`] and [`MIN_DAY_SHARE`].
    pub fn build(obs: impl IntoIterator<Item = Obs>) -> Self {
        Self::build_with(obs, SPOT_IOU, MIN_DAY_SHARE)
    }

    /// The spots of `obs` at the spot IoU `t` (the measurement compares several).
    pub fn build_at(obs: impl IntoIterator<Item = Obs>, t: f32) -> Self {
        Self::build_with(obs, t, MIN_DAY_SHARE)
    }

    /// The spots of `obs` at the spot IoU `t` and the dwell share `share` (0 = every day counts).
    pub fn build_with(obs: impl IntoIterator<Item = Obs>, t: f32, share: f32) -> Self {
        Self::build_counted(obs, t, share, &mut Work::default())
    }

    /// [`Self::build_with`], counting what it computed into `work`.
    pub fn build_counted(
        obs: impl IntoIterator<Item = Obs>,
        t: f32,
        share: f32,
        work: &mut Work,
    ) -> Self {
        Self {
            iou: t,
            min_day_share: share,
            cameras: build(obs, t, share, work),
        }
    }

    /// The spots of `camera` (none: empty).
    pub fn camera(&self, camera: &str) -> &[Spot] {
        self.cameras.get(camera).map_or(&[], Vec::as_slice)
    }

    /// Every spot, by camera.
    pub fn spots(&self) -> impl Iterator<Item = &Spot> {
        self.cameras.values().flatten()
    }

    /// Spots in all.
    pub fn len(&self) -> usize {
        self.cameras.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    /// The spot with id `id`.
    pub fn get(&self, id: &str) -> Option<&Spot> {
        self.spots().find(|s| s.id == id)
    }

    /// The spot of `camera` a box of `label` at `bbox` (normalised) lies at ([`spot_at_iou`]
    /// at this table's IoU).
    pub fn spot_at(&self, camera: &str, label: &str, bbox: &[f32; 4]) -> Option<&Spot> {
        spot_at_iou(self.camera(camera), label, bbox, self.iou)
    }

    /// The spot of this table an `add` suggestion of a frame at `at` lies at, by geometry (its
    /// box in pixels, its label through the class list): [`Self::spot_at`]. The id the check
    /// stored ([`Suggestion::spot`]) plays no part: it names a spot of the check's own table, which
    /// can differ from this one (another set of frames, labels since). None for another kind, an
    /// unwanted label or a frame of unknown size.
    pub fn spot_of_add(
        &self,
        at: FrameAt<'_>,
        s: &Suggestion,
        wanted: &Wanted<'_>,
    ) -> Option<&Spot> {
        if s.kind != Kind::Add {
            return None;
        }
        let class = wanted.class_of(s.label.as_deref()?)?;
        let nb = normalised(&s.bbox?, at.width, at.height)?;
        self.spot_at(at.camera, class, &nb)
    }

    /// [`is_static_only`] of a frame of `camera` at this table's IoU.
    pub fn static_only<L: AsRef<str>>(
        &self,
        camera: &str,
        boxes: impl IntoIterator<Item = (L, [f32; 4])>,
    ) -> Option<Vec<&str>> {
        static_only_at(self.camera(camera), boxes, self.iou)
    }
}

/// The spot of `spots` (one camera's) a box of `label` at `bbox` (normalised) lies at: the same
/// object at IoU >= [`SPOT_IOU`] with its representative box; of several the best IoU, ties to
/// the first id.
pub fn spot_at<'a>(spots: &'a [Spot], label: &str, bbox: &[f32; 4]) -> Option<&'a Spot> {
    spot_at_iou(spots, label, bbox, SPOT_IOU)
}

/// [`spot_at`] at IoU `t`.
pub fn spot_at_iou<'a>(
    spots: &'a [Spot],
    label: &str,
    bbox: &[f32; 4],
    t: f32,
) -> Option<&'a Spot> {
    spots
        .iter()
        .filter_map(|s| s.iou_with(label, bbox, t).map(|v| (v, s)))
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.id.cmp(&a.1.id)))
        .map(|(_, s)| s)
}

/// The ids of the spots (sorted, each once) of a frame of one camera whose every training box
/// (`(label, normalised box)`) lies at a spot of `spots` ([`spot_at`]); None when one does not,
/// and for a frame without boxes (an empty frame is a negative, not a static one).
pub fn is_static_only<L: AsRef<str>>(
    boxes: impl IntoIterator<Item = (L, [f32; 4])>,
    spots: &[Spot],
) -> Option<Vec<&str>> {
    static_only_at(spots, boxes, SPOT_IOU)
}

fn static_only_at<L: AsRef<str>>(
    spots: &[Spot],
    boxes: impl IntoIterator<Item = (L, [f32; 4])>,
    t: f32,
) -> Option<Vec<&str>> {
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for (label, b) in boxes {
        ids.insert(spot_at_iou(spots, label.as_ref(), &b, t)?.id.as_str());
    }
    (!ids.is_empty()).then(|| ids.into_iter().collect())
}

/// `b` (pixels of a `width` x `height` frame) normalised; None for a frame of unknown size.
pub fn normalised(b: &[f32; 4], width: u32, height: u32) -> Option<[f32; 4]> {
    if width == 0 || height == 0 {
        return None;
    }
    let (w, h) = (width as f32, height as f32);
    Some([b[0] / w, b[1] / h, b[2] / w, b[3] / h])
}

/// The observations of reviewed frames (a check's selection, an export's): every training box
/// that is no ignore region, of a wanted class (`label` = the list's spelling), normalised; the
/// frame key is its `<scene>/<camera>/<date>/<stem>.jpg`. Frames of unknown size give none.
pub fn observations<'a>(
    frames: impl IntoIterator<Item = &'a ExportFrame>,
    wanted: &Wanted<'_>,
) -> Vec<Obs> {
    let mut out = Vec::new();
    for f in frames {
        let key = f.rel.jpg();
        for b in f.labels.training().filter(|b| !b.is_ignore()) {
            let (Some(class), Some(bbox)) = (
                wanted.class_of(&b.label),
                normalised(&b.bbox, f.width, f.height),
            ) else {
                continue;
            };
            out.push(Obs {
                camera: f.rel.camera.clone(),
                frame: key.clone(),
                day: f.rel.date.clone(),
                scene: f.rel.scene.clone(),
                label: class.to_string(),
                bbox,
            });
        }
    }
    out
}

/// A frame with a live `add` at a spot ([`SpotSummary::open_frames`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenFrame {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub rel: String,
    pub width: u32,
    pub height: u32,
}

/// One spot of a folder's [`overview`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpotSummary {
    #[serde(flatten)]
    pub spot: Spot,
    /// Reviewed frames with a training box at the spot ([`Table::spot_at`]; the same frames as
    /// the spot's `frames` when the table is made from these frames).
    pub labelled: usize,
    /// Reviewed frames with a live `add` suggestion at this spot ([`super::refine::live`], placed
    /// by [`Table::spot_of_add`]).
    pub open_adds: usize,
    /// Those frames, in the order given.
    pub open_frames: Vec<OpenFrame>,
}

/// [`overview`] with the table it made.
pub fn overview_table(frames: &[ExportFrame], wanted: &Wanted<'_>) -> (Table, Vec<SpotSummary>) {
    let table = Table::build(observations(frames, wanted));
    let mut out: BTreeMap<&str, SpotSummary> = table
        .spots()
        .map(|s| {
            (
                s.id.as_str(),
                SpotSummary {
                    spot: s.clone(),
                    labelled: 0,
                    open_adds: 0,
                    open_frames: Vec::new(),
                },
            )
        })
        .collect();
    for f in frames {
        let mut labelled: BTreeSet<&str> = BTreeSet::new();
        for b in f.labels.training().filter(|b| !b.is_ignore()) {
            let Some(class) = wanted.class_of(&b.label) else {
                continue;
            };
            if let Some(nb) = normalised(&b.bbox, f.width, f.height)
                && let Some(s) = table.spot_at(&f.rel.camera, class, &nb)
            {
                labelled.insert(s.id.as_str());
            }
        }
        for id in labelled {
            if let Some(x) = out.get_mut(id) {
                x.labelled += 1;
            }
        }
        let at = FrameAt {
            camera: &f.rel.camera,
            width: f.width,
            height: f.height,
        };
        let open: BTreeSet<&str> = super::refine::live(&f.labels, Some(wanted))
            .iter()
            .filter_map(|l| table.spot_of_add(at, l.suggestion, wanted))
            .map(|s| s.id.as_str())
            .collect();
        for id in open {
            if let Some(x) = out.get_mut(id) {
                x.open_adds += 1;
                x.open_frames.push(OpenFrame {
                    rel: f.rel.jpg(),
                    width: f.width,
                    height: f.height,
                });
            }
        }
    }
    let list = out.into_values().collect();
    (table, list)
}

/// The static spots of `frames` (reviewed frames, e.g. [`super::export::reviewed_frames_filtered`]
/// of the folder under review) with their counts, by camera then id: what a "Static objects" card
/// lists. The spots are made from these frames ([`observations`], [`Table::build`]); a live `add`
/// counts for the spot it lies at now ([`Table::spot_of_add`]), whatever spot id the check that
/// made it stored (that check's spots came from its own selection, maybe scene-filtered, and from
/// the labels as they were then).
pub fn overview(frames: &[ExportFrame], wanted: &Wanted<'_>) -> Vec<SpotSummary> {
    overview_table(frames, wanted).1
}

/// The indices (into [`super::LabelFile::refine`]) of the live `add` suggestions of `f` (a frame
/// at `at`) that lie at spot `id` of `table` ([`Table::spot_of_add`], with the class list): what a
/// bulk "accept the adds at this spot" hands to [`super::refine::accept`] one by one, highest
/// index first (an accept removes its suggestion, so a lower index stays valid).
pub fn live_adds_at(
    f: &super::LabelFile,
    table: &Table,
    at: FrameAt<'_>,
    id: &str,
    wanted: &Wanted<'_>,
) -> Vec<usize> {
    let mut v: Vec<usize> = super::refine::live(f, Some(wanted))
        .iter()
        .filter(|l| {
            table
                .spot_of_add(at, l.suggestion, wanted)
                .is_some_and(|s| s.id == id)
        })
        .map(|l| l.index)
        .collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// A car box on frame `i` of `camera` on day `d` (of 2026-10-0d), at `b` shifted by `j`.
    fn obs(camera: &str, i: usize, d: usize, label: &str, b: [f32; 4], j: f32) -> Obs {
        Obs {
            camera: camera.into(),
            frame: format!("{camera}/{d}/{i:04}"),
            day: format!("2026-10-{d:02}"),
            scene: if i.is_multiple_of(3) { "night" } else { "day" }.into(),
            label: label.into(),
            bbox: [b[0] + j, b[1] + j, b[2] + j, b[3] + j],
        }
    }

    const A: [f32; 4] = [0.10, 0.50, 0.30, 0.70];
    /// Beside [`A`]: IoU 0.
    const B: [f32; 4] = [0.32, 0.50, 0.52, 0.70];

    /// `n` frames of `camera` over `days` days with a car at `b`, jittered a little.
    fn parked(camera: &str, n: usize, days: usize, b: [f32; 4], from: usize) -> Vec<Obs> {
        (0..n)
            .map(|i| {
                let j = ((i * 7) % 5) as f32 * 0.002;
                obs(camera, from + i, i % days + 1, "car", b, j)
            })
            .collect()
    }

    #[test]
    fn a_place_on_enough_frames_and_days_is_a_spot() {
        let t = Table::build(parked("patio", 12, 3, A, 0));
        let s = t.camera("patio");
        assert_eq!(s.len(), 1, "{t:?}");
        let s = &s[0];
        assert_eq!((s.frames, s.days, s.label.as_str()), (12, 3, "car"));
        assert_eq!(s.object, VEHICLE);
        assert_eq!(s.scenes.values().sum::<usize>(), 12);
        assert!(iou(&s.bbox, &A) > 0.9, "{:?}", s.bbox);
        assert_eq!(s.id, "patio/vehicle/5-25-15-35");
        // Too few frames, or one day only: no spot.
        assert!(Table::build(parked("patio", 9, 3, A, 0)).is_empty());
        assert!(Table::build(parked("patio", 30, 1, A, 0)).is_empty());
        // Another camera is its own.
        let mut two = parked("patio", 12, 3, A, 0);
        two.extend(parked("drive", 5, 3, A, 100));
        let t = Table::build(two);
        assert_eq!(t.len(), 1);
        assert!(t.camera("drive").is_empty());
    }

    #[test]
    fn neighbouring_places_stay_two_spots() {
        // Two cars side by side on the same frames, and a chain of boxes from one to the other,
        // each at IoU > 0.9 with the next (single linkage would make it all one spot).
        let mut v = parked("cam", 20, 4, A, 0);
        v.extend(parked("cam", 20, 4, B, 0));
        for k in 0..=22 {
            let x = 0.10 + k as f32 * 0.01;
            v.push(obs("cam", 500 + k, 1, "car", [x, 0.50, x + 0.2, 0.70], 0.0));
        }
        let t = Table::build(v);
        let s = t.camera("cam");
        assert_eq!(s.len(), 2, "{t:?}");
        assert!(s.iter().all(|s| s.frames >= 20), "{t:?}");
        assert!(
            iou(&s[0].bbox, &A) > 0.9 && iou(&s[1].bbox, &B) > 0.9,
            "{t:?}"
        );
        // A box half way lies at neither.
        assert!(t.spot_at("cam", "car", &[0.21, 0.50, 0.41, 0.70]).is_none());
    }

    #[test]
    fn vehicles_are_one_object_and_the_label_is_the_majority() {
        let mut v = parked("cam", 12, 3, A, 0);
        for o in v.iter_mut().take(5) {
            o.label = "truck".into();
        }
        let t = Table::build(v.clone());
        assert_eq!(t.len(), 1);
        assert_eq!(t.camera("cam")[0].label, "car");
        // A tie goes to the first by name.
        for o in v.iter_mut().take(6) {
            o.label = "truck".into();
        }
        assert_eq!(Table::build(v).camera("cam")[0].label, "car");
        // A person at the place is another object: no spot of it, and it is not at the car's.
        let t = Table::build(parked("cam", 12, 3, A, 0));
        assert!(t.spot_at("cam", "person", &A).is_none());
        assert!(t.spot_at("cam", "truck", &A).is_some());
        assert!(t.spot_at("cam", "Car", &A).is_some(), "case-insensitive");
    }

    #[test]
    fn the_result_does_not_depend_on_the_order() {
        let mut v = parked("cam", 30, 5, A, 0);
        v.extend(parked("cam", 15, 3, B, 100));
        v.extend(parked("other", 11, 2, B, 300));
        let first = Table::build(v.clone());
        assert_eq!(first.len(), 3);
        // Reversed, and rotated: the same spots and ids.
        let mut r = v.clone();
        r.reverse();
        assert_eq!(Table::build(r), first);
        let mut rot = v;
        rot.rotate_left(17);
        assert_eq!(Table::build(rot), first);
    }

    #[test]
    fn static_only_needs_every_box_at_a_spot() {
        let t = Table::build(parked("cam", 12, 3, A, 0));
        let id = t.camera("cam")[0].id.clone();
        assert_eq!(t.static_only("cam", [("car", A)]), Some(vec![id.as_str()]));
        assert_eq!(
            is_static_only([("truck", A)], t.camera("cam")),
            Some(vec![id.as_str()])
        );
        assert_eq!(t.static_only("cam", [("car", A), ("person", B)]), None);
        assert_eq!(t.static_only("cam", Vec::<(&str, [f32; 4])>::new()), None);
        assert_eq!(t.static_only("elsewhere", [("car", A)]), None);
    }

    #[test]
    fn a_spot_without_a_day_is_counted_without_it() {
        let t = Table::build(parked("cam", 12, 2, A, 0));
        let s = &t.camera("cam")[0];
        assert_eq!(s.by_day.get("2026-10-01"), Some(&6));
        assert_eq!(s.frames_without("2026-10-01"), 6);
        assert_eq!(s.days_without("2026-10-01"), 1);
        assert!(!s.holds_without("2026-10-01"));
        assert_eq!(s.frames_without("2026-11-30"), 12);
        let t = Table::build(parked("cam", 33, 3, A, 0));
        assert!(t.camera("cam")[0].holds_without("2026-10-01"));
    }

    #[test]
    fn ids_round_to_steps_and_collisions_get_a_suffix() {
        assert_eq!(
            spot_id("c", "vehicle", &[0.1, 0.2, 0.3, 0.4]),
            "c/vehicle/5-10-15-20"
        );
        assert_eq!(
            spot_id("c", "dog", &[0.0, 0.0, 1.0, 1.2]),
            "c/dog/0-0-50-50"
        );
        // A little jitter keeps the id.
        let a = Table::build(parked("cam", 12, 3, A, 0));
        let b = Table::build(parked("cam", 13, 3, A, 50));
        assert_eq!(a.camera("cam")[0].id, b.camera("cam")[0].id);
        // Two bench places so close they round alike (they cannot both be spots at the spot IoU,
        // so a high test IoU makes them).
        let benches = |b: [f32; 4], from: usize| -> Vec<Obs> {
            (0..12)
                .map(|i| obs("cam", from + i, i % 3 + 1, "bench", b, 0.0))
                .collect()
        };
        let mut v = benches([0.40, 0.40, 0.60, 0.60], 0);
        v.extend(benches([0.405, 0.40, 0.605, 0.60], 100));
        let t = Table::build_at(v, 0.99);
        let ids: Vec<&str> = t.spots().map(|s| s.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "{t:?}");
        assert_eq!(ids[1], format!("{}~2", ids[0]));
    }

    #[test]
    fn a_large_camera_is_clustered_on_a_fixed_sample() {
        let v: Vec<Obs> = (0..MAX_OBS_PER_CAMERA + 50)
            .map(|i| obs("big", i, i % 4 + 1, "car", A, 0.0))
            .collect();
        let s = sample_frames(&v);
        assert_eq!(s.len(), MAX_OBS_PER_CAMERA);
        let mut r = v.clone();
        r.reverse();
        assert_eq!(
            sample_frames(&r),
            s,
            "the sample does not depend on the order"
        );
        // Frames without a box that may be static come along without using the bound.
        let mut with_people = v.clone();
        with_people.extend((0..30).map(|i| obs("big", 90_000 + i, 1, "person", B, 0.0)));
        let car_frames: BTreeSet<&str> = v.iter().map(|o| o.frame.as_str()).collect();
        let s2 = sample_frames(&with_people);
        let cars2: BTreeSet<&str> = s2
            .iter()
            .copied()
            .filter(|f| car_frames.contains(f))
            .collect();
        assert_eq!(cars2, s);
        // The spot is found on the sample and counted on every frame.
        let t = Table::build(v);
        let sp = &t.camera("big")[0];
        assert_eq!((sp.frames, sp.days), (MAX_OBS_PER_CAMERA + 50, 4));
    }

    /// One car jittered in every one of `n` frames over 20 days of camera `cam`, plus `noise`
    /// small boxes scattered over the frame on frames of their own.
    fn one_car_everywhere(n: usize, noise: usize) -> Vec<Obs> {
        let mut v: Vec<Obs> = (0..n)
            .map(|i| {
                let j = (i % 7) as f32 * 0.0015;
                obs("cam", i, i % 20 + 1, "car", A, j)
            })
            .collect();
        for k in 0..noise {
            // A fixed scatter (a multiplicative hash), 3% boxes away from the car.
            let h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let x = 0.35 + (h % 1000) as f32 / 1000.0 * 0.6;
            let y = ((h >> 20) % 1000) as f32 / 1000.0 * 0.9;
            v.push(obs(
                "cam",
                1_000_000 + k,
                k % 20 + 1,
                "car",
                [x, y, x + 0.03, y + 0.03],
                0.0,
            ));
        }
        v
    }

    #[test]
    fn clustering_is_bounded_by_the_sample_and_counting_is_linear() {
        // The case that took 21.6 s and 4.1 GB when every neighbour pair was stored: one car in
        // 20,000 frames (all mutual neighbours), plus noise. The work is counted, not timed.
        let n = 20_000;
        let mut work = Work::default();
        let t = Table::build_counted(
            one_car_everywhere(n, 2_000),
            SPOT_IOU,
            MIN_DAY_SHARE,
            &mut work,
        );
        let spots = t.camera("cam");
        let car = spots
            .iter()
            .find(|s| iou(&s.bbox, &A) > 0.9)
            .unwrap_or_else(|| panic!("{t:?}"));
        // Counted over every frame, not only the sample.
        assert_eq!((car.frames, car.days, car.stay_days), (n, 20, 20));
        assert_eq!(car.by_day.values().sum::<usize>(), n);
        let s = MAX_OBS_PER_CAMERA as u64;
        assert!(work.clustered <= s, "{work:?}");
        // At most the sweep over every pair of the sample, plus the rounds' gathers; and in any
        // case what a sample of a few thousand boxes costs (a bound of 20,000 boxes, as before,
        // would cost 200 million here).
        assert!(
            work.cluster_ious <= s * s + 2 * MAX_ROUNDS as u64 * s * 4,
            "{work:?}"
        );
        assert!(work.cluster_ious <= 20_000_000, "{work:?}");
        // Counting: each box against the few candidates of its object.
        let boxes = (n + 2_000) as u64;
        assert!(
            work.count_ious <= boxes * 2 * spots.len() as u64,
            "{work:?}"
        );
        // Twice the frames: the clustering costs the same order, the counting twice.
        let mut more = Work::default();
        let t2 = Table::build_counted(
            one_car_everywhere(2 * n, 4_000),
            SPOT_IOU,
            MIN_DAY_SHARE,
            &mut more,
        );
        let car2 = t2
            .camera("cam")
            .iter()
            .find(|s| iou(&s.bbox, &A) > 0.9)
            .unwrap();
        assert_eq!(car2.frames, 2 * n);
        assert!(more.clustered <= s, "{more:?}");
        assert!(
            more.cluster_ious <= s * s + 2 * MAX_ROUNDS as u64 * s * 4,
            "{more:?}"
        );
        // What storing every pair took: n^2 / 2 IoUs for the car alone.
        assert!(
            work.cluster_ious + work.count_ious < (n as u64 * n as u64) / 10,
            "{work:?}"
        );
    }

    /// The clustering as it was before the bound (every neighbour pair stored, supports kept
    /// exact eagerly), without the dwell rule: the reference the lazy seed must equal.
    fn eager(obs: &[&Obs], t: f32) -> Vec<[f32; 4]> {
        let n = obs.len();
        let mut frame_ids: BTreeMap<&str, usize> = BTreeMap::new();
        for o in obs {
            let next = frame_ids.len();
            frame_ids.entry(o.frame.as_str()).or_insert(next);
        }
        let frame_of: Vec<usize> = obs.iter().map(|o| frame_ids[o.frame.as_str()]).collect();
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if frame_of[i] != frame_of[j] && iou(&obs[i].bbox, &obs[j].bbox) >= t {
                    nbrs[i].push(j);
                    nbrs[j].push(i);
                }
            }
        }
        let mut assigned = vec![false; n];
        let mut no_seed = vec![false; n];
        let support_of = |i: usize, assigned: &[bool]| -> usize {
            let f: BTreeSet<usize> = std::iter::once(&i)
                .chain(&nbrs[i])
                .filter(|&&j| !assigned[j])
                .map(|&j| frame_of[j])
                .collect();
            f.len()
        };
        let gather = |rep: &[f32; 4], assigned: &[bool]| -> Vec<usize> {
            let mut best: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
            for (i, o) in obs.iter().enumerate() {
                let v = iou(rep, &o.bbox);
                if assigned[i] || v < t {
                    continue;
                }
                match best.get(&frame_of[i]) {
                    Some((bv, _)) if *bv >= v => {}
                    _ => {
                        best.insert(frame_of[i], (v, i));
                    }
                }
            }
            let mut m: Vec<usize> = best.into_values().map(|(_, i)| i).collect();
            m.sort_unstable();
            m
        };
        let median = |m: &[usize]| -> [f32; 4] {
            let at = |k: usize| lower_median(m.iter().map(|&i| obs[i].bbox[k]).collect());
            [at(0), at(1), at(2), at(3)]
        };
        let mut out = Vec::new();
        loop {
            let support: Vec<usize> = (0..n).map(|i| support_of(i, &assigned)).collect();
            let Some(seed) = (0..n)
                .filter(|&i| !assigned[i] && !no_seed[i] && support[i] >= MIN_FRAMES)
                .max_by(|&a, &b| support[a].cmp(&support[b]).then(b.cmp(&a)))
            else {
                break;
            };
            let mut rep = obs[seed].bbox;
            let mut members = gather(&rep, &assigned);
            for _ in 0..MAX_ROUNDS {
                if members.is_empty() {
                    break;
                }
                let next_rep = median(&members);
                let next = gather(&next_rep, &assigned);
                let same = next == members;
                rep = next_rep;
                members = next;
                if same {
                    break;
                }
            }
            let days: BTreeSet<&str> = members.iter().map(|&i| obs[i].day.as_str()).collect();
            if members.len() < MIN_FRAMES || days.len() < MIN_DAYS || !proper(&rep) {
                no_seed[seed] = true;
                members.iter().for_each(|&i| no_seed[i] = true);
                continue;
            }
            members.iter().for_each(|&i| assigned[i] = true);
            out.push(rep);
        }
        out
    }

    #[test]
    fn the_lazy_seed_finds_what_storing_every_pair_found() {
        // Neighbouring places, a chain between them, places of uneven support that overlap, one
        // that fails the days rule, and scattered boxes: the representatives are those of the
        // eager rule, in the same order, at several IoUs.
        let mut v = parked("cam", 30, 3, A, 0);
        v.extend(parked("cam", 14, 2, B, 100));
        v.extend(parked("cam", 25, 1, [0.13, 0.52, 0.33, 0.72], 200));
        v.extend(parked("cam", 18, 4, [0.16, 0.50, 0.36, 0.70], 300));
        for k in 0..=22 {
            let x = 0.10 + k as f32 * 0.01;
            v.push(obs(
                "cam",
                500 + k,
                k % 3 + 1,
                "car",
                [x, 0.50, x + 0.2, 0.70],
                0.0,
            ));
        }
        for k in 0..60usize {
            let h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let x = (h % 700) as f32 / 1000.0;
            let y = ((h >> 24) % 700) as f32 / 1000.0;
            let w = 0.05 + ((h >> 40) % 100) as f32 / 1000.0;
            v.push(obs(
                "cam",
                700 + k,
                k % 4 + 1,
                "car",
                [x, y, x + w, y + w],
                0.0,
            ));
        }
        let mut sorted: Vec<&Obs> = v.iter().collect();
        sorted.sort_by(|a, b| canonical(a, b));
        let totals = day_totals(v.iter());
        for t in [0.5f32, 0.6, 0.7, 0.8] {
            let mut work = Work::default();
            let lazy = cluster(&sorted, t, 0.0, &totals, &mut work);
            let reference = eager(&sorted, t);
            assert!(!reference.is_empty(), "{t}");
            assert_eq!(lazy, reference, "IoU {t}");
        }
        // The table is the same whatever the order, and every box was clustered (no sample).
        let mut work = Work::default();
        let t = Table::build_counted(v.clone(), SPOT_IOU, MIN_DAY_SHARE, &mut work);
        assert_eq!(work.clustered, v.len() as u64);
        let mut r = v;
        r.reverse();
        assert_eq!(Table::build(r), t);
    }

    /// `per_day` frames of camera `gate` on each of `days` days with people only, and on `with`
    /// of them a car at `b` too.
    fn busy(per_day: usize, days: usize, with: usize, b: [f32; 4], label: &str) -> Vec<Obs> {
        let mut v = Vec::new();
        for d in 1..=days {
            for i in 0..per_day {
                let frame = d * 1000 + i;
                let x = 0.02 * (i % 20) as f32;
                v.push(obs(
                    "gate",
                    frame,
                    d,
                    "person",
                    [x, 0.1, x + 0.05, 0.3],
                    0.0,
                ));
                if i < with {
                    let j = (i % 5) as f32 * 0.002;
                    v.push(obs("gate", frame, d, label, b, j));
                }
            }
        }
        v
    }

    #[test]
    fn visitors_stopping_now_and_then_are_no_spot_a_parked_car_is() {
        // Visitors' cars stop at the gate on 6 of 40 busy frames a day, 4 days: 24 frames on 4
        // days would have made a spot; on 15% of each day's frames they did not stay.
        let visitors = busy(40, 4, 6, A, "car");
        assert!(Table::build_with(visitors.clone(), SPOT_IOU, 0.0).len() == 1);
        let t = Table::build(visitors.clone());
        assert!(t.is_empty(), "{t:?}");
        assert!(t.spot_at("gate", "car", &A).is_none());
        // A car parked through the day is on most frames: a spot, every day counts.
        let parked_car = busy(40, 4, 30, A, "car");
        let t = Table::build(parked_car);
        let s = &t.camera("gate")[0];
        assert_eq!((s.frames, s.days, s.stay_days), (120, 4, 4));
        assert_eq!(s.day_frames.get("2026-10-01"), Some(&40));
        assert!((s.day_shares()["2026-10-01"] - 0.75).abs() < 1e-9);
        // 3 of 10 reach 0.3 exactly.
        assert!(reaches(3, 10, 0.3) && !reaches(2, 10, 0.3));
    }

    #[test]
    fn a_day_it_did_not_stay_counts_frames_but_not_days() {
        // Parked on day 1 (all of 20 frames) and day 2 (all of 20); on day 3 it shows on 5 of 40.
        let mut v = busy(20, 2, 20, A, "car");
        for i in 0..40 {
            let frame = 3000 + i;
            let x = 0.02 * (i % 20) as f32;
            v.push(obs(
                "gate",
                frame,
                3,
                "person",
                [x, 0.1, x + 0.05, 0.3],
                0.0,
            ));
            if i < 5 {
                v.push(obs("gate", frame, 3, "car", A, 0.0));
            }
        }
        let t = Table::build(v);
        let s = &t.camera("gate")[0];
        assert_eq!((s.frames, s.days, s.stay_days), (45, 3, 2));
        assert!(s.stayed_on("2026-10-01") && !s.stayed_on("2026-10-03"));
        // Without day 3 it still holds; without day 1 one day it stayed is left: no evidence.
        assert!(s.holds_without("2026-10-03"));
        assert!(!s.holds_without("2026-10-01"));
        assert_eq!(s.stay_days_without("2026-10-01"), 1);
    }

    #[test]
    fn spot_at_takes_the_best_iou() {
        let mut v = parked("cam", 12, 3, A, 0);
        let near = [0.16, 0.50, 0.36, 0.70];
        v.extend(parked("cam", 12, 3, near, 100));
        let t = Table::build_at(v, 0.7);
        assert_eq!(t.len(), 2, "{t:?}");
        let probe = [0.15, 0.50, 0.35, 0.70];
        let s = t.spot_at("cam", "car", &probe).unwrap();
        let other = t.spots().find(|x| x.id != s.id).unwrap();
        assert!(iou(&s.bbox, &probe) >= iou(&other.bbox, &probe));
    }

    #[test]
    fn only_things_that_park_may_be_static() {
        for l in [
            "car",
            "Truck",
            "bus",
            "van",
            "motorcycle",
            "motorbike",
            "bicycle",
            "boat",
        ] {
            assert!(may_be_static(l), "{l}");
            assert!(!is_animate(l), "{l}");
        }
        // Known inanimate COCO classes, also under an alias.
        for l in ["bench", "potted plant", "pottedplant", "chair"] {
            assert!(may_be_static(l), "{l}");
        }
        // People and animals: COCO's, and those the auto-labeler knows besides.
        for l in [
            "person", " Person ", "dog", "cat", "bird", "horse", "raccoon", "deer", "fox",
        ] {
            assert!(is_animate(l), "{l}");
            assert!(!may_be_static(l), "{l}");
        }
        // A class this build cannot place is never static either (the safe side).
        for l in ["groundhog", "trash can", ""] {
            assert!(!is_animate(l), "{l}");
            assert!(!may_be_static(l), "{l}");
        }
    }

    #[test]
    fn people_and_animals_never_form_a_spot() {
        // A person sitting at one place on 12 frames over 3 days, a raccoon at its den, and a
        // parked car: only the car is a spot.
        let mut v = parked("cam", 12, 3, A, 0);
        for label in ["person", "raccoon", "groundhog"] {
            for o in parked("cam", 12, 3, B, 100) {
                v.push(Obs {
                    label: label.into(),
                    ..o
                });
            }
        }
        let t = Table::build(v.clone());
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!(t.camera("cam")[0].object, VEHICLE);
        for label in ["person", "raccoon", "groundhog"] {
            assert!(t.spot_at("cam", label, &B).is_none(), "{label}");
            assert_eq!(t.static_only("cam", [(label, B)]), None, "{label}");
        }
        // A parked car with a person beside it: not static-only, as before.
        assert_eq!(t.static_only("cam", [("car", A), ("person", B)]), None);
        assert!(t.static_only("cam", [("car", A)]).is_some());
        // The same boxes as a bench form a spot.
        let benches: Vec<Obs> = v
            .into_iter()
            .filter(|o| o.label == "person")
            .map(|o| Obs {
                label: "bench".into(),
                ..o
            })
            .collect();
        assert_eq!(Table::build(benches).len(), 1);
        // A person matched against a spot of persons (a table made elsewhere) is not at it.
        let mut fake = t.clone();
        if let Some(s) = fake.cameras.get_mut("cam").and_then(|v| v.first_mut()) {
            s.object = "person".into();
            s.bbox = B;
        }
        assert!(fake.spot_at("cam", "person", &B).is_none());
    }
}
