//! Static spots of fixed cameras: the places where the same object (a parked car, a boat on its
//! trailer) is labelled frame after frame. Pure. A spot is a cluster of training boxes (accepted
//! or added, no ignore region) of one object ([`object_of`]: the vehicle classes are one object,
//! like [`super::repeats::same_object`]) at one place of one camera, in coordinates normalised to
//! the frame (frames of a camera can differ in size), on at least [`MIN_FRAMES`] distinct frames
//! of at least [`MIN_DAYS`] distinct days ([`Table::build`]).
//!
//! **Only things that park** ([`may_be_static`]): a vehicle, or another known inanimate COCO class
//! (a bench, a potted plant). People and animals ([`is_animate`]) never form a spot: they are never
//! parked, their frames are the scarce data a dataset needs most, and the dataset check's lower
//! cut-off was measured on vehicles. A class this build cannot place (a custom name outside the
//! person, vehicle, animal and COCO lists, such as `groundhog`) forms no spot either: a missed cap
//! or a missed low-score `add` costs little, a capped-away animal frame costs the class.
//!
//! Two readers: the training plan caps the frames that show nothing but such objects
//! (`training::dataset`, `static_cap`), and the dataset check takes a detection at a spot as an
//! `add` from a lower score than elsewhere (`training::refine::judge`). The Review side lists a
//! folder's spots with [`overview`].
//!
//! **Clustering** (per camera and object, [`cluster`]): the observations are put in a canonical
//! order first (box coordinates, then frame, day, scene, label), so the result depends on the set
//! of observations only, never on the order a walk found them in. Then, repeatedly: the
//! unassigned observation with the most support (distinct frames with an unassigned box of its
//! object at IoU >= the spot IoU with it; ties to the first in the canonical order) seeds a
//! cluster; its members are, per frame, the unassigned box closest to the representative at an
//! IoU of the spot IoU or more; the representative becomes the coordinate-wise (lower) median of
//! the members,
//! and members and representative are taken again until they do not change (at most
//! [`MAX_ROUNDS`] times; the last is kept, so it ends however the data look). A cluster on
//! [`MIN_FRAMES`] frames of [`MIN_DAYS`] days is a spot and its members leave the pool; one that is
//! not leaves its members in the pool but none of them seeds again. It stops when no observation
//! left has enough support. Every member is within the spot IoU of the one representative box, so
//! two neighbouring parking places do not chain into one the way single linkage would: a box
//! between them joins the one it overlaps more, and a place counts as a spot only on its own.
//!
//! **Ids** ([`Spot::id`]): `<camera>/<object>/<x0>-<y0>-<x1>-<y1>`, the representative box in
//! steps of 1/[`ID_STEPS`] of the frame (2%). A rerun on the same labels gives the same ids; a rerun
//! after more labels keeps an id unless the median moves across a step (a pixel of jitter does so
//! for a coordinate that sits on a step's edge: an id is a convenience for grouping, never a key
//! that must survive). Two spots of one camera and object whose boxes round alike get `~2`, `~3`
//! in the canonical order.
//!
//! **Bounds**: a camera with more than [`MAX_OBS_PER_CAMERA`] boxes is clustered on the boxes of a
//! fixed sample of its frames (the frames with the lowest [`fnv1a`] hash of their key until the
//! cap); the spots it finds are matched against every frame all the same ([`Table::spot_at`]).

use super::bulk::Wanted;
use super::export::ExportFrame;
use super::iou;
use super::repeats::VEHICLES;
use crate::model::coco::coco_class;
use crate::training::autolabel::Category;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// IoU (normalised boxes) from which a box lies at a spot: with the spot's representative box,
/// and between two boxes of a cluster's seed. Measured on 851 reviewed frames (docs/TRAINING.md,
/// "Static spots").
pub const SPOT_IOU: f32 = 0.6;
/// Fewest distinct frames with a box at a place that make it a spot.
pub const MIN_FRAMES: usize = 10;
/// Fewest distinct days among them: one afternoon of a car parked is no static object.
pub const MIN_DAYS: usize = 2;
/// Most boxes of one camera clustered (see the module docs).
pub const MAX_OBS_PER_CAMERA: usize = 20_000;
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

/// A static spot (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Spot {
    /// `<camera>/<object>/<x0>-<y0>-<x1>-<y1>` (see the module docs).
    pub id: String,
    pub camera: String,
    /// [`object_of`] its labels.
    pub object: String,
    /// The members' most common label (ties: the first by name).
    pub label: String,
    /// The representative box: the coordinate-wise median of the members, normalised.
    pub bbox: [f32; 4],
    /// Distinct frames (members: one box per frame).
    pub frames: usize,
    /// Distinct days.
    pub days: usize,
    /// Frames per scene.
    pub scenes: BTreeMap<String, usize>,
    /// Frames per day.
    pub by_day: BTreeMap<String, usize>,
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

    /// Whether the spot still is one without the frames of `day` ([`MIN_FRAMES`] frames on
    /// [`MIN_DAYS`] days): the evidence a frame of `day` may use (one day's frames are near
    /// copies of each other, so a spot only that day makes is no evidence for a frame of it). The
    /// counts are the spot's minus that day's; the cluster is not made again without it.
    pub fn holds_without(&self, day: &str) -> bool {
        self.frames_without(day) >= MIN_FRAMES && self.days_without(day) >= MIN_DAYS
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

/// The canonical order of observations: the box, then frame, day, scene, label.
fn canonical(a: &Obs, b: &Obs) -> Ordering {
    a.bbox
        .iter()
        .zip(&b.bbox)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
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

/// A spot before its id.
struct Found {
    object: String,
    label: String,
    bbox: [f32; 4],
    frames: usize,
    scenes: BTreeMap<String, usize>,
    by_day: BTreeMap<String, usize>,
}

/// The clustering of one camera's boxes of one object (see the module docs); `obs` in canonical
/// order, `t` the spot IoU.
#[allow(
    clippy::indexing_slicing,
    reason = "every index is below n (obs, nbrs, assigned, no_seed, support, frame_of) or a frame \
              index below the frame count (seen)"
)]
fn cluster(object: &str, obs: &[Obs], t: f32) -> Vec<Found> {
    let n = obs.len();
    // Frame and day indices.
    let mut frame_ids: BTreeMap<&str, usize> = BTreeMap::new();
    for o in obs {
        let next = frame_ids.len();
        frame_ids.entry(o.frame.as_str()).or_insert(next);
    }
    let frame_of: Vec<usize> = obs
        .iter()
        .map(|o| frame_ids.get(o.frame.as_str()).copied().unwrap_or(0))
        .collect();
    // Neighbours at IoU >= t on other frames. The canonical order sorts by x0 first, so a sweep
    // stops at the first box that starts right of this one's end (no overlap, IoU 0).
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let a = &obs[i].bbox;
        for j in (i + 1)..n {
            let b = &obs[j].bbox;
            if b[0] >= a[2] {
                break;
            }
            if frame_of[i] != frame_of[j] && iou(a, b) >= t {
                nbrs[i].push(j);
                nbrs[j].push(i);
            }
        }
    }
    let mut assigned = vec![false; n];
    let mut no_seed = vec![false; n];
    // Distinct frames marker, by generation (no clearing between counts).
    let mut seen = vec![0u32; frame_ids.len()];
    let mut stamp = 0u32;
    let mut support_of = |i: usize, assigned: &[bool], seen: &mut [u32]| -> usize {
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            seen.iter_mut().for_each(|s| *s = 0);
            stamp = 1;
        }
        let mut k = 0;
        for &j in std::iter::once(&i).chain(&nbrs[i]) {
            if assigned[j] {
                continue;
            }
            if let Some(s) = seen.get_mut(frame_of[j])
                && *s != stamp
            {
                *s = stamp;
                k += 1;
            }
        }
        k
    };
    let mut support: Vec<usize> = (0..n)
        .map(|i| support_of(i, &assigned, &mut seen))
        .collect();
    // The unassigned box per frame closest to `rep` at IoU >= t (ties: the first), sorted.
    let gather = |rep: &[f32; 4], assigned: &[bool]| -> Vec<usize> {
        let mut best: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
        for (i, o) in obs.iter().enumerate() {
            if assigned[i] || o.bbox[0] >= rep[2] {
                continue;
            }
            let v = iou(rep, &o.bbox);
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
    let mut out = Vec::new();
    loop {
        let seed = (0..n)
            .filter(|&i| !assigned[i] && !no_seed[i] && support[i] >= MIN_FRAMES)
            .max_by(|&a, &b| support[a].cmp(&support[b]).then(b.cmp(&a)));
        let Some(seed) = seed else { break };
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
            for &i in &members {
                no_seed[i] = true;
            }
            continue;
        }
        let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
        let mut scenes: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_day: BTreeMap<String, usize> = BTreeMap::new();
        for &i in &members {
            let o = &obs[i];
            *labels.entry(o.label.as_str()).or_default() += 1;
            *scenes.entry(o.scene.clone()).or_default() += 1;
            *by_day.entry(o.day.clone()).or_default() += 1;
        }
        // Most common, ties to the first by name (the map's order; `max_by` keeps the last max).
        let label = labels
            .iter()
            .rev()
            .max_by_key(|(_, k)| **k)
            .map(|(l, _)| (*l).to_string())
            .unwrap_or_default();
        let mut touched: BTreeSet<usize> = BTreeSet::new();
        for &i in &members {
            assigned[i] = true;
            touched.extend(nbrs[i].iter().copied());
        }
        for i in touched {
            if !assigned[i] {
                support[i] = support_of(i, &assigned, &mut seen);
            }
        }
        out.push(Found {
            object: object.to_string(),
            label,
            bbox: rep,
            frames: members.len(),
            scenes,
            by_day,
        });
    }
    out
}

/// The spots of every camera in `obs`, by camera, at `t` (see the module docs).
fn build(obs: impl IntoIterator<Item = Obs>, t: f32) -> BTreeMap<String, Vec<Spot>> {
    let mut by_camera: BTreeMap<String, Vec<Obs>> = BTreeMap::new();
    for o in obs {
        // People, animals and classes this build cannot place are not clustered at all.
        if proper(&o.bbox) && !o.camera.is_empty() && may_be_static(&o.label) {
            by_camera.entry(o.camera.clone()).or_default().push(o);
        }
    }
    let mut out = BTreeMap::new();
    for (camera, mut list) in by_camera {
        if list.len() > MAX_OBS_PER_CAMERA {
            list = sample(list);
        }
        let mut by_object: BTreeMap<String, Vec<Obs>> = BTreeMap::new();
        for o in list {
            by_object.entry(object_of(&o.label)).or_default().push(o);
        }
        let mut found: Vec<Found> = Vec::new();
        for (object, mut list) in by_object {
            list.sort_by(canonical);
            found.extend(cluster(&object, &list, t));
        }
        // Ids in a fixed order: object, then the box.
        found.sort_by(|a, b| {
            a.object.cmp(&b.object).then_with(|| {
                a.bbox
                    .iter()
                    .zip(&b.bbox)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(Ordering::Equal)
            })
        });
        let mut used: BTreeMap<String, usize> = BTreeMap::new();
        let spots: Vec<Spot> = found
            .into_iter()
            .map(|f| {
                let base = spot_id(&camera, &f.object, &f.bbox);
                let n = used.entry(base.clone()).or_default();
                *n += 1;
                let id = if *n == 1 { base } else { format!("{base}~{n}") };
                Spot {
                    id,
                    camera: camera.clone(),
                    object: f.object,
                    label: f.label,
                    bbox: f.bbox,
                    frames: f.frames,
                    days: f.by_day.len(),
                    scenes: f.scenes,
                    by_day: f.by_day,
                }
            })
            .collect();
        if !spots.is_empty() {
            out.insert(camera, spots);
        }
    }
    out
}

/// The boxes of a fixed sample of frames, at most [`MAX_OBS_PER_CAMERA`]: frames by [`fnv1a`] of
/// their key (then the key), whole frames only.
fn sample(list: Vec<Obs>) -> Vec<Obs> {
    let mut per_frame: BTreeMap<&str, usize> = BTreeMap::new();
    for o in &list {
        *per_frame.entry(o.frame.as_str()).or_default() += 1;
    }
    let mut frames: Vec<(&str, usize)> = per_frame.into_iter().collect();
    frames.sort_by(|a, b| fnv1a(a.0).cmp(&fnv1a(b.0)).then(a.0.cmp(b.0)));
    let mut keep: BTreeSet<String> = BTreeSet::new();
    let mut total = 0;
    for (f, k) in frames {
        if total + k > MAX_OBS_PER_CAMERA {
            break;
        }
        total += k;
        keep.insert(f.to_string());
    }
    list.into_iter()
        .filter(|o| keep.contains(&o.frame))
        .collect()
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

/// The spots of a set of observations, per camera (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Table {
    /// The IoU spots were made and are matched at.
    pub iou: f32,
    /// Spots per camera, by object then box.
    pub cameras: BTreeMap<String, Vec<Spot>>,
}

impl Table {
    /// The spots of `obs` at [`SPOT_IOU`].
    pub fn build(obs: impl IntoIterator<Item = Obs>) -> Self {
        Self::build_at(obs, SPOT_IOU)
    }

    /// The spots of `obs` at the spot IoU `t` (the measurement compares several).
    pub fn build_at(obs: impl IntoIterator<Item = Obs>, t: f32) -> Self {
        Self {
            iou: t,
            cameras: build(obs, t),
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

/// One spot of a folder's [`overview`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpotSummary {
    #[serde(flatten)]
    pub spot: Spot,
    /// Reviewed frames with a training box at the spot ([`Table::spot_at`]: each box counts for
    /// its best spot, so where two spots overlap this can differ from the cluster's `frames`).
    pub labelled: usize,
    /// Reviewed frames with a live `add` suggestion of this spot ([`super::refine::live`] with
    /// [`super::refine::Suggestion::spot`] = the id).
    pub open_adds: usize,
    /// Those frames (`<scene>/<camera>/<date>/<stem>.jpg`), in the order given.
    pub open_frames: Vec<String>,
}

/// The static spots of `frames` (reviewed frames, e.g. [`super::export::reviewed_frames_filtered`]
/// of the folder under review) with their counts, by camera then id: what a "Static objects" card
/// lists. The spots are made from these frames ([`observations`], [`Table::build`]), so they are
/// the dataset check's spots when `frames` is its selection.
pub fn overview(frames: &[ExportFrame], wanted: &Wanted<'_>) -> Vec<SpotSummary> {
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
        let open: BTreeSet<&str> = super::refine::live(&f.labels, Some(wanted))
            .iter()
            .filter(|l| l.suggestion.kind == super::refine::Kind::Add)
            .filter_map(|l| l.suggestion.spot.as_deref())
            .collect();
        for id in open {
            if let Some(x) = out.get_mut(id) {
                x.open_adds += 1;
                x.open_frames.push(f.rel.jpg());
            }
        }
    }
    out.into_values().collect()
}

/// The indices (into [`super::LabelFile::refine`]) of the live `add` suggestions of spot `id` on
/// `f` ([`super::refine::live`] with the class list): what a bulk "accept the adds at this spot"
/// hands to [`super::refine::accept`] one by one, highest index first (an accept removes its
/// suggestion, so a lower index stays valid).
pub fn live_adds_at(f: &super::LabelFile, id: &str, wanted: &Wanted<'_>) -> Vec<usize> {
    let mut v: Vec<usize> = super::refine::live(f, Some(wanted))
        .iter()
        .filter(|l| {
            l.suggestion.kind == super::refine::Kind::Add
                && l.suggestion.spot.as_deref() == Some(id)
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
        let s = sample(v.clone());
        assert_eq!(s.len(), MAX_OBS_PER_CAMERA);
        let mut r = v;
        r.reverse();
        let mut a: Vec<String> = s.into_iter().map(|o| o.frame).collect();
        let mut b: Vec<String> = sample(r).into_iter().map(|o| o.frame).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
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
