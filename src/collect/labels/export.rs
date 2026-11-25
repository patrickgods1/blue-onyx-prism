//! Exports of the reviewed frames (`POST /v1/review/export`): YOLO txt (`labels/<split>/*.txt`,
//! `classes.txt`, `data.yaml`), COCO instances JSON (`annotations/instances_<split>.json`),
//! X-AnyLabeling JSON (`xanylabeling/`) and a benchmark `dir:` set (`benchmark/images/` +
//! `benchmark/manifest.json`, tags `<scene>`, `camera:<c>`, `date:<d>`, `split:<s>`,
//! `bucket:<b>` and `hard-negative` for a frame confirmed empty). Only reviewed frames are
//! exported (X-AnyLabeling may add the queue's unreviewed frames, pre-filled, unchecked).
//!
//! Split ([`assign_splits`]): whole (camera, day) groups, so frames of one camera on one day never
//! sit on both sides. Other days of a held-out group's camera still train, unless a camera is
//! named. Explicit `holdout_cameras` / `holdout_days` decide when given;
//! else a (camera, date) group is held out when the first 8 bytes of SHA-256(`camera/date`), as a
//! fraction of 2^64, are below `holdout_fraction` ([`hash_heldout`], stable when days are added;
//! the Review queue's held-out steering asks the same function), with at least
//! one group on each side once there are two groups. With [`ExportRequest::heldout`] (the Dataset
//! tab's export only) the review folder's `heldout-days.json` ([`super::heldout`]) holds out every
//! day it lists too, and an export of the default split adds its held-out days to it, plus days
//! for cameras with fewer than two ([`ExportReport::heldout`]); exports of other settings only
//! read it. Selections for other uses ([`reviewed_frames_filtered`], [`run_selected`], the dataset
//! check's folds) never touch it.
//!
//! Class filter ([`ClassFilter`]): with `labels`, only frames with a training box of one of
//! them are exported (other-class boxes in a kept frame stay labelled, class ids still follow
//! the class list); frames confirmed empty are exported while `include_empty` is set (default).
//! Applied before the split, recorded in `export.json`.
//!
//! Ignore regions ([`LabelBox::is_ignore`]: unsure boxes, boxes rejected as too small) are real
//! objects not to train on. They stay out of the YOLO labels, so `ignore.json` ([`IgnoreFile`],
//! at the export's root) names them per image: a training run knows the object is there (it
//! trains the frame with the object as background, or leaves the frame out:
//! `training::dataset::IgnoreMode`; validation never holds such a frame). They are `ignore` in
//! the benchmark set.
//!
//! The size floor ([`ExportRequest::size_floor`], `collect.size_floor`, off by default): a
//! training box whose short side at the floor's model input is under it ([`super::under_floor`])
//! is exported as an ignore region exactly as an unsure box is (out of YOLO, in `ignore.json`
//! marked `size_floor`, `ignore` in the benchmark set, `attributes.difficult` + `size_floor` in
//! COCO); the X-AnyLabeling files keep it as a plain box (an import would otherwise make it
//! `difficult` in the label file for good). The label file is never changed: a floor change is a
//! re-export. `export.json` records the floor and counts the boxes under it ([`UnderFloor`]). A
//! frame whose only training boxes are under the floor has ignore regions and no label, as a frame
//! of unsure boxes only: an empty YOLO file, listed in `ignore.json`, never "confirmed empty"
//! (that is the reviewer's flag) and left out of training.
//!
//! Frames whose only boxes a labelers run accepted at parked spots ([`ExportFrame::parked_spots`])
//! are listed in `ignore.json` too (`parked`): identical crops can dominate training, so a training
//! run caps them per spot. They train normally otherwise.
//!
//! `frames.json` ([`FrameNotes`], at the export's root, every format): the frames a reviewer
//! confirmed empty, the frames accepted in bulk and the Review queue's random picks
//! ([`ReviewPick::Random`]), so a training run knows them whichever formats were ticked (it scores
//! the held-out random picks apart, `heldout:random`). The benchmark manifest tags a random pick
//! `random`. `hard_negatives.json` lists the rejected boxes.
//!
//! Pure builders ([`yolo_txt`], [`coco_json`], [`benchmark_manifest`], [`ignore_file`],
//! [`frame_notes`]) plus
//! [`run`], which exports into an empty or new folder, or a new `export-<UTC time>` folder inside a
//! non-empty one, never into the dataset (images hard-linked, else copied). An export appears
//! whole or not at all, since a part of one reads as a valid folder (its `classes.txt`, without
//! its `ignore.json`, teaches unsure objects as background): see [`Staging`].
//!
//! A reviewed frame that lost its image or sidecar and a folder that cannot be listed cannot be
//! exported; the report counts them (`lost_frames`, `unreadable_folders`) and warns.
//!
//! Progress ([`run_filtered_with`], [`ExportStep`]): the label files read, every image placed and
//! per-frame label file written, then the publish, for a caller that shows a running export (the
//! Jobs tab of the Train area). The plain entry points report to nobody. The export runs one
//! writer per format ([`formats`]), each beside the count of the files it reports, so the count
//! shown up front is what it then writes.
//!
//! Stopping ([`run_filtered_until`]): a caller's stop (the service stopping) is asked between two
//! label files read and two files written; once it says yes the export ends with [`Stopped`]
//! (before the held-out list is touched when it comes during the walk) and its staging folder is
//! removed. [`Staging`] also removes itself when dropped unpublished, so a panic leaves no
//! `.building-<id>` folder either; one a kill or a crash leaves is held busy, so the next storage
//! sweep removes it ([`crate::storage::sweep`]; [`run_filtered_recorded`] also records it, for a
//! folder of the user's choice).

use super::super::layout::{self, Rel, Walker};
use super::super::record::{Bucket, Record};
use super::heldout::{self, HeldoutPolicy, HeldoutReport};
use super::xany;
use super::{
    BoxState, Frame, LabelBox, LabelFile, ReviewPick, SizeFloor, Status, clamp_box, spots,
};
use crate::benchmark::images::{Manifest, ManifestImage, ManifestObject};
use anyhow::{Context, Result, bail};
use formats::Writer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

mod formats;

/// An export or import ended early because its caller asked it to stop ([`StopFn`]): the service
/// is stopping. An export keeps nothing of itself; an import keeps the frames it had done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("stopped because the service is stopping")]
pub struct Stopped;

/// Whether `e` (or its cause chain) is a [`Stopped`].
pub fn is_stopped(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Stopped>().is_some())
}

/// Asked between two steps of an export or import: true = stop now ([`Stopped`]). Called from the
/// walk's threads too, so it must be `Sync` and cheap.
pub type StopFn<'a> = &'a (dyn Fn() -> bool + Sync);

/// Default share of (camera, day) groups held out for validation.
pub const DEFAULT_HOLDOUT_FRACTION: f32 = 0.2;

/// An export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Yolo,
    Coco,
    #[serde(rename = "xanylabeling")]
    XAnyLabeling,
    Benchmark,
}

impl Format {
    pub const ALL: [Format; 4] = [
        Format::Yolo,
        Format::Coco,
        Format::XAnyLabeling,
        Format::Benchmark,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Yolo => "yolo",
            Format::Coco => "coco",
            Format::XAnyLabeling => "xanylabeling",
            Format::Benchmark => "benchmark",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.as_str().eq_ignore_ascii_case(s.trim()))
    }
}

/// Train or validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
        }
    }
}

/// How frames are split (see the module doc).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplitConfig {
    /// Cameras whose frames all go to `val`.
    pub holdout_cameras: Vec<String>,
    /// Dates (`YYYY-MM-DD`) whose frames all go to `val`.
    pub holdout_days: Vec<String>,
    /// Share of (camera, day) groups held out when no camera or day is named (0..=1).
    pub holdout_fraction: f32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            holdout_cameras: Vec::new(),
            holdout_days: Vec::new(),
            holdout_fraction: DEFAULT_HOLDOUT_FRACTION,
        }
    }
}

/// A (camera, date) group.
pub type Group = (String, String);

/// First 8 bytes of SHA-256(`camera/date`): the rank of a group in the split.
pub fn group_hash(g: &Group) -> u64 {
    day_hash(&g.0, &g.1)
}

/// [`group_hash`] of the camera day `camera`, `date`.
fn day_hash(camera: &str, date: &str) -> u64 {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(camera.as_bytes());
    h.update(b"/");
    h.update(date.as_bytes());
    let d = h.finalize();
    let mut b = [0u8; 8];
    for (o, i) in b.iter_mut().zip(d.iter()) {
        *o = *i;
    }
    u64::from_be_bytes(b)
}

/// Whether the hash split holds the camera day `camera`, `date` out at fraction `frac` (0..=1):
/// its [`group_hash`] as a fraction of 2^64 is below `frac`. The per-day rule of
/// [`assign_splits`] when no camera or day is named, before its "one group on each side" guard
/// (which looks at every group of an export); the Review queue's held-out steering
/// ([`super::need::Need::heldout_day`]) asks it at [`DEFAULT_HOLDOUT_FRACTION`].
pub fn hash_heldout(camera: &str, date: &str, frac: f64) -> bool {
    #[allow(clippy::cast_precision_loss, reason = "a fraction of 2^64")]
    let x = day_hash(camera, date) as f64 / 18_446_744_073_709_551_616.0;
    x < frac
}

/// The split of every group (see the module doc). Pure and deterministic.
pub fn assign_splits(groups: &BTreeSet<Group>, cfg: &SplitConfig) -> BTreeMap<Group, Split> {
    let explicit = !cfg.holdout_cameras.is_empty() || !cfg.holdout_days.is_empty();
    if explicit {
        return groups
            .iter()
            .map(|g| {
                let val = cfg.holdout_cameras.contains(&g.0) || cfg.holdout_days.contains(&g.1);
                (g.clone(), if val { Split::Val } else { Split::Train })
            })
            .collect();
    }
    let frac = f64::from(cfg.holdout_fraction.clamp(0.0, 1.0));
    let mut out: BTreeMap<Group, Split> = groups
        .iter()
        .map(|g| {
            let val = hash_heldout(&g.0, &g.1, frac);
            (g.clone(), if val { Split::Val } else { Split::Train })
        })
        .collect();
    if groups.len() >= 2 && frac > 0.0 {
        let ranked = || {
            let mut v: Vec<&Group> = groups.iter().collect();
            v.sort_by_key(|g| (group_hash(g), (*g).clone()));
            v
        };
        if !out.values().any(|s| *s == Split::Val)
            && let Some(g) = ranked().first()
        {
            out.insert((*g).clone(), Split::Val);
        }
        if !out.values().any(|s| *s == Split::Train)
            && let Some(g) = ranked().last()
        {
            out.insert((*g).clone(), Split::Train);
        }
    }
    out
}

/// The flat file name of a frame in an export: `<scene>.<camera>.<date>.<stem>.jpg` (no
/// component may contain a `.`, so it parses back with [`parse_flat_name`]).
pub fn flat_name(r: &Rel) -> String {
    format!("{}.{}.{}.{}.jpg", r.scene, r.camera, r.date, r.stem)
}

/// The frame of a [`flat_name`].
pub fn parse_flat_name(name: &str) -> Option<Rel> {
    let stem = name.strip_suffix(".jpg")?;
    let parts: Vec<&str> = stem.split('.').collect();
    let [scene, camera, date, stem] = parts.as_slice() else {
        return None;
    };
    layout::parse_rel(&format!("{scene}/{camera}/{date}/{stem}.jpg"))
}

/// One frame of an export.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportFrame {
    pub rel: Rel,
    /// [`flat_name`].
    pub name: String,
    /// The frame's image.
    pub image: PathBuf,
    pub width: u32,
    pub height: u32,
    pub buckets: Vec<Bucket>,
    pub split: Split,
    pub labels: LabelFile,
    /// The export's size floor ([`ExportRequest::size_floor`], set on every frame by the export;
    /// None = off): a training box under it is exported as an ignore region ([`Self::under_floor`]).
    pub floor: Option<SizeFloor>,
}

impl ExportFrame {
    /// The boxes `keep` takes that are not empty once clamped to the frame.
    fn placed(&self, keep: impl Fn(&LabelBox) -> bool) -> Vec<&LabelBox> {
        self.labels
            .boxes
            .iter()
            .filter(|b| {
                let c = clamp_box(b.bbox, self.width, self.height);
                keep(b) && c[2] > c[0] && c[3] > c[1]
            })
            .collect()
    }

    /// A training box (accepted or added, no ignore region already) under the frame's size floor
    /// ([`super::under_floor`]): the export makes it an ignore region, exactly as an unsure box is.
    /// The label file keeps it as it is. Never with the floor off.
    pub fn under_floor(&self, b: &LabelBox) -> bool {
        b.is_training()
            && !b.is_ignore()
            && super::under_floor(&b.bbox, self.width, self.height, self.floor)
    }

    /// Training boxes (unsure ones included: COCO marks them), clamped to the frame, empty ones
    /// and those under the size floor ([`Self::under_floor`]) left out.
    pub fn boxes(&self) -> Vec<&LabelBox> {
        self.placed(|b| b.is_training() && !self.under_floor(b))
    }

    /// Every training box, the size floor regardless (the X-AnyLabeling files: a box left out there
    /// would come back from an import as a rejected box), clamped to the frame, empty ones left out.
    pub fn training_boxes(&self) -> Vec<&LabelBox> {
        self.placed(LabelBox::is_training)
    }

    /// Ignore regions ([`LabelBox::is_ignore`], and the training boxes under the size floor),
    /// clamped to the frame, empty ones left out.
    pub fn ignore_boxes(&self) -> Vec<&LabelBox> {
        self.placed(|b| b.is_ignore() || self.under_floor(b))
    }

    /// The parked spots of a frame whose training boxes all come from one, sorted; empty for any
    /// other frame. A box a reviewer moved or renamed since is the reviewer's box.
    pub fn parked_spots(&self) -> Vec<String> {
        let keys: Option<BTreeSet<String>> = self
            .boxes()
            .into_iter()
            .map(|b| {
                let mark = b.auto.parked.as_ref().filter(|_| {
                    b.reason.as_deref() == Some(spots::PARKED_REASON)
                        && b.orig_bbox.is_none()
                        && b.model_label.is_none()
                })?;
                let [x0, y0, x1, y1] = mark.bbox;
                Some(format!(
                    "{}/{}x{}/{}/{x0:.0},{y0:.0},{x1:.0},{y1:.0}",
                    self.rel.camera, self.width, self.height, b.label
                ))
            })
            .collect();
        keys.map(|k| k.into_iter().collect()).unwrap_or_default()
    }
}

/// File at the root of an export: the ignore regions of its images. A missing file means none
/// (exports of older versions).
pub const IGNORE_FILE: &str = "ignore.json";
/// Version [`ignore_file`] writes.
pub const IGNORE_VERSION: u32 = 1;

/// One ignore region, in the numbers of a YOLO label line (normalised to the image, center and
/// size, 6 decimals), so a reader treats it like the frame's boxes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreBox {
    pub label: String,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    /// A training box the export's size floor made an ignore region ([`ExportFrame::under_floor`]),
    /// not one a reviewer marked. Written only when true; older builds ignore the key and read the
    /// region like any other.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub size_floor: bool,
}

/// [`IGNORE_FILE`]: image file name (the [`flat_name`] of `images/<split>/`) -> its ignore regions.
/// Images without any are not listed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreFile {
    pub version: u32,
    pub images: BTreeMap<String, Vec<IgnoreBox>>,
    /// Image file name -> the spots of a frame with nothing but parked-spot boxes
    /// ([`ExportFrame::parked_spots`]). These are no ignore regions: the frames train, up to a cap
    /// per spot. Absent in exports of older versions, and older builds skip it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parked: BTreeMap<String, Vec<String>>,
    /// The size floor the export applied ([`ExportRequest::size_floor`]); absent when off (and in
    /// exports of older versions, which older builds read the same).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
}

/// File at the root of an export: what a training run needs to know about its frames beyond their
/// boxes. A missing file means not known (exports of older versions).
pub const FRAMES_FILE: &str = "frames.json";
/// File at the root of an export: its rejected boxes ([`LabelFile::hard_negatives`]).
pub const HARD_NEGATIVES_FILE: &str = "hard_negatives.json";
/// Version [`frame_notes`] writes.
pub const FRAMES_VERSION: u32 = 1;

/// [`FRAMES_FILE`]: image file names ([`flat_name`]), sorted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameNotes {
    pub version: u32,
    /// Confirmed empty by a reviewer.
    #[serde(default)]
    pub empty_confirmed: Vec<String>,
    /// Accepted in bulk: nobody looked at the frame on its own.
    #[serde(default)]
    pub bulk_accepted: Vec<String>,
    /// Served by the Review queue as a random pick ([`ReviewPick::Random`]): chosen whatever it
    /// shows. Added within version 1: older builds ignore it, older files have none.
    #[serde(default)]
    pub random_pick: Vec<String>,
}

/// The [`FrameNotes`] of `frames`.
pub fn frame_notes(frames: &[ExportFrame]) -> FrameNotes {
    let names = |keep: &dyn Fn(&LabelFile) -> bool| {
        let mut v: Vec<String> = frames
            .iter()
            .filter(|f| keep(&f.labels))
            .map(|f| f.name.clone())
            .collect();
        v.sort_unstable();
        v
    };
    FrameNotes {
        version: FRAMES_VERSION,
        empty_confirmed: names(&|l| l.empty_confirmed),
        bulk_accepted: names(&|l| l.bulk_accepted),
        random_pick: names(&|l| l.review_pick == Some(ReviewPick::Random)),
    }
}

/// `x` rounded to `d` decimals, as f64 (clean JSON numbers).
fn round_to(x: f64, d: i32) -> f64 {
    let p = 10f64.powi(d);
    (x * p).round() / p
}

/// A frame's YOLO label file: one `class cx cy w h` line per training box, normalised to the
/// frame and clamped to 0..1, 6 decimals; an empty file for a frame without objects. Boxes of a
/// label not in `classes` are an error (the frame must not be exported half-labelled). Unsure
/// (`difficult`) boxes are left out, for [`ignore_file`]; the frame and its other boxes stay.
pub fn yolo_txt(f: &ExportFrame, classes: &[String]) -> Result<String> {
    let mut out = String::new();
    for b in f.boxes().into_iter().filter(|b| !b.difficult) {
        let Some(id) = classes.iter().position(|c| *c == b.label) else {
            bail!(
                "{}: label '{}' is not in the class list",
                f.rel.jpg(),
                b.label
            );
        };
        let [cx, cy, bw, bh] = yolo_numbers(b, f);
        out.push_str(&format!("{id} {cx:.6} {cy:.6} {bw:.6} {bh:.6}\n"));
    }
    Ok(out)
}

/// `[cx, cy, w, h]` of `b`, normalised to the frame and clamped to 0..1, 6 decimals.
fn yolo_numbers(b: &LabelBox, f: &ExportFrame) -> [f64; 4] {
    let (w, h) = (f64::from(f.width.max(1)), f64::from(f.height.max(1)));
    let c = clamp_box(b.bbox, f.width, f.height).map(f64::from);
    let n = |v: f64| round_to(v.clamp(0.0, 1.0), 6);
    [
        n((c[0] + c[2]) / 2.0 / w),
        n((c[1] + c[3]) / 2.0 / h),
        n((c[2] - c[0]) / w),
        n((c[3] - c[1]) / h),
    ]
}

/// The ignore regions of `frames` ([`IgnoreFile`]; the training boxes under a frame's size floor
/// among them, marked `size_floor`). Labels are kept as reviewed, in the class list or not: such a
/// box needs no class id, and the object is just as real. `size_floor` is left for the caller.
pub fn ignore_file(frames: &[ExportFrame]) -> IgnoreFile {
    let images = frames
        .iter()
        .filter_map(|f| {
            let list: Vec<IgnoreBox> = f
                .ignore_boxes()
                .into_iter()
                .map(|b| {
                    let [cx, cy, w, h] = yolo_numbers(b, f);
                    IgnoreBox {
                        label: b.label.clone(),
                        cx,
                        cy,
                        w,
                        h,
                        size_floor: f.under_floor(b),
                    }
                })
                .collect();
            (!list.is_empty()).then(|| (f.name.clone(), list))
        })
        .collect();
    let parked = frames
        .iter()
        .filter_map(|f| {
            let spots = f.parked_spots();
            (!spots.is_empty()).then(|| (f.name.clone(), spots))
        })
        .collect();
    IgnoreFile {
        version: IGNORE_VERSION,
        images,
        parked,
        size_floor: None,
    }
}

/// COCO instances JSON of `frames` (file names as given): category ids 1..=K in `classes`
/// order, image and annotation ids 1.. in order, `bbox` `[x, y, w, h]` in pixels (2 decimals),
/// `area` = w * h, `iscrowd` 0, `attributes.difficult` on unsure boxes, and on the boxes under a
/// frame's size floor (with `attributes.size_floor`), as on the other ignore regions that are
/// training boxes. A frame without objects is an image without annotations.
pub fn coco_json(frames: &[&ExportFrame], classes: &[String]) -> Result<serde_json::Value> {
    use serde_json::json;
    let mut images = Vec::new();
    let mut anns = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let image_id = i + 1;
        images.push(json!({
            "id": image_id,
            "file_name": f.name,
            "width": f.width,
            "height": f.height,
        }));
        for b in f.training_boxes() {
            let Some(cat) = classes.iter().position(|c| *c == b.label) else {
                bail!(
                    "{}: label '{}' is not in the class list",
                    f.rel.jpg(),
                    b.label
                );
            };
            let c = clamp_box(b.bbox, f.width, f.height).map(f64::from);
            let (w, h) = (round_to(c[2] - c[0], 2), round_to(c[3] - c[1], 2));
            let mut ann = json!({
                "id": anns.len() + 1,
                "image_id": image_id,
                "category_id": cat + 1,
                "bbox": [round_to(c[0], 2), round_to(c[1], 2), w, h],
                "area": round_to(w * h, 2),
                "iscrowd": 0,
            });
            let attributes = if f.under_floor(b) {
                Some(json!({"difficult": true, "size_floor": true}))
            } else {
                b.difficult.then(|| json!({"difficult": true}))
            };
            if let Some(a) = attributes
                && let Some(o) = ann.as_object_mut()
            {
                o.insert("attributes".into(), a);
            }
            anns.push(ann);
        }
    }
    let categories: Vec<serde_json::Value> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| json!({"id": i + 1, "name": c, "supercategory": "object"}))
        .collect();
    Ok(json!({
        "info": {"description": "Blue Onyx Prism review export", "version": "1"},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": categories,
    }))
}

/// Tags of a frame in the benchmark manifest.
fn tags(f: &ExportFrame) -> Vec<String> {
    let mut t = vec![
        f.rel.scene.clone(),
        format!("camera:{}", f.rel.camera),
        format!("date:{}", f.rel.date),
        format!("split:{}", f.split.as_str()),
    ];
    t.extend(f.buckets.iter().map(|b| format!("bucket:{}", b.as_str())));
    if f.labels.empty_confirmed {
        t.push("hard-negative".into());
    }
    if f.labels.review_pick == Some(ReviewPick::Random) {
        t.push("random".into());
    }
    t
}

/// The benchmark manifest of `frames`: their training boxes as ground truth, scored on
/// `classes` only (predictions of other labels count neither way); ignore regions
/// ([`LabelBox::is_ignore`], also the too-small rejects that are no training box, and the
/// training boxes under the size floor, [`ExportFrame::under_floor`]) are `ignore`.
pub fn benchmark_manifest(frames: &[&ExportFrame], classes: &[String], id: &str) -> Manifest {
    Manifest {
        id: id.to_string(),
        title: format!("Reviewed frames ({id})"),
        description: "Frames collected from your cameras and reviewed on the Review tab.".into(),
        license: "private: your own camera footage, do not share".into(),
        scored_labels: Some(classes.to_vec()),
        images: frames
            .iter()
            .map(|f| ManifestImage {
                file: f.name.clone(),
                url: None,
                sha256: None,
                size: None,
                width: Some(f.width),
                height: Some(f.height),
                tags: tags(f),
                objects: Some(
                    f.placed(|b| b.is_training() || b.is_ignore())
                        .into_iter()
                        .map(|b| ManifestObject {
                            label: b.label.clone(),
                            bbox: clamp_box(b.bbox, f.width, f.height),
                            ignore: b.is_ignore() || f.under_floor(b),
                            crowd: false,
                        })
                        .collect(),
                ),
            })
            .collect(),
        ..Manifest::default()
    }
}

/// Which frames the benchmark set holds. Held-out frames by default: a score on frames a model
/// trained on says little about the frames it will see next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchSplit {
    All,
    Train,
    #[default]
    Val,
}

/// Which reviewed frames an export keeps (see the module doc).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClassFilter {
    /// Keep frames with a training box of one of these classes; empty = every class.
    pub labels: Vec<String>,
    /// Keep frames confirmed empty (hard-negative images).
    pub include_empty: bool,
}

impl Default for ClassFilter {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            include_empty: true,
        }
    }
}

impl ClassFilter {
    /// Every label must be in `classes` (exports name classes by their id in that list).
    pub fn check(&self, classes: &[String]) -> Result<()> {
        if let Some(l) = self.labels.iter().find(|l| !classes.contains(l)) {
            bail!(
                "labels: '{}' is not in the class list ({}); pick classes from the list, or add \
                 '{}' to the class list on the Dataset tab first",
                l.escape_debug(),
                classes.join(", "),
                l.escape_debug()
            );
        }
        Ok(())
    }

    /// Whether a frame with `labels` is exported: a frame confirmed empty when `include_empty`
    /// is set; any other frame when no class is named, else only with a training box of a
    /// named class.
    pub fn keeps(&self, labels: &LabelFile) -> bool {
        if labels.empty_confirmed {
            return self.include_empty;
        }
        self.labels.is_empty() || labels.training().any(|b| self.labels.contains(&b.label))
    }
}

/// An export request (`POST /v1/review/export`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExportRequest {
    pub formats: Vec<Format>,
    pub out_dir: PathBuf,
    pub split: SplitConfig,
    pub benchmark_split: BenchSplit,
    /// X-AnyLabeling only: also the queue's unreviewed frames, pre-filled and unchecked.
    pub include_unreviewed: bool,
    /// `collect.hidden_classes`: the pre-fill of those unreviewed frames leaves them out
    /// ([`super::prefill`]).
    pub hidden_classes: Vec<String>,
    /// `collect.size_floor`: training boxes under it are exported as ignore regions
    /// ([`ExportFrame::under_floor`]; the X-AnyLabeling files keep them as they are). None = off,
    /// the export as before.
    pub size_floor: Option<SizeFloor>,
    /// The camera days held out for good ([`heldout`]): read, and grown by an export of the
    /// default split. None = the split alone, the list untouched (every export but the Dataset
    /// tab's). Applied by [`run`] / [`run_filtered`] / [`run_filtered_with`], never by
    /// [`run_selected`], whose frames carry their split.
    pub heldout: Option<HeldoutPolicy>,
}

/// The training boxes an export's size floor made ignore regions ([`ExportReport::under_floor`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnderFloor {
    /// Per class, on frames of the `train` split.
    pub train: BTreeMap<String, usize>,
    /// Per class, on held-out (`val`) frames: ignore regions there, detections on them do not
    /// count.
    pub val: BTreeMap<String, usize>,
    /// Frames left with no label but ignore regions because of the floor: a training run leaves
    /// them out, they are no "nothing here" frame.
    pub frames_only: usize,
}

/// What [`run`] wrote.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ExportReport {
    pub out_dir: PathBuf,
    pub formats: Vec<Format>,
    pub frames: usize,
    pub train: usize,
    pub val: usize,
    pub empty_confirmed: usize,
    pub per_class: BTreeMap<String, u64>,
    /// Rejected model / auto boxes listed in `hard_negatives.json` (duplicates and too-small boxes
    /// sit on real objects and are left out, [`LabelFile::hard_negatives`]).
    pub hard_negative_boxes: usize,
    /// Ignore regions listed in [`IGNORE_FILE`].
    pub ignore_boxes: usize,
    /// Groups and their split.
    pub splits: Vec<GroupSplit>,
    /// Format -> path written.
    pub paths: BTreeMap<String, PathBuf>,
    /// `dir:<path>` for `benchmark --dataset` (benchmark format only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_dataset: Option<String>,
    /// Unreviewed frames added to the X-AnyLabeling folder.
    pub unreviewed_frames: usize,
    pub warnings: Vec<String>,
    pub classes: Vec<String>,
    /// The class filter applied.
    pub class_filter: ClassFilter,
    /// Reviewed frames the class filter left out.
    pub filtered_out: usize,
    /// Reviewed frames left out because their image or sidecar is gone.
    pub lost_frames: usize,
    /// Folders of the dataset that could not be listed: their frames are not in this export.
    pub unreadable_folders: usize,
    /// The size floor applied ([`ExportRequest::size_floor`]); absent when off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
    /// The training boxes it made ignore regions, per split and class (in `ignore_boxes` too, not
    /// in `per_class`); absent when off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub under_floor: Option<UnderFloor>,
    /// What the export did with the camera days held out for good ([`ExportRequest::heldout`]);
    /// absent without.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heldout: Option<HeldoutReport>,
}

/// One (camera, date) group of an export.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupSplit {
    pub camera: String,
    pub date: String,
    pub split: Split,
    pub frames: usize,
}

/// The reviewed frames of `dir` with their split. Frames whose image or sidecar is gone, whose
/// files do not parse, or whose training labels are not all in `classes` are left out with a
/// warning.
pub fn reviewed_frames(
    dir: &Path,
    classes: &[String],
    split: &SplitConfig,
) -> (Vec<ExportFrame>, Vec<String>) {
    let s = reviewed_frames_filtered(dir, classes, split, &ClassFilter::default());
    (s.frames, s.warnings)
}

/// What [`reviewed_frames_filtered`] selected.
#[derive(Debug, Default)]
pub struct Selection {
    pub frames: Vec<ExportFrame>,
    pub warnings: Vec<String>,
    /// [`Self::warnings`] without the sentences of the frames counted in [`Self::pending`] and
    /// [`Self::off_class_list`], in the same order: for a reader that counts those frames apart
    /// (the dataset check). An unreadable frame's sentence stays: it names the error.
    pub other_warnings: Vec<String>,
    /// Reviewed frames left out because they still hold a pending box (each named in
    /// [`Self::warnings`]).
    pub pending: usize,
    /// Reviewed frames left out because a training label is not in the class list (each named).
    pub off_class_list: usize,
    /// Reviewed frames left out because their label file or sidecar does not read (each named).
    pub unreadable_frames: usize,
    /// Reviewed frames the filter left out.
    pub filtered_out: usize,
    /// [`ExportReport::lost_frames`].
    pub lost_frames: usize,
    /// [`ExportReport::unreadable_folders`].
    pub unreadable_folders: usize,
    /// The label file [`super::revision`] of every selected frame, by its [`flat_name`] (read
    /// with the labels, from the same bytes).
    pub revisions: BTreeMap<String, String>,
    /// Reviewed frames whose label file changed while the walk read them ([`reviewed_snapshot`]
    /// only): left out, never read half-way.
    pub changed: usize,
    /// The [`flat_name`]s of the frames this walk could not judge: changed while read, a label
    /// file or sidecar that does not read, an image or sidecar gone from a frame that may be
    /// reviewed. Whether they are still reviewed is unknown (a reader that compares with an
    /// earlier walk must not take them as gone).
    pub unsure: BTreeSet<String>,
    /// The folders of [`Self::unreadable_folders`]: their frames are unknown too.
    pub unreadable_dirs: Vec<PathBuf>,
}

/// Whether the label file at `path` is, or may be, of a reviewed frame.
fn may_be_reviewed(path: &Path) -> bool {
    super::read(path).map_or(true, |l| l.is_some_and(|l| l.status == Status::Reviewed))
}

/// The warning of `lost`, frames that lost a file.
fn lost_warning(lost: &[String]) -> String {
    let more = lost.len().saturating_sub(super::bulk::LISTED);
    let tail = if more > 0 {
        format!(" and {more} more")
    } else {
        String::new()
    };
    format!(
        "{} reviewed frame(s) left out because the image or sidecar is gone (deleted by hand, or \
         a folder moved?): {}{tail}. Restore the files, or review other frames, for them to count",
        lost.len(),
        lost.iter()
            .take(super::bulk::LISTED)
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// What the walk of [`reviewed_frames_filtered`] read of one label file.
enum ReadFrame {
    /// The image or sidecar is gone from a frame that may have been reviewed.
    Lost,
    /// The label file or sidecar does not read: why.
    Failed(String),
    Reviewed(Box<(Frame, LabelFile, Record, String)>),
    /// The label file changed while it was read (its revision before and after the sidecar).
    Changed,
}

/// The reviewed frame of the label file at `path`, with its revision; None when the frame is not
/// reviewed, or is gone and was not. `recheck`: the label file's revision is read again after the
/// sidecar, and a frame whose file changed meanwhile is [`ReadFrame::Changed`].
fn read_reviewed(
    walker: &mut Walker<'_>,
    rel: &str,
    path: &Path,
    recheck: bool,
) -> Option<ReadFrame> {
    let Some(frame) = Frame::resolve_in(walker, rel) else {
        return may_be_reviewed(path).then_some(ReadFrame::Lost);
    };
    let (labels, revision) = match super::read_with_revision(&frame.labels) {
        Ok((Some(l), rev)) if l.status == Status::Reviewed => (l, rev),
        Ok(_) => return None,
        Err(e) => return Some(ReadFrame::Failed(format!("{e:#}"))),
    };
    let rec = match frame.record() {
        Ok(rec) => rec,
        Err(e) => return Some(ReadFrame::Failed(format!("{e:#}"))),
    };
    if recheck && super::revision(&frame.labels).ok().as_ref() != Some(&revision) {
        return Some(ReadFrame::Changed);
    }
    Some(ReadFrame::Reviewed(Box::new((
        frame, labels, rec, revision,
    ))))
}

/// [`reviewed_frames`] that `filter` keeps, split after filtering (so both sides hold kept
/// frames); also how many the filter left out and what could not be looked at.
pub fn reviewed_frames_filtered(
    dir: &Path,
    classes: &[String],
    split: &SplitConfig,
    filter: &ClassFilter,
) -> Selection {
    select_with(dir, classes, split, filter, false, &|_| {}, &|| false)
}

/// [`reviewed_frames_filtered`] for a reader that keeps what it read beside the label files (the
/// benchmark's "Your cameras" set and its ground-truth snapshot, `crate::benchmark::cameras`): each
/// label file's revision is read again after its sidecar, and a frame whose file changed
/// meanwhile (a save during the walk) is left out and counted ([`Selection::changed`]). Reads only:
/// no lock is taken and nothing is written.
pub fn reviewed_snapshot(
    dir: &Path,
    classes: &[String],
    split: &SplitConfig,
    filter: &ClassFilter,
) -> Selection {
    select_with(dir, classes, split, filter, true, &|_| {}, &|| false)
}

/// Label files read between two [`ExportStep::Reading`] reports (the walk's threads share one
/// count), and always the last one.
const READ_REPORT_EVERY: usize = 64;

/// [`reviewed_frames_filtered`], telling `progress` how far the walk is. Once `stop` says yes the
/// walk reads no more label files: the selection is then incomplete, and the caller must ask
/// `stop` again before using it.
fn select_with(
    dir: &Path,
    classes: &[String],
    split: &SplitConfig,
    filter: &ClassFilter,
    recheck: bool,
    progress: ProgressFn<'_>,
    stop: StopFn<'_>,
) -> Selection {
    let mut filtered_out = 0;
    let mut changed = 0;
    let (mut pending, mut off_class_list, mut unreadable_frames) = (0, 0, 0);
    let mut unsure = BTreeSet::new();
    let flat_of = |rel: &str| layout::parse_rel(rel).map(|r| flat_name(&r));
    let mut revisions = BTreeMap::new();
    // Every warning, and the same without the counted frames' ([`Selection::other_warnings`]).
    let mut warnings = Vec::new();
    let mut other_warnings = Vec::new();
    let mut frames = Vec::new();
    progress(ExportStep::Listing);
    let scan = super::scan_labels(dir);
    let of = scan.files.len();
    progress(ExportStep::Reading { done: 0, of });
    let read_count = std::sync::atomic::AtomicUsize::new(0);
    for folder in &scan.unreadable {
        warnings.push(format!(
            "{} could not be read, so its frames are not in this export; check the folder's \
             permissions and the drive, then export again",
            folder.display()
        ));
    }
    other_warnings.clone_from(&warnings);
    let mut lost: Vec<String> = scan
        .orphans
        .iter()
        .filter(|(_, path)| may_be_reviewed(path))
        .map(|(rel, _)| rel.clone())
        .collect();
    // Read on the walk's threads, judged here in the list's order, so the warnings keep theirs.
    let walker = Walker::new(dir);
    let read = super::queue::over_days(&scan.files, &|day| {
        let mut walker = walker.clone();
        day.iter()
            .filter_map(|(rel, path)| {
                if stop() {
                    return None;
                }
                let read = read_reviewed(&mut walker, rel, path, recheck);
                let done = read_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done.is_multiple_of(READ_REPORT_EVERY) || done == of {
                    progress(ExportStep::Reading { done, of });
                }
                Some((rel.clone(), read?))
            })
            .collect()
    });
    for (rel, read) in read {
        let (frame, labels, rec, revision) = match read {
            ReadFrame::Lost => {
                lost.push(rel);
                continue;
            }
            ReadFrame::Failed(e) => {
                unsure.extend(flat_of(&rel));
                unreadable_frames += 1;
                let w = format!("{rel}: {e}; left out");
                other_warnings.push(w.clone());
                warnings.push(w);
                continue;
            }
            ReadFrame::Changed => {
                unsure.extend(flat_of(&rel));
                changed += 1;
                continue;
            }
            ReadFrame::Reviewed(read) => *read,
        };
        let unknown: BTreeSet<&str> = labels
            .training()
            .filter(|b| !classes.contains(&b.label))
            .map(|b| b.label.as_str())
            .collect();
        if !unknown.is_empty() {
            off_class_list += 1;
            warnings.push(format!(
                "{rel}: label(s) {} are not in the class list; left out (review it again or add \
                 the class)",
                unknown.into_iter().collect::<Vec<_>>().join(", ")
            ));
            continue;
        }
        if labels.boxes.iter().any(|b| b.state == BoxState::Pending) {
            pending += 1;
            warnings.push(format!("{rel}: reviewed but has pending boxes; left out"));
            continue;
        }
        if !filter.keeps(&labels) {
            filtered_out += 1;
            continue;
        }
        revisions.insert(flat_name(&frame.rel), revision);
        frames.push(ExportFrame {
            name: flat_name(&frame.rel),
            rel: frame.rel,
            image: frame.jpg,
            width: rec.width,
            height: rec.height,
            buckets: rec.buckets,
            split: Split::Train,
            labels,
            floor: None,
        });
    }
    let groups: BTreeSet<Group> = frames
        .iter()
        .map(|f| (f.rel.camera.clone(), f.rel.date.clone()))
        .collect();
    let splits = assign_splits(&groups, split);
    for f in &mut frames {
        f.split = splits
            .get(&(f.rel.camera.clone(), f.rel.date.clone()))
            .copied()
            .unwrap_or(Split::Train);
    }
    unsure.extend(lost.iter().filter_map(|rel| flat_of(rel)));
    if !lost.is_empty() {
        let w = lost_warning(&lost);
        other_warnings.push(w.clone());
        warnings.push(w);
    }
    Selection {
        frames,
        warnings,
        other_warnings,
        pending,
        off_class_list,
        unreadable_frames,
        filtered_out,
        lost_frames: lost.len(),
        unreadable_folders: scan.unreadable.len(),
        revisions,
        changed,
        unsure,
        unreadable_dirs: scan.unreadable,
    }
}

/// What a size floor would do to the reviewed frames of a folder ([`floor_counts`]): the numbers of
/// a settings card, before any export.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct FloorCounts {
    /// Reviewed frames an export would hold.
    pub frames: usize,
    /// Their training boxes per class (accepted or added, no ignore region).
    pub boxes: BTreeMap<String, usize>,
    /// Of `boxes`, those under the floor: ignore regions in the next export.
    pub under: BTreeMap<String, usize>,
    /// Frames holding at least one box under the floor.
    pub frames_with: usize,
    /// Of them, frames left with no label but ignore regions (a training run leaves them out).
    pub frames_only: usize,
    /// What the walk could not read ([`reviewed_frames_filtered`]).
    pub warnings: Vec<String>,
}

/// The training boxes of the reviewed frames of `dir` under `floor`, per class, and the frames they
/// are on: what an export with that floor would turn into ignore regions. Reads only (the walk of
/// an export); slow on a large folder, so call it off the async threads (`server::blocking`).
pub fn floor_counts(dir: &Path, classes: &[String], floor: SizeFloor) -> FloorCounts {
    let sel = reviewed_frames_filtered(
        dir,
        classes,
        &SplitConfig::default(),
        &ClassFilter::default(),
    );
    let mut out = FloorCounts {
        frames: sel.frames.len(),
        warnings: sel.warnings,
        ..FloorCounts::default()
    };
    for f in sel.frames {
        let f = ExportFrame {
            floor: Some(floor),
            ..f
        };
        let mut under = 0usize;
        for b in f.placed(|b| b.is_training() && !b.is_ignore()) {
            *out.boxes.entry(b.label.clone()).or_default() += 1;
            if f.under_floor(b) {
                *out.under.entry(b.label.clone()).or_default() += 1;
                under += 1;
            }
        }
        if under > 0 {
            out.frames_with += 1;
            out.frames_only += usize::from(!f.boxes().iter().any(|b| !b.difficult));
        }
    }
    out
}

#[cfg(test)]
thread_local! {
    /// [`write_file`] fails on a file of this name: a full disk at a chosen point of an export.
    static FAIL_WRITE: std::cell::RefCell<Option<std::ffi::OsString>> =
        const { std::cell::RefCell::new(None) };
    /// [`write_file`] panics on a file of this name: a bug in the middle of an export.
    static PANIC_WRITE: std::cell::RefCell<Option<std::ffi::OsString>> =
        const { std::cell::RefCell::new(None) };
}

/// Hard-link `src` to `dst`, else copy it.
fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if std::fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)
        .map(|_| ())
        .with_context(|| format!("copying {} to {}", src.display(), dst.display()))
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    #[cfg(test)]
    if FAIL_WRITE.with_borrow(|n| path.file_name().is_some_and(|f| Some(f.to_owned()) == *n)) {
        bail!("{}: writing failed (test)", path.display());
    }
    #[cfg(test)]
    if PANIC_WRITE.with_borrow(|n| path.file_name().is_some_and(|f| Some(f.to_owned()) == *n)) {
        panic!("{}: writing panicked (test)", path.display());
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).with_context(|| format!("creating {}", p.display()))?;
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn json_bytes<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    let mut b = serde_json::to_vec_pretty(v)?;
    b.push(b'\n');
    Ok(b)
}

/// `p` with symlinks and `..` resolved through its deepest existing ancestor (the rest of
/// the path appended as given), so a folder that does not exist yet compares too.
fn resolved(p: &Path) -> PathBuf {
    let mut rest = Vec::new();
    let mut cur = p;
    loop {
        if let Ok(c) = std::fs::canonicalize(cur) {
            return rest.iter().rev().fold(c, |acc, n| acc.join(n));
        }
        match (cur.parent(), cur.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                cur = parent;
            }
            _ => return p.to_path_buf(),
        }
    }
}

/// The folder an export to `out` becomes: `out` when it is new or empty, else a new
/// `export-<UTC time>` folder inside it. Never the dataset `dir` or a folder inside it (the export
/// would mix with the frames it reads). Nothing is created.
fn prepare_out(dir: &Path, out: &Path, now: std::time::SystemTime) -> Result<PathBuf> {
    if resolved(out).starts_with(resolved(dir)) {
        let hint = match dir.parent() {
            Some(p) if !p.as_os_str().is_empty() => format!(" (e.g. {})", p.display()),
            _ => String::new(),
        };
        bail!(
            "{} is the dataset folder under review (or inside it); choose a folder outside \
             it{hint}, or leave the field empty for a new one under training/exports",
            out.display()
        );
    }
    match std::fs::read_dir(out).map(|mut rd| rd.next().is_some()) {
        Ok(true) => {
            let t = out.join(format!("export-{}", crate::clock::frame_stamp(now)));
            if t.exists() {
                bail!("{} already exists; export again in a moment", t.display());
            }
            Ok(t)
        }
        Ok(false) => Ok(out.to_path_buf()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(out.to_path_buf()),
        Err(e) => Err(e).with_context(|| format!("reading {}", out.display())),
    }
}

/// How many times publishing an export is tried while Windows refuses it as in use, and how far
/// apart: thousands of new files are scanned by Defender / indexed for a few seconds after they are
/// written, and a folder holding one of them open cannot be renamed meanwhile ("Access is denied",
/// which once failed a dataset check's last fold).
const PUBLISH_ATTEMPTS: u32 = 40;
const PUBLISH_PAUSE: std::time::Duration = std::time::Duration::from_millis(250);

/// `rename` retried for [`PUBLISH_ATTEMPTS`] x [`PUBLISH_PAUSE`] on a Windows sharing violation.
/// Windows also answers "Access is denied" when `to` is taken, which no wait cures: then one try.
fn rename_patiently(from: &Path, to: &Path) -> std::io::Result<()> {
    if std::fs::symlink_metadata(to).is_ok() {
        return std::fs::rename(from, to);
    }
    crate::private_dir::rename_retrying(from, to, PUBLISH_ATTEMPTS, PUBLISH_PAUSE)
}

/// Files published last from an existing folder ([`Staging::publish`]), `classes.txt` and
/// `data.yaml` being what makes a folder look like an export.
const PUBLISHED_LAST: [&str; 3] = ["classes.txt", "data.yaml", "export.json"];

/// Where an export is built until it is whole. A new target is built in a hidden folder beside
/// it, inside a subfolder so that nothing looking for exports one level down can take it for one
/// (a crash leaves it there), and appears by one rename: the sibling is on the target's
/// volume, as the target would have been created in the same parent. An existing empty target may
/// be a mount point, or sit in a parent this service cannot write, and a user's folder keeps its
/// owner and permissions: it is built in a hidden folder inside the target and moved up file by
/// file, the files that make a folder look like an export last.
///
/// The scratch folder is held busy ([`crate::storage::Busy`], its marker beside it) from before it
/// is made until it is dealt with, so a storage sweep after a crash or a kill removes it
/// ([`crate::storage::sweep`]: under `<training dir>/exports` by its name, elsewhere through the
/// record [`crate::storage::sweep::write_export_record`] writes when the caller names a pending
/// folder). Removed, or kept on purpose by a failed publish: the marker goes (a kept export then
/// reads as ended, never swept). A removal that fails keeps the marker, unlocked, for the sweep.
#[derive(Debug)]
struct Staging {
    target: PathBuf,
    /// The folder the export is written to.
    build: PathBuf,
    /// What remains to remove once `build` is published.
    scratch: PathBuf,
    /// `target` is an existing folder ([`Self::publish`] moves the files).
    inside: bool,
    /// Published, kept on purpose by a failed publish, or discarded: the drop leaves it alone.
    settled: std::cell::Cell<bool>,
    /// The hold on `scratch` until it is dealt with ([`Self::release`]).
    busy: std::cell::RefCell<Option<crate::storage::Busy>>,
    /// The record naming `scratch` in the pending folder, when one was written.
    record: Option<PathBuf>,
}

impl Staging {
    /// The staging of `target`, unrecorded (a caller whose scratch folders a sweep finds by
    /// their place: under `<training dir>/exports`, or inside a job's work folder).
    #[cfg(test)]
    fn new(target: &Path) -> Result<Self> {
        Self::recorded(target, None)
    }

    /// The staging of `target`; with `pending`, a record of its scratch folder is written there
    /// first ([`crate::storage::sweep::EXPORTS_PENDING_DIR`]).
    fn recorded(target: &Path, pending: Option<&Path>) -> Result<Self> {
        let id = uuid::Uuid::new_v4().simple();
        let inside = target.is_dir();
        let scratch = if inside {
            target.join(format!(".building-{id}"))
        } else {
            let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
                bail!(
                    "{} cannot be created; choose another folder",
                    target.display()
                );
            };
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
            parent.join(format!(".{}.building-{id}", name.to_string_lossy()))
        };
        let build = if inside {
            scratch.clone()
        } else {
            scratch.join("export")
        };
        // Recorded, then held, then made: a crash at any point leaves nothing a sweep cannot
        // tell from a running export.
        let record = pending.and_then(|dir| {
            crate::storage::sweep::write_export_record(dir, &scratch)
                .map_err(|e| {
                    tracing::warn!(
                        "export: could not note {} in {} ({e}); if the service stops during this \
                         export, delete that folder by hand",
                        scratch.display(),
                        dir.display()
                    );
                })
                .ok()
        });
        let staging = Self {
            target: target.to_path_buf(),
            busy: std::cell::RefCell::new(Some(crate::storage::Busy::hold(&scratch))),
            build,
            scratch,
            inside,
            settled: std::cell::Cell::new(false),
            record,
        };
        if let Err(e) = std::fs::create_dir_all(&staging.build) {
            staging.discard();
            return Err(e).with_context(|| format!("creating {}", staging.build.display()));
        }
        Ok(staging)
    }

    /// The scratch folder is dealt with: `gone` (removed, or kept on purpose by a failed
    /// publish) lets its marker and record go; otherwise the marker stays, unlocked, so the next
    /// storage sweep sees a crashed export and removes the folder.
    fn release(&self, gone: bool) {
        let Some(busy) = self.busy.borrow_mut().take() else {
            return;
        };
        if !gone {
            busy.abandon();
            return;
        }
        // A hold dropped while a panic unwinds keeps its marker (the job did not end in order),
        // but this scratch folder is gone: nothing is left for a sweep. Deleted while still
        // locked, as the hold's own drop does.
        if std::thread::panicking()
            && let Err(e) = std::fs::remove_file(busy.marker())
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("export: removing {}: {e}", busy.marker().display());
        }
        drop(busy);
        if let Some(r) = &self.record
            && let Err(e) = std::fs::remove_file(r)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::debug!("export: removing the record {}: {e}", r.display());
        }
    }

    /// Give up: nothing of the export stays (the dataset is its source).
    fn discard(&self) {
        self.settled.set(true);
        let gone = match std::fs::remove_dir_all(&self.scratch) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => {
                tracing::warn!(
                    "export: could not remove the unfinished {}: {e}; the next storage sweep \
                     removes it",
                    self.scratch.display()
                );
                false
            }
        };
        self.release(gone);
    }

    fn move_up(&self) -> std::io::Result<()> {
        let mut names: Vec<std::ffi::OsString> = std::fs::read_dir(&self.build)?
            .map(|e| e.map(|e| e.file_name()))
            .collect::<std::io::Result<_>>()?;
        let rank = |n: &std::ffi::OsString| {
            PUBLISHED_LAST
                .iter()
                .position(|l| n == l)
                .map_or(0, |at| at + 1)
        };
        names.sort_by_key(rank);
        for name in names {
            rename_patiently(&self.build.join(&name), &self.target.join(&name))?;
        }
        Ok(())
    }

    /// The finished export takes its place. A failure keeps what was built (it is complete, and
    /// its `data.yaml` already names the target) and says where.
    fn publish(&self) -> Result<()> {
        // Whatever happens now, what was built is either in place or kept and named.
        self.settled.set(true);
        let moved = if self.inside {
            self.move_up()
        } else {
            rename_patiently(&self.build, &self.target)
        };
        if moved.is_err() {
            // Kept on purpose: its marker goes, so no sweep ever takes it.
            self.release(true);
        }
        moved.with_context(|| {
            format!(
                "the export is complete in {} but could not be moved to {} (the folder is in use, \
                 or something else was created there meanwhile); move it there by hand, or \
                 export again",
                self.build.display(),
                self.target.display()
            )
        })?;
        let gone = match std::fs::remove_dir_all(&self.scratch) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    "export: could not remove the empty {}: {e}; the next storage sweep removes it",
                    self.scratch.display()
                );
                false
            }
        };
        self.release(gone);
        Ok(())
    }
}

/// An export that never got to [`Staging::publish`] or [`Staging::discard`] (a panic, or an
/// early return) goes like a failed one.
impl Drop for Staging {
    fn drop(&mut self) {
        if !self.settled.get() {
            self.discard();
        }
    }
}

/// Export the reviewed frames of the dataset `dir` (see the module doc), every class.
pub fn run(dir: &Path, classes: &[String], req: &ExportRequest) -> Result<ExportReport> {
    run_filtered(dir, classes, req, &ClassFilter::default())
}

/// [`run`] keeping only the frames `filter` keeps ([`ClassFilter::keeps`]; the queue's
/// unreviewed X-AnyLabeling frames by the boxes a save would accept).
pub fn run_filtered(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
) -> Result<ExportReport> {
    run_filtered_with(dir, classes, req, filter, &|_| {})
}

/// Where a running export is ([`run_filtered_with`]): the steps in order. A step without a count
/// is one the export cannot count cheaply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportStep {
    /// Listing the dataset's label files.
    Listing,
    /// `done` of the `of` label files read (to find the reviewed frames).
    Reading { done: usize, of: usize },
    /// Reading the review queue's unreviewed frames (X-AnyLabeling with `include_unreviewed`).
    Queue,
    /// `done` of the `of` files written: every image placed (hard link or copy) and every
    /// per-frame label file (YOLO txt, X-AnyLabeling JSON). The COCO, manifest and list files are
    /// written in this step too, uncounted (one file each).
    Writing { done: usize, of: usize },
    /// Moving the finished export into place ([`Staging::publish`]).
    Publishing,
}

/// What an export tells about its progress: called on the export's thread, and from the walk's
/// threads while label files are read, so it must be `Sync` and cheap.
pub type ProgressFn<'a> = &'a (dyn Fn(ExportStep) + Sync);

/// [`run_filtered`], telling `progress` where it is ([`ExportStep`]).
pub fn run_filtered_with(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    progress: ProgressFn<'_>,
) -> Result<ExportReport> {
    run_filtered_until(dir, classes, req, filter, progress, &|| false)
}

/// [`run_filtered_with`], asking `stop` between two label files read and two files written: once
/// it says yes the export ends with [`Stopped`] and nothing of it stays (the held-out list is
/// left alone when the stop comes before the split).
pub fn run_filtered_until(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    progress: ProgressFn<'_>,
    stop: StopFn<'_>,
) -> Result<ExportReport> {
    run_filtered_recorded(dir, classes, req, filter, (progress, stop), None)
}

/// [`run_filtered_until`] (`hooks` = its progress and stop), first writing a record of the
/// export's scratch folder into `pending` when given
/// ([`crate::storage::sweep::EXPORTS_PENDING_DIR`] of the training folder): an export into a
/// folder of the user's choice, which a sweep after a crash finds only through it.
pub fn run_filtered_recorded(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    hooks: (ProgressFn<'_>, StopFn<'_>),
    pending: Option<&Path>,
) -> Result<ExportReport> {
    let (progress, stop) = hooks;
    if req.formats.is_empty() {
        bail!("no export format given (yolo, coco, xanylabeling, benchmark)");
    }
    filter.check(classes)?;
    let mut selected = select_with(dir, classes, &req.split, filter, false, progress, stop);
    if stop() {
        return Err(Stopped.into());
    }
    let held = match &req.heldout {
        Some(policy) => Some(frozen_split(dir, &mut selected, &req.split, policy)?),
        None => None,
    };
    run_selected_with(
        dir,
        classes,
        req,
        filter,
        selected,
        held,
        (progress, stop),
        pending,
    )
}

/// The split of `selected` by [`heldout::apply`] (the listed days held out, the list grown by an
/// export of the default split): after the walk, under the list's lock for the decision only.
fn frozen_split(
    dir: &Path,
    selected: &mut Selection,
    split: &SplitConfig,
    policy: &HeldoutPolicy,
) -> Result<HeldoutReport> {
    let groups: BTreeSet<Group> = selected
        .frames
        .iter()
        .map(|f| (f.rel.camera.clone(), f.rel.date.clone()))
        .collect();
    let (splits, report) =
        heldout::apply(dir, &groups, split, policy, std::time::SystemTime::now())?;
    for f in &mut selected.frames {
        f.split = splits
            .get(&(f.rel.camera.clone(), f.rel.date.clone()))
            .copied()
            .unwrap_or(Split::Train);
    }
    if policy.seen.is_none() && heldout::is_default_split(split) {
        selected.warnings.push(
            "the frame lists of past training runs (<training dir>/<run>/data/train.txt, \
             valid.txt) could not all be read, so no camera day was added for a camera with too \
             few held-out days: one of them may have been trained on. Check the training folder, \
             then export again"
                .into(),
        );
    }
    Ok(report)
}

/// [`run_filtered`] of frames selected already ([`reviewed_frames_filtered`]), each with the split
/// it carries: a caller that needs one label snapshot for several exports, or splits of its own
/// (the dataset check, `crate::training::refine`, holds out one fold of camera days per export),
/// selects once and exports a copy each time. `req.split` is not applied here; `filter` is only
/// recorded (and checks the unreviewed X-AnyLabeling frames). Images are hard links (else
/// copies) of the dataset's: nothing here writes to them.
pub fn run_selected(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    selected: Selection,
) -> Result<ExportReport> {
    run_selected_with(
        dir,
        classes,
        req,
        filter,
        selected,
        None,
        (&|_| {}, &|| false),
        None,
    )
}

/// [`run_selected`], telling `progress` where it is and asking `stop` between two files
/// (`hooks`); `heldout` is what [`frozen_split`] did; `pending` is where the scratch folder is
/// recorded ([`Staging::recorded`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the export's inputs, what the split did, the caller's progress and stop, and where \
              its scratch folder is recorded"
)]
fn run_selected_with(
    dir: &Path,
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    selected: Selection,
    heldout: Option<HeldoutReport>,
    hooks: (ProgressFn<'_>, StopFn<'_>),
    pending: Option<&Path>,
) -> Result<ExportReport> {
    let (progress, stop) = hooks;
    if req.formats.is_empty() {
        bail!("no export format given (yolo, coco, xanylabeling, benchmark)");
    }
    filter.check(classes)?;
    let mut selected = selected;
    let unreviewed: Vec<(Frame, LabelFile, u32, u32)> =
        if req.include_unreviewed && req.formats.contains(&Format::XAnyLabeling) {
            progress(ExportStep::Queue);
            unreviewed_queue_frames(dir, classes, &req.hidden_classes, &mut selected.warnings)
                .into_iter()
                .filter(|(_, labels, _, _)| filter.keeps(&preview(labels)))
                .collect()
        } else {
            Vec::new()
        };
    if selected.frames.is_empty() && unreviewed.is_empty() {
        if selected.lost_frames + selected.unreadable_folders > 0 {
            bail!(
                "nothing could be exported from {}: {}",
                dir.display(),
                selected.warnings.join("; ")
            );
        }
        if selected.filtered_out > 0 {
            bail!(
                "no reviewed frame of {} in {} ({} reviewed frame(s) left out by the class \
                 filter); clear the class filter or review frames of these classes first",
                filter.labels.join(", "),
                dir.display(),
                selected.filtered_out
            );
        }
        bail!(
            "no reviewed frame to export in {}; review frames on the Review tab first (save \
             them as reviewed)",
            dir.display()
        );
    }
    if stop() {
        return Err(Stopped.into());
    }
    let target = prepare_out(dir, &req.out_dir, std::time::SystemTime::now())?;
    let staging = Staging::recorded(&target, pending)?;
    let report = build_export(
        selected,
        &unreviewed,
        classes,
        req,
        filter,
        (&target, &staging.build),
        heldout,
        (progress, stop),
    );
    match report {
        Ok(r) => {
            progress(ExportStep::Publishing);
            staging.publish().map(|()| r)
        }
        Err(e) => {
            staging.discard();
            Err(e)
        }
    }
}

/// Write the export that will be at `out` into the folder `build` ([`Staging`]). Every path a
/// reader of the export sees names `out`.
#[allow(
    clippy::too_many_arguments,
    reason = "the selection, what goes with it, the two folders, the held-out list's report and \
              the progress report"
)]
fn build_export(
    selected: Selection,
    unreviewed: &[(Frame, LabelFile, u32, u32)],
    classes: &[String],
    req: &ExportRequest,
    filter: &ClassFilter,
    (out, build): (&Path, &Path),
    heldout: Option<HeldoutReport>,
    (progress, stop): (ProgressFn<'_>, StopFn<'_>),
) -> Result<ExportReport> {
    let Selection {
        mut frames,
        mut warnings,
        filtered_out,
        lost_frames,
        unreadable_folders,
        ..
    } = selected;
    for f in &mut frames {
        f.floor = req.size_floor;
    }
    let mut r = ExportReport {
        out_dir: out.to_path_buf(),
        formats: req.formats.clone(),
        frames: frames.len(),
        classes: classes.to_vec(),
        unreviewed_frames: unreviewed.len(),
        class_filter: filter.clone(),
        filtered_out,
        lost_frames,
        unreadable_folders,
        size_floor: req.size_floor,
        under_floor: req.size_floor.map(|_| UnderFloor::default()),
        heldout,
        ..ExportReport::default()
    };
    count_frames(&mut r, &frames);
    let writers = formats::plan(req, !frames.is_empty());
    let mut written = Written {
        done: 0,
        of: written_files(&writers, &frames, unreviewed.len(), req.benchmark_split),
        progress,
        stop,
    };
    written.report();
    let dirs = formats::Dirs { out, build };
    for w in writers {
        match w {
            Writer::Images => formats::write_images(dirs, &frames, &mut written)?,
            Writer::Yolo => {
                formats::write_yolo(
                    dirs,
                    &frames,
                    classes,
                    &mut written,
                    (&mut r, &mut warnings),
                )?;
            }
            Writer::Coco => formats::write_coco(dirs, &frames, classes, &mut r)?,
            Writer::XAnyLabeling => {
                formats::write_xany(dirs, (&frames, unreviewed), classes, &mut written, &mut r)?;
            }
            Writer::Benchmark => formats::write_benchmark(
                dirs,
                (&frames, req.benchmark_split),
                classes,
                &mut written,
                (&mut r, &mut warnings),
            )?,
        }
    }
    formats::write_extras(dirs, &frames, req, &mut r)?;
    r.warnings = warnings;
    write_file(&build.join("export.json"), &json_bytes(&r)?)?;
    Ok(r)
}

/// The report's statistics of the exported `frames`: per split, confirmed empty, per class, under
/// the floor (when `r` has one), and the split of every camera day.
fn count_frames(r: &mut ExportReport, frames: &[ExportFrame]) {
    let mut groups: BTreeMap<Group, (Split, usize)> = BTreeMap::new();
    for f in frames {
        match f.split {
            Split::Train => r.train += 1,
            Split::Val => r.val += 1,
        }
        r.empty_confirmed += usize::from(f.labels.empty_confirmed);
        for b in f.boxes() {
            *r.per_class.entry(b.label.clone()).or_default() += 1;
        }
        if let Some(u) = &mut r.under_floor {
            let under = f.placed(|b| f.under_floor(b));
            let side = match f.split {
                Split::Train => &mut u.train,
                Split::Val => &mut u.val,
            };
            for b in &under {
                *side.entry(b.label.clone()).or_default() += 1;
            }
            let labelled = f.boxes().iter().any(|b| !b.difficult);
            u.frames_only += usize::from(!under.is_empty() && !labelled);
        }
        groups
            .entry((f.rel.camera.clone(), f.rel.date.clone()))
            .or_insert((f.split, 0))
            .1 += 1;
    }
    r.splits = groups
        .into_iter()
        .map(|((camera, date), (split, frames))| GroupSplit {
            camera,
            date,
            split,
            frames,
        })
        .collect();
}

/// The files [`build_export`] counts in [`ExportStep::Writing`]: what the `writers` it runs
/// write ([`Writer::files`]).
fn written_files(
    writers: &[Writer],
    frames: &[ExportFrame],
    unreviewed: usize,
    bench: BenchSplit,
) -> usize {
    writers
        .iter()
        .map(|w| w.files(frames, unreviewed, bench))
        .sum()
}

/// The count of [`ExportStep::Writing`], and where the caller's stop is asked.
struct Written<'a> {
    done: usize,
    of: usize,
    progress: ProgressFn<'a>,
    stop: StopFn<'a>,
}

impl Written<'_> {
    fn report(&self) {
        (self.progress)(ExportStep::Writing {
            done: self.done,
            of: self.of,
        });
    }

    /// One more file written; [`Stopped`] when the caller asks to stop.
    fn one(&mut self) -> Result<()> {
        self.done += 1;
        self.report();
        if (self.stop)() {
            return Err(Stopped.into());
        }
        Ok(())
    }
}

/// An unreviewed frame's labels as a save could leave them: pending boxes resolved, training
/// boxes only, without the flagged boxes a reviewer has yet to decide.
fn preview(labels: &LabelFile) -> LabelFile {
    LabelFile {
        boxes: labels
            .boxes
            .iter()
            .filter(|b| !super::needs_decision(b))
            .map(|b| LabelBox {
                state: super::resolve_pending(b),
                ..b.clone()
            })
            .filter(LabelBox::is_training)
            .collect(),
        ..labels.clone()
    }
}

/// The review queue's frames that are not reviewed or skipped, with their labels (saved or
/// pre-filled with the boxes of `classes` not in `hidden`) and size.
fn unreviewed_queue_frames(
    dir: &Path,
    classes: &[String],
    hidden: &[String],
    warnings: &mut Vec<String>,
) -> Vec<(Frame, LabelFile, u32, u32)> {
    let mut out = Vec::new();
    let mut walker = Walker::new(dir);
    for c in super::queue::candidates(dir, super::queue::QueueSource::Queue) {
        let Some(frame) = Frame::resolve_in(&mut walker, &c.rel) else {
            continue;
        };
        let (rec, labels) = match (frame.record(), frame.labels()) {
            (Ok(r), Ok(l)) => (r, l),
            (Err(e), _) | (_, Err(e)) => {
                warnings.push(format!("{}: {e:#}; left out", c.rel));
                continue;
            }
        };
        let labels = match labels {
            Some(l) if l.status != Status::Unreviewed => continue,
            Some(l) => l,
            None => super::prefill(&rec, classes, hidden),
        };
        out.push((frame, labels, rec.width, rec.height));
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::Source;
    use super::*;

    #[test]
    fn a_preview_leaves_out_boxes_a_reviewer_has_yet_to_decide() {
        let mut flagged = super::super::tests_support::auto_box("person");
        flagged.auto.triage = Some(super::super::Triage::Review);
        let plain = super::super::tests_support::auto_box("car");
        let p = preview(&LabelFile {
            boxes: vec![flagged, plain],
            ..LabelFile::default()
        });
        assert_eq!(p.boxes.len(), 1);
        assert_eq!(p.boxes[0].label, "car");
        assert_eq!(p.boxes[0].state, BoxState::Accepted);
    }

    #[test]
    fn only_frames_with_nothing_but_parked_boxes_are_listed() {
        let parked = |bbox: [f32; 4]| {
            let mut b = lb("car", bbox, BoxState::Accepted);
            b.reason = Some(spots::PARKED_REASON.into());
            b.auto.parked = Some(super::super::ParkedMark {
                since: "2026-10-01".into(),
                bbox: [10.0, 10.0, 50.0, 50.0],
            });
            b
        };
        let f = frame(vec![parked([11.0, 11.0, 50.0, 50.0])], false);
        let key = "cam/200x100/car/10,10,50,50".to_string();
        assert_eq!(f.parked_spots(), std::slice::from_ref(&key));
        // Two boxes at one spot are one spot.
        let two = frame(
            vec![
                parked([11.0, 11.0, 50.0, 50.0]),
                parked([10.0, 10.0, 50.0, 50.0]),
            ],
            false,
        );
        assert_eq!(two.parked_spots(), std::slice::from_ref(&key));
        // Another label on the frame, and a box a reviewer moved or renamed.
        let mixed = vec![
            parked([1.0, 1.0, 9.0, 9.0]),
            lb("car", [20.0, 20.0, 60.0, 60.0], BoxState::Accepted),
        ];
        assert!(frame(mixed, false).parked_spots().is_empty());
        let mut moved = parked([1.0, 1.0, 9.0, 9.0]);
        moved.orig_bbox = Some([2.0, 2.0, 9.0, 9.0]);
        assert!(frame(vec![moved], false).parked_spots().is_empty());
        let mut renamed = parked([1.0, 1.0, 9.0, 9.0]);
        renamed.model_label = Some("truck".into());
        assert!(frame(vec![renamed], false).parked_spots().is_empty());
        // A rejected box is no training box, and an empty frame has no spot.
        let side = lb("car", [20.0, 20.0, 60.0, 60.0], BoxState::Rejected);
        assert_eq!(
            frame(vec![parked([11.0, 11.0, 50.0, 50.0]), side], false).parked_spots(),
            std::slice::from_ref(&key)
        );
        assert!(frame(Vec::new(), true).parked_spots().is_empty());
        let file = ignore_file(&[f]);
        assert_eq!(file.parked.values().next(), Some(&vec![key]));
        assert!(file.images.is_empty());
    }

    fn g(c: &str, d: &str) -> Group {
        (c.to_string(), d.to_string())
    }

    fn frame(boxes: Vec<LabelBox>, empty: bool) -> ExportFrame {
        ExportFrame {
            rel: layout::parse_rel("ir/cam/2026-10-22/20261022T010203004Z_abcd.jpg").unwrap(),
            name: "ir.cam.2026-10-22.20261022T010203004Z_abcd.jpg".into(),
            image: PathBuf::from("x.jpg"),
            width: 200,
            height: 100,
            buckets: vec![Bucket::Suppressed],
            split: Split::Val,
            labels: LabelFile {
                status: Status::Reviewed,
                empty_confirmed: empty,
                boxes,
                ..LabelFile::default()
            },
            floor: None,
        }
    }

    fn lb(label: &str, b: [f32; 4], state: BoxState) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: b,
            source: Source::Model,
            score: Some(0.9),
            state,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    fn classes() -> Vec<String> {
        vec!["person".into(), "dog".into(), "raccoon".into()]
    }

    #[test]
    fn yolo_lines_are_normalised_and_clamped() {
        let f = frame(
            vec![
                lb("dog", [20.0, 10.0, 60.0, 50.0], BoxState::Accepted),
                lb("person", [150.0, 50.0, 260.0, 130.0], BoxState::Added),
                lb("person", [0.0, 0.0, 10.0, 10.0], BoxState::Rejected),
                lb("raccoon", [250.0, 0.0, 300.0, 10.0], BoxState::Accepted),
            ],
            false,
        );
        let txt = yolo_txt(&f, &classes()).unwrap();
        assert_eq!(
            txt, "1 0.200000 0.300000 0.200000 0.400000\n0 0.875000 0.750000 0.250000 0.500000\n",
            "rejected and outside boxes left out, the clipped box clamped"
        );
        let bad = frame(
            vec![lb("unicorn", [0.0, 0.0, 5.0, 5.0], BoxState::Accepted)],
            false,
        );
        assert!(yolo_txt(&bad, &classes()).is_err());
        assert_eq!(yolo_txt(&frame(vec![], true), &classes()).unwrap(), "");
    }

    #[test]
    fn coco_ids_and_boxes() {
        let a = frame(
            vec![
                lb("raccoon", [10.1, 20.2, 30.3, 40.4], BoxState::Accepted),
                lb("dog", [0.0, 0.0, 5.0, 5.0], BoxState::Rejected),
            ],
            false,
        );
        let mut b = frame(
            vec![lb("person", [0.0, 0.0, 300.0, 50.0], BoxState::Added)],
            false,
        );
        b.name = "second.jpg".into();
        let empty = frame(vec![], true);
        let v = coco_json(&[&a, &b, &empty], &classes()).unwrap();
        assert_eq!(
            v["categories"][0],
            serde_json::json!({"id": 1, "name": "person", "supercategory": "object"})
        );
        assert_eq!(v["categories"][2]["id"], 3);
        assert_eq!(v["images"][1]["id"], 2);
        assert_eq!(v["images"][1]["file_name"], "second.jpg");
        assert_eq!(v["images"].as_array().unwrap().len(), 3);
        let anns = v["annotations"].as_array().unwrap();
        assert_eq!(anns.len(), 2);
        assert_eq!(anns[0]["id"], 1);
        assert_eq!(anns[0]["image_id"], 1);
        assert_eq!(anns[0]["category_id"], 3);
        assert_eq!(anns[0]["bbox"], serde_json::json!([10.1, 20.2, 20.2, 20.2]));
        assert_eq!(anns[0]["area"], 408.04);
        assert_eq!(anns[1]["id"], 2);
        assert_eq!(anns[1]["image_id"], 2);
        assert_eq!(
            anns[1]["bbox"],
            serde_json::json!([0.0, 0.0, 200.0, 50.0]),
            "clamped"
        );
        // It reads back through the benchmark's COCO parser.
        let parsed = crate::benchmark::images::parse_coco(&v.to_string()).unwrap();
        assert_eq!(parsed[0].1[0].label, "raccoon");
        assert!(parsed[2].1.is_empty());
    }

    #[test]
    fn benchmark_manifest_tags_and_objects() {
        let f = frame(
            vec![lb("dog", [1.0, 2.0, 3.0, 4.0], BoxState::Accepted)],
            false,
        );
        let e = frame(vec![], true);
        let m = benchmark_manifest(&[&f, &e], &classes(), "review-x");
        assert_eq!(m.scored_labels.as_deref(), Some(classes().as_slice()));
        assert_eq!(
            m.images[0].tags,
            [
                "ir",
                "camera:cam",
                "date:2026-10-22",
                "split:val",
                "bucket:suppressed"
            ]
        );
        assert!(m.images[1].tags.contains(&"hard-negative".to_string()));
        assert_eq!(m.images[1].objects.as_ref().unwrap().len(), 0);
        assert!(m.has_ground_truth());
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(Manifest::parse(&text).unwrap(), m);
        // A random pick of the Review queue is tagged `random`; an unknown pick is not.
        let mut r = frame(vec![], true);
        r.labels.review_pick = Some(ReviewPick::Random);
        let mut u = frame(vec![], true);
        u.labels.review_pick = Some(ReviewPick::Other("stratified".into()));
        let m = benchmark_manifest(&[&f, &r, &u], &classes(), "review-x");
        assert_eq!(m.images[1].tags.last().map(String::as_str), Some("random"));
        for i in [0, 2] {
            assert!(!m.images[i].tags.iter().any(|t| t == "random"), "{i}");
        }
    }

    #[test]
    fn frame_notes_list_the_random_picks_and_read_without_them() {
        let named = |name: &str, pick: Option<ReviewPick>, bulk: bool| {
            let mut f = frame(vec![], false);
            f.name = name.into();
            f.labels.review_pick = pick;
            f.labels.bulk_accepted = bulk;
            f
        };
        let frames = [
            named("c.jpg", Some(ReviewPick::Random), false),
            named("a.jpg", Some(ReviewPick::Random), true),
            named("b.jpg", None, true),
            named("d.jpg", Some(ReviewPick::Other("stratified".into())), false),
        ];
        let notes = frame_notes(&frames);
        assert_eq!(notes.version, FRAMES_VERSION);
        assert_eq!(notes.version, 1, "an added list, no new version");
        assert_eq!(notes.random_pick, ["a.jpg", "c.jpg"], "sorted");
        assert_eq!(notes.bulk_accepted, ["a.jpg", "b.jpg"]);
        let v = serde_json::to_value(&notes).unwrap();
        assert_eq!(v["random_pick"], serde_json::json!(["a.jpg", "c.jpg"]));
        let back: FrameNotes = serde_json::from_value(v).unwrap();
        assert_eq!(back, notes);
        // A frames.json of an earlier build has no such key: no random picks.
        let old: FrameNotes = serde_json::from_str(
            r#"{"version": 1, "empty_confirmed": ["e.jpg"], "bulk_accepted": []}"#,
        )
        .unwrap();
        assert!(old.random_pick.is_empty());
        assert_eq!(old.empty_confirmed, ["e.jpg"]);
    }

    #[test]
    fn splits_are_deterministic_and_respect_holdouts() {
        let groups: BTreeSet<Group> = [
            g("drive", "2026-10-20"),
            g("drive", "2026-10-21"),
            g("yard", "2026-10-20"),
            g("yard", "2026-10-21"),
            g("porch", "2026-10-21"),
        ]
        .into_iter()
        .collect();
        let cfg = SplitConfig::default();
        let a = assign_splits(&groups, &cfg);
        assert_eq!(a, assign_splits(&groups, &cfg), "deterministic");
        assert!(a.values().any(|s| *s == Split::Val));
        assert!(a.values().any(|s| *s == Split::Train));
        // Adding a group keeps the others where they were unless the at-least-one rule applies.
        let mut more = groups.clone();
        more.insert(g("drive", "2026-10-22"));
        let b = assign_splits(&more, &cfg);
        let moved = groups.iter().filter(|k| a[*k] != b[*k]).count();
        assert!(moved <= 1, "{a:?} vs {b:?}");
        let explicit = SplitConfig {
            holdout_cameras: vec!["yard".into()],
            holdout_days: vec!["2026-10-21".into()],
            ..SplitConfig::default()
        };
        let e = assign_splits(&groups, &explicit);
        assert_eq!(e[&g("drive", "2026-10-20")], Split::Train);
        assert_eq!(e[&g("drive", "2026-10-21")], Split::Val);
        assert_eq!(e[&g("yard", "2026-10-20")], Split::Val);
        let none = SplitConfig {
            holdout_fraction: 0.0,
            ..SplitConfig::default()
        };
        assert!(
            assign_splits(&groups, &none)
                .values()
                .all(|s| *s == Split::Train)
        );
        let all = SplitConfig {
            holdout_fraction: 1.0,
            ..SplitConfig::default()
        };
        let x = assign_splits(&groups, &all);
        assert_eq!(x.values().filter(|s| **s == Split::Train).count(), 1);
        let one: BTreeSet<Group> = [g("a", "2026-01-01")].into_iter().collect();
        assert_eq!(assign_splits(&one, &cfg).len(), 1);
    }

    /// [`hash_heldout`] is [`assign_splits`]' rule per day: over a grid of cameras, dates and
    /// fractions, every group splits as the function says unless the guard had to move one, and
    /// the hash is still the one of `camera/date`.
    #[test]
    fn the_per_day_hash_rule_agrees_with_assign_splits() {
        let cameras = ["drive", "yard", "porch", "back-door", "cam 7"];
        let dates: Vec<String> = (1..=28).map(|d| format!("2026-10-{d:02}")).collect();
        let grid: BTreeSet<Group> = cameras
            .iter()
            .flat_map(|c| dates.iter().map(|d| g(c, d)))
            .collect();
        let small: BTreeSet<Group> = grid.iter().take(3).cloned().collect();
        let (mut held, mut trained, mut guarded) = (0, 0, 0);
        for groups in [&grid, &small] {
            for frac in [
                0.0f32,
                0.01,
                0.05,
                0.2,
                DEFAULT_HOLDOUT_FRACTION,
                0.5,
                0.9,
                1.0,
            ] {
                let cfg = SplitConfig {
                    holdout_fraction: frac,
                    ..SplitConfig::default()
                };
                let splits = assign_splits(groups, &cfg);
                let rule: BTreeMap<&Group, bool> = groups
                    .iter()
                    .map(|x| (x, hash_heldout(&x.0, &x.1, f64::from(frac))))
                    .collect();
                // The guard moves one group, only when a side would be empty.
                let one_sided = rule.values().all(|v| *v) || !rule.values().any(|v| *v);
                let moved = rule
                    .iter()
                    .filter(|(x, val)| (splits[**x] == Split::Val) != **val)
                    .count();
                if one_sided && frac > 0.0 {
                    assert_eq!(moved, 1, "{frac}");
                    guarded += 1;
                } else {
                    assert_eq!(moved, 0, "{frac}");
                }
                held += rule.values().filter(|v| **v).count();
                trained += rule.values().filter(|v| !**v).count();
            }
        }
        assert!(
            held > 100 && trained > 100 && guarded > 0,
            "{held} {trained} {guarded}"
        );
        // The hash is that of the joined name, as before the factoring.
        use sha2::Digest;
        let d = sha2::Sha256::digest(b"drive/2026-10-20");
        let want = u64::from_be_bytes(d[..8].try_into().unwrap());
        assert_eq!(group_hash(&g("drive", "2026-10-20")), want);
        assert!(!hash_heldout("drive", "2026-10-20", 0.0));
        assert!(hash_heldout("drive", "2026-10-20", 1.0));
    }

    #[test]
    fn the_class_filter_keeps_frames_with_a_training_box_of_a_named_class() {
        let dog = frame(
            vec![
                lb("dog", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted),
                lb("person", [20.0, 1.0, 29.0, 9.0], BoxState::Added),
            ],
            false,
        );
        let rejected_dog = frame(
            vec![
                lb("dog", [1.0, 1.0, 9.0, 9.0], BoxState::Rejected),
                lb("person", [20.0, 1.0, 29.0, 9.0], BoxState::Accepted),
            ],
            false,
        );
        let empty = frame(vec![], true);
        let all = ClassFilter::default();
        assert!(
            all.keeps(&dog.labels) && all.keeps(&rejected_dog.labels) && all.keeps(&empty.labels)
        );
        let only_dog = ClassFilter {
            labels: vec!["dog".into()],
            include_empty: true,
        };
        assert!(only_dog.keeps(&dog.labels));
        assert!(
            !only_dog.keeps(&rejected_dog.labels),
            "a rejected box is no training box"
        );
        assert!(only_dog.keeps(&empty.labels));
        let no_empty = ClassFilter {
            include_empty: false,
            ..only_dog.clone()
        };
        assert!(!no_empty.keeps(&empty.labels));
        let every_class_no_empty = ClassFilter {
            labels: vec![],
            include_empty: false,
        };
        assert!(
            every_class_no_empty.keeps(&dog.labels) && !every_class_no_empty.keeps(&empty.labels)
        );
        assert!(only_dog.check(&classes()).is_ok());
        let e = ClassFilter {
            labels: vec!["unicorn".into()],
            include_empty: true,
        }
        .check(&classes())
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("'unicorn' is not in the class list") && e.contains("Dataset tab"),
            "{e}"
        );
    }

    /// A reviewed frame (64x48 image, sidecar, label file) of `camera` on `date`.
    fn put_reviewed(
        dir: &Path,
        camera: &str,
        date: &str,
        stem: &str,
        boxes: Vec<LabelBox>,
        empty: bool,
    ) {
        let day = dir.join("day").join(camera).join(date);
        std::fs::create_dir_all(&day).unwrap();
        image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]))
            .save(day.join(format!("{stem}.jpg")))
            .unwrap();
        let mut r = super::super::super::record::empty_record();
        r.id = stem.into();
        r.camera = camera.into();
        r.width = 64;
        r.height = 48;
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let f = LabelFile {
            status: Status::Reviewed,
            empty_confirmed: empty,
            boxes,
            ..LabelFile::default()
        };
        super::super::write(
            &day.join(format!("{stem}{}", super::super::LABELS_SUFFIX)),
            &f,
        )
        .unwrap();
    }

    #[test]
    fn a_filtered_export_keeps_other_class_boxes_and_records_the_filter() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![
                lb("raccoon", [4.0, 4.0, 20.0, 20.0], BoxState::Accepted),
                lb("person", [30.0, 4.0, 60.0, 40.0], BoxState::Added),
            ],
            false,
        );
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "b",
            vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
            false,
        );
        put_reviewed(
            &dir,
            "yard",
            "2026-10-22",
            "c",
            vec![lb("raccoon", [1.0, 1.0, 9.0, 9.0], BoxState::Rejected)],
            true,
        );
        let req = |out: &str| ExportRequest {
            formats: vec![Format::Yolo, Format::Benchmark],
            out_dir: root.join(out),
            split: SplitConfig {
                holdout_cameras: vec!["yard".into()],
                ..SplitConfig::default()
            },
            benchmark_split: BenchSplit::All,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
            size_floor: None,
            heldout: None,
        };
        let filter = ClassFilter {
            labels: vec!["raccoon".into()],
            include_empty: true,
        };
        let r = run_filtered(&dir, &classes(), &req("one"), &filter).unwrap();
        assert_eq!((r.frames, r.filtered_out, r.empty_confirmed), (2, 1, 1));
        assert_eq!(
            r.per_class.get("person"),
            Some(&1),
            "the kept frame's person stays labelled"
        );
        let txt = std::fs::read_to_string(root.join("one/labels/train/day.drive.2026-10-21.a.txt"))
            .unwrap();
        let ids: Vec<&str> = txt.lines().map(|l| &l[..1]).collect();
        assert_eq!(ids, ["2", "0"], "class ids follow the class list");
        assert!(
            !root
                .join("one/labels/train/day.drive.2026-10-21.b.txt")
                .exists()
        );
        assert!(
            root.join("one/labels/val/day.yard.2026-10-22.c.txt")
                .is_file()
        );
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("one/export.json")).unwrap())
                .unwrap();
        assert_eq!(
            report["class_filter"],
            serde_json::json!({"labels": ["raccoon"], "include_empty": true})
        );
        assert_eq!(report["filtered_out"], 1);
        // Without confirmed-empty frames.
        let r = run_filtered(
            &dir,
            &classes(),
            &req("two"),
            &ClassFilter {
                include_empty: false,
                ..filter.clone()
            },
        )
        .unwrap();
        assert_eq!((r.frames, r.filtered_out, r.empty_confirmed), (1, 2, 0));
        // No filter: every reviewed frame, as before.
        let r = run(&dir, &classes(), &req("three")).unwrap();
        assert_eq!((r.frames, r.filtered_out), (3, 0));
        assert_eq!(r.class_filter, ClassFilter::default());
        // A filter nothing matches says so; a label outside the class list is refused.
        let e = run_filtered(
            &dir,
            &classes(),
            &req("four"),
            &ClassFilter {
                labels: vec!["dog".into()],
                include_empty: false,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("left out by the class filter"), "{e}");
        assert!(
            run_filtered(
                &dir,
                &classes(),
                &req("five"),
                &ClassFilter {
                    labels: vec!["fox".into()],
                    include_empty: true
                }
            )
            .is_err()
        );
        assert!(!root.join("four").exists() && !root.join("five").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unsure_boxes_left_out_of_yolo_marked_in_coco_and_ignored_in_the_benchmark() {
        let unsure = LabelBox {
            difficult: true,
            ..lb("raccoon", [100.0, 10.0, 140.0, 40.0], BoxState::Accepted)
        };
        let f = frame(
            vec![
                lb("dog", [20.0, 10.0, 60.0, 50.0], BoxState::Accepted),
                unsure,
            ],
            false,
        );
        assert_eq!(
            yolo_txt(&f, &classes()).unwrap(),
            "1 0.200000 0.300000 0.200000 0.400000\n",
            "the unsure box is left out, the frame's other box stays"
        );
        let only = frame(f.labels.boxes[1..].to_vec(), false);
        assert_eq!(
            yolo_txt(&only, &classes()).unwrap(),
            "",
            "a frame of unsure boxes stays"
        );
        let v = coco_json(&[&f], &classes()).unwrap();
        let anns = v["annotations"].as_array().unwrap();
        assert_eq!(anns.len(), 2);
        assert!(anns[0].get("attributes").is_none());
        assert_eq!(
            anns[1]["attributes"],
            serde_json::json!({"difficult": true})
        );
        let m = benchmark_manifest(&[&f], &classes(), "t");
        let objs = m.images[0].objects.as_ref().unwrap();
        assert_eq!(
            objs.iter().map(|o| o.ignore).collect::<Vec<_>>(),
            [false, true]
        );
    }

    #[test]
    fn too_small_rejects_are_ignored_in_the_benchmark_and_listed_for_training() {
        let tiny = LabelBox {
            reject_reason: Some(super::super::RejectReason::TooSmall),
            ..lb("person", [20.0, 10.0, 24.0, 18.0], BoxState::Rejected)
        };
        let unsure = LabelBox {
            difficult: true,
            ..lb("raccoon", [100.0, 10.0, 140.0, 40.0], BoxState::Accepted)
        };
        let wrong = lb("dog", [50.0, 50.0, 60.0, 60.0], BoxState::Rejected);
        // Off the frame once clamped: nothing to ignore.
        let gone = LabelBox {
            reject_reason: Some(super::super::RejectReason::TooSmall),
            ..lb("person", [300.0, 10.0, 320.0, 30.0], BoxState::Rejected)
        };
        let keep = lb("dog", [20.0, 10.0, 60.0, 50.0], BoxState::Accepted);
        let f = frame(vec![keep, tiny, unsure, wrong, gone], false);
        // Neither reaches YOLO or COCO as a label: the rejected box was never one.
        assert_eq!(
            yolo_txt(&f, &classes()).unwrap(),
            "1 0.200000 0.300000 0.200000 0.400000\n"
        );
        assert_eq!(
            coco_json(&[&f], &classes()).unwrap()["annotations"]
                .as_array()
                .unwrap()
                .len(),
            2,
            "the accepted box and the unsure one"
        );
        let m = benchmark_manifest(&[&f], &classes(), "t");
        let objs: Vec<(&str, bool)> = m.images[0]
            .objects
            .iter()
            .flatten()
            .map(|o| (o.label.as_str(), o.ignore))
            .collect();
        assert_eq!(
            objs,
            [("dog", false), ("person", true), ("raccoon", true)],
            "the plain reject is no ground truth"
        );
        let plain = frame(
            vec![lb("dog", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
            true,
        );
        let file = ignore_file(&[f.clone(), plain]);
        assert_eq!(file.version, IGNORE_VERSION);
        assert_eq!(file.images.len(), 1, "a frame without any is not listed");
        assert_eq!(
            file.images[&f.name],
            [
                IgnoreBox {
                    label: "person".into(),
                    cx: 0.11,
                    cy: 0.14,
                    w: 0.02,
                    h: 0.08,
                    size_floor: false,
                },
                IgnoreBox {
                    label: "raccoon".into(),
                    cx: 0.6,
                    cy: 0.25,
                    w: 0.2,
                    h: 0.3,
                    size_floor: false,
                },
            ]
        );
    }

    fn floor_384() -> Option<SizeFloor> {
        Some(SizeFloor {
            px: 10.0,
            input: 384,
        })
    }

    #[test]
    fn a_box_under_the_size_floor_is_exported_as_an_unsure_box_is() {
        // 200 x 100 at 384: 4 px wide is 7.7 px, under 10; the dog is 77 px.
        let tiny = lb("person", [100.0, 10.0, 104.0, 30.0], BoxState::Accepted);
        let dog = lb("dog", [20.0, 10.0, 60.0, 50.0], BoxState::Accepted);
        let unsure = LabelBox {
            difficult: true,
            ..lb("raccoon", [120.0, 10.0, 160.0, 40.0], BoxState::Accepted)
        };
        let mut f = frame(vec![dog.clone(), tiny.clone(), unsure.clone()], false);
        let off = f.clone();
        f.floor = floor_384();
        assert!(f.under_floor(&f.labels.boxes[1]));
        assert!(!f.under_floor(&f.labels.boxes[0]) && !f.under_floor(&f.labels.boxes[2]));
        assert_eq!(
            yolo_txt(&f, &classes()).unwrap(),
            "1 0.200000 0.300000 0.200000 0.400000\n",
            "out of the labels"
        );
        assert_eq!(
            yolo_txt(&off, &classes()).unwrap(),
            "1 0.200000 0.300000 0.200000 0.400000\n0 0.510000 0.200000 0.020000 0.200000\n",
            "the floor off: a label as before"
        );
        let file = ignore_file(&[f.clone()]);
        let regions: Vec<(&str, bool)> = file.images[&f.name]
            .iter()
            .map(|b| (b.label.as_str(), b.size_floor))
            .collect();
        assert_eq!(regions, [("person", true), ("raccoon", false)]);
        let text = serde_json::to_string(&ignore_file(std::slice::from_ref(&off))).unwrap();
        assert!(!text.contains("size_floor"), "{text}");
        // An older reader of ignore.json reads the marked region like any other.
        #[derive(Deserialize)]
        struct OldBox {
            label: String,
        }
        #[derive(Deserialize)]
        struct OldFile {
            images: BTreeMap<String, Vec<OldBox>>,
        }
        let mut marked = ignore_file(&[f.clone()]);
        marked.size_floor = floor_384();
        let old: OldFile = serde_json::from_slice(&json_bytes(&marked).unwrap()).unwrap();
        assert_eq!(old.images[&f.name][0].label, "person");
        let coco = coco_json(&[&f], &classes()).unwrap();
        let attrs: Vec<serde_json::Value> = coco["annotations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.get("attributes").cloned().unwrap_or_default())
            .collect();
        assert_eq!(
            attrs,
            [
                serde_json::Value::Null,
                serde_json::json!({"difficult": true, "size_floor": true}),
                serde_json::json!({"difficult": true}),
            ]
        );
        assert_eq!(
            coco_json(&[&off], &classes()).unwrap()["annotations"][1].get("attributes"),
            None
        );
        let m = benchmark_manifest(&[&f], &classes(), "t");
        let objs: Vec<(&str, bool)> = m.images[0]
            .objects
            .iter()
            .flatten()
            .map(|o| (o.label.as_str(), o.ignore))
            .collect();
        assert_eq!(objs, [("dog", false), ("person", true), ("raccoon", true)]);
        // X-AnyLabeling keeps every training box as it is.
        assert_eq!(f.training_boxes().len(), 3);
        assert_eq!(f.boxes().len(), 2);
        // A box already an ignore region, or no training box, is not the floor's.
        let mut rejected = tiny.clone();
        rejected.state = BoxState::Rejected;
        let tiny_unsure = LabelBox {
            difficult: true,
            ..tiny
        };
        assert!(!f.under_floor(&rejected) && !f.under_floor(&tiny_unsure));
    }

    #[test]
    fn an_export_with_a_size_floor_changes_no_label_file_and_says_what_it_did() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        // 64 x 48 frames at 384: 1 px wide is 6 px, under the floor of 10.
        let tiny = |label: &str| lb(label, [1.0, 1.0, 2.0, 9.0], BoxState::Accepted);
        let big = |label: &str| lb(label, [30.0, 4.0, 60.0, 40.0], BoxState::Accepted);
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![big("dog"), tiny("person")],
            false,
        );
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "b",
            vec![tiny("person")],
            false,
        );
        put_reviewed(
            &dir,
            "yard",
            "2026-10-22",
            "c",
            vec![tiny("raccoon")],
            false,
        );
        put_reviewed(&dir, "yard", "2026-10-22", "d", vec![big("person")], false);
        let files: Vec<PathBuf> = [
            "drive/2026-10-21/a",
            "drive/2026-10-21/b",
            "yard/2026-10-22/c",
        ]
        .iter()
        .map(|p| {
            dir.join("day")
                .join(format!("{p}{}", super::super::LABELS_SUFFIX))
        })
        .collect();
        let bytes = || {
            files
                .iter()
                .map(|p| std::fs::read(p).unwrap())
                .collect::<Vec<_>>()
        };
        let before = bytes();
        let req = |out: &str, floor: Option<SizeFloor>| ExportRequest {
            formats: Format::ALL.to_vec(),
            out_dir: root.join(out),
            split: SplitConfig {
                holdout_cameras: vec!["yard".into()],
                ..SplitConfig::default()
            },
            benchmark_split: BenchSplit::All,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
            size_floor: floor,
            heldout: None,
        };
        let off = run(&dir, &classes(), &req("off", None)).unwrap();
        let on = run(&dir, &classes(), &req("on", floor_384())).unwrap();
        assert_eq!(bytes(), before, "the label files are never rewritten");
        // Off: nothing new in the files.
        for name in ["export.json", IGNORE_FILE] {
            let text = std::fs::read_to_string(root.join("off").join(name)).unwrap();
            assert!(!text.contains("floor"), "{name}: {text}");
        }
        assert_eq!((off.size_floor, off.under_floor.as_ref()), (None, None));
        assert_eq!(off.per_class.get("person"), Some(&3));
        // On: recorded and counted per split and class.
        assert_eq!(on.size_floor, floor_384());
        let u = on.under_floor.clone().unwrap();
        assert_eq!(u.train, BTreeMap::from([("person".to_string(), 2)]));
        assert_eq!(u.val, BTreeMap::from([("raccoon".to_string(), 1)]));
        assert_eq!(
            u.frames_only, 2,
            "b and c are left with ignore regions only"
        );
        assert_eq!(on.per_class.get("person"), Some(&1));
        assert_eq!((on.ignore_boxes, off.ignore_boxes), (3, 0));
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("on/export.json")).unwrap())
                .unwrap();
        assert_eq!(
            report["size_floor"],
            serde_json::json!({"px": 10.0, "input": 384})
        );
        assert_eq!(report["under_floor"]["train"]["person"], 2);
        let ignore: IgnoreFile =
            serde_json::from_slice(&std::fs::read(root.join("on").join(IGNORE_FILE)).unwrap())
                .unwrap();
        assert_eq!(ignore.size_floor, floor_384());
        assert!(ignore.images.values().flatten().all(|b| b.size_floor));
        // A frame of tiny boxes only is no confirmed-empty frame, and its label file is empty.
        let notes: FrameNotes =
            serde_json::from_slice(&std::fs::read(root.join("on").join(FRAMES_FILE)).unwrap())
                .unwrap();
        assert!(notes.empty_confirmed.is_empty());
        assert_eq!(
            std::fs::read_to_string(root.join("on/labels/train/day.drive.2026-10-21.b.txt"))
                .unwrap(),
            ""
        );
        // X-AnyLabeling gets every box as reviewed, so an import changes nothing.
        let x: xany::XFile = serde_json::from_slice(
            &std::fs::read(root.join("on/xanylabeling/day.drive.2026-10-21.a.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(x.shapes.len(), 2);
        assert!(x.shapes.iter().all(|s| !s.difficult));
        // What a training run reads: the tiny person an ignore region beside the dog.
        let source = crate::training::dataset::read_export(&root.join("on")).unwrap();
        let a = source
            .frames
            .iter()
            .find(|f| f.image.ends_with("day.drive.2026-10-21.a.jpg"))
            .unwrap();
        assert_eq!((a.boxes.len(), a.ignore.len()), (1, 1));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn floor_counts_name_the_boxes_an_export_would_turn_into_ignore_regions() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        let tiny = |label: &str| lb(label, [1.0, 1.0, 2.0, 9.0], BoxState::Accepted);
        let big = |label: &str| lb(label, [30.0, 4.0, 60.0, 40.0], BoxState::Accepted);
        let unsure = LabelBox {
            difficult: true,
            ..tiny("dog")
        };
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![big("dog"), tiny("person")],
            false,
        );
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "b",
            vec![tiny("person"), unsure],
            false,
        );
        put_reviewed(&dir, "yard", "2026-10-22", "c", vec![big("person")], false);
        let c = floor_counts(&dir, &classes(), floor_384().unwrap());
        assert_eq!(c.frames, 3);
        assert_eq!(
            c.boxes,
            BTreeMap::from([("dog".to_string(), 1), ("person".to_string(), 3)]),
            "an unsure box is an ignore region already"
        );
        assert_eq!(c.under, BTreeMap::from([("person".to_string(), 2)]));
        assert_eq!((c.frames_with, c.frames_only), (2, 1));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hard_negatives_carry_the_reject_reason() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![
                lb("person", [30.0, 4.0, 60.0, 40.0], BoxState::Accepted),
                LabelBox {
                    reject_reason: Some(super::super::RejectReason::InsectWeb),
                    ..lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Rejected)
                },
                lb("dog", [70.0, 1.0, 90.0, 20.0], BoxState::Rejected),
                // A second box on the accepted person: no false positive.
                LabelBox {
                    reject_reason: Some(super::super::RejectReason::Duplicate),
                    ..lb("person", [30.0, 4.0, 60.0, 39.0], BoxState::Rejected)
                },
                // Real, only too small to label: no false positive either.
                LabelBox {
                    reject_reason: Some(super::super::RejectReason::TooSmall),
                    ..lb("person", [50.0, 30.0, 60.0, 45.0], BoxState::Rejected)
                },
                LabelBox {
                    reason: Some(super::super::bulk::DISMISSED_REASON.into()),
                    ..lb("person", [100.0, 1.0, 120.0, 20.0], BoxState::Rejected)
                },
            ],
            false,
        );
        let req = ExportRequest {
            formats: vec![Format::Yolo],
            out_dir: root.join("out"),
            split: SplitConfig::default(),
            benchmark_split: BenchSplit::All,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
            size_floor: None,
            heldout: None,
        };
        let r = run(&dir, &classes(), &req).unwrap();
        assert_eq!(
            r.hard_negative_boxes, 2,
            "duplicate, too small and dismissed left out"
        );
        assert_eq!(r.ignore_boxes, 1);
        let ignore: IgnoreFile =
            serde_json::from_slice(&std::fs::read(root.join("out").join(IGNORE_FILE)).unwrap())
                .unwrap();
        let (name, list) = ignore.images.iter().next().unwrap();
        assert!(root.join("out/images/train").join(name).is_file(), "{name}");
        assert_eq!(list[0].label, "person");
        // What a training run reads back from a real export.
        let source = crate::training::dataset::read_export(&root.join("out")).unwrap();
        let frame = &source.frames[0];
        assert_eq!(frame.boxes.len(), 1);
        assert_eq!(frame.ignore.len(), 1);
        assert_eq!(frame.ignore[0].class, "person");
        assert!((frame.ignore[0].cx - 0.859_375).abs() < 1e-6);
        assert!((frame.ignore[0].h - 0.3125).abs() < 1e-6);
        let neg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("out/hard_negatives.json")).unwrap(),
        )
        .unwrap();
        let reasons: Vec<&serde_json::Value> = neg
            .as_array()
            .unwrap()
            .iter()
            .map(|n| &n["reject_reason"])
            .collect();
        assert_eq!(
            reasons,
            [&serde_json::json!("insect_web"), &serde_json::Value::Null]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn out_folder_new_empty_nonempty_or_dataset() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(dir.join("day")).unwrap();
        let now = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_800_000_000);
        // New and empty folders are used as they are.
        let new = root.join("new");
        assert_eq!(prepare_out(&dir, &new, now).unwrap(), new);
        assert!(
            !new.exists(),
            "nothing is created before the export is whole"
        );
        std::fs::create_dir_all(&new).unwrap();
        assert_eq!(prepare_out(&dir, &new, now).unwrap(), new);
        // A non-empty one gets a new export-<frame stamp> folder inside it.
        std::fs::write(new.join("x.txt"), b"x").unwrap();
        let sub = prepare_out(&dir, &new, now).unwrap();
        assert_eq!(sub.parent(), Some(new.as_path()));
        let name = sub.file_name().unwrap().to_string_lossy().into_owned();
        let stamp = name.strip_prefix("export-").expect("export- prefix");
        let parsed = crate::clock::parse_frame_stamp(stamp).expect("a frame stamp");
        let want: chrono::DateTime<chrono::Utc> = now.into();
        assert_eq!(parsed.timestamp(), want.timestamp());
        std::fs::create_dir_all(&sub).unwrap();
        assert!(
            prepare_out(&dir, &new, now).is_err(),
            "that folder exists now"
        );
        // The dataset, a folder inside it (existing or not) and a `..` path into it are refused.
        for out in [
            dir.clone(),
            dir.join("day"),
            dir.join("exports/a"),
            root.join("new/../dataset"),
        ] {
            let e = prepare_out(&dir, &out, now).unwrap_err().to_string();
            assert!(e.contains("dataset folder under review"), "{e}");
        }
        // The dataset's parent is fine (non-empty, so a subfolder).
        assert!(prepare_out(&dir, &root, now).unwrap().starts_with(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_benchmark_set_defaults_to_held_out_frames() {
        assert_eq!(BenchSplit::default(), BenchSplit::Val);
        // The page starts there too, and offers the others with what they mean.
        let page = include_str!("../../../templates/train/dataset.html");
        assert!(page.contains(r#"<option value="val" selected>held-out frames"#));
        assert!(page.contains("all frames (includes training frames"));
    }

    #[test]
    fn flat_names_round_trip() {
        let r =
            layout::parse_rel("night/back_yard-2/2026-10-22/20261022T010203004Z_ab-c.jpg").unwrap();
        let n = flat_name(&r);
        assert_eq!(
            n,
            "night.back_yard-2.2026-10-22.20261022T010203004Z_ab-c.jpg"
        );
        assert_eq!(parse_flat_name(&n), Some(r));
        for bad in [
            "a.jpg",
            "x.cam.2026-10-22.s.jpg",
            "ir.cam.2026-10-22.s.t.jpg",
            "ir.cam.2026-10-22.s.png",
        ] {
            assert_eq!(parse_flat_name(bad), None, "{bad}");
        }
    }

    fn all_formats(out: PathBuf) -> ExportRequest {
        ExportRequest {
            formats: Format::ALL.to_vec(),
            out_dir: out,
            split: SplitConfig::default(),
            benchmark_split: BenchSplit::All,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
            size_floor: None,
            heldout: None,
        }
    }

    fn fail_write(name: Option<&str>) {
        FAIL_WRITE.with_borrow_mut(|n| *n = name.map(Into::into));
    }

    fn names_in(p: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(p)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn a_failed_export_leaves_nothing_that_looks_valid() {
        for name in [IGNORE_FILE, FRAMES_FILE, "export.json", "classes.txt"] {
            let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
            let dir = root.join("dataset");
            put_reviewed(
                &dir,
                "drive",
                "2026-10-21",
                "a",
                vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
                false,
            );
            // A new folder under exports/ (the page's default).
            let exports = root.join("exports");
            let out = exports.join("new");
            fail_write(Some(name));
            let e = run(&dir, &classes(), &all_formats(out.clone())).unwrap_err();
            fail_write(None);
            assert!(format!("{e:#}").contains(name), "{e:#}");
            assert!(!out.exists(), "{name}: {:?}", names_in(&exports));
            assert_eq!(
                names_in(&exports),
                Vec::<String>::new(),
                "{name}: the partial folder goes"
            );
            // An empty folder a user chose stays empty.
            let chosen = root.join("chosen");
            std::fs::create_dir_all(&chosen).unwrap();
            fail_write(Some(name));
            run(&dir, &classes(), &all_formats(chosen.clone())).unwrap_err();
            fail_write(None);
            assert_eq!(names_in(&chosen), Vec::<String>::new(), "{name}");
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// Audit 2026-11 phase 1: an export stopped by the service, or cut by a panic, leaves no
    /// `.building-*` folder (and a stop during the walk leaves the held-out list alone).
    #[test]
    fn a_stopped_or_panicking_export_leaves_no_staging_behind() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        for stem in ["a", "b", "c"] {
            put_reviewed(
                &dir,
                "drive",
                "2026-10-21",
                stem,
                vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
                false,
            );
        }
        let exports = root.join("exports");
        let chosen = root.join("chosen");
        std::fs::create_dir_all(&chosen).unwrap();
        let none = || false;
        for out in [exports.join("new"), chosen.clone()] {
            // Stop after the second file written.
            let files = std::sync::atomic::AtomicUsize::new(0);
            let progress = |s: ExportStep| {
                if let ExportStep::Writing { done, .. } = s {
                    files.store(done, std::sync::atomic::Ordering::SeqCst);
                }
            };
            let stop = || files.load(std::sync::atomic::Ordering::SeqCst) >= 2;
            let req = all_formats(out.clone());
            let filter = ClassFilter::default();
            let e =
                run_filtered_until(&dir, &classes(), &req, &filter, &progress, &stop).unwrap_err();
            assert!(is_stopped(&e), "{e:#}");
            let left = |p: &Path| if p.exists() { names_in(p) } else { Vec::new() };
            assert_eq!(left(&exports), Vec::<String>::new(), "{}", out.display());
            assert_eq!(left(&chosen), Vec::<String>::new(), "{}", out.display());
            // A panic in the middle (here: writing ignore.json) unwinds past the error path.
            PANIC_WRITE.with_borrow_mut(|n| *n = Some(IGNORE_FILE.into()));
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_filtered_until(&dir, &classes(), &req, &filter, &|_| {}, &none)
            }));
            PANIC_WRITE.with_borrow_mut(|n| *n = None);
            assert!(panicked.is_err());
            assert_eq!(left(&exports), Vec::<String>::new(), "{}", out.display());
            assert_eq!(left(&chosen), Vec::<String>::new(), "{}", out.display());
        }
        // A stop during the walk: no export, and the held-out list is never written.
        let mut req = all_formats(exports.join("walk"));
        req.heldout = Some(HeldoutPolicy { seen: None });
        let e = run_filtered_until(
            &dir,
            &classes(),
            &req,
            &ClassFilter::default(),
            &|_| {},
            &|| true,
        )
        .unwrap_err();
        assert!(is_stopped(&e), "{e:#}");
        assert!(!exports.join("walk").exists());
        assert!(!dir.join(heldout::FILE).exists(), "the list is left alone");
        // Not stopped: the same export goes through.
        run_filtered_until(
            &dir,
            &classes(),
            &req,
            &ClassFilter::default(),
            &|_| {},
            &none,
        )
        .unwrap();
        assert!(exports.join("walk/export.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_reviewed_frame_that_lost_a_file_is_counted_and_warned_about() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        let reviewed = |stem: &str| {
            put_reviewed(
                &dir,
                "drive",
                "2026-10-21",
                stem,
                vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
                false,
            );
        };
        reviewed("keep");
        reviewed("nosidecar");
        reviewed("noimage");
        let day = dir.join("day/drive/2026-10-21");
        std::fs::remove_file(day.join("nosidecar.json")).unwrap();
        std::fs::remove_file(day.join("noimage.jpg")).unwrap();
        let r = run(&dir, &classes(), &all_formats(root.join("out"))).unwrap();
        assert_eq!(r.frames, 1);
        let text = r.warnings.join("\n");
        assert!(text.contains("nosidecar"), "{text}");
        assert!(text.contains("noimage"), "{text}");
        assert_eq!((r.lost_frames, r.unreadable_folders), (2, 0));
        // Only reviewed frames count: an unreviewed one is simply not exported.
        let empty = root.join("nothing");
        let day = dir.join("day/drive/2026-10-21");
        std::fs::remove_file(day.join("keep.jpg")).unwrap();
        let e = run(&dir, &classes(), &all_formats(empty.clone())).unwrap_err();
        assert!(format!("{e:#}").contains("keep"), "{e:#}");
        assert!(!empty.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_day_folder_is_warned_about() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        for date in ["2026-10-21", "2026-10-22"] {
            put_reviewed(
                &dir,
                "drive",
                date,
                "a",
                vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
                false,
            );
        }
        let day = dir.join("day/drive/2026-10-22");
        let mode = |m: u32| std::fs::Permissions::from_mode(m);
        std::fs::set_permissions(&day, mode(0o000)).unwrap();
        if std::fs::read_dir(&day).is_ok() {
            // Root reads everything: nothing to show here.
            std::fs::set_permissions(&day, mode(0o755)).unwrap();
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let r = run(&dir, &classes(), &all_formats(root.join("out")));
        std::fs::set_permissions(&day, mode(0o755)).unwrap();
        let r = r.unwrap();
        assert_eq!(r.frames, 1);
        let text = r.warnings.join("\n");
        assert!(text.contains("2026-10-22"), "{text}");
        assert_eq!((r.lost_frames, r.unreadable_folders), (0, 1));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_export_into_an_empty_folder_keeps_the_folder_and_publishes_everything() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted)],
            false,
        );
        let (fresh, chosen) = (root.join("exports/fresh"), root.join("chosen"));
        std::fs::create_dir_all(&chosen).unwrap();
        // A marker the folder keeps through the export: the same folder, not a replacement.
        #[cfg(unix)]
        let inode = {
            use std::os::unix::fs::MetadataExt;
            std::fs::metadata(&chosen).unwrap().ino()
        };
        let a = run(&dir, &classes(), &all_formats(fresh.clone())).unwrap();
        let b = run(&dir, &classes(), &all_formats(chosen.clone())).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(&chosen).unwrap().ino(), inode);
        }
        assert_eq!(names_in(&fresh), names_in(&chosen));
        assert!(names_in(&fresh).iter().all(|n| !n.starts_with('.')));
        assert_eq!(
            names_in(&root.join("exports")),
            ["fresh"],
            "no scratch left"
        );
        for (r, out) in [(&a, &fresh), (&b, &chosen)] {
            for must in [
                "classes.txt",
                "data.yaml",
                "export.json",
                IGNORE_FILE,
                "benchmark/manifest.json",
            ] {
                assert!(out.join(must).is_file(), "{must}");
            }
            // Paths in the export name where it ended up, never the scratch folder.
            assert_eq!(&r.out_dir, out);
            assert_eq!(r.paths["yolo"], out.join("data.yaml"));
            assert!(
                r.benchmark_dataset
                    .as_deref()
                    .unwrap()
                    .contains(&*out.to_string_lossy())
            );
            let yaml = std::fs::read_to_string(out.join("data.yaml")).unwrap();
            assert!(!yaml.contains("building"), "{yaml}");
            let report = std::fs::read_to_string(out.join("export.json")).unwrap();
            assert!(!report.contains("building"), "{report}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_scratch_folder_left_by_a_crash_does_not_look_like_an_export() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let s = Staging::new(&root.join("exports/new")).unwrap();
        write_file(&s.build.join("classes.txt"), b"person\n").unwrap();
        // What `training::discover_exports` looks for, one level under exports/.
        for entry in names_in(&root.join("exports")) {
            let p = root.join("exports").join(entry);
            assert!(!p.join("classes.txt").exists() && !p.join("data.yaml").exists());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_final_move_keeps_the_finished_export_and_says_where() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let target = root.join("exports/new");
        let s = Staging::new(&target).unwrap();
        write_file(&s.build.join("export.json"), b"{}").unwrap();
        // Something else took the name meanwhile.
        write_file(&target.join("x.txt"), b"x").unwrap();
        let e = format!("{:#}", s.publish().unwrap_err());
        assert!(e.contains("complete in"), "{e}");
        assert!(e.contains(&*s.build.to_string_lossy()), "{e}");
        assert!(e.contains(&*target.to_string_lossy()), "{e}");
        assert!(s.build.join("export.json").is_file(), "kept");
        assert!(target.join("x.txt").is_file(), "what was there stays");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A scanner holding a new file open (no delete sharing) refuses the folder's rename with
    /// "Access is denied" for a moment; the publish waits it out instead of failing the job.
    #[cfg(windows)]
    #[test]
    fn a_folder_held_open_for_a_moment_is_still_published() {
        use std::os::windows::fs::OpenOptionsExt;
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let target = root.join("exports/new");
        let s = Staging::new(&target).unwrap();
        write_file(&s.build.join("export.json"), b"{}").unwrap();
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(s.build.join("export.json"))
            .unwrap();
        let refused = std::fs::rename(&s.build, &target).unwrap_err();
        assert!(crate::fsx::is_sharing_violation(&refused), "{refused:?}");
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            drop(held);
        });
        s.publish().unwrap();
        release.join().unwrap();
        assert!(target.join("export.json").is_file());
        assert!(!s.scratch.exists(), "the scratch folder is gone");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn files_of_an_export_move_up_with_what_marks_it_valid_last() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let s = Staging::new(&root).unwrap();
        for f in ["export.json", "classes.txt", "ignore.json", "data.yaml"] {
            write_file(&s.build.join(f), b"x").unwrap();
        }
        // The move stops at a file that cannot arrive: a folder holds its name.
        std::fs::create_dir_all(root.join("data.yaml/full")).unwrap();
        s.publish().unwrap_err();
        assert!(root.join("ignore.json").is_file(), "first");
        assert!(!root.join("export.json").exists(), "last");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (RA-055, RA-060): the scratch folder is held busy (marker beside it)
    /// and recorded while the export runs; a publish, a discard and a panic each leave neither
    /// marker nor record; a failed publish keeps the complete export with no marker (ended, never
    /// swept); a scratch folder that cannot be removed keeps its marker, unlocked, for the sweep.
    #[test]
    fn the_scratch_folder_is_held_while_building_and_released_when_dealt_with() {
        use crate::storage::busy::{self, State};
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let pending = root
            .join("training")
            .join(crate::storage::sweep::EXPORTS_PENDING_DIR);
        let records = || names_in_or_none(&pending);
        // A new target: published.
        let s = Staging::recorded(&root.join("exports/new"), Some(&pending)).unwrap();
        assert!(busy::marker_of(&s.scratch).is_file());
        assert!(
            matches!(busy::probe(&s.scratch), State::Live),
            "held while building"
        );
        assert_eq!(records().len(), 1);
        let rec = pending.join(&records()[0]);
        assert_eq!(
            std::fs::read_to_string(&rec).unwrap(),
            std::path::absolute(&s.scratch)
                .unwrap()
                .display()
                .to_string()
        );
        write_file(&s.build.join("classes.txt"), b"person\n").unwrap();
        s.publish().unwrap();
        assert!(!busy::marker_of(&s.scratch).exists() && !s.scratch.exists());
        assert_eq!(records(), Vec::<String>::new());
        assert_eq!(names_in(&root.join("exports")), ["new"]);
        // An existing folder: discarded.
        let chosen = root.join("chosen");
        std::fs::create_dir_all(&chosen).unwrap();
        let s = Staging::recorded(&chosen, Some(&pending)).unwrap();
        assert!(busy::marker_of(&s.scratch).starts_with(&chosen));
        s.discard();
        assert_eq!(names_in(&chosen), Vec::<String>::new());
        assert_eq!(records(), Vec::<String>::new());
        // A panic while building: dropped unsettled while unwinding.
        let target = root.join("exports/panicked");
        let r = std::panic::catch_unwind(|| {
            let s = Staging::recorded(&target, Some(&pending)).unwrap();
            write_file(&s.build.join("a.txt"), b"x").unwrap();
            panic!("a bug while building");
        });
        assert!(r.is_err());
        assert_eq!(names_in(&root.join("exports")), ["new"]);
        assert_eq!(records(), Vec::<String>::new());
        // A failed publish keeps the export, which then reads as ended (never swept).
        let target = root.join("exports/taken");
        let s = Staging::recorded(&target, Some(&pending)).unwrap();
        write_file(&s.build.join("classes.txt"), b"person\n").unwrap();
        write_file(&target.join("x.txt"), b"x").unwrap();
        s.publish().unwrap_err();
        assert!(s.build.join("classes.txt").is_file());
        assert!(!busy::marker_of(&s.scratch).exists());
        assert!(matches!(busy::probe(&s.scratch), State::Ended));
        assert_eq!(records(), Vec::<String>::new());
        drop(s);
        assert!(target.parent().unwrap().is_dir());
        // A scratch folder that cannot be removed: its marker stays, unlocked (a crash to the sweep).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let s = Staging::recorded(&root.join("exports/stuck"), Some(&pending)).unwrap();
            let sub = s.build.join("images");
            write_file(&sub.join("a.jpg"), b"x").unwrap();
            std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o555)).unwrap();
            if std::fs::write(sub.join("probe"), b"x").is_err() {
                s.discard();
                assert!(s.scratch.is_dir());
                let State::Crashed(claim) = busy::probe(&s.scratch) else {
                    panic!("a scratch folder its export could not remove reads as crashed")
                };
                drop(claim);
                assert_eq!(records().len(), 1, "the record stays for the sweep");
            } else {
                eprintln!("skipped: permissions are not enforced for this user");
                s.discard();
            }
            let _ = std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o755));
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The names in `dir`, none when it does not exist.
    fn names_in_or_none(dir: &Path) -> Vec<String> {
        if dir.exists() {
            names_in(dir)
        } else {
            Vec::new()
        }
    }

    /// An export cut off by a crash: what the next storage sweep finds of it and removes,
    /// beside a new target under exports/ and inside a folder of the user's choice.
    #[test]
    fn a_crashed_exports_scratch_is_what_the_storage_sweep_removes() {
        let root = std::env::temp_dir().join(format!("bop-export-{}", uuid::Uuid::new_v4()));
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: String::new(),
            ..Default::default()
        };
        c.models.clear();
        let settings = crate::storage::sweep::Settings::for_config(&c);
        let training = settings.training_dir();
        let pending = training.join(crate::storage::sweep::EXPORTS_PENDING_DIR);
        let chosen = root.join("chosen");
        std::fs::create_dir_all(&chosen).unwrap();
        let mut scratch = Vec::new();
        for target in [training.join("exports/20261123-1200"), chosen.clone()] {
            let s = Staging::recorded(&target, Some(&pending)).unwrap();
            write_file(&s.build.join("images/a.jpg"), b"x").unwrap();
            // A crash: the process is gone, so its lock is, while marker and folder stay.
            let busy = s.busy.borrow_mut().take().unwrap();
            busy.abandon();
            s.settled.set(true);
            scratch.push(s.scratch.clone());
        }
        let mut out = Vec::new();
        crate::storage::sweep::sweep_export_scratch(
            &settings,
            std::time::SystemTime::now(),
            &mut out,
        );
        assert_eq!(out.len(), 2, "{out:?}");
        for p in &scratch {
            assert!(!p.exists(), "{}", p.display());
            assert!(!crate::storage::busy::marker_of(p).exists());
        }
        assert_eq!(names_in(&chosen), Vec::<String>::new());
        assert_eq!(names_in_or_none(&pending), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every file under `dir`, relative, `/`-separated, sorted.
    fn tree(dir: &Path) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p.strip_prefix(dir).unwrap().to_string_lossy();
                    out.push(rel.replace('\\', "/"));
                }
            }
        }
        out.sort();
        out
    }

    /// Characterises what an export writes in every format and in each alone (QA-177, before
    /// `build_export` was split into one writer per format): the file list, the last `Writing`
    /// count against the files counted up front, the paths and the warnings.
    #[test]
    fn every_format_writes_the_files_it_counts() {
        let root = crate::test_dirs::TempDir::new("export-formats");
        let dir = root.join("dataset");
        let mut negative = lb("dog", [30.0, 20.0, 40.0, 30.0], BoxState::Rejected);
        negative.reject_reason = Some(super::super::RejectReason::FalsePositive);
        let mut unsure = lb("person", [40.0, 30.0, 50.0, 40.0], BoxState::Accepted);
        unsure.difficult = true;
        put_reviewed(
            &dir,
            "drive",
            "2026-10-21",
            "a",
            vec![
                lb("person", [1.0, 1.0, 9.0, 9.0], BoxState::Accepted),
                negative,
                unsure,
            ],
            false,
        );
        put_reviewed(&dir, "drive", "2026-10-22", "b", vec![], true);
        put_reviewed(
            &dir,
            "yard",
            "2026-10-21",
            "c",
            vec![lb("dog", [5.0, 5.0, 25.0, 20.0], BoxState::Accepted)],
            false,
        );
        // A queue frame with no label file: the X-AnyLabeling folder's unreviewed frame.
        let day = dir.join("day").join("yard").join("2026-10-23");
        std::fs::create_dir_all(&day).unwrap();
        image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]))
            .save(day.join("d.jpg"))
            .unwrap();
        let mut rec = super::super::super::record::empty_record();
        rec.id = "d".into();
        rec.camera = "yard".into();
        rec.width = 64;
        rec.height = 48;
        std::fs::write(day.join("d.json"), serde_json::to_vec(&rec).unwrap()).unwrap();
        layout::append_review(&dir, &["day/yard/2026-10-23/d.jpg".to_string()]).unwrap();
        let split = SplitConfig {
            holdout_cameras: vec!["yard".into()],
            ..SplitConfig::default()
        };
        let cases: Vec<(&str, Vec<Format>, BenchSplit, SplitConfig)> = vec![
            ("all", Format::ALL.to_vec(), BenchSplit::All, split.clone()),
            ("yolo", vec![Format::Yolo], BenchSplit::Val, split.clone()),
            ("coco", vec![Format::Coco], BenchSplit::Val, split.clone()),
            (
                "xany",
                vec![Format::XAnyLabeling],
                BenchSplit::Val,
                split.clone(),
            ),
            (
                "bench",
                vec![Format::Benchmark],
                BenchSplit::Val,
                split.clone(),
            ),
            (
                "yolo-coco",
                vec![Format::Yolo, Format::Coco],
                BenchSplit::Val,
                split.clone(),
            ),
            (
                "no-val",
                vec![Format::Yolo, Format::Benchmark],
                BenchSplit::Val,
                SplitConfig {
                    holdout_fraction: 0.0,
                    ..SplitConfig::default()
                },
            ),
        ];
        let mut seen = Vec::new();
        for (name, formats, benchmark_split, split) in cases {
            let out = root.join("out").join(name);
            let req = ExportRequest {
                formats,
                out_dir: out.clone(),
                split,
                benchmark_split,
                include_unreviewed: true,
                hidden_classes: Vec::new(),
                size_floor: None,
                heldout: None,
            };
            let last = std::sync::Mutex::new(None);
            let progress = |s: ExportStep| {
                if let ExportStep::Writing { done, of } = s {
                    *last.lock().unwrap() = Some((done, of));
                }
            };
            let r = run_filtered_until(
                &dir,
                &classes(),
                &req,
                &ClassFilter::default(),
                &progress,
                &|| false,
            )
            .unwrap();
            let at = out.display().to_string();
            let (done, of) = last.lock().unwrap().unwrap();
            assert_eq!(done, of, "{name}: every counted file written");
            seen.push(format!(
                "{name}: {done}/{of} files={:?} paths={:?} warnings={:?}",
                tree(&out),
                r.paths.keys().collect::<Vec<_>>(),
                r.warnings
                    .iter()
                    .map(|w| w.replace(&at, "<out>"))
                    .collect::<Vec<_>>(),
            ));
        }
        // As the export wrote before its split into one writer per format.
        let want = [
            "all: 17/17 files=[\"annotations/instances_train.json\", \"annotations/instances_val.json\", \"benchmark/images/day.drive.2026-10-21.a.jpg\", \"benchmark/images/day.drive.2026-10-22.b.jpg\", \"benchmark/images/day.yard.2026-10-21.c.jpg\", \"benchmark/manifest.json\", \"classes.txt\", \"data.yaml\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"images/train/day.drive.2026-10-21.a.jpg\", \"images/train/day.drive.2026-10-22.b.jpg\", \"images/val/day.yard.2026-10-21.c.jpg\", \"labels/train/day.drive.2026-10-21.a.txt\", \"labels/train/day.drive.2026-10-22.b.txt\", \"labels/val/day.yard.2026-10-21.c.txt\", \"xanylabeling/classes.txt\", \"xanylabeling/day.drive.2026-10-21.a.jpg\", \"xanylabeling/day.drive.2026-10-21.a.json\", \"xanylabeling/day.drive.2026-10-22.b.jpg\", \"xanylabeling/day.drive.2026-10-22.b.json\", \"xanylabeling/day.yard.2026-10-21.c.jpg\", \"xanylabeling/day.yard.2026-10-21.c.json\", \"xanylabeling/day.yard.2026-10-23.d.jpg\", \"xanylabeling/day.yard.2026-10-23.d.json\"] paths=[\"benchmark\", \"coco\", \"xanylabeling\", \"yolo\"] warnings=[]",
            "yolo: 6/6 files=[\"classes.txt\", \"data.yaml\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"images/train/day.drive.2026-10-21.a.jpg\", \"images/train/day.drive.2026-10-22.b.jpg\", \"images/val/day.yard.2026-10-21.c.jpg\", \"labels/train/day.drive.2026-10-21.a.txt\", \"labels/train/day.drive.2026-10-22.b.txt\", \"labels/val/day.yard.2026-10-21.c.txt\"] paths=[\"yolo\"] warnings=[]",
            "coco: 3/3 files=[\"annotations/instances_train.json\", \"annotations/instances_val.json\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"images/train/day.drive.2026-10-21.a.jpg\", \"images/train/day.drive.2026-10-22.b.jpg\", \"images/val/day.yard.2026-10-21.c.jpg\"] paths=[\"coco\"] warnings=[]",
            "xany: 8/8 files=[\"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"xanylabeling/classes.txt\", \"xanylabeling/day.drive.2026-10-21.a.jpg\", \"xanylabeling/day.drive.2026-10-21.a.json\", \"xanylabeling/day.drive.2026-10-22.b.jpg\", \"xanylabeling/day.drive.2026-10-22.b.json\", \"xanylabeling/day.yard.2026-10-21.c.jpg\", \"xanylabeling/day.yard.2026-10-21.c.json\", \"xanylabeling/day.yard.2026-10-23.d.jpg\", \"xanylabeling/day.yard.2026-10-23.d.json\"] paths=[\"xanylabeling\"] warnings=[]",
            "bench: 1/1 files=[\"benchmark/images/day.yard.2026-10-21.c.jpg\", \"benchmark/manifest.json\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\"] paths=[\"benchmark\"] warnings=[]",
            "yolo-coco: 6/6 files=[\"annotations/instances_train.json\", \"annotations/instances_val.json\", \"classes.txt\", \"data.yaml\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"images/train/day.drive.2026-10-21.a.jpg\", \"images/train/day.drive.2026-10-22.b.jpg\", \"images/val/day.yard.2026-10-21.c.jpg\", \"labels/train/day.drive.2026-10-21.a.txt\", \"labels/train/day.drive.2026-10-22.b.txt\", \"labels/val/day.yard.2026-10-21.c.txt\"] paths=[\"coco\", \"yolo\"] warnings=[]",
            "no-val: 6/6 files=[\"classes.txt\", \"data.yaml\", \"export.json\", \"frames.json\", \"hard_negatives.json\", \"ignore.json\", \"images/train/day.drive.2026-10-21.a.jpg\", \"images/train/day.drive.2026-10-22.b.jpg\", \"images/train/day.yard.2026-10-21.c.jpg\", \"labels/train/day.drive.2026-10-21.a.txt\", \"labels/train/day.drive.2026-10-22.b.txt\", \"labels/train/day.yard.2026-10-21.c.txt\"] paths=[\"yolo\"] warnings=[\"no frame was held out for validation; data.yaml validates on the training images (name holdout cameras or days, or review more cameras/days)\", \"the benchmark set was not written: no frame is in the chosen split (a share of 0 holds nothing out); hold out some cameras or days, or choose another set\"]",
        ];
        assert_eq!(seen, want);
    }
}
