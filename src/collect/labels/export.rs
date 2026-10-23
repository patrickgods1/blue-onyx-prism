//! Exports of the reviewed frames (`POST /v1/review/export`): YOLO txt (`labels/<split>/*.txt`,
//! `classes.txt`, `data.yaml`), COCO instances JSON (`annotations/instances_<split>.json`),
//! X-AnyLabeling JSON (`xanylabeling/`) and a benchmark `dir:` set (`benchmark/images/` +
//! `benchmark/manifest.json`, tags `<scene>`, `camera:<c>`, `date:<d>`, `split:<s>`,
//! `bucket:<b>` and `hard-negative` for a frame confirmed empty). Only reviewed frames are
//! exported (X-AnyLabeling may add the queue's unreviewed frames, pre-filled, unchecked).
//!
//! Split ([`assign_splits`]): by camera and day, so held-out frames come from cameras and days
//! the model did not train on. Explicit `holdout_cameras` / `holdout_days` decide when given;
//! else a (camera, date) group is held out when the first 8 bytes of SHA-256(`camera/date`), as a
//! fraction of 2^64, are below `holdout_fraction` (stable when days are added), with at least
//! one group on each side once there are two groups.
//!
//! Pure builders ([`yolo_txt`], [`coco_json`], [`benchmark_manifest`]) plus [`run`], which
//! writes into an empty or new folder (images hard-linked, else copied) and writes
//! `export.json` last.

use super::super::layout::{self, Rel};
use super::super::record::Bucket;
use super::xany;
use super::{BoxState, Frame, LabelBox, LabelFile, Status, clamp_box, label_files};
use crate::benchmark::images::{Manifest, ManifestImage, ManifestObject};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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

fn group_hash(g: &Group) -> u64 {
    use sha2::Digest;
    let d = sha2::Sha256::digest(format!("{}/{}", g.0, g.1).as_bytes());
    let mut b = [0u8; 8];
    for (o, i) in b.iter_mut().zip(d.iter()) {
        *o = *i;
    }
    u64::from_be_bytes(b)
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
            #[allow(clippy::cast_precision_loss, reason = "a fraction of 2^64")]
            let x = group_hash(g) as f64 / 18_446_744_073_709_551_616.0;
            (g.clone(), if x < frac { Split::Val } else { Split::Train })
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
}

impl ExportFrame {
    /// Training boxes, clamped to the frame, empty ones left out.
    pub fn boxes(&self) -> Vec<&LabelBox> {
        self.labels
            .training()
            .filter(|b| {
                let c = clamp_box(b.bbox, self.width, self.height);
                c[2] > c[0] && c[3] > c[1]
            })
            .collect()
    }
}

/// `x` rounded to `d` decimals, as f64 (clean JSON numbers).
fn round_to(x: f64, d: i32) -> f64 {
    let p = 10f64.powi(d);
    (x * p).round() / p
}

/// A frame's YOLO label file: one `class cx cy w h` line per training box, normalised to the
/// frame and clamped to 0..1, 6 decimals; an empty file for a frame without objects. Boxes of a
/// label not in `classes` are an error (the frame must not be exported half-labelled).
pub fn yolo_txt(f: &ExportFrame, classes: &[String]) -> Result<String> {
    let (w, h) = (f64::from(f.width.max(1)), f64::from(f.height.max(1)));
    let mut out = String::new();
    for b in f.boxes() {
        let Some(id) = classes.iter().position(|c| *c == b.label) else {
            bail!(
                "{}: label '{}' is not in the class list",
                f.rel.jpg(),
                b.label
            );
        };
        let c = clamp_box(b.bbox, f.width, f.height).map(f64::from);
        let n = |v: f64| round_to(v.clamp(0.0, 1.0), 6);
        let (cx, cy) = (n((c[0] + c[2]) / 2.0 / w), n((c[1] + c[3]) / 2.0 / h));
        let (bw, bh) = (n((c[2] - c[0]) / w), n((c[3] - c[1]) / h));
        out.push_str(&format!("{id} {cx:.6} {cy:.6} {bw:.6} {bh:.6}\n"));
    }
    Ok(out)
}

/// COCO instances JSON of `frames` (file names as given): category ids 1..=K in `classes`
/// order, image and annotation ids 1.. in order, `bbox` `[x, y, w, h]` in pixels (2 decimals),
/// `area` = w * h, `iscrowd` 0. A frame without objects is an image without annotations.
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
        for b in f.boxes() {
            let Some(cat) = classes.iter().position(|c| *c == b.label) else {
                bail!(
                    "{}: label '{}' is not in the class list",
                    f.rel.jpg(),
                    b.label
                );
            };
            let c = clamp_box(b.bbox, f.width, f.height).map(f64::from);
            let (w, h) = (round_to(c[2] - c[0], 2), round_to(c[3] - c[1], 2));
            anns.push(json!({
                "id": anns.len() + 1,
                "image_id": image_id,
                "category_id": cat + 1,
                "bbox": [round_to(c[0], 2), round_to(c[1], 2), w, h],
                "area": round_to(w * h, 2),
                "iscrowd": 0,
            }));
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
    t
}

/// The benchmark manifest of `frames`: their training boxes as ground truth, scored on
/// `classes` only (predictions of other labels count neither way).
pub fn benchmark_manifest(frames: &[&ExportFrame], classes: &[String], id: &str) -> Manifest {
    Manifest {
        id: id.to_string(),
        title: format!("Reviewed frames ({id})"),
        description: "Frames collected from your cameras and reviewed on the Review page.".into(),
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
                    f.boxes()
                        .into_iter()
                        .map(|b| ManifestObject {
                            label: b.label.clone(),
                            bbox: clamp_box(b.bbox, f.width, f.height),
                            ignore: false,
                            crowd: false,
                        })
                        .collect(),
                ),
            })
            .collect(),
        ..Manifest::default()
    }
}

/// Which frames the benchmark set holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchSplit {
    #[default]
    All,
    Train,
    Val,
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
    /// Rejected model / auto boxes listed in `hard_negatives.json`.
    pub hard_negative_boxes: usize,
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
    let mut warnings = Vec::new();
    let mut frames = Vec::new();
    for (rel, _) in label_files(dir) {
        let Some(frame) = Frame::resolve(dir, &rel) else {
            continue;
        };
        let labels = match frame.labels() {
            Ok(Some(l)) if l.status == Status::Reviewed => l,
            Ok(_) => continue,
            Err(e) => {
                warnings.push(format!("{rel}: {e:#}; left out"));
                continue;
            }
        };
        let rec = match frame.record() {
            Ok(r) => r,
            Err(e) => {
                warnings.push(format!("{rel}: {e:#}; left out"));
                continue;
            }
        };
        let unknown: BTreeSet<&str> = labels
            .training()
            .filter(|b| !classes.contains(&b.label))
            .map(|b| b.label.as_str())
            .collect();
        if !unknown.is_empty() {
            warnings.push(format!(
                "{rel}: label(s) {} are not in the class list; left out (review it again or add \
                 the class)",
                unknown.into_iter().collect::<Vec<_>>().join(", ")
            ));
            continue;
        }
        if labels.boxes.iter().any(|b| b.state == BoxState::Pending) {
            warnings.push(format!("{rel}: reviewed but has pending boxes; left out"));
            continue;
        }
        frames.push(ExportFrame {
            name: flat_name(&frame.rel),
            rel: frame.rel,
            image: frame.jpg,
            width: rec.width,
            height: rec.height,
            buckets: rec.buckets,
            split: Split::Train,
            labels,
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
    (frames, warnings)
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

/// `out` must be a new folder or an empty one; it is created.
fn prepare_out(out: &Path) -> Result<()> {
    match std::fs::read_dir(out) {
        Ok(mut rd) => {
            if rd.next().is_some() {
                bail!(
                    "{} is not empty; choose a new folder (or leave it empty for a new one under \
                     training/exports)",
                    out.display()
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("reading {}", out.display())),
    }
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))
}

/// Export the reviewed frames of the dataset `dir` (see the module doc).
pub fn run(dir: &Path, classes: &[String], req: &ExportRequest) -> Result<ExportReport> {
    if req.formats.is_empty() {
        bail!("no export format given (yolo, coco, xanylabeling, benchmark)");
    }
    let (frames, mut warnings) = reviewed_frames(dir, classes, &req.split);
    let unreviewed: Vec<(Frame, LabelFile, u32, u32)> =
        if req.include_unreviewed && req.formats.contains(&Format::XAnyLabeling) {
            unreviewed_queue_frames(dir, &mut warnings)
        } else {
            Vec::new()
        };
    if frames.is_empty() && unreviewed.is_empty() {
        bail!(
            "no reviewed frame to export in {}; review frames on the Review page first (save \
             them as reviewed)",
            dir.display()
        );
    }
    let out = &req.out_dir;
    prepare_out(out)?;
    let mut r = ExportReport {
        out_dir: out.clone(),
        formats: req.formats.clone(),
        frames: frames.len(),
        classes: classes.to_vec(),
        unreviewed_frames: unreviewed.len(),
        ..ExportReport::default()
    };
    let mut groups: BTreeMap<Group, (Split, usize)> = BTreeMap::new();
    for f in &frames {
        match f.split {
            Split::Train => r.train += 1,
            Split::Val => r.val += 1,
        }
        r.empty_confirmed += usize::from(f.labels.empty_confirmed);
        for b in f.boxes() {
            *r.per_class.entry(b.label.clone()).or_default() += 1;
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
    let by_split = |s: Split| frames.iter().filter(|f| f.split == s).collect::<Vec<_>>();
    let wants = |f: Format| req.formats.contains(&f);
    if (wants(Format::Yolo) || wants(Format::Coco)) && !frames.is_empty() {
        for f in &frames {
            let dst = out.join("images").join(f.split.as_str()).join(&f.name);
            if let Some(p) = dst.parent() {
                std::fs::create_dir_all(p)?;
            }
            link_or_copy(&f.image, &dst)?;
        }
    }
    if wants(Format::Yolo) && !frames.is_empty() {
        for f in &frames {
            let txt = yolo_txt(f, classes)?;
            let stem = f.name.trim_end_matches(".jpg");
            write_file(
                &out.join("labels")
                    .join(f.split.as_str())
                    .join(format!("{stem}.txt")),
                txt.as_bytes(),
            )?;
        }
        let mut names = classes.join("\n");
        names.push('\n');
        write_file(&out.join("classes.txt"), names.as_bytes())?;
        let val = if r.val > 0 {
            "images/val"
        } else {
            warnings.push(
                "no frame was held out for validation; data.yaml validates on the training \
                 images (name holdout cameras or days, or review more cameras/days)"
                    .into(),
            );
            "images/train"
        };
        let mut yaml = format!(
            "# Blue Onyx Prism review export (Ultralytics dataset)\npath: {}\ntrain: images/train\nval: {val}\nnames:\n",
            serde_json::to_string(&out.to_string_lossy())?
        );
        for (i, c) in classes.iter().enumerate() {
            yaml.push_str(&format!("  {i}: {}\n", serde_json::to_string(c)?));
        }
        write_file(&out.join("data.yaml"), yaml.as_bytes())?;
        r.paths.insert("yolo".into(), out.join("data.yaml"));
    }
    if wants(Format::Coco) && !frames.is_empty() {
        for s in [Split::Train, Split::Val] {
            let list = by_split(s);
            if list.is_empty() {
                continue;
            }
            let v = coco_json(&list, classes)?;
            write_file(
                &out.join("annotations")
                    .join(format!("instances_{}.json", s.as_str())),
                &json_bytes(&v)?,
            )?;
        }
        r.paths.insert("coco".into(), out.join("annotations"));
    }
    if wants(Format::XAnyLabeling) {
        let xdir = out.join("xanylabeling");
        std::fs::create_dir_all(&xdir)?;
        for f in &frames {
            link_or_copy(&f.image, &xdir.join(&f.name))?;
            let x = xany::to_xany(&f.name, f.width, f.height, f.boxes(), true);
            let stem = f.name.trim_end_matches(".jpg");
            write_file(&xdir.join(format!("{stem}.json")), &json_bytes(&x)?)?;
        }
        for (frame, labels, w, h) in &unreviewed {
            let name = flat_name(&frame.rel);
            link_or_copy(&frame.jpg, &xdir.join(&name))?;
            let preview: Vec<LabelBox> = labels
                .boxes
                .iter()
                .map(|b| LabelBox {
                    state: super::resolve_pending(b),
                    ..b.clone()
                })
                .filter(LabelBox::is_training)
                .collect();
            let x = xany::to_xany(&name, *w, *h, &preview, false);
            let stem = name.trim_end_matches(".jpg");
            write_file(&xdir.join(format!("{stem}.json")), &json_bytes(&x)?)?;
        }
        let mut names = classes.join("\n");
        names.push('\n');
        write_file(&xdir.join("classes.txt"), names.as_bytes())?;
        r.paths.insert("xanylabeling".into(), xdir);
    }
    if wants(Format::Benchmark) && !frames.is_empty() {
        let bdir = out.join("benchmark");
        let list: Vec<&ExportFrame> = frames
            .iter()
            .filter(|f| match req.benchmark_split {
                BenchSplit::All => true,
                BenchSplit::Train => f.split == Split::Train,
                BenchSplit::Val => f.split == Split::Val,
            })
            .collect();
        if list.is_empty() {
            warnings.push("the benchmark set is empty for the chosen split; not written".into());
        } else {
            let img = bdir.join("images");
            std::fs::create_dir_all(&img)?;
            for f in &list {
                link_or_copy(&f.image, &img.join(&f.name))?;
            }
            let id = out
                .file_name()
                .map_or_else(|| "review".into(), |n| n.to_string_lossy().into_owned());
            let m = benchmark_manifest(&list, classes, &format!("review-{id}"));
            write_file(&bdir.join("manifest.json"), &json_bytes(&m)?)?;
            r.benchmark_dataset = Some(format!(
                "{}{}",
                crate::benchmark::images::DIR_PREFIX,
                bdir.display()
            ));
            r.paths.insert("benchmark".into(), bdir);
        }
    }
    let negatives: Vec<serde_json::Value> = frames
        .iter()
        .flat_map(|f| {
            f.labels.rejected().map(move |b| {
                serde_json::json!({
                    "frame": f.rel.jpg(),
                    "file": f.name,
                    "split": f.split,
                    "label": b.label,
                    "bbox": b.bbox,
                    "score": b.score,
                    "source": b.source,
                    "origin": b.origin,
                    "reason": b.reason,
                })
            })
        })
        .collect();
    r.hard_negative_boxes = negatives.len();
    if !frames.is_empty() {
        write_file(&out.join("hard_negatives.json"), &json_bytes(&negatives)?)?;
    }
    r.warnings = warnings;
    write_file(&out.join("export.json"), &json_bytes(&r)?)?;
    Ok(r)
}

/// The review queue's frames that are not reviewed or skipped, with their labels (saved or
/// pre-filled) and size.
fn unreviewed_queue_frames(
    dir: &Path,
    warnings: &mut Vec<String>,
) -> Vec<(Frame, LabelFile, u32, u32)> {
    let mut out = Vec::new();
    for c in super::queue::candidates(dir, super::queue::QueueSource::Queue) {
        let Some(frame) = Frame::resolve(dir, &c.rel) else {
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
            None => super::prefill(&rec),
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
}
