//! The writers of [`super::build_export`], one per part of an export, in the order it runs them
//! ([`plan`]): the images YOLO and COCO share, YOLO, COCO, X-AnyLabeling, the benchmark set, then
//! the extras every export has (`hard_negatives.json`, `frames.json`, `ignore.json`). Each writer
//! that reports its files in [`super::ExportStep::Writing`] has its count beside it
//! ([`Writer::files`]), so the count shown up front ([`super::written_files`]) is the sum of what
//! the chosen writers then write.

use super::{
    BenchSplit, ExportFrame, ExportReport, ExportRequest, FRAMES_FILE, Format, HARD_NEGATIVES_FILE,
    IGNORE_FILE, Split, Written, benchmark_manifest, coco_json, flat_name, frame_notes,
    ignore_file, json_bytes, link_or_copy, preview, write_file, xany, yolo_txt,
};
use super::{Frame, LabelFile};
use anyhow::Result;
use std::path::Path;

/// An unreviewed queue frame of the X-AnyLabeling folder: its frame, its labels (saved or
/// pre-filled) and its size ([`super::unreviewed_queue_frames`]).
pub(super) type QueueFrame = (Frame, LabelFile, u32, u32);

/// The folder an export will be at (`out`: every path a reader sees names it) and the folder it
/// is written into meanwhile (`build`, [`super::Staging`]).
#[derive(Clone, Copy)]
pub(super) struct Dirs<'a> {
    pub out: &'a Path,
    pub build: &'a Path,
}

/// One part of an export, in [`plan`]'s order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Writer {
    /// `images/<split>/`, for YOLO and COCO.
    Images,
    Yolo,
    Coco,
    XAnyLabeling,
    Benchmark,
}

/// The writers an export of `req` runs, in order: every format needs a reviewed frame
/// (`has_frames`) but X-AnyLabeling, which may hold the queue's unreviewed frames only.
pub(super) fn plan(req: &ExportRequest, has_frames: bool) -> Vec<Writer> {
    let wants = |f: Format| req.formats.contains(&f);
    let mut out = Vec::new();
    if (wants(Format::Yolo) || wants(Format::Coco)) && has_frames {
        out.push(Writer::Images);
    }
    if wants(Format::Yolo) && has_frames {
        out.push(Writer::Yolo);
    }
    if wants(Format::Coco) && has_frames {
        out.push(Writer::Coco);
    }
    if wants(Format::XAnyLabeling) {
        out.push(Writer::XAnyLabeling);
    }
    if wants(Format::Benchmark) && has_frames {
        out.push(Writer::Benchmark);
    }
    out
}

impl Writer {
    /// The files this writer counts in [`super::ExportStep::Writing`] for `frames`, `unreviewed`
    /// queue frames and the benchmark split `bench`.
    pub fn files(self, frames: &[ExportFrame], unreviewed: usize, bench: BenchSplit) -> usize {
        match self {
            Writer::Images => images_files(frames),
            Writer::Yolo => yolo_files(frames),
            Writer::Coco => coco_files(),
            Writer::XAnyLabeling => xany_files(frames, unreviewed),
            Writer::Benchmark => benchmark_files(frames, bench),
        }
    }
}

/// Whether the frame `f` is in the benchmark split `bench`.
pub(super) fn in_bench(bench: BenchSplit, f: &ExportFrame) -> bool {
    match bench {
        BenchSplit::All => true,
        BenchSplit::Train => f.split == Split::Train,
        BenchSplit::Val => f.split == Split::Val,
    }
}

/// The class list, one per line.
fn class_lines(classes: &[String]) -> String {
    let mut names = classes.join("\n");
    names.push('\n');
    names
}

/// [`write_images`]' files: one image per frame.
fn images_files(frames: &[ExportFrame]) -> usize {
    frames.len()
}

/// Every frame's image under `images/<split>/` (hard-linked, else copied).
pub(super) fn write_images(
    dirs: Dirs<'_>,
    frames: &[ExportFrame],
    written: &mut Written<'_>,
) -> Result<()> {
    for f in frames {
        let dst = dirs
            .build
            .join("images")
            .join(f.split.as_str())
            .join(&f.name);
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p)?;
        }
        link_or_copy(&f.image, &dst)?;
        written.one()?;
    }
    Ok(())
}

/// [`write_yolo`]'s files: one label file per frame (`classes.txt` and `data.yaml` not counted).
fn yolo_files(frames: &[ExportFrame]) -> usize {
    frames.len()
}

/// The YOLO labels (`labels/<split>/<stem>.txt`), `classes.txt` and `data.yaml` (validating on the
/// training images, with a warning, when no frame is held out); its path in `r.paths`.
pub(super) fn write_yolo(
    dirs: Dirs<'_>,
    frames: &[ExportFrame],
    classes: &[String],
    written: &mut Written<'_>,
    (r, warnings): (&mut ExportReport, &mut Vec<String>),
) -> Result<()> {
    let build = dirs.build;
    for f in frames {
        let txt = yolo_txt(f, classes)?;
        let stem = f.name.trim_end_matches(".jpg");
        write_file(
            &build
                .join("labels")
                .join(f.split.as_str())
                .join(format!("{stem}.txt")),
            txt.as_bytes(),
        )?;
        written.one()?;
    }
    write_file(&build.join("classes.txt"), class_lines(classes).as_bytes())?;
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
        serde_json::to_string(&dirs.out.to_string_lossy())?
    );
    for (i, c) in classes.iter().enumerate() {
        yaml.push_str(&format!("  {i}: {}\n", serde_json::to_string(c)?));
    }
    write_file(&build.join("data.yaml"), yaml.as_bytes())?;
    r.paths.insert("yolo".into(), dirs.out.join("data.yaml"));
    Ok(())
}

/// [`write_coco`]'s files: none counted (one annotation file per split).
const fn coco_files() -> usize {
    0
}

/// The COCO instances of each split that has frames (`annotations/instances_<split>.json`); its
/// path in `r.paths`.
pub(super) fn write_coco(
    dirs: Dirs<'_>,
    frames: &[ExportFrame],
    classes: &[String],
    r: &mut ExportReport,
) -> Result<()> {
    for s in [Split::Train, Split::Val] {
        let list: Vec<&ExportFrame> = frames.iter().filter(|f| f.split == s).collect();
        if list.is_empty() {
            continue;
        }
        let v = coco_json(&list, classes)?;
        write_file(
            &dirs
                .build
                .join("annotations")
                .join(format!("instances_{}.json", s.as_str())),
            &json_bytes(&v)?,
        )?;
    }
    r.paths.insert("coco".into(), dirs.out.join("annotations"));
    Ok(())
}

/// [`write_xany`]'s files: an image and a JSON file per reviewed and per unreviewed frame
/// (`classes.txt` not counted).
fn xany_files(frames: &[ExportFrame], unreviewed: usize) -> usize {
    2 * (frames.len() + unreviewed)
}

/// The X-AnyLabeling folder: every reviewed frame with all its training boxes, then the queue's
/// `unreviewed` frames pre-filled and unchecked, and `classes.txt`; its path in `r.paths`.
pub(super) fn write_xany(
    dirs: Dirs<'_>,
    (frames, unreviewed): (&[ExportFrame], &[QueueFrame]),
    classes: &[String],
    written: &mut Written<'_>,
    r: &mut ExportReport,
) -> Result<()> {
    let xdir = dirs.build.join("xanylabeling");
    std::fs::create_dir_all(&xdir)?;
    for f in frames {
        link_or_copy(&f.image, &xdir.join(&f.name))?;
        written.one()?;
        // Every training box, the floor regardless: an import treats a box missing here as
        // rejected, and `difficult` on it would stay in the label file for good.
        let x = xany::to_xany(&f.name, f.width, f.height, f.training_boxes(), true);
        let stem = f.name.trim_end_matches(".jpg");
        write_file(&xdir.join(format!("{stem}.json")), &json_bytes(&x)?)?;
        written.one()?;
    }
    for (frame, labels, w, h) in unreviewed {
        let name = flat_name(&frame.rel);
        link_or_copy(&frame.jpg, &xdir.join(&name))?;
        written.one()?;
        let preview = preview(labels).boxes;
        let x = xany::to_xany(&name, *w, *h, &preview, false);
        let stem = name.trim_end_matches(".jpg");
        write_file(&xdir.join(format!("{stem}.json")), &json_bytes(&x)?)?;
        written.one()?;
    }
    write_file(&xdir.join("classes.txt"), class_lines(classes).as_bytes())?;
    r.paths
        .insert("xanylabeling".into(), dirs.out.join("xanylabeling"));
    Ok(())
}

/// [`write_benchmark`]'s files: one image per frame of the split (the manifest not counted).
fn benchmark_files(frames: &[ExportFrame], bench: BenchSplit) -> usize {
    frames.iter().filter(|f| in_bench(bench, f)).count()
}

/// The benchmark set of the frames in `bench` (`benchmark/images/` + `manifest.json`) and its
/// `dir:` name in `r`; none, with a warning, when no frame is in that split.
pub(super) fn write_benchmark(
    dirs: Dirs<'_>,
    (frames, bench): (&[ExportFrame], BenchSplit),
    classes: &[String],
    written: &mut Written<'_>,
    (r, warnings): (&mut ExportReport, &mut Vec<String>),
) -> Result<()> {
    let bdir = dirs.build.join("benchmark");
    let list: Vec<&ExportFrame> = frames.iter().filter(|f| in_bench(bench, f)).collect();
    if list.is_empty() {
        warnings.push(
            "the benchmark set was not written: no frame is in the chosen split (a share of \
             0 holds nothing out); hold out some cameras or days, or choose another set"
                .into(),
        );
        return Ok(());
    }
    let img = bdir.join("images");
    std::fs::create_dir_all(&img)?;
    for f in &list {
        link_or_copy(&f.image, &img.join(&f.name))?;
        written.one()?;
    }
    let id = dirs
        .out
        .file_name()
        .map_or_else(|| "review".into(), |n| n.to_string_lossy().into_owned());
    let m = benchmark_manifest(&list, classes, &format!("review-{id}"));
    write_file(&bdir.join("manifest.json"), &json_bytes(&m)?)?;
    let published = dirs.out.join("benchmark");
    r.benchmark_dataset = Some(format!(
        "{}{}",
        crate::benchmark::images::DIR_PREFIX,
        published.display()
    ));
    r.paths.insert("benchmark".into(), published);
    Ok(())
}

/// What every export has: the rejected boxes (counted in `r` always; [`HARD_NEGATIVES_FILE`]),
/// the frame notes ([`FRAMES_FILE`]) and the ignore regions ([`IGNORE_FILE`], counted in `r`),
/// the files written only with a reviewed frame. None is counted in
/// [`super::ExportStep::Writing`].
pub(super) fn write_extras(
    dirs: Dirs<'_>,
    frames: &[ExportFrame],
    req: &ExportRequest,
    r: &mut ExportReport,
) -> Result<()> {
    let negatives: Vec<serde_json::Value> = frames
        .iter()
        .flat_map(|f| {
            f.labels.hard_negatives().map(move |b| {
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
                    "reject_reason": b.reject_reason,
                })
            })
        })
        .collect();
    r.hard_negative_boxes = negatives.len();
    if !frames.is_empty() {
        let build = dirs.build;
        write_file(&build.join(HARD_NEGATIVES_FILE), &json_bytes(&negatives)?)?;
        write_file(&build.join(FRAMES_FILE), &json_bytes(&frame_notes(frames))?)?;
        let mut ignore = ignore_file(frames);
        ignore.size_floor = req.size_floor;
        r.ignore_boxes = ignore.images.values().map(Vec::len).sum();
        write_file(&build.join(IGNORE_FILE), &json_bytes(&ignore)?)?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn request(formats: &[Format]) -> ExportRequest {
        ExportRequest {
            formats: formats.to_vec(),
            out_dir: std::path::PathBuf::new(),
            split: super::super::SplitConfig::default(),
            benchmark_split: BenchSplit::Val,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
            size_floor: None,
            heldout: None,
        }
    }

    #[test]
    fn the_writers_run_in_one_order_and_only_x_anylabeling_without_a_reviewed_frame() {
        use Writer::*;
        assert_eq!(
            plan(&request(&Format::ALL), true),
            [Images, Yolo, Coco, XAnyLabeling, Benchmark]
        );
        // The order is the writers', not the request's.
        assert_eq!(
            plan(&request(&[Format::Benchmark, Format::Coco]), true),
            [Images, Coco, Benchmark]
        );
        assert_eq!(plan(&request(&[Format::Yolo]), true), [Images, Yolo]);
        assert_eq!(plan(&request(&Format::ALL), false), [XAnyLabeling]);
        assert!(plan(&request(&[Format::Yolo, Format::Benchmark]), false).is_empty());
    }

    #[test]
    fn each_writer_counts_what_it_reports() {
        let frame = |split| ExportFrame {
            rel: super::super::layout::parse_rel("day/cam/2026-10-22/a.jpg").unwrap(),
            name: "day.cam.2026-10-22.a.jpg".into(),
            image: std::path::PathBuf::from("a.jpg"),
            width: 64,
            height: 48,
            buckets: Vec::new(),
            split,
            labels: LabelFile::default(),
            floor: None,
        };
        let frames = [frame(Split::Train), frame(Split::Val), frame(Split::Val)];
        let files = |w: Writer, bench| w.files(&frames, 4, bench);
        assert_eq!(files(Writer::Images, BenchSplit::All), 3);
        assert_eq!(files(Writer::Yolo, BenchSplit::All), 3);
        assert_eq!(files(Writer::Coco, BenchSplit::All), 0);
        assert_eq!(files(Writer::XAnyLabeling, BenchSplit::All), 14);
        assert_eq!(files(Writer::Benchmark, BenchSplit::All), 3);
        assert_eq!(files(Writer::Benchmark, BenchSplit::Val), 2);
        assert_eq!(files(Writer::Benchmark, BenchSplit::Train), 1);
    }
}
