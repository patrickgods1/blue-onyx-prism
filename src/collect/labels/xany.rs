//! X-AnyLabeling interop. X-AnyLabeling (github.com/CVHub520/X-AnyLabeling) saves one
//! LabelMe-style JSON per image beside it: `{version, flags, shapes, imagePath, imageData,
//! imageHeight, imageWidth, description, checked}`, each shape `{label, score, points,
//! group_id, description, difficult, shape_type, flags, attributes, kie_linking}`. Its
//! `label_file.py` saves a rectangle as four corner points (`[xmin, ymin]`, `[xmax, ymin]`,
//! `[xmax, ymax]`, `[xmin, ymax]`) and loads two-point (diagonal) rectangles too.
//!
//! Export ([`to_xany`]) writes the training boxes of a frame as rectangles, `checked: true` for
//! a reviewed frame. Import ([`import_dir`], [`apply_import`]) reads such files back: a file
//! counts as reviewed when `checked` is true (or the caller treats unchecked files as reviewed)
//! and its shapes become the frame's complete set of training boxes (see [`apply_import`]).

use super::export::parse_flat_name;
use super::{BoxState, Frame, LabelBox, LabelFile, Source, Status, iou, normalize};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// `version` written in exported files (X-AnyLabeling only checks it is present).
pub const XANY_VERSION: &str = "2.5.4";
/// IoU at or above which an imported rectangle is an existing box (moved or relabeled).
pub const IMPORT_MATCH_IOU: f32 = 0.9;
/// Most JSON files read by one import.
pub const MAX_IMPORT_FILES: usize = 100_000;
/// Largest X-AnyLabeling JSON read (embedded `imageData` makes files large).
pub const MAX_IMPORT_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// One shape of an X-AnyLabeling file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XShape {
    pub label: String,
    #[serde(default)]
    pub score: Option<f64>,
    pub points: Vec<Vec<f64>>,
    #[serde(default)]
    pub group_id: Option<i64>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub difficult: bool,
    #[serde(default = "polygon")]
    pub shape_type: String,
    #[serde(default)]
    pub flags: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub attributes: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub kie_linking: Vec<serde_json::Value>,
}

fn polygon() -> String {
    "polygon".into()
}

/// An X-AnyLabeling file (unknown keys are ignored on import).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct XFile {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub flags: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub shapes: Vec<XShape>,
    pub image_path: String,
    #[serde(default)]
    pub image_data: Option<String>,
    #[serde(default)]
    pub image_height: Option<i64>,
    #[serde(default)]
    pub image_width: Option<i64>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub checked: bool,
}

fn round2(x: f32) -> f64 {
    (f64::from(x) * 100.0).round() / 100.0
}

/// The X-AnyLabeling file of a frame saved as `name` (`w` x `h`): one rectangle per box of
/// `boxes` (the caller picks training boxes), `checked` as given.
pub fn to_xany<'a>(
    name: &str,
    w: u32,
    h: u32,
    boxes: impl IntoIterator<Item = &'a LabelBox>,
    checked: bool,
) -> XFile {
    let shapes = boxes
        .into_iter()
        .map(|b| {
            let [x0, y0, x1, y1] = b.bbox.map(round2);
            XShape {
                label: b.label.clone(),
                score: b
                    .score
                    .map(|s| (f64::from(s) * 10_000.0).round() / 10_000.0),
                points: vec![vec![x0, y0], vec![x1, y0], vec![x1, y1], vec![x0, y1]],
                group_id: None,
                description: Some(String::new()),
                difficult: b.difficult,
                shape_type: "rectangle".into(),
                flags: Some(serde_json::Map::new()),
                attributes: serde_json::Map::new(),
                kie_linking: Vec::new(),
            }
        })
        .collect();
    XFile {
        version: Some(XANY_VERSION.into()),
        flags: serde_json::Map::new(),
        shapes,
        image_path: name.to_string(),
        image_data: None,
        image_height: Some(i64::from(h)),
        image_width: Some(i64::from(w)),
        description: Some(String::new()),
        checked,
    }
}

/// An imported rectangle: label, `[x0, y0, x1, y1]`, score, unsure (`difficult`).
pub type Rect = (String, [f32; 4], Option<f32>, bool);

/// The rectangles of `x`: `rectangle` (2 or 4 points), `polygon` and `rotation` shapes (their
/// bounding box). Other shape types (points, lines, circles) are skipped with a warning.
#[allow(
    clippy::cast_possible_truncation,
    reason = "pixel coordinates of camera frames fit f32"
)]
pub fn rects(x: &XFile) -> (Vec<Rect>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    for (i, s) in x.shapes.iter().enumerate() {
        let pts: Vec<(f64, f64)> = s
            .points
            .iter()
            .filter_map(|p| match p.as_slice() {
                [a, b, ..] if a.is_finite() && b.is_finite() => Some((*a, *b)),
                _ => None,
            })
            .collect();
        let need = match s.shape_type.as_str() {
            "rectangle" => 2,
            "polygon" | "rotation" => 3,
            other => {
                warnings.push(format!(
                    "shape {} ({}): a '{other}' shape is not a box; skipped",
                    i + 1,
                    s.label
                ));
                continue;
            }
        };
        if pts.len() < need || pts.len() != s.points.len() {
            warnings.push(format!(
                "shape {} ({}): {} point(s), not a valid {}; skipped",
                i + 1,
                s.label,
                s.points.len(),
                s.shape_type
            ));
            continue;
        }
        let fold = |f: fn(f64, f64) -> f64, init: f64, sel: fn(&(f64, f64)) -> f64| {
            pts.iter().map(sel).fold(init, f)
        };
        let b = [
            fold(f64::min, f64::INFINITY, |p| p.0) as f32,
            fold(f64::min, f64::INFINITY, |p| p.1) as f32,
            fold(f64::max, f64::NEG_INFINITY, |p| p.0) as f32,
            fold(f64::max, f64::NEG_INFINITY, |p| p.1) as f32,
        ];
        let score = s
            .score
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .map(|v| v as f32);
        out.push((s.label.trim().to_string(), b, score, s.difficult));
    }
    (out, warnings)
}

/// `base` (the frame's saved labels, else its pre-fill) with the imported rectangles as the
/// complete set of training boxes, reviewed:
///
/// - a rectangle matching a box of `base` (IoU >= [`IMPORT_MATCH_IOU`]; a box of the same label
///   preferred, then the highest IoU; each box matched once) takes that box's place: its
///   coordinates and label (the old label kept in `model_label`), accepted (added for a human
///   box);
/// - any other rectangle is a new human box (`added`);
/// - every box of `base` no rectangle matched is rejected (a model or auto box: hard-negative
///   evidence) or dropped (a human box);
/// - `empty_confirmed` = no rectangle.
pub fn apply_import(base: LabelFile, rects: &[Rect]) -> LabelFile {
    let mut boxes = base.boxes;
    let mut matched = vec![false; boxes.len()];
    let mut added = Vec::new();
    for (label, bbox, score, difficult) in rects {
        let best = boxes
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched.get(*i).copied().unwrap_or(true))
            .map(|(i, b)| (i, b.label == *label, iou(&b.bbox, bbox)))
            .filter(|(_, _, v)| *v >= IMPORT_MATCH_IOU)
            .max_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)))
            .map(|(i, _, _)| i);
        match best.and_then(|i| Some((i, boxes.get_mut(i)?))) {
            Some((i, b)) => {
                if let Some(m) = matched.get_mut(i) {
                    *m = true;
                }
                if b.label != *label {
                    if b.model_label.is_none() {
                        b.model_label = Some(b.label.clone());
                    }
                    b.label = label.clone();
                }
                b.bbox = *bbox;
                b.difficult = *difficult;
                b.state = if b.source == Source::Human {
                    BoxState::Added
                } else {
                    BoxState::Accepted
                };
            }
            None => added.push(LabelBox {
                label: label.clone(),
                bbox: *bbox,
                source: Source::Human,
                score: *score,
                state: BoxState::Added,
                origin: None,
                reason: None,
                model_label: None,
                orig_bbox: None,
                difficult: *difficult,
                reject_reason: None,
                auto: Default::default(),
            }),
        }
    }
    let mut out: Vec<LabelBox> = boxes
        .into_iter()
        .zip(matched)
        .filter_map(|(mut b, m)| {
            if m {
                return Some(b);
            }
            if b.source == Source::Human {
                return None;
            }
            b.state = BoxState::Rejected;
            Some(b)
        })
        .collect();
    out.extend(added);
    LabelFile {
        status: Status::Reviewed,
        empty_confirmed: rects.is_empty(),
        boxes: out,
        ..LabelFile::default()
    }
}

/// What [`import_dir`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ImportReport {
    /// Frames whose labels were written.
    pub imported: usize,
    /// Files left alone because `checked` is false.
    pub unchecked: usize,
    /// Files naming no frame of the dataset.
    pub unmatched: usize,
    /// Per-file problems (file: message).
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// The frame an X-AnyLabeling file is for: its `imagePath` file name (any folder dropped), else
/// the JSON's own stem, as an exported flat name (`<scene>.<camera>.<date>.<stem>.jpg`).
fn frame_of(x: &XFile, json_name: &str) -> Option<String> {
    let base = x
        .image_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .to_string();
    parse_flat_name(&base)
        .or_else(|| {
            let stem = json_name.strip_suffix(".json")?;
            parse_flat_name(&format!("{stem}.jpg"))
        })
        .map(|r| r.jpg())
}

/// Import every `*.json` directly in `src` into the dataset `dir` (see the module doc). A file
/// whose image size differs from the frame's (a resized copy) or whose labels are not in
/// `classes` is an error for that file only. A frame without a label file starts from its
/// pre-fill without the `hidden` classes ([`super::prefill`]). Returns the report; fails only
/// when `src` cannot be listed.
pub fn import_dir(
    dir: &Path,
    src: &Path,
    classes: &[String],
    hidden: &[String],
    treat_unchecked_as_reviewed: bool,
    now: &str,
) -> Result<ImportReport> {
    let mut names: Vec<String> = std::fs::read_dir(src)
        .with_context(|| format!("reading {}", src.display()))?
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.ends_with(".json") && !n.starts_with('.'))
        .collect();
    names.sort();
    if names.len() > MAX_IMPORT_FILES {
        bail!(
            "{} holds {} JSON files; at most {MAX_IMPORT_FILES} per import",
            src.display(),
            names.len()
        );
    }
    let mut r = ImportReport::default();
    for name in names {
        let path = src.join(&name);
        let one = (|| -> Result<Option<()>> {
            let bytes = super::read_capped(&path, MAX_IMPORT_FILE_BYTES)?;
            let Ok(x) = serde_json::from_slice::<XFile>(&bytes) else {
                // Not an X-AnyLabeling file (export.json, a classes file, ...).
                return Ok(None);
            };
            let Some(rel) = frame_of(&x, &name) else {
                r.unmatched += 1;
                return Ok(Some(()));
            };
            let Some(frame) = Frame::resolve(dir, &rel) else {
                r.unmatched += 1;
                return Ok(Some(()));
            };
            if !(x.checked || treat_unchecked_as_reviewed) {
                r.unchecked += 1;
                return Ok(Some(()));
            }
            let rec = frame.record()?;
            if let (Some(iw), Some(ih)) = (x.image_width, x.image_height)
                && iw > 0
                && ih > 0
                && (iw != i64::from(rec.width) || ih != i64::from(rec.height))
            {
                bail!(
                    "the image is {iw}x{ih} but the frame is {}x{}; label the original frame",
                    rec.width,
                    rec.height
                );
            }
            let (rects, warnings) = rects(&x);
            r.warnings
                .extend(warnings.into_iter().map(|w| format!("{name}: {w}")));
            let base = frame
                .labels()?
                .unwrap_or_else(|| super::prefill(&rec, classes, hidden));
            let classes_version = base.classes_version.clone();
            let mut merged = apply_import(base, &rects);
            merged.classes_version = classes_version;
            let f = normalize(merged, classes, rec.width, rec.height, now)?;
            super::write(&frame.labels, &f)?;
            r.imported += 1;
            Ok(Some(()))
        })();
        if let Err(e) = one {
            r.errors.push(format!("{name}: {e:#}"));
        }
    }
    Ok(r)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::{Origin, prefill};
    use super::*;

    fn model_box(label: &str, b: [f32; 4], state: BoxState) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: b,
            source: Source::Model,
            score: Some(0.8),
            state,
            origin: Some(Origin::Prediction),
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    #[test]
    fn export_writes_four_corner_rectangles() {
        let b = model_box("person", [10.0, 20.0, 30.5, 40.25], BoxState::Accepted);
        let x = to_xany("ir.cam.2026-10-22.s.jpg", 640, 480, [&b], true);
        let v = serde_json::to_value(&x).unwrap();
        assert_eq!(v["imagePath"], "ir.cam.2026-10-22.s.jpg");
        assert_eq!(v["imageWidth"], 640);
        assert_eq!(v["imageHeight"], 480);
        assert_eq!(v["imageData"], serde_json::Value::Null);
        assert_eq!(v["checked"], true);
        let s = &v["shapes"][0];
        assert_eq!(s["shape_type"], "rectangle");
        assert_eq!(
            s["points"],
            serde_json::json!([[10.0, 20.0], [30.5, 20.0], [30.5, 40.25], [10.0, 40.25]])
        );
        assert_eq!(s["score"], 0.8);
        assert_eq!(s["kie_linking"], serde_json::json!([]));
    }

    #[test]
    fn reads_two_point_rectangles_polygons_and_skips_others() {
        let x: XFile = serde_json::from_str(
            r#"{"version": "2.4.0", "flags": {}, "imagePath": "a.jpg", "imageData": null,
                "imageHeight": 480, "imageWidth": 640, "extra": 1,
                "shapes": [
                  {"label": "dog", "points": [[50, 60], [10, 20]], "shape_type": "rectangle"},
                  {"label": "cat", "points": [[1, 1], [9, 2], [5, 8]], "shape_type": "polygon", "score": 0.4, "difficult": true},
                  {"label": "p", "points": [[1, 1]], "shape_type": "point"},
                  {"label": "bad", "points": [[1, 1]], "shape_type": "rectangle"}
                ]}"#,
        )
        .unwrap();
        let (r, w) = rects(&x);
        assert_eq!(r[0], ("dog".into(), [10.0, 20.0, 50.0, 60.0], None, false));
        assert_eq!(r[1], ("cat".into(), [1.0, 1.0, 9.0, 8.0], Some(0.4), true));
        assert_eq!(r.len(), 2);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(!x.checked);
    }

    #[test]
    fn import_replaces_training_boxes_and_keeps_evidence() {
        let base = LabelFile {
            boxes: vec![
                model_box("person", [10.0, 10.0, 50.0, 90.0], BoxState::Pending),
                model_box("dog", [100.0, 100.0, 150.0, 140.0], BoxState::Accepted),
                LabelBox {
                    source: Source::Human,
                    ..model_box("cat", [300.0, 300.0, 340.0, 340.0], BoxState::Added)
                },
            ],
            ..LabelFile::default()
        };
        let rects = vec![
            // The person, moved a little and relabeled.
            ("raccoon".to_string(), [10.0, 10.0, 50.0, 88.0], None, false),
            // New.
            ("deer".to_string(), [400.0, 10.0, 500.0, 100.0], None, false),
        ];
        let f = apply_import(base.clone(), &rects);
        assert_eq!(f.status, Status::Reviewed);
        assert!(!f.empty_confirmed);
        let got: Vec<(&str, BoxState, Source)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.state, b.source))
            .collect();
        assert_eq!(
            got,
            [
                ("raccoon", BoxState::Accepted, Source::Model),
                ("dog", BoxState::Rejected, Source::Model),
                ("deer", BoxState::Added, Source::Human),
            ]
        );
        assert_eq!(f.boxes[0].model_label.as_deref(), Some("person"));
        assert_eq!(f.boxes[0].bbox, [10.0, 10.0, 50.0, 88.0]);
        let empty = apply_import(base, &[]);
        assert!(empty.empty_confirmed);
        assert_eq!(empty.training().count(), 0);
        assert_eq!(empty.rejected().count(), 2);
    }

    #[test]
    fn round_trip_through_the_files() {
        let mut rec = super::super::super::record::empty_record();
        rec.width = 640;
        rec.height = 480;
        rec.predictions = vec![super::super::super::record::Det {
            label: "person".into(),
            score: 0.81,
            bbox: [10.0, 10.0, 50.0, 90.0],
        }];
        let classes = super::super::super::config::default_classes();
        let mut reviewed = prefill(&rec, &classes, &[]);
        reviewed.status = Status::Reviewed;
        reviewed.boxes.push(LabelBox {
            label: "dog".into(),
            bbox: [100.5, 100.25, 150.0, 140.0],
            source: Source::Human,
            score: None,
            state: BoxState::Added,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        });
        let reviewed = normalize(reviewed, &classes, 640, 480, "t").unwrap();
        let x = to_xany("n.jpg", 640, 480, reviewed.training(), true);
        let text = serde_json::to_string_pretty(&x).unwrap();
        let back: XFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back, x);
        let (r, w) = rects(&back);
        assert!(w.is_empty());
        let again = normalize(apply_import(reviewed.clone(), &r), &classes, 640, 480, "t").unwrap();
        assert_eq!(again, reviewed, "an unchanged round trip changes nothing");
    }

    #[test]
    fn unsure_round_trips_through_the_files() {
        let unsure = LabelBox {
            difficult: true,
            ..model_box("person", [10.0, 10.0, 50.0, 90.0], BoxState::Accepted)
        };
        let sure = model_box("dog", [100.0, 100.0, 150.0, 140.0], BoxState::Accepted);
        let x = to_xany("n.jpg", 640, 480, [&unsure, &sure], true);
        let v = serde_json::to_value(&x).unwrap();
        assert_eq!(v["shapes"][0]["difficult"], true);
        assert_eq!(v["shapes"][1]["difficult"], false);
        let (r, _) = rects(&x);
        // Marked unsure in X-AnyLabeling: the matched box and a new one take the flag.
        let mut flipped = r.clone();
        flipped[0].3 = false;
        flipped[1].3 = true;
        flipped.push(("cat".into(), [300.0, 300.0, 340.0, 340.0], None, true));
        let base = LabelFile {
            boxes: vec![unsure, sure],
            ..LabelFile::default()
        };
        let f = apply_import(base, &flipped);
        let got: Vec<(&str, bool)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.difficult))
            .collect();
        assert_eq!(got, [("person", false), ("dog", true), ("cat", true)]);
    }
}
