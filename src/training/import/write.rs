//! Writing the selected frames of an import into the dataset folder, in the collector's layout
//! (`<dir>/<scene>/<camera>/<date>/<stem>.{jpg,json}` + `index.jsonl`, `docs/DATASET.md`): a
//! version-1 sidecar without detections, `source: "import:<id>"` and its provenance; then the
//! frames are queued for review (`review-queue.jsonl`) and the import is listed in
//! `<dir>/imports.jsonl`. Each file is written atomically, image first, then sidecar, then the
//! index line, like the collector does.

use crate::collect::layout::{self, Rel};
use crate::collect::record::{self, ImageInfo, IndexLine, Provenance, Record, Stats};
use crate::image_stats::ImageStats;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The list of imports in the dataset folder.
pub const IMPORTS_FILE: &str = "imports.jsonl";

/// One frame to write.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The image file to copy (a JPEG from the extraction, or the source image).
    pub file: PathBuf,
    /// The original file name (Blue Iris names carry the camera).
    pub image_name: String,
    pub camera: String,
    /// Recording time of the frame (the camera folder's date and the file name come from it).
    pub time: SystemTime,
    pub width: u32,
    pub height: u32,
    pub stats: ImageStats,
    pub dhash: u64,
    pub provenance: Provenance,
}

/// One line of `imports.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportLine {
    pub id: String,
    pub created: String,
    /// What was imported (file name, URL or dataset).
    pub source: String,
    /// `path | url | dataset`.
    pub kind: String,
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    pub frames: usize,
    pub cameras: Vec<String>,
    /// Frames per scene.
    pub scenes: std::collections::BTreeMap<String, usize>,
}

/// The imports listed in `dir` (newest last; lines that do not parse are skipped).
pub fn read_imports(dir: &Path) -> Vec<ImportLine> {
    std::fs::read_to_string(dir.join(IMPORTS_FILE))
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn stem_of(time: SystemTime, id4: &str, n: usize) -> String {
    format!("{}_imp-{id4}-{n:05}", crate::collect::utc_file_ts(time))
}

/// The record of an imported frame (no detections).
fn record_of(f: &Frame, rel: &Rel, import_id: &str, bytes: u64, original: bool) -> Record {
    let mut r = record::empty_record();
    r.id = rel.stem.clone();
    r.request_id = rel
        .stem
        .split_once('_')
        .map_or_else(|| rel.stem.clone(), |(_, x)| x.to_string());
    r.timestamp = crate::collect::utc_rfc3339(f.time);
    r.model = "import".into();
    r.image_name = f.image_name.clone();
    r.camera = f.camera.clone();
    r.camera_key = format!("import|cam:{}", f.camera);
    r.scene = rel.scene.clone();
    r.stats = Stats::from(f.stats);
    r.width = f.width;
    r.height = f.height;
    r.image = ImageInfo {
        file: format!("{}.jpg", rel.stem),
        bytes,
        original,
    };
    r.threshold = 0.0;
    r.dhash = crate::collect::dhash::hex(f.dhash);
    r.source = Some(format!("import:{import_id}"));
    r.provenance = Some(f.provenance.clone());
    r
}

/// Whether `day` already holds an imported frame of this millisecond (`<ts>_imp-*.jpg`) with
/// the dHash `hash`: the same footage imported again.
fn already_imported(day: &Path, ts: &str, hash: u64) -> bool {
    let Ok(rd) = std::fs::read_dir(day) else {
        return false;
    };
    let prefix = format!("{ts}_imp-");
    rd.flatten().any(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        name.starts_with(&prefix)
            && name.ends_with(".json")
            && !name.ends_with(crate::collect::labels::LABELS_SUFFIX)
            && std::fs::read(e.path())
                .ok()
                .and_then(|b| serde_json::from_slice::<Record>(&b).ok())
                .is_some_and(|r| r.dhash == crate::collect::dhash::hex(hash))
    })
}

/// Write `frames` of import `import_id` into `dir`; returns the image paths relative to `dir`
/// (`<scene>/<camera>/<date>/<stem>.jpg`), in the order written, and how many frames were
/// skipped as already imported (the same camera, millisecond and dHash). Frames of one import at
/// the same millisecond get distinct numbers; an existing file is never overwritten.
pub fn write_frames(dir: &Path, import_id: &str, frames: &[Frame]) -> Result<(Vec<String>, usize)> {
    let id4: String = import_id
        .rsplit('-')
        .next()
        .unwrap_or("0000")
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    let mut out = Vec::with_capacity(frames.len());
    let mut skipped = 0;
    for (n, f) in frames.iter().enumerate() {
        let scene = f.stats.scene().as_str().to_string();
        let date = crate::collect::utc_date(f.time);
        let ts = crate::collect::utc_file_ts(f.time);
        if already_imported(&dir.join(&scene).join(&f.camera).join(&date), &ts, f.dhash) {
            skipped += 1;
            continue;
        }
        let mut k = n;
        let rel = loop {
            let rel = Rel {
                scene: scene.clone(),
                camera: f.camera.clone(),
                date: date.clone(),
                stem: stem_of(f.time, &id4, k),
            };
            if !rel.to_path(dir, "jpg").exists() && !rel.to_path(dir, "json").exists() {
                break rel;
            }
            k += frames.len().max(1);
        };
        if layout::parse_rel(&rel.jpg()).is_none() {
            anyhow::bail!(
                "internal: {} is not a valid dataset path (camera {:?})",
                rel.jpg(),
                f.camera
            );
        }
        let day = dir.join(&rel.scene).join(&rel.camera).join(&rel.date);
        std::fs::create_dir_all(&day).with_context(|| format!("creating {}", day.display()))?;
        let raw =
            std::fs::read(&f.file).with_context(|| format!("reading {}", f.file.display()))?;
        let (bytes, original) = if raw.starts_with(&[0xFF, 0xD8]) {
            (raw, true)
        } else {
            let img = crate::image::decode(&raw)
                .with_context(|| format!("decoding {}", f.file.display()))?;
            (crate::image::encode_jpeg(&img, 92)?, false)
        };
        let jpg = rel.to_path(dir, "jpg");
        layout::write_atomic(&jpg, &bytes).with_context(|| format!("writing {}", jpg.display()))?;
        let rec = record_of(f, &rel, import_id, bytes.len() as u64, original);
        let json = serde_json::to_vec_pretty(&rec)?;
        let side = rel.to_path(dir, "json");
        if let Err(e) = layout::write_atomic(&side, &json) {
            let _ = std::fs::remove_file(&jpg);
            return Err(e).with_context(|| format!("writing {}", side.display()));
        }
        let line = IndexLine::of(&rec, (bytes.len() + json.len()) as u64);
        layout::append_line(
            &day.join(layout::INDEX_FILE),
            &serde_json::to_string(&line)?,
        )
        .with_context(|| format!("writing {}", day.join(layout::INDEX_FILE).display()))?;
        out.push(rel.jpg());
    }
    Ok((out, skipped))
}

/// Append `line` to `<dir>/imports.jsonl`.
pub fn append_import(dir: &Path, line: &ImportLine) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    layout::append_line(&dir.join(IMPORTS_FILE), &serde_json::to_string(line)?)
        .with_context(|| format!("writing {}", dir.join(IMPORTS_FILE).display()))
}

/// The recording time of a frame `offset_s` seconds into a video that started at `base`.
pub fn frame_time(base: SystemTime, offset_s: Option<f64>) -> SystemTime {
    match offset_s.filter(|o| o.is_finite() && *o >= 0.0) {
        Some(o) => base + Duration::from_secs_f64(o.min(1e9)),
        None => base,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::labels::Frame as LabelFrame;

    fn jpeg(gray: bool) -> Vec<u8> {
        let mut img = crate::image::RgbImage {
            width: 32,
            height: 24,
            rgb: vec![0; 32 * 24 * 3],
        };
        for (i, px) in img.rgb.chunks_mut(3).enumerate() {
            let v = u8::try_from(i % 251).unwrap();
            px[0] = v;
            px[1] = if gray { v } else { v / 2 };
            px[2] = if gray { v } else { 255 - v };
        }
        crate::image::encode_jpeg(&img, 90).unwrap()
    }

    #[test]
    fn frames_land_in_the_dataset_layout_with_provenance() {
        let root = std::env::temp_dir().join(format!("bop-import-write-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("f_000001.jpg");
        let bytes = jpeg(true);
        std::fs::write(&src, &bytes).unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        let stats = ImageStats::compute(&img);
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_520_812_500);
        let prov = Provenance {
            import_id: "20261022-120000-ab12".into(),
            kind: "video".into(),
            path: Some("/v/clip.mp4".into()),
            license: "CC BY 4.0".into(),
            license_confirmed: true,
            time_offset_s: Some(12.5),
            ..Provenance::default()
        };
        let f = Frame {
            file: src.clone(),
            image_name: "clip.mp4".into(),
            camera: "meva-school-g639".into(),
            time: frame_time(base, Some(12.5)),
            width: img.width,
            height: img.height,
            stats,
            dhash: 0xabcd,
            provenance: prov,
        };
        let mut other = f.clone();
        other.dhash = 0x1234;
        let (rels, skipped) =
            write_frames(&dir, "20261022-120000-ab12", &[f.clone(), other.clone()]).unwrap();
        assert_eq!((rels.len(), skipped), (2, 0));
        // The same footage again: skipped.
        let (again, skipped) = write_frames(&dir, "20261023-000000-cd34", &[f, other]).unwrap();
        assert_eq!((again.len(), skipped), (0, 2));
        assert_ne!(rels[0], rels[1], "same time: distinct numbers");
        assert!(
            rels[0].starts_with("ir/meva-school-g639/2018-03-11/"),
            "{}",
            rels[0]
        );
        let fr = LabelFrame::resolve(&dir, &rels[0]).unwrap();
        let rec = fr.record().unwrap();
        assert_eq!(rec.source.as_deref(), Some("import:20261022-120000-ab12"));
        assert_eq!(rec.provenance.as_ref().unwrap().time_offset_s, Some(12.5));
        assert!(rec.image.original);
        assert!(rec.raw.is_empty() && rec.predictions.is_empty());
        assert_eq!(rec.scene, "ir");
        assert_eq!(
            std::fs::read(&fr.jpg).unwrap(),
            bytes,
            "the JPEG is copied as is"
        );
        let idx = layout::read_index(&fr.jpg.parent().unwrap().join(layout::INDEX_FILE));
        assert_eq!(idx.len(), 2);
        // The sidecar is a version-1 record: an older reader sees an unknown field only.
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fr.sidecar).unwrap()).unwrap();
        assert_eq!(v["version"], 1);
        assert_eq!(v["provenance"]["license"], "CC BY 4.0");
        let summary = layout::Summary::scan(&dir);
        assert_eq!(summary.frames, 2);
        append_import(
            &dir,
            &ImportLine {
                id: "x".into(),
                created: "c".into(),
                source: "clip.mp4".into(),
                kind: "path".into(),
                license: "CC BY 4.0".into(),
                attribution: None,
                frames: 2,
                cameras: vec!["meva-school-g639".into()],
                scenes: [("ir".to_string(), 2)].into(),
            },
        )
        .unwrap();
        assert_eq!(read_imports(&dir).len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
