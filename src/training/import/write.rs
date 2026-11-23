//! Writing the selected frames of an import into the dataset folder, in the collector's layout
//! (`<dir>/<scene>/<camera>/<date>/<stem>.{jpg,json}` + `index.jsonl`, `docs/DATASET.md`): a
//! version-1 sidecar without detections, `source: "import:<id>"` and its provenance; then the
//! frames are queued for review (`review-queue.jsonl`) and the import is listed in
//! `<dir>/imports.jsonl`. Each file is written atomically, image first, then sidecar, then the
//! index line, like the collector does; when the sidecar or the index line fails (a full disk),
//! the frame's files are removed again, so a frame is either complete or absent. A write that
//! stops part way ([`write_frames_with`]) answers the frames written so far.
//!
//! The dataset folder is shared with the collector thread, whose retention rewrites day indexes and
//! removes emptied folders: index lines go through [`layout::append_index`] (the dataset's
//! cross-process index lock), and the image through [`layout::write_in_day`], which makes the day
//! folder again when retention removed it, still empty, after this made it (why a retry rather than
//! holding the lock across the frame's decode: its doc).

use crate::clock::Zone;
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

impl ImportLine {
    /// `created` (stored UTC) for people, in `zone` ([`Zone::display`], `2026-11-16 14:02 PST`);
    /// the stored text when it is not RFC 3339.
    pub fn created_in(&self, zone: Zone) -> String {
        zone.display_stored(&self.created)
    }

    /// [`Self::created_in`] the process-wide zone, for the import lists of the Import and
    /// Collect tabs.
    pub fn created_shown(&self) -> String {
        self.created_in(crate::clock::zone())
    }
}

/// The imports listed in `dir` (newest last; lines that do not parse are skipped).
pub fn read_imports(dir: &Path) -> Vec<ImportLine> {
    layout::read_lines(&dir.join(IMPORTS_FILE)).lines
}

/// The timestamp an imported frame's stem starts with: [`crate::clock::Zone::frame_stamp`] of its
/// recording time in `zone` (the import job matches written frames by it, so both use this).
pub fn stamp(zone: Zone, time: SystemTime) -> String {
    zone.frame_stamp(time)
}

fn stem_of(zone: Zone, time: SystemTime, id4: &str, n: usize) -> String {
    format!("{}_imp-{id4}-{n:05}", stamp(zone, time))
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

/// Whether the camera folder `camera_dir` already holds an imported frame (`<stamp>_imp-*.json`)
/// of the millisecond `ts` (a frame stamp) with the dHash `hash`: the same footage imported
/// again. Stamps are compared as instants, so a frame imported before the `timezone` key (UTC
/// names) or in another zone is found: it looks in the day folder of `ts` in `zone`, then in the
/// UTC day and the days either side of it (every zone's date is one of those three).
fn already_imported(camera_dir: &Path, zone: Zone, ts: &str, hash: u64) -> bool {
    let Some(at) = crate::clock::parse_frame_stamp(ts) else {
        return false;
    };
    let utc = at.date_naive();
    let mut days = vec![zone.day(at)];
    for d in [utc.pred_opt(), Some(utc), utc.succ_opt()]
        .into_iter()
        .flatten()
    {
        if !days.contains(&d) {
            days.push(d);
        }
    }
    days.iter().any(|day| {
        let folder = day.format("%Y-%m-%d").to_string();
        imported_in(&camera_dir.join(folder), at, hash)
    })
}

/// [`already_imported`] in one day folder.
fn imported_in(day: &Path, at: chrono::DateTime<chrono::Utc>, hash: u64) -> bool {
    let Ok(rd) = std::fs::read_dir(day) else {
        return false;
    };
    rd.flatten().any(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        crate::clock::parse_frame_stamp(&name) == Some(at)
            && name.contains("_imp-")
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
    write_frames_with(
        dir,
        crate::clock::zone(),
        import_id,
        frames,
        &mut |_| Ok(()),
    )
    .map_err(|p| p.error)
}

/// What [`write_frames_with`] tells its caller around each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Frame `n` is next; an error stops the write before it.
    Before(usize),
    /// Frame `n` is in the dataset (its image, sidecar and index line).
    Written(usize),
}

/// A write that stopped part way: the frames written before `error` (complete, in the index).
#[derive(Debug)]
pub struct Partial {
    pub rels: Vec<String>,
    pub skipped: usize,
    pub error: anyhow::Error,
}

/// [`write_frames`] with the day folders and stems in `zone` (the process-wide zone for
/// [`write_frames`]) and `hook` called before and after each frame (free-space checks, cleanup).
pub fn write_frames_with(
    dir: &Path,
    zone: Zone,
    import_id: &str,
    frames: &[Frame],
    hook: &mut dyn FnMut(Step) -> Result<()>,
) -> std::result::Result<(Vec<String>, usize), Partial> {
    let mut out = Vec::with_capacity(frames.len());
    let mut skipped = 0;
    match write_into(dir, zone, import_id, frames, hook, &mut out, &mut skipped) {
        Ok(()) => Ok((out, skipped)),
        Err(error) => Err(Partial {
            rels: out,
            skipped,
            error,
        }),
    }
}

fn write_into(
    dir: &Path,
    zone: Zone,
    import_id: &str,
    frames: &[Frame],
    hook: &mut dyn FnMut(Step) -> Result<()>,
    out: &mut Vec<String>,
    skipped: &mut usize,
) -> Result<()> {
    let id4: String = import_id
        .rsplit('-')
        .next()
        .unwrap_or("0000")
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    for (n, f) in frames.iter().enumerate() {
        hook(Step::Before(n))?;
        let scene = f.stats.scene().as_str().to_string();
        let date = layout::day_folder(zone, f.time);
        let ts = stamp(zone, f.time);
        if already_imported(&dir.join(&scene).join(&f.camera), zone, &ts, f.dhash) {
            *skipped += 1;
            continue;
        }
        let mut k = n;
        let rel = loop {
            let rel = Rel {
                scene: scene.clone(),
                camera: f.camera.clone(),
                date: date.clone(),
                stem: stem_of(zone, f.time, &id4, k),
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
        let raw =
            std::fs::read(&f.file).with_context(|| format!("reading {}", f.file.display()))?;
        let (bytes, original) = if raw.starts_with(&[0xFF, 0xD8]) {
            (raw, true)
        } else {
            let img = crate::image::decode(&raw)
                .with_context(|| format!("decoding {}", f.file.display()))?;
            (crate::image::encode_jpeg(&img, 92)?, false)
        };
        // Made after the decode, right before the image goes in: the folder is empty until then.
        std::fs::create_dir_all(&day).with_context(|| format!("creating {}", day.display()))?;
        let jpg = rel.to_path(dir, "jpg");
        layout::write_in_day(&day, &jpg, &bytes)
            .with_context(|| format!("writing {}", jpg.display()))?;
        let rec = record_of(f, &rel, import_id, bytes.len() as u64, original);
        let json = serde_json::to_vec_pretty(&rec)?;
        let side = rel.to_path(dir, "json");
        if let Err(e) = layout::write_atomic(&side, &json) {
            let _ = std::fs::remove_file(&jpg);
            return Err(e).with_context(|| format!("writing {}", side.display()));
        }
        let line = IndexLine::of(&rec, (bytes.len() + json.len()) as u64);
        if let Err(e) = layout::append_index(&day, &line) {
            let _ = std::fs::remove_file(&side);
            let _ = std::fs::remove_file(&jpg);
            return Err(e)
                .with_context(|| format!("writing {}", day.join(layout::INDEX_FILE).display()));
        }
        out.push(rel.jpg());
        hook(Step::Written(n))?;
    }
    Ok(())
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

    /// [`write_frames`] with UTC folders and stems (never the process-wide zone, which other
    /// tests may change).
    fn utc(dir: &Path, import_id: &str, frames: &[Frame]) -> Result<(Vec<String>, usize)> {
        write_frames_with(dir, Zone::Utc, import_id, frames, &mut |_| Ok(())).map_err(|p| p.error)
    }

    #[test]
    fn folders_and_stems_follow_the_zone_and_a_reimport_in_another_zone_is_skipped() {
        let root = std::env::temp_dir().join(format!("bop-import-zone-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("f.jpg");
        let bytes = jpeg(false);
        std::fs::write(&src, &bytes).unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        // 2018-03-11T23:55:00Z + 12.5 s: 16:55 PDT the same day, 13:55 the next day at +14:00.
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_520_812_500);
        let frame = |dhash: u64| Frame {
            file: src.clone(),
            image_name: "c.mp4".into(),
            camera: "cam".into(),
            time: frame_time(base, Some(12.5)),
            width: img.width,
            height: img.height,
            stats: ImageStats::compute(&img),
            dhash,
            provenance: Provenance::default(),
        };
        let write = |zone: &str, id: &str, f: Frame| {
            let zone: Zone = zone.parse().unwrap();
            write_frames_with(&dir, zone, id, &[f], &mut |_| Ok(())).unwrap()
        };
        let (la, _) = write("America/Los_Angeles", "20261022-120000-ab12", frame(1));
        assert_eq!(
            la,
            ["day/cam/2018-03-11/20180311T165512500-0700_imp-ab12-00000.jpg"]
        );
        let (east, _) = write("+14:00", "20261022-120000-cd34", frame(2));
        assert_eq!(
            east,
            ["day/cam/2018-03-12/20180312T135512500p1400_imp-cd34-00000.jpg"]
        );
        // The sidecar's time stays UTC; the record's id is the stem.
        let rec = LabelFrame::resolve(&dir, &la[0]).unwrap().record().unwrap();
        assert_eq!(rec.timestamp, "2018-03-11T23:55:12.500Z");
        assert_eq!(rec.id, "20180311T165512500-0700_imp-ab12-00000");
        // The same footage again, in a zone whose folder is another day: still skipped, both
        // the frame named in Los Angeles and the one named at +14:00.
        for (zone, dhash) in [("utc", 1), ("Asia/Tokyo", 1), ("utc", 2), ("-08:00", 2)] {
            let (rels, skipped) = write(zone, "20261023-000000-ef56", frame(dhash));
            assert_eq!((rels.len(), skipped), (0, 1), "{zone} {dhash}");
        }
        // Another picture at that millisecond is not the same footage.
        let (rels, skipped) = write("utc", "20261023-000000-ef56", frame(3));
        assert_eq!(
            (rels, skipped),
            (
                vec!["day/cam/2018-03-11/20180311T235512500Z_imp-ef56-00000.jpg".to_string()],
                0
            )
        );
        // The matching helper of the import job gives the stem's start.
        let la_zone: Zone = "America/Los_Angeles".parse().unwrap();
        assert!(la[0].contains(&stamp(la_zone, frame_time(base, Some(12.5)))));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2 (Q:collect-1): the import lists its frames under the dataset's index
    /// lock, which retention's index rewrite holds, so the line is appended only once a rewrite
    /// in hand is done (and never lost to it).
    #[test]
    fn an_import_lists_its_frame_under_the_index_lock() {
        let root = std::env::temp_dir().join(format!("bop-import-lock-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(&dir).unwrap();
        let src = root.join("f.jpg");
        let bytes = jpeg(false);
        std::fs::write(&src, &bytes).unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        let f = Frame {
            file: src.clone(),
            image_name: "c.mp4".into(),
            camera: "cam".into(),
            time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_520_812_500),
            width: img.width,
            height: img.height,
            stats: ImageStats::compute(&img),
            dhash: 7,
            provenance: Provenance::default(),
        };
        // Retention (another process, or the collector thread) holds the lock.
        let held = std::fs::File::create(dir.join(layout::INDEX_LOCK_FILE)).unwrap();
        fs4::FileExt::try_lock(&held).unwrap();
        let write = {
            let dir = dir.clone();
            std::thread::spawn(move || utc(&dir, "20261022-120000-ab12", &[f]))
        };
        std::thread::sleep(Duration::from_millis(200));
        assert!(!write.is_finished(), "the append waits for the index lock");
        let day = dir.join("day/cam/2018-03-11");
        assert!(
            std::fs::read_dir(&day)
                .unwrap()
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".jpg")),
            "the image is in before the lock is asked for"
        );
        assert!(layout::read_index(&day.join(layout::INDEX_FILE)).is_empty());
        fs4::FileExt::unlock(&held).unwrap();
        let (rels, _) = write.join().unwrap().unwrap();
        let index = layout::read_index(&day.join(layout::INDEX_FILE));
        assert_eq!(index.len(), 1);
        assert!(rels[0].ends_with(&format!("{}.jpg", index[0].id)));
        let _ = std::fs::remove_dir_all(&root);
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
            utc(&dir, "20261022-120000-ab12", &[f.clone(), other.clone()]).unwrap();
        assert_eq!((rels.len(), skipped), (2, 0));
        // The same footage again: skipped.
        let (again, skipped) = utc(&dir, "20261023-000000-cd34", &[f, other]).unwrap();
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

    #[test]
    fn a_write_stopped_part_way_answers_the_complete_frames() {
        let root =
            std::env::temp_dir().join(format!("bop-import-partial-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("f.jpg");
        let bytes = jpeg(false);
        std::fs::write(&src, &bytes).unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let frames: Vec<Frame> = (0u32..4)
            .map(|i| Frame {
                file: src.clone(),
                image_name: "c.mp4".into(),
                camera: "cam".into(),
                time: frame_time(base, Some(f64::from(i))),
                width: img.width,
                height: img.height,
                stats: ImageStats::compute(&img),
                dhash: u64::from(i) * 0x0101_0101,
                provenance: Provenance::default(),
            })
            .collect();
        let mut steps = Vec::new();
        let p = write_frames_with(&dir, Zone::Utc, "20261022-120000-ab12", &frames, &mut |s| {
            steps.push(s);
            if s == Step::Before(2) {
                anyhow::bail!(std::io::Error::from(std::io::ErrorKind::StorageFull));
            }
            Ok(())
        })
        .unwrap_err();
        assert_eq!(p.rels.len(), 2);
        assert_eq!(
            steps,
            [
                Step::Before(0),
                Step::Written(0),
                Step::Before(1),
                Step::Written(1),
                Step::Before(2)
            ]
        );
        assert!(crate::training::import::disk::is_disk_full(&p.error));
        // Exactly the two frames: images, sidecars and index lines, no temporary files.
        let day = LabelFrame::resolve(&dir, &p.rels[0])
            .unwrap()
            .jpg
            .parent()
            .unwrap()
            .to_path_buf();
        let mut names: Vec<String> = std::fs::read_dir(&day)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names.len(), 5, "{names:?}");
        assert!(names.iter().all(|n| !n.ends_with(".tmp")), "{names:?}");
        assert_eq!(layout::read_index(&day.join(layout::INDEX_FILE)).len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_damaged_imports_file_costs_only_the_damaged_lines() {
        let dir = std::env::temp_dir().join(format!("bop-imports-{}", uuid::Uuid::new_v4()));
        let line = |id: &str| ImportLine {
            id: id.into(),
            created: "t".into(),
            source: "s".into(),
            kind: "path".into(),
            license: "l".into(),
            attribution: None,
            frames: 1,
            cameras: vec![],
            scenes: Default::default(),
        };
        append_import(&dir, &line("a")).unwrap();
        // An invalid byte, and a torn line with no newline.
        let path = dir.join(IMPORTS_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"\xff\n{\"id\":\"to");
        std::fs::write(&path, &bytes).unwrap();
        append_import(&dir, &line("b")).unwrap();
        let ids: Vec<String> = read_imports(&dir).into_iter().map(|l| l.id).collect();
        assert_eq!(ids, ["a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_frame_after_a_torn_index_line_is_indexed_whole() {
        let root = std::env::temp_dir().join(format!("bop-import-torn-{}", uuid::Uuid::new_v4()));
        let dir = root.join("dataset");
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("f.jpg");
        let bytes = jpeg(false);
        std::fs::write(&src, &bytes).unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let frame = |i: u32| Frame {
            file: src.clone(),
            image_name: "c.mp4".into(),
            camera: "cam".into(),
            time: frame_time(base, Some(f64::from(i))),
            width: img.width,
            height: img.height,
            stats: ImageStats::compute(&img),
            dhash: u64::from(i + 1) * 0x0101_0101,
            provenance: Provenance::default(),
        };
        let (rels, _) = utc(&dir, "20261022-120000-ab12", &[frame(0)]).unwrap();
        let index = LabelFrame::resolve(&dir, &rels[0])
            .unwrap()
            .jpg
            .parent()
            .unwrap()
            .join(layout::INDEX_FILE);
        // ENOSPC stopped an append part way: the frames went, the half line stayed.
        let mut text = std::fs::read(&index).unwrap();
        text.extend_from_slice(b"{\"id\":\"half");
        std::fs::write(&index, &text).unwrap();
        utc(&dir, "20261022-120000-ab12", &[frame(1)]).unwrap();
        let read = layout::read_lines::<IndexLine>(&index);
        assert_eq!((read.lines.len(), read.damaged), (2, 1));
        let _ = std::fs::remove_dir_all(&root);
    }
}
