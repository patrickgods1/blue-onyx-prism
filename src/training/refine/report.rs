//! The report of the dataset check, `<review folder>/refine/report.json` ([`REPORT_FILE`]): what
//! was checked, what was suggested, and how often a fold model found the labelled objects of camera
//! days it never saw (out-of-fold recall) per camera x scene x class and per size at model input,
//! each with a Wilson 95% interval ([`wilson`]): 57 held-out persons give a recall to about +/- 10
//! points, and the interval says so.
//!
//! The `refine/` folder sits beside the scene folders (`day/`, `night/`, `ir/`) of the folder under
//! review: the dataset walkers ([`crate::collect::layout::day_dirs`]) and retention list only scene
//! folders, so they never see it (like `spots/`).

use super::judge::{BoxFact, FOUND_IOU, FOUND_SCORE};
use crate::collect::labels::SizeFloor;
use crate::training::dataset::IgnoreMode;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Folder of the report in the folder under review.
pub const REPORT_DIR: &str = "refine";
/// The report file in [`REPORT_DIR`].
pub const REPORT_FILE: &str = "report.json";
/// Version of the report's layout.
pub const REPORT_VERSION: u32 = 1;
/// z of a two-sided 95% interval: the 0.975 quantile of the standard normal distribution.
pub const Z95: f64 = 1.959_963_984_540_054;
/// Upper edges (pixels, exclusive) of the size buckets of the short side at model input; the last
/// bucket has none. 8 and 12: about one RF-DETR patch; 24 and 48: a person far and near.
pub const SIZE_EDGES: [f32; 4] = [8.0, 12.0, 24.0, 48.0];

/// `<review folder>/refine/report.json`.
pub fn report_path(review_dir: &Path) -> PathBuf {
    review_dir.join(REPORT_DIR).join(REPORT_FILE)
}

/// The Wilson score interval of `found` successes in `n` trials at 95% ([`Z95`]): `(low, high)`
/// within 0..=1; `(0, 1)` for no trial. Better than the normal approximation near 0 and 1 and on
/// small counts, where most per-camera recalls sit.
pub fn wilson(found: u64, n: u64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let found = found.min(n);
    #[allow(clippy::cast_precision_loss, reason = "counts of boxes")]
    let (k, n) = (found as f64, n as f64);
    let p = k / n;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    let low = if found == 0 { 0.0 } else { center - half };
    let high = if found as f64 == n {
        1.0
    } else {
        center + half
    };
    (low.clamp(0.0, 1.0), high.clamp(0.0, 1.0))
}

/// The size bucket of a short side at model input: an index into [`size_label`]'s buckets.
pub fn size_bucket(px: f32) -> usize {
    SIZE_EDGES.iter().take_while(|e| px >= **e).count()
}

/// The label of size bucket `i`: `under 8`, `8-12`, `12-24`, `24-48`, `48+`.
pub fn size_label(i: usize) -> String {
    match (
        i.checked_sub(1).and_then(|j| SIZE_EDGES.get(j)),
        SIZE_EDGES.get(i),
    ) {
        (None, Some(hi)) => format!("under {hi}"),
        (Some(lo), Some(hi)) => format!("{lo}-{hi}"),
        (Some(lo), None) => format!("{lo}+"),
        (None, None) => "any".into(),
    }
}

/// A recall: `found` of `boxes`, with its Wilson 95% interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Recall {
    pub boxes: u64,
    pub found: u64,
    /// `found / boxes`; None for no box.
    pub recall: Option<f64>,
    pub low: f64,
    pub high: f64,
}

/// Decimals a recall is written with: far below its interval, and a short decimal reads back as
/// exactly the number written.
pub const RECALL_DECIMALS: i32 = 6;

/// `x` rounded to [`RECALL_DECIMALS`].
fn rounded(x: f64) -> f64 {
    let p = 10f64.powi(RECALL_DECIMALS);
    (x * p).round() / p
}

impl Recall {
    fn of(found: u64, boxes: u64) -> Self {
        let (low, high) = wilson(found, boxes);
        #[allow(clippy::cast_precision_loss, reason = "counts of boxes")]
        let recall = (boxes > 0).then(|| rounded(found as f64 / boxes as f64));
        Self {
            boxes,
            found,
            recall,
            low: rounded(low),
            high: rounded(high),
        }
    }
}

/// Recall of one camera x scene x class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupRecall {
    pub camera: String,
    pub scene: String,
    pub class: String,
    #[serde(flatten)]
    pub recall: Recall,
}

/// Recall of one size bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizeRecall {
    /// `under 8`, `8-12`, ... ([`size_label`]).
    pub bucket: String,
    /// Short side at model input, pixels: from (inclusive) ...
    pub min_px: Option<f32>,
    /// ... to (exclusive); None = no upper edge.
    pub max_px: Option<f32>,
    #[serde(flatten)]
    pub recall: Recall,
}

/// Suggestions per kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindCounts {
    pub add: usize,
    pub verify: usize,
    pub relabel: usize,
}

impl KindCounts {
    pub fn total(&self) -> usize {
        self.add + self.verify + self.relabel
    }
}

/// Why reviewed frames were not checked (the export's selection left them out).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeftOut {
    /// Reviewed, but boxes still pending.
    pub pending: usize,
    /// Training labels not in the class list.
    pub off_class_list: usize,
    /// Image or sidecar gone.
    pub lost: usize,
    /// Label file or sidecar does not read.
    pub unreadable: usize,
    /// Folders that could not be listed (their frames are in no count).
    pub unreadable_folders: usize,
    /// Left out by the request's frame filter.
    pub filtered: usize,
}

impl LeftOut {
    /// Frames left out (the folders not counted).
    pub fn frames(&self) -> usize {
        self.pending + self.off_class_list + self.lost + self.unreadable + self.filtered
    }
}

/// The recipe the fold models trained with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecipeInfo {
    /// `rfdetr-nano`, `yolo26n`, ...
    pub model: String,
    pub imgsz: u32,
    pub epochs: u32,
    pub batch: u32,
    /// Where it came from: the newest finished training run, or the default.
    pub from: String,
}

/// `report.json` (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefineReport {
    pub version: u32,
    /// The check's id ([`crate::collect::labels::refine::Suggestion::run`]).
    pub run: String,
    /// When it finished (UTC, RFC 3339).
    pub created: String,
    /// Stopped by Cancel before every fold ran.
    pub cancelled: bool,
    pub folds: usize,
    pub folds_done: usize,
    pub recipe: RecipeInfo,
    pub camera_days: usize,
    /// Frames selected (reviewed, exportable, in the filter).
    pub frames_selected: usize,
    /// Frames judged (reviewed when written; finished folds only).
    pub frames_checked: usize,
    pub left_out: LeftOut,
    /// Suggestions written now, per kind (a frame's unchanged suggestions are not counted again).
    pub written: KindCounts,
    /// Suggestions found, per kind (written now or already in the file).
    pub found: KindCounts,
    /// Frames whose label file changed.
    pub frames_written: usize,
    /// Matched pairs below the judge's good IoU (loose labels).
    pub loose: usize,
    /// The production-like score and IoU at which a box counts as found.
    pub found_score: f32,
    pub found_iou: f32,
    pub recall: Recall,
    pub recall_by_group: Vec<GroupRecall>,
    pub recall_by_size: Vec<SizeRecall>,
    /// Static spots of the check's selection
    /// ([`crate::collect::labels::statics`]). Reports of earlier checks: 0.
    #[serde(default)]
    pub static_spots: usize,
    /// Recall of the training boxes at a static spot (one that holds without the frame's
    /// camera-day, [`super::judge::spot_of`]) ...
    #[serde(default)]
    pub recall_at_spots: Recall,
    /// ... and of the others.
    #[serde(default)]
    pub recall_elsewhere: Recall,
    /// Of `found.add`, the adds at a static spot (from [`super::judge::ADD_SCORE_AT_SPOT`]).
    #[serde(default)]
    pub adds_at_spots: usize,
    /// The fold models trained on every label (their exports apply no size floor), so the recall by
    /// size measures what a model of the recipe can find. False in reports of earlier checks, whose
    /// fold exports applied the floor then in force.
    #[serde(default)]
    pub folds_every_label: bool,
    /// `collect.size_floor` in force: the judge's floor (boxes under it were ignore regions for the
    /// suggestions and the recall but for its size table). None = off, or an earlier report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
    /// `training.ignore` the fold models trained with. None in reports of earlier checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_mode: Option<IgnoreMode>,
    pub warnings: Vec<String>,
}

/// Counts toward the recall tables, frame by frame.
#[derive(Debug, Clone, Default)]
pub struct RecallTally {
    groups: BTreeMap<(String, String, String), (u64, u64)>,
    sizes: BTreeMap<usize, (u64, u64)>,
    /// (boxes, found) at a static spot, and elsewhere.
    at_spots: (u64, u64),
    elsewhere: (u64, u64),
}

impl RecallTally {
    /// Count the training boxes of a frame of `camera` and `scene`. A box under the size floor
    /// ([`BoxFact::under_floor`]) counts in the recall by size only: an ignore region otherwise.
    pub fn add(&mut self, camera: &str, scene: &str, boxes: &[BoxFact]) {
        for b in boxes {
            let hit = u64::from(b.found);
            let s = self.sizes.entry(size_bucket(b.short_px)).or_default();
            s.0 += 1;
            s.1 += hit;
            if b.under_floor {
                continue;
            }
            let g = self
                .groups
                .entry((camera.to_string(), scene.to_string(), b.class.clone()))
                .or_default();
            g.0 += 1;
            g.1 += hit;
            let w = if b.at_spot {
                &mut self.at_spots
            } else {
                &mut self.elsewhere
            };
            w.0 += 1;
            w.1 += hit;
        }
    }

    /// Over the boxes at a static spot.
    pub fn at_spots(&self) -> Recall {
        Recall::of(self.at_spots.1, self.at_spots.0)
    }

    /// Over the boxes at no static spot.
    pub fn elsewhere(&self) -> Recall {
        Recall::of(self.elsewhere.1, self.elsewhere.0)
    }

    /// Over every box.
    pub fn total(&self) -> Recall {
        let (n, k) = self
            .groups
            .values()
            .fold((0, 0), |(n, k), (a, b)| (n + a, k + b));
        Recall::of(k, n)
    }

    pub fn by_group(&self) -> Vec<GroupRecall> {
        self.groups
            .iter()
            .map(|((camera, scene, class), (n, k))| GroupRecall {
                camera: camera.clone(),
                scene: scene.clone(),
                class: class.clone(),
                recall: Recall::of(*k, *n),
            })
            .collect()
    }

    /// Every size bucket, empty ones too (so a table has its rows).
    pub fn by_size(&self) -> Vec<SizeRecall> {
        (0..=SIZE_EDGES.len())
            .map(|i| {
                let (n, k) = self.sizes.get(&i).copied().unwrap_or((0, 0));
                SizeRecall {
                    bucket: size_label(i),
                    min_px: i.checked_sub(1).and_then(|j| SIZE_EDGES.get(j)).copied(),
                    max_px: SIZE_EDGES.get(i).copied(),
                    recall: Recall::of(k, n),
                }
            })
            .collect()
    }
}

impl RefineReport {
    /// An empty report of run `run`.
    pub fn new(run: &str, folds: usize, recipe: RecipeInfo) -> Self {
        Self {
            version: REPORT_VERSION,
            run: run.to_string(),
            created: String::new(),
            cancelled: false,
            folds,
            folds_done: 0,
            recipe,
            camera_days: 0,
            frames_selected: 0,
            frames_checked: 0,
            left_out: LeftOut::default(),
            written: KindCounts::default(),
            found: KindCounts::default(),
            frames_written: 0,
            loose: 0,
            found_score: FOUND_SCORE,
            found_iou: FOUND_IOU,
            recall: Recall::default(),
            recall_by_group: Vec::new(),
            recall_by_size: Vec::new(),
            static_spots: 0,
            recall_at_spots: Recall::default(),
            recall_elsewhere: Recall::default(),
            adds_at_spots: 0,
            folds_every_label: false,
            size_floor: None,
            ignore_mode: None,
            warnings: Vec::new(),
        }
    }

    /// Fill the recall tables from `tally` and stamp the time.
    pub fn finish(&mut self, tally: &RecallTally) {
        self.recall = tally.total();
        self.recall_by_group = tally.by_group();
        self.recall_by_size = tally.by_size();
        self.recall_at_spots = tally.at_spots();
        self.recall_elsewhere = tally.elsewhere();
        self.created = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    }

    /// Write it to `<review_dir>/refine/report.json` atomically (a reader never sees half of it);
    /// returns the path. Refused when the folder under review is not a folder any more (an
    /// unmounted drive: creating it would put the report on the disk below the mount point); only
    /// `refine/` is created.
    pub fn write(&self, review_dir: &Path) -> Result<PathBuf> {
        if !review_dir.is_dir() {
            bail!(
                "the folder under review {} is not available (an unplugged drive?), so the report \
                 was not written; the suggestions written before are in the label files",
                review_dir.display()
            );
        }
        let path = report_path(review_dir);
        let dir = review_dir.join(REPORT_DIR);
        match std::fs::create_dir(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && dir.is_dir() => {}
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "creating {} (is the folder under review writable?)",
                        dir.display()
                    )
                });
            }
        }
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        crate::collect::layout::write_atomic(&path, &bytes).with_context(|| {
            format!(
                "writing {}; the suggestions in the label files are saved all the same",
                path.display()
            )
        })?;
        Ok(path)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4
    }

    #[test]
    fn wilson_matches_known_values() {
        // Reference values (Wilson 1927; e.g. Newcombe 1998, method 3).
        assert!(
            close(wilson(5, 10), (0.236_594, 0.763_406)),
            "{:?}",
            wilson(5, 10)
        );
        assert!(
            close(wilson(0, 10), (0.0, 0.277_533)),
            "{:?}",
            wilson(0, 10)
        );
        assert!(
            close(wilson(10, 10), (0.722_467, 1.0)),
            "{:?}",
            wilson(10, 10)
        );
        assert!(
            close(wilson(81, 100), (0.722_21, 0.874_85)),
            "{:?}",
            wilson(81, 100)
        );
        assert!(close(wilson(1, 1), (0.206_543, 1.0)), "{:?}", wilson(1, 1));
        // Exact ends, no trial, more found than tried (clamped).
        assert_eq!(wilson(0, 57).0, 0.0);
        assert_eq!(wilson(57, 57).1, 1.0);
        assert_eq!(wilson(0, 0), (0.0, 1.0));
        assert_eq!(wilson(12, 10), wilson(10, 10));
        // Symmetric, and narrower with more boxes.
        let (a, b) = wilson(3, 20);
        let (c, d) = wilson(17, 20);
        assert!((a - (1.0 - d)).abs() < 1e-12 && (b - (1.0 - c)).abs() < 1e-12);
        let w57 = wilson(50, 57);
        let w570 = wilson(500, 570);
        assert!(w570.1 - w570.0 < w57.1 - w57.0);
        // 57 persons give about +/- 10 points at 88%.
        assert!((w57.1 - w57.0) > 0.15 && (w57.1 - w57.0) < 0.22, "{w57:?}");
    }

    #[test]
    fn size_buckets_have_their_edges_in_the_upper_bucket() {
        let cases = [
            (0.0, 0),
            (7.99, 0),
            (8.0, 1),
            (11.99, 1),
            (12.0, 2),
            (23.9, 2),
            (24.0, 3),
            (47.99, 3),
            (48.0, 4),
            (900.0, 4),
            (f32::INFINITY, 4),
        ];
        for (px, b) in cases {
            assert_eq!(size_bucket(px), b, "{px}");
        }
        let labels: Vec<String> = (0..5).map(size_label).collect();
        assert_eq!(labels, ["under 8", "8-12", "12-24", "24-48", "48+"]);
        // NaN (a broken box) lands in the first bucket, never panics.
        assert_eq!(size_bucket(f32::NAN), 0);
    }

    #[test]
    fn the_tally_and_the_json_round_trip() {
        let mut t = RecallTally::default();
        let fact = |class: &str, px: f32, found: bool| BoxFact {
            class: class.into(),
            short_px: px,
            found,
            // The cars stand at a static spot.
            at_spot: class == "car",
            under_floor: false,
        };
        t.add(
            "patio",
            "night",
            &[
                fact("person", 9.0, false),
                fact("person", 30.0, true),
                fact("car", 60.0, true),
            ],
        );
        t.add("patio", "night", &[fact("person", 30.0, true)]);
        t.add("front", "day", &[fact("car", 100.0, false)]);
        let mut r = RefineReport::new(
            "20261111-120000-abcd",
            4,
            RecipeInfo {
                model: "rfdetr-nano".into(),
                imgsz: 384,
                epochs: 8,
                batch: 4,
                from: "default".into(),
            },
        );
        r.finish(&t);
        assert_eq!((r.recall.boxes, r.recall.found), (5, 3));
        assert_eq!(
            (r.recall_at_spots.boxes, r.recall_at_spots.found),
            (2, 1),
            "the cars"
        );
        assert_eq!((r.recall_elsewhere.boxes, r.recall_elsewhere.found), (3, 2));
        assert_eq!(r.recall_at_spots.recall, Some(0.5));
        // A report of an earlier check (no spot fields) reads.
        let mut old = serde_json::to_value(&r).unwrap();
        for k in [
            "static_spots",
            "recall_at_spots",
            "recall_elsewhere",
            "adds_at_spots",
        ] {
            old.as_object_mut().unwrap().remove(k);
        }
        let back: RefineReport = serde_json::from_value(old).unwrap();
        assert_eq!(back.recall_at_spots, Recall::default());
        let g: Vec<(&str, &str, &str, u64, u64)> = r
            .recall_by_group
            .iter()
            .map(|g| {
                (
                    g.camera.as_str(),
                    g.scene.as_str(),
                    g.class.as_str(),
                    g.recall.boxes,
                    g.recall.found,
                )
            })
            .collect();
        assert_eq!(
            g,
            [
                ("front", "day", "car", 1, 0),
                ("patio", "night", "car", 1, 1),
                ("patio", "night", "person", 3, 2),
            ]
        );
        let person = &r.recall_by_group[2].recall;
        assert_eq!(person.recall, Some(0.666_667));
        let (lo, hi) = wilson(2, 3);
        assert!((person.low - lo).abs() < 1e-6 && (person.high - hi).abs() < 1e-6);
        let sizes: Vec<(String, u64, u64)> = r
            .recall_by_size
            .iter()
            .map(|s| (s.bucket.clone(), s.recall.boxes, s.recall.found))
            .collect();
        assert_eq!(
            sizes,
            [
                ("under 8".to_string(), 0, 0),
                ("8-12".to_string(), 1, 0),
                ("12-24".to_string(), 0, 0),
                ("24-48".to_string(), 2, 2),
                ("48+".to_string(), 2, 1),
            ]
        );
        assert_eq!(r.recall_by_size[0].recall.recall, None);
        assert_eq!(
            (r.recall_by_size[1].min_px, r.recall_by_size[1].max_px),
            (Some(8.0), Some(12.0))
        );
        assert_eq!(r.recall_by_size[4].max_px, None);
        // The JSON reads back as it was, and the file is written whole.
        let dir = std::env::temp_dir().join(format!("bop-refine-report-{}", uuid::Uuid::new_v4()));
        // A folder under review that is not there (an unmounted drive) is not created.
        let e = r.write(&dir).unwrap_err().to_string();
        assert!(e.contains("is not available"), "{e}");
        assert!(!dir.exists(), "the vanished folder was created again");
        std::fs::create_dir_all(&dir).unwrap();
        let path = r.write(&dir).unwrap();
        // A second write replaces it in the existing refine/ folder.
        assert_eq!(r.write(&dir).unwrap(), path);
        assert_eq!(path, dir.join("refine").join("report.json"));
        let back: RefineReport = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back, r);
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["recall_by_group"][2]["boxes"], 3, "flattened: {v}");
        assert_eq!(v["found_score"], 0.3);
        let left: Vec<_> = std::fs::read_dir(dir.join("refine")).unwrap().collect();
        assert_eq!(left.len(), 1, "no temporary file left");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_box_under_the_size_floor_counts_in_the_size_table_only() {
        let fact = |px: f32, found: bool, under_floor: bool| BoxFact {
            class: "person".into(),
            short_px: px,
            found,
            at_spot: false,
            under_floor,
        };
        let mut t = RecallTally::default();
        t.add(
            "patio",
            "night",
            &[
                fact(6.0, false, true),
                fact(7.0, true, true),
                fact(30.0, true, false),
            ],
        );
        let total = t.total();
        assert_eq!((total.boxes, total.found), (1, 1), "an ignore region");
        assert_eq!((t.elsewhere().boxes, t.at_spots().boxes), (1, 0));
        assert_eq!(t.by_group()[0].recall.boxes, 1);
        let sizes = t.by_size();
        assert_eq!(
            (sizes[0].recall.boxes, sizes[0].recall.found),
            (2, 1),
            "the table a floor is chosen from keeps them"
        );
    }
}
