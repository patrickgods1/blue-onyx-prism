//! The "Where the data is thin" table of the Refine tab's Coverage card (admin only;
//! `assets/train-refine.js`): one row per camera x scene of the folder under review with what the
//! reviewer has put in (reviewed frames, person and vehicle boxes, frames with nothing in them), how well
//! the last check's models found those boxes there (the report's recall, joined here), and how many
//! unreviewed frames hold a person proposal that a review would turn into training boxes. One route:
//!
//! - `GET /v1/refine/coverage`: one walk over every label file of the folder
//!   ([`queue::labelled_map`], status any, off the async workers), kept 60 s per folder and class
//!   list and dropped when a check starts or ends or a write of the tab changes boxes. The recall is
//!   joined from the current `refine/report.json` on every request (a small read), so a new report
//!   shows at once. Boxes of classes that are not wanted (`collect.classes` without the hidden ones)
//!   count nowhere.
//!
//! Counting rules: `reviewed` / `persons` / `vehicles` / `empty` are over frames with status
//! reviewed (training boxes, ignore regions left out, as an export keeps them); `empty` = reviewed frames
//! with no training box and no ignore region (the training plan's negatives; `empty_confirmed` is how many
//! of them the reviewer ticked); `unreviewed_person_frames`
//! is over unreviewed frames holding a box labelled `person` that is not rejected, which is the
//! Review page's own class filter (`label=person`), so the link the page builds
//! (`status=unreviewed&source=all&scene=..&camera=..&label=person`) opens exactly those frames. The
//! vehicle classes are [`VEHICLE_CLASSES`]. A frame of an unknown scene or camera name (not a path
//! the walkers accept) is in no row.

use super::refine::{queue_stamp, read_report};
use super::{AppState, blocking, training};
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::queue::{self, Filters};
use crate::collect::labels::repeats::VEHICLES;
use crate::collect::labels::{LabelFile, Status};
use crate::collect::layout::{self, SCENES};
use crate::sync::LockExt;
use crate::training::refine::report::{self, RECALL_DECIMALS, RefineReport};
use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a walk's answer is kept.
const KEEP_FOR: Duration = Duration::from_secs(60);
/// Most answers kept at once (folders under review x class lists).
const KEEP: usize = 4;
/// The class the table is about.
const PERSON: &str = "person";
/// The classes summed as "vehicles": the vehicle classes the Review page and the static spots
/// share ([`VEHICLES`]) and the two-wheelers.
const VEHICLE_CLASSES: [&str; 6] = [
    VEHICLES[0],
    VEHICLES[1],
    VEHICLES[2],
    VEHICLES[3],
    "motorcycle",
    "bicycle",
];

/// What one frame adds to its row.
struct Tally {
    camera: String,
    scene: String,
    reviewed: bool,
    persons: usize,
    vehicles: usize,
    empty: bool,
    empty_confirmed: bool,
    unreviewed_person: bool,
}

/// One camera x scene, counted over its frames.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Row {
    pub(super) camera: String,
    pub(super) scene: String,
    pub(super) reviewed: usize,
    pub(super) persons: usize,
    pub(super) vehicles: usize,
    pub(super) empty: usize,
    pub(super) empty_confirmed: usize,
    pub(super) unreviewed_person_frames: usize,
}

/// What `file` (frame `rel`) adds to its row; None for a path no walker accepts.
fn tally_of(rel: &str, file: &LabelFile, wanted: &Wanted<'_>) -> Option<Tally> {
    let r = layout::parse_rel(rel)?;
    let reviewed = file.status == Status::Reviewed;
    // Empty for training: reviewed, no training box and no ignore region (an unsure or
    // too-small object is no negative), whether or not the reviewer ticked "empty".
    let empty = reviewed && !file.boxes.iter().any(|b| b.is_training() || b.is_ignore());
    let (mut persons, mut vehicles) = (0, 0);
    if reviewed {
        for b in file.training().filter(|b| !b.is_ignore()) {
            match wanted.class_of(&b.label) {
                Some(PERSON) => persons += 1,
                Some(c) if VEHICLE_CLASSES.contains(&c) => vehicles += 1,
                _ => {}
            }
        }
    }
    // The Review page's class filter: any box of the label that is not rejected.
    let unreviewed_person = file.status == Status::Unreviewed
        && file.boxes.iter().any(|b| {
            b.state != crate::collect::labels::BoxState::Rejected
                && b.label == PERSON
                && wanted.class_of(&b.label) == Some(PERSON)
        });
    Some(Tally {
        camera: r.camera,
        scene: r.scene,
        reviewed,
        persons,
        vehicles,
        empty,
        empty_confirmed: reviewed && file.empty_confirmed && empty,
        unreviewed_person,
    })
}

/// The position of a scene in [`SCENES`] (day, night, ir); unknown ones last.
fn scene_rank(scene: &str) -> usize {
    SCENES
        .iter()
        .position(|s| *s == scene)
        .unwrap_or(SCENES.len())
}

/// A fresh walk of the label files of `dir`: the rows, by camera then scene (day, night, ir).
fn compute(dir: &Path, classes: &[String], hidden: &[String]) -> Vec<Row> {
    let wanted = Wanted { classes, hidden };
    let tallies = queue::labelled_map(dir, &Filters::default(), &|_| true, &|rel, file| {
        tally_of(rel, &file, &wanted)
    });
    let mut rows: BTreeMap<(String, usize), Row> = BTreeMap::new();
    for t in tallies {
        let row = rows
            .entry((t.camera.clone(), scene_rank(&t.scene)))
            .or_insert_with(|| Row {
                camera: t.camera.clone(),
                scene: t.scene.clone(),
                ..Row::default()
            });
        row.reviewed += usize::from(t.reviewed);
        row.persons += t.persons;
        row.vehicles += t.vehicles;
        row.empty += usize::from(t.empty);
        row.empty_confirmed += usize::from(t.empty_confirmed);
        row.unreviewed_person_frames += usize::from(t.unreviewed_person);
    }
    rows.into_values().collect()
}

struct Kept {
    at: Instant,
    stamp: String,
    value: Arc<Vec<Row>>,
}

/// The walks' answers per folder and class list (one per process, [`CACHE`]; a test makes its
/// own, so other tests clearing the process's cannot touch it).
struct Cache {
    kept: Mutex<BTreeMap<String, Kept>>,
    /// One walk at a time: a second caller waits and finds the first one's answer.
    walking: Mutex<()>,
}

/// The process's walk answers.
static CACHE: Cache = Cache::new();

/// Drop every kept answer of the process (a label write changed what a walk finds).
pub(super) fn forget() {
    CACHE.forget();
}

impl Cache {
    const fn new() -> Self {
        Self {
            kept: Mutex::new(BTreeMap::new()),
            walking: Mutex::new(()),
        }
    }

    fn forget(&self) {
        self.kept.lock_recover().clear();
    }

    /// [`compute`], kept [`KEEP_FOR`]; `stamp` names the checks on the queue: when one starts,
    /// ends or is cancelled the kept answer is dropped.
    fn rows(
        &self,
        dir: &Path,
        classes: &[String],
        hidden: &[String],
        stamp: &str,
    ) -> Arc<Vec<Row>> {
        let key = format!(
            "{}\u{1f}{}\u{1f}{}",
            dir.display(),
            classes.join("\u{1e}"),
            hidden.join("\u{1e}")
        );
        let fresh = |k: &Kept| k.at.elapsed() < KEEP_FOR && k.stamp == stamp;
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let _walking = self.walking.lock_recover();
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let value = Arc::new(compute(dir, classes, hidden));
        let mut map = self.kept.lock_recover();
        map.retain(|k, v| *k != key && v.at.elapsed() < KEEP_FOR);
        while map.len() >= KEEP {
            let Some(oldest) = map.iter().min_by_key(|(_, v)| v.at).map(|(k, _)| k.clone()) else {
                break;
            };
            map.remove(&oldest);
        }
        map.insert(
            key,
            Kept {
                at: Instant::now(),
                stamp: stamp.to_string(),
                value: value.clone(),
            },
        );
        value
    }
}

// ---------------------------------------------------------------------------------------------
// The report's recall

/// A recall as the page reads it.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct RecallJson {
    boxes: u64,
    found: u64,
    recall: Option<f64>,
    low: f64,
    high: f64,
}

fn round(x: f64) -> f64 {
    let p = 10f64.powi(RECALL_DECIMALS);
    (x * p).round() / p
}

/// The check's recall of `camera` x `scene` for the classes `pick` keeps, summed over them (the
/// interval recomputed from the sums, [`report::wilson`]); None when the report has no row there.
fn recall_of(
    r: &RefineReport,
    camera: &str,
    scene: &str,
    pick: impl Fn(&str) -> bool,
) -> Option<RecallJson> {
    let (mut boxes, mut found, mut any) = (0u64, 0u64, false);
    for g in &r.recall_by_group {
        if g.camera == camera && g.scene == scene && pick(&g.class) {
            boxes += g.recall.boxes;
            found += g.recall.found;
            any = true;
        }
    }
    if !any {
        return None;
    }
    let (low, high) = report::wilson(found, boxes);
    #[allow(clippy::cast_precision_loss, reason = "counts of boxes")]
    let recall = (boxes > 0).then(|| round(found as f64 / boxes as f64));
    Some(RecallJson {
        boxes,
        found,
        recall,
        low: round(low),
        high: round(high),
    })
}

// ---------------------------------------------------------------------------------------------
// GET /v1/refine/coverage

/// The answer of `GET /v1/refine/coverage` for `state`.
fn coverage_of(state: &AppState) -> serde_json::Value {
    let config = state.config_read().clone();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    // An unplugged drive is never an empty dataset: nothing is read from it.
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        return serde_json::json!({
            "success": true,
            "folder_missing": true,
            "message": why,
            "rows": [],
            "totals": totals(&[]),
        });
    }
    let stamp = queue_stamp(&training::tasks(state));
    let rows = CACHE.rows(
        &dir,
        &config.collect.classes,
        &config.collect.hidden_classes,
        &stamp,
    );
    let (report, _) = read_report(&dir);
    answer(&rows, report.as_ref())
}

/// The sums over `rows`.
fn totals(rows: &[Row]) -> serde_json::Value {
    serde_json::json!({
        "rows": rows.len(),
        "reviewed": rows.iter().map(|r| r.reviewed).sum::<usize>(),
        "persons": rows.iter().map(|r| r.persons).sum::<usize>(),
        "vehicles": rows.iter().map(|r| r.vehicles).sum::<usize>(),
        "empty": rows.iter().map(|r| r.empty).sum::<usize>(),
        "empty_confirmed": rows.iter().map(|r| r.empty_confirmed).sum::<usize>(),
        "unreviewed_person_frames": rows.iter().map(|r| r.unreviewed_person_frames).sum::<usize>(),
    })
}

/// The rows with the recall of `report` joined on.
fn answer(rows: &[Row], report: Option<&RefineReport>) -> serde_json::Value {
    let json_rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            let person = report.and_then(|x| recall_of(x, &r.camera, &r.scene, |c| c == PERSON));
            let vehicle = report
                .and_then(|x| recall_of(x, &r.camera, &r.scene, |c| VEHICLE_CLASSES.contains(&c)));
            serde_json::json!({
                "camera": r.camera,
                "scene": r.scene,
                "reviewed": r.reviewed,
                "persons": r.persons,
                "vehicles": r.vehicles,
                "empty": r.empty,
                "empty_confirmed": r.empty_confirmed,
                "unreviewed_person_frames": r.unreviewed_person_frames,
                "person": person,
                "vehicle": vehicle,
            })
        })
        .collect();
    serde_json::json!({
        "success": true,
        "folder_missing": false,
        "rows": json_rows,
        "totals": totals(rows),
    })
}

/// `GET /v1/refine/coverage`: see the module docs.
pub(super) async fn coverage(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || Json(coverage_of(&state)).into_response()).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, test_state_with};
    use super::*;
    use crate::collect::labels::bulk::test_support::put_frame_at;
    use crate::collect::labels::{BoxState, Frame, LabelBox, Source};
    use crate::training::refine::judge::BoxFact;
    use crate::training::refine::report::{Mode, ModelInfo};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::path::PathBuf;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_cover_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn state_in(dir: &Path) -> Arc<AppState> {
        let dir = dir.to_path_buf();
        test_state_with(move |c| {
            c.download_dir = Some(dir.clone());
            c.collect.dir = Some(dir.join("dataset"));
            c.training.dir = Some(dir.join("training"));
            c.collect.classes = vec![
                "person".into(),
                "car".into(),
                "truck".into(),
                "dog".into(),
                "bicycle".into(),
            ];
        })
    }

    async fn json(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let (s, _, b) = call(state, req).await;
        (
            s,
            serde_json::from_slice(&b).unwrap_or(serde_json::Value::Null),
        )
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    fn lbox(label: &str, state: BoxState) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: [8.0, 8.0, 28.0, 40.0],
            source: Source::Auto,
            score: Some(0.8),
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

    /// A frame of `camera` in `scene` with `boxes`, in `status`.
    fn frame(
        dir: &Path,
        scene: &str,
        camera: &str,
        stem: &str,
        status: Status,
        empty: bool,
        boxes: Vec<LabelBox>,
    ) {
        // The test support writes scene `ir`; the frame is moved into the scene's folder.
        put_frame_at(dir, camera, stem, (64, 48), None, boxes);
        if scene != "ir" {
            let from = dir.join("ir").join(camera).join("2026-10-21");
            let to = dir.join(scene).join(camera).join("2026-10-21");
            std::fs::create_dir_all(&to).unwrap();
            for ext in ["jpg", "json", "labels.json"] {
                std::fs::rename(
                    from.join(format!("{stem}.{ext}")),
                    to.join(format!("{stem}.{ext}")),
                )
                .unwrap();
            }
        }
        // The day's index, as the collector writes it: "all collected" lists frames from it.
        let line = crate::collect::record::IndexLine {
            id: stem.into(),
            timestamp: "2026-10-21T01:02:03.004Z".into(),
            scene: scene.into(),
            camera: camera.into(),
            model: "m".into(),
            buckets: vec![],
            labels: vec![],
            bytes: 1,
        };
        let day = dir.join(scene).join(camera).join("2026-10-21");
        let mut text = std::fs::read_to_string(day.join("index.jsonl")).unwrap_or_default();
        text.push_str(&serde_json::to_string(&line).unwrap());
        text.push('\n');
        std::fs::write(day.join("index.jsonl"), text).unwrap();
        let rel = format!("{scene}/{camera}/2026-10-21/{stem}.jpg");
        let path = Frame::resolve(dir, &rel).unwrap().labels;
        let mut f = crate::collect::labels::read(&path).unwrap().unwrap();
        f.status = status;
        f.empty_confirmed = empty;
        if status == Status::Reviewed {
            f.reviewed_at = Some("2026-10-22T00:00:00Z".into());
        }
        crate::collect::labels::write(&path, &f).unwrap();
    }

    /// Two cameras and scenes:
    /// - `yard` / `night`: 2 reviewed frames (3 accepted persons, 1 accepted car, 1 rejected
    ///   person, a dog, a bicycle), 1 confirmed empty, 2 unreviewed with a pending person (one
    ///   with its person rejected on the other box too), 1 unreviewed with a pending car only.
    /// - `yard` / `day`: 1 reviewed frame with a truck.
    /// - `gate` / `ir`: 1 unreviewed frame with a pending person, nothing reviewed.
    /// - a hidden class (`dog` hidden in the config of the tests that say so).
    fn folder(dir: &Path) -> PathBuf {
        let review = dir.join("dataset");
        let acc = |l: &str| lbox(l, BoxState::Accepted);
        let pend = |l: &str| lbox(l, BoxState::Pending);
        frame(
            &review,
            "night",
            "yard",
            "n1",
            Status::Reviewed,
            false,
            vec![
                acc("person"),
                acc("person"),
                acc("car"),
                lbox("person", BoxState::Rejected),
                acc("dog"),
            ],
        );
        frame(
            &review,
            "night",
            "yard",
            "n2",
            Status::Reviewed,
            false,
            vec![acc("person"), acc("bicycle")],
        );
        frame(
            &review,
            "night",
            "yard",
            "n3",
            Status::Reviewed,
            true,
            vec![],
        );
        // Reviewed with its only box rejected: an empty frame for training, though not ticked.
        frame(
            &review,
            "night",
            "yard",
            "n5",
            Status::Reviewed,
            false,
            vec![lbox("person", BoxState::Rejected)],
        );
        // Reviewed with only an unsure object: no negative, so not empty.
        let mut unsure = acc("dog");
        unsure.difficult = true;
        frame(
            &review,
            "night",
            "yard",
            "n6",
            Status::Reviewed,
            false,
            vec![unsure],
        );
        frame(
            &review,
            "night",
            "yard",
            "u1",
            Status::Unreviewed,
            false,
            vec![pend("person"), pend("car")],
        );
        frame(
            &review,
            "night",
            "yard",
            "u2",
            Status::Unreviewed,
            false,
            vec![pend("person")],
        );
        frame(
            &review,
            "night",
            "yard",
            "u3",
            Status::Unreviewed,
            false,
            vec![pend("car")],
        );
        // An unreviewed frame whose only person was rejected: not a person frame.
        frame(
            &review,
            "night",
            "yard",
            "u4",
            Status::Unreviewed,
            false,
            vec![lbox("person", BoxState::Rejected)],
        );
        frame(
            &review,
            "day",
            "yard",
            "d1",
            Status::Reviewed,
            false,
            vec![acc("truck")],
        );
        frame(
            &review,
            "ir",
            "gate",
            "g1",
            Status::Unreviewed,
            false,
            vec![pend("person")],
        );
        review
    }

    fn classes() -> Vec<String> {
        ["person", "car", "truck", "dog", "bicycle"]
            .map(String::from)
            .to_vec()
    }

    fn row<'a>(v: &'a serde_json::Value, camera: &str, scene: &str) -> &'a serde_json::Value {
        v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["camera"] == camera && r["scene"] == scene)
            .unwrap_or_else(|| panic!("no row {camera}/{scene}: {v}"))
    }

    /// `(camera, scene, class, boxes, found)`: one recall row of a report.
    type Group = (&'static str, &'static str, &'static str, u64, u64);

    /// A report with the recall rows `groups` (built by the report's own tally), written into
    /// `review/refine/report.json`; a quick check's when `mode` says so.
    fn put_report(review: &Path, mode: Mode, groups: &[Group]) {
        let mut r = RefineReport::new(
            "run1",
            4,
            crate::training::refine::recipe_info(
                &crate::training::recipe::TrainRecipe::new(
                    crate::resources::catalog::TrainFamily::Rfdetr,
                    "nano",
                )
                .unwrap(),
                "the default",
            ),
        );
        let mut tally = report::RecallTally::default();
        for (camera, scene, class, boxes, found) in groups {
            let facts: Vec<BoxFact> = (0..*boxes)
                .map(|i| BoxFact {
                    class: (*class).into(),
                    short_px: 30.0,
                    found: i < *found,
                    at_spot: false,
                    under_floor: false,
                })
                .collect();
            tally.add(camera, scene, &facts);
        }
        r.finish(&tally);
        if mode == Mode::Model {
            r.mode = Mode::Model;
            r.model = Some(ModelInfo {
                name: "custom-1".into(),
                run: "run-1".into(),
                recipe: "rfdetr-nano".into(),
                imgsz: 384,
                ..ModelInfo::default()
            });
        }
        let path = report::report_path(review);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&r).unwrap()).unwrap();
    }

    #[tokio::test]
    async fn the_route_needs_a_sign_in() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, _) = json(
            &state,
            super::super::test_util::anonymous(get("/v1/refine/coverage")),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        let (s, v) = json(&state, get("/v1/refine/coverage")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
    }

    #[test]
    fn rows_count_reviewed_boxes_empty_frames_and_unreviewed_person_frames() {
        let dir = scratch();
        let review = folder(&dir);
        let rows = compute(&review, &classes(), &[]);
        let v = answer(&rows, None);
        // Sorted by camera, then day / night / ir.
        let keys: Vec<(String, String)> = v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r["camera"].as_str().unwrap().to_string(),
                    r["scene"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            keys,
            [("gate", "ir"), ("yard", "day"), ("yard", "night")]
                .map(|(a, b)| (a.to_string(), b.to_string()))
        );
        let night = row(&v, "yard", "night");
        assert_eq!(night["reviewed"], 5);
        // The rejected person, the dog (a class but no person or vehicle) and pending boxes of
        // unreviewed frames are not counted.
        assert_eq!(night["persons"], 3);
        // One car and one bicycle.
        assert_eq!(night["vehicles"], 2);
        // n3 (ticked) and n5 (only a rejected box); n6 holds an unsure object.
        assert_eq!(night["empty"], 2);
        assert_eq!(night["empty_confirmed"], 1);
        // u1 and u2; u3 has a car only, u4 only a rejected person.
        assert_eq!(night["unreviewed_person_frames"], 2);
        assert!(night["person"].is_null() && night["vehicle"].is_null());
        let day = row(&v, "yard", "day");
        assert_eq!(
            (
                day["reviewed"].as_u64(),
                day["persons"].as_u64(),
                day["vehicles"].as_u64()
            ),
            (Some(1), Some(0), Some(1))
        );
        let gate = row(&v, "gate", "ir");
        assert_eq!(gate["reviewed"], 0);
        assert_eq!(gate["unreviewed_person_frames"], 1);
        assert_eq!(v["totals"]["rows"], 3);
        assert_eq!(v["totals"]["reviewed"], 6);
        assert_eq!(v["totals"]["persons"], 3);
        assert_eq!(v["totals"]["vehicles"], 3);
        assert_eq!(v["totals"]["empty"], 2);
        assert_eq!(v["totals"]["empty_confirmed"], 1);
        assert_eq!(v["totals"]["unreviewed_person_frames"], 3);
    }

    #[test]
    fn a_hidden_or_unlisted_class_counts_nowhere() {
        let dir = scratch();
        let review = folder(&dir);
        // Person hidden: no person boxes, no person frames; the vehicles stay.
        let rows = compute(&review, &classes(), &["person".to_string()]);
        let v = answer(&rows, None);
        let night = row(&v, "yard", "night");
        assert_eq!(night["persons"], 0);
        assert_eq!(night["unreviewed_person_frames"], 0);
        assert_eq!(night["vehicles"], 2);
        // Bicycle off the class list.
        let short: Vec<String> = ["person", "car"].map(String::from).to_vec();
        let v = answer(&compute(&review, &short, &[]), None);
        assert_eq!(row(&v, "yard", "night")["vehicles"], 1);
        assert_eq!(row(&v, "yard", "day")["vehicles"], 0, "truck is not listed");
    }

    #[test]
    fn the_recall_is_joined_from_the_report_with_vehicles_summed() {
        let dir = scratch();
        let review = folder(&dir);
        put_report(
            &review,
            Mode::Folds,
            &[
                ("yard", "night", "person", 40, 30),
                ("yard", "night", "car", 20, 18),
                ("yard", "night", "bicycle", 10, 4),
                ("yard", "night", "dog", 5, 5),
                ("yard", "day", "truck", 12, 12),
            ],
        );
        let (report, problem) = read_report(&review);
        assert!(problem.is_none());
        let v = answer(&compute(&review, &classes(), &[]), report.as_ref());
        let night = row(&v, "yard", "night");
        assert_eq!(night["person"]["boxes"], 40);
        assert_eq!(night["person"]["found"], 30);
        assert!((night["person"]["recall"].as_f64().unwrap() - 0.75).abs() < 1e-9);
        let (lo, hi) = report::wilson(30, 40);
        assert!((night["person"]["low"].as_f64().unwrap() - lo).abs() < 1e-5);
        assert!((night["person"]["high"].as_f64().unwrap() - hi).abs() < 1e-5);
        // Car and bicycle summed (the dog is no vehicle); the interval is of the sums.
        assert_eq!(night["vehicle"]["boxes"], 30);
        assert_eq!(night["vehicle"]["found"], 22);
        let (lo, hi) = report::wilson(22, 30);
        assert!((night["vehicle"]["low"].as_f64().unwrap() - lo).abs() < 1e-5);
        assert!((night["vehicle"]["high"].as_f64().unwrap() - hi).abs() < 1e-5);
        // No row for persons by day, none for the gate: null, never zero.
        assert!(row(&v, "yard", "day")["person"].is_null());
        assert_eq!(row(&v, "yard", "day")["vehicle"]["boxes"], 12);
        assert!(row(&v, "gate", "ir")["person"].is_null());
        assert!(row(&v, "gate", "ir")["vehicle"].is_null());
    }

    #[tokio::test]
    async fn a_quick_checks_recall_is_joined_the_same_way() {
        let dir = scratch();
        let review = folder(&dir);
        put_report(&review, Mode::Model, &[("yard", "night", "person", 8, 6)]);
        let state = state_in(&dir);
        let (s, v) = json(&state, get("/v1/refine/coverage")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(row(&v, "yard", "night")["person"]["boxes"], 8);
        assert_eq!(row(&v, "yard", "night")["person"]["found"], 6);
        // The route answers the counts too, through the process cache's own path.
        assert_eq!(row(&v, "yard", "night")["persons"], 3);
        let (report, _) = read_report(&review);
        assert_eq!(report.unwrap().mode, Mode::Model);
    }

    #[tokio::test]
    async fn an_unplugged_folder_is_reported_and_never_an_empty_dataset() {
        let dir = scratch();
        let state = test_state_with({
            let dir = dir.clone();
            move |c| {
                c.download_dir = Some(dir.clone());
                c.collect.dir = Some(dir.join("dataset"));
                c.collect.review_dir = Some(dir.join("not-there"));
                c.training.dir = Some(dir.join("training"));
            }
        });
        let (s, v) = json(&state, get("/v1/refine/coverage")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["folder_missing"], true);
        assert!(v["message"].as_str().is_some_and(|m| !m.is_empty()));
        assert_eq!(v["rows"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn the_walk_is_kept_until_the_queue_changes_or_a_write_drops_it() {
        let dir = scratch();
        let review = folder(&dir);
        let none: Vec<String> = Vec::new();
        // A cache of its own: other tests clear the process's at any time.
        let cache = Cache::new();
        let first = cache.rows(&review, &classes(), &none, "");
        assert_eq!(first.len(), 3);
        let again = cache.rows(&review, &classes(), &none, "");
        assert!(Arc::ptr_eq(&first, &again), "kept");
        forget();
        let kept = cache.rows(&review, &classes(), &none, "");
        assert!(Arc::ptr_eq(&first, &kept), "another cache's forget");
        let other_classes = cache.rows(&review, &classes()[..2], &none, "");
        assert!(!Arc::ptr_eq(&first, &other_classes), "another class list");
        let other = cache.rows(&review, &classes(), &none, "t1:running");
        assert!(!Arc::ptr_eq(&first, &other), "a check changed state");
        cache.forget();
        let third = cache.rows(&review, &classes(), &none, "t1:running");
        assert!(!Arc::ptr_eq(&other, &third), "dropped");
        // Bounded: more folders than KEEP leave at most KEEP.
        for i in 0..(KEEP + 3) {
            cache.rows(&review.join(format!("x{i}")), &classes(), &none, "");
        }
        assert!(cache.kept.lock_recover().len() <= KEEP);
    }

    /// The link the page builds opens exactly the frames the count names: the Review queue under
    /// the same query (`RefineFloor.coverageUrl`) lists as many frames as `unreviewed_person_frames`.
    #[test]
    fn the_review_link_opens_exactly_the_counted_frames() {
        let dir = scratch();
        let review = folder(&dir);
        let v = answer(&compute(&review, &classes(), &[]), None);
        for r in v["rows"].as_array().unwrap() {
            let q: std::collections::BTreeMap<String, String> = [
                ("status", "unreviewed"),
                ("source", "all"),
                ("scene", r["scene"].as_str().unwrap()),
                ("camera", r["camera"].as_str().unwrap()),
                ("label", "person"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
            let filters = Filters::from_query(&q).unwrap();
            let queue = queue::build(&review, &filters);
            assert_eq!(
                queue.items.len() as u64,
                r["unreviewed_person_frames"].as_u64().unwrap(),
                "{r}"
            );
        }
    }
}
