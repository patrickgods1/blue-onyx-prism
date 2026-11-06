//! The lineup end to end on synthetic runs: who is on a set, the shared label space, the marks and
//! the dashes.

use super::*;
use crate::benchmark::images::GroundTruth;
use crate::benchmark::metrics::GtBox;
use crate::benchmark::report::test_support::{dev, model as base_model};
use crate::benchmark::report::{DeviceResult, ImageInfo};
use crate::benchmark::search::{StoredImage, StoredPred, StoredPreds, StoredRun};
use crate::benchmark::{ImageRun, images::SetKind};

pub(super) fn results() -> BenchmarkResults {
    serde_json::from_str("{}").unwrap()
}

pub(super) fn model(name: &str, datasets: &[&str], devices: Vec<DeviceResult>) -> ModelResult {
    let mut m = base_model(devices, None);
    m.model = name.into();
    m.datasets = datasets.iter().map(ToString::to_string).collect();
    m
}

const BOX: [f32; 4] = [10.0, 10.0, 110.0, 110.0];
const OTHER: [f32; 4] = [300.0, 300.0, 400.0, 400.0];

fn img(file: &str, tags: &[&str], labels: &[&str]) -> ImageInfo {
    ImageInfo {
        file: file.into(),
        width: 640,
        height: 640,
        tags: tags.iter().map(ToString::to_string).collect(),
        objects: Some(
            labels
                .iter()
                .map(|l| GtBox {
                    label: (*l).into(),
                    bbox: BOX,
                    ignore: false,
                    crowd: false,
                })
                .collect(),
        ),
    }
}

fn set(id: &str, kind: SetKind, images: Vec<ImageInfo>) -> ImageSetInfo {
    ImageSetInfo {
        id: id.into(),
        title: id.into(),
        kind,
        dir: None,
        license: String::new(),
        attribution: String::new(),
        source: String::new(),
        ground_truth: GroundTruth::Yolo,
        scored_labels: None,
        images,
    }
}

/// (class index, confidence, box).
type Pred = (u32, f32, [f32; 4]);

/// A stored run: per file, its predictions.
fn stored(
    name: &str,
    device: &str,
    classes: &[&str],
    set: &str,
    images: Vec<(&str, Vec<Pred>)>,
) -> StoredRun {
    StoredRun {
        model: name.into(),
        device: device.into(),
        timestamp: String::new(),
        classes: classes.iter().map(ToString::to_string).collect(),
        configured: 0.5,
        eval_confidence: 0.05,
        images: images
            .into_iter()
            .map(|(file, preds)| StoredImage {
                set: set.into(),
                file: file.into(),
                preds: preds
                    .into_iter()
                    .map(|(class, confidence, bbox)| StoredPred {
                        class,
                        confidence,
                        bbox,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn timed(mut d: DeviceResult, set: &str, files: &[(&str, f64)]) -> DeviceResult {
    if let Some(r) = d.run.as_mut() {
        r.per_image = files
            .iter()
            .map(|(file, ms)| ImageRun {
                set: set.into(),
                file: (*file).into(),
                total_ms: *ms,
                infer_ms: *ms,
                preds: Vec::new(),
                gt_matched: Vec::new(),
                counts: None,
            })
            .collect();
    }
    d
}

struct World {
    results: BenchmarkResults,
    preds: StoredPreds,
    now: HashMap<String, ModelNow>,
    configured: Vec<Configured>,
    trained: HashMap<String, BTreeSet<String>>,
    default_model: Option<String>,
}

impl World {
    fn new(results: BenchmarkResults, runs: Vec<StoredRun>) -> Self {
        let configured = results
            .models
            .iter()
            .map(|m| Configured {
                name: m.model.clone(),
                blocked: None,
            })
            .collect();
        Self {
            results,
            preds: StoredPreds {
                version: String::new(),
                runs,
            },
            now: HashMap::new(),
            configured,
            trained: HashMap::new(),
            default_model: None,
        }
    }

    fn build(&self) -> Lineup {
        let history = History::default();
        build(&Inputs {
            results: &self.results,
            preds: Some(&self.preds),
            history: &history,
            now: &self.now,
            default_model: self.default_model.as_deref(),
            configured: &self.configured,
            trained: &self.trained,
            reachable: &|_| true,
        })
    }
}

fn now_at(t: f32, device: &str) -> ModelNow {
    ModelNow::new(t, None, Some(device.to_string()))
}

/// Two models on a set of ten day frames with a person and ten IR frames with a car, plus five
/// empty frames: `good` finds all, `poor` half and raises a false alert on one empty frame.
fn pair() -> World {
    let (mut images, mut good, mut poor) = (Vec::new(), Vec::new(), Vec::new());
    let mut good_ms = Vec::new();
    let mut poor_ms = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for i in 0..10 {
        files.push(format!("d{i}.jpg"));
        files.push(format!("i{i}.jpg"));
    }
    for i in 0..5 {
        files.push(format!("e{i}.jpg"));
    }
    for f in &files {
        let (tags, labels, class): (&[&str], &[&str], u32) = match f.chars().next() {
            Some('d') => (&["day"], &["person"], 0),
            Some('i') => (&["ir"], &["car"], 1),
            _ => (&["day"], &[], 0),
        };
        images.push(img(f, tags, labels));
        let n: u32 = f[1..f.len() - 4].parse().unwrap();
        let g = if labels.is_empty() {
            Vec::new()
        } else {
            vec![(class, 0.9, BOX)]
        };
        let p = if f.starts_with('e') && n == 0 {
            vec![(0, 0.9, OTHER)]
        } else if !labels.is_empty() && n < 5 {
            vec![(class, 0.9, BOX)]
        } else {
            Vec::new()
        };
        good.push((f.as_str(), g));
        poor.push((f.as_str(), p));
        good_ms.push((f.as_str(), 100.0));
        poor_ms.push((f.as_str(), 50.0));
    }
    let mut r = results();
    r.image_sets = vec![set("own", SetKind::Dir, images)];
    r.models = vec![
        model(
            "good",
            &["own"],
            vec![timed(dev("ort:coreml", 1.0, Vec::new()), "own", &good_ms)],
        ),
        model(
            "poor",
            &["own"],
            vec![timed(dev("ort:coreml", 1.0, Vec::new()), "own", &poor_ms)],
        ),
    ];
    let runs = vec![
        stored("good", "ort:coreml", &["person", "car"], "own", good),
        stored("poor", "ort:coreml", &["person", "car"], "own", poor),
    ];
    let mut w = World::new(r, runs);
    w.now.insert("good".into(), now_at(0.5, "ort:coreml"));
    w.now.insert("poor".into(), now_at(0.5, "ort:coreml"));
    w
}

fn row<'a>(s: &'a SetView, model: &str) -> &'a Row {
    s.views[0].rows.iter().find(|r| r.model == model).unwrap()
}

#[test]
fn every_model_answers_for_the_same_objects() {
    let l = pair().build();
    assert_eq!(l.default_set.as_deref(), Some("own"));
    let s = &l.sets[0];
    assert_eq!((s.images, s.objects, s.empty_frames), (25, 20, 5));
    assert_eq!(s.classes, ["person", "vehicle"]);
    let (g, p) = (row(s, "good"), row(s, "poor"));
    assert_eq!(g.objects["all"].finds.all.value, Some(1.0));
    assert_eq!(p.objects["all"].finds.all.value, Some(0.5));
    assert_eq!(g.objects["all"].finds.all.n, 20);
    assert_eq!(p.objects["all"].finds.all.n, 20);
    // Per class and per scene.
    assert_eq!(p.objects["person"].finds.all.value, Some(0.5));
    assert_eq!(p.objects["vehicle"].finds.all.n, 10);
    assert_eq!(p.objects["all"].finds.day.n, 10);
    assert_eq!(p.objects["all"].finds.ir.value, Some(0.5));
    // No night frames: a dash, not 0.
    assert_eq!(g.objects["all"].finds.night.value, None);
    assert_eq!(g.objects["all"].correct.night.value, None);
}

#[test]
fn false_alerts_are_measured_on_empty_frames_only() {
    let l = pair().build();
    let s = &l.sets[0];
    assert_eq!(row(s, "poor").false_alerts.value, Some(0.2));
    assert_eq!(row(s, "poor").alerts, 1);
    assert_eq!(row(s, "poor").false_alerts.n, 5);
    assert_eq!(row(s, "good").false_alerts.value, Some(0.0));
    // A false alert on one frame of five is inside the noise (one frame in five).
    assert!(!row(s, "good").false_alerts.best);
}

#[test]
fn best_follows_the_noise_rule_and_time_comes_from_these_images() {
    let l = pair().build();
    let s = &l.sets[0];
    let (g, p) = (row(s, "good"), row(s, "poor"));
    assert!(g.objects["all"].finds.all.best, "100% against 50% of 20");
    assert!(!p.objects["all"].finds.all.best);
    assert_eq!(g.speed.value, Some(100.0));
    assert_eq!(g.speed.n, 25);
    assert!(p.speed.best && !g.speed.best);
    assert_eq!(s.views[0].headline.call, verdict::Call::Clear);
    assert!(
        s.views[0].headline.text.starts_with("good finds more"),
        "{}",
        s.views[0].headline.text
    );
}

#[test]
fn a_model_with_training_frames_in_the_set_is_shown_but_not_best() {
    let mut w = pair();
    w.results.models[0].trained_on_share = Some(0.4);
    let l = w.build();
    let s = &l.sets[0];
    let g = row(s, "good");
    assert!(g.excluded);
    assert_eq!(g.trained_on_share, Some(0.4));
    assert!(!g.objects["all"].finds.all.best, "excluded from best");
    assert_eq!(s.views[0].headline.call, verdict::Call::Alone);
    // The frame list, when known, decides per set: no frame of this set is in it.
    w.trained.insert(
        "good".into(),
        ["day/back/2026-10-25/other.jpg".to_string()].into(),
    );
    let l = w.build();
    assert!(!row(&l.sets[0], "good").excluded);
    assert_eq!(row(&l.sets[0], "good").trained_on_share, Some(0.0));
}

#[test]
fn the_default_model_is_labelled() {
    let mut w = pair();
    w.default_model = Some("POOR".into());
    let l = w.build();
    assert!(row(&l.sets[0], "poor").is_default);
    assert!(!row(&l.sets[0], "good").is_default);
}

#[test]
fn each_model_sits_at_its_own_confidence_and_best_is_a_second_view() {
    let mut w = pair();
    w.now.insert("good".into(), now_at(0.95, "ort:coreml"));
    let l = w.build();
    let s = &l.sets[0];
    let g = row(s, "good");
    assert!((g.confidence - 0.95).abs() < 1e-6);
    assert_eq!(
        g.objects["all"].finds.all.value,
        Some(0.0),
        "0.9 is under 0.95"
    );
    assert_eq!(
        s.views
            .iter()
            .map(|v| v.confidence.as_str())
            .collect::<Vec<_>>(),
        ["now", "best"]
    );
}

#[test]
fn a_model_on_other_images_is_listed_with_why_and_a_way_to_run_it() {
    let mut w = pair();
    let mut other = model("other", &["coco"], vec![dev("ort:coreml", 1.0, Vec::new())]);
    other.images = 40;
    w.results.models.push(other);
    w.results.image_sets.push(set(
        "coco",
        SetKind::Builtin,
        vec![img("c.jpg", &[], &["person"])],
    ));
    w.preds.runs.push(stored(
        "other",
        "ort:coreml",
        &["person"],
        "coco",
        vec![("c.jpg", vec![(0, 0.9, BOX)])],
    ));
    w.configured.push(Configured {
        name: "other".into(),
        blocked: None,
    });
    w.configured.push(Configured {
        name: "fresh".into(),
        blocked: Some("the model file is missing".into()),
    });
    let l = w.build();
    let own = l.sets.iter().find(|s| s.id == "own").unwrap();
    assert_eq!(own.views[0].rows.len(), 2);
    let o = own.missing.iter().find(|m| m.model == "other").unwrap();
    assert_eq!(o.why, "scored on 40 other images (coco)");
    assert_eq!(o.device.as_deref(), Some("ort:coreml"));
    assert_eq!(o.blocked, None);
    let f = own.missing.iter().find(|m| m.model == "fresh").unwrap();
    assert_eq!(f.why, "not benchmarked yet");
    assert_eq!(f.blocked.as_deref(), Some("the model file is missing"));
    assert_eq!(own.run.datasets, ["own"]);
    assert_eq!(own.run.max_images, 25);
    // The one model on coco is alone there.
    let coco = l.sets.iter().find(|s| s.id == "coco").unwrap();
    assert_eq!(coco.views[0].headline.call, verdict::Call::Alone);
    // The default is the set with the most models.
    assert_eq!(l.default_set.as_deref(), Some("own"));
    assert_eq!(l.sets[0].id, "own", "best set first");
}

#[test]
fn a_run_of_part_of_the_images_is_not_on_the_set() {
    let mut w = pair();
    w.preds.runs[1].images.truncate(10);
    let l = w.build();
    let s = &l.sets[0];
    assert_eq!(s.views[0].rows.len(), 1);
    assert_eq!(s.missing[0].model, "poor");
    assert_eq!(s.missing[0].why, "scored on only 10 of these 25 frames");
}

#[test]
fn a_run_without_stored_predictions_says_so() {
    let mut w = pair();
    w.preds.runs.remove(1);
    let l = w.build();
    assert_eq!(
        l.sets[0].missing[0].why,
        "its last run kept no predictions to score on these images"
    );
}

#[test]
fn classes_only_some_models_output_are_left_out_for_all() {
    let mut w = pair();
    // `poor` has a vehicle group class and no person: only vehicles are shared.
    w.preds.runs[1].classes = vec!["vehicle".into(), "dog".into()];
    let l = w.build();
    let s = &l.sets[0];
    assert_eq!(s.classes, ["vehicle"]);
    assert_eq!(s.left_out, ["dog", "person"]);
    assert_eq!(s.objects, 10, "the persons left the count for everyone");
    assert_eq!(row(s, "good").objects["all"].finds.all.n, 10);
    assert_eq!(row(s, "poor").objects["all"].finds.all.n, 10);
}

#[test]
fn car_truck_and_bus_are_one_vehicle_for_every_model() {
    // Labelled as car, detected as truck: a hit for both, as a vehicle-class model is scored.
    let mut r = results();
    r.image_sets = vec![set("s", SetKind::Dir, vec![img("a.jpg", &[], &["car"])])];
    r.models = vec![
        model("fine", &["s"], vec![dev("ort:coreml", 1.0, Vec::new())]),
        model("coarse", &["s"], vec![dev("ort:coreml", 1.0, Vec::new())]),
    ];
    let w = World::new(
        r,
        vec![
            stored(
                "fine",
                "ort:coreml",
                &["car", "truck"],
                "s",
                vec![("a.jpg", vec![(1, 0.9, BOX)])],
            ),
            stored(
                "coarse",
                "ort:coreml",
                &["vehicle"],
                "s",
                vec![("a.jpg", vec![(0, 0.9, BOX)])],
            ),
        ],
    );
    let l = w.build();
    let s = &l.sets[0];
    assert_eq!(s.classes, ["vehicle"]);
    assert_eq!(row(s, "fine").objects["all"].finds.all.value, Some(1.0));
    assert_eq!(row(s, "coarse").objects["all"].finds.all.value, Some(1.0));
}

#[test]
fn models_on_different_devices_flag_speed_and_keep_accuracy() {
    let mut w = pair();
    w.results.models[1].devices = vec![dev("openvino:cpu", 1.0, Vec::new())];
    w.preds.runs[1].device = "openvino:cpu".into();
    let l = w.build();
    let s = &l.sets[0];
    assert!(!s.speed.comparable);
    assert!(
        s.speed.reasons[0].contains("no device in common"),
        "{:?}",
        s.speed
    );
    assert!(!row(s, "good").speed.best && !row(s, "poor").speed.best);
    assert!(
        row(s, "good").objects["all"].finds.all.best,
        "accuracy stays"
    );
}

#[test]
fn a_shared_device_is_preferred_over_each_models_own() {
    let mut w = pair();
    for m in &mut w.results.models {
        m.devices.push(dev("openvino:cpu", 2.0, Vec::new()));
    }
    let extra: Vec<StoredRun> = w
        .preds
        .runs
        .iter()
        .map(|r| {
            let mut c = r.clone();
            c.device = "openvino:cpu".into();
            c
        })
        .collect();
    w.preds.runs.extend(extra);
    w.now.insert("good".into(), now_at(0.5, "openvino:cpu"));
    w.now.insert("poor".into(), now_at(0.5, "ort:coreml"));
    w.default_model = Some("good".into());
    let l = w.build();
    let s = &l.sets[0];
    assert_eq!(row(s, "good").device, "openvino:cpu");
    assert_eq!(row(s, "poor").device, "openvino:cpu");
    assert!(s.speed.comparable, "{:?}", s.speed);
}

#[test]
fn held_out_frames_of_an_export_are_their_own_set_and_open_first() {
    let mut w = pair();
    for i in &mut w.results.image_sets[0].images {
        if i.file.ends_with("0.jpg") || i.file.ends_with("1.jpg") {
            i.tags.push("split:val".into());
        }
    }
    w.results.models[0].trained_on_share = Some(0.5);
    // `good` trained on the others: the frame list says none of the held-out ones.
    let frames: BTreeSet<String> = (2..10)
        .map(|i| format!("day/back/2026-10-10/d{i}.jpg"))
        .collect();
    w.trained.insert("good".into(), frames);
    for i in &mut w.results.image_sets[0].images {
        i.file = format!("day.back.2026-10-10.{}", i.file);
    }
    for r in &mut w.preds.runs {
        for i in &mut r.images {
            i.file = format!("day.back.2026-10-10.{}", i.file);
        }
    }
    let l = w.build();
    assert_eq!(l.default_set.as_deref(), Some("own#split:val"));
    let val = &l.sets[0];
    assert_eq!(val.role, SetRole::HeldOut);
    assert_eq!(row(val, "good").trained_on_share, Some(0.0));
    let all = l.sets.iter().find(|s| s.id == "own").unwrap();
    assert!(row(all, "good").excluded);
    assert!(row(all, "good").trained_on_share.unwrap() > 0.0);
}

#[test]
fn the_reference_model_does_not_score_itself() {
    let mut w = pair();
    w.results.image_sets[0].ground_truth = GroundTruth::Pseudo("good on openvino:cpu".into());
    let l = w.build();
    let s = &l.sets[0];
    assert!(s.relative);
    assert_eq!(s.views[0].rows.len(), 1);
    assert_eq!(s.missing[0].model, "good");
    assert!(
        s.missing[0].why.contains("its own labels"),
        "{}",
        s.missing[0].why
    );
    assert!(s.missing[0].blocked.is_some(), "no button that cannot help");
}

#[test]
fn an_unreachable_folder_blocks_the_button() {
    let mut w = pair();
    w.preds.runs.remove(1);
    let history = History::default();
    let l = build(&Inputs {
        results: &w.results,
        preds: Some(&w.preds),
        history: &history,
        now: &w.now,
        default_model: None,
        configured: &w.configured,
        trained: &w.trained,
        reachable: &|_| false,
    });
    assert!(
        l.sets[0].missing[0]
            .blocked
            .as_deref()
            .unwrap()
            .contains("cannot be read")
    );
}

#[test]
fn no_results_no_sets_and_pairs_cover_the_models() {
    let w = World::new(results(), Vec::new());
    let l = w.build();
    assert!(l.sets.is_empty() && l.default_set.is_none());
    let l = pair().build();
    assert_eq!(l.pairs.len(), 1);
    assert!(l.pairs[0].accuracy.comparable, "{:?}", l.pairs[0]);
}

#[test]
fn the_contract_of_a_cell_is_null_not_zero() {
    let l = pair().build();
    let v = serde_json::to_value(&l).unwrap();
    let cell = &v["sets"][0]["views"][0]["rows"][0]["objects"]["all"]["finds"]["night"];
    assert!(cell["value"].is_null(), "{cell}");
    assert_eq!(v["defaultSet"], "own");
    assert_eq!(v["sets"][0]["views"][0]["rows"][0]["isDefault"], false);
    assert_eq!(v["sets"][0]["views"][0]["headline"]["call"], "clear");
}
