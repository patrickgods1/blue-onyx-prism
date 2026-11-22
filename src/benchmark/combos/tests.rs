//! Who is on a set and on which frames, end to end through the Compare view on synthetic runs: the
//! shared label space, the fair frames of "Your cameras", the left-out and missing models, the
//! reference model of a relative set.

use super::*;
use crate::benchmark::current::ModelNow;
use crate::benchmark::eval::Goal;
use crate::benchmark::eval::compare::{self, CompareView};
use crate::benchmark::eval::rank::Block;
use crate::benchmark::images::GroundTruth;
use crate::benchmark::metrics::GtBox;
use crate::benchmark::report::test_support::{dev, model as base_model};
use crate::benchmark::report::{DeviceResult, ImageInfo};
use crate::benchmark::search::{StoredImage, StoredPred, StoredPreds, StoredRun};
use crate::benchmark::{ImageRun, images::SetKind};
use sets::SetRole;

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
        scene: None,
        group: None,
    }
}

fn set(id: &str, kind: SetKind, images: Vec<ImageInfo>) -> ImageSetInfo {
    ImageSetInfo {
        snapshot: None,
        labels_stale: None,
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
                        window: None,
                    })
                    .collect(),
                group: None,
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
                scene: None,
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
    seen: HashMap<String, crate::benchmark::cameras::seen::Seen>,
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
            seen: HashMap::new(),
            default_model: None,
        }
    }

    fn inputs(&self) -> Inputs<'_> {
        Inputs {
            results: &self.results,
            preds: Some(&self.preds),
            now: &self.now,
            default_model: self.default_model.as_deref(),
            configured: &self.configured,
            trained: &self.trained,
            seen: &self.seen,
            reachable: &|_| true,
        }
    }

    /// The Compare view of `set` (None: the default set) for the default goal (F1).
    fn compare(&self, set: Option<&str>) -> CompareView {
        compare_on(&self.inputs(), set)
    }
}

fn compare_on(inputs: &Inputs<'_>, set: Option<&str>) -> CompareView {
    compare::build(
        inputs,
        compare::Request {
            goal: &Goal::default(),
            weights: Default::default(),
            set,
            scene: None,
        },
    )
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

fn row<'a>(v: &'a CompareView, model: &str) -> &'a compare::Row {
    v.rows
        .iter()
        .find(|r| r.model == model)
        .unwrap_or_else(|| panic!("{model} is not on {:?}", v.set.as_ref().map(|s| &s.id)))
}

fn scene<'a>(r: &'a compare::Row, s: &str) -> Option<&'a compare::SceneRow> {
    r.scenes.iter().find(|x| x.scene == s)
}

/// Every model answers for the same objects: one label space (car, truck and bus are one
/// vehicle), the same frames, per scene where the set has them.
#[test]
fn every_model_answers_for_the_same_objects() {
    let v = pair().compare(None);
    let s = v.set.as_ref().unwrap();
    assert_eq!(s.id, "own");
    assert_eq!((s.frames, s.objects, s.empty_frames), (25, 20, 5));
    assert_eq!(s.scored_classes, ["person", "vehicle"]);
    let (g, p) = (row(&v, "good"), row(&v, "poor"));
    assert_eq!(g.recall, Some(1.0));
    assert_eq!(p.recall, Some(0.5));
    let person = p.classes.iter().find(|c| c.class == "person").unwrap();
    assert_eq!((person.objects, person.recall), (10, Some(0.5)));
    let vehicle = p.classes.iter().find(|c| c.class == "vehicle").unwrap();
    assert_eq!(vehicle.objects, 10);
    assert_eq!(scene(p, "day").unwrap().objects, 10);
    assert_eq!(scene(p, "ir").unwrap().recall, Some(0.5));
    // No night frames: no night row, not a row of zeros.
    assert!(scene(g, "night").is_none_or(|n| n.frames == 0 && n.recall.is_none()));
}

/// The Day / Night / IR rows follow the scene the run computed for each frame over the
/// manifest's tags (here every frame computed as night: the day and IR tags no longer count).
#[test]
fn scene_rows_follow_the_computed_scene() {
    let mut w = pair();
    for s in &mut w.results.image_sets {
        for i in &mut s.images {
            i.scene = Some(crate::image_stats::Scene::Night);
        }
    }
    let v = w.compare(None);
    let p = row(&v, "poor");
    let night = scene(p, "night").unwrap();
    assert_eq!((night.objects, night.recall), (20, Some(0.5)));
    assert!(scene(p, "day").is_none_or(|d| d.objects == 0));
    assert!(scene(p, "ir").is_none_or(|i| i.recall.is_none()));
}

#[test]
fn false_alerts_are_measured_on_empty_frames_only() {
    let v = pair().compare(None);
    assert_eq!(v.set.as_ref().unwrap().empty_frames, 5);
    assert_eq!(
        row(&v, "poor").fa100,
        Some(20.0),
        "one of five empty frames"
    );
    assert_eq!(row(&v, "good").fa100, Some(0.0));
}

/// The better model ranks first, clearly; each row's speed is its p50 on these images.
#[test]
fn the_rank_follows_the_goal_and_time_comes_from_these_images() {
    let v = pair().compare(None);
    assert_eq!(v.rows[0].model, "good");
    let (g, p) = (row(&v, "good"), row(&v, "poor"));
    assert_eq!((g.rank, g.block), (1, Block::Ok));
    assert!(
        p.rank > 1,
        "100% against 50% of 20 objects is clearly better"
    );
    assert_eq!(g.speed.p50_ms, Some(100.0));
    assert_eq!(p.speed.p50_ms, Some(50.0));
    assert_eq!(g.speed.frames, 25);
    assert!(
        v.headline.starts_with("good on ort:coreml"),
        "{}",
        v.headline
    );
}

#[test]
fn a_model_with_training_frames_in_the_set_is_shown_but_not_ranked_fairly() {
    let mut w = pair();
    w.results.models[0].trained_on_share = Some(0.4);
    let v = w.compare(None);
    let g = row(&v, "good");
    assert!(g.flags.trained_here);
    assert_eq!(g.flags.trained_on_share, Some(0.4));
    assert_eq!(g.block, Block::Unfair);
    assert_eq!(v.rows.last().unwrap().model, "good", "after the fair rows");
    assert_eq!(row(&v, "poor").rank, 1);
    // The frame list, when known, decides per set: no frame of this set is in it.
    w.trained.insert(
        "good".into(),
        ["day/back/2026-10-25/other.jpg".to_string()].into(),
    );
    let v = w.compare(None);
    let g = row(&v, "good");
    assert!(!g.flags.trained_here);
    assert_eq!(g.flags.trained_on_share, Some(0.0));
    assert_eq!((g.rank, g.block), (1, Block::Ok));
}

#[test]
fn the_default_model_is_labelled() {
    let mut w = pair();
    w.default_model = Some("POOR".into());
    let v = w.compare(None);
    assert!(row(&v, "poor").is_default);
    assert!(!row(&v, "good").is_default);
}

/// Each row shows the confidence the config gives it on that device beside the suggested one;
/// the figures are at the suggestion, not at the saved value.
#[test]
fn each_row_carries_its_saved_and_its_suggested_confidence() {
    let mut w = pair();
    w.now.insert("good".into(), now_at(0.95, "ort:coreml"));
    let v = w.compare(None);
    let g = row(&v, "good");
    assert_eq!(g.served_threshold, Some(0.95));
    assert!(g.confidence.unwrap() <= 0.9, "{:?}", g.confidence);
    assert_eq!(
        g.recall,
        Some(1.0),
        "0.9 detections count at the suggestion"
    );
    let at = g.at_saved.as_ref().unwrap();
    assert_eq!(at.recall, Some(0.0), "0.9 is under the saved 0.95");
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
    let v = w.compare(Some("own"));
    assert_eq!(v.rows.len(), 2);
    let o = v.missing.iter().find(|m| m.model == "other").unwrap();
    assert_eq!(o.why, "scored on 40 other images (coco)");
    assert_eq!(o.device.as_deref(), Some("ort:coreml"));
    assert_eq!(o.blocked, None);
    let f = v.missing.iter().find(|m| m.model == "fresh").unwrap();
    assert_eq!(f.why, "not benchmarked yet");
    assert_eq!(f.blocked.as_deref(), Some("the model file is missing"));
    let s = v.set.as_ref().unwrap();
    assert_eq!(s.run.datasets, ["own"]);
    assert_eq!(s.run.max_images, 25);
    // The one model on coco is alone there; the default is the set with the most models.
    let coco = w.compare(Some("coco"));
    assert_eq!(coco.rows.len(), 1);
    assert_eq!(
        v.sets.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        ["own", "coco"]
    );
    assert_eq!(w.compare(None).set.unwrap().id, "own");
}

#[test]
fn a_run_of_part_of_the_images_is_not_on_the_set() {
    let mut w = pair();
    w.preds.runs[1].images.truncate(10);
    let v = w.compare(None);
    assert_eq!(v.rows.len(), 1);
    assert_eq!(v.missing[0].model, "poor");
    assert_eq!(v.missing[0].why, "scored on only 10 of these 25 frames");
}

#[test]
fn a_run_without_stored_predictions_says_so() {
    let mut w = pair();
    w.preds.runs.remove(1);
    let v = w.compare(None);
    assert_eq!(
        v.missing[0].why,
        "its last run kept no predictions to score on these images"
    );
}

#[test]
fn classes_only_some_models_output_are_left_out_for_all() {
    let mut w = pair();
    // `poor` has a vehicle group class and no person: only vehicles are shared.
    w.preds.runs[1].classes = vec!["vehicle".into(), "dog".into()];
    let v = w.compare(None);
    let s = v.set.as_ref().unwrap();
    assert_eq!(s.scored_classes, ["vehicle"]);
    assert_eq!(s.objects, 10, "the persons left the count for everyone");
    for m in ["good", "poor"] {
        let r = row(&v, m);
        assert_eq!(
            r.classes
                .iter()
                .map(|c| c.class.as_str())
                .collect::<Vec<_>>(),
            ["vehicle"],
            "{m}"
        );
        assert_eq!(r.classes[0].objects, 10, "{m}");
    }
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
    let v = w.compare(None);
    assert_eq!(v.set.as_ref().unwrap().scored_classes, ["vehicle"]);
    assert_eq!(row(&v, "fine").recall, Some(1.0));
    assert_eq!(row(&v, "coarse").recall, Some(1.0));
}

/// Every device a model ran the set on is its own row, each with its own speed; accuracy is the
/// same predictions' wherever they ran.
#[test]
fn every_device_is_its_own_row() {
    let mut w = pair();
    w.results.models[1].devices = vec![dev("openvino:cpu", 1.0, Vec::new())];
    w.preds.runs[1].device = "openvino:cpu".into();
    for m in &mut w.results.models {
        if m.model == "good" {
            m.devices.push(timed(
                dev("openvino:cpu", 2.0, Vec::new()),
                "own",
                &[("d0.jpg", 300.0)],
            ));
        }
    }
    let mut extra = w.preds.runs[0].clone();
    extra.device = "openvino:cpu".into();
    w.preds.runs.push(extra);
    let v = w.compare(None);
    let combos: Vec<(&str, &str)> = v
        .rows
        .iter()
        .map(|r| (r.model.as_str(), r.device.as_str()))
        .collect();
    assert_eq!(combos.len(), 3, "{combos:?}");
    assert!(
        combos.contains(&("good", "openvino:cpu")) && combos.contains(&("poor", "openvino:cpu"))
    );
    let on = |m: &str, d: &str| {
        v.rows
            .iter()
            .find(|r| r.model == m && r.device == d)
            .unwrap()
    };
    assert_eq!(
        on("good", "ort:coreml").recall,
        on("good", "openvino:cpu").recall
    );
    assert_ne!(
        on("good", "ort:coreml").speed.p50_ms,
        on("good", "openvino:cpu").speed.p50_ms
    );
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
    let v = w.compare(None);
    let val = v.set.as_ref().unwrap();
    assert_eq!(val.id, "own#split:val");
    assert_eq!(val.what, SetRole::HeldOut.what());
    let g = row(&v, "good");
    assert_eq!(g.flags.trained_on_share, Some(0.0));
    assert!(!g.flags.trained_here);
    let all = w.compare(Some("own"));
    let g = row(&all, "good");
    assert!(g.flags.trained_here);
    assert!(g.flags.trained_on_share.unwrap() > 0.0);
}

#[test]
fn the_reference_model_does_not_score_itself() {
    let mut w = pair();
    w.results.image_sets[0].ground_truth = GroundTruth::Pseudo("good on openvino:cpu".into());
    let v = w.compare(None);
    let s = v.set.as_ref().unwrap();
    assert!(s.relative);
    assert_eq!(v.rows.len(), 1);
    assert!(row(&v, "poor").flags.relative);
    assert_eq!(v.missing[0].model, "good");
    assert!(
        v.missing[0].why.contains("its own labels"),
        "{}",
        v.missing[0].why
    );
    assert!(v.missing[0].blocked.is_some(), "no button that cannot help");
}

#[test]
fn an_unreachable_folder_blocks_the_button() {
    let mut w = pair();
    w.preds.runs.remove(1);
    let inputs = Inputs {
        reachable: &|_| false,
        ..w.inputs()
    };
    let v = compare_on(&inputs, None);
    assert!(
        v.missing[0]
            .blocked
            .as_deref()
            .unwrap()
            .contains("cannot be read")
    );
}

#[test]
fn no_results_no_sets() {
    let w = World::new(results(), Vec::new());
    let v = w.compare(None);
    assert!(v.sets.is_empty() && v.set.is_none() && v.rows.is_empty());
    assert!(
        v.headline.starts_with("Nothing to compare"),
        "{}",
        v.headline
    );
}

/// The JSON: no value is null, never 0; every row has its 1-based place (the table's
/// `data-best`).
#[test]
fn the_contract_of_a_row_is_null_not_zero() {
    let v = pair().compare(None);
    let j = serde_json::to_value(&v).unwrap();
    let r0 = &j["rows"][0];
    assert_eq!(r0["isDefault"], false);
    assert!(r0["precision"].is_number());
    let night = r0["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["scene"] == "night");
    assert!(night.is_none_or(|n| n["recall"].is_null()), "{night:?}");
    let mut places: Vec<usize> = v.rows.iter().map(|r| r.position).collect();
    places.sort_unstable();
    assert_eq!(places, (1..=v.rows.len()).collect::<Vec<_>>());
}

/// "Your cameras": each trained model ran without the frames it trained or validated on, so
/// Compare scores every listed model on the frames none of them saw, says how many were left out
/// and by whom, and ranks them all (no trained share there). A model whose lists do not read is
/// shown, never ranked among the fair rows.
#[test]
fn your_cameras_compares_on_the_frames_no_listed_model_saw() {
    use crate::benchmark::cameras::{ID, seen::Seen};
    let names: Vec<String> = (0..6)
        .map(|i| format!("day.cam.2026-11-0{}.f{i}.jpg", 1 + i % 2))
        .collect();
    let id_of = |f: &str| crate::training::autolabel::frame_of_file(f).unwrap();
    let images: Vec<ImageInfo> = names
        .iter()
        .map(|f| img(f, &["day"], &["person"]))
        .collect();
    let preds: Vec<(&str, Vec<Pred>)> = names
        .iter()
        .map(|f| (f.as_str(), vec![(0, 0.9, BOX)]))
        .collect();
    let ms: Vec<(&str, f64)> = names.iter().map(|f| (f.as_str(), 10.0)).collect();
    let mut r = results();
    r.image_sets = vec![set(ID, SetKind::Dir, images)];
    r.models = vec![
        model(
            "plain",
            &[ID],
            vec![timed(dev("ort:coreml", 1.0, Vec::new()), ID, &ms)],
        ),
        // Its run left out frames 0 (trained) and 1 (validated).
        model(
            "tuned",
            &[ID],
            vec![timed(dev("ort:coreml", 1.0, Vec::new()), ID, &ms[2..])],
        ),
    ];
    // A trained share recorded with the run never counts here: the left-out frames decide.
    r.models[1].trained_on_share = Some(0.33);
    let runs = vec![
        stored("plain", "ort:coreml", &["person"], ID, preds.clone()),
        stored("tuned", "ort:coreml", &["person"], ID, preds[2..].to_vec()),
    ];
    let mut w = World::new(r, runs);
    w.seen.insert(
        "tuned".into(),
        Seen::Frames {
            trained: [id_of(&names[0])].into(),
            validated: [id_of(&names[1])].into(),
        },
    );
    let v = w.compare(None);
    let s = v.set.as_ref().unwrap();
    assert_eq!(s.id, ID, "the headline set");
    assert_eq!((s.frames, s.frames_left_out), (4, 2));
    assert_eq!(s.left_out_by, ["tuned"]);
    for m in ["plain", "tuned"] {
        let r = row(&v, m);
        assert!(
            !r.flags.trained_here,
            "{m}: the left-out frames make it fair"
        );
        assert_eq!(r.block, Block::Ok, "{m}: ranked");
        assert_eq!(r.speed.frames, 4, "{m}");
    }

    // Without knowing what it saw, its run lacks two frames: it is not on the set.
    let mut blind = World::new(w.results.clone(), w.preds.runs.clone());
    blind.seen.clear();
    let v = blind.compare(None);
    let s = v.set.as_ref().unwrap();
    assert_eq!((s.frames, s.frames_left_out), (6, 0));
    let missing = v.missing.iter().find(|m| m.model == "tuned").unwrap();
    assert!(missing.why.contains("4 of these 6"), "{}", missing.why);

    // Lists that do not read: scored on everything it ran, never ranked fairly.
    w.seen.insert(
        "plain".into(),
        Seen::Unknown("its lists do not read".into()),
    );
    let v = w.compare(None);
    let p = row(&v, "plain");
    assert!(p.flags.trained_here && p.flags.unknown_seen.is_some());
    assert_eq!(p.block, Block::Unfair);
    assert_eq!(row(&v, "tuned").block, Block::Ok);
}
