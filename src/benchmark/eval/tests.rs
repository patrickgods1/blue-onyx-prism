//! The Compare and Tune views end to end on synthetic runs: ranks and rank groups, the budget,
//! targets on their bound, low confidence, flat curves, determinism and the answers' shapes.

use super::compare::{self, Request};
use super::goal::{Goal, Optimize};
use super::rank::Block;
use super::tune;
use crate::benchmark::combos::{Configured, Inputs};
use crate::benchmark::current::ModelNow;
use crate::benchmark::grade::Weights;
use crate::benchmark::images::{GroundTruth, SetKind};
use crate::benchmark::metrics::{GtBox, Window};
use crate::benchmark::report::test_support::{dev, model as base_model};
use crate::benchmark::report::{BenchmarkResults, DeviceResult, ImageInfo, ImageSetInfo};
use crate::benchmark::search::{StoredImage, StoredPred, StoredPreds, StoredRun};
use crate::benchmark::threshold::Objective;
use crate::benchmark::{ImageRun, ModelResult};
use std::collections::{BTreeSet, HashMap};

const BOX: [f32; 4] = [10.0, 10.0, 110.0, 110.0];
const OTHER: [f32; 4] = [300.0, 300.0, 400.0, 400.0];

/// (class index, confidence, box, window).
type Pred = (u32, f32, [f32; 4], Option<Window>);

/// A frame: its file, group, scene tag, whether it has a person, and each model's predictions.
struct Frame {
    file: String,
    group: String,
    scene: &'static str,
    person: bool,
}

/// `groups` camera days (cameras a and b), `per` frames each: every frame but each fifth has a
/// person; day frames, and night ones on odd days.
fn frames(groups: usize, per: usize) -> Vec<Frame> {
    let mut v = Vec::new();
    for g in 0..groups {
        let cam = if g % 2 == 0 { "a" } else { "b" };
        let day = g / 2 + 1;
        for i in 0..per {
            v.push(Frame {
                file: format!("{cam}_{day}_{i}.jpg"),
                group: format!("{cam}/2026-11-{day:02}"),
                scene: if day % 2 == 1 && i % 2 == 1 {
                    "night"
                } else {
                    "day"
                },
                person: i % 5 != 4,
            });
        }
    }
    v
}

fn info(f: &Frame) -> ImageInfo {
    ImageInfo {
        file: f.file.clone(),
        width: 640,
        height: 640,
        tags: vec![
            f.scene.to_string(),
            format!("camera:{}", &f.group[..1]),
            format!("date:{}", &f.group[2..]),
        ],
        objects: Some(if f.person {
            vec![GtBox {
                label: "person".into(),
                bbox: BOX,
                ignore: false,
                crowd: false,
            }]
        } else {
            Vec::new()
        }),
        scene: None,
        group: Some(f.group.clone()),
    }
}

fn stored(name: &str, device: &str, preds: Vec<(&Frame, Vec<Pred>)>) -> StoredRun {
    StoredRun {
        model: name.into(),
        device: device.into(),
        timestamp: String::new(),
        classes: vec!["person".into(), "car".into()],
        configured: 0.5,
        eval_confidence: 0.05,
        images: preds
            .into_iter()
            .map(|(f, p)| StoredImage {
                set: "own".into(),
                file: f.file.clone(),
                preds: p
                    .into_iter()
                    .map(|(class, confidence, bbox, window)| StoredPred {
                        class,
                        confidence,
                        bbox,
                        window,
                    })
                    .collect(),
                group: Some(f.group.clone()),
            })
            .collect(),
    }
}

fn timed(mut d: DeviceResult, frames: &[Frame], ms: f64) -> DeviceResult {
    if let Some(r) = d.run.as_mut() {
        r.per_image = frames
            .iter()
            .map(|f| ImageRun {
                set: "own".into(),
                file: f.file.clone(),
                total_ms: ms,
                infer_ms: ms,
                preds: Vec::new(),
                gt_matched: Vec::new(),
                counts: None,
                scene: None,
            })
            .collect();
        r.live_requests_per_min = Some(12.0);
    }
    d
}

fn model(name: &str, devices: Vec<DeviceResult>) -> ModelResult {
    let mut m = base_model(devices, None);
    m.model = name.into();
    m.datasets = vec!["own".into()];
    m
}

/// "good" finds every person at 0.9 (one false alert at 0.3 per camera day), on ort:coreml at
/// 40 ms and on openvino:cpu at 400 ms; "poor" finds every other person at 0.6 and alerts at 0.7
/// on most empty frames (ort:coreml, 30 ms).
pub(crate) fn fixture(groups: usize, per: usize) -> (BenchmarkResults, StoredPreds) {
    let fs = frames(groups, per);
    let good: Vec<(&Frame, Vec<Pred>)> = fs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut p = Vec::new();
            if f.person {
                p.push((0, 0.9, BOX, None));
            } else if i % per == 4 {
                p.push((0, 0.3, OTHER, None));
            }
            (f, p)
        })
        .collect();
    let poor: Vec<(&Frame, Vec<Pred>)> = fs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut p = Vec::new();
            if f.person && i % 2 == 0 {
                p.push((0, 0.6, BOX, None));
            }
            if !f.person && i % 3 != 0 {
                p.push((0, 0.7, OTHER, None));
            }
            (f, p)
        })
        .collect();
    let mut r: BenchmarkResults = serde_json::from_str("{}").unwrap();
    r.image_sets = vec![ImageSetInfo {
        id: "own".into(),
        title: "Own frames".into(),
        kind: SetKind::Dir,
        dir: None,
        license: String::new(),
        attribution: String::new(),
        source: String::new(),
        ground_truth: GroundTruth::Yolo,
        scored_labels: None,
        images: fs.iter().map(info).collect(),
    }];
    r.models = vec![
        model(
            "ipcam-general",
            vec![
                timed(dev("ort:coreml", 40.0, Vec::new()), &fs, 40.0),
                timed(dev("openvino:cpu", 400.0, Vec::new()), &fs, 400.0),
            ],
        ),
        model(
            "dfine-s",
            vec![timed(dev("ort:coreml", 30.0, Vec::new()), &fs, 30.0)],
        ),
    ];
    let runs = vec![
        stored("ipcam-general", "ort:coreml", good.clone()),
        stored("ipcam-general", "openvino:cpu", good),
        stored("dfine-s", "ort:coreml", poor),
    ];
    (
        r,
        StoredPreds {
            version: String::new(),
            runs,
        },
    )
}

struct World {
    results: BenchmarkResults,
    preds: StoredPreds,
    now: HashMap<String, ModelNow>,
    configured: Vec<Configured>,
}

impl World {
    fn new(groups: usize, per: usize) -> Self {
        let (results, preds) = fixture(groups, per);
        let configured = results
            .models
            .iter()
            .map(|m| Configured {
                name: m.model.clone(),
                blocked: None,
            })
            .collect();
        let mut now = HashMap::new();
        for m in ["ipcam-general", "dfine-s"] {
            now.insert(
                m.to_string(),
                ModelNow::new(0.5, None, Some("ort:coreml".into())),
            );
        }
        Self {
            results,
            preds,
            now,
            configured,
        }
    }

    fn compare(&self, goal: &Goal, scene: Option<&str>) -> compare::CompareView {
        let trained: HashMap<String, BTreeSet<String>> = HashMap::new();
        let seen = HashMap::new();
        compare::build(
            &Inputs {
                results: &self.results,
                preds: Some(&self.preds),
                now: &self.now,
                default_model: Some("ipcam-general"),
                configured: &self.configured,
                trained: &trained,
                seen: &seen,
                reachable: &|_| true,
            },
            Request {
                goal,
                weights: Weights::default(),
                set: None,
                scene,
            },
        )
    }
}

fn row<'a>(v: &'a compare::CompareView, model: &str, device: &str) -> &'a compare::Row {
    v.rows
        .iter()
        .find(|r| r.model == model && r.device == device)
        .unwrap()
}

#[test]
fn compare_ranks_groups_budget_and_headline() {
    let w = World::new(10, 10);
    let goal = Goal::default();
    let v = w.compare(&goal, None);
    assert_eq!(v.rows.len(), 3);
    let good = row(&v, "ipcam-general", "ort:coreml");
    let cpu = row(&v, "ipcam-general", "openvino:cpu");
    let poor = row(&v, "dfine-s", "ort:coreml");
    // The same predictions on two devices are not clearly different; the poor model is.
    assert_eq!((good.rank, cpu.rank), (1, 1));
    assert_eq!(poor.rank, 3);
    assert!(poor.vs_leader.unwrap().clearly_below());
    let m = good.metric.unwrap();
    assert!(
        m.low.unwrap() <= m.value && m.value <= m.high.unwrap(),
        "{m:?}"
    );
    assert!(good.flags.low_confidence.is_none());
    assert_eq!(good.folds, 5);
    assert!(v.headline.contains("has the best F1"), "{}", v.headline);
    assert!(
        v.headline.contains("not clearly different"),
        "{}",
        v.headline
    );
    // A budget makes the CPU combo too slow: ranked after, its own group.
    let budget = Goal {
        speed_budget_ms: Some(200.0),
        ..Goal::default()
    };
    let v = w.compare(&budget, None);
    let cpu = row(&v, "ipcam-general", "openvino:cpu");
    assert!(cpu.flags.too_slow);
    assert_eq!(cpu.block, Block::TooSlow);
    assert_eq!(v.rows.last().unwrap().device, "openvino:cpu");
    assert!(v.headline.contains("within budget"), "{}", v.headline);
    // Breakdowns.
    let good = row(&v, "ipcam-general", "ort:coreml");
    let scenes: Vec<&str> = good.scenes.iter().map(|s| s.scene.as_str()).collect();
    assert_eq!(scenes, ["day", "night"]);
    assert_eq!(good.classes[0].class, "person");
    assert_eq!(good.classes[0].objects, 80);
    assert!(good.cameras.is_empty(), "cameras only on Your cameras");
    assert_eq!(good.flags.live_load, Some(12.0));
    assert_eq!(good.speed.p95_ms, Some(40.0));
    assert_eq!(good.served_threshold, Some(0.5));
}

#[test]
fn compare_is_deterministic_and_camel_case() {
    let w = World::new(6, 10);
    let goal = Goal::default();
    let a = serde_json::to_value(w.compare(&goal, None)).unwrap();
    let mut b = serde_json::to_value(w.compare(&goal, None)).unwrap();
    b["elapsedMs"] = a["elapsedMs"].clone();
    assert_eq!(a, b, "the same inputs give the same answer");
    let r = &a["rows"][0];
    for k in [
        "model",
        "device",
        "metric",
        "rank",
        "position",
        "block",
        "recall",
        "precision",
        "fa100",
        "confidence",
        "pickInformative",
        "servedThreshold",
        "speed",
        "agreement",
        "flags",
        "scenes",
        "classes",
        "cameras",
    ] {
        assert!(r.get(k).is_some(), "row has no {k}: {r}");
    }
    for k in ["value", "low", "high"] {
        assert!(r["metric"].get(k).is_some(), "{k}");
    }
    for k in [
        "tooSlow",
        "unmet",
        "trainedHere",
        "relative",
        "served",
        "staleRules",
        "lowConfidence",
        "liveLoad",
        "videoDecode",
    ] {
        assert!(r["flags"].get(k).is_some(), "flags has no {k}");
    }
    for k in ["p50Ms", "p95Ms", "p99Ms", "burstP95Ms", "firstLoad"] {
        assert!(r["speed"].get(k).is_some(), "speed has no {k}");
    }
    for k in [
        "framesLeftOut",
        "classes",
        "ignoredClasses",
        "groups",
        "run",
    ] {
        assert!(a["set"].get(k).is_some(), "set has no {k}");
    }
    assert!(a["goal"]["metric"] == "F1");
    assert!(a["missing"].is_array() && a["sets"].is_array());
}

#[test]
fn a_target_unmet_on_its_bound_ranks_after_and_rank_only_goals_have_intervals() {
    let w = World::new(10, 10);
    let goal = Goal {
        optimize_for: Optimize::Threshold(Objective::Precision(0.95)),
        ..Goal::default()
    };
    let v = w.compare(&goal, None);
    let poor = row(&v, "dfine-s", "ort:coreml");
    assert!(poor.flags.unmet, "{:?}", poor.target);
    assert_eq!(poor.block, Block::Unmet);
    assert_eq!(v.rows.last().unwrap().model, "dfine-s");
    for g in [
        Optimize::Ap50,
        Optimize::RocAuc,
        Optimize::Grade,
        Optimize::Ap5095,
    ] {
        let goal = Goal {
            optimize_for: g,
            ..Goal::default()
        };
        let v = w.compare(&goal, None);
        let good = row(&v, "ipcam-general", "ort:coreml");
        let m = good.metric.unwrap_or_else(|| panic!("{g}: no metric"));
        assert!(m.low.is_some(), "{g}: no interval");
        assert_eq!(v.rows[0].model, "ipcam-general", "{g}");
        assert!(good.confidence.is_some(), "{g}: tuned for f1");
    }
    // Classes the goal names but no combo scores are listed.
    let goal = Goal {
        classes: vec!["person".into(), "giraffe".into()],
        ..Goal::default()
    };
    let v = w.compare(&goal, None);
    let set = v.set.unwrap();
    assert_eq!(set.classes, ["person"]);
    assert_eq!(set.ignored_classes, ["giraffe"]);
}

#[test]
fn too_little_data_is_low_confidence_and_a_scene_filters() {
    let w = World::new(3, 10);
    let v = w.compare(&Goal::default(), None);
    let good = row(&v, "ipcam-general", "ort:coreml");
    let why = good.flags.low_confidence.as_deref().unwrap();
    assert!(why.contains("3 group"), "{why}");
    assert_eq!(good.folds, 0);
    assert!(
        good.metric.unwrap().low.is_some(),
        "intervals still computed"
    );
    let night = w.compare(&Goal::default(), Some("night"));
    let s = night.set.unwrap();
    assert_eq!(s.scene.as_deref(), Some("night"));
    assert!(s.frames < 30);
}

#[test]
fn tune_lists_the_grid_the_pick_and_the_examples() {
    let w = World::new(10, 10);
    let goal = Goal::default();
    let t = tune::build(
        &w.results,
        &w.preds,
        tune::Request {
            model: "dfine-s",
            device: None,
            set: None,
            goal: &goal,
            weights: Weights::default(),
            saved: Some((0.5, "model")),
        },
    )
    .unwrap();
    assert_eq!(t.grid.len(), 96);
    assert_eq!(t.grid[0].threshold, 0.05);
    let s = t.suggested.as_ref().unwrap();
    assert!(s.plateau[0] <= s.threshold && s.threshold <= s.plateau[1]);
    assert!(s.metric.unwrap().low.is_some());
    assert!(t.savable && t.not_savable.is_none());
    assert_eq!(t.saved.as_ref().unwrap().source, "model");
    assert_eq!(t.frames, 100);
    assert_eq!(t.folds, 5);
    let j = serde_json::to_value(&t).unwrap();
    for k in [
        "grid",
        "suggested",
        "lowConfidence",
        "pickInformative",
        "savable",
        "notSavable",
    ] {
        assert!(j.get(k).is_some(), "{k}");
    }
    assert!(j["grid"][0].get("fn").is_some() && j["grid"][0].get("fa100").is_some());
    // A precision target it cannot reach is reported and not savable.
    let strict = Goal {
        optimize_for: Optimize::Threshold(Objective::Precision(0.99)),
        ..Goal::default()
    };
    let t = tune::build(
        &w.results,
        &w.preds,
        tune::Request {
            model: "dfine-s",
            device: Some("ort:coreml"),
            set: Some("own"),
            goal: &strict,
            weights: Weights::default(),
            saved: None,
        },
    )
    .unwrap();
    let tg = t.target.as_ref().unwrap();
    assert!(!tg.met && tg.text.starts_with("not met"), "{}", tg.text);
    assert!(!t.savable);
    // Misses, most missed first; false alerts, most confident first.
    let req = |kind, offset, limit| tune::ImagesRequest {
        model: "dfine-s",
        device: None,
        set: None,
        goal: &goal,
        threshold: 0.5,
        kind,
        camera: None,
        scene: None,
        offset,
        limit,
    };
    let ex = tune::images(&w.results, &w.preds, req(tune::Kind::Miss, 0, 5)).unwrap();
    assert_eq!(ex.frames.len(), 5);
    assert!(ex.total >= 5);
    assert!(ex.frames.iter().all(|f| f.missed >= 1));
    assert_eq!(ex.frames[0].camera.as_deref(), Some("a"));
    let fa = tune::images(&w.results, &w.preds, req(tune::Kind::Fa, 0, 200)).unwrap();
    assert!(fa.frames.iter().all(|f| f.top_false == Some(0.7)));
    // Every frame, the most errors first, in pages; one camera; the boxes' space.
    let all = tune::images(&w.results, &w.preds, req(tune::Kind::All, 0, 200)).unwrap();
    assert!(all.total >= ex.total.max(fa.total), "every frame");
    assert_eq!(all.frames.len(), all.total);
    assert!(
        all.frames
            .windows(2)
            .all(|p| p[0].missed + p[0].false_positives >= p[1].missed + p[1].false_positives)
    );
    assert!(all.frames.iter().all(|f| f.width > 0 && f.height > 0));
    let page = tune::images(&w.results, &w.preds, req(tune::Kind::All, 2, 3)).unwrap();
    assert_eq!((page.total, page.offset), (all.total, 2));
    assert_eq!(
        page.frames.iter().map(|f| &f.file).collect::<Vec<_>>(),
        all.frames[2..5].iter().map(|f| &f.file).collect::<Vec<_>>()
    );
    assert!(!all.cameras.is_empty());
    let cam = all.cameras[0].clone();
    let one = tune::images(
        &w.results,
        &w.preds,
        tune::ImagesRequest {
            camera: Some(&cam),
            ..req(tune::Kind::All, 0, 200)
        },
    )
    .unwrap();
    assert!(one.total > 0 && one.total <= all.total);
    assert!(
        one.frames
            .iter()
            .all(|f| f.camera.as_deref() == Some(cam.as_str()))
    );
    let none = tune::images(
        &w.results,
        &w.preds,
        tune::ImagesRequest {
            camera: Some("no such camera"),
            ..req(tune::Kind::All, 0, 200)
        },
    )
    .unwrap();
    assert_eq!(none.total, 0);
    assert!(
        tune::build(
            &w.results,
            &w.preds,
            tune::Request {
                model: "nobody",
                device: None,
                set: None,
                goal: &goal,
                weights: Weights::default(),
                saved: None,
            },
        )
        .is_err()
    );
}

#[test]
fn a_flat_curve_is_not_informative_and_not_savable() {
    let mut w = World::new(6, 10);
    // Every prediction returned at every threshold (as a night model's answers are).
    for run in &mut w.preds.runs {
        for img in &mut run.images {
            for p in &mut img.preds {
                p.window = Some(Window::ALWAYS);
            }
        }
    }
    let goal = Goal::default();
    let t = tune::build(
        &w.results,
        &w.preds,
        tune::Request {
            model: "ipcam-general",
            device: Some("ort:coreml"),
            set: None,
            goal: &goal,
            weights: Weights::default(),
            saved: None,
        },
    )
    .unwrap();
    assert!(!t.pick_informative);
    assert!(t.flat_note.is_some());
    assert!(!t.savable);
    let v = w.compare(&goal, None);
    assert!(v.rows.iter().all(|r| r.flags.flat));
    assert!(v.notes.iter().any(|n| n.contains("says nothing")));
}

#[test]
fn the_run_snapshot_carries_the_goal_metric() {
    let w = World::new(6, 10);
    let run = &w.preds.runs[0];
    let s = tune::run_snapshot(
        &w.results.image_sets,
        run,
        Some(40.0),
        &Goal::default(),
        Weights::default(),
    )
    .unwrap();
    assert_eq!(s.optimize_for, "f1");
    assert_eq!(s.frames, 60);
    assert!(s.low.is_some() && s.high.is_some());
    // 48 labelled persons: under the 50 objects of the low-confidence rule.
    assert!(s.low_confidence);
    let w = World::new(10, 10);
    let s = tune::run_snapshot(
        &w.results.image_sets,
        &w.preds.runs[0],
        Some(40.0),
        &Goal::default(),
        Weights::default(),
    )
    .unwrap();
    assert!(!s.low_confidence);
    assert_eq!(s.objects, 80);
}

/// Measures the views on saved results (not a check): `BOP_EVAL_DIR=<folder with benchmark.json
/// and benchmark-preds.json> [BOP_EVAL_SCALE=<copies>] cargo test --lib eval_on_saved_results --
/// --ignored --nocapture`. A scale above 1 repeats every frame that many times (new file names,
/// each copy of a camera day one of three groups) to time a larger set with the same prediction
/// density.
#[test]
#[ignore = "measures saved results named by BOP_EVAL_DIR"]
fn eval_on_saved_results() {
    let Ok(dir) = std::env::var("BOP_EVAL_DIR") else {
        eprintln!("BOP_EVAL_DIR is not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let scale: usize = std::env::var("BOP_EVAL_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let mut results = BenchmarkResults::load(&dir.join("benchmark.json"))
        .unwrap()
        .unwrap();
    let mut preds = StoredPreds::load(&dir.join("benchmark-preds.json"))
        .unwrap()
        .unwrap();
    if scale > 1 {
        let copy = |c: usize, f: &str| format!("x{c}_{f}");
        let regroup = |c: usize, g: &Option<String>| g.as_ref().map(|g| format!("{g}~{}", c % 3));
        for s in &mut results.image_sets {
            s.images = (0..scale)
                .flat_map(|c| {
                    s.images.iter().map(move |i| ImageInfo {
                        file: copy(c, &i.file),
                        group: regroup(c, &i.group),
                        ..i.clone()
                    })
                })
                .collect();
        }
        for m in &mut results.models {
            for d in &mut m.devices {
                if let Some(r) = d.run.as_mut() {
                    r.per_image = (0..scale)
                        .flat_map(|c| {
                            r.per_image.iter().map(move |i| ImageRun {
                                file: copy(c, &i.file),
                                ..i.clone()
                            })
                        })
                        .collect();
                }
            }
        }
        for run in &mut preds.runs {
            run.images = (0..scale)
                .flat_map(|c| {
                    run.images.iter().map(move |i| StoredImage {
                        file: copy(c, &i.file),
                        group: regroup(c, &i.group),
                        ..i.clone()
                    })
                })
                .collect();
        }
    }
    let configured: Vec<Configured> = results
        .models
        .iter()
        .map(|m| Configured {
            name: m.model.clone(),
            blocked: None,
        })
        .collect();
    let now = HashMap::new();
    let (trained, seen) = (HashMap::new(), HashMap::new());
    let inputs = Inputs {
        results: &results,
        preds: Some(&preds),
        now: &now,
        default_model: None,
        configured: &configured,
        trained: &trained,
        seen: &seen,
        reachable: &|_| true,
    };
    for goal in [
        "f1",
        "f2",
        "precision:0.9",
        "recall_at_fa:5",
        "ap50",
        "roc_auc",
    ] {
        let g = Goal {
            optimize_for: goal.parse().unwrap(),
            ..Goal::default()
        };
        let v = compare::build(
            &inputs,
            Request {
                goal: &g,
                weights: Weights::default(),
                set: None,
                scene: None,
            },
        );
        let set = v.set.as_ref().unwrap();
        println!(
            "compare goal {goal}: {} frames, {} groups, {} objects, {} combos in {:.0} ms",
            set.frames,
            set.groups,
            set.objects,
            v.rows.len(),
            v.elapsed_ms
        );
        for r in &v.rows {
            let m = r.metric.unwrap();
            println!(
                "  #{} {} on {}: {:.3} [{:.3}, {:.3}] conf {:?} P {:?} R {:?} FA/100 {:?} {:?}{}{}",
                r.rank,
                r.model,
                r.device,
                m.value,
                m.low.unwrap_or(f64::NAN),
                m.high.unwrap_or(f64::NAN),
                r.confidence,
                r.precision.map(|x| (x * 1000.0).round() / 1000.0),
                r.recall.map(|x| (x * 1000.0).round() / 1000.0),
                r.fa100.map(|x| (x * 10.0).round() / 10.0),
                r.block,
                r.target
                    .as_ref()
                    .map(|t| format!(" target met {} bound {:?}", t.met, t.bound))
                    .unwrap_or_default(),
                r.flags
                    .low_confidence
                    .as_ref()
                    .map(|l| format!(" [{l}]"))
                    .unwrap_or_default()
            );
        }
        println!("  {}", v.headline);
    }
    for run in &preds.runs {
        let t = tune::build(
            &results,
            &preds,
            tune::Request {
                model: &run.model,
                device: Some(&run.device),
                set: None,
                goal: &Goal::default(),
                weights: Weights::default(),
                saved: None,
            },
        )
        .unwrap();
        let s = t.suggested.as_ref().unwrap();
        println!(
            "tune {} on {}: pick {} plateau {:?} in-sample F1 {:?} OOF {:?} folds {} in {:.0} ms",
            t.model, t.device, s.threshold, s.plateau, s.in_sample, s.metric, t.folds, t.elapsed_ms
        );
        let started = std::time::Instant::now();
        let goal = Goal::default();
        let ex = tune::images(
            &results,
            &preds,
            tune::ImagesRequest {
                model: &run.model,
                device: Some(&run.device),
                set: None,
                goal: &goal,
                threshold: s.threshold,
                kind: tune::Kind::Miss,
                camera: None,
                scene: None,
                offset: 0,
                limit: 24,
            },
        )
        .unwrap();
        println!(
            "  images: {} miss frames, top {} in {:.0} ms",
            ex.total,
            ex.frames.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
    }
}
