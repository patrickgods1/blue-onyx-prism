//! Tests of the quick check: which frames a model saw (a fixture run folder), the refusals, a
//! model with a class subset, the preview, and the whole job over a fake predictor (only the frames
//! it never saw are written, nothing else changes). No GPU, Python or network.

use super::*;
use crate::collect::labels::bulk::test_support::put_frame_at;
use crate::collect::labels::refine::{Answer, Kind, Suggestion};
use crate::collect::labels::{self, BoxState, Frame, LabelBox, LabelFile, Source, Status};
use crate::model::ModelFamilyKind;
use crate::resources::catalog::TrainFamily;
use crate::training::autolabel::relabel::{Detections, Image};
use crate::training::autolabel::triage::Det;
use crate::training::refine::predict::{FoldModel, FoldPredictor};
use crate::training::refine::{RefineJob, plan_refine, preview_text};
use crate::training::task::{Task, TaskContext, TaskKind};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// The model's training run.
pub(crate) const RUN: &str = "20261111-020504-cad6";
/// A labelled car on every frame (320 x 240).
pub(crate) const CAR: [f32; 4] = [20.0, 20.0, 120.0, 100.0];
/// Where the fake model finds an unlabelled person.
pub(crate) const FAR: [f32; 4] = [200.0, 60.0, 260.0, 200.0];

/// A fixture: a data root with a dataset folder of reviewed frames, a model `mine` in `models/`
/// with its training record, and its run folder under `training/`. Removed on drop.
pub(crate) struct Fixture {
    pub root: PathBuf,
    pub dataset: PathBuf,
    /// Frames by role: `trained` (alpha 1, 2), `validated` (bravo 1), `heldout` (charlie 1),
    /// `new` (delta 1, 2), `same_day` (alpha 3: new, of a camera day it trained on).
    pub rels: BTreeMap<&'static str, Vec<String>>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn accepted(label: &str, bbox: [f32; 4]) -> LabelBox {
    LabelBox {
        label: label.into(),
        bbox,
        source: Source::Auto,
        score: Some(0.8),
        state: BoxState::Accepted,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: Default::default(),
    }
}

/// The flat name of `rel` (`ir/cam/2026-10-21/stem.jpg`).
fn flat(rel: &str) -> String {
    export::flat_name(&crate::collect::layout::parse_rel(rel).unwrap())
}

fn path_of(dir: &Path, rel: &str) -> PathBuf {
    Frame::resolve(dir, rel).unwrap().labels
}

fn edit(dir: &Path, rel: &str, f: impl FnOnce(&mut LabelFile)) {
    let p = path_of(dir, rel);
    let mut l = labels::read(&p).unwrap().unwrap();
    f(&mut l);
    labels::write(&p, &l).unwrap();
}

/// A reviewed frame of `camera` with an accepted car.
fn reviewed(dir: &Path, camera: &str, stem: &str) -> String {
    let rel = put_frame_at(
        dir,
        camera,
        stem,
        (320, 240),
        None,
        vec![accepted("car", CAR)],
    );
    edit(dir, &rel, |l| {
        l.status = Status::Reviewed;
        l.reviewed_at = Some("2026-10-22T00:00:00Z".into());
    });
    rel
}

/// The model's config entry, as a user would have it: a threshold and an object filter of its own,
/// a device.
pub(crate) fn entry() -> ModelConfig {
    ModelConfig {
        name: Some("mine".into()),
        path: PathBuf::from("models").join("mine.onnx"),
        family: ModelFamilyKind::RfDetr,
        classes: Some(PathBuf::from("models").join("mine.yaml")),
        device: Some("openvino:cpu".into()),
        confidence_threshold: Some(0.6),
        object_filter: Some(vec!["car".into()]),
        enabled: true,
        ..ModelConfig::default()
    }
}

/// The training record of `mine` naming `run` with `classes`.
fn write_record(root: &Path, run: &str, classes: &[&str]) {
    let models = root.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("mine.onnx"), b"onnx").unwrap();
    let names: String = classes.iter().map(|c| format!("  - {c}\n")).collect();
    std::fs::write(models.join("mine.yaml"), format!("NAMES:\n{names}")).unwrap();
    let mut recipe = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
    recipe.imgsz = 384;
    recipe.epochs = 100;
    recipe.batch = 8;
    std::fs::write(
        models.join("mine.provenance.json"),
        serde_json::json!({"version": 1, "name": "mine", "created": "2026-11-11T04:09:55Z",
            "run_id": run, "recipe": recipe, "classes": classes})
        .to_string(),
    )
    .unwrap();
}

/// The run folder's `data/`: `train` listed twice (balancing repeats), `valid` once, `heldout`
/// in `data/heldout/images`, the training box counts, and with `sources` the record of where each
/// image came from (an export's `images/<split>/<flat name>`).
fn write_run(
    run_dir: &Path,
    train: &[String],
    valid: &[String],
    heldout: &[String],
    sources: bool,
) {
    let data = run_dir.join("data");
    std::fs::create_dir_all(data.join("heldout").join("images")).unwrap();
    let line = |rel: &String| format!("./images/s0-{}\n", flat(rel));
    let t: String = train.iter().flat_map(|r| [line(r), line(r)]).collect();
    std::fs::write(data.join("train.txt"), t).unwrap();
    let v: String = valid.iter().map(line).collect();
    std::fs::write(data.join("valid.txt"), v).unwrap();
    for h in heldout {
        std::fs::write(
            data.join("heldout")
                .join("images")
                .join(format!("s0-{}", flat(h))),
            b"x",
        )
        .unwrap();
    }
    std::fs::write(
        data.join("dataset.json"),
        serde_json::json!({"stats": {"train_boxes": {"car": 120, "person": 60, "motorbike": 3}}})
            .to_string(),
    )
    .unwrap();
    std::fs::write(data.join("classes.txt"), "person\ncar\nmotorbike\n").unwrap();
    if sources {
        let export = run_dir.join("export");
        let mut m: BTreeMap<String, PathBuf> = BTreeMap::new();
        for (split, list) in [("train", train), ("val", valid)] {
            for r in list {
                m.insert(
                    format!("s0-{}", flat(r)),
                    export.join("images").join(split).join(flat(r)),
                );
            }
        }
        std::fs::write(
            data.join(crate::training::dataset::IMAGE_SOURCES_FILE),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
    }
}

/// The fixture (see [`Fixture`]); `sources`: the run records its image sources.
pub(crate) fn fixture(tag: &str, sources: bool) -> Fixture {
    let root = std::env::temp_dir().join(format!("bop-quick-{tag}-{}", uuid::Uuid::new_v4()));
    let dataset = root.join("dataset");
    let mut rels: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    rels.insert(
        "trained",
        vec![
            reviewed(&dataset, "alpha", "a1"),
            reviewed(&dataset, "alpha", "a2"),
        ],
    );
    rels.insert("validated", vec![reviewed(&dataset, "bravo", "b1")]);
    rels.insert("heldout", vec![reviewed(&dataset, "charlie", "c1")]);
    rels.insert(
        "new",
        vec![
            reviewed(&dataset, "delta", "d1"),
            reviewed(&dataset, "delta", "d2"),
        ],
    );
    rels.insert("same_day", vec![reviewed(&dataset, "alpha", "a3")]);
    write_record(&root, RUN, &["person", "car", "motorbike"]);
    write_run(
        &root.join("training").join(RUN),
        &rels["trained"],
        &rels["validated"],
        &rels["heldout"],
        sources,
    );
    Fixture {
        root,
        dataset,
        rels,
    }
}

/// A config over the fixture: its data root, dataset and the model `mine`.
pub(crate) fn config(f: &Fixture) -> Config {
    let mut c = Config {
        download_dir: Some(f.root.clone()),
        allow_large_downloads: true,
        ..Config::default()
    };
    c.collect.dir = Some(f.dataset.clone());
    c.collect.classes = ["person", "car", "motorcycle", "dog"]
        .map(String::from)
        .to_vec();
    c.models = vec![entry()];
    c
}

fn host() -> Capability {
    crate::training::capability::host_for_tests()
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

#[test]
fn the_frames_it_trained_or_validated_on_are_skipped_and_held_out_and_new_ones_kept() {
    for sources in [false, true] {
        let f = fixture("seen", sources);
        let c = config(&f);
        let seen = seen_of(&c, &entry()).unwrap();
        assert_eq!(seen.by_source, sources);
        let l = seen.located.as_ref().unwrap();
        assert_eq!((l.run.as_str(), l.imgsz), (RUN, 384));
        assert_eq!(l.recipe, "rfdetr-nano");
        assert_eq!(l.run_dir, f.root.join("training").join(RUN));
        let qs = select(
            &f.dataset,
            &c.collect.classes,
            &[],
            &RefineFrames::default(),
            &seen,
        );
        assert!(qs.s.refused.is_none(), "{:?}", qs.s.refused);
        let checked: Vec<String> = qs.s.frames.iter().map(|x| x.rel.jpg()).collect();
        let mut want = f.rels["heldout"].clone();
        want.extend(f.rels["new"].iter().cloned());
        want.extend(f.rels["same_day"].iter().cloned());
        assert_eq!(sorted(checked), sorted(want), "sources {sources}");
        // Repeats in train.txt count once.
        assert_eq!((qs.trained_on, qs.validated_on), (2, 1));
        assert_eq!(qs.same_day, 1, "alpha 3: a day it trained on");
        assert_eq!(qs.s.groups.len(), 3, "charlie, delta, alpha");
        assert_eq!(qs.s.k, 0);
        assert!(qs.s.folds.is_empty());
        // The class counts are the run's, through the class list (motorbike = motorcycle).
        assert_eq!(
            qs.s.class_boxes,
            BTreeMap::from([
                ("car".to_string(), 120),
                ("motorcycle".to_string(), 3),
                ("person".to_string(), 60)
            ])
        );
        // A scene or camera filter narrows it like the full check's.
        let only = select(
            &f.dataset,
            &c.collect.classes,
            &[],
            &RefineFrames::parse("camera=delta").unwrap(),
            &seen,
        );
        assert_eq!(only.s.frames.len(), 2);
        assert_eq!((only.trained_on, only.validated_on), (0, 0));
        assert_eq!(only.s.left_out.filtered, 5);
    }
}

#[test]
fn the_strictest_reading_matches_a_cut_name_and_a_renamed_source() {
    let f = fixture("names", false);
    let c = config(&f);
    let data = f.root.join("training").join(RUN).join("data");
    // A frame whose flat name the dataset cut at 120 bytes, listed under its cut name (camera
    // and stem at their longest: 150 bytes with the prefix).
    let long = reviewed(&f.dataset, &"e".repeat(64), &"x".repeat(64));
    assert_eq!(flat(&long).len() + 3, 150);
    let cut = spelled("s0", &flat(&long));
    assert_eq!(cut.len(), 120);
    let mut t = std::fs::read_to_string(data.join("train.txt")).unwrap();
    t.push_str(&format!("./images/{cut}\n"));
    // A second source's frame (s1-), in the validation list.
    let other = reviewed(&f.dataset, "foxtrot", "f1");
    let mut v = std::fs::read_to_string(data.join("valid.txt")).unwrap();
    v.push_str(&format!("./images/s1-{}\n", flat(&other)));
    // A public image: names no frame, skipped nothing.
    t.push_str("./images/s2-000000123456.jpg\n");
    std::fs::write(data.join("train.txt"), &t).unwrap();
    std::fs::write(data.join("valid.txt"), &v).unwrap();
    let seen = seen_of(&c, &entry()).unwrap();
    let qs = select(
        &f.dataset,
        &c.collect.classes,
        &[],
        &RefineFrames::default(),
        &seen,
    );
    let checked: Vec<String> = qs.s.frames.iter().map(|x| x.rel.jpg()).collect();
    assert!(!checked.contains(&long), "the cut name is the frame");
    assert!(!checked.contains(&other), "a second source's prefix");
    assert_eq!((qs.trained_on, qs.validated_on), (3, 2));
    // With the record of sources: a listed image named otherwise (a `-2` for a name taken
    // twice) is matched by its source; one the record does not name refuses the check.
    let g = fixture("sources", true);
    let c = config(&g);
    let data = g.root.join("training").join(RUN).join("data");
    let new = &g.rels["new"][0];
    let mut t = std::fs::read_to_string(data.join("train.txt")).unwrap();
    t.push_str("./images/s0-ir.renamed-2.jpg\n");
    std::fs::write(data.join("train.txt"), &t).unwrap();
    let record = data.join(crate::training::dataset::IMAGE_SOURCES_FILE);
    let mut m: BTreeMap<String, PathBuf> =
        serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    m.insert(
        "s0-ir.renamed-2.jpg".into(),
        g.root
            .join("export")
            .join("images")
            .join("train")
            .join(flat(new)),
    );
    std::fs::write(&record, serde_json::to_vec(&m).unwrap()).unwrap();
    let seen = seen_of(&c, &entry()).unwrap();
    let frames = select(
        &g.dataset,
        &c.collect.classes,
        &[],
        &RefineFrames::default(),
        &seen,
    );
    assert!(frames.s.frames.iter().all(|x| x.rel.jpg() != *new));
    assert_eq!(frames.trained_on, 3);
    m.remove("s0-ir.renamed-2.jpg");
    std::fs::write(&record, serde_json::to_vec(&m).unwrap()).unwrap();
    let e = format!("{:#}", seen_of(&c, &entry()).unwrap_err());
    assert!(
        e.contains("s0-ir.renamed-2.jpg") && e.contains("cannot tell"),
        "{e}"
    );
}

#[test]
fn a_model_whose_frames_cannot_be_known_is_refused_with_why() {
    let f = fixture("refuse", false);
    let c = config(&f);
    let models = f.root.join("models");
    let run_dir = f.root.join("training").join(RUN);
    let refused = |c: &Config| format!("{:#}", seen_of(c, &entry()).unwrap_err());
    let common = "a model judging frames it trained on finds every label right";
    // A stock model: no record.
    let record = models.join("mine.provenance.json");
    let kept = std::fs::read(&record).unwrap();
    std::fs::remove_file(&record).unwrap();
    let e = refused(&c);
    assert!(
        e.contains("has no training record") && e.contains(common),
        "{e}"
    );
    assert!(e.contains("Run the full check"), "{e}");
    // A record that names no run, or does not read.
    std::fs::write(&record, r#"{"name":"mine"}"#).unwrap();
    assert!(
        refused(&c).contains("names no training run"),
        "{}",
        refused(&c)
    );
    std::fs::write(&record, r#"{"run_id":"../escape"}"#).unwrap();
    assert!(refused(&c).contains("names no training run"));
    std::fs::write(&record, "{broken").unwrap();
    assert!(refused(&c).contains("does not read"));
    std::fs::write(&record, &kept).unwrap();
    // The lists, one at a time.
    for list in ["valid.txt", "train.txt"] {
        let p = run_dir.join("data").join(list);
        let text = std::fs::read(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        let e = refused(&c);
        assert!(e.contains(list) && e.contains(common), "{e}");
        std::fs::write(&p, text).unwrap();
    }
    // The run folder elsewhere: found through runs.json's run_dir; gone: refused.
    let moved = f.root.join("elsewhere").join(RUN);
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&run_dir, &moved).unwrap();
    let e = refused(&c);
    assert!(e.contains("whose folder is gone") && e.contains(RUN), "{e}");
    let runs = crate::training::runs_file(&c);
    std::fs::write(
        &runs,
        serde_json::json!([{"id": RUN, "run_dir": moved}]).to_string(),
    )
    .unwrap();
    let before = std::fs::read(&runs).unwrap();
    let seen = seen_of(&c, &entry()).unwrap();
    assert_eq!(seen.located.unwrap().run_dir, moved);
    assert_eq!(std::fs::read(&runs).unwrap(), before, "read only");
    // The preview says it as a refusal (not an error), and a plan refuses.
    std::fs::remove_file(&runs).unwrap();
    let req = RefineRequest {
        model: Some("mine".into()),
        ..RefineRequest::default()
    };
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(
        p.refused
            .as_deref()
            .unwrap()
            .contains("whose folder is gone")
    );
    assert_eq!(p.mode, Mode::Model);
    assert_eq!(p.model.as_ref().unwrap().name, "mine");
    assert!(
        preview_text(&p).contains("refused: "),
        "{}",
        preview_text(&p)
    );
    let e = format!("{:#}", plan_refine(&c, req.clone(), &host()).unwrap_err());
    assert!(e.contains("whose folder is gone"), "{e}");
    // A name no configured model has: an error naming the ones trained here.
    let mut unknown = req;
    unknown.model = Some("nope".into());
    let e = format!(
        "{:#}",
        super::super::preview(&c, &unknown, &host()).unwrap_err()
    );
    assert!(e.contains("no configured model is named 'nope'"), "{e}");
}

#[test]
fn a_model_with_a_class_subset_judges_only_its_classes() {
    let f = fixture("subset", false);
    let c = config(&f);
    let seen = seen_of(&c, &entry()).unwrap();
    let classes = c.collect.classes.clone();
    let hidden = vec!["person".to_string()];
    let wanted = Wanted {
        classes: &classes,
        hidden: &hidden,
    };
    let j = model_judge(&seen, &wanted);
    assert_eq!(j.name, "mine");
    // person is hidden, motorbike is the list's motorcycle, dog it lacks.
    assert_eq!(
        j.classes,
        BTreeSet::from(["car".to_string(), "motorcycle".to_string()])
    );
    assert_eq!(classes_missing(&j, &wanted), ["dog"]);
    let w = missing_classes_warning("mine", &["dog".into()]);
    assert!(w.contains("does not detect dog"), "{w}");
    // Without the record's classes: the run's classes.txt.
    let record = f.root.join("models").join("mine.provenance.json");
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("classes");
    std::fs::write(&record, v.to_string()).unwrap();
    let l = locate(&c, &entry()).unwrap();
    assert_eq!(l.classes, ["person", "car", "motorbike"]);
    // Without the run's box counts: no class counts, so no verify, and said so.
    std::fs::remove_file(
        f.root
            .join("training")
            .join(RUN)
            .join("data")
            .join("dataset.json"),
    )
    .unwrap();
    let seen = seen_of(&c, &entry()).unwrap();
    assert!(seen.train_boxes.is_none());
    assert!(model_class_boxes(&seen, &wanted).is_empty());
    let p = super::super::preview(
        &c,
        &RefineRequest {
            model: Some("mine".into()),
            ..RefineRequest::default()
        },
        &host(),
    )
    .unwrap();
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("do not read, so no label gets a verify")),
        "{:?}",
        p.warnings
    );
}

#[test]
fn the_entry_it_runs_with_is_the_configured_one_without_what_hides_detections() {
    let e = entry();
    let p = predict_entry(&e);
    assert_eq!(p.family, e.family);
    assert_eq!(p.path, e.path);
    assert_eq!(p.classes, e.classes);
    assert_eq!(p.device, e.device);
    assert_eq!(p.confidence_threshold, None);
    assert!(p.device_thresholds.is_empty());
    assert_eq!(p.object_filter, None);
    assert!(!p.enabled);
}

fn suggestion(kind: Kind, target: Option<[f32; 4]>, bbox: Option<[f32; 4]>) -> Suggestion {
    Suggestion {
        kind,
        target,
        target_label: target.map(|_| "car".into()),
        bbox,
        label: Some("car".into()),
        score: Some(0.7),
        why: "an earlier check".into(),
        spot: None,
        answer: None,
        run: "earlier".into(),
    }
}

#[test]
fn the_preview_counts_only_the_frames_it_would_check() {
    let f = fixture("preview", false);
    let c = config(&f);
    // Open suggestions on a trained frame and on a new one; an answered one on another new one.
    edit(&f.dataset, &f.rels["trained"][0], |l| {
        l.refine = vec![suggestion(Kind::Add, None, Some(FAR))];
    });
    edit(&f.dataset, &f.rels["new"][0], |l| {
        l.refine = vec![
            suggestion(Kind::Add, None, Some(FAR)),
            suggestion(Kind::Verify, Some(CAR), None),
        ];
    });
    edit(&f.dataset, &f.rels["new"][1], |l| {
        let mut s = suggestion(Kind::Add, None, Some(FAR));
        s.answer = Some(Answer::Dismissed);
        l.refine = vec![s];
    });
    let req = RefineRequest {
        model: Some(" mine ".into()),
        ..RefineRequest::default()
    };
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert_eq!(p.mode, Mode::Model);
    assert!(p.refused.is_none(), "{:?}", p.refused);
    assert_eq!((p.frames, p.camera_days), (4, 3));
    assert!(p.folds.is_empty());
    assert_eq!(
        (p.open_suggestions, p.open_tightens, p.open_frames),
        (2, 0, 1),
        "the trained frame's open add is not this check's"
    );
    let replace = p.replace_text();
    assert_eq!(p.warnings[0], replace);
    assert!(
        replace.starts_with(
            "this check replaces the 2 open suggestions on 1 frame with what mine finds"
        ),
        "{replace}"
    );
    let m = p.model.as_ref().unwrap();
    assert_eq!(
        (m.trained_on, m.validated_on, m.same_day, m.imgsz),
        (2, 1, 1, 384)
    );
    assert_eq!(m.run, RUN);
    assert_eq!(m.device, "openvino:cpu");
    assert_eq!(m.classes_missing, ["dog"]);
    assert_eq!(p.recipe.model, "rfdetr-nano");
    assert_eq!(p.recipe.imgsz, 384);
    // The time: the documented guess, then a quick check's own measure of the same model.
    assert_eq!(p.time_basis, Some(TimeBasis::Guess));
    let want = QUICK_FIXED_S + QUICK_S_PER_FRAME * 4.0;
    assert!((p.seconds.unwrap() - want).abs() < 1e-9);
    let mut last = RefineReport::new(
        "r",
        0,
        RecipeInfo {
            model: "rfdetr-nano".into(),
            imgsz: 384,
            epochs: 0,
            batch: 0,
            from: String::new(),
        },
    );
    last.predict_s_per_frame = Some(0.05);
    last.predict_load_s = Some(45.0);
    assert_eq!(quick_seconds(4, "mine", Some(&last)).1, TimeBasis::Guess);
    last.mode = Mode::Model;
    last.model = Some(ModelInfo {
        name: "Mine".into(),
        ..ModelInfo::default()
    });
    let (s, b) = quick_seconds(4, "mine", Some(&last));
    assert_eq!(b, TimeBasis::LastCheck);
    assert!((s - (45.0 + 0.2)).abs() < 1e-9, "{s}");
    assert_eq!(quick_seconds(4, "other", Some(&last)).1, TimeBasis::Guess);
    // The load once, the frames each: a 45 s compile measured over 3 frames does not make 3,000
    // frames 12.5 h, and many frames are not one load per frame.
    last.predict_s_per_frame = Some(0.1);
    let (s, _) = quick_seconds(3000, "mine", Some(&last));
    assert!((s - (45.0 + 300.0)).abs() < 1e-9, "{s}");
    let (s, _) = quick_seconds(3, "mine", Some(&last));
    assert!((s - 45.3).abs() < 1e-9, "{s}");
    // A report without the load beside its time per frame (the first quick checks' held the load
    // in it) is no measure.
    last.predict_load_s = None;
    assert_eq!(quick_seconds(3000, "mine", Some(&last)).1, TimeBasis::Guess);
    // Warnings: the same-day frame, the missing class, the by-name note, the closer look.
    let joined = p.warnings.join("\n");
    assert!(joined.contains("1 of the 4 frames mine never saw are of camera days it trained"));
    assert!(joined.contains("mine does not detect dog"));
    assert!(joined.contains("matched by name only"));
    assert!(p.warnings.contains(&tighten::skipped_warning()));
    let text = preview_text(&p);
    for want in [
        "quick dataset check of all reviewed frames in",
        "with mine: 4 frames it never saw on 3 camera days",
        "model: mine (rfdetr-nano at 384 px, training run 20261111-020504-cad6, runs on \
         openvino:cpu); skipped: 2 frames it trained on, 1 it validated on",
        "frames matched by name (the run has no record of its image sources)",
        "loose labels: counted only (no closer look)",
        "(a rough guess)",
        "nothing was run or written (--dry-run)",
    ] {
        assert!(text.contains(want), "{want}\n{text}");
    }
    assert!(!text.contains("folds, each"), "{text}");
    let start = super::super::start_warnings(&p, None);
    assert_eq!(start[0], replace);
    assert!(start[1].starts_with("1 of the 4 frames mine never saw"));
    // With the record of sources: no by-name note.
    let g = fixture("preview-src", true);
    let p = super::super::preview(&config(&g), &req, &host()).unwrap();
    assert!(!p.warnings.join("\n").contains("matched by name only"));
    assert!(p.model.unwrap().by_source);
}

#[test]
fn every_frame_seen_or_no_gpu_for_an_asked_look_is_refused_and_no_gpu_is_needed_otherwise() {
    let f = fixture("all-seen", false);
    let c = config(&f);
    let mut none = host();
    none.accelerator = None;
    none.cuda_options.clear();
    let req = RefineRequest {
        model: Some("mine".into()),
        ..RefineRequest::default()
    };
    // No GPU: the quick check plans all the same (the model runs on its own device).
    let job = plan_refine(&c, req.clone(), &none).unwrap();
    assert!(job.train.is_none());
    assert_eq!(job.request.tighten, Some(false));
    assert!(job.quick.as_ref().unwrap().look.is_none());
    // The closer look asked for without a GPU: refused at once.
    let ask = RefineRequest {
        tighten: Some(true),
        ..req.clone()
    };
    let e = format!("{:#}", plan_refine(&c, ask.clone(), &none).unwrap_err());
    assert!(e.contains("NVIDIA GPU or Apple silicon"), "{e}");
    let p = super::super::preview(&c, &ask, &none).unwrap();
    assert!(p.refused.unwrap().contains("closer look"));
    // Every reviewed frame seen: nothing to check.
    for rel in f.rels.values().flatten() {
        if !f.rels["trained"].contains(rel) {
            std::fs::remove_file(path_of(&f.dataset, rel)).unwrap();
        }
    }
    let p = super::super::preview(&c, &req, &host()).unwrap();
    let why = p.refused.unwrap();
    assert!(
        why.contains("mine trained or validated on every reviewed frame"),
        "{why}"
    );
    assert!(plan_refine(&c, req, &host()).is_err());
}

/// A fake model: on every frame the labelled car and an unlabelled person.
#[derive(Default)]
struct ModelFake {
    asked: Mutex<Vec<(FoldModel, Vec<String>)>>,
    cancel: bool,
}

impl FoldPredictor for ModelFake {
    fn predict(
        &self,
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections> {
        assert_eq!(threshold, judge::MATCH_MIN_SCORE);
        // A measurable pass: the report's time per frame.
        std::thread::sleep(std::time::Duration::from_millis(10));
        self.asked.lock().unwrap().push((
            model.clone(),
            images.iter().map(|i| i.frame.clone()).collect(),
        ));
        if self.cancel {
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            return Ok(Detections {
                cancelled: true,
                ..Detections::default()
            });
        }
        let dets = vec![
            Det {
                label: "car".into(),
                score: 0.93,
                bbox: CAR,
            },
            Det {
                label: "person".into(),
                score: 0.82,
                bbox: FAR,
            },
        ];
        Ok(Detections {
            frames: images
                .iter()
                .map(|i| (i.frame.clone(), dets.clone()))
                .collect(),
            device: "openvino:cpu".into(),
            cancelled: false,
            load_s: Some(0.004),
        })
    }
}

/// Every file under `dir` with its bytes.
fn files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(
                    p.strip_prefix(dir).unwrap().to_path_buf(),
                    std::fs::read(&p).unwrap(),
                );
            }
        }
    }
    out
}

fn run(job: &RefineJob, cancel: &AtomicBool) -> Result<crate::training::task::TaskOutcome> {
    let fake = crate::resources::export::fake::Fake::new();
    job.run(
        TaskContext {
            runner: &fake,
            fetcher: &fake,
        },
        &mut |_| {},
        cancel,
    )
}

#[test]
fn a_quick_check_writes_only_the_frames_it_never_saw_and_nothing_else() {
    let f = fixture("job", false);
    let c = config(&f);
    let cfg_path = f.root.join("blue_onyx_prism_config.json");
    c.save(&cfg_path).unwrap();
    // A trained frame holds an open and an answered suggestion of an earlier check; a new frame
    // an answered one, which stays.
    let trained = f.rels["trained"][0].clone();
    edit(&f.dataset, &trained, |l| {
        let mut answered = suggestion(Kind::Verify, Some(CAR), None);
        answered.answer = Some(Answer::Kept);
        l.refine = vec![suggestion(Kind::Add, None, Some(FAR)), answered];
    });
    let new = f.rels["new"][0].clone();
    let mut dismissed = suggestion(Kind::Add, None, Some([0.0, 150.0, 40.0, 230.0]));
    dismissed.answer = Some(Answer::Dismissed);
    edit(&f.dataset, &new, |l| l.refine = vec![dismissed.clone()]);
    let watched = |root: &Path| {
        let mut m = files(&root.join("models"));
        m.extend(files(&root.join("training")));
        m.insert(cfg_path.clone(), std::fs::read(&cfg_path).unwrap());
        m
    };
    let kept = watched(&f.root);
    let before = files(&f.dataset);
    let predictor = Arc::new(ModelFake::default());
    let req = RefineRequest {
        model: Some("mine".into()),
        ..RefineRequest::default()
    };
    let mut job = plan_refine(&c, req, &host()).unwrap();
    job.predictor = Some(predictor.clone());
    assert_eq!(job.kind(), TaskKind::Refine);
    assert_eq!(job.steps(), 3);
    assert_eq!(
        job.title(),
        "quick dataset check with mine (4 frames it never saw, all reviewed frames)"
    );
    let out = run(&job, &AtomicBool::new(false)).unwrap();
    assert!(
        out.summary.starts_with(
            "4 frames mine never saw checked (2 it trained on and 1 it validated on left out); \
             suggested: 4 missing boxes, 0 labels to verify, 0 other classes (4 of these \
             suggestions new since the last check, on 4 frames written)"
        ),
        "{}",
        out.summary
    );
    assert!(
        out.summary.contains("No box was changed"),
        "{}",
        out.summary
    );
    // The model ran once, on the frames it never saw, with its configured entry made blind to
    // nothing.
    let asked = predictor.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 1);
    let (m, frames) = &asked[0];
    let mut want = f.rels["heldout"].clone();
    want.extend(f.rels["new"].iter().cloned());
    want.extend(f.rels["same_day"].iter().cloned());
    assert_eq!(sorted(frames.clone()), sorted(want.clone()));
    assert_eq!(m.name, "mine");
    assert_eq!(m.onnx, f.root.join("models").join("mine.onnx"));
    assert_eq!(m.yaml, f.root.join("models").join("mine.yaml"));
    let e = m.entry.as_ref().unwrap();
    assert_eq!(e.device.as_deref(), Some("openvino:cpu"));
    assert_eq!((e.confidence_threshold, &e.object_filter), (None, &None));
    // The label files: the checked ones gained an add in the model's words, their answered
    // suggestion stays; the others are byte for byte as they were.
    let after = files(&f.dataset);
    for (p, was) in &before {
        let now = &after[p];
        let rel = p.to_string_lossy().replace('\\', "/");
        let checked = want
            .iter()
            .any(|w| rel.ends_with(&w.replace(".jpg", ".labels.json")));
        if !p.to_string_lossy().ends_with(".labels.json") || !checked {
            assert_eq!(now, was, "{} changed", p.display());
            continue;
        }
        let l: LabelFile = serde_json::from_slice(now).unwrap();
        let old: LabelFile = serde_json::from_slice(was).unwrap();
        let mut bare = l.clone();
        bare.refine.clear();
        let mut old_bare = old.clone();
        old_bare.refine.clear();
        assert_eq!(bare, old_bare, "{}: nothing but refine", p.display());
        let adds: Vec<&Suggestion> = l
            .refine
            .iter()
            .filter(|s| s.kind == Kind::Add && s.answer.is_none())
            .collect();
        assert_eq!(adds.len(), 1, "{}", p.display());
        assert_eq!(adds[0].bbox, Some(FAR));
        assert_eq!(adds[0].run, job.id);
        assert_eq!(
            adds[0].why,
            "mine (never trained on this frame) found a person here, score 0.82"
        );
        if rel.ends_with(&new.replace(".jpg", ".labels.json")) {
            assert!(l.refine.contains(&dismissed), "the answered one stays");
        }
    }
    // The only new file is the quick check's report (the full check's report.json is its own).
    let new_files: Vec<&PathBuf> = after.keys().filter(|p| !before.contains_key(*p)).collect();
    let quick_file = PathBuf::from("refine").join("report-quick.json");
    assert_eq!(new_files, [&quick_file]);
    let r: RefineReport = serde_json::from_slice(&after[&quick_file]).unwrap();
    assert_eq!(r.mode, Mode::Model);
    assert_eq!((r.folds, r.folds_done), (0, 0));
    assert_eq!((r.frames_selected, r.frames_checked), (4, 4));
    let m = r.model.as_ref().unwrap();
    assert_eq!((m.name.as_str(), m.run.as_str()), ("mine", RUN));
    assert_eq!((m.trained_on, m.validated_on, m.same_day), (2, 1, 1));
    assert_eq!(m.device, "openvino:cpu");
    assert_eq!(m.classes_missing, ["dog"]);
    assert_eq!(r.recipe.model, "rfdetr-nano");
    assert_eq!(
        r.recall.boxes, 4,
        "the recall covers the checked frames only"
    );
    // The load (what the predictor measured) apart from the time per frame.
    assert_eq!(r.predict_load_s, Some(0.004));
    let per = r.predict_s_per_frame.unwrap();
    assert!(per > 0.0 && per < 1.0, "{per}");
    assert!(r.warnings.iter().any(|w| w.contains("does not detect dog")));
    // The fixture's record has no hash: said, not refused.
    assert!(
        r.warnings.iter().any(|w| w.contains("names no hash")),
        "{:?}",
        r.warnings
    );
    // The data root: models, training runs and the config untouched; no work folder left.
    assert_eq!(watched(&f.root), kept);
    assert!(!job.work.exists());
    // The preview of the next quick check takes its time from this one.
    let p = super::super::preview(
        &c,
        &RefineRequest {
            model: Some("mine".into()),
            ..RefineRequest::default()
        },
        &host(),
    )
    .unwrap();
    assert_eq!(p.time_basis, Some(TimeBasis::LastCheck));
    assert!(preview_text(&p).contains("(from the last quick check with this model)"));
    // A rerun with the same answers writes nothing new.
    let mut again = plan_refine(
        &c,
        RefineRequest {
            model: Some("mine".into()),
            ..RefineRequest::default()
        },
        &host(),
    )
    .unwrap();
    again.predictor = Some(Arc::new(ModelFake::default()));
    let labels_before: BTreeMap<PathBuf, Vec<u8>> = files(&f.dataset)
        .into_iter()
        .filter(|(p, _)| p.to_string_lossy().ends_with(".labels.json"))
        .collect();
    let out = run(&again, &AtomicBool::new(false)).unwrap();
    assert!(
        out.summary.contains("(0 of these suggestions new"),
        "{}",
        out.summary
    );
    let labels_after: BTreeMap<PathBuf, Vec<u8>> = files(&f.dataset)
        .into_iter()
        .filter(|(p, _)| p.to_string_lossy().ends_with(".labels.json"))
        .collect();
    assert_eq!(labels_after, labels_before);
}

#[test]
fn a_cancel_during_the_models_pass_writes_nothing() {
    let f = fixture("cancel", false);
    let c = config(&f);
    let before = files(&f.dataset);
    let mut job = plan_refine(
        &c,
        RefineRequest {
            model: Some("mine".into()),
            ..RefineRequest::default()
        },
        &host(),
    )
    .unwrap();
    job.predictor = Some(Arc::new(ModelFake {
        cancel: true,
        ..ModelFake::default()
    }));
    let e = run(&job, &AtomicBool::new(false)).unwrap_err();
    assert!(crate::training::pyenv::is_cancel(&e), "{e:#}");
    assert_eq!(files(&f.dataset), before);
    assert!(!job.work.exists());
}

#[test]
fn the_models_list_has_those_trained_here_and_says_why_others_cannot_be_used() {
    let f = fixture("models", false);
    let mut c = config(&f);
    // A stock model (no record): in neither list. A trained one whose run is gone: unusable.
    c.models.push(ModelConfig {
        name: Some("stock".into()),
        path: PathBuf::from("models").join("stock.onnx"),
        ..ModelConfig::default()
    });
    let gone = ModelConfig {
        name: Some("gone".into()),
        path: PathBuf::from("models").join("gone.onnx"),
        ..ModelConfig::default()
    };
    std::fs::write(
        f.root.join("models").join("gone.provenance.json"),
        r#"{"run_id":"20260101-000000-dead","recipe":{"imgsz":640}}"#,
    )
    .unwrap();
    std::fs::write(f.root.join("models").join("gone.onnx"), b"onnx").unwrap();
    c.models.push(gone);
    // A record and its run, but the model file is gone.
    let models_dir = f.root.join("models");
    std::fs::copy(
        models_dir.join("mine.provenance.json"),
        models_dir.join("lost.provenance.json"),
    )
    .unwrap();
    c.models.push(ModelConfig {
        name: Some("lost".into()),
        path: PathBuf::from("models").join("lost.onnx"),
        ..ModelConfig::default()
    });
    let (ok, bad) = models(&c);
    assert_eq!(ok.len(), 1);
    assert_eq!(
        (ok[0].name.as_str(), ok[0].run.as_str(), ok[0].imgsz),
        ("mine", RUN, 384)
    );
    assert_eq!(ok[0].recipe, "rfdetr-nano");
    assert_eq!(bad.len(), 2);
    assert_eq!(bad[0].name, "gone");
    assert!(
        bad[0].why.contains("whose folder is gone"),
        "{}",
        bad[0].why
    );
    assert_eq!(bad[1].name, "lost");
    assert!(
        bad[1].why.contains("model file") && bad[1].why.contains("lost.onnx"),
        "{}",
        bad[1].why
    );
    // The plan refuses it naming the file; the preview says the same as a refusal.
    let req = RefineRequest {
        model: Some("lost".into()),
        ..RefineRequest::default()
    };
    let e = format!("{:#}", plan_refine(&c, req.clone(), &host()).unwrap_err());
    assert!(e.contains("lost.onnx") && e.contains("is missing"), "{e}");
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(p.refused.unwrap().contains("lost.onnx"));
    let v = serde_json::to_value(&ok[0]).unwrap();
    assert!(v.get("created").is_none(), "{v}");
    // An unknown name names the usable ones.
    let e = no_model(&c, "x").to_string();
    assert!(e.contains("models trained here: mine"), "{e}");
}

/// Set (or with None, drop) `onnx_sha256` in `mine`'s training record.
fn set_hash(root: &Path, hash: Option<&str>) {
    let record = root.join("models").join("mine.provenance.json");
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    let o = v.as_object_mut().unwrap();
    match hash {
        Some(h) => o.insert("onnx_sha256".into(), h.into()),
        None => o.remove("onnx_sha256"),
    };
    std::fs::write(&record, v.to_string()).unwrap();
}

#[test]
fn a_model_file_that_is_not_the_one_its_run_produced_is_refused() {
    let f = fixture("hash", false);
    let c = config(&f);
    let req = RefineRequest {
        model: Some("mine".into()),
        ..RefineRequest::default()
    };
    let onnx = f.root.join("models").join("mine.onnx");
    let right = crate::resources::manager::sha256_file(&onnx).unwrap();
    // The record's hash is the file's (in upper case too): planned, no warning about it.
    set_hash(&f.root, Some(&right.to_ascii_uppercase()));
    let l = locate(&c, &entry()).unwrap();
    assert_eq!(check_weights(&l).unwrap(), None);
    plan_refine(&c, req.clone(), &host()).unwrap();
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(p.refused.is_none(), "{:?}", p.refused);
    assert!(!p.warnings.join("\n").contains("hash"), "{:?}", p.warnings);
    // The file replaced since: refused at plan and preview time, never by the models list.
    std::fs::write(&onnx, b"another model").unwrap();
    let e = format!("{:#}", plan_refine(&c, req.clone(), &host()).unwrap_err());
    assert!(
        e.contains("is not the file training run")
            && e.contains(RUN)
            && e.contains("the frames it trained on are unknown"),
        "{e}"
    );
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(
        p.refused
            .as_deref()
            .unwrap()
            .contains("is not the file training run"),
        "{:?}",
        p.refused
    );
    assert_eq!(models(&c).0.len(), 1, "the list never hashes");
    // No hash in the record: a warning, not a refusal.
    set_hash(&f.root, None);
    let job = plan_refine(&c, req.clone(), &host()).unwrap();
    let w = job.quick.unwrap().weights_warning.unwrap();
    assert!(w.contains("names no hash"), "{w}");
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(p.refused.is_none());
    assert!(p.warnings.contains(&w));
}

#[test]
fn listed_names_that_are_no_frame_names_are_counted_and_lists_naming_no_frame_refuse() {
    let f = fixture("odd", false);
    let c = config(&f);
    let data = f.root.join("training").join(RUN).join("data");
    let req = RefineRequest {
        model: Some("mine".into()),
        ..RefineRequest::default()
    };
    let seen = seen_of(&c, &entry()).unwrap();
    assert_eq!((seen.odd_names, seen.listed()), (0, 3));
    // Hand-made names beside the app's: two with a prefix, one without; a public set's image
    // (s1 in dataset.json) is not counted; a cut name is no odd name.
    let mut t = std::fs::read_to_string(data.join("train.txt")).unwrap();
    t.push_str("./images/s0-IMG_0001.jpg\n./images/frame_17.jpg\n./images/s1-000000123456.jpg\n");
    t.push_str(&format!(
        "./images/{}\n",
        "s0-".to_string() + &"y".repeat(117)
    ));
    std::fs::write(data.join("train.txt"), &t).unwrap();
    let mut v = std::fs::read_to_string(data.join("valid.txt")).unwrap();
    v.push_str("./images/s0-IMG_0002.jpg\n");
    std::fs::write(data.join("valid.txt"), &v).unwrap();
    let summary = data.join("dataset.json");
    let mut s: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summary).unwrap()).unwrap();
    s["sources"] = serde_json::json!([
        {"dir": "/x", "kind": "review", "license": "", "frames": 9, "used_train": 2, "used_heldout": 1},
        {"dir": "/y", "kind": "train-data:lila-night", "license": "", "frames": 9, "used_train": 1, "used_heldout": 0},
    ]);
    std::fs::write(&summary, s.to_string()).unwrap();
    let seen = seen_of(&c, &entry()).unwrap();
    assert_eq!((seen.odd_names, seen.listed()), (3, 8));
    let p = super::super::preview(&c, &req, &host()).unwrap();
    assert!(p.refused.is_none(), "{:?}", p.refused);
    let w = p
        .warnings
        .iter()
        .find(|w| w.contains("do not carry this program's frame names"))
        .unwrap_or_else(|| panic!("{:?}", p.warnings));
    assert!(w.starts_with("3 of the 8 images in run "), "{w}");
    // Lists that name no reviewed frame of the folder at all (a run on renamed copies): refused,
    // even under a frame filter that leaves out every frame it did see.
    for list in ["train.txt", "valid.txt"] {
        std::fs::write(data.join(list), "./images/s0-IMG_0003.jpg\n").unwrap();
    }
    let p = super::super::preview(&c, &req, &host()).unwrap();
    let why = p.refused.unwrap();
    assert!(
        why.contains("none of the 2 images mine trained or validated on is a reviewed frame")
            && why.contains("the safe side"),
        "{why}"
    );
    let e = format!("{:#}", plan_refine(&c, req.clone(), &host()).unwrap_err());
    assert!(e.contains("cannot tell which frames it saw"), "{e}");
    // A frame filter that leaves out the frames it saw does not make the lists look foreign.
    let line = |rel: &String| format!("./images/s0-{}\n", flat(rel));
    std::fs::write(data.join("train.txt"), line(&f.rels["trained"][0])).unwrap();
    let filtered = RefineRequest {
        frames: RefineFrames::parse("camera=delta").unwrap(),
        ..req
    };
    let p = super::super::preview(&c, &filtered, &host()).unwrap();
    assert!(p.refused.is_none(), "{:?}", p.refused);
    assert_eq!(p.frames, 2);
}

/// The quick check's preview on a copy of a real dataset, read only (no GPU, nothing written):
/// `BOP_REFINE_CONFIG=<config file> [BOP_REFINE_MODEL=<name>] cargo test -j 2 --lib
/// quick_preview_real -- --ignored --nocapture`. Lists the configured models trained here (and
/// those that cannot be used, with why), then previews the named model (default: the newest).
#[test]
#[ignore = "needs a copy of a dataset and a model trained on it"]
fn quick_preview_real_on_a_copy() {
    let Some(path) = std::env::var_os("BOP_REFINE_CONFIG").map(PathBuf::from) else {
        println!("skipped: BOP_REFINE_CONFIG is not set");
        return;
    };
    let cfg = Config::load(&path).unwrap();
    let (usable, unusable) = models(&cfg);
    for m in &usable {
        println!(
            "usable: {} ({} at {} px, run {}, enabled {})",
            m.name, m.recipe, m.imgsz, m.run, m.enabled
        );
    }
    for u in &unusable {
        println!("not usable: {}: {}", u.name, u.why);
    }
    let name = std::env::var("BOP_REFINE_MODEL")
        .ok()
        .or_else(|| usable.first().map(|m| m.name.clone()))
        .expect("no model trained here");
    let entry = find_entry(&cfg, &name).unwrap();
    let started = std::time::Instant::now();
    match seen_of(&cfg, entry) {
        Ok(seen) => {
            let l = seen.located.as_ref().unwrap();
            println!(
                "record {} -> run {} in {} (image sources recorded: {}); {} train and {} valid \
                 image names; run box counts: {:?}",
                l.record.display(),
                l.run,
                l.run_dir.display(),
                seen.by_source,
                seen.train.len(),
                seen.valid.len(),
                seen.train_boxes
            );
            println!(
                "listed images without this program's frame names: {} of {}",
                seen.odd_names,
                seen.listed()
            );
            let hashed = std::time::Instant::now();
            match check_weights(l) {
                Ok(None) => println!(
                    "model file {} matches the record's onnx_sha256 ({:.1} s)",
                    l.model_file.display(),
                    hashed.elapsed().as_secs_f64()
                ),
                Ok(Some(w)) => println!("model file not checked: {w}"),
                Err(e) => println!("model file refused: {e:#}"),
            }
        }
        Err(e) => println!("refused: {e:#}"),
    }
    let p = super::super::preview(
        &cfg,
        &RefineRequest {
            model: Some(name),
            ..RefineRequest::default()
        },
        &crate::training::capability_now(),
    )
    .unwrap();
    println!(
        "{:.1} s\n{}",
        started.elapsed().as_secs_f64(),
        preview_text(&p)
    );
    println!("model: {:?}", p.model);
}
