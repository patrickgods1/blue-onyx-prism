//! "Your cameras" on temp review folders: the selection is the Dataset export's, a missing folder
//! is no empty set, trained models leave out what they saw, the snapshot keeps results put and
//! a re-score catches up with label edits made through `labels::update`.

use super::snapshot::{self, GtSnapshot};
use super::*;
use crate::benchmark::images::{self as bench_images, CamerasLoaded, SetImage};
use crate::benchmark::metrics::PredBox;
use crate::benchmark::report::{BenchmarkResults, test_support};
use crate::benchmark::search::StoredPreds;
use crate::collect::labels::{
    self, BoxState, LabelBox, LabelFile, RejectReason, Source as BoxSource, Status,
};
use crate::config::{Config, DatasetRef, ModelConfig};

/// A temp folder removed on drop.
struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("bop-cameras-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn lb(label: &str, b: [f32; 4], state: BoxState) -> LabelBox {
    LabelBox {
        label: label.into(),
        bbox: b,
        source: BoxSource::Model,
        score: Some(0.9),
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

pub(crate) fn reviewed(boxes: Vec<LabelBox>) -> LabelFile {
    LabelFile {
        status: Status::Reviewed,
        reviewed_at: Some("2026-11-20T00:00:00Z".into()),
        boxes,
        ..LabelFile::default()
    }
}

/// A frame (64x48 image, sidecar, label file) at `rel` (`<scene>/<camera>/<date>/<stem>`).
pub(crate) fn put(dir: &Path, rel: &str, f: &LabelFile) {
    let r = crate::collect::layout::parse_rel(&format!("{rel}.jpg")).unwrap();
    let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
    std::fs::create_dir_all(&day).unwrap();
    image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]))
        .save(day.join(format!("{}.jpg", r.stem)))
        .unwrap();
    let mut rec = crate::collect::record::empty_record();
    rec.id = r.stem.clone();
    rec.camera = r.camera.clone();
    rec.width = 64;
    rec.height = 48;
    std::fs::write(
        day.join(format!("{}.json", r.stem)),
        serde_json::to_vec(&rec).unwrap(),
    )
    .unwrap();
    labels::write(&labels_path(dir, rel), f).unwrap();
}

pub(crate) fn labels_path(dir: &Path, rel: &str) -> PathBuf {
    dir.join(format!("{rel}{}", labels::LABELS_SUFFIX))
}

pub(crate) fn flat(rel: &str) -> String {
    format!("{}.jpg", rel.replace('/', "."))
}

const FLOOR: SizeFloor = SizeFloor {
    px: 32.0,
    input: 640,
};

/// A config whose dataset folder (the folder under review) is `<root>/dataset`.
fn config(root: &Path) -> Config {
    let mut c = Config {
        download_dir: Some(root.to_path_buf()),
        ..Config::default()
    };
    c.collect.dir = Some(root.join("dataset"));
    c.collect.classes = vec!["person".into(), "car".into(), "dog".into()];
    c.collect.hidden_classes = vec!["dog".into()];
    c.collect.size_floor = Some(FLOOR);
    c
}

/// Every kind of frame the export's rules decide on.
fn every_kind(dir: &Path) {
    let person = || lb("person", [4.0, 4.0, 40.0, 40.0], BoxState::Accepted);
    put(
        dir,
        "day/front/2026-11-01/a",
        &reviewed(vec![
            person(),
            lb("car", [30.0, 10.0, 60.0, 40.0], BoxState::Added),
        ]),
    );
    // An unsure box and a box under the size floor: ignore regions.
    let mut unsure = person();
    unsure.difficult = true;
    put(
        dir,
        "day/front/2026-11-01/b",
        &reviewed(vec![
            unsure,
            lb("person", [1.0, 1.0, 3.0, 3.0], BoxState::Accepted),
        ]),
    );
    // Rejected as too small: an ignore region too.
    let mut small = lb("car", [5.0, 5.0, 9.0, 9.0], BoxState::Rejected);
    small.reject_reason = Some(RejectReason::TooSmall);
    put(
        dir,
        "night/front/2026-11-02/c",
        &reviewed(vec![small, person()]),
    );
    // Confirmed empty: a hard negative.
    put(
        dir,
        "day/back/2026-11-01/d",
        &LabelFile {
            empty_confirmed: true,
            ..reviewed(Vec::new())
        },
    );
    // A hidden class: kept, as the export keeps it.
    put(
        dir,
        "day/back/2026-11-02/e",
        &reviewed(vec![lb(
            "dog",
            [10.0, 10.0, 50.0, 40.0],
            BoxState::Accepted,
        )]),
    );
    // Reopened by a re-label: out until saved.
    put(
        dir,
        "day/back/2026-11-03/f",
        &LabelFile {
            status: Status::Unreviewed,
            disagreement: Some("a labeler found an object".into()),
            ..reviewed(vec![person()])
        },
    );
    // Skipped: out.
    put(
        dir,
        "ir/back/2026-11-03/g",
        &LabelFile {
            status: Status::Skipped,
            ..reviewed(vec![person()])
        },
    );
    // A label outside the class list: out, with the export's warning.
    put(
        dir,
        "day/back/2026-11-04/h",
        &reviewed(vec![lb(
            "raccoon",
            [10.0, 10.0, 50.0, 40.0],
            BoxState::Accepted,
        )]),
    );
    put(dir, "ir/side/2026-11-05/i", &reviewed(vec![person()]));
}

pub(crate) fn ready(l: Load) -> Loaded {
    match l {
        Load::Ready(l) => *l,
        other => panic!("not ready: {other:?}"),
    }
}

fn by_file(images: &[SetImage]) -> std::collections::BTreeMap<String, SetImage> {
    images.iter().map(|i| (i.file.clone(), i.clone())).collect()
}

#[test]
fn the_selection_is_the_dataset_exports_benchmark_manifest() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    every_kind(&dir);
    // The Dataset tab's export, benchmark format, every split (it also grows heldout-days.json,
    // which the set then reads, so the split tags agree).
    let report = labels::export::run(
        &dir,
        &cfg.collect.classes,
        &labels::export::ExportRequest {
            formats: vec![labels::export::Format::Benchmark],
            out_dir: t.0.join("export"),
            split: SplitConfig::default(),
            benchmark_split: labels::export::BenchSplit::All,
            include_unreviewed: false,
            hidden_classes: cfg.collect.hidden_classes.clone(),
            size_floor: cfg.collect.size_floor,
            heldout: Some(labels::heldout::HeldoutPolicy {
                seen: Some(BTreeSet::new()),
            }),
        },
    )
    .unwrap();
    let (exported, _) =
        bench_images::load_dir(&report.paths["benchmark"], None, Some("export")).unwrap();
    let l = ready(load(&Source::of(&cfg)));
    let (ours, theirs) = (by_file(&l.set.images), by_file(&exported.images));
    assert_eq!(
        ours.keys().collect::<Vec<_>>(),
        theirs.keys().collect::<Vec<_>>()
    );
    for (file, a) in &ours {
        let b = &theirs[file];
        let sorted = |v: &[String]| {
            let mut v = v.to_vec();
            v.sort();
            v
        };
        assert_eq!((a.width, a.height), (b.width, b.height), "{file}");
        assert_eq!(sorted(&a.tags), sorted(&b.tags), "{file}");
        assert_eq!(a.gt, b.gt, "{file}");
        let frame = dir.join(labels::export::parse_flat_name(file).unwrap().jpg());
        assert_eq!(
            a.path.as_ref().map(|p| p.canonicalize().unwrap()),
            Some(frame.canonicalize().unwrap()),
            "{file}: the review frame itself, read in place"
        );
    }
    assert_eq!(l.set.scored_labels, exported.scored_labels);
    // What that means, frame by frame.
    let names: Vec<&str> = ours.keys().map(String::as_str).collect();
    for gone in ["f", "g", "h"] {
        assert!(
            !names.iter().any(|n| n.ends_with(&format!(".{gone}.jpg"))),
            "{gone}: reopened, skipped or of an unknown class"
        );
    }
    let img = |rel: &str| &ours[&flat(rel)];
    let gt = |rel: &str| img(rel).gt.clone().unwrap();
    assert!(
        gt("day/front/2026-11-01/b").iter().all(|b| b.ignore),
        "unsure and under the floor"
    );
    let c = gt("night/front/2026-11-02/c");
    assert_eq!(
        c.iter()
            .map(|b| (b.label.as_str(), b.ignore))
            .collect::<Vec<_>>(),
        [("car", true), ("person", false)]
    );
    assert!(gt("day/back/2026-11-01/d").is_empty());
    assert!(
        img("day/back/2026-11-01/d")
            .tags
            .contains(&"hard-negative".to_string())
    );
    assert_eq!(gt("day/back/2026-11-02/e")[0].label, "dog");
    let a = &img("day/front/2026-11-01/a").tags;
    for t in ["day", "camera:front", "date:2026-11-01", "res:sd"] {
        assert!(a.contains(&t.to_string()), "{t} in {a:?}");
    }
    assert!(a.iter().any(|t| t.starts_with("split:")), "{a:?}");
    assert!(
        l.warnings.iter().any(|w| w.contains("raccoon")),
        "{:?}",
        l.warnings
    );
    // Every frame's revision, as the label file has it; the survey.
    for rel in ["day/front/2026-11-01/a", "ir/side/2026-11-05/i"] {
        assert_eq!(
            l.reviewed[&flat(rel)],
            labels::revision(&labels_path(&dir, rel)).unwrap()
        );
    }
    assert_eq!(l.reviewed.len(), 6);
    let s = &l.survey;
    assert_eq!((s.frames, s.cameras, s.days), (6, 3, 5));
    assert_eq!((s.day, s.night, s.ir, s.empty), (4, 1, 1, 1));
    assert_eq!(s.objects, 5, "a's two, c's person, e's dog, i's person");
    assert!(s.low_confidence() && s.warning().unwrap().contains("low confidence"));
    assert_eq!(s.resolution_range(), "64x48");
    let sized = |sizes, a, b| Survey {
        sizes,
        smallest: Some(a),
        largest: Some(b),
        ..Survey::default()
    };
    assert_eq!(
        sized(2, (1520, 2688), (2688, 1520)).resolution_range(),
        "1520x2688 and 2688x1520",
        "portrait and landscape of one sensor"
    );
    assert_eq!(
        sized(3, (1280, 720), (3840, 2160)).resolution_range(),
        "1280x720 to 3840x2160 (3 sizes)"
    );
    assert!(s.line().contains("6 frames") && s.line().contains("night 1"));
    // The cross-validation groups.
    assert_eq!(
        group_of(&img("day/front/2026-11-01/a").tags).as_deref(),
        Some("front/2026-11-01")
    );
    assert_eq!(group_of(&["day".to_string()]), None);
}

#[test]
fn a_walk_takes_no_lock_and_writes_nothing() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    put(
        &dir,
        "day/front/2026-11-01/a",
        &reviewed(vec![lb(
            "person",
            [4.0, 4.0, 40.0, 40.0],
            BoxState::Accepted,
        )]),
    );
    let lock = dir.join(".labels.lock");
    std::fs::remove_file(&lock).ok();
    let before: Vec<_> = walkdir(&dir);
    let l = ready(load(&Source::of(&cfg)));
    assert_eq!(l.set.images.len(), 1);
    assert_eq!(walkdir(&dir), before, "nothing created or changed");
    assert!(!lock.exists());
}

/// Every file under `dir` with its length and modification time.
fn walkdir(dir: &Path) -> Vec<(PathBuf, u64, Option<std::time::SystemTime>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let m = std::fs::metadata(&p).unwrap();
                out.push((p, m.len(), m.modified().ok()));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn a_missing_folder_is_folder_missing_and_an_empty_one_is_skipped() {
    let t = Tmp::new();
    let mut cfg = config(&t.0);
    cfg.collect.review_dir = Some(t.0.join("unplugged"));
    let src = Source::of(&cfg);
    let l = load(&src);
    assert!(matches!(l, Load::Missing(_)), "{l:?}");
    assert!(l.skip_note().unwrap().contains("folder_missing"));
    // A run skips it with the note and goes on (here with the sample: nothing else is listed).
    let refs = [DatasetRef::Id(ID.into())];
    let r = bench_images::resolve_with(&refs, &cfg.data_root(), 0, Some(&src)).unwrap();
    assert_eq!(r.sets.len(), 1);
    assert_eq!(r.sets[0].id, bench_images::SAMPLE_ID);
    assert!(
        r.warnings.iter().any(|w| w.contains("folder_missing")),
        "{:?}",
        r.warnings
    );
    assert!(r.cameras.is_none());
    // A folder without a reviewed frame: skipped too, never an error.
    let cfg = config(&t.0);
    std::fs::create_dir_all(cfg.collect.dir.as_ref().unwrap()).unwrap();
    let src = Source::of(&cfg);
    assert!(matches!(load(&src), Load::Empty(_)));
    let refs = [
        DatasetRef::Id(ID.into()),
        DatasetRef::Id(bench_images::SAMPLE_ID.into()),
    ];
    let r = bench_images::resolve_with(&refs, &cfg.data_root(), 0, Some(&src)).unwrap();
    assert_eq!(r.sets.len(), 1);
    assert!(
        r.warnings.iter().any(|w| w.contains("no reviewed frame")),
        "{:?}",
        r.warnings
    );
    // Without the config (a plain `resolve`), it says so.
    let r = bench_images::resolve(&refs, &cfg.data_root(), 0).unwrap();
    assert!(
        r.warnings.iter().any(|w| w.contains("--config")),
        "{:?}",
        r.warnings
    );
    // The id is known to validation.
    assert!(bench_images::validate(&refs, &cfg.data_root()).is_ok());
}

#[test]
fn a_run_resolves_the_set_with_its_revisions_and_a_small_set_warns() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    every_kind(&dir);
    let src = Source::of(&cfg);
    let refs = [DatasetRef::Id(ID.into())];
    let r = bench_images::resolve_with(&refs, &cfg.data_root(), 2, Some(&src)).unwrap();
    assert_eq!(r.sets[0].id, ID);
    assert_eq!(r.sets[0].images.len(), 2, "the run's image limit");
    let c = r.cameras.unwrap();
    assert_eq!(
        c.reviewed.len(),
        6,
        "every reviewed frame, before the limit"
    );
    assert!(
        r.warnings.iter().any(|w| w.contains("low confidence")),
        "{:?}",
        r.warnings
    );
}

/// A model with a training record whose run listed `train` and `valid` frames (identities).
fn trained_model(cfg: &Config, run: &str, train: &[&str], valid: &[&str]) -> ModelConfig {
    let models = cfg.data_root().join("models");
    std::fs::create_dir_all(&models).unwrap();
    let m = ModelConfig {
        name: Some("custom".into()),
        path: models.join("custom.onnx"),
        ..Default::default()
    };
    std::fs::write(
        models.join("custom.provenance.json"),
        format!(r#"{{"run_id":"{run}"}}"#),
    )
    .unwrap();
    if !train.is_empty() {
        let data = crate::training::training_dir(cfg).join(run).join("data");
        std::fs::create_dir_all(&data).unwrap();
        let list = |v: &[&str]| {
            v.iter()
                .map(|r| format!("./images/s0-{}", flat(r)))
                .collect::<Vec<_>>()
                .join("\n")
        };
        std::fs::write(data.join("train.txt"), list(train)).unwrap();
        std::fs::write(data.join("valid.txt"), list(valid)).unwrap();
    }
    m
}

#[test]
fn a_trained_model_leaves_out_the_frames_it_trained_or_validated_on() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    every_kind(&dir);
    let set = ready(load(&Source::of(&cfg))).set;
    let m = trained_model(
        &cfg,
        "run-1",
        &["day/front/2026-11-01/a"],
        &["night/front/2026-11-02/c"],
    );
    let (sets, left) = sets_for_model(&cfg, &m, std::slice::from_ref(&set));
    let own = &sets.unwrap()[0];
    assert_eq!(own.images.len(), set.images.len() - 2);
    for rel in ["day/front/2026-11-01/a", "night/front/2026-11-02/c"] {
        assert!(!own.images.iter().any(|i| i.file == flat(rel)), "{rel}");
    }
    let left = left.unwrap();
    assert_eq!((left.frames, left.trained, left.validated), (2, 1, 1));
    assert!(left.unknown.is_none());
    assert!(
        left.why().contains("1 frame(s) it trained on and 1"),
        "{}",
        left.why()
    );
    // Other sets are not touched.
    let (sets, _) = sets_for_model(
        &cfg,
        &m,
        &[set.clone(), crate::benchmark::images::ImageSet::sample()],
    );
    assert_eq!(sets.unwrap()[1].images.len(), 1);
    // No training record: every frame, nothing recorded.
    let plain = ModelConfig {
        name: Some("plain".into()),
        path: cfg.data_root().join("models/plain.onnx"),
        ..Default::default()
    };
    assert_eq!(
        seen::of_model(&cfg, &plain, std::iter::empty()),
        seen::Seen::NoRecord
    );
    assert!(matches!(
        sets_for_model(&cfg, &plain, std::slice::from_ref(&set)),
        (None, None)
    ));
    // Trained on every frame of the set: the set is out of its run (nothing fair to score).
    let all: Vec<String> = set
        .images
        .iter()
        .map(|i| labels::export::parse_flat_name(&i.file).unwrap().jpg())
        .collect();
    let all: Vec<&str> = all.iter().map(|r| r.trim_end_matches(".jpg")).collect();
    let greedy = trained_model(&cfg, "run-2", &all, &[]);
    let (sets, left) = sets_for_model(&cfg, &greedy, std::slice::from_ref(&set));
    assert!(sets.unwrap().is_empty());
    assert!(nothing_left(left.as_ref()).contains("trained or validated on"));
    // A record whose run folder is gone: unknown, scored on every frame, said why.
    let gone = trained_model(&cfg, "gone", &[], &[]);
    let (sets, left) = sets_for_model(&cfg, &gone, std::slice::from_ref(&set));
    assert_eq!(sets.unwrap()[0].images.len(), set.images.len());
    let left = left.unwrap();
    assert_eq!(left.frames, 0);
    assert!(left.unknown.unwrap().contains("gone"));
}

/// A scored run of a model that finds every labelled object of `set`, as a run stores it.
pub(crate) fn scored_results(set: &ImageSet) -> (BenchmarkResults, crate::benchmark::ModelResult) {
    let all: Vec<(&ImageSet, &SetImage)> = set.images.iter().map(|i| (set, i)).collect();
    let eval_preds: Vec<Vec<PredBox>> = set
        .images
        .iter()
        .map(|i| {
            i.gt.iter()
                .flatten()
                .filter(|g| !g.ignore)
                .map(|g| PredBox {
                    label: g.label.clone(),
                    confidence: 0.9,
                    bbox: g.bbox,
                    window: None,
                })
                .collect()
        })
        .collect();
    let mut per_image: Vec<crate::benchmark::ImageRun> = set
        .images
        .iter()
        .map(|i| crate::benchmark::ImageRun {
            set: set.id.clone(),
            file: i.file.clone(),
            total_ms: 10.0,
            infer_ms: 5.0,
            preds: Vec::new(),
            gt_matched: Vec::new(),
            counts: None,
            scene: None,
        })
        .collect();
    let job = crate::benchmark::Job {
        name: "m".into(),
        path: PathBuf::from("m.onnx"),
        family: Default::default(),
        classes: vec!["person".into(), "car".into()],
        device: "ort:cpu".into(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
        variant: None,
        concurrency: None,
        device_thresholds: Default::default(),
    };
    let (acc, note) = crate::benchmark::scoring::score(
        &job,
        &all,
        &mut per_image,
        &eval_preds,
        0.5,
        Default::default(),
    );
    assert!(acc.is_some(), "{note:?}");
    let mut d = test_support::dev("ort:cpu", 10.0, Vec::new());
    let run = d.run.as_mut().unwrap();
    run.accuracy = acc;
    run.images = per_image.len();
    run.per_image = per_image;
    run.eval_preds = eval_preds;
    let mut m = test_support::model(vec![d], None);
    m.datasets = vec![ID.into()];
    m.finish();
    let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m.clone()])
        .with_sets(std::slice::from_ref(set));
    (r, m)
}

fn recall(r: &BenchmarkResults) -> Option<f64> {
    r.models[0].devices[0]
        .run
        .as_ref()?
        .accuracy
        .as_ref()?
        .overall
        .recall
}

#[test]
fn the_snapshot_keeps_results_put_and_a_rescore_catches_up_with_label_edits() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    let person = |x: f32| lb("person", [x, 4.0, x + 20.0, 40.0], BoxState::Accepted);
    for rel in [
        "day/front/2026-11-01/a",
        "day/front/2026-11-02/b",
        "day/back/2026-11-01/c",
    ] {
        put(&dir, rel, &reviewed(vec![person(4.0)]));
    }
    let src = Source::of(&cfg);
    let l = ready(load(&src));
    let (results, m) = scored_results(&l.set);
    assert_eq!(recall(&results), Some(1.0));
    let results_path = t.0.join("benchmark.json");
    let preds_path = t.0.join("benchmark-preds.json");
    let id = write_snapshot(
        &results_path,
        std::slice::from_ref(&l.set),
        Some(&CamerasLoaded {
            source: src.clone(),
            reviewed: l.reviewed.clone(),
        }),
        None,
    )
    .unwrap()
    .unwrap();
    let mut results = results;
    snapshot::stamp(&mut results, &id);
    results.save(&results_path).unwrap();
    StoredPreds::merge(None, &results, &[m])
        .save(&preds_path)
        .unwrap();
    // benchmark.json holds the frames, not their labels; a load scores against the snapshot.
    let text = std::fs::read_to_string(&results_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let imgs = v["image_sets"][0]["images"].as_array().unwrap();
    assert_eq!(imgs.len(), 3);
    assert!(imgs.iter().all(|i| i.get("objects").is_none()), "{text}");
    assert!(
        imgs.iter().all(|i| i["group"].is_string()),
        "camera days as groups"
    );
    let stored = StoredPreds::load(&preds_path).unwrap().unwrap();
    assert!(stored.runs[0].images.iter().all(|i| i.group.is_some()));
    let loaded = BenchmarkResults::load(&results_path).unwrap().unwrap();
    assert_eq!(loaded.image_sets, results.image_sets);
    let snap = GtSnapshot::load(&snapshot::path_beside(&results_path))
        .unwrap()
        .unwrap();
    assert_eq!(snap.frames.len(), 3);
    assert_eq!(
        snapshot::changes(&snap, &l.reviewed),
        snapshot::Changes::default()
    );

    // Edits: b gains a second person (through the label write path), c is reopened, z is
    // reviewed for the first time.
    labels::update(&labels_path(&dir, "day/front/2026-11-02/b"), |f| {
        let mut f = f.unwrap();
        f.boxes.push(person(40.0));
        Ok(Some(f))
    })
    .unwrap();
    labels::update(&labels_path(&dir, "day/back/2026-11-01/c"), |f| {
        let mut f = f.unwrap();
        f.status = Status::Unreviewed;
        Ok(Some(f))
    })
    .unwrap();
    put(&dir, "day/back/2026-11-03/z", &reviewed(vec![person(4.0)]));
    let now = ready(load(&src));
    let changes = snapshot::changes(&snap, &now.reviewed);
    assert_eq!(
        changes,
        snapshot::Changes {
            frames: 3,
            changed: 1,
            left: 1,
            joined: 1,
            unknown: 0,
        }
    );
    // The saved results stay put meanwhile: still scored against the snapshot.
    let still = BenchmarkResults::load(&results_path).unwrap().unwrap();
    assert_eq!(still.image_sets, results.image_sets);
    assert_eq!(recall(&still), Some(1.0));

    let r = rescore::rescore(&cfg, &results_path, &preds_path).unwrap();
    assert_eq!((r.models, r.runs, r.frames, r.need_run), (1, 1, 2, 1));
    assert_eq!(r.changes, changes);
    assert!(r.not_rescored.is_empty());
    let after = BenchmarkResults::load(&results_path).unwrap().unwrap();
    let set = after.image_set(ID).unwrap();
    assert_eq!(set.images.len(), 2, "c left the set");
    let run = after.models[0].devices[0].run.as_ref().unwrap();
    assert_eq!(run.per_image.len(), 2);
    // a: 1 of 1, b: 1 of 2 (the new person has no prediction).
    let acc = run.accuracy.as_ref().unwrap();
    assert_eq!(acc.overall.counts.tp, 2);
    assert_eq!(acc.overall.counts.fn_, 1);
    let snap = GtSnapshot::load(&snapshot::path_beside(&results_path))
        .unwrap()
        .unwrap();
    assert_eq!(snap.frames.len(), 2);
    assert_eq!(
        snapshot::changes(&snap, &now.reviewed),
        snapshot::Changes {
            frames: 1,
            changed: 0,
            left: 0,
            joined: 1,
            unknown: 0,
        },
        "z still needs a run"
    );
}

#[test]
fn a_rescore_without_a_run_on_the_set_is_refused() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let results_path = t.0.join("benchmark.json");
    let preds_path = t.0.join("benchmark-preds.json");
    let e = rescore::rescore(&cfg, &results_path, &preds_path).unwrap_err();
    assert!(e.to_string().contains("no benchmark results"), "{e}");
    BenchmarkResults::new(Default::default(), Default::default(), Vec::new())
        .save(&results_path)
        .unwrap();
    let e = rescore::rescore(&cfg, &results_path, &preds_path).unwrap_err();
    assert!(e.to_string().contains("benchmark-gt.json"), "{e}");
}

#[test]
fn a_newer_snapshot_is_refused_and_a_missing_one_leaves_the_set_unscored() {
    let t = Tmp::new();
    let p = t.0.join(snapshot::FILE);
    std::fs::write(&p, r#"{"version": 99, "set": "cameras"}"#).unwrap();
    let e = GtSnapshot::load(&p).unwrap_err();
    assert!(format!("{e:#}").contains("newer format"), "{e:#}");
    // Results with the set but no snapshot: its images carry no labels, with a warning.
    std::fs::remove_file(&p).unwrap();
    let mut r: BenchmarkResults = serde_json::from_str(
        r#"{"image_sets": [{"id": "cameras", "kind": "dir", "ground_truth": {"kind": "manifest"},
            "images": [{"file": "day.cam.2026-11-01.a.jpg", "width": 1, "height": 1,
                        "objects": []}]}]}"#,
    )
    .unwrap();
    let w = snapshot::overlay(&mut r, &t.0.join("benchmark.json")).unwrap();
    assert!(w.contains("missing"), "{w}");
    assert!(r.image_sets[0].images[0].objects.is_none());
}

#[test]
fn only_review_frames_inside_the_folder_are_served() {
    let t = Tmp::new();
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    let rel = "day/front/2026-11-01/a";
    put(
        &dir,
        rel,
        &reviewed(vec![lb(
            "person",
            [4.0, 4.0, 40.0, 40.0],
            BoxState::Accepted,
        )]),
    );
    let l = ready(load(&Source::of(&cfg)));
    let info = crate::benchmark::report::ImageSetInfo::of(&l.set);
    let p = info.image_path(&flat(rel)).unwrap();
    assert_eq!(
        p,
        dir.join(rel).with_extension("jpg").canonicalize().unwrap()
    );
    for bad in [
        "../a.jpg",
        "day.front.2026-11-01.b.jpg",
        "x.jpg",
        "day/front/2026-11-01/a.jpg",
    ] {
        assert!(info.image_path(bad).is_none(), "{bad}");
    }
}

/// Three reviewed frames scored by `models` (each the same scored run of every frame), saved
/// with their snapshot; returns (results path, preds path, review folder, config).
fn saved_run(t: &Tmp, models: &[&str]) -> (PathBuf, PathBuf, PathBuf, Config) {
    let cfg = config(&t.0);
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    let person = lb("person", [4.0, 4.0, 24.0, 40.0], BoxState::Accepted);
    for rel in [
        "day/front/2026-11-01/a",
        "day/front/2026-11-02/b",
        "day/back/2026-11-01/c",
    ] {
        put(&dir, rel, &reviewed(vec![person.clone()]));
    }
    let src = Source::of(&cfg);
    let l = ready(load(&src));
    let results_path = t.0.join("benchmark.json");
    let preds_path = t.0.join("benchmark-preds.json");
    let id = write_snapshot(
        &results_path,
        std::slice::from_ref(&l.set),
        Some(&CamerasLoaded {
            source: src.clone(),
            reviewed: l.reviewed.clone(),
        }),
        None,
    )
    .unwrap()
    .unwrap();
    let mut all = Vec::new();
    for name in models {
        let (_, mut m) = scored_results(&l.set);
        m.model = (*name).to_string();
        all.push(m);
    }
    let mut r = BenchmarkResults::new(Default::default(), Default::default(), all.clone())
        .with_sets(std::slice::from_ref(&l.set));
    snapshot::stamp(&mut r, &id);
    r.save(&results_path).unwrap();
    StoredPreds::merge(None, &r, &all)
        .save(&preds_path)
        .unwrap();
    (results_path, preds_path, dir, cfg)
}

/// D1: a frame the walk could not read this time (its label file does not parse) is unknown,
/// not "left": it keeps its place, its labels and its figures, and the re-score lists it.
#[test]
fn a_rescore_keeps_frames_it_could_not_read() {
    let t = Tmp::new();
    let (results_path, preds_path, dir, cfg) = saved_run(&t, &["m"]);
    std::fs::write(labels_path(&dir, "day/back/2026-11-01/c"), b"{ not json").unwrap();
    let r = rescore::rescore(&cfg, &results_path, &preds_path).unwrap();
    assert_eq!(r.changes.left, 0, "{:?}", r.changes);
    assert_eq!(r.frames, 3);
    let after = BenchmarkResults::load(&results_path).unwrap().unwrap();
    assert_eq!(after.image_set(ID).unwrap().images.len(), 3);
    let run = after.models[0].devices[0].run.as_ref().unwrap();
    assert_eq!(run.per_image.len(), 3);
    let c = after
        .image_set(ID)
        .unwrap()
        .images
        .iter()
        .find(|i| i.file == flat("day/back/2026-11-01/c"))
        .unwrap();
    assert_eq!(c.objects.as_ref().map(Vec::len), Some(1), "its old labels");
    let snap = GtSnapshot::load(&snapshot::path_beside(&results_path))
        .unwrap()
        .unwrap();
    assert_eq!(snap.frames.len(), 3);
    let listed = serde_json::to_value(&r).unwrap();
    assert_eq!(
        listed["unknown"],
        serde_json::json!([flat("day/back/2026-11-01/c")]),
        "{listed}"
    );
}

/// D1: a re-score against another folder or another class list is refused: the snapshot is of
/// another set.
#[test]
fn a_rescore_of_another_folder_or_class_list_is_refused() {
    let t = Tmp::new();
    let (results_path, preds_path, _dir, mut cfg) = saved_run(&t, &["m"]);
    cfg.collect.classes.push("bicycle".into());
    let e = rescore::rescore(&cfg, &results_path, &preds_path).unwrap_err();
    assert!(format!("{e:#}").contains("run the benchmark"), "{e:#}");
    let mut cfg2 = config(&t.0);
    cfg2.collect.dir = Some(t.0.join("elsewhere"));
    std::fs::create_dir_all(t.0.join("elsewhere")).unwrap();
    put(
        &t.0.join("elsewhere"),
        "day/front/2026-11-01/a",
        &reviewed(vec![lb(
            "person",
            [4.0, 4.0, 24.0, 40.0],
            BoxState::Accepted,
        )]),
    );
    let e = rescore::rescore(&cfg2, &results_path, &preds_path).unwrap_err();
    assert!(format!("{e:#}").contains("run the benchmark"), "{e:#}");
}

/// D3: a snapshot that is not the one the results were saved with (a crash between the two
/// writes, a hand edit) is reported, not scored silently.
#[test]
fn a_snapshot_of_another_save_marks_the_set_stale() {
    let t = Tmp::new();
    let (results_path, _preds, _dir, cfg) = saved_run(&t, &["m"]);
    let (_, w) = BenchmarkResults::load_with_warnings(&results_path).unwrap();
    assert!(w.is_empty(), "{w:?}");
    // Another snapshot over it, the results untouched.
    let src = Source::of(&cfg);
    let l = ready(load(&src));
    let mut other = GtSnapshot::of(&src, &l.set, &l.reviewed);
    other.created = "2030-01-01T00:00:00Z".into();
    other.save(&snapshot::path_beside(&results_path)).unwrap();
    let (r, w) = BenchmarkResults::load_with_warnings(&results_path).unwrap();
    assert!(
        w.iter().any(|w| w.contains("Re-score")),
        "a warning that offers a re-score: {w:?}"
    );
    assert!(r.unwrap().image_set(ID).unwrap().labels_stale.is_some());
}

/// D4: a later run of fewer frames keeps the snapshot entries (and set images) the kept runs
/// still score: the earlier models re-score and compare on their frames.
#[test]
fn a_smaller_later_run_keeps_the_frames_of_earlier_runs() {
    let t = Tmp::new();
    let (results_path, preds_path, _dir, cfg) = saved_run(&t, &["m1", "m2"]);
    // m2 alone on the first two frames.
    let src = Source::of(&cfg);
    let l = ready(load(&src));
    let mut two = l.set.clone();
    two.images.truncate(2);
    // As a run's save does it: the previous results, the snapshot, the merge, the identity.
    let prev = BenchmarkResults::load(&results_path).unwrap();
    let id = write_snapshot(
        &results_path,
        std::slice::from_ref(&two),
        Some(&CamerasLoaded {
            source: src.clone(),
            reviewed: l.reviewed.clone(),
        }),
        prev.as_ref(),
    )
    .unwrap()
    .unwrap();
    let (_, mut m2) = scored_results(&two);
    m2.model = "m2".into();
    let new = BenchmarkResults::new(Default::default(), Default::default(), vec![m2.clone()])
        .with_sets(std::slice::from_ref(&two));
    let mut merged = BenchmarkResults::merge(prev, new);
    snapshot::stamp(&mut merged, &id);
    merged.save(&results_path).unwrap();
    let warnings = crate::benchmark::search::save_beside(&results_path, &merged, &[m2]);
    assert!(warnings.is_empty(), "{warnings:?}");
    let back = BenchmarkResults::load(&results_path).unwrap().unwrap();
    let set = back.image_set(ID).unwrap();
    assert_eq!(set.images.len(), 3, "m1 still scores the third frame");
    assert!(set.images.iter().all(|i| i.objects.is_some()), "{set:?}");
    let snap = GtSnapshot::load(&snapshot::path_beside(&results_path))
        .unwrap()
        .unwrap();
    assert_eq!(snap.frames.len(), 3);
    let r = rescore::rescore(&cfg, &results_path, &preds_path).unwrap();
    assert!(r.not_rescored.is_empty(), "{:?}", r.not_rescored);
    let after = BenchmarkResults::load(&results_path).unwrap().unwrap();
    let images = |name: &str| {
        after.find(name).unwrap().devices[0]
            .run
            .as_ref()
            .unwrap()
            .per_image
            .len()
    };
    assert_eq!((images("m1"), images("m2")), (3, 2));
}
