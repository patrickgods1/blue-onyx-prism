//! The auto-label stage machine over fakes: the export's [`Fake`] for uv and downloads, a fake
//! `autolabel.py` that proposes boxes for every frame it is given, and a fake [`Detector`] for
//! the self-training loop.

use super::*;
use crate::backend::detect::HardwareInfo;
use crate::collect::labels::{BoxState, Source, Status};
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::{Cmd, CommandRunner, Fetcher};
use crate::training::autolabel::triage::Det;
use crate::training::autolabel::{FrameFilter, plan_autolabel};
use crate::training::capability::{self, Capability};
use std::path::Path;
use std::sync::Mutex;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-autolabel-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

fn host() -> Capability {
    // The host's platform: a job finds the training environment of the OS it runs on.
    capability::host_for_tests()
}

/// `n` imported 64x48 frames (gray = IR, else colour) in `<root>/dataset`; returns their paths.
fn frames(root: &Path, n: usize, gray: bool) -> Vec<String> {
    let dir = root.join("dataset");
    let src = root.join("src.jpg");
    let mut rgb = Vec::with_capacity(64 * 48 * 3);
    for i in 0..64 * 48 {
        let v = u8::try_from(i % 200).unwrap();
        if gray {
            rgb.extend([v, v, v]);
        } else {
            rgb.extend([v, 255 - v, 128]);
        }
    }
    let img = crate::image::RgbImage {
        width: 64,
        height: 48,
        rgb,
    };
    let bytes = crate::image::encode_jpeg(&img, 90).unwrap();
    std::fs::write(&src, &bytes).unwrap();
    let decoded = crate::image::decode(&bytes).unwrap();
    let stats = crate::image_stats::ImageStats::compute(&decoded);
    let fs: Vec<crate::training::import::write::Frame> = (0..n)
        .map(|i| crate::training::import::write::Frame {
            file: src.clone(),
            image_name: "cam.jpg".into(),
            camera: "cam".into(),
            time: std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000 + i as u64),
            width: 64,
            height: 48,
            stats,
            dhash: i as u64,
            provenance: crate::collect::record::Provenance {
                import_id: "20261022-000000-abcd".into(),
                ..Default::default()
            },
        })
        .collect();
    let (rels, _) =
        crate::training::import::write::write_frames(&dir, "20261022-000000-abcd", &fs).unwrap();
    crate::collect::layout::append_review(&dir, &rels).unwrap();
    rels
}

#[derive(Default)]
struct LabelFake {
    inner: Fake,
    cmds: Mutex<Vec<Cmd>>,
}

impl CommandRunner for LabelFake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        if !cmd.display().contains("autolabel.py") {
            return self.inner.run(cmd, on_line, cancel);
        }
        if cmd.args.iter().any(|a| a == "--self-check") {
            on_line("@autolabel {\"event\":\"self_check\",\"mps\":true}");
            on_line("@autolabel {\"event\":\"self_test\",\"passed\":27}");
            return Ok(());
        }
        self.cmds.lock().unwrap().push(cmd.clone());
        let list = arg_after(cmd, "--frames").unwrap();
        let out = arg_after(cmd, "--out").unwrap();
        let mut text = String::new();
        for (i, l) in std::fs::read_to_string(list).unwrap().lines().enumerate() {
            let f: serde_json::Value = serde_json::from_str(l).unwrap();
            let line = serde_json::json!({"frame": f["frame"], "width": 64, "height": 48, "boxes": [
                {"label": "car", "bbox": [1, 1, 30, 20], "score": 0.82, "agree": true,
                 "sources": {"gdino": 0.7, "rfdetr": 0.94}},
                {"label": "raccoon", "bbox": [35, 20, 60, 46], "score": 0.31, "agree": false,
                 "sources": {"gdino": 0.31},
                 "vlm": {"model": "qwen3-vl-2b", "answer": "rock", "expected": "animal", "ok": false},
                 "species_scores": {"raccoon": 0.6, "cat": 0.4}},
                {"label": "unicorn", "bbox": [1, 1, 5, 5], "score": 0.9, "agree": true, "sources": {}}
            ], "ms": {"gdino": 10}});
            text.push_str(&line.to_string());
            text.push('\n');
            on_line(&format!(
                "@autolabel {{\"event\":\"frame\",\"index\":{},\"frames\":2,\"boxes\":3}}",
                i + 1
            ));
        }
        std::fs::write(out, text).unwrap();
        on_line("@autolabel {\"event\":\"done\",\"frames\":2}");
        Ok(())
    }
}

impl Fetcher for LabelFake {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.inner.fetch(res, dir, progress, cancel)
    }
}

fn config(root: &Path) -> Config {
    let mut c = Config {
        download_dir: Some(root.to_path_buf()),
        allow_large_downloads: true,
        ..Config::default()
    };
    c.seed_default_model();
    c
}

fn run_job(job: &AutolabelJob, fake: &LabelFake) -> Result<TaskOutcome> {
    job.run(
        TaskContext {
            runner: fake,
            fetcher: fake,
        },
        &mut |_| {},
        &AtomicBool::new(false),
    )
}

#[test]
fn labelers_propose_triaged_pending_boxes_and_queue_the_frames() {
    let t = tmp("label");
    let rels = frames(&t.0, 2, true);
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake::default();
    let out = run_job(&job, &fake).unwrap();
    let rep = &out.result;
    assert_eq!(rep["frames"], 2);
    assert_eq!(rep["proposals"], 4, "unicorn is not a class: {rep}");
    assert_eq!(rep["accept"], 2);
    assert_eq!(rep["review"], 2);
    assert_eq!(rep["added"], 4);
    assert_eq!(rep["vlm_objections"], 2);
    // Every labeler checkpoint was fetched (fake) and passed to the script.
    let fetched = fake.inner.fetched.lock().unwrap().clone();
    for id in [
        "autolabel:mm-gdino-tiny",
        "train-base:rfdetr-large",
        "autolabel:qwen3-vl-2b",
        "autolabel:siglip2-base",
    ] {
        assert!(
            fetched.contains(&id.to_string()),
            "{id} not fetched: {fetched:?}"
        );
    }
    let cmd = fake.cmds.lock().unwrap()[0].display();
    for flag in [
        "--gdino",
        "--rfdetr",
        "--vlm",
        "--siglip",
        "--categories",
        "--synonyms",
    ] {
        assert!(cmd.contains(flag), "{flag}: {cmd}");
    }
    assert!(cmd.contains("rf-detr-large-2026.pth"), "{cmd}");
    // The label files: pending auto boxes with their triage; IR frames, so the car is accepted
    // (vehicles are not night-reviewed) and the raccoon reviewed.
    let dir = t.0.join("dataset");
    for rel in &rels {
        let f = Frame::resolve(&dir, rel)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap();
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(f.boxes.len(), 2);
        assert!(
            f.boxes
                .iter()
                .all(|b| b.source == Source::Auto && b.state == BoxState::Pending)
        );
        let car = &f.boxes[0];
        assert_eq!(car.auto.triage, Some(Triage::Accept));
        let rac = &f.boxes[1];
        assert_eq!(rac.auto.triage, Some(Triage::Review));
        assert!(
            rac.auto
                .why
                .iter()
                .any(|w| w.contains("qwen3-vl-2b says rock")),
            "{:?}",
            rac.auto.why
        );
        assert!(rac.auto.why.iter().any(|w| w.contains("at night")));
        assert_eq!(rac.auto.species_scores.get("raccoon"), Some(&0.6));
    }
    // Queued for review once each (the import queued them already).
    let q =
        crate::collect::layout::read_review(&dir.join(crate::collect::layout::REVIEW_QUEUE_FILE));
    assert_eq!(q.len(), 2);
    // A second run proposes the same boxes: all covered, nothing added.
    let job2 =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out2 = run_job(&job2, &fake).unwrap();
    assert_eq!(out2.result["added"], 0);
    assert_eq!(out2.result["dropped"], 4);
    assert!(!job.work.exists());
}

struct FakeDetector(BTreeMap<String, Vec<Det>>);

impl Detector for FakeDetector {
    fn detect(
        &self,
        model: &str,
        images: &[Image],
        _threshold: f32,
        _cancel: &AtomicBool,
    ) -> Result<BTreeMap<String, Vec<Det>>> {
        assert_eq!(model, "IPcam-general");
        Ok(images
            .iter()
            .map(|i| {
                (
                    i.frame.clone(),
                    self.0.get(&i.frame).cloned().unwrap_or_default(),
                )
            })
            .collect())
    }
}

fn det(label: &str, score: f32, bbox: [f32; 4]) -> Det {
    Det {
        label: label.into(),
        score,
        bbox,
    }
}

#[test]
fn self_training_sends_disagreeing_reviewed_frames_back_first() {
    let t = tmp("relabel");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    // Both frames reviewed with one drawn car.
    for rel in &rels {
        let fr = Frame::resolve(&dir, rel).unwrap();
        let mut f = labels::LabelFile {
            status: Status::Reviewed,
            ..Default::default()
        };
        f.boxes.push(labels::LabelBox {
            label: "car".into(),
            bbox: [1.0, 1.0, 30.0, 20.0],
            source: Source::Human,
            score: None,
            state: BoxState::Added,
            origin: None,
            reason: None,
            model_label: None,
            auto: Default::default(),
        });
        labels::write(&fr.labels, &f).unwrap();
    }
    let cfg = config(&t.0);
    let mut req =
        AutolabelRequest::new(FrameFilter::parse("", super::super::default_status(true)).unwrap());
    req.with_model = Some("IPcam-general".into());
    // No GPU needed for a re-label with a configured model.
    let none = capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
    let mut job = plan_autolabel(&cfg, req, &none).unwrap();
    // Frame 0: the model agrees. Frame 1: a truck where the car is, and a new dog.
    let dets: BTreeMap<String, Vec<Det>> = [
        (
            rels[0].clone(),
            vec![det("car", 0.9, [1.0, 1.0, 30.0, 21.0])],
        ),
        (
            rels[1].clone(),
            vec![
                det("truck", 0.7, [1.0, 1.0, 30.0, 20.0]),
                det("dog", 0.6, [35.0, 20.0, 60.0, 46.0]),
            ],
        ),
    ]
    .into();
    job.detector = Some(Arc::new(FakeDetector(dets)));
    let fake = LabelFake::default();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["disagreements"], 1, "{}", out.result);
    assert!(fake.cmds.lock().unwrap().is_empty(), "no Python ran");
    assert_eq!(fake.inner.runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    let f0 = Frame::resolve(&dir, &rels[0])
        .unwrap()
        .labels()
        .unwrap()
        .unwrap();
    assert_eq!(f0.status, Status::Reviewed);
    assert_eq!(f0.disagreement, None);
    let f1 = Frame::resolve(&dir, &rels[1])
        .unwrap()
        .labels()
        .unwrap()
        .unwrap();
    assert_eq!(f1.status, Status::Unreviewed);
    assert_eq!(
        f1.disagreement.as_deref(),
        Some("re-label by IPcam-general: 1 new box, 1 class change")
    );
    // The Review queue lists it first.
    let q = queue::build(
        &dir,
        &queue::Filters::from_query(&[("status".to_string(), "any".to_string())].into()).unwrap(),
    );
    assert_eq!(q.items[0].frame, rels[1]);
    assert!(q.items[0].uncertainty.relabel.is_some());
}

#[test]
fn planning_refuses_unknown_models_missing_gpus_and_empty_selections() {
    let t = tmp("plan");
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("nope".into());
    assert!(
        plan_autolabel(&cfg, req, &host())
            .unwrap_err()
            .to_string()
            .contains("no configured model")
    );
    let none = capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
    let e = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &none)
        .unwrap_err()
        .to_string();
    assert!(e.contains("--with-model"), "{e}");
    let mut small = cfg.clone();
    small.allow_large_downloads = false;
    let e = plan_autolabel(
        &small,
        AutolabelRequest::new(FrameFilter::default()),
        &host(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("--allow-large"), "{e}");
    // No frames: the run fails with a hint.
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let e = run_job(&job, &LabelFake::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("no frame matches"), "{e}");
}
