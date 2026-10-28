//! The auto-label stage machine over fakes: the export's [`Fake`] for uv and downloads, a fake
//! `autolabel.py` that proposes boxes for every frame it is given, and a fake [`Detector`] for
//! the self-training loop.

use super::*;
use crate::backend::detect::HardwareInfo;
use crate::collect::labels::{BoxState, Source, Status};
use crate::config::ModelConfig;
use crate::resources::catalog::Platform;
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::{Cmd, CommandRunner, Fetcher};
use crate::training::autolabel::models::{ConfigWriter, Install, Target};
use crate::training::autolabel::triage::Det;
use crate::training::autolabel::{FrameFilter, plan_autolabel};
use crate::training::capability::{self, Capability};
use crate::training::service::RunState;
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
    /// Frames the fake labelers find nothing in.
    empty: Vec<String>,
    /// Cancel after writing this many proposal lines, the last one cut off mid-line (as the
    /// real runner does: the flag set, the script killed, `Cancelled`).
    stop_after: Option<usize>,
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
            if let Some(k) = self.stop_after
                && i + 1 == k
            {
                // Killed while writing line k.
                let line = serde_json::json!({"frame": "x", "boxes": []}).to_string();
                text.push_str(&line[..line.len() / 2]);
                std::fs::write(&out, &text).unwrap();
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(anyhow!(crate::resources::export::Cancelled));
            }
            let f: serde_json::Value = serde_json::from_str(l).unwrap();
            if self.empty.iter().any(|e| f["frame"] == e.as_str()) {
                let line = serde_json::json!({"frame": f["frame"], "boxes": []});
                text.push_str(&line.to_string());
                text.push('\n');
                continue;
            }
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
        let run = f.auto_run.as_ref().unwrap();
        assert_eq!(run.labeler, rep["labeler"].as_str().unwrap());
        assert!(run.at.starts_with("20"), "{run:?}");
    }
    // Queued for review once each (the import queued them already).
    let q =
        crate::collect::layout::read_review(&dir.join(crate::collect::layout::REVIEW_QUEUE_FILE));
    assert_eq!(q.len(), 2);
    // A second run of the same labelers skips the frames they looked at.
    let job2 =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let e = run_job(&job2, &fake).unwrap_err().to_string();
    assert!(
        e.contains("all 2 frame(s)") && e.contains("again=1") && e.contains("Label again"),
        "{e}"
    );
    // Another set of labelers is a new run.
    let mut other = AutolabelRequest::new(FrameFilter::default());
    other.siglip = false;
    let job3 = plan_autolabel(&cfg, other, &host()).unwrap();
    let out3 = run_job(&job3, &fake).unwrap();
    assert_eq!(
        (
            out3.result["frames"].as_u64(),
            out3.result["skipped_done"].as_u64()
        ),
        (Some(2), Some(0))
    );
    // again=1: the same boxes, all covered, nothing added.
    let again = FrameFilter::parse("again=1", "unreviewed").unwrap();
    let job4 = plan_autolabel(&cfg, AutolabelRequest::new(again), &host()).unwrap();
    let out4 = run_job(&job4, &fake).unwrap();
    assert_eq!(out4.result["added"], 0);
    assert_eq!(out4.result["dropped"], 4);
    assert_eq!(out4.result["frames"], 2);
    assert!(!job.work.exists());
}

#[test]
fn hidden_classes_are_neither_labelled_nor_left_pending() {
    let t = tmp("hidden");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    // Frame 0 was pre-filled before `raccoon` was hidden: a pending model raccoon.
    let fr = Frame::resolve(&dir, &rels[0]).unwrap();
    let raccoon = labels::LabelBox {
        label: "raccoon".into(),
        bbox: [35.0, 20.0, 60.0, 46.0],
        source: Source::Model,
        score: Some(0.6),
        state: BoxState::Pending,
        origin: Some(labels::Origin::Prediction),
        reason: None,
        model_label: None,
        orig_bbox: None,
        auto: Default::default(),
    };
    let f = LabelFile {
        boxes: vec![raccoon],
        ..LabelFile::default()
    };
    labels::write(&fr.labels, &f).unwrap();
    let mut cfg = config(&t.0);
    cfg.collect.hidden_classes = vec!["raccoon".into()];
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    assert!(
        !job.classes.iter().any(|c| c == "raccoon"),
        "{:?}",
        job.classes
    );
    let fake = LabelFake::default();
    run_job(&job, &fake).unwrap();
    let cmd = fake.cmds.lock().unwrap()[0].clone();
    let classes = arg_after(&cmd, "--classes").unwrap();
    assert!(
        !classes.to_string_lossy().contains("raccoon"),
        "{classes:?}"
    );
    for rel in &rels {
        let l = Frame::resolve(&dir, rel)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap();
        assert_eq!(l.status, Status::Unreviewed);
        let pending: Vec<&str> = l
            .boxes
            .iter()
            .filter(|b| b.state == BoxState::Pending)
            .map(|b| b.label.as_str())
            .collect();
        assert_eq!(pending, ["car"], "{rel}: {:?}", l.boxes);
    }
    let f0 = fr.labels().unwrap().unwrap();
    assert_eq!(
        (f0.boxes[0].state, f0.boxes[0].reason.as_deref()),
        (BoxState::Rejected, Some(labels::bulk::OTHER_CLASS_REASON))
    );
}

#[test]
fn empty_frames_get_the_marker_and_other_class_boxes_a_suggestion() {
    let t = tmp("marker");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    // Frame 0: the serving model saw a dog where the labelers find a car.
    let fr = Frame::resolve(&dir, &rels[0]).unwrap();
    let mut f = labels::prefill(
        &fr.record().unwrap(),
        &crate::collect::config::default_classes(),
        &[],
    );
    f.boxes.push(labels::LabelBox {
        label: "dog".into(),
        bbox: [1.0, 1.0, 30.0, 20.0],
        source: Source::Model,
        score: Some(0.6),
        state: BoxState::Pending,
        origin: Some(labels::Origin::Prediction),
        reason: None,
        model_label: None,
        orig_bbox: None,
        auto: Default::default(),
    });
    labels::write(&fr.labels, &f).unwrap();
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake {
        empty: vec![rels[2].clone()],
        ..LabelFake::default()
    };
    let out = run_job(&job, &fake).unwrap();
    let rep = &out.result;
    assert_eq!(
        (rep["frames"].as_u64(), rep["with_boxes"].as_u64()),
        (Some(3), Some(2)),
        "{rep}"
    );
    assert_eq!(rep["class_suggestions"], 1, "{rep}");
    assert!(
        out.summary.contains("1 class suggestions"),
        "{}",
        out.summary
    );
    let f0 = fr.labels().unwrap().unwrap();
    let dog = &f0.boxes[0];
    assert_eq!(
        (dog.label.as_str(), dog.auto.auto_label.as_deref()),
        ("dog", Some("car"))
    );
    assert_eq!(dog.auto.triage, Some(Triage::Review));
    assert_eq!(
        f0.boxes.len(),
        2,
        "the car is not added twice; the raccoon is"
    );
    // The empty frame: a label file with the marker and no boxes; not queued as changed.
    let f2 = Frame::resolve(&dir, &rels[2])
        .unwrap()
        .labels()
        .unwrap()
        .unwrap();
    assert!(f2.boxes.is_empty());
    assert_eq!(
        f2.auto_run.as_ref().unwrap().labeler,
        rep["labeler"].as_str().unwrap()
    );
    let sample: Vec<&str> = rep["sample"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert!(!sample.contains(&rels[2].as_str()), "{sample:?}");
    assert_eq!(sample.len(), 2);
    let q = |auto: &str| {
        let mut v: Vec<String> = queue::build(
            &dir,
            &queue::Filters::from_query(
                &[
                    ("source".to_string(), "all".to_string()),
                    ("auto".to_string(), auto.to_string()),
                ]
                .into(),
            )
            .unwrap(),
        )
        .items
        .into_iter()
        .map(|i| i.frame)
        .collect();
        v.sort();
        v
    };
    assert_eq!(q("nothing"), [rels[2].clone()]);
    assert_eq!(q("review").len(), 2);
    assert!(q("none").is_empty());
    // Everything was looked at: a second run says so.
    let job2 =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let e = run_job(&job2, &fake).unwrap_err().to_string();
    assert!(e.contains("all 3 frame(s)"), "{e}");
    // Only the frames never auto-labelled: none left.
    let never = FrameFilter::parse("auto=none,again=1", "unreviewed").unwrap();
    let job3 = plan_autolabel(&cfg, AutolabelRequest::new(never), &host()).unwrap();
    let e = run_job(&job3, &fake).unwrap_err().to_string();
    assert!(e.contains("no frame matches"), "{e}");
}

/// Detections per frame; records the model, device and config it was asked for.
#[derive(Default)]
struct FakeDetector {
    dets: BTreeMap<String, Vec<Det>>,
    calls: Mutex<Vec<(String, Option<String>, bool)>>,
    /// Cancel after this many frames (the flag set, the finished frames returned).
    stop_after: Option<usize>,
}

impl FakeDetector {
    fn new(dets: BTreeMap<String, Vec<Det>>) -> Self {
        Self {
            dets,
            calls: Mutex::default(),
            stop_after: None,
        }
    }
}

impl Detector for FakeDetector {
    fn detect(
        &self,
        config: &Config,
        model: &ModelConfig,
        device: Option<&str>,
        images: &[Image],
        _threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<super::super::relabel::Detections> {
        let name = model.effective_name();
        let n = self.stop_after.unwrap_or(images.len()).min(images.len());
        if self.stop_after.is_some() {
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        assert!(
            config.models.iter().any(|m| m.effective_name() == name),
            "the entry is in the config the detector gets"
        );
        self.calls
            .lock()
            .unwrap()
            .push((name, device.map(str::to_string), model.enabled));
        Ok(super::super::relabel::Detections {
            frames: images[..n]
                .iter()
                .map(|i| {
                    (
                        i.frame.clone(),
                        self.dets.get(&i.frame).cloned().unwrap_or_default(),
                    )
                })
                .collect(),
            device: device.unwrap_or("openvino:cpu").to_string(),
            cancelled: self.stop_after.is_some(),
        })
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
            orig_bbox: None,
            auto: Default::default(),
        });
        labels::write(&fr.labels, &f).unwrap();
    }
    let cfg = config(&t.0);
    // Self-training: an explicit status=any (the default is unreviewed frames only).
    let mut req = AutolabelRequest::new(FrameFilter::parse("status=any", "unreviewed").unwrap());
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
    job.detector = Some(Arc::new(FakeDetector::new(dets)));
    let fake = LabelFake::default();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["disagreements"], 1, "{}", out.result);
    assert_eq!(out.result["labeler"], "model:IPcam-general@auto");
    assert_eq!(out.result["requested_device"], "auto");
    assert!(fake.cmds.lock().unwrap().is_empty(), "no Python ran");
    assert_eq!(fake.inner.runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    let f0 = Frame::resolve(&dir, &rels[0])
        .unwrap()
        .labels()
        .unwrap()
        .unwrap();
    assert_eq!(f0.status, Status::Reviewed);
    assert_eq!(f0.disagreement, None);
    assert_eq!(
        f0.auto_run.as_ref().map(|r| r.labeler.as_str()),
        Some("model:IPcam-general@auto"),
        "every frame the model looked at gets the marker"
    );
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
    let e = plan_autolabel(&cfg, req, &host()).unwrap_err().to_string();
    assert!(e.contains("not a configured model"), "{e}");
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

/// Two NVIDIA GPUs on Linux: an RTX 3090 (24 GB, CUDA 12.6 build, the automatic pick) and an
/// RTX 5060 (8 GB, Blackwell: CUDA 13.0 build).
fn two_gpus() -> Capability {
    use crate::backend::detect::{ComputeCapability, CudaInfo, GpuAdapter, GpuVendor};
    let gpu = |name: &str, cc: (u32, u32), vram_mb: u64, ordinal: u32| GpuAdapter {
        vendor: GpuVendor::Nvidia,
        name: name.into(),
        vram_mb,
        index: ordinal,
        discrete: true,
        cuda: Some(CudaInfo {
            ordinal,
            compute_capability: ComputeCapability::new(cc.0, cc.1),
        }),
    };
    capability::capability(
        &HardwareInfo::new(
            "linux",
            "x86_64",
            vec![
                gpu("NVIDIA GeForce RTX 3090", (8, 6), 24576, 0),
                gpu("NVIDIA GeForce RTX 5060", (12, 0), 8192, 1),
            ],
        )
        .with_cuda_driver(13, 0),
        None,
    )
}

#[test]
fn the_labelers_run_on_the_chosen_gpu_in_its_lanes_environment() {
    let t = tmp("gpu");
    frames(&t.0, 1, false);
    let cfg = config(&t.0);
    let cap = two_gpus();
    let run = |gpu: Option<&str>| {
        let mut req = AutolabelRequest::new(FrameFilter::parse("again=1", "unreviewed").unwrap());
        req.gpu = gpu.map(str::to_string);
        let mut job = plan_autolabel(&cfg, req, &cap).unwrap();
        // The environment of the planned GPU's platform (the tests may run on a Mac).
        job.platform = Platform::LINUX_X64;
        let fake = LabelFake::default();
        let out = run_job(&job, &fake).unwrap();
        let cmd = fake.cmds.lock().unwrap().last().unwrap().clone();
        let pip = fake
            .inner
            .cmds()
            .into_iter()
            .find(|c| c.contains(" pip install "))
            .unwrap();
        (job, out, cmd, pip)
    };
    let (job, out, cmd, pip) = run(Some("CUDA:1"));
    assert_eq!(job.request.gpu.as_deref(), Some("cuda:1"), "canonical");
    assert_eq!(
        job.title(),
        "auto-label on cuda:1 (again=1, source=all, status=unreviewed, limit 200)"
    );
    assert_eq!(
        arg_after(&cmd, "--device").unwrap().to_str(),
        Some("cuda:1")
    );
    assert!(pip.contains("--torch-backend cu130"), "{pip}");
    assert_eq!(out.result["device"], "cuda:1");
    // Automatic: the largest GPU and its lane; the title names no device.
    let (job, out, cmd, pip) = run(None);
    assert_eq!(job.request.gpu, None);
    assert!(job.title().starts_with("auto-label ("), "{}", job.title());
    assert_eq!(
        arg_after(&cmd, "--device").unwrap().to_str(),
        Some("cuda:0")
    );
    assert!(pip.contains("--torch-backend cu126"), "{pip}");
    assert_eq!(out.result["device"], "cuda:0");
}

#[test]
fn a_gpu_choice_is_checked_when_planned() {
    let t = tmp("gpu-plan");
    let cfg = config(&t.0);
    let plan = |gpu: &str, cap: &Capability| {
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.gpu = Some(gpu.into());
        plan_autolabel(&cfg, req, cap).map(|j| j.request.gpu)
    };
    let e = plan("cuda:2", &two_gpus()).unwrap_err().to_string();
    assert!(
        e.contains("no such CUDA device") && e.contains("RTX 5060 (cuda:1"),
        "{e}"
    );
    let e = plan("gpu1", &two_gpus()).unwrap_err().to_string();
    assert!(e.contains("is not cuda:N"), "{e}");
    let mac = capability::capability(
        &HardwareInfo::new(
            "macos",
            "aarch64",
            vec![crate::backend::detect::apple_gpu("Apple M1")],
        ),
        Some(16.0),
    );
    let e = plan("cuda:0", &mac).unwrap_err().to_string();
    assert!(e.contains("no GPU to choose"), "{e}");
    assert_eq!(plan("mps", &mac).unwrap().as_deref(), Some("mps"));
    assert_eq!(plan("auto", &two_gpus()).unwrap(), None);
    // A run with a model takes a device, not a GPU.
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("IPcam-general".into());
    req.gpu = Some("cuda:1".into());
    let e = plan_autolabel(&cfg, req, &two_gpus())
        .unwrap_err()
        .to_string();
    assert!(e.contains("--device"), "{e}");
}

#[test]
fn a_run_for_a_class_the_last_run_did_not_look_for_labels_the_frames_again() {
    let t = tmp("subset");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let mut cfg = config(&t.0);
    cfg.collect.classes = vec!["person".into(), "car".into(), "raccoon".into()];
    // A run for car only: the marker names it.
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.classes = Some(vec!["CAR".into()]);
    let job = plan_autolabel(&cfg, req.clone(), &host()).unwrap();
    assert_eq!(job.classes, ["car"], "the list's spelling");
    let fake = LabelFake::default();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["frames"], 2);
    assert_eq!(
        out.result["by_label"],
        serde_json::json!({"car": 2}),
        "car only"
    );
    let marker = |rel: &str| {
        Frame::resolve(&dir, rel)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap()
            .auto_run
            .unwrap()
    };
    assert_eq!(marker(&rels[0]).classes, ["car"]);
    // car again: covered, skipped.
    let e = run_job(&plan_autolabel(&cfg, req, &host()).unwrap(), &fake)
        .unwrap_err()
        .to_string();
    assert!(e.contains("all 2 frame(s)"), "{e}");
    // The whole list: raccoon and person were not looked for, so the frames are labelled.
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(
        (
            out.result["frames"].as_u64(),
            out.result["skipped_done"].as_u64()
        ),
        (Some(2), Some(0))
    );
    assert_eq!(marker(&rels[1]).classes, ["person", "car", "raccoon"]);
    // Now a subset is covered by the full run.
    let mut sub = AutolabelRequest::new(FrameFilter::default());
    sub.classes = Some(vec!["raccoon".into()]);
    let e = run_job(&plan_autolabel(&cfg, sub, &host()).unwrap(), &fake)
        .unwrap_err()
        .to_string();
    assert!(e.contains("all 2 frame(s)"), "{e}");
    // A marker of an old file (no classes) covers every class.
    let fr = Frame::resolve(&dir, &rels[0]).unwrap();
    labels::update(&fr.labels, |f| {
        let mut f = f.unwrap();
        f.auto_run.as_mut().unwrap().classes.clear();
        Ok(Some(f))
    })
    .unwrap();
    cfg.collect.classes.push("package".into());
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(
        (
            out.result["frames"].as_u64(),
            out.result["skipped_done"].as_u64()
        ),
        (Some(1), Some(1)),
        "the old marker covers package; the other frame did not look for it"
    );
    // A class outside collect.classes is refused.
    let mut bad = AutolabelRequest::new(FrameFilter::default());
    bad.classes = Some(vec!["unicorn".into()]);
    let e = plan_autolabel(&cfg, bad, &host()).unwrap_err().to_string();
    assert!(
        e.contains("unicorn") && e.contains("collect.classes"),
        "{e}"
    );
}

#[test]
fn self_training_for_some_classes_ignores_the_others() {
    let t = tmp("relabel-subset");
    let rels = frames(&t.0, 1, false);
    let dir = t.0.join("dataset");
    let fr = Frame::resolve(&dir, &rels[0]).unwrap();
    let mut f = labels::LabelFile {
        status: Status::Reviewed,
        ..Default::default()
    };
    // A reviewed dog the model will not see.
    f.boxes.push(labels::LabelBox {
        label: "dog".into(),
        bbox: [35.0, 20.0, 60.0, 46.0],
        source: Source::Human,
        score: None,
        state: BoxState::Added,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        auto: Default::default(),
    });
    labels::write(&fr.labels, &f).unwrap();
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::parse("status=any", "unreviewed").unwrap());
    req.with_model = Some("IPcam-general".into());
    req.classes = Some(vec!["car".into()]);
    let none = capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
    let mut job = plan_autolabel(&cfg, req, &none).unwrap();
    // The model sees a person (not in the run) and nothing else.
    let dets: BTreeMap<String, Vec<Det>> = [(
        rels[0].clone(),
        vec![det("person", 0.9, [1.0, 1.0, 30.0, 20.0])],
    )]
    .into();
    job.detector = Some(Arc::new(FakeDetector::new(dets)));
    let out = run_job(&job, &LabelFake::default()).unwrap();
    assert_eq!(out.result["disagreements"], 0, "{}", out.result);
    assert_eq!(
        out.result["dropped_other_class"], 1,
        "the person is counted"
    );
    let f = fr.labels().unwrap().unwrap();
    assert_eq!(f.status, Status::Reviewed);
    assert_eq!(f.disagreement, None);
    assert_eq!(f.boxes.len(), 1, "the person is not proposed");
}

/// The config file of a run that adds a model: kept in memory.
struct MemWriter(Mutex<Config>);

impl ConfigWriter for MemWriter {
    fn update(&self, edit: &mut dyn FnMut(&mut Config) -> Result<()>) -> Result<(Config, String)> {
        let mut c = self.0.lock().unwrap();
        edit(&mut c)?;
        Ok((c.clone(), "saved (test)".into()))
    }
}

fn no_gpu() -> Capability {
    capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None)
}

fn labels_of(dir: &Path, rel: &str) -> labels::LabelFile {
    Frame::resolve(dir, rel).unwrap().labels().unwrap().unwrap()
}

#[test]
fn model_runs_mark_their_frames_map_labels_and_pre_accept_sure_boxes() {
    let t = tmp("model-mark");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let dets: BTreeMap<String, Vec<Det>> = [
        (
            rels[0].clone(),
            vec![
                det("car", 0.9, [1.0, 1.0, 30.0, 20.0]),
                det("motorbike", 0.85, [35.0, 20.0, 60.0, 46.0]),
                det("tvmonitor", 0.95, [2.0, 25.0, 20.0, 46.0]),
            ],
        ),
        (
            rels[1].clone(),
            vec![det("car", 0.5, [1.0, 1.0, 30.0, 20.0])],
        ),
    ]
    .into();
    let detector = Arc::new(FakeDetector::new(dets));
    let plan = |filter: &str, device: Option<&str>| {
        let mut req = AutolabelRequest::new(FrameFilter::parse(filter, "unreviewed").unwrap());
        req.with_model = Some("IPcam-general".into());
        req.model_accept_score = Some(0.8);
        req.device = device.map(str::to_string);
        let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        job.detector = Some(detector.clone());
        job
    };
    let job = plan("", None);
    assert_eq!(job.steps(), MODEL_STAGES.len());
    assert!(
        job.title().starts_with("label with IPcam-general on auto"),
        "{}",
        job.title()
    );
    let out = run_job(&job, &LabelFake::default()).unwrap();
    let rep = &out.result;
    assert_eq!(
        (
            rep["proposals"].as_u64(),
            rep["accept"].as_u64(),
            rep["review"].as_u64()
        ),
        (Some(3), Some(2), Some(1)),
        "{rep}"
    );
    assert_eq!(rep["dropped_other_class"], 1, "the tv: {rep}");
    assert_eq!(
        rep["renamed"],
        serde_json::json!({"motorbike": "motorcycle"})
    );
    assert_eq!(
        rep["by_label"],
        serde_json::json!({"car": 2, "motorcycle": 1})
    );
    assert_eq!(rep["labeler"], "model:IPcam-general@auto");
    assert_eq!(rep["device"], "openvino:cpu", "the device that ran");
    assert!(
        out.summary.contains("2 accept, 1 review"),
        "{}",
        out.summary
    );
    let f0 = labels_of(&dir, &rels[0]);
    assert_eq!(f0.boxes.len(), 2);
    assert!(
        f0.boxes
            .iter()
            .all(|b| b.auto.triage == Some(Triage::Accept))
    );
    let moto = &f0.boxes[1];
    assert_eq!(
        (moto.label.as_str(), moto.model_label.as_deref()),
        ("motorcycle", Some("motorbike"))
    );
    assert_eq!(moto.auto.why, ["model IPcam-general at 0.85 >= 0.80"]);
    assert!(
        labels::auto_accept_eligible(&f0),
        "the bulk accept takes it"
    );
    let f1 = labels_of(&dir, &rels[1]);
    assert_eq!(f1.boxes[0].auto.triage, Some(Triage::Review));
    assert!(!labels::auto_accept_eligible(&f1));
    for f in [&f0, &f1] {
        let m = f.auto_run.as_ref().unwrap();
        assert_eq!(m.labeler, "model:IPcam-general@auto");
        assert_eq!(m.classes, cfg.collect.classes);
    }
    // `auto=` filters see the marker.
    let q = queue::build(
        &dir,
        &queue::Filters::from_query(
            &[
                ("source".to_string(), "all".to_string()),
                ("auto".to_string(), "accept".to_string()),
            ]
            .into(),
        )
        .unwrap(),
    );
    assert_eq!(
        q.items.iter().map(|i| i.frame.clone()).collect::<Vec<_>>(),
        [rels[0].clone()]
    );
    // The same model on the same device: skipped unless again=1.
    let e = run_job(&plan("", None), &LabelFake::default())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("all 2 frame(s)") && e.contains("model:IPcam-general@auto"),
        "{e}"
    );
    let again = run_job(&plan("again=1", None), &LabelFake::default()).unwrap();
    assert_eq!(
        (
            again.result["frames"].as_u64(),
            again.result["added"].as_u64()
        ),
        (Some(2), Some(0))
    );
    // Another device is another labeler; the detector gets the run's device.
    let other = plan("", Some("CPU"));
    assert!(
        other.title().contains("on openvino:cpu"),
        "{}",
        other.title()
    );
    let out = run_job(&other, &LabelFake::default()).unwrap();
    assert_eq!(out.result["frames"], 2);
    assert_eq!(out.result["labeler"], "model:IPcam-general@openvino:cpu");
    assert_eq!(out.result["requested_device"], "openvino:cpu");
    let calls = detector.calls.lock().unwrap().clone();
    assert_eq!(calls.last().unwrap().1.as_deref(), Some("openvino:cpu"));
    assert_eq!(calls[0].1, None, "no override: the model's own device");
}

#[test]
fn night_person_boxes_of_a_model_stay_in_review() {
    let t = tmp("model-night");
    let rels = frames(&t.0, 1, true);
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("IPcam-general".into());
    req.model_accept_score = Some(0.5);
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    job.detector = Some(Arc::new(FakeDetector::new(
        [(
            rels[0].clone(),
            vec![
                det("person", 0.97, [1.0, 1.0, 30.0, 40.0]),
                det("car", 0.97, [35.0, 20.0, 60.0, 46.0]),
            ],
        )]
        .into(),
    )));
    let out = run_job(&job, &LabelFake::default()).unwrap();
    assert_eq!(
        (out.result["accept"].as_u64(), out.result["review"].as_u64()),
        (Some(1), Some(1))
    );
    let f = labels_of(&t.0.join("dataset"), &rels[0]);
    let person = f.boxes.iter().find(|b| b.label == "person").unwrap();
    assert_eq!(person.auto.triage, Some(Triage::Review));
    assert!(
        person.auto.why.iter().any(|w| w.contains("at night")),
        "{:?}",
        person.auto.why
    );
}

#[test]
fn a_catalog_model_is_downloaded_added_disabled_and_used() {
    let t = tmp("model-catalog");
    let rels = frames(&t.0, 1, false);
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("resource:model:rfdetr-base".into());
    let mut job = plan_autolabel(&cfg, req.clone(), &no_gpu()).unwrap();
    assert_eq!(job.model.as_ref().unwrap().install, Some(Install::Download));
    let det = Arc::new(FakeDetector::new(
        [(
            rels[0].clone(),
            vec![det("car", 0.9, [1.0, 1.0, 30.0, 20.0])],
        )]
        .into(),
    ));
    job.detector = Some(det.clone());
    let writer = Arc::new(MemWriter(Mutex::new(cfg.clone())));
    job.config_writer = Some(writer.clone());
    let fake = LabelFake::default();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(
        *fake.inner.fetched.lock().unwrap(),
        ["model:rfdetr-base".to_string()]
    );
    assert!(
        out.result["installed"]
            .as_str()
            .unwrap()
            .starts_with("downloaded RF-DETR base"),
        "{}",
        out.result
    );
    assert!(
        out.result["config_entry"]
            .as_str()
            .unwrap()
            .contains("disabled")
    );
    assert_eq!(out.result["added"], 1);
    let saved = writer.0.lock().unwrap().clone();
    let m = saved
        .models
        .iter()
        .find(|m| m.effective_name() == "rfdetr-base")
        .unwrap();
    assert!(!m.enabled, "never loaded for serving");
    assert_eq!(
        det.calls.lock().unwrap()[0],
        ("rfdetr-base".to_string(), None, false)
    );
    assert!(t.0.join("models/rfdetr-base.onnx").is_file());
    // Next time it is a configured model: nothing to install.
    let again = plan_autolabel(&saved, req, &no_gpu()).unwrap();
    let p = again.model.as_ref().unwrap();
    assert!(matches!(p.target, Target::Config { .. }) && p.install.is_none());
}

#[test]
fn an_export_model_is_exported_first_and_gated() {
    let t = tmp("model-export");
    let rels = frames(&t.0, 1, false);
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("resource:model:yolo26n".into());
    let p = Platform::current();
    if let Some(why) = crate::resources::export::unsupported_reason(p.os, p.arch) {
        let e = plan_autolabel(&cfg, req, &no_gpu())
            .unwrap_err()
            .to_string();
        assert!(e.contains(&why), "{e}");
        return;
    }
    // Without the large-download permission the toolchain setup is refused.
    let mut small = cfg.clone();
    small.allow_large_downloads = false;
    let e = plan_autolabel(&small, req.clone(), &no_gpu())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("--allow-large") && e.contains("export toolchain"),
        "{e}"
    );
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    assert_eq!(job.model.as_ref().unwrap().install, Some(Install::Export));
    job.detector = Some(Arc::new(FakeDetector::new(
        [(
            rels[0].clone(),
            vec![det("person", 0.9, [1.0, 1.0, 30.0, 40.0])],
        )]
        .into(),
    )));
    let writer = Arc::new(MemWriter(Mutex::new(cfg.clone())));
    job.config_writer = Some(writer.clone());
    let fake = LabelFake {
        inner: Fake::new(),
        ..LabelFake::default()
    };
    let mut stages: Vec<String> = Vec::new();
    let out = job
        .run(
            TaskContext {
                runner: &fake,
                fetcher: &fake,
            },
            &mut |r| {
                if let TaskReport::Stage { stage, detail, .. } = r {
                    stages.push(format!("{stage}: {detail}"));
                }
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    assert!(
        fake.inner
            .cmds()
            .iter()
            .any(|c| c.contains("export_model.py")),
        "{:?}",
        fake.inner.cmds()
    );
    assert!(
        stages
            .iter()
            .any(|s| s.starts_with("installing the model: exporting yolo26n")),
        "{stages:?}"
    );
    assert!(
        out.result["installed"]
            .as_str()
            .unwrap()
            .contains("exported yolo26n")
    );
    assert!(t.0.join("models/yolo26n.onnx").is_file());
    let saved = writer.0.lock().unwrap().clone();
    assert!(
        saved
            .models
            .iter()
            .any(|m| m.effective_name() == "yolo26n" && !m.enabled)
    );
}

#[test]
fn restricted_licences_need_a_confirmation() {
    let t = tmp("model-licence");
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("resource:model:deimv2-s".into());
    let p = Platform::current();
    let e = plan_autolabel(&cfg, req.clone(), &no_gpu())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("--confirm-license") && e.contains("Non-commercial"),
        "{e}"
    );
    req.confirm_license = true;
    let r = plan_autolabel(&cfg, req, &no_gpu());
    if crate::resources::export::unsupported_reason(p.os, p.arch).is_none() {
        assert!(r.is_ok(), "{:?}", r.err());
    }
}

#[test]
fn a_local_model_file_is_added_with_family_auto() {
    let t = tmp("model-local");
    let rels = frames(&t.0, 1, false);
    std::fs::create_dir_all(t.0.join("models")).unwrap();
    std::fs::write(t.0.join("models/mine.onnx"), b"onnx").unwrap();
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("local:mine.onnx".into());
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    job.detector = Some(Arc::new(FakeDetector::new(
        [(
            rels[0].clone(),
            vec![det("Car", 0.9, [1.0, 1.0, 30.0, 20.0])],
        )]
        .into(),
    )));
    // No writer: the run uses the entry, says it was not saved.
    let out = run_job(&job, &LabelFake::default()).unwrap();
    assert_eq!(out.result["by_label"], serde_json::json!({"car": 1}));
    assert!(
        out.result["warnings"][0]
            .as_str()
            .unwrap()
            .contains("not saved"),
        "{}",
        out.result
    );
    let f = labels_of(&t.0.join("dataset"), &rels[0]);
    assert_eq!(f.boxes[0].model_label, None, "a case-only change");
    assert_eq!(f.auto_run.unwrap().labeler, "model:mine@auto");
}

#[test]
fn a_failing_auto_family_names_the_fix() {
    struct Broken;
    impl Detector for Broken {
        fn detect(
            &self,
            _: &Config,
            _: &ModelConfig,
            _: Option<&str>,
            _: &[Image],
            _: f32,
            _: &AtomicBool,
        ) -> Result<super::super::relabel::Detections> {
            bail!("could not auto-detect the model family from its outputs")
        }
    }
    let t = tmp("model-broken");
    frames(&t.0, 1, false);
    std::fs::create_dir_all(t.0.join("models")).unwrap();
    std::fs::write(t.0.join("models/odd.onnx"), b"onnx").unwrap();
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.with_model = Some("local:odd.onnx".into());
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    job.detector = Some(Arc::new(Broken));
    let e = format!("{:#}", run_job(&job, &LabelFake::default()).unwrap_err());
    assert!(
        e.contains("family auto") && e.contains("Models card"),
        "{e}"
    );
}

fn run_cancellable(
    job: &AutolabelJob,
    fake: &LabelFake,
    cancel: &AtomicBool,
) -> Result<TaskOutcome> {
    job.run(
        TaskContext {
            runner: fake,
            fetcher: fake,
        },
        &mut |_| {},
        cancel,
    )
}

#[test]
fn script_flags_follow_the_run_options() {
    let t = tmp("flags");
    frames(&t.0, 1, false);
    let cfg = config(&t.0);
    let fake = LabelFake::default();
    let argv = |edit: &dyn Fn(&mut AutolabelRequest)| -> Vec<String> {
        let mut req = AutolabelRequest::new(FrameFilter::parse("again=1", "unreviewed").unwrap());
        edit(&mut req);
        let job = plan_autolabel(&cfg, req, &host()).unwrap();
        run_job(&job, &fake).unwrap();
        let cmd = fake.cmds.lock().unwrap().last().unwrap().clone();
        cmd.args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    };
    let after = |a: &[String], flag: &str| {
        a.iter()
            .position(|x| x == flag)
            .and_then(|i| a.get(i + 1).cloned())
    };
    let has = |a: &[String], flag: &str| a.iter().any(|x| x == flag);
    // The defaults: batches of 8, every box checked, the next frame prefetched.
    let a = argv(&|_| {});
    assert_eq!(after(&a, "--vlm-batch").as_deref(), Some("8"), "{a:?}");
    assert!(
        !has(&a, "--vlm-min-score") && has(&a, "--prefetch"),
        "{a:?}"
    );
    // Everything off.
    let a = argv(&|r| {
        r.vlm_batch = false;
        r.prefetch = false;
    });
    assert_eq!(after(&a, "--vlm-batch").as_deref(), Some("1"), "{a:?}");
    assert!(
        !has(&a, "--vlm-min-score") && !has(&a, "--prefetch"),
        "{a:?}"
    );
    // A minimum score.
    let a = argv(&|r| r.vlm_min_score = Some(0.2));
    assert_eq!(
        after(&a, "--vlm-min-score").as_deref(),
        Some("0.2"),
        "{a:?}"
    );
    assert_eq!(after(&a, "--vlm-batch").as_deref(), Some("8"), "{a:?}");
    // No VLM: no VLM flags, whatever the options say.
    let a = argv(&|r| {
        r.vlm = None;
        r.vlm_min_score = Some(0.5);
    });
    assert!(
        !has(&a, "--vlm") && !has(&a, "--vlm-batch") && !has(&a, "--vlm-min-score"),
        "{a:?}"
    );
    assert!(has(&a, "--prefetch"), "{a:?}");
}

#[test]
fn a_cancelled_labelers_run_keeps_the_finished_frames_and_a_rerun_resumes() {
    let t = tmp("keep");
    let rels = frames(&t.0, 5, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    // Four lines written: three frames, the fourth cut off by the kill.
    let fake = LabelFake {
        stop_after: Some(4),
        ..LabelFake::default()
    };
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let cancel = AtomicBool::new(false);
    let out = run_cancellable(&job, &fake, &cancel).unwrap();
    assert!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    let rep = &out.result;
    assert_eq!(rep["cancelled"], true, "{rep}");
    assert_eq!(
        (rep["frames_done"].as_u64(), rep["frames_total"].as_u64()),
        (Some(3), Some(5)),
        "{rep}"
    );
    assert_eq!(rep["frames"], 3);
    assert_eq!(
        rep["added"], 6,
        "two class boxes on each of three frames: {rep}"
    );
    assert!(
        rep["warnings"].as_array().unwrap().is_empty(),
        "the cut-off line is no warning: {rep}"
    );
    assert!(
        out.summary
            .starts_with("stopped by Cancel after 3 of 5 frames; those 3 were saved for review"),
        "{}",
        out.summary
    );
    assert!(!job.work.exists(), "the work folder is removed");
    let labeler = rep["labeler"].as_str().unwrap();
    // The proposals follow the frame list's order: the first three are done.
    let list: Vec<String> = queue::build(&dir, &job.request.frames.filters().unwrap())
        .items
        .into_iter()
        .map(|i| i.frame)
        .collect();
    assert_eq!(list.len(), 5);
    for rel in &list[..3] {
        let f = labels_of(&dir, rel);
        assert_eq!(f.boxes.len(), 2, "{rel}");
        assert_eq!(f.auto_run.as_ref().unwrap().labeler, labeler);
    }
    for rel in &list[3..] {
        let f = Frame::resolve(&dir, rel).unwrap().labels().unwrap();
        assert!(f.is_none_or(|f| f.auto_run.is_none()), "{rel}");
    }
    // A rerun of the same labelers skips the three and labels the other two.
    let job2 =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out2 = run_job(&job2, &LabelFake::default()).unwrap();
    assert_eq!(
        (
            out2.result["frames"].as_u64(),
            out2.result["skipped_done"].as_u64(),
            out2.result["cancelled"].as_bool()
        ),
        (Some(2), Some(3), Some(false)),
        "{}",
        out2.result
    );
    for rel in &rels {
        assert!(labels_of(&dir, rel).auto_run.is_some(), "{rel}");
    }
}

#[test]
fn a_cancelled_run_without_keep_on_cancel_writes_nothing() {
    let t = tmp("discard");
    let rels = frames(&t.0, 5, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let fake = LabelFake {
        stop_after: Some(4),
        ..LabelFake::default()
    };
    let mut req = AutolabelRequest::new(FrameFilter::default());
    req.keep_on_cancel = false;
    let job = plan_autolabel(&cfg, req, &host()).unwrap();
    let e = run_cancellable(&job, &fake, &AtomicBool::new(false)).unwrap_err();
    assert!(
        e.downcast_ref::<crate::resources::export::Cancelled>()
            .is_some(),
        "{e:#}"
    );
    for rel in &rels {
        let f = Frame::resolve(&dir, rel).unwrap().labels().unwrap();
        assert!(
            f.is_none_or(|f| f.auto_run.is_none() && f.boxes.is_empty()),
            "{rel}"
        );
    }
    assert!(!job.work.exists());
    // Cancelled before any frame finished: a plain cancel with keep_on_cancel too.
    let first = LabelFake {
        stop_after: Some(1),
        ..LabelFake::default()
    };
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let e = run_cancellable(&job, &first, &AtomicBool::new(false)).unwrap_err();
    assert!(
        e.downcast_ref::<crate::resources::export::Cancelled>()
            .is_some(),
        "{e:#}"
    );
    assert!(!job.work.exists());
}

#[test]
fn a_cancelled_model_run_keeps_the_finished_frames() {
    let t = tmp("keep-model");
    let rels = frames(&t.0, 4, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let dets: BTreeMap<String, Vec<Det>> = rels
        .iter()
        .map(|r| (r.clone(), vec![det("car", 0.9, [1.0, 1.0, 30.0, 20.0])]))
        .collect();
    let plan = |keep: bool| {
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.with_model = Some("IPcam-general".into());
        req.keep_on_cancel = keep;
        let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        let mut d = FakeDetector::new(dets.clone());
        d.stop_after = Some(3);
        job.detector = Some(Arc::new(d));
        job
    };
    // Without keep_on_cancel: nothing written.
    let e =
        run_cancellable(&plan(false), &LabelFake::default(), &AtomicBool::new(false)).unwrap_err();
    assert!(
        e.downcast_ref::<crate::resources::export::Cancelled>()
            .is_some(),
        "{e:#}"
    );
    for rel in &rels {
        let f = Frame::resolve(&dir, rel).unwrap().labels().unwrap();
        assert!(f.is_none_or(|f| f.auto_run.is_none()), "{rel}");
    }
    let job = plan(true);
    let cancel = AtomicBool::new(false);
    let out = run_cancellable(&job, &LabelFake::default(), &cancel).unwrap();
    let rep = &out.result;
    assert_eq!(rep["cancelled"], true, "{rep}");
    assert_eq!(
        (rep["frames_done"].as_u64(), rep["frames_total"].as_u64()),
        (Some(3), Some(4)),
        "{rep}"
    );
    assert_eq!(rep["added"], 3, "{rep}");
    assert!(
        out.summary
            .starts_with("stopped by Cancel after 3 of 4 frames"),
        "{}",
        out.summary
    );
    let marked: Vec<&String> = rels
        .iter()
        .filter(|r| {
            Frame::resolve(&dir, r)
                .unwrap()
                .labels()
                .unwrap()
                .is_some_and(|f| f.auto_run.is_some())
        })
        .collect();
    assert_eq!(marked.len(), 3, "{marked:?}");
}

/// An import of `<root>/clips` (one image, dHash, no Python) planned with the id the test
/// frames carry.
fn import_job(cfg: &Config, root: &Path, frames: usize) -> crate::training::import::job::ImportJob {
    use crate::training::import::{ImportRequest, ImportSource, Selection, plan_import};
    std::fs::create_dir_all(root.join("clips")).unwrap();
    std::fs::copy(root.join("src.jpg"), root.join("clips/a.jpg")).unwrap();
    let mut r = ImportRequest::new(ImportSource::Path(root.join("clips")));
    r.license_confirmed = true;
    r.selection = Selection::Dhash;
    r.frames = frames;
    let mut j = plan_import(cfg, r, &host()).unwrap();
    j.id = "20261022-000000-abcd".into();
    j
}

#[test]
fn a_run_follows_an_import_and_is_skipped_when_it_wrote_nothing() {
    let t = tmp("follow");
    let _ = frames(&t.0, 1, true);
    let cfg = config(&t.0);
    let mut imp = import_job(&cfg, &t.0, 50);
    imp.dataset_dir = t.0.join("elsewhere");
    let mut job = plan_autolabel(
        &cfg,
        AutolabelRequest::new(FrameFilter::parse("scene=ir,again=1,limit=3", "any").unwrap()),
        &host(),
    )
    .unwrap();
    assert_eq!((job.after.as_deref(), job.finish.is_none()), (None, true));
    super::super::follow_import(&mut job, &imp);
    assert_eq!(job.after.as_deref(), Some("20261022-000000-abcd"));
    assert_eq!(job.dataset_dir, t.0.join("elsewhere"));
    let f = &job.request.frames;
    assert_eq!(f.import(), Some("20261022-000000-abcd"));
    assert_eq!(
        (f.query["status"].as_str(), f.query["source"].as_str()),
        ("any", "all")
    );
    assert!(!f.again() && !f.query.contains_key("scene"), "{f:?}");
    assert_eq!(f.limit, 50);
    assert_eq!(Task::after(&job), Some("20261022-000000-abcd"));
    assert!(
        job.title().contains("import 20261022-000000-abcd"),
        "{}",
        job.title()
    );
    let rec = TaskRecord::new(&job, String::new());
    assert_eq!(rec.describe(), "queued (after 20261022-000000-abcd)");
    // The limit follows the import's frame count, capped.
    imp.request.frames = super::super::MAX_LIMIT + 5;
    super::super::follow_import(&mut job, &imp);
    assert_eq!(job.request.frames.limit, super::super::MAX_LIMIT);

    // The import it follows.
    let mut dep = TaskRecord::new(&imp, String::new());
    dep.state = RunState::Done;
    dep.result = Some(serde_json::json!({ "frames": 7 }));
    assert_eq!(job.skip_reason(&dep), None);
    dep.result = Some(serde_json::json!({ "frames": 0 }));
    assert_eq!(
        job.skip_reason(&dep).as_deref(),
        Some(super::super::NO_NEW_FRAMES)
    );
    for (st, why) in [
        (RunState::Failed, "not run: the import it follows failed"),
        (
            RunState::Cancelled,
            "not run: the import it follows was cancelled",
        ),
    ] {
        dep.state = st;
        assert_eq!(job.skip_reason(&dep).as_deref(), Some(why));
    }
}

#[test]
fn a_chained_run_rejects_low_boxes_then_accepts_then_switches_the_folder() {
    use crate::training::finish::Finish;
    let t = tmp("finish");
    let rels = frames(&t.0, 2, true);
    let cfg = config(&t.0);
    let imp = import_job(&cfg, &t.0, 2);
    let mut job =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    super::super::follow_import(&mut job, &imp);
    let writer = Arc::new(MemWriter(Mutex::new(cfg.clone())));
    let folder = t.0.join("dataset");
    let other = t.0.join("other");
    std::fs::create_dir_all(&other).unwrap();
    job.finish = Some(Finish {
        reject_below: Some(0.5),
        accept_auto: true,
        review_folder: Some(other.clone()),
        config_writer: Some(writer.clone()),
    });
    let out = run_job(&job, &LabelFake::default()).unwrap();
    // The raccoon (0.31, review) is rejected; then the car (accept) alone settles the frames.
    assert!(
        out.summary
            .contains("then: rejected 2 auto box(es) scoring below 0.5 on 2 frame(s) of import"),
        "{}",
        out.summary
    );
    assert!(
        out.summary.contains("accepted 2 frame(s)"),
        "{}",
        out.summary
    );
    assert!(
        out.summary.contains("the Review page now shows"),
        "{}",
        out.summary
    );
    assert_eq!(out.result["finish"].as_array().unwrap().len(), 3);
    for rel in &rels {
        let f = labels_of(&folder, rel);
        assert_eq!(f.status, Status::Reviewed, "{rel}");
        assert!(f.bulk_accepted);
        let raccoon = f.boxes.iter().find(|b| b.label == "raccoon").unwrap();
        assert_eq!(raccoon.state, BoxState::Rejected);
        assert_eq!(raccoon.reason.as_deref(), Some("low_score"));
        let car = f.boxes.iter().find(|b| b.label == "car").unwrap();
        assert_eq!(car.state, BoxState::Accepted);
    }
    assert_eq!(writer.0.lock().unwrap().collect.review_dir, Some(other));
}
