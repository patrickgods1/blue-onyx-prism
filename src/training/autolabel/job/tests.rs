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
use crate::training::autolabel::FrameFilter;
use crate::training::autolabel::models::{ConfigWriter, Install, Target};
use crate::training::autolabel::triage::Det;
use crate::training::capability::{self, Capability};
use crate::training::service::RunState;
use std::path::Path;
use std::sync::Mutex;

/// [`crate::training::autolabel::plan_autolabel`]; nothing here reads the config file's path.
fn plan_autolabel(
    config: &Config,
    request: AutolabelRequest,
    cap: &Capability,
) -> Result<AutolabelJob> {
    crate::training::autolabel::plan_autolabel(config, Path::new("config.json"), request, cap)
}

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
    frames_every(root, n, gray, 1)
}

/// [`frames`], `step` seconds apart (a day apart = each in its own folder).
fn frames_every(root: &Path, n: usize, gray: bool, step: u64) -> Vec<String> {
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
            time: std::time::SystemTime::UNIX_EPOCH
                + Duration::from_secs(1_800_000_000 + i as u64 * step),
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
    /// Die after this many proposal lines and half of the next one, with a plain error.
    fail_after: Option<usize>,
    /// Write only this many lines, then exit cleanly without the `done` event.
    no_summary_after: Option<usize>,
    /// The boxes of every frame, in place of the default car, raccoon and unicorn.
    boxes: Option<serde_json::Value>,
    /// Label files a reviewer saves while the script runs (path, new content).
    saved_during_run: Vec<(PathBuf, LabelFile)>,
    /// Label files that cannot be written from the moment the script runs ([`unwritable`]).
    broken_during_run: Vec<PathBuf>,
}

/// Make [`labels::update`] fail on the label file at `path`, on every OS, as a full disk or a lost
/// permission fails it after the frames were selected. The update stops at the read, before the
/// triage; [`a_box_that_was_not_written_is_not_counted_as_added`] covers a write that fails after it.
fn unwritable(path: &Path) {
    std::fs::write(path, b"not a label file").unwrap();
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
        for (path, f) in &self.saved_during_run {
            labels::write(path, f).unwrap();
        }
        for path in &self.broken_during_run {
            unwritable(path);
        }
        let list = arg_after(cmd, "--frames").unwrap();
        let out = arg_after(cmd, "--out").unwrap();
        let mut text = String::new();
        for (i, l) in std::fs::read_to_string(list).unwrap().lines().enumerate() {
            if self.fail_after == Some(i) {
                text.push_str("{\"frame\": \"torn");
                std::fs::write(&out, &text).unwrap();
                return Err(anyhow!("the GPU fell off the bus"));
            }
            if self.no_summary_after == Some(i) {
                std::fs::write(&out, &text).unwrap();
                return Ok(());
            }
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
            let seat = if cmd.args.iter().any(|a| a == "--sam3") {
                "sam3"
            } else {
                "gdino"
            };
            let boxes = self.boxes.clone().unwrap_or_else(|| serde_json::json!([
                {"label": "car", "bbox": [1, 1, 30, 20], "score": 0.82, "agree": true,
                 "sources": {(seat): 0.7, "rfdetr": 0.94}},
                {"label": "raccoon", "bbox": [35, 20, 60, 46], "score": 0.31, "agree": false,
                 "sources": {(seat): 0.31},
                 "vlm": {"model": "qwen3-vl-2b", "answer": "rock", "expected": "animal", "ok": false},
                 "species_scores": {"raccoon": 0.6, "cat": 0.4}},
                {"label": "unicorn", "bbox": [1, 1, 5, 5], "score": 0.9, "agree": true, "sources": {}}
            ]));
            let line = serde_json::json!({"frame": f["frame"], "width": 64, "height": 48,
                                          "boxes": boxes, "ms": {"gdino": 10}});
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
        difficult: false,
        reject_reason: None,
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
        difficult: false,
        reject_reason: None,
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
            difficult: false,
            reject_reason: None,
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
        difficult: false,
        reject_reason: None,
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
    // The defaults: batches of 8, a score cutoff with its audit sample, the next frame prefetched.
    let a = argv(&|_| {});
    assert_eq!(after(&a, "--vlm-batch").as_deref(), Some("8"), "{a:?}");
    assert_eq!(
        after(&a, "--vlm-min-score").as_deref(),
        Some("0.15"),
        "{a:?}"
    );
    assert_eq!(after(&a, "--vlm-sample").as_deref(), Some("0.02"), "{a:?}");
    assert!(has(&a, "--prefetch"), "{a:?}");
    // Everything off.
    let a = argv(&|r| {
        r.vlm_batch = false;
        r.vlm_min_score = None;
        r.prefetch = false;
    });
    assert_eq!(after(&a, "--vlm-batch").as_deref(), Some("1"), "{a:?}");
    assert!(
        !has(&a, "--vlm-min-score") && !has(&a, "--vlm-sample") && !has(&a, "--prefetch"),
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

/// A SAM 3 run, past the plan's login gate (it looks at this machine's token).
fn sam3_job(cfg: &Config) -> AutolabelJob {
    let mut job =
        plan_autolabel(cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    job.request.gdino = None;
    job.request.sam3 = true;
    job
}

#[test]
fn sam3_runs_in_the_gdino_seat_with_its_own_marker_and_never_accepts() {
    let t = tmp("sam3");
    let rels = frames(&t.0, 2, false);
    let cfg = config(&t.0);
    let fake = LabelFake::default();

    let job = sam3_job(&cfg);
    assert_eq!(
        job.labeler(),
        "sam3 + rfdetr-large + qwen3-vl-2b + siglip2-base"
    );
    assert_eq!(Task::uses(&job), [catalog::SAM3_ID]);
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["frames"], 2);
    // The car both found at 0.82 would be `accept` with MM-Grounding-DINO.
    assert_eq!(out.result["accept"], 0, "{}", out.result);
    assert_eq!(out.result["review"], 4, "{}", out.result);
    let dir = t.0.join("dataset");
    let f = labels_of(&dir, &rels[0]);
    let car = f.boxes.iter().find(|b| b.label == "car").unwrap();
    assert_eq!(car.auto.triage, Some(labels::Triage::Review));
    assert!(car.auto.why.iter().any(|w| w.contains("SAM 3")), "{car:?}");
    assert_eq!(car.auto.labelers.keys().next().unwrap(), "rfdetr");
    assert!(car.auto.labelers.contains_key("sam3"), "{car:?}");
    assert_eq!(f.auto_run.unwrap().labeler, job.labeler());

    // The script's arguments: --sam3 with its folder, no --gdino.
    let cmd = fake.cmds.lock().unwrap().last().unwrap().clone();
    let weights = job.paths.weights.join("sam3");
    assert_eq!(arg_after(&cmd, "--sam3").unwrap(), weights);
    assert_eq!(
        arg_after(&cmd, "--sam3-threshold").unwrap(),
        Path::new("0.3")
    );
    assert!(arg_after(&cmd, "--gdino").is_none(), "{}", cmd.display());
    assert!(
        fake.inner
            .fetched
            .lock()
            .unwrap()
            .contains(&catalog::SAM3_ID.to_string())
    );

    // The same run again is covered; MM-Grounding-DINO's is another run.
    let e = run_job(&sam3_job(&cfg), &fake).unwrap_err().to_string();
    assert!(e.contains("all 2 frame(s)"), "{e}");
    let gdino =
        plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    assert!(Task::uses(&gdino).is_empty());
    let out = run_job(&gdino, &fake).unwrap();
    assert_eq!(out.result["frames"], 2);
    assert!(!out.summary.contains("SAM 3 threshold"), "{}", out.summary);
    // So is SAM 3 after it: a gdino marker does not skip these frames.
    assert_eq!(run_job(&sam3_job(&cfg), &fake).unwrap().result["frames"], 2);
}

#[test]
fn sam3_can_run_without_rfdetr() {
    let t = tmp("sam3-alone");
    frames(&t.0, 1, false);
    let cfg = config(&t.0);
    let fake = LabelFake::default();
    let mut job = sam3_job(&cfg);
    job.request.rfdetr = false;
    job.request.sam3_threshold = 0.45;
    let out = run_job(&job, &fake).unwrap();
    let cmd = fake.cmds.lock().unwrap().last().unwrap().clone();
    assert_eq!(
        arg_after(&cmd, "--sam3-threshold").unwrap(),
        Path::new("0.45")
    );
    assert!(arg_after(&cmd, "--rfdetr").is_none(), "{}", cmd.display());
    assert!(
        out.summary.contains("SAM 3 threshold 0.45"),
        "{}",
        out.summary
    );
}

#[test]
fn a_rerun_at_a_new_sam3_threshold_is_skipped_by_the_marker_without_again() {
    let t = tmp("sam3-again");
    frames(&t.0, 2, false);
    let cfg = config(&t.0);
    let fake = LabelFake::default();
    run_job(&sam3_job(&cfg), &fake).unwrap();
    let mut rerun = sam3_job(&cfg);
    rerun.request.sam3_threshold = 0.45;
    let p = rerun.preview().unwrap();
    assert_eq!((p.taken, p.skipped_done), (0, 2), "{p:?}");
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("not the SAM 3 threshold")),
        "{p:?}"
    );
    assert!(run_job(&rerun, &fake).is_err(), "nothing left to label");
    rerun.request.frames = FrameFilter::parse("again=1", "unreviewed").unwrap();
    assert_eq!(rerun.preview().unwrap().taken, 2);
    assert_eq!(run_job(&rerun, &fake).unwrap().result["frames"], 2);
    let cmd = fake.cmds.lock().unwrap().last().unwrap().clone();
    assert_eq!(
        arg_after(&cmd, "--sam3-threshold").unwrap(),
        Path::new("0.45")
    );
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
    // The order the run takes them in: its boxes change the queue's order afterwards.
    let list: Vec<String> = queue::build(&dir, &job.request.frames.filters().unwrap())
        .items
        .into_iter()
        .map(|i| i.frame)
        .collect();
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
        out.summary.contains("the Review tab now shows"),
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

/// `missing_only` runs.
mod missing {
    use super::*;
    use crate::collect::labels::export::flat_name;
    use crate::collect::labels::{LabelBox, LabelFile};
    use crate::collect::layout;

    /// A reviewed file with an accepted model box per `(label, bbox)`.
    fn reviewed(boxes: &[(&str, [f32; 4])]) -> LabelFile {
        LabelFile {
            status: Status::Reviewed,
            boxes: boxes
                .iter()
                .map(|(label, bbox)| LabelBox {
                    label: (*label).into(),
                    bbox: *bbox,
                    source: Source::Model,
                    score: Some(0.9),
                    state: BoxState::Accepted,
                    origin: Some(labels::Origin::Prediction),
                    reason: None,
                    model_label: None,
                    orig_bbox: None,
                    difficult: false,
                    reject_reason: None,
                    auto: Default::default(),
                })
                .collect(),
            ..LabelFile::default()
        }
    }

    fn put(dir: &Path, rel: &str, f: &LabelFile) {
        labels::write(&Frame::resolve(dir, rel).unwrap().labels, f).unwrap();
    }

    fn bytes(dir: &Path, rel: &str) -> Vec<u8> {
        std::fs::read(Frame::resolve(dir, rel).unwrap().labels).unwrap()
    }

    /// A planned run of the model `name`, on `cfg`, with `detector`.
    fn plan(
        cfg: &Config,
        name: &str,
        filter: &str,
        detector: Arc<dyn Detector>,
    ) -> Result<AutolabelJob> {
        let mut req = AutolabelRequest::new(FrameFilter::parse(filter, "unreviewed")?);
        req.with_model = Some(name.into());
        req.missing_only = true;
        let mut job = plan_autolabel(cfg, req, &no_gpu())?;
        job.detector = Some(detector);
        Ok(job)
    }

    fn fake(dets: &[(&str, Vec<Det>)]) -> Arc<dyn Detector> {
        Arc::new(FakeDetector::new(
            dets.iter()
                .map(|(rel, d)| ((*rel).to_string(), d.clone()))
                .collect(),
        ))
    }

    /// The default model trained in run `id` on `rels`: its provenance and `train.txt`, each frame
    /// repeated as balancing does, under source prefixes as the dataset writes them.
    fn trained_on(cfg: &Config, id: &str, rels: &[&str]) {
        let entry = cfg.models.first().unwrap();
        let path = cfg.data_path(&entry.path);
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path.with_file_name(format!("{stem}.provenance.json")),
            serde_json::json!({"run_id": id}).to_string(),
        )
        .unwrap();
        let data = crate::training::training_dir(cfg).join(id).join("data");
        std::fs::create_dir_all(&data).unwrap();
        let mut text = String::new();
        for (i, rel) in rels.iter().enumerate() {
            let flat = flat_name(&layout::parse_rel(rel).unwrap());
            for _ in 0..3 {
                text.push_str(&format!("./images/s{i}-{flat}\n"));
            }
        }
        std::fs::write(data.join("train.txt"), text).unwrap();
    }

    const CAR: [f32; 4] = [1.0, 1.0, 30.0, 20.0];
    const SPOT: [f32; 4] = [35.0, 20.0, 60.0, 46.0];

    #[test]
    fn plan_forces_every_frame_and_raises_the_defaults() {
        let t = tmp("missing-plan");
        let cfg = config(&t.0);
        let job = plan(&cfg, "IPcam-general", "", fake(&[])).unwrap();
        let f = &job.request.frames;
        assert_eq!(f.query["status"], "any");
        assert_eq!(f.query["source"], "all");
        assert_eq!(f.limit, crate::training::autolabel::MAX_LIMIT);
        assert_eq!(job.request.threshold, 0.5);
        assert!(
            job.title()
                .starts_with("find missing labels with IPcam-general")
        );
        // What a page posts: the queue's filters and its own status.
        let job = plan(
            &cfg,
            "IPcam-general",
            "status=reviewed,source=queue,scene=ir,limit=50",
            fake(&[]),
        )
        .unwrap();
        let f = &job.request.frames;
        assert_eq!(f.query["status"], "any");
        assert_eq!(f.query["source"], "all");
        assert_eq!(f.query["scene"], "ir", "other filters stay");
        assert_eq!(f.limit, 50, "a limit somebody chose stays");
        // A threshold somebody chose stays.
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.with_model = Some("IPcam-general".into());
        req.missing_only = true;
        req.threshold = 0.7;
        let job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        assert_eq!(job.request.threshold, 0.7);
        // A normal run is not changed.
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.with_model = Some("IPcam-general".into());
        let job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        assert_eq!(job.request.frames.query["status"], "unreviewed");
        assert_eq!(job.request.frames.limit, 200);
        assert!((job.request.threshold - 0.3).abs() < 1e-6);
        // Bad requests are refused before anything else.
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.missing_only = true;
        let e = plan_autolabel(&cfg, req, &no_gpu())
            .unwrap_err()
            .to_string();
        assert!(e.contains("--with-model"), "{e}");
    }

    #[test]
    fn only_frames_gaining_a_box_are_written() {
        let t = tmp("missing-write");
        let rels = frames(&t.0, 6, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        // Gains a box.
        let mut gains = reviewed(&[("person", SPOT)]);
        gains.overlaps_checked = true;
        put(&dir, &rels[0], &gains);
        // The detection is the labelled object, under the other vehicle name.
        put(&dir, &rels[1], &reviewed(&[("truck", CAR)]));
        // The model trained on this one.
        put(&dir, &rels[2], &reviewed(&[("person", SPOT)]));
        // Confirmed empty.
        let mut empty = reviewed(&[]);
        empty.empty_confirmed = true;
        put(&dir, &rels[3], &empty);
        // Skipped.
        let mut skipped = reviewed(&[]);
        skipped.status = Status::Skipped;
        put(&dir, &rels[4], &skipped);
        // rels[5]: no label file.
        trained_on(&cfg, "20261031-1", &[&rels[2]]);
        let before: Vec<Vec<u8>> = (1..5).map(|i| bytes(&dir, &rels[i])).collect();
        // Frames reviewed under "all collected": not in the queue file, and still reached.
        let queue_file = dir.join(layout::REVIEW_QUEUE_FILE);
        std::fs::remove_file(&queue_file).unwrap();
        let car = || vec![det("car", 0.9, CAR)];
        let detector = fake(&[
            (&rels[0], car()),
            (&rels[1], car()),
            (&rels[2], car()),
            (&rels[3], car()),
            (&rels[4], car()),
            (&rels[5], car()),
        ]);
        let job = plan(&cfg, "IPcam-general", "", detector).unwrap();
        // The preview walks the frames as the run does, the training record included.
        let p = job.preview().unwrap();
        assert_eq!(p.matched, 3, "{p:?}");
        assert_eq!((p.taken, p.skipped_trained, p.skipped_done), (2, 1, 0));
        assert!(p.frames.contains(&rels[0]) && p.frames.contains(&rels[1]));
        assert!(!p.frames.contains(&rels[2]), "trained on");
        let mut short = job.clone();
        short.request.frames.limit = 1;
        let cut = short.preview().unwrap();
        assert_eq!((cut.taken, cut.over_limit), (1, 1), "{cut:?}");
        assert_eq!(cut.skipped_trained, 1, "past the limit too: {cut:?}");
        let out = run_job(&job, &LabelFake::default()).unwrap();
        let rep = &out.result;
        assert_eq!(rep["missing_only"], true, "{rep}");
        assert_eq!(rep["skipped_trained"], p.skipped_trained, "{rep}");
        assert_eq!(rep["frames"], p.taken, "{rep}");
        assert_eq!(rep["frames"], 2, "reviewed, not trained on: {rep}");
        assert_eq!(rep["skipped_trained"], 1, "{rep}");
        assert_eq!(rep["proposals"], 2, "{rep}");
        assert_eq!(rep["candidates"], 1, "{rep}");
        assert_eq!(rep["added"], 1, "{rep}");
        assert_eq!(rep["dropped"], 1, "{rep}");
        assert_eq!(rep["reopened"], 1, "{rep}");
        assert_eq!(rep["disagreements"], 0, "{rep}");
        let summary = &out.summary;
        assert!(summary.contains("1 candidate boxes"), "{summary}");
        assert!(summary.contains("1 frames reopened"), "{summary}");
        assert!(
            summary.contains("leave exports until you save"),
            "{summary}"
        );
        // Untouched frames are byte-identical, and no file appears for a frame without one.
        let after: Vec<Vec<u8>> = (1..5).map(|i| bytes(&dir, &rels[i])).collect();
        assert_eq!(before, after);
        assert!(!Frame::resolve(&dir, &rels[5]).unwrap().labels.exists());
        // The changed frame.
        let f = labels_of(&dir, &rels[0]);
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by IPcam-general: 1 new box")
        );
        assert!(!f.overlaps_checked);
        assert_eq!(f.auto_run, None, "no marker");
        assert_eq!(f.boxes.len(), 2);
        assert_eq!(f.boxes[1].state, BoxState::Pending);
        assert_eq!(f.boxes[1].auto.triage, Some(Triage::Review));
        // Only that frame was queued.
        let queued: Vec<String> = layout::read_review(&queue_file)
            .into_iter()
            .map(|l| l.frame)
            .collect();
        assert_eq!(queued, [rels[0].clone()]);
        // A save through the page's normalisation takes it back to reviewed, boxes accepted.
        let classes = cfg.collect.classes.clone();
        let mut f = f;
        f.status = Status::Reviewed;
        f.boxes
            .iter_mut()
            .for_each(|b| b.state = BoxState::Accepted);
        let saved = labels::normalize(f, &classes, 64, 48, "2026-11-04T00:00:00.000Z").unwrap();
        assert_eq!(saved.training().count(), 2);
        assert_eq!(saved.disagreement, None);
    }

    #[test]
    fn a_missing_only_run_writes_no_vote_on_a_box_it_finds_again() {
        let t = tmp("missing-vote");
        let rels = frames(&t.0, 1, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        let mut f = reviewed(&[("car", CAR)]);
        f.boxes[0].auto.labelers.insert("gdino".into(), 0.41);
        put(&dir, &rels[0], &f);
        let before = bytes(&dir, &rels[0]);
        let detector = fake(&[(&rels[0], vec![det("car", 0.9, CAR)])]);
        let job = plan(&cfg, "IPcam-general", "", detector).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["votes"], 0, "{}", out.result);
        assert_eq!(out.result["dropped"], 1, "{}", out.result);
        assert!(!out.summary.contains("vote"), "{}", out.summary);
        assert_eq!(bytes(&dir, &rels[0]), before, "the file is not written");
    }

    #[test]
    fn a_model_run_votes_on_boxes_it_finds_again_and_queues_no_frame_for_it() {
        let t = tmp("model-vote");
        let rels = frames(&t.0, 3, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        // A pending box of a labelers run, a reviewed box with evidence, an evidence-free car.
        let mut open = reviewed(&[("car", CAR)]);
        open.status = Status::Unreviewed;
        open.boxes[0].state = BoxState::Pending;
        open.boxes[0].source = Source::Auto;
        open.boxes[0].origin = None;
        open.boxes[0].auto.labelers.insert("gdino".into(), 0.41);
        open.boxes[0].auto.triage = Some(Triage::Review);
        put(&dir, &rels[0], &open);
        let mut done = reviewed(&[("car", CAR)]);
        done.boxes[0].auto.labelers.insert("gdino".into(), 0.41);
        put(&dir, &rels[1], &done);
        let plain = LabelFile {
            status: Status::Unreviewed,
            ..reviewed(&[("car", CAR)])
        };
        put(&dir, &rels[2], &plain);
        let queue_file = dir.join(layout::REVIEW_QUEUE_FILE);
        std::fs::remove_file(&queue_file).unwrap();

        let car = || vec![det("car", 0.9, CAR)];
        let mut req =
            AutolabelRequest::new(FrameFilter::parse("status=any", "unreviewed").unwrap());
        req.with_model = Some("IPcam-general".into());
        let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        job.detector = Some(fake(&[
            (&rels[0], car()),
            (&rels[1], car()),
            (&rels[2], car()),
        ]));
        let out = run_job(&job, &LabelFake::default()).unwrap();
        let rep = &out.result;
        assert_eq!(rep["votes"], 2, "{rep}");
        // The third frame's box takes the evidence, as before: annotated, and queued for that.
        assert_eq!(rep["annotated"], 1, "{rep}");
        assert_eq!(
            (&rep["added"], &rep["dropped"]),
            (&0.into(), &3.into()),
            "{rep}"
        );
        assert_eq!(
            (&rep["disagreements"], &rep["reopened"]),
            (&0.into(), &0.into())
        );
        assert!(
            out.summary
                .contains("2 existing boxes gained a labeler's vote"),
            "{}",
            out.summary
        );

        // Only `labelers` and the marker changed on the voted frames.
        for (rel, before) in [(&rels[0], open), (&rels[1], done)] {
            let mut f = labels_of(&dir, rel);
            let voted = &f.boxes[0].auto.labelers;
            assert_eq!(voted["gdino"], 0.41, "{voted:?}");
            assert_eq!(voted["IPcam-general"], 0.9, "{voted:?}");
            assert!(f.auto_run.is_some(), "every frame gets the marker");
            f.boxes[0].auto.labelers = before.boxes[0].auto.labelers.clone();
            f.auto_run = None;
            assert_eq!(f, before);
        }
        // A vote queues nothing; the annotated frame is queued as before.
        let queued: Vec<String> = layout::read_review(&queue_file)
            .into_iter()
            .map(|l| l.frame)
            .collect();
        assert_eq!(queued, [rels[2].clone()]);
    }

    #[test]
    fn two_models_run_in_turn_and_a_rerun_adds_nothing() {
        let t = tmp("missing-turn");
        let rels = frames(&t.0, 1, false);
        let dir = t.0.join("dataset");
        let mut cfg = config(&t.0);
        cfg.models.push(ModelConfig {
            name: Some("second".into()),
            path: "models/second.onnx".into(),
            enabled: false,
            ..ModelConfig::default()
        });
        put(&dir, &rels[0], &reviewed(&[]));
        let first = fake(&[(&rels[0], vec![det("car", 0.9, CAR)])]);
        let job = plan(&cfg, "IPcam-general", "", first.clone()).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["candidates"], 1);
        // No marker, so the second model is not turned away; the frame is unreviewed now, with
        // the note, so it is still in scope. It sees the same car and a dog.
        let both = fake(&[(&rels[0], vec![det("car", 0.8, CAR), det("dog", 0.7, SPOT)])]);
        let job = plan(&cfg, "second", "", both).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["frames"], 1, "{}", out.result);
        assert_eq!(out.result["candidates"], 1, "the dog only: {}", out.result);
        assert_eq!(out.result["reopened"], 0, "it was open already");
        let f = labels_of(&dir, &rels[0]);
        assert_eq!(f.boxes.len(), 2);
        assert_eq!(
            f.disagreement.as_deref(),
            Some("re-label by IPcam-general, second: 2 new boxes")
        );
        // A rerun of the first adds nothing and writes nothing.
        let before = bytes(&dir, &rels[0]);
        let job = plan(&cfg, "IPcam-general", "", first).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["candidates"], 0);
        assert!(!out.summary.contains("leave exports"), "{}", out.summary);
        assert_eq!(bytes(&dir, &rels[0]), before);
    }

    #[test]
    fn the_limit_counts_frames_in_scope() {
        let t = tmp("missing-limit");
        let rels = frames(&t.0, 3, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        // One of three frames in scope; the others sort anywhere.
        put(&dir, &rels[2], &reviewed(&[]));
        let job = plan(
            &cfg,
            "IPcam-general",
            "limit=1",
            fake(&[(&rels[2], vec![det("car", 0.9, CAR)])]),
        )
        .unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["frames"], 1);
        assert_eq!(out.result["candidates"], 1);
    }

    /// Takes a frame out of scope while the model runs, as a reviewer on the page can.
    struct SkipDuring {
        inner: Arc<dyn Detector>,
        dir: PathBuf,
        rel: String,
    }

    impl Detector for SkipDuring {
        fn detect(
            &self,
            config: &Config,
            model: &ModelConfig,
            device: Option<&str>,
            images: &[Image],
            threshold: f32,
            cancel: &AtomicBool,
        ) -> Result<super::super::super::relabel::Detections> {
            let mut f = labels_of(&self.dir, &self.rel);
            f.status = Status::Skipped;
            put(&self.dir, &self.rel, &f);
            self.inner
                .detect(config, model, device, images, threshold, cancel)
        }
    }

    #[test]
    fn the_scope_is_checked_again_under_the_write_lock() {
        let t = tmp("missing-recheck");
        let rels = frames(&t.0, 1, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        put(&dir, &rels[0], &reviewed(&[]));
        let detector = Arc::new(SkipDuring {
            inner: fake(&[(&rels[0], vec![det("car", 0.9, CAR)])]),
            dir: dir.clone(),
            rel: rels[0].clone(),
        });
        let job = plan(&cfg, "IPcam-general", "", detector).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["candidates"], 0);
        let f = labels_of(&dir, &rels[0]);
        assert_eq!(f.status, Status::Skipped);
        assert!(f.boxes.is_empty());
    }

    #[test]
    fn a_model_without_a_record_warns_and_runs_a_lost_list_refuses() {
        let t = tmp("missing-record");
        let rels = frames(&t.0, 1, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        put(&dir, &rels[0], &reviewed(&[]));
        let detector = fake(&[(&rels[0], vec![det("car", 0.9, CAR)])]);
        let job = plan(&cfg, "IPcam-general", "", detector.clone()).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["candidates"], 1);
        let warn = out.result["warnings"][0].as_str().unwrap();
        assert!(warn.contains("no training record"), "{warn}");
        assert!(
            out.summary.contains("no training record"),
            "{}",
            out.summary
        );
        // The record is there, its run folder is not: nothing runs, nothing is written.
        trained_on(&cfg, "20261031-1", &[]);
        std::fs::remove_dir_all(crate::training::training_dir(&cfg).join("20261031-1")).unwrap();
        let before = bytes(&dir, &rels[0]);
        let job = plan(&cfg, "IPcam-general", "", detector).unwrap();
        let e = run_job(&job, &LabelFake::default())
            .unwrap_err()
            .to_string();
        assert!(e.contains("20261031-1") && e.contains("train.txt"), "{e}");
        assert_eq!(bytes(&dir, &rels[0]), before);
    }

    #[test]
    fn a_local_model_finds_its_record_before_it_is_in_the_config() {
        let t = tmp("missing-local");
        let rels = frames(&t.0, 2, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        put(&dir, &rels[0], &reviewed(&[]));
        put(&dir, &rels[1], &reviewed(&[]));
        // The default entry's record, then the same record for a file no entry uses.
        trained_on(&cfg, "20261031-1", &[&rels[0]]);
        let models = cfg.data_path(&cfg.models_dir);
        std::fs::create_dir_all(&models).unwrap();
        let path = cfg.data_path(&cfg.models.first().unwrap().path);
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        std::fs::write(models.join("mine.onnx"), b"x").unwrap();
        std::fs::rename(
            path.with_file_name(format!("{stem}.provenance.json")),
            models.join("mine.provenance.json"),
        )
        .unwrap();
        let detector = fake(&[(&rels[1], vec![det("car", 0.9, CAR)])]);
        let mut job = plan(&cfg, "local:mine.onnx", "", detector).unwrap();
        job.config_writer = Some(Arc::new(MemWriter(Mutex::new(cfg.clone()))));
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["skipped_trained"], 1, "{}", out.result);
        assert_eq!(out.result["frames"], 1, "{}", out.result);
    }

    #[test]
    fn nothing_in_scope_says_why() {
        let t = tmp("missing-none");
        let rels = frames(&t.0, 1, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        put(&dir, &rels[0], &reviewed(&[]));
        trained_on(&cfg, "20261031-1", &[&rels[0]]);
        let job = plan(&cfg, "IPcam-general", "", fake(&[])).unwrap();
        let e = run_job(&job, &LabelFake::default())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("1 reviewed frame(s)") && e.contains("trained on"),
            "{e}"
        );
    }

    #[test]
    fn a_cancel_keeps_the_finished_frames_and_says_a_rerun_looks_again() {
        let t = tmp("missing-cancel");
        let rels = frames(&t.0, 2, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        put(&dir, &rels[0], &reviewed(&[]));
        put(&dir, &rels[1], &reviewed(&[]));
        let dets = rels
            .iter()
            .map(|r| (r.clone(), vec![det("car", 0.9, CAR)]))
            .collect();
        let detector = FakeDetector {
            stop_after: Some(1),
            ..FakeDetector::new(dets)
        };
        let job = plan(&cfg, "IPcam-general", "", Arc::new(detector)).unwrap();
        let out = run_job(&job, &LabelFake::default()).unwrap();
        assert_eq!(out.result["cancelled"], true);
        assert_eq!(out.result["candidates"], 1);
        assert!(
            out.summary.starts_with("stopped by Cancel after 1 of 2"),
            "{}",
            out.summary
        );
        assert!(out.summary.contains("run it again"), "{}", out.summary);
    }
}

/// A reviewed file with one drawn car where the fake labelers find theirs, plus `extra` boxes.
fn reviewed_car(extra: Vec<labels::LabelBox>) -> LabelFile {
    let mut boxes = vec![labels::LabelBox {
        label: "car".into(),
        bbox: [1.0, 1.0, 30.0, 20.0],
        source: Source::Human,
        score: None,
        state: BoxState::Added,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: Default::default(),
    }];
    boxes.extend(extra);
    LabelFile {
        status: Status::Reviewed,
        reviewed_at: Some("2026-10-22T12:00:00.000Z".into()),
        boxes,
        ..LabelFile::default()
    }
}

/// A decided model box.
fn decided(label: &str, bbox: [f32; 4], state: BoxState) -> labels::LabelBox {
    labels::LabelBox {
        label: label.into(),
        bbox,
        source: Source::Model,
        score: Some(0.4),
        state,
        origin: Some(labels::Origin::Prediction),
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: Default::default(),
    }
}

fn review_queue(dir: &Path) -> Vec<String> {
    crate::collect::layout::read_review(&dir.join(crate::collect::layout::REVIEW_QUEUE_FILE))
        .into_iter()
        .map(|l| l.frame)
        .collect()
}

/// A SAM 3 run over `filter`, no VLM or SigLIP.
fn sam3_over(cfg: &Config, filter: &str) -> AutolabelJob {
    let mut job = sam3_job(cfg);
    job.request.frames = FrameFilter::parse(filter, "unreviewed").unwrap();
    job.request.vlm = None;
    job.request.siglip = false;
    job
}

/// The boxes every frame gets in the reviewed-frame runs: a confident second sight of the drawn
/// car, a confident person (both with a VLM yes, which the fake's script can give whatever the
/// request says), a weak dog.
fn sam3_boxes() -> serde_json::Value {
    let yes = serde_json::json!({"answer": "person", "expected": "person", "ok": true});
    serde_json::json!([
        {"label": "car", "bbox": [1, 1, 30, 20], "score": 0.35, "agree": false,
         "sources": {"sam3": 0.7}, "vlm": yes},
        {"label": "person", "bbox": [40, 5, 60, 40], "score": 0.45, "agree": false,
         "sources": {"sam3": 0.9}, "vlm": yes},
        {"label": "dog", "bbox": [2, 25, 30, 46], "score": 0.1, "agree": false,
         "sources": {"sam3": 0.2}}
    ])
}

#[test]
fn a_lone_box_with_no_vlm_answer_is_counted_and_reopens_nothing() {
    let t = tmp("unchecked");
    let rels = frames(&t.0, 1, false);
    let dir = t.0.join("dataset");
    labels::write(
        &Frame::resolve(&dir, &rels[0]).unwrap().labels,
        &reviewed_car(vec![]),
    )
    .unwrap();
    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "status=any,limit=20");
    let fake = LabelFake {
        boxes: Some(serde_json::json!([
            {"label": "person", "bbox": [40, 5, 60, 40], "score": 0.45, "agree": false,
             "sources": {"sam3": 0.9}},
            {"label": "dog", "bbox": [2, 25, 30, 46], "score": 0.45, "agree": false,
             "sources": {"sam3": 0.9}, "vlm_skipped": "score"}
        ])),
        ..LabelFake::default()
    };
    let out = run_job(&job, &fake).unwrap();
    let rep = &out.result;
    assert_eq!(rep["filtered_unchecked"], 2, "{rep}");
    assert_eq!((&rep["added"], &rep["reopened"]), (&0.into(), &0.into()));
    assert!(
        out.summary
            .contains("2 found by one labeler with no VLM yes"),
        "{}",
        out.summary
    );
    let f = labels_of(&dir, &rels[0]);
    assert_eq!((f.status, f.boxes.len()), (Status::Reviewed, 1));
    assert!(f.auto_run.is_some());
}

#[test]
fn a_confirmed_empty_frame_nobody_reviewed_takes_only_the_marker() {
    let t = tmp("empty-unreviewed");
    let rels = frames(&t.0, 1, false);
    let dir = t.0.join("dataset");
    let path = Frame::resolve(&dir, &rels[0]).unwrap().labels;
    let empty = LabelFile {
        empty_confirmed: true,
        ..LabelFile::default()
    };
    labels::write(&path, &empty).unwrap();
    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "limit=20");
    let fake = LabelFake {
        boxes: Some(sam3_boxes()),
        ..LabelFake::default()
    };
    let rep = run_job(&job, &fake).unwrap().result;
    assert_eq!(rep["added"], 0, "{rep}");
    assert_eq!(rep["filtered_closed"], 3, "{rep}");
    let f = labels_of(&dir, &rels[0]);
    assert!(f.boxes.is_empty(), "{f:?}");
    assert!(f.auto_run.is_some(), "{f:?}");
}

#[test]
fn a_rerun_over_reviewed_frames_only_reopens_for_confident_new_objects() {
    let t = tmp("reviewed");
    let rels = frames(&t.0, 5, false);
    let dir = t.0.join("dataset");
    let write = |rel: &str, f: &LabelFile| {
        labels::write(&Frame::resolve(&dir, rel).unwrap().labels, f).unwrap();
    };
    // 0: the person is new. 1: a person is labelled, nothing is new. 2: the person sits on an
    // accepted cat. 3: skipped. 4: confirmed empty.
    let person = [40.0, 5.0, 60.0, 40.0];
    write(
        &rels[0],
        &reviewed_car(vec![decided(
            "dog",
            [2.0, 25.0, 30.0, 46.0],
            BoxState::Rejected,
        )]),
    );
    write(
        &rels[1],
        &reviewed_car(vec![decided("person", person, BoxState::Accepted)]),
    );
    write(
        &rels[2],
        &reviewed_car(vec![decided("cat", person, BoxState::Accepted)]),
    );
    let mut skipped = reviewed_car(vec![]);
    skipped.status = Status::Skipped;
    write(&rels[3], &skipped);
    write(
        &rels[4],
        &LabelFile {
            empty_confirmed: true,
            boxes: Vec::new(),
            ..reviewed_car(vec![])
        },
    );
    let before: Vec<LabelFile> = rels.iter().map(|r| labels_of(&dir, r)).collect();
    // Only the frames of this run in the queue file: nothing was reviewed through it.
    let queued_before = review_queue(&dir);

    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "status=any,limit=20");
    let fake = LabelFake {
        boxes: Some(sam3_boxes()),
        ..LabelFake::default()
    };
    let out = run_job(&job, &fake).unwrap();
    let rep = &out.result;
    assert_eq!(rep["frames"], 5, "{rep}");
    assert_eq!(rep["added"], 1, "the person of frame 0: {rep}");
    assert_eq!(rep["reopened"], 1, "{rep}");
    assert_eq!(
        rep["filtered_low_score"], 3,
        "the dog, frames 0 to 2: {rep}"
    );
    assert_eq!(
        rep["filtered_other_object"], 1,
        "the person on the cat: {rep}"
    );
    assert_eq!(
        rep["filtered_closed"], 6,
        "three boxes on frames 3 and 4: {rep}"
    );
    assert_eq!(
        rep["votes"], 6,
        "cars 0 to 3, the dog on frame 0's rejected dog, the person of frame 1: {rep}"
    );
    assert_eq!(rep["annotated"], 0, "a vote is no annotation: {rep}");
    assert!(
        out.summary.contains("leave exports until you save"),
        "{}",
        out.summary
    );
    assert!(out.summary.contains("Ask again"), "{}", out.summary);
    assert!(out.summary.contains("Restore reopened"), "{}", out.summary);
    assert!(out.summary.contains("3 not confident"), "{}", out.summary);

    let after: Vec<LabelFile> = rels.iter().map(|r| labels_of(&dir, r)).collect();
    // Decided boxes: every field but the `labelers` a vote adds, byte for byte.
    let bare = |b: &labels::LabelBox| {
        let mut b = b.clone();
        b.auto.labelers.clear();
        serde_json::to_string(&b).unwrap()
    };
    for (i, (a, b)) in after.iter().zip(&before).enumerate() {
        for (x, y) in a.boxes.iter().zip(&b.boxes) {
            assert_eq!(bare(x), bare(y), "frame {i}");
        }
        assert_eq!(
            a.auto_run.as_ref().unwrap().labeler,
            rep["labeler"].as_str().unwrap(),
            "frame {i}"
        );
    }
    // Frame 0: reopened, with a note and one new pending box; the vote on its drawn car.
    let f0 = &after[0];
    assert_eq!(f0.status, Status::Unreviewed);
    assert_eq!(f0.boxes.len(), 3);
    assert_eq!(f0.boxes[0].auto.labelers["sam3"], 0.7);
    assert_eq!(f0.boxes[2].state, BoxState::Pending);
    assert_eq!(
        f0.disagreement.as_deref(),
        Some("re-label by sam3: 1 new box"),
        "the short name"
    );
    // The others keep their status and their boxes.
    for i in 1..5 {
        assert_eq!(after[i].status, before[i].status, "frame {i}");
        assert_eq!(after[i].boxes.len(), before[i].boxes.len(), "frame {i}");
        assert_eq!(after[i].disagreement, None, "frame {i}");
    }
    assert!(!labels::auto_accept_eligible(&after[3]));
    // Only the reopened frame is queued, once more; a vote queues nothing.
    let queued = review_queue(&dir);
    assert_eq!(
        queued.len(),
        queued_before.len(),
        "every frame was queued by the import: {queued:?}"
    );
    assert_eq!(rep["sample"], serde_json::json!([rels[0]]));
    // A rerun of the same labelers finds its markers.
    let job2 = sam3_over(&cfg, "status=any,limit=20");
    let e = run_job(&job2, &fake).unwrap_err().to_string();
    assert!(e.contains("all 5 frame(s)"), "{e}");
}

#[test]
fn a_vote_only_write_adds_nothing_to_the_review_queue() {
    let t = tmp("vote-queue");
    let rels = frames(&t.0, 1, false);
    let dir = t.0.join("dataset");
    let fr = Frame::resolve(&dir, &rels[0]).unwrap();
    labels::write(&fr.labels, &reviewed_car(vec![])).unwrap();
    // The frame left the queue file: a reviewed frame from an older queue, or `source=all`.
    std::fs::write(dir.join(crate::collect::layout::REVIEW_QUEUE_FILE), "").unwrap();
    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "status=reviewed,source=all");
    let fake = LabelFake {
        boxes: Some(serde_json::json!([sam3_boxes()[0]])),
        ..LabelFake::default()
    };
    let rep = run_job(&job, &fake).unwrap().result;
    assert_eq!(
        (rep["votes"].as_u64(), rep["reopened"].as_u64()),
        (Some(1), Some(0))
    );
    assert!(review_queue(&dir).is_empty(), "{:?}", review_queue(&dir));
    assert_eq!(rep["sample"], serde_json::json!([]));
    let f = labels_of(&dir, &rels[0]);
    assert_eq!(f.status, Status::Reviewed);
    assert_eq!(f.boxes[0].auto.labelers["sam3"], 0.7);
    assert_eq!(f.boxes[0].state, BoxState::Added);
}

#[test]
fn the_frames_status_is_read_when_the_file_is_written() {
    let t = tmp("write-time");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "limit=20");
    // Both unreviewed when selected. A reviewer saves frame 0 and skips frame 1 while the labelers
    // run, with the car drawn and nothing else on it.
    let mut skipped = reviewed_car(vec![]);
    skipped.status = Status::Skipped;
    let fake = LabelFake {
        saved_during_run: vec![
            (
                Frame::resolve(&dir, &rels[0]).unwrap().labels,
                reviewed_car(vec![]),
            ),
            (Frame::resolve(&dir, &rels[1]).unwrap().labels, skipped),
        ],
        boxes: Some(serde_json::json!([
            sam3_boxes()[2],
            {"label": "dog", "bbox": [35, 25, 60, 46], "score": 0.15, "agree": false,
             "sources": {"sam3": 0.3}}
        ])),
        ..LabelFake::default()
    };
    let rep = run_job(&job, &fake).unwrap().result;
    assert_eq!(rep["added"], 0, "{rep}");
    assert_eq!(rep["reopened"], 0, "{rep}");
    assert_eq!(rep["filtered_low_score"], 2, "{rep}");
    assert_eq!(rep["filtered_closed"], 2, "{rep}");
    let f0 = labels_of(&dir, &rels[0]);
    assert_eq!((f0.status, f0.boxes.len()), (Status::Reviewed, 1));
    assert!(f0.auto_run.is_some());
    let f1 = labels_of(&dir, &rels[1]);
    assert_eq!((f1.status, f1.boxes.len()), (Status::Skipped, 1));
    assert!(f1.auto_run.is_some());
}

#[test]
fn a_failed_run_keeps_its_finished_frames_and_still_fails() {
    let t = tmp("salvage");
    let rels = frames(&t.0, 5, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    for keep in [true, false] {
        let mut request = AutolabelRequest::new(FrameFilter::default());
        request.keep_on_cancel = keep;
        let job = plan_autolabel(&cfg, request, &host()).unwrap();
        let fake = LabelFake {
            fail_after: Some(if keep { 2 } else { 1 }),
            ..LabelFake::default()
        };
        let e = run_job(&job, &fake).unwrap_err();
        let text = format!("{e:#}");
        let n = if keep { 2 } else { 1 };
        assert!(
            text.contains(&format!("failed after {n} of ")),
            "keep {keep}: {text}"
        );
        assert!(text.contains("labels only the rest"), "{text}");
        assert!(text.contains("the GPU fell off the bus"), "{text}");
        assert!(!crate::training::pyenv::is_cancel(&e), "a failure: {text}");
        assert!(!job.work.exists(), "the work folder is removed");
    }
    // The finished frames are written with their markers, the others left alone.
    let labelled: Vec<&String> = rels
        .iter()
        .filter(|r| labels_of_opt(&dir, r).is_some_and(|f| f.auto_run.is_some()))
        .collect();
    assert_eq!(labelled.len(), 3, "2 frames, then 1 more: {labelled:?}");
    for r in &labelled {
        assert_eq!(labels_of(&dir, r).boxes.len(), 2, "{r}");
    }
    // A rerun resumes after them.
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let rep = run_job(&job, &LabelFake::default()).unwrap().result;
    assert_eq!(
        (rep["frames"].as_u64(), rep["skipped_done"].as_u64()),
        (Some(2), Some(3)),
        "{rep}"
    );
}

/// [`labels_of`] of a frame that may have no label file yet.
fn labels_of_opt(dir: &Path, rel: &str) -> Option<LabelFile> {
    Frame::resolve(dir, rel).unwrap().labels().unwrap()
}

#[test]
fn a_failed_run_says_so_when_it_reopened_frames() {
    let t = tmp("salvage-reopen");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    for r in &rels {
        labels::write(
            &Frame::resolve(&dir, r).unwrap().labels,
            &reviewed_car(vec![]),
        )
        .unwrap();
    }
    let cfg = config(&t.0);
    let job = sam3_over(&cfg, "status=reviewed");
    let fake = LabelFake {
        fail_after: Some(1),
        boxes: Some(sam3_boxes()),
        ..LabelFake::default()
    };
    let text = format!("{:#}", run_job(&job, &fake).unwrap_err());
    assert!(text.contains("failed after 1 of 3"), "{text}");
    assert!(text.contains("1 frames reopened"), "{text}");
    assert!(text.contains("leave exports"), "{text}");
    let reopened: Vec<LabelFile> = rels
        .iter()
        .map(|r| labels_of(&dir, r))
        .filter(|f| f.status == Status::Unreviewed)
        .collect();
    assert_eq!(reopened.len(), 1);
    assert!(reopened[0].disagreement.is_some());
    assert_eq!(review_queue(&dir).len(), 3, "still queued once");
}

#[test]
fn a_run_that_fails_with_nothing_finished_stays_a_plain_failure() {
    let t = tmp("salvage-none");
    frames(&t.0, 2, false);
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake {
        fail_after: Some(0),
        ..LabelFake::default()
    };
    let text = format!("{:#}", run_job(&job, &fake).unwrap_err());
    assert!(text.starts_with("the auto-labeler failed: "), "{text}");
    assert!(text.contains("the GPU fell off the bus"), "{text}");
    assert!(!text.contains("saved"), "{text}");
}

#[test]
fn a_script_that_exits_without_its_summary_is_a_failure_that_keeps_its_frames() {
    let t = tmp("salvage-summary");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake {
        no_summary_after: Some(2),
        ..LabelFake::default()
    };
    let text = format!("{:#}", run_job(&job, &fake).unwrap_err());
    assert!(text.contains("failed after 2 of 3"), "{text}");
    assert!(text.contains("without its summary"), "{text}");
    let n = rels
        .iter()
        .filter(|r| labels_of_opt(&dir, r).is_some_and(|f| f.auto_run.is_some()))
        .count();
    assert_eq!(n, 2);
}

#[test]
fn a_cancel_without_keep_on_cancel_still_discards_a_failing_run() {
    /// Sets the cancel flag, then dies with a plain error.
    struct DiesCancelled(LabelFake);
    impl CommandRunner for DiesCancelled {
        fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
            let r = self.0.run(cmd, on_line, cancel);
            if r.is_err() {
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            r
        }
    }
    impl Fetcher for DiesCancelled {
        fn fetch(
            &self,
            res: &'static Resource,
            dir: &Path,
            progress: &mut dyn FnMut(u64, u64),
            cancel: &AtomicBool,
        ) -> Result<()> {
            self.0.fetch(res, dir, progress, cancel)
        }
    }
    let t = tmp("salvage-cancel");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let mut request = AutolabelRequest::new(FrameFilter::default());
    request.keep_on_cancel = false;
    let job = plan_autolabel(&cfg, request, &host()).unwrap();
    let fake = DiesCancelled(LabelFake {
        fail_after: Some(2),
        ..LabelFake::default()
    });
    let cancel = AtomicBool::new(false);
    let ctx = TaskContext {
        runner: &fake,
        fetcher: &fake,
    };
    assert!(job.run(ctx, &mut |_| {}, &cancel).is_err());
    for r in &rels {
        assert!(labels_of_opt(&dir, r).is_none(), "{r}: asked to discard");
    }
}

/// The frames holding an `auto_run` marker, sorted.
fn marked(dir: &Path, rels: &[String]) -> Vec<String> {
    let mut v: Vec<String> = rels
        .iter()
        .filter(|r| labels_of_opt(dir, r).is_some_and(|f| f.auto_run.is_some()))
        .cloned()
        .collect();
    v.sort();
    v
}

fn request(filter: &str) -> AutolabelRequest {
    AutolabelRequest::new(FrameFilter::parse(filter, "unreviewed").unwrap())
}

#[test]
fn the_preview_takes_what_the_run_then_takes() {
    let t = tmp("preview-run");
    let rels = frames(&t.0, 6, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    // Two frames carry the labelers' marker.
    let first = plan_autolabel(&cfg, request("limit=2"), &host()).unwrap();
    let p = first.preview().unwrap();
    assert_eq!((p.matched, p.taken, p.over_limit), (6, 2, 4), "{p:?}");
    assert_eq!(p.skipped_done, 0, "{p:?}");
    run_job(&first, &LabelFake::default()).unwrap();
    let before = marked(&dir, &rels);
    assert_eq!(before.len(), 2);
    assert!(p.frames.iter().all(|f| before.contains(f)), "{p:?}");

    // Limit below what is left: same frames, same skipped count as the run's own stage 1.
    let job = plan_autolabel(&cfg, request("limit=3"), &host()).unwrap();
    let p = job.preview().unwrap();
    assert_eq!((p.matched, p.taken, p.over_limit), (6, 3, 1), "{p:?}");
    assert_eq!(p.skipped_done, 2, "{p:?}");
    assert_eq!(p.matched_by_status.unreviewed, 6, "{p:?}");
    assert_eq!(p.taken_by_status.unreviewed, 3, "{p:?}");
    let labeler = job.run_labeler();
    let sel = job.select(Some(&labeler), None).unwrap();
    let selected: Vec<String> = sel.frames.iter().map(|f| f.rel.clone()).collect();
    assert_eq!(selected, p.frames);
    assert_eq!(sel.skipped_done, p.skipped_done);
    let rep = run_job(&job, &LabelFake::default()).unwrap().result;
    assert_eq!(rep["frames"], p.taken, "{rep}");
    assert_eq!(rep["skipped_done"], p.skipped_done, "{rep}");
    let after = marked(&dir, &rels);
    let new: BTreeSet<&String> = after.iter().filter(|f| !before.contains(f)).collect();
    assert_eq!(new, p.frames.iter().collect(), "the frames it previewed");

    // The count of the left-out frames does not stop at the limit.
    let job = plan_autolabel(&cfg, request("limit=1"), &host()).unwrap();
    let p = job.preview().unwrap();
    assert_eq!((p.skipped_done, p.taken, p.over_limit), (5, 1, 0), "{p:?}");

    // again=1 takes them back, and a rerun would start from the same frames.
    let job = plan_autolabel(&cfg, request("again=1,limit=4"), &host()).unwrap();
    let p = job.preview().unwrap();
    assert_eq!((p.skipped_done, p.taken, p.over_limit), (0, 4, 2), "{p:?}");
    assert!(p.warnings[0].contains("same frames first"), "{p:?}");
    let rep = run_job(&job, &LabelFake::default()).unwrap().result;
    assert_eq!(rep["frames"], 4, "{rep}");
    assert_eq!(rep["skipped_done"], 0, "{rep}");

    // A class the labelers did not look for: the marker does not cover it.
    let mut wider = cfg.clone();
    wider.collect.classes.push("wombat".into());
    let job = plan_autolabel(&wider, request("limit=20"), &host()).unwrap();
    let p = job.preview().unwrap();
    assert_eq!((p.skipped_done, p.taken), (0, 6), "{p:?}");
}

#[test]
fn the_preview_needs_no_gpu_and_splits_by_status() {
    let t = tmp("preview-gates");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let mut cfg = config(&t.0);
    cfg.allow_large_downloads = false;
    labels::write(
        &Frame::resolve(&dir, &rels[0]).unwrap().labels,
        &reviewed_car(vec![]),
    )
    .unwrap();
    let skipped = LabelFile {
        status: Status::Skipped,
        ..LabelFile::default()
    };
    labels::write(&Frame::resolve(&dir, &rels[1]).unwrap().labels, &skipped).unwrap();
    assert!(
        plan_autolabel(&cfg, request("status=any"), &no_gpu()).is_err(),
        "a run needs a GPU"
    );
    let p = crate::training::autolabel::preview_autolabel(&cfg, request("status=any")).unwrap();
    assert_eq!(
        (p.matched_by_status.reviewed, p.matched_by_status.skipped),
        (1, 1),
        "{p:?}"
    );
    assert_eq!(p.matched_by_status.unreviewed, 1, "{p:?}");
    assert_eq!(p.taken_by_status, p.matched_by_status, "{p:?}");
    assert!(
        p.warnings.iter().any(|w| w.contains("leave exports")),
        "a reviewed frame may reopen: {p:?}"
    );
    // A run of SAM 3 needs a login; a count does not.
    let mut sam3 = request("status=unreviewed");
    sam3.sam3 = true;
    sam3.gdino = None;
    let p = crate::training::autolabel::preview_autolabel(&cfg, sam3).unwrap();
    let e = p.estimate.unwrap();
    let phrases = p.phrases as f64;
    assert!(
        phrases >= cfg.collect.classes.len() as f64,
        "synonyms count"
    );
    assert!(
        (e.total_seconds - (2.0 + 0.2 * phrases)).abs() < 1e-9,
        "{e:?}"
    );
    assert!(p.warnings.is_empty(), "no reviewed frame: {:?}", p.warnings);
    // A bad filter and an unknown class stay errors.
    let mut bad = request("");
    bad.classes = Some(vec!["unicorn".into()]);
    assert!(crate::training::autolabel::preview_autolabel(&cfg, bad).is_err());
}

#[test]
fn phrases_are_counted_as_the_script_folds_them() {
    let t = tmp("preview-phrases");
    let mut cfg = config(&t.0);
    cfg.collect.classes = vec!["car".into(), "truck".into()];
    cfg.training.synonyms = [
        (
            "car".to_string(),
            vec!["Pickup  Truck".to_string(), "car".to_string()],
        ),
        ("truck".to_string(), vec!["pickup truck".to_string()]),
    ]
    .into();
    let p = crate::training::autolabel::preview_autolabel(&cfg, request("")).unwrap();
    assert_eq!(p.phrases, 2, "{p:?}");
}

/// Two boxes of `label` on one object, accepted: an overlap for a reviewer to look at, unless the
/// class is unwanted.
fn overlapping(label: &str) -> LabelFile {
    let at = |bbox: [f32; 4]| labels::LabelBox {
        label: label.into(),
        bbox,
        source: Source::Model,
        score: Some(0.9),
        state: BoxState::Accepted,
        origin: Some(labels::Origin::Prediction),
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: Default::default(),
    };
    LabelFile {
        status: Status::Reviewed,
        reviewed_at: Some("2026-10-22T12:00:00.000Z".into()),
        boxes: vec![at([1.0, 1.0, 30.0, 20.0]), at([2.0, 1.0, 30.0, 20.0])],
        ..LabelFile::default()
    }
}

#[test]
fn a_job_and_the_page_agree_on_an_overlap_of_an_unwanted_class() {
    let t = tmp("preview-wanted");
    let rels = frames(&t.0, 4, false);
    let dir = t.0.join("dataset");
    let mut cfg = config(&t.0);
    cfg.collect.classes = vec!["car".into(), "person".into(), "dog".into()];
    cfg.collect.hidden_classes = vec!["dog".into()];
    let put = |i: usize, f: &LabelFile| {
        labels::write(&Frame::resolve(&dir, &rels[i]).unwrap().labels, f).unwrap();
    };
    put(0, &overlapping("car"));
    put(1, &overlapping("dog"));
    put(2, &overlapping("pottery"));
    put(3, &reviewed_car(vec![]));

    let query: BTreeMap<String, String> = [
        ("status", "reviewed"),
        ("auto", "overlaps"),
        ("source", "all"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .into();
    let mut page = queue::Filters::from_query(&query).unwrap();
    // Without the class rules the two unwanted frames count: the gap this closes.
    assert_eq!(queue::build(&dir, &page).items.len(), 3);
    page.wanted = Some((
        cfg.collect.classes.clone(),
        cfg.collect.hidden_classes.clone(),
    ));
    let on_page: Vec<String> = queue::build(&dir, &page)
        .items
        .into_iter()
        .map(|i| i.frame)
        .collect();
    assert_eq!(on_page, [rels[0].clone()]);

    let job = plan_autolabel(&cfg, request("status=reviewed,auto=overlaps"), &host()).unwrap();
    let selected: Vec<String> = job
        .select(None, None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.rel)
        .collect();
    assert_eq!(selected, on_page);
    assert_eq!(job.preview().unwrap().frames, on_page);
}

#[test]
fn the_page_the_preview_and_the_run_agree_on_auto_both() {
    let t = tmp("preview-both");
    let rels = frames(&t.0, 6, false);
    let dir = t.0.join("dataset");
    let mut cfg = config(&t.0);
    cfg.collect.classes = vec!["car".into(), "person".into(), "dog".into()];
    cfg.collect.hidden_classes = vec!["dog".into()];
    let found = |label: &str, labelers: &[&str], state: BoxState| labels::LabelBox {
        bbox: [1.0, 1.0, 30.0, 20.0],
        state,
        auto: labels::AutoInfo {
            labelers: labelers.iter().map(|k| ((*k).to_string(), 0.5)).collect(),
            ..Default::default()
        },
        ..labels::tests_support::auto_box(label)
    };
    let pair = ["sam3", "rfdetr"];
    let put = |i: usize, status: Status, boxes: Vec<labels::LabelBox>| {
        let f = LabelFile {
            status,
            boxes,
            ..LabelFile::default()
        };
        labels::write(&Frame::resolve(&dir, &rels[i]).unwrap().labels, &f).unwrap();
    };
    let open = Status::Unreviewed;
    put(0, open, vec![found("car", &["sam3"], BoxState::Pending)]);
    put(1, open, vec![found("car", &pair, BoxState::Pending)]);
    let two = found("car", &pair, BoxState::Pending);
    put(2, open, vec![two.clone(), two]);
    put(3, open, vec![found("dog", &pair, BoxState::Pending)]);
    put(
        4,
        Status::Reviewed,
        vec![found("car", &pair, BoxState::Accepted)],
    );
    // rels[5]: no label file.
    let want = [rels[2].clone(), rels[1].clone()];

    let query: BTreeMap<String, String> = [("status", "any"), ("auto", "both"), ("source", "all")]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .into();
    let mut page = queue::Filters::from_query(&query).unwrap();
    page.wanted = Some((
        cfg.collect.classes.clone(),
        cfg.collect.hidden_classes.clone(),
    ));
    let on_page: Vec<String> = queue::build(&dir, &page)
        .items
        .into_iter()
        .map(|i| i.frame)
        .collect();
    assert_eq!(on_page, want);

    let job = plan_autolabel(&cfg, request("status=any,auto=both"), &host()).unwrap();
    let p = job.preview().unwrap();
    assert_eq!(p.frames, want);
    assert_eq!((p.matched, p.taken), (2, 2), "{p:?}");
    // `limit` keeps the most agreed frame.
    let first = plan_autolabel(&cfg, request("status=any,auto=both,limit=1"), &host()).unwrap();
    assert_eq!(first.preview().unwrap().frames, want[..1]);

    let out = run_job(&job, &LabelFake::default()).unwrap();
    assert_eq!(out.result["frames"], p.taken, "{}", out.result);
    let mut sorted = want.to_vec();
    sorted.sort();
    assert_eq!(
        marked(&dir, &rels),
        sorted,
        "the run took what it previewed"
    );
}

/// Every file under `root` with its time and bytes.
fn tree(root: &Path) -> BTreeMap<PathBuf, (std::time::SystemTime, Vec<u8>)> {
    let mut out = BTreeMap::new();
    let mut todo = vec![root.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let e = e.unwrap();
            let path = e.path();
            if path.is_dir() {
                todo.push(path);
            } else {
                let at = e.metadata().unwrap().modified().unwrap();
                let bytes = std::fs::read(&path).unwrap();
                out.insert(path, (at, bytes));
            }
        }
    }
    out
}

#[test]
fn dry_run_writes_nothing() {
    use clap::Parser;
    #[derive(clap::Parser)]
    struct T {
        #[command(subcommand)]
        c: crate::training::cli::TrainCommand,
    }
    let t = tmp("preview-dry");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    labels::write(
        &Frame::resolve(&dir, &rels[0]).unwrap().labels,
        &overlapping("car"),
    )
    .unwrap();
    let mut cfg = config(&t.0);
    cfg.collect.dir = Some(dir);
    let path = t.0.join("config.json");
    cfg.save(&path).unwrap();
    let before = tree(&t.0);
    for filter in ["status=reviewed,auto=overlaps", "status=any,limit=1"] {
        let args = T::parse_from([
            "train",
            "autolabel",
            "--dry-run",
            "--sam3",
            "--frames",
            filter,
        ]);
        let args = crate::training::cli::TrainArgs { command: args.c };
        assert!(crate::training::cli::run(args, Some(path.clone())).unwrap());
    }
    let after = tree(&t.0);
    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "no file came or went"
    );
    assert!(before == after, "no file changed");
}

#[test]
fn the_summary_says_what_the_tiers_did() {
    let mut rep = AutolabelReport::default();
    assert_eq!(
        rep.labelers_notes(),
        "",
        "a run of an older kind reads as before"
    );
    rep.count_labelers(&triage::LabelersReport {
        apply: triage::ApplyReport {
            merge: labels::MergeReport {
                added: 6,
                ..labels::MergeReport::default()
            },
            ..triage::ApplyReport::default()
        },
        dismissed: 3,
        audit: 1,
        likely: 2,
        ..triage::LabelersReport::default()
    });
    assert_eq!(
        (rep.added, rep.dismissed, rep.audit, rep.likely),
        (3, 3, 1, 2)
    );
    let s = rep.labelers_notes();
    assert!(s.contains("2 boxes likely real"), "{s}");
    assert!(
        s.contains("3 boxes below score 0.15 hidden, 1 more kept visible"),
        "{s}"
    );
    assert!(s.contains("Show hidden on the Review tab"), "{s}");
    let mut spots = AutolabelReport::default();
    spots.count_labelers(&triage::LabelersReport {
        apply: triage::ApplyReport {
            merge: labels::MergeReport {
                added: 6,
                ..labels::MergeReport::default()
            },
            ..triage::ApplyReport::default()
        },
        spot_dismissed: 2,
        parked_accepted: 3,
        parked_new: 1,
        ..triage::LabelersReport::default()
    });
    assert_eq!(
        (spots.added, spots.spot_dismissed, spots.parked_accepted),
        (3, 2, 3)
    );
    let s = spots.labelers_notes();
    assert!(s.contains("2 boxes hidden at spots you rejected"), "{s}");
    assert!(s.contains("3 boxes accepted at parked spots"), "{s}");
}

/// `rels` saved as reviewed with a rejected car at `bbox`.
fn rejected_on(dir: &Path, rels: &[String], bbox: [f32; 4]) {
    for rel in rels {
        let f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![labels::LabelBox {
                bbox,
                state: BoxState::Rejected,
                ..labels::tests_support::auto_box("car")
            }],
            ..LabelFile::default()
        };
        labels::write(&Frame::resolve(dir, rel).unwrap().labels, &f).unwrap();
    }
}

fn car_at_spot(score: f32) -> serde_json::Value {
    serde_json::json!([{"label": "car", "bbox": [4, 4, 40, 40], "score": score, "agree": false,
                        "sources": {"gdino": score}}])
}

#[test]
fn a_run_hides_boxes_at_rejected_spots_from_a_snapshot_taken_once() {
    const SPOT: [f32; 4] = [4.0, 4.0, 40.0, 40.0];
    let t = tmp("spot-hidden");
    let rels = frames(&t.0, 4, false);
    let dir = t.0.join("dataset");
    rejected_on(&dir, &rels[..2], SPOT);
    // While the script runs a reviewer accepts a car there. The run read the spots at its start,
    // so this mixed spot still counts until the next run.
    let accepted = LabelFile {
        status: Status::Reviewed,
        boxes: vec![labels::LabelBox {
            bbox: SPOT,
            state: BoxState::Accepted,
            ..labels::tests_support::auto_box("car")
        }],
        ..LabelFile::default()
    };
    let fake = LabelFake {
        boxes: Some(car_at_spot(0.3)),
        saved_during_run: vec![(Frame::resolve(&dir, &rels[3]).unwrap().labels, accepted)],
        ..LabelFake::default()
    };
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["spot_dismissed"], 1, "{}", out.result);
    assert_eq!(out.result["added"], 0, "hidden boxes are left out");
    assert!(
        out.summary.contains("1 boxes hidden at spots you rejected"),
        "{}",
        out.summary
    );
    let b = &labels_of(&dir, &rels[2]).boxes[0];
    assert_eq!(
        b.reason.as_deref(),
        Some(labels::bulk::SPOT_DISMISSED_REASON)
    );
    assert_eq!(b.auto.spot_rejections, Some(2));
    // The next run sees it: a mixed spot is no spot any more.
    let f = LabelFile::default();
    labels::write(&Frame::resolve(&dir, &rels[2]).unwrap().labels, &f).unwrap();
    let fake = LabelFake {
        boxes: Some(car_at_spot(0.3)),
        ..LabelFake::default()
    };
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["spot_dismissed"], 0, "{}", out.result);
    assert_eq!(labels_of(&dir, &rels[2]).boxes[0].state, BoxState::Pending);
}

#[test]
fn a_run_accepts_boxes_at_parked_spots_and_the_frame_can_be_saved_in_bulk() {
    let t = tmp("spot-parked");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let at = labels::spots::Place {
        camera: "cam",
        width: 64,
        height: 48,
        label: "car",
        bbox: [4.0, 4.0, 40.0, 40.0],
    };
    labels::spots::add_parked(&dir, &at, "2026-10-01").unwrap();
    let fake = LabelFake {
        boxes: Some(car_at_spot(0.4)),
        ..LabelFake::default()
    };
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &fake).unwrap();
    assert_eq!(out.result["parked_accepted"], 2, "{}", out.result);
    assert_eq!(out.result["added"], 0);
    assert!(
        out.summary.contains("2 boxes accepted at parked spots"),
        "{}",
        out.summary
    );
    let f = labels_of(&dir, &rels[0]);
    assert_eq!(f.status, Status::Unreviewed);
    assert_eq!(f.boxes[0].state, BoxState::Accepted);
    assert_eq!(
        f.boxes[0].reason.as_deref(),
        Some(labels::spots::PARKED_REASON)
    );
    assert!(labels::repeats::repeats_only(&f));
    let classes = cfg.collect.classes.clone();
    let filters = queue::Filters::default();
    let o = labels::bulk::bulk_accept_repeats(&dir, &filters, &classes, 10, "t", false, &|_| true);
    assert_eq!(o.done_count, 2, "{o:?}");
    let saved = labels_of(&dir, &rels[0]);
    assert_eq!(saved.status, Status::Reviewed);
    assert_eq!(
        saved.boxes[0].reason.as_deref(),
        Some(labels::spots::PARKED_REASON)
    );
}

#[test]
fn a_sam3_run_reads_no_spot() {
    let t = tmp("spot-sam3");
    let rels = frames(&t.0, 4, false);
    let dir = t.0.join("dataset");
    rejected_on(&dir, &rels[..2], [4.0, 4.0, 40.0, 40.0]);
    let cfg = config(&t.0);
    let fake = LabelFake {
        boxes: Some(
            serde_json::json!([{"label": "car", "bbox": [4, 4, 40, 40], "score": 0.3,
                                        "agree": false, "sources": {"sam3": 0.3}}]),
        ),
        ..LabelFake::default()
    };
    let out = run_job(&sam3_job(&cfg), &fake).unwrap();
    assert_eq!(out.result["spot_dismissed"], 0, "{}", out.result);
    assert_eq!(labels_of(&dir, &rels[2]).boxes[0].state, BoxState::Pending);
}

/// The label files of `rels` (the failure seam of [`LabelFake::broken_during_run`]).
fn label_paths(dir: &Path, rels: &[String]) -> Vec<PathBuf> {
    rels.iter()
        .map(|r| Frame::resolve(dir, r).unwrap().labels)
        .collect()
}

#[test]
fn a_frame_that_cannot_be_written_keeps_the_proposals_and_says_where() {
    let t = tmp("write-fails");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let broken = Frame::resolve(&dir, &rels[1]).unwrap().labels;
    std::fs::remove_file(dir.join(crate::collect::layout::REVIEW_QUEUE_FILE)).unwrap();
    let fake = LabelFake {
        broken_during_run: vec![broken.clone()],
        ..LabelFake::default()
    };
    let out = run_job(&job, &fake).unwrap();
    let kept = job.work.join("proposals.jsonl");
    assert!(kept.is_file(), "the only copy of the frame's proposals");
    let rep = &out.result;
    assert_eq!(
        (&rep["frames_saved"], &rep["frames_failed"]),
        (&2.into(), &1.into()),
        "{rep}"
    );
    // The two frames written, two boxes each: the failed frame's boxes are not counted.
    assert_eq!(rep["added"], 4, "{rep}");
    assert!(
        out.summary
            .contains("1 of 3 label files could not be written"),
        "{}",
        out.summary
    );
    assert!(
        out.summary.contains(&kept.display().to_string()),
        "{}",
        out.summary
    );
    assert_eq!(std::fs::read_to_string(&kept).unwrap().lines().count(), 3);
    // Queued: the saved frames only.
    assert_eq!(review_queue(&dir).len(), 2);

    // Once the folder is writable again, a rerun labels only the frame that was lost.
    std::fs::remove_file(&broken).unwrap();
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &LabelFake::default()).unwrap();
    assert_eq!(out.result["frames"], 1, "{}", out.result);
    assert_eq!(out.result["frames_failed"], 0, "{}", out.result);
    assert!(!job.work.exists(), "a clean run removes its work folder");
    assert!(
        !out.summary.contains("could not be written"),
        "{}",
        out.summary
    );
}

/// The file reads and only the write fails, as on a full disk: the triage ran, and what it would
/// have added must not be reported as added.
#[cfg(unix)]
#[test]
fn a_box_that_was_not_written_is_not_counted_as_added() {
    use std::os::unix::fs::PermissionsExt;
    let t = tmp("write-fails-added");
    let rels = frames_every(&t.0, 2, false, 86_400);
    let dir = t.0.join("dataset");
    let day = Frame::resolve(&dir, &rels[0])
        .unwrap()
        .labels
        .parent()
        .unwrap()
        .to_path_buf();
    let set = |mode| std::fs::set_permissions(&day, std::fs::Permissions::from_mode(mode));
    set(0o555).unwrap();
    if std::fs::File::create(day.join("probe")).is_ok() {
        // Root ignores the mode: nothing to test here.
        let _ = set(0o755);
        return;
    }
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let out = run_job(&job, &LabelFake::default());
    set(0o755).unwrap();
    let rep = out.unwrap().result;
    assert_eq!(
        (&rep["frames_saved"], &rep["frames_failed"]),
        (&1.into(), &1.into()),
        "{rep}"
    );
    assert_eq!(rep["added"], 2, "the saved frame's two boxes: {rep}");
}

#[test]
fn a_run_that_saves_no_frame_fails_and_keeps_the_proposals() {
    let t = tmp("write-fails-all");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake {
        broken_during_run: label_paths(&dir, &rels),
        ..LabelFake::default()
    };
    let text = format!("{:#}", run_job(&job, &fake).unwrap_err());
    assert!(
        text.contains("no label file could be written (2 tried"),
        "{text}"
    );
    let kept = job.work.join("proposals.jsonl");
    assert!(text.contains(&kept.display().to_string()), "{text}");
    assert_eq!(std::fs::read_to_string(&kept).unwrap().lines().count(), 2);
}

#[test]
fn a_failed_run_with_a_failed_write_keeps_the_proposals_too() {
    for (broken, says) in [
        (1, "1 of 2 label files could not be written"),
        (2, "no label file could be written (2 tried"),
    ] {
        let t = tmp("write-fails-crash");
        let rels = frames(&t.0, 4, false);
        let dir = t.0.join("dataset");
        let cfg = config(&t.0);
        let job =
            plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
        // The queue's order is newest first: the last two frames finish, then the script dies,
        // and `broken` of those two cannot be saved.
        let fake = LabelFake {
            fail_after: Some(2),
            broken_during_run: label_paths(&dir, &rels[4 - broken..]),
            ..LabelFake::default()
        };
        let text = format!("{:#}", run_job(&job, &fake).unwrap_err());
        assert!(text.contains("the GPU fell off the bus"), "{text}");
        assert!(text.contains(says), "{text}");
        let kept = job.work.join("proposals.jsonl");
        assert!(text.contains(&kept.display().to_string()), "{text}");
        assert!(kept.is_file(), "the only copy of those proposals");
    }
}

#[test]
fn a_stopped_run_says_which_frames_were_not_saved() {
    let t = tmp("write-fails-stop");
    let rels = frames(&t.0, 4, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    let fake = LabelFake {
        stop_after: Some(3),
        // Newest first: the last frame is one of the two that finish.
        broken_during_run: label_paths(&dir, &rels[3..]),
        ..LabelFake::default()
    };
    let cancel = AtomicBool::new(false);
    let ctx = TaskContext {
        runner: &fake,
        fetcher: &fake,
    };
    let out = job.run(ctx, &mut |_| {}, &cancel).unwrap();
    assert!(
        out.summary.contains("those 1 were saved"),
        "the frame not written is not called saved: {}",
        out.summary
    );
    assert!(
        out.summary
            .contains("1 of 2 label files could not be written"),
        "{}",
        out.summary
    );
    assert!(job.work.join("proposals.jsonl").is_file());
}

#[test]
fn a_model_run_counts_what_it_saved() {
    let t = tmp("write-fails-model");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let dets: BTreeMap<String, Vec<Det>> = rels
        .iter()
        .map(|r| (r.clone(), vec![det("car", 0.9, [1.0, 1.0, 30.0, 20.0])]))
        .collect();
    let run = |broken: &[String]| {
        let mut req = AutolabelRequest::new(FrameFilter::default());
        req.with_model = Some("IPcam-general".into());
        let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
        job.detector = Some(Arc::new(StallingDetector {
            dets: dets.clone(),
            before_returning: label_paths(&dir, broken),
        }));
        run_job(&job, &LabelFake::default())
    };
    // One frame fails: the run is done, and says so.
    let out = run(&rels[..1]).unwrap();
    let rep = &out.result;
    assert_eq!(
        (&rep["frames_saved"], &rep["frames_failed"]),
        (&2.into(), &1.into()),
        "{rep}"
    );
    assert_eq!(rep["added"], 2, "{rep}");
    assert!(
        out.summary
            .contains("1 of 3 label files could not be written"),
        "{}",
        out.summary
    );
    // The two saved frames are skipped by their marker, and the last one still cannot be saved.
    // No frame saved: a failure.
    std::fs::remove_file(Frame::resolve(&dir, &rels[0]).unwrap().labels).unwrap();
    let text = format!("{:#}", run(&rels[..1]).unwrap_err());
    assert!(
        text.contains("no label file could be written (1 tried"),
        "{text}"
    );
}

#[test]
fn a_missing_only_run_counts_what_it_saved() {
    let t = tmp("write-fails-missing");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    for r in &rels {
        labels::write(
            &Frame::resolve(&dir, r).unwrap().labels,
            &reviewed_car(vec![]),
        )
        .unwrap();
    }
    let cfg = config(&t.0);
    let far = det("person", 0.9, [40.0, 20.0, 60.0, 46.0]);
    let dets: BTreeMap<String, Vec<Det>> = rels
        .iter()
        .map(|r| (r.clone(), vec![far.clone()]))
        .collect();
    let mut req =
        AutolabelRequest::new(FrameFilter::parse("status=reviewed", "unreviewed").unwrap());
    req.with_model = Some("IPcam-general".into());
    req.missing_only = true;
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    job.detector = Some(Arc::new(StallingDetector {
        dets,
        before_returning: label_paths(&dir, &rels[..1]),
    }));
    let out = run_job(&job, &LabelFake::default()).unwrap();
    let rep = &out.result;
    assert_eq!(
        (
            &rep["frames_saved"],
            &rep["frames_failed"],
            &rep["candidates"]
        ),
        (&1.into(), &1.into(), &1.into()),
        "{rep}"
    );
    assert!(
        out.summary
            .contains("1 of 2 label files could not be written"),
        "{}",
        out.summary
    );
}

/// [`FakeDetector`] for frames that were selected fine and fail to save: their label files break
/// once the model ran.
struct StallingDetector {
    dets: BTreeMap<String, Vec<Det>>,
    before_returning: Vec<PathBuf>,
}

impl Detector for StallingDetector {
    fn detect(
        &self,
        _: &Config,
        _: &ModelConfig,
        device: Option<&str>,
        images: &[Image],
        _: f32,
        _: &AtomicBool,
    ) -> Result<super::super::relabel::Detections> {
        for p in &self.before_returning {
            unwritable(p);
        }
        Ok(super::super::relabel::Detections {
            frames: images
                .iter()
                .map(|i| {
                    (
                        i.frame.clone(),
                        self.dets.get(&i.frame).cloned().unwrap_or_default(),
                    )
                })
                .collect(),
            device: device.unwrap_or("openvino:cpu").to_string(),
            cancelled: false,
        })
    }
}

/// A `status=any` re-label over confirmed-empty and skipped frames: the marker and nothing else,
/// so the frame stays in the exports with its hard negatives.
#[test]
fn a_model_run_leaves_closed_frames_closed() {
    let t = tmp("relabel-closed");
    let rels = frames(&t.0, 2, false);
    let dir = t.0.join("dataset");
    let closed = [
        LabelFile {
            status: Status::Reviewed,
            empty_confirmed: true,
            boxes: vec![decided("car", [5.0, 5.0, 30.0, 20.0], BoxState::Rejected)],
            ..LabelFile::default()
        },
        LabelFile {
            status: Status::Skipped,
            ..LabelFile::default()
        },
    ];
    for (r, f) in rels.iter().zip(&closed) {
        labels::write(&Frame::resolve(&dir, r).unwrap().labels, f).unwrap();
    }
    let queue_file = dir.join(crate::collect::layout::REVIEW_QUEUE_FILE);
    std::fs::remove_file(&queue_file).unwrap();
    let cfg = config(&t.0);
    let mut req = AutolabelRequest::new(FrameFilter::parse("status=any", "unreviewed").unwrap());
    req.with_model = Some("IPcam-general".into());
    let mut job = plan_autolabel(&cfg, req, &no_gpu()).unwrap();
    let dets: BTreeMap<String, Vec<Det>> = rels
        .iter()
        .map(|r| {
            (
                r.clone(),
                vec![det("person", 0.9, [40.0, 20.0, 60.0, 46.0])],
            )
        })
        .collect();
    job.detector = Some(Arc::new(FakeDetector::new(dets)));
    let out = run_job(&job, &LabelFake::default()).unwrap();
    let rep = &out.result;
    assert_eq!(
        (&rep["added"], &rep["disagreements"]),
        (&0.into(), &0.into()),
        "{rep}"
    );
    assert_eq!(rep["filtered_closed"], 2, "{rep}");
    for (r, before) in rels.iter().zip(&closed) {
        let mut f = labels_of(&dir, r);
        assert!(f.auto_run.is_some(), "{r}: the marker, so a rerun skips it");
        f.auto_run = None;
        assert_eq!(&f, before);
    }
    assert!(review_queue(&dir).is_empty(), "nothing to review");
    assert!(
        out.summary
            .contains("2 detections left out on skipped or confirmed-empty frames"),
        "{}",
        out.summary
    );
}

/// A run whose process died leaves `work` with its marker: the storage sweep writes the finished
/// proposals through the run's own write stage (markers included) and only then removes the
/// folder; while a label file cannot be written, the folder and its marker stay.
#[test]
fn a_crashed_runs_proposals_are_written_by_the_sweep_before_its_folder_goes() {
    use crate::storage::busy::marker_of;
    use crate::storage::sweep;
    let t = tmp("crash-sweep");
    let rels = frames(&t.0, 3, false);
    let dir = t.0.join("dataset");
    let cfg = config(&t.0);
    let job = plan_autolabel(&cfg, AutolabelRequest::new(FrameFilter::default()), &host()).unwrap();
    // Every write fails, so the run keeps its proposals: the state a crash before the write stage
    // leaves, apart from the marker.
    let paths = label_paths(&dir, &rels);
    let fake = LabelFake {
        broken_during_run: paths.clone(),
        ..LabelFake::default()
    };
    run_job(&job, &fake).unwrap_err();
    assert!(job.work.join(PROPOSALS_FILE).is_file());
    assert!(job.work.join(JOB_FILE).is_file());
    assert!(
        !marker_of(&job.work).exists(),
        "an orderly end: kept on purpose, not crashed"
    );
    let s = crate::storage::Settings::for_config(&cfg);
    let mut rep = sweep::Report::default();
    sweep::sweep_work(&s, &mut rep);
    assert!(job.work.is_dir(), "a kept folder is never swept");

    // The process died instead: the marker stays behind, unlocked.
    std::fs::write(marker_of(&job.work), b"1\n").unwrap();
    for p in &paths[1..] {
        std::fs::remove_file(p).unwrap();
    }
    std::fs::remove_file(dir.join(crate::collect::layout::REVIEW_QUEUE_FILE)).unwrap();
    let mut rep = sweep::Report::default();
    sweep::sweep_work(&s, &mut rep);
    assert!(job.work.is_dir(), "one frame could not be written: kept");
    assert!(
        marker_of(&job.work).exists(),
        "so the next sweep tries again"
    );
    assert_eq!(rep.kept.len(), 1, "{rep:?}");
    assert!(rep.kept[0].contains("1 of 3"), "{}", rep.kept[0]);
    assert!(rep.removed.is_empty());
    let first = labels_of(&dir, &rels[1]);

    std::fs::remove_file(&paths[0]).unwrap();
    let mut rep = sweep::Report::default();
    sweep::sweep_work(&s, &mut rep);
    assert!(rep.kept.is_empty(), "{rep:?}");
    assert_eq!(rep.removed.len(), 1);
    assert!(!job.work.exists());
    assert!(!marker_of(&job.work).exists());
    for r in &rels {
        let f = labels_of(&dir, r);
        assert!(f.auto_run.is_some(), "{r}: the run's marker");
        assert_eq!(f.boxes.len(), first.boxes.len(), "{r}");
    }
    // Written twice, merged once.
    assert_eq!(labels_of(&dir, &rels[1]).boxes, first.boxes);
    assert_eq!(review_queue(&dir).len(), 3);
}
