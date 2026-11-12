//! The dataset check over fakes: the export's [`Fake`] for uv and downloads, a fake
//! `train_model.py` that "trains" and writes a model, a fake predictor that answers each frame. No
//! Python, GPU or network.

use super::*;
use crate::collect::labels::bulk::test_support::put_frame_at;
use crate::collect::labels::refine::Kind;
use crate::collect::labels::{self, BoxState, Frame, LabelBox, LabelFile, Source, Status};
use crate::resources::catalog::Resource;
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::{ChildFailed, Cmd, CommandRunner, Fetcher};
use crate::training::autolabel::relabel::Detections;
use crate::training::autolabel::triage::Det;
use crate::training::capability::{self, Capability};
use crate::training::refine::{RefineRequest, plan_refine, report};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-refine-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// A labelled car on every frame.
const CAR: [f32; 4] = [100.0, 100.0, 200.0, 180.0];
/// Where the fake model finds an unlabelled person.
const FAR: [f32; 4] = [240.0, 120.0, 290.0, 220.0];
const CAMERAS: [&str; 4] = ["alpha", "bravo", "charlie", "delta"];
const PER_CAMERA: usize = 3;

/// The fake `train_model.py`.
#[derive(Default)]
struct TrainerFake {
    inner: Fake,
    /// Training runs so far.
    trains: AtomicUsize,
    /// Fail the n-th training run (1-based), with a checkpoint left behind.
    fail_on: Option<usize>,
    /// Cancel at the start of the n-th training run.
    cancel_on: Option<usize>,
    /// Put a held-out frame into train.txt.
    leak: bool,
    /// Put a held-out frame into train.txt under another name, as the dataset would after
    /// cutting a long name or numbering a name taken twice, with its source in the dataset's
    /// record.
    leak_renamed: bool,
    /// What each run trained on: `train.txt` + `valid.txt` names.
    trained: Mutex<Vec<String>>,
    /// The `export.json` of each run's fold export.
    exports: Mutex<Vec<serde_json::Value>>,
}

impl CommandRunner for TrainerFake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        let d = cmd.display();
        if !d.contains("train_model.py") {
            return self.inner.run(cmd, on_line, cancel);
        }
        if d.contains("--self-check") {
            on_line(
                "@train {\"event\":\"self_check\",\"cuda\":true,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}",
            );
            return Ok(());
        }
        let n = self.trains.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cancel_on == Some(n) {
            cancel.store(true, Ordering::SeqCst);
            return Err(anyhow!(Cancelled));
        }
        // Early stopping is off and every epoch runs.
        assert_eq!(arg_after(cmd, "--patience").unwrap(), PathBuf::from("0"));
        assert_eq!(arg_after(cmd, "--anneal").unwrap(), PathBuf::from("0"));
        let data = arg_after(cmd, "--data").unwrap();
        if self.leak {
            let held = std::fs::read_dir(data.join("heldout").join("images"))
                .unwrap()
                .flatten()
                .next()
                .unwrap()
                .file_name();
            let mut t = std::fs::read_to_string(data.join("train.txt")).unwrap();
            // Held-out images carry the dataset's own `s0-` name already.
            t.push_str(&format!("./images/{}\n", held.to_string_lossy()));
            std::fs::write(data.join("train.txt"), t).unwrap();
        }
        if self.leak_renamed {
            let held = std::fs::read_dir(data.join("heldout").join("images"))
                .unwrap()
                .flatten()
                .next()
                .unwrap()
                .path();
            let flat = held
                .file_name()
                .unwrap()
                .to_string_lossy()
                .trim_start_matches("s0-")
                .to_string();
            let name = "s0-ir.renamed-2.jpg";
            std::fs::copy(&held, data.join("images").join(name)).unwrap();
            let mut t = std::fs::read_to_string(data.join("train.txt")).unwrap();
            t.push_str(&format!("./images/{name}\n"));
            std::fs::write(data.join("train.txt"), t).unwrap();
            let record = data.join(crate::training::dataset::IMAGE_SOURCES_FILE);
            let mut m: BTreeMap<String, PathBuf> =
                serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
            // The fold's export holds its held-out frames under images/val.
            let export = data.parent().unwrap().join("export");
            m.insert(name.into(), export.join("images").join("val").join(flat));
            std::fs::write(&record, serde_json::to_vec(&m).unwrap()).unwrap();
        }
        for list in ["train.txt", "valid.txt"] {
            let t = std::fs::read_to_string(data.join(list)).unwrap();
            self.trained
                .lock()
                .unwrap()
                .extend(t.lines().map(str::to_string));
        }
        let report = data.parent().unwrap().join("export").join("export.json");
        if let Ok(b) = std::fs::read(&report) {
            self.exports
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&b).unwrap());
        }
        on_line(
            r#"@train {"event":"start","device":"cuda:0","train_images":9,"valid_images":3,"epochs":2}"#,
        );
        for e in 1..=2 {
            on_line(&format!(
                "@train {{\"event\":\"epoch\",\"epoch\":{e},\"epochs\":2,\"loss\":{}}}",
                3.0 - f64::from(e)
            ));
        }
        if self.fail_on == Some(n) {
            let out = arg_after(cmd, "--work").unwrap().join("rfdetr-out");
            std::fs::create_dir_all(&out).unwrap();
            std::fs::write(out.join("training_config.json"), "{}").unwrap();
            std::fs::write(out.join("checkpoint_best_total.pth"), vec![7u8; 2048]).unwrap();
            let lines = vec!["RuntimeError: the export broke".to_string()];
            on_line(&lines[0]);
            return Err(anyhow::Error::new(ChildFailed {
                program: "python".into(),
                status: "exit status: 1".into(),
                lines,
            }));
        }
        let name = arg_after(cmd, "--name").unwrap().display().to_string();
        let out = arg_after(cmd, "--out-dir").unwrap();
        let classes: Vec<String> = std::fs::read_to_string(data.join("classes.txt"))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        std::fs::write(out.join(format!("{name}.onnx")), vec![1u8; 4096]).unwrap();
        let names: String = classes.iter().map(|c| format!("  - {c}\n")).collect();
        std::fs::write(out.join(format!("{name}.yaml")), format!("NAMES:\n{names}")).unwrap();
        on_line(&format!(
            "@train {}",
            serde_json::json!({"event": "done", "onnx": out.join(format!("{name}.onnx")),
                "yaml": "y", "family": "rfdetr", "classes": classes, "imgsz": 384,
                "seconds": 2.5, "device": "cuda:0"})
        ));
        Ok(())
    }
}

impl Fetcher for TrainerFake {
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

fn trainer() -> TrainerFake {
    TrainerFake {
        inner: Fake::new(),
        ..TrainerFake::default()
    }
}

type Meanwhile = Box<dyn FnOnce(&[Image]) + Send>;
/// A [`PredictorFake::on_call`] hook.
type OnCall = Box<dyn FnMut(&[Image]) + Send>;

/// The fake fold model: on every frame the labelled car and an unlabelled person.
#[derive(Default)]
struct PredictorFake {
    calls: AtomicUsize,
    /// What each call was asked: (model, frames).
    asked: Mutex<Vec<(FoldModel, Vec<String>)>>,
    /// Called during the first call: a reviewer saving frames while the check runs.
    meanwhile: Mutex<Option<Meanwhile>>,
    /// Called during every call.
    on_call: Mutex<Option<OnCall>>,
}

impl FoldPredictor for PredictorFake {
    fn predict(
        &self,
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        _cancel: &AtomicBool,
    ) -> Result<Detections> {
        assert_eq!(threshold, judge::MATCH_MIN_SCORE);
        assert!(model.onnx.is_file() && model.yaml.is_file());
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.asked.lock().unwrap().push((
            model.clone(),
            images.iter().map(|i| i.frame.clone()).collect(),
        ));
        if let Some(f) = self.meanwhile.lock().unwrap().take() {
            f(images);
        }
        if let Some(f) = self.on_call.lock().unwrap().as_mut() {
            f(images);
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
            device: "ort:cuda".into(),
            cancelled: false,
        })
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

fn path_of(dir: &Path, rel: &str) -> PathBuf {
    Frame::resolve(dir, rel).unwrap().labels
}

fn labels_of(dir: &Path, rel: &str) -> LabelFile {
    labels::read(&path_of(dir, rel)).unwrap().unwrap()
}

fn edit(dir: &Path, rel: &str, f: impl FnOnce(&mut LabelFile)) {
    let mut l = labels_of(dir, rel);
    f(&mut l);
    labels::write(&path_of(dir, rel), &l).unwrap();
}

/// `PER_CAMERA` reviewed frames on each of the four cameras (four camera days), each with an
/// accepted car; their rels.
fn dataset(dir: &Path) -> Vec<String> {
    let mut rels = Vec::new();
    for cam in CAMERAS {
        for i in 0..PER_CAMERA {
            let rel = put_frame_at(
                dir,
                cam,
                &format!("20261021T0102030{i}Z-{cam}"),
                (320, 240),
                None,
                vec![accepted("car", CAR)],
            );
            edit(dir, &rel, |l| {
                l.status = Status::Reviewed;
                l.reviewed_at = Some("2026-10-22T00:00:00Z".into());
            });
            rels.push(rel);
        }
    }
    rels
}

fn config(root: &Path) -> Config {
    let mut c = Config {
        download_dir: Some(root.to_path_buf()),
        allow_large_downloads: true,
        ..Config::default()
    };
    c.seed_default_model();
    c.collect.dir = Some(root.join("dataset"));
    c
}

fn host() -> Capability {
    capability::host_for_tests()
}

fn plan(root: &Path, predictor: &Arc<PredictorFake>) -> RefineJob {
    let mut job = plan_refine(&config(root), RefineRequest::default(), &host()).unwrap();
    job.min_frames = 4;
    job.predictor = Some(predictor.clone());
    job
}

fn run(job: &RefineJob, fake: &TrainerFake) -> Result<TaskOutcome> {
    run_cancellable(job, fake, &AtomicBool::new(false))
}

fn run_cancellable(
    job: &RefineJob,
    fake: &TrainerFake,
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

/// One stage report: (stage, step, steps, percent, detail).
type Shown = (String, usize, usize, u8, String);

/// `run`, keeping every stage report.
fn run_shown(job: &RefineJob, fake: &TrainerFake) -> (Result<TaskOutcome>, Vec<Shown>) {
    let mut shown = Vec::new();
    let out = job.run(
        TaskContext {
            runner: fake,
            fetcher: fake,
        },
        &mut |r| {
            if let TaskReport::Stage {
                stage,
                step,
                steps,
                percent,
                detail,
                ..
            } = r
            {
                shown.push((stage.to_string(), step, steps, percent, detail.to_string()));
            }
        },
        &AtomicBool::new(false),
    );
    (out, shown)
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

fn is_label_file(p: &Path) -> bool {
    p.to_string_lossy().ends_with(".labels.json")
}

/// The pretty JSON a label write produces, of `l` without its suggestions.
fn bytes_without_refine(l: &LabelFile) -> Vec<u8> {
    let mut l = l.clone();
    l.refine.clear();
    let mut b = serde_json::to_vec_pretty(&l).unwrap();
    b.push(b'\n');
    b
}

/// The fold (0-based) of every rel, as the job folds them.
fn folds_of(job: &RefineJob) -> BTreeMap<String, usize> {
    let s = job.select();
    s.frames
        .iter()
        .map(|f| {
            (
                f.rel.jpg(),
                s.folds[&(f.rel.camera.clone(), f.rel.date.clone())],
            )
        })
        .collect()
}

#[test]
fn a_check_writes_suggestions_and_the_report_and_nothing_else() {
    let t = tmp("invariant");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    // The data root's own files: the config, the models folder, runs.json, benchmark.json.
    let cfg = config(&t.0);
    let cfg_path = t.0.join("blue_onyx_prism_config.json");
    cfg.save(&cfg_path).unwrap();
    std::fs::create_dir_all(t.0.join("models")).unwrap();
    std::fs::write(t.0.join("models").join("mine.onnx"), b"model").unwrap();
    let runs = crate::training::runs_file(&cfg);
    std::fs::create_dir_all(runs.parent().unwrap()).unwrap();
    std::fs::write(&runs, "[]").unwrap();
    std::fs::write(t.0.join("benchmark.json"), b"{\"models\":[]}").unwrap();
    let watched = |root: &Path| -> BTreeMap<PathBuf, Vec<u8>> {
        let mut m = files(&root.join("models"));
        for p in [&cfg_path, &runs, &root.join("benchmark.json")] {
            m.insert(p.clone(), std::fs::read(p).unwrap());
        }
        m
    };
    let kept = watched(&t.0);
    let before = files(&dir);
    let predictor = Arc::new(PredictorFake::default());
    let fake = trainer();
    let job = plan(&t.0, &predictor);
    assert_eq!(job.kind(), TaskKind::Refine);
    assert_eq!(job.steps(), 6);
    assert!(job.work.starts_with(t.0.join("training").join("refine")));
    let out = run(&job, &fake).unwrap();
    assert!(
        out.summary.contains("No box was changed"),
        "{}",
        out.summary
    );
    // The new count covers every kind, not the last one named.
    assert!(
        out.summary.contains(
            "12 missing boxes, 0 labels to verify, 0 other classes (12 of these suggestions new \
             since the last check, on 12 frames written)"
        ),
        "{}",
        out.summary
    );
    let r = &out.result;
    assert_eq!(r["folds"], 4, "{r}");
    assert_eq!(r["folds_done"], 4);
    assert_eq!(r["frames_checked"], 12);
    assert_eq!(r["frames_written"], 12);
    assert_eq!(r["found"]["add"], 12);
    assert_eq!(r["written"]["add"], 12);
    assert_eq!(r["recall"]["boxes"], 12);
    assert_eq!(r["recall"]["found"], 12);
    assert_eq!(r["cancelled"], false);
    // Four folds trained, each on the other three cameras, and asked about its own.
    assert_eq!(fake.trains.load(Ordering::SeqCst), 4);
    let asked = predictor.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 4);
    let mut checked: Vec<String> = asked.iter().flat_map(|(_, f)| f.clone()).collect();
    checked.sort();
    let mut all = rels.clone();
    all.sort();
    assert_eq!(checked, all, "every frame checked exactly once");
    for (i, (m, frames)) in asked.iter().enumerate() {
        assert_eq!(m.name, format!("refine-fold-{}", i + 1));
        assert!(m.onnx.starts_with(&job.work), "{}", m.onnx.display());
        let cams: BTreeSet<&str> = frames
            .iter()
            .map(|f| f.split('/').nth(1).unwrap())
            .collect();
        assert_eq!(cams.len(), 1, "a fold is whole camera days: {frames:?}");
    }
    // What the folds trained on never included a frame they were asked about.
    let trained = fake.trained.lock().unwrap().clone();
    assert_eq!(
        trained.len(),
        4 * 9,
        "3 cameras x 3 frames per fold: {trained:?}"
    );
    // The label files: equal to their former selves once `refine` is cleared.
    let after = files(&dir);
    for (p, was) in &before {
        let now = &after[p];
        if is_label_file(p) {
            let l: LabelFile = serde_json::from_slice(now).unwrap();
            assert_eq!(&bytes_without_refine(&l), was, "{}", p.display());
            assert_eq!(l.refine.len(), 1, "{}", p.display());
            let s = &l.refine[0];
            assert_eq!(s.kind, Kind::Add);
            assert_eq!((s.bbox, s.label.as_deref()), (Some(FAR), Some("person")));
            assert_eq!(s.run, job.id);
            assert!(s.why.contains("never saw this camera-day"), "{}", s.why);
        } else {
            assert_eq!(now, was, "{} changed", p.display());
        }
    }
    // The only new file is the report.
    let new: Vec<&PathBuf> = after.keys().filter(|p| !before.contains_key(*p)).collect();
    assert_eq!(new, [&PathBuf::from("refine").join("report.json")]);
    let report: report::RefineReport =
        serde_json::from_slice(&after[&PathBuf::from("refine").join("report.json")]).unwrap();
    assert_eq!(report.run, job.id);
    assert_eq!(report.recipe.model, "rfdetr-nano");
    assert_eq!(report.recall_by_group.len(), 4);
    // The data root: config, models folder, runs.json, benchmark.json untouched.
    assert_eq!(watched(&t.0), kept);
    // The work folder is gone, and so is its empty parent.
    assert!(!job.work.exists());
    assert!(!t.0.join("training").join("refine").exists());
}

#[test]
fn the_fold_models_train_on_every_label_and_the_report_names_the_floor() {
    let t = tmp("floor-folds");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    // A second car on every frame, 4 px wide of 320: 4.8 px at 384, under a 10 px floor.
    for rel in &rels {
        edit(&dir, rel, |l| {
            l.boxes.push(accepted("car", [250.0, 10.0, 254.0, 60.0]));
        });
    }
    let mut cfg = config(&t.0);
    cfg.collect.size_floor = Some(crate::collect::labels::SizeFloor {
        px: 10.0,
        input: 384,
    });
    let predictor = Arc::new(PredictorFake::default());
    let fake = trainer();
    let mut job = plan_refine(&cfg, RefineRequest::default(), &host()).unwrap();
    job.min_frames = 4;
    job.predictor = Some(predictor.clone());
    let out = run(&job, &fake).unwrap();
    let exports = fake.exports.lock().unwrap();
    assert_eq!(exports.len(), 4);
    for e in exports.iter() {
        assert!(
            e.get("size_floor").is_none() && e.get("under_floor").is_none(),
            "a fold export applies no floor: {e}"
        );
        assert_eq!(e["ignore_boxes"], 0, "{e}");
    }
    let r = &out.result;
    assert_eq!(r["folds_every_label"], true, "{r}");
    assert_eq!(
        r["size_floor"],
        serde_json::json!({"px": 10.0, "input": 384})
    );
    assert_eq!(r["ignore_mode"], "background");
    // The fold models trained on both cars of each frame, so the class counts hold them: the small
    // car is still no verify (an ignore region for the judge).
    assert_eq!(r["found"]["verify"], 0, "{r}");
}

#[test]
fn a_second_identical_run_writes_no_label_file() {
    let t = tmp("rerun");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let predictor = Arc::new(PredictorFake::default());
    run(&plan(&t.0, &predictor), &trainer()).unwrap();
    let stamp = |rel: &String| {
        let p = path_of(&dir, rel);
        (
            std::fs::read(&p).unwrap(),
            std::fs::metadata(&p).unwrap().modified().unwrap(),
        )
    };
    let first: Vec<_> = rels.iter().map(stamp).collect();
    std::thread::sleep(Duration::from_millis(20));
    let again = plan(&t.0, &predictor);
    let out = run(&again, &trainer()).unwrap();
    assert_eq!(out.result["frames_written"], 0, "{}", out.result);
    assert_eq!(out.result["found"]["add"], 12);
    assert_eq!(out.result["written"]["add"], 0);
    let second: Vec<_> = rels.iter().map(stamp).collect();
    assert_eq!(first, second, "no label file was written");
    // An answered suggestion is never asked again, and a rerun keeps the answer.
    edit(&dir, &rels[0], |l| {
        assert!(crate::collect::labels::refine::dismiss(l, 0));
    });
    let out = run(&plan(&t.0, &predictor), &trainer()).unwrap();
    assert_eq!(out.result["frames_written"], 0, "{}", out.result);
    let l = labels_of(&dir, &rels[0]);
    assert_eq!(l.refine.len(), 1);
    assert!(l.refine[0].answered());
}

#[test]
fn a_cancel_after_two_folds_keeps_them_and_reports_a_partial_run() {
    let t = tmp("cancel");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let folds = folds_of(&job);
    let fake = TrainerFake {
        cancel_on: Some(3),
        ..trainer()
    };
    let cancel = AtomicBool::new(false);
    let out = run_cancellable(&job, &fake, &cancel).unwrap();
    assert!(
        out.summary
            .starts_with("stopped by Cancel after 2 of 4 folds"),
        "{}",
        out.summary
    );
    assert_eq!(out.result["cancelled"], true);
    assert_eq!(out.result["folds_done"], 2);
    for rel in &rels {
        let n = labels_of(&dir, rel).refine.len();
        assert_eq!(
            n,
            usize::from(folds[rel] < 2),
            "{rel} of fold {}",
            folds[rel]
        );
    }
    let report: report::RefineReport =
        serde_json::from_slice(&std::fs::read(report::report_path(&dir)).unwrap()).unwrap();
    assert!(report.cancelled);
    assert_eq!((report.folds, report.folds_done), (4, 2));
    assert!(!job.work.exists());
    // A cancel before any fold finished is a plain cancel and writes nothing.
    let t = tmp("cancel-early");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let before: Vec<Vec<u8>> = rels
        .iter()
        .map(|r| std::fs::read(path_of(&dir, r)).unwrap())
        .collect();
    let fake = TrainerFake {
        cancel_on: Some(1),
        ..trainer()
    };
    let job = plan(&t.0, &predictor);
    let e = run_cancellable(&job, &fake, &AtomicBool::new(false)).unwrap_err();
    assert!(is_cancel(&e), "{e:#}");
    let after: Vec<Vec<u8>> = rels
        .iter()
        .map(|r| std::fs::read(path_of(&dir, r)).unwrap())
        .collect();
    assert_eq!(before, after);
    assert!(!report::report_path(&dir).exists());
    assert!(!job.work.exists());
}

#[test]
fn a_failed_fold_keeps_the_earlier_folds_and_names_itself() {
    let t = tmp("fail");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let folds = folds_of(&job);
    let fake = TrainerFake {
        fail_on: Some(3),
        ..trainer()
    };
    let e = run(&job, &fake).unwrap_err();
    let text = format!("{e:#}");
    assert!(text.contains("fold 3 of 4"), "{text}");
    assert!(text.contains("the export broke"), "{text}");
    assert!(
        text.contains("suggestions of the 2 finished fold(s) were saved"),
        "{text}"
    );
    assert!(text.contains("nothing to retry"), "{text}");
    // No advice to retry the export of a run that does not exist.
    assert!(!text.contains("retry-export"), "{text}");
    assert!(!text.contains("checkpoint is kept"), "{text}");
    for rel in &rels {
        let n = labels_of(&dir, rel).refine.len();
        assert_eq!(n, usize::from(folds[rel] < 2), "{rel}");
    }
    let report: report::RefineReport =
        serde_json::from_slice(&std::fs::read(report::report_path(&dir)).unwrap()).unwrap();
    assert_eq!(report.folds_done, 2);
    assert!(report.warnings.iter().any(|w| w.contains("fold 3 of 4")));
    assert!(
        !job.work.exists(),
        "the kept checkpoint went with the work folder"
    );
}

#[test]
fn a_folds_progress_names_no_config_and_no_benchmark() {
    let t = tmp("progress");
    dataset(&t.0.join("dataset"));
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let (out, shown) = run_shown(&job, &trainer());
    out.unwrap();
    let folds: Vec<&Shown> = shown
        .iter()
        .filter(|s| s.0.starts_with("checking fold"))
        .collect();
    assert!(!folds.is_empty());
    for (stage, _, _, pct, detail) in &folds {
        let text = format!("{stage}: {detail}").to_lowercase();
        for word in ["config", "benchmark", "installing the model files"] {
            assert!(!text.contains(word), "{pct}% {text}");
        }
        // No path of the fold folder either.
        assert!(
            !detail.contains(&*job.work.to_string_lossy()),
            "{pct}% {text}"
        );
    }
    // The fold's training stages still show, the install as the fold model.
    for want in ["training: ", "preparing the fold model: done"] {
        assert!(
            folds
                .iter()
                .any(|s| s.0 == "checking fold 2 of 4" && s.4.starts_with(want)),
            "{want}: {folds:#?}"
        );
    }
    // Percent never goes backwards.
    let pcts: Vec<u8> = shown.iter().map(|s| s.3).collect();
    assert!(pcts.windows(2).all(|w| w[0] <= w[1]), "{pcts:?}");
    // The words of ordinary training runs are unchanged.
    assert_eq!(
        TrainStage::Register.label(),
        "adding the model to the config"
    );
    assert_eq!(fold_stage_text(TrainStage::Register, 100, "x"), None);
    assert_eq!(fold_stage_text(TrainStage::Benchmark, 0, "waiting"), None);
    assert_eq!(
        fold_stage_text(TrainStage::Install, 100, "/w/fold-1/model/m.onnx").as_deref(),
        Some("preparing the fold model: done")
    );
    assert_eq!(
        fold_stage_text(TrainStage::Train, 40, "epoch 2 of 8").as_deref(),
        Some("training: epoch 2 of 8")
    );
    assert_eq!(fold_stage_label(TrainStage::Benchmark), FOLD_MODEL);
    assert_eq!(
        fold_stage_label(TrainStage::Dataset),
        "assembling the dataset"
    );
}

#[test]
fn the_step_count_follows_the_folds_lowered_to_the_camera_days() {
    let t = tmp("steps");
    dataset(&t.0.join("dataset"));
    let predictor = Arc::new(PredictorFake::default());
    let req = RefineRequest {
        folds: 8,
        ..RefineRequest::default()
    };
    let mut job = plan_refine(&config(&t.0), req, &host()).unwrap();
    job.min_frames = 4;
    job.predictor = Some(predictor.clone());
    // Queued: the asked count (planning reads no frames).
    assert_eq!(job.steps(), 10);
    let (out, shown) = run_shown(&job, &trainer());
    let out = out.unwrap();
    assert_eq!(out.result["folds"], 4, "four camera days: four folds");
    // The first report is before the selection; every later one has the lowered count.
    assert_eq!((shown[0].1, shown[0].2), (1, 10), "{:?}", shown[0]);
    assert!(shown[1..].iter().all(|s| s.2 == 6), "{shown:#?}");
    let last = shown.last().unwrap();
    assert_eq!(
        (last.0.as_str(), last.1, last.2, last.3),
        (REPORTING, 6, 6, 100)
    );
    assert!(
        shown
            .iter()
            .any(|s| s.0 == "checking fold 4 of 4" && s.1 == 5)
    );
    assert!(!shown.iter().any(|s| s.0.contains("of 8")));
}

#[test]
fn a_reviewer_saving_during_the_check_wins() {
    let t = tmp("reviewer");
    let dir = t.0.join("dataset");
    dataset(&dir);
    let d = dir.clone();
    let predictor = Arc::new(PredictorFake {
        meanwhile: Mutex::new(Some(Box::new(move |images: &[Image]| {
            // One frame gets the person labelled, one goes back to unreviewed, one is skipped.
            edit(&d, &images[0].frame, |l| {
                l.boxes.push(LabelBox {
                    source: Source::Human,
                    state: BoxState::Added,
                    score: None,
                    ..accepted("person", [241.0, 121.0, 291.0, 219.0])
                });
            });
            edit(&d, &images[1].frame, |l| l.status = Status::Unreviewed);
            edit(&d, &images[2].frame, |l| l.status = Status::Skipped);
        }))),
        ..PredictorFake::default()
    });
    let job = plan(&t.0, &predictor);
    let out = run(&job, &trainer()).unwrap();
    assert_eq!(out.result["frames_checked"], 10, "{}", out.result);
    let first = predictor.asked.lock().unwrap()[0].1.clone();
    let labelled = labels_of(&dir, &first[0]);
    assert!(labelled.refine.is_empty(), "the person is labelled now");
    assert_eq!(labelled.boxes.len(), 2, "the reviewer's box is kept");
    assert_eq!(labelled.boxes[1].source, Source::Human);
    let back = labels_of(&dir, &first[1]);
    assert!(back.refine.is_empty());
    assert_eq!(back.status, Status::Unreviewed);
    let skipped = labels_of(&dir, &first[2]);
    assert!(skipped.refine.is_empty());
    assert_eq!(skipped.status, Status::Skipped);
}

#[test]
fn a_fold_frame_in_the_training_lists_fails_loudly() {
    let t = tmp("leak");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let before: Vec<Vec<u8>> = rels
        .iter()
        .map(|r| std::fs::read(path_of(&dir, r)).unwrap())
        .collect();
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let fake = TrainerFake {
        leak: true,
        ..trainer()
    };
    let e = format!("{:#}", run(&job, &fake).unwrap_err());
    assert!(
        e.contains("held-out fold are in the fold model's training or validation lists"),
        "{e}"
    );
    assert!(e.contains("fold 1 of 4"), "{e}");
    assert_eq!(predictor.calls.load(Ordering::SeqCst), 0, "never predicted");
    let after: Vec<Vec<u8>> = rels
        .iter()
        .map(|r| std::fs::read(path_of(&dir, r)).unwrap())
        .collect();
    assert_eq!(before, after);
    assert!(!job.work.exists());
}

#[test]
fn a_fold_frame_trained_on_under_another_name_fails_loudly() {
    let t = tmp("leak-renamed");
    dataset(&t.0.join("dataset"));
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let fake = TrainerFake {
        leak_renamed: true,
        ..trainer()
    };
    let e = format!("{:#}", run(&job, &fake).unwrap_err());
    assert!(
        e.contains("held-out fold are in the fold model's training or validation lists"),
        "{e}"
    );
    assert_eq!(predictor.calls.load(Ordering::SeqCst), 0, "never predicted");
}

/// Accepted boxes `label` at `bbox` and a car, on `n` reviewed frames of `camera`.
fn reviewed_frames(dir: &Path, camera: &str, n: usize, extra: Option<(&str, [f32; 4])>) {
    for i in 0..n {
        let mut boxes = vec![accepted("car", CAR)];
        if let Some((label, bbox)) = extra {
            boxes.push(accepted(label, bbox));
        }
        let rel = put_frame_at(dir, camera, &format!("f{i:02}"), (320, 240), None, boxes);
        edit(dir, &rel, |l| {
            l.status = Status::Reviewed;
            l.reviewed_at = Some("2026-10-22T00:00:00Z".into());
        });
    }
}

/// The warnings of a check's result.
fn warnings_of(r: &serde_json::Value) -> Vec<String> {
    r["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_verify_needs_the_boxes_its_fold_model_trained_on() {
    let t = tmp("per-fold");
    let dir = t.0.join("dataset");
    // One camera day with 55 dogs, three small ones with 5 between them: 60 dogs in all. The
    // fake model finds no dog anywhere.
    const DOG: [f32; 4] = [20.0, 20.0, 60.0, 60.0];
    reviewed_frames(&dir, "big", 55, Some(("dog", DOG)));
    reviewed_frames(&dir, "alpha", 3, Some(("dog", DOG)));
    reviewed_frames(&dir, "bravo", 2, Some(("dog", DOG)));
    reviewed_frames(&dir, "charlie", 4, None);
    let predictor = Arc::new(PredictorFake::default());
    let req = RefineRequest {
        folds: 2,
        ..RefineRequest::default()
    };
    let mut job = plan_refine(&config(&t.0), req.clone(), &host()).unwrap();
    job.min_frames = 4;
    job.predictor = Some(predictor.clone());
    let folds = folds_of(&job);
    assert!(
        folds
            .iter()
            .filter(|(r, _)| r.contains("/big/"))
            .all(|(_, f)| *f == 0)
    );
    assert!(
        folds
            .iter()
            .filter(|(r, _)| !r.contains("/big/"))
            .all(|(_, f)| *f == 1)
    );
    // The preview says so before anything runs.
    let p = super::super::preview(&config(&t.0), &req, &host()).unwrap();
    let says = |ws: &[String], what: &str| ws.iter().any(|w| w.contains(what));
    for want in [
        "fold 1 of 2 holds out 55 frames and its model trains on only 9",
        "fold 1 of 2 trains on fewer than 50 boxes of car (9), dog (5)",
    ] {
        assert!(says(&p.warnings, want), "{want}: {:?}", p.warnings);
    }
    assert!(!says(&p.warnings, "fold 2 of 2"), "{:?}", p.warnings);
    let out = run(&job, &trainer()).unwrap();
    // The big day's model trained on 5 dogs: its 55 dogs get no verify. The small days' model
    // trained on 55: their 5 dogs do.
    assert_eq!(out.result["found"]["verify"], 5, "{}", out.result);
    for (rel, fold) in &folds {
        let l = labels_of(&dir, rel);
        let verifies = l.refine.iter().filter(|s| s.kind == Kind::Verify).count();
        let has_dog = l.boxes.iter().any(|b| b.label == "dog");
        assert_eq!(verifies, usize::from(*fold == 1 && has_dog), "{rel}");
    }
    let ws = warnings_of(&out.result);
    assert!(says(&ws, "fold 1 of 2 holds out 55 frames"), "{ws:?}");
}

#[test]
fn the_title_counts_epochs_in_the_singular_and_the_plural() {
    let t = tmp("title");
    let title = |folds: usize, epochs: u32, frames: &str| {
        let req = RefineRequest {
            folds,
            epochs,
            frames: super::super::RefineFrames::parse(frames).unwrap(),
            ..RefineRequest::default()
        };
        Task::title(&plan_refine(&config(&t.0), req, &host()).unwrap())
    };
    assert_eq!(
        title(2, 1, ""),
        "dataset check (2 folds x 1 epoch, all reviewed frames)"
    );
    assert_eq!(
        title(4, 8, "scene=ir"),
        "dataset check (4 folds x 8 epochs, reviewed frames of scene ir)"
    );
}

#[test]
fn the_training_lock_is_held_from_the_first_fold_to_the_last_write() {
    let t = tmp("lock");
    dataset(&t.0.join("dataset"));
    let lock_file =
        crate::resources::export::train::TrainPaths::for_config(&config(&t.0)).train_lock_file();
    let held: Arc<Mutex<Vec<bool>>> = Arc::default();
    let (h, lf) = (held.clone(), lock_file.clone());
    let predictor = Arc::new(PredictorFake {
        // Another process asking for the lock while a fold model predicts.
        on_call: Mutex::new(Some(Box::new(move |_: &[Image]| {
            h.lock().unwrap().push(TrainLock::try_acquire(&lf).is_err());
        }))),
        ..PredictorFake::default()
    });
    let fake = trainer();
    run(&plan(&t.0, &predictor), &fake).unwrap();
    assert_eq!(*held.lock().unwrap(), [true; 4], "held during every fold");
    assert_eq!(fake.trains.load(Ordering::SeqCst), 4);
    // Released at the end.
    drop(TrainLock::try_acquire(&lock_file).unwrap());
    // Held by another process before the check: refused, nothing trained.
    let other = TrainLock::try_acquire(&lock_file).unwrap();
    let fake = trainer();
    let e = run(&plan(&t.0, &Arc::new(PredictorFake::default())), &fake)
        .unwrap_err()
        .to_string();
    assert!(e.contains("is training a model and holds"), "{e}");
    assert_eq!(fake.trains.load(Ordering::SeqCst), 0);
    drop(other);
}

#[test]
fn frames_gone_since_the_selection_are_left_out_with_a_warning() {
    let t = tmp("gone");
    let dir = t.0.join("dataset");
    dataset(&dir);
    let job0 = plan(&t.0, &Arc::new(PredictorFake::default()));
    let folds = folds_of(&job0);
    let of = |k: usize| -> Vec<String> {
        folds
            .iter()
            .filter(|(_, f)| **f == k)
            .map(|(r, _)| r.clone())
            .collect()
    };
    // While fold 1 predicts: an image of fold 2 and a label file of fold 3 go.
    let (img, lab) = (of(1)[0].clone(), of(2)[0].clone());
    let d = dir.clone();
    let (img2, lab2) = (img.clone(), lab.clone());
    let predictor = Arc::new(PredictorFake {
        meanwhile: Mutex::new(Some(Box::new(move |_: &[Image]| {
            let f = Frame::resolve(&d, &img2).unwrap();
            std::fs::remove_file(&f.jpg).unwrap();
            let f = Frame::resolve(&d, &lab2).unwrap();
            std::fs::remove_file(&f.labels).unwrap();
        }))),
        ..PredictorFake::default()
    });
    let job = plan(&t.0, &predictor);
    let out = run(&job, &trainer()).unwrap();
    assert_eq!(out.result["folds_done"], 4, "{}", out.result);
    assert_eq!(out.result["frames_checked"], 10, "{}", out.result);
    let asked = predictor.asked.lock().unwrap().clone();
    let all: Vec<&String> = asked.iter().flat_map(|(_, f)| f).collect();
    assert!(!all.contains(&&img) && !all.contains(&&lab), "{all:?}");
    let ws = warnings_of(&out.result);
    for n in 2..=4 {
        assert!(
            ws.iter().any(|w| w.starts_with(&format!("fold {n} of 4: "))
                && w.contains("are gone (deleted or moved since the check started)")),
            "fold {n}: {ws:?}"
        );
    }
}

#[test]
fn a_fold_whose_writes_all_fail_is_not_done() {
    let t = tmp("unwritable");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let d = dir.clone();
    // Every frame of fold 1 goes while its model predicts: nothing of it can be written.
    let predictor = Arc::new(PredictorFake {
        meanwhile: Mutex::new(Some(Box::new(move |images: &[Image]| {
            for i in images {
                std::fs::remove_file(&Frame::resolve(&d, &i.frame).unwrap().jpg).unwrap();
            }
        }))),
        ..PredictorFake::default()
    });
    let job = plan(&t.0, &predictor);
    let e = format!("{:#}", run(&job, &trainer()).unwrap_err());
    assert!(
        e.contains("fold 1 of 4: none of its 3 frames could be written"),
        "{e}"
    );
    assert!(e.contains("the frame is gone"), "{e}");
    assert!(
        !report::report_path(&dir).exists(),
        "no fold done, no report"
    );
    assert_eq!(
        predictor.calls.load(Ordering::SeqCst),
        1,
        "the check stopped"
    );
    let written = rels
        .iter()
        .filter(|r| Frame::resolve(&dir, r).is_some())
        .filter(|r| !labels_of(&dir, r).refine.is_empty())
        .count();
    assert_eq!(written, 0);
}

#[test]
fn write_failures_are_named_up_to_a_cap_and_counted_beyond_it() {
    let recipe = report::RecipeInfo {
        model: "rfdetr-nano".into(),
        imgsz: 384,
        epochs: 8,
        batch: 4,
        from: "default".into(),
    };
    let mut acc = Acc::new(RefineReport::new("r", 2, recipe));
    let fold = |failed: usize, kept: usize| write::Written {
        attempted: failed + 1,
        failed,
        failures: (0..kept).map(|i| format!("f{i}: boom")).collect(),
        ..write::Written::default()
    };
    acc.note_failures(&fold(15, 15));
    acc.note_failures(&fold(40, write::MAX_FAILURES));
    acc.close_failures();
    let ws = &acc.rep.warnings;
    assert_eq!(ws.len(), write::MAX_FAILURES + 1, "{ws:?}");
    assert_eq!(
        ws.last().unwrap(),
        "... and 35 more frames whose suggestions could not be written"
    );
    // Nothing more: no line.
    let mut acc = Acc::new(acc.rep.clone());
    acc.rep.warnings.clear();
    acc.note_failures(&fold(2, 2));
    acc.close_failures();
    assert_eq!(acc.rep.warnings.len(), 2);
}

#[test]
fn an_unavailable_review_folder_is_not_an_empty_dataset() {
    let t = tmp("unplugged");
    dataset(&t.0.join("dataset"));
    let mut cfg = config(&t.0);
    let gone = t.0.join("unplugged-drive").join("review");
    cfg.collect.review_dir = Some(gone.clone());
    let p = super::super::preview(&cfg, &RefineRequest::default(), &host()).unwrap();
    assert!(p.folder_missing);
    let why = p.refused.clone().unwrap();
    assert!(
        why.starts_with("The folder under review is not available"),
        "{why}"
    );
    assert!(!why.contains("review more frames"), "{why}");
    assert!(super::super::preview_text(&p).contains("is not available"));
    let mut job = plan_refine(&cfg, RefineRequest::default(), &host()).unwrap();
    job.min_frames = 4;
    job.predictor = Some(Arc::new(PredictorFake::default()));
    let fake = trainer();
    let e = run(&job, &fake).unwrap_err().to_string();
    assert!(
        e.starts_with("The folder under review is not available"),
        "{e}"
    );
    assert_eq!(fake.trains.load(Ordering::SeqCst), 0);
    assert!(
        !gone.exists() && !t.0.join("unplugged-drive").exists(),
        "nothing created"
    );
    // The folder there: a check runs.
    let p = super::super::preview(&config(&t.0), &RefineRequest::default(), &host()).unwrap();
    assert!(!p.folder_missing);
}

#[test]
fn the_unseen_check_reads_the_datasets_names() {
    assert_eq!(
        strip_source("s0-day.cam.2026-10-01.x.jpg"),
        "day.cam.2026-10-01.x.jpg"
    );
    assert_eq!(strip_source("s12-a.jpg"), "a.jpg");
    assert_eq!(strip_source("sun.jpg"), "sun.jpg");
    assert_eq!(strip_source("s-a.jpg"), "s-a.jpg");
    assert_eq!(
        dataset_spelling("ir.my cam.2026-10-01.x.jpg"),
        "ir.my_cam.2026-10-01.x.jpg"
    );
    assert_eq!(
        without_kept_note("boom. Its best checkpoint is kept (1 MB in w): Retry export"),
        "boom"
    );
    assert_eq!(without_kept_note("boom"), "boom");
    let t = tmp("unseen");
    let export = t.0.join("export").join("images");
    // The dataset's record: its image name -> the export image it came from.
    let sources = |pairs: &[(&str, &str)]| {
        let m: BTreeMap<&str, PathBuf> = pairs
            .iter()
            .map(|(n, src)| (*n, export.join(src)))
            .collect();
        std::fs::write(
            t.0.join(crate::training::dataset::IMAGE_SOURCES_FILE),
            serde_json::to_string(&m).unwrap(),
        )
        .unwrap();
    };
    std::fs::write(t.0.join("train.txt"), "./images/s0-ir.a.2026-10-21.x.jpg\n").unwrap();
    std::fs::write(t.0.join("valid.txt"), "").unwrap();
    sources(&[("s0-ir.a.2026-10-21.x.jpg", "train/ir.a.2026-10-21.x.jpg")]);
    let frame = |name: &str| ExportFrame {
        rel: crate::collect::layout::parse_rel(&format!(
            "ir/{}/2026-10-21/{}",
            name.split('.').nth(1).unwrap(),
            name.splitn(4, '.').nth(3).unwrap()
        ))
        .unwrap(),
        name: name.into(),
        image: PathBuf::new(),
        width: 1,
        height: 1,
        buckets: Vec::new(),
        split: Split::Val,
        labels: LabelFile::default(),
        floor: None,
    };
    let ok = frame("ir.b.2026-10-21.x.jpg");
    let bad = frame("ir.a.2026-10-21.x.jpg");
    check_unseen(&t.0, &[&ok]).unwrap();
    let e = check_unseen(&t.0, &[&ok, &bad]).unwrap_err().to_string();
    assert!(e.contains("ir.a.2026-10-21.x.jpg"), "{e}");
    // A frame the dataset wrote under another name (its 120-byte cut, a `-2` of a name taken
    // twice): no name says so, the record does.
    std::fs::write(
        t.0.join("train.txt"),
        "./images/s0-ir.a.2026-10-21.x.jpg\n./images/s0-ir.b.2026-10-2-2.jpg\n",
    )
    .unwrap();
    sources(&[
        ("s0-ir.a.2026-10-21.x.jpg", "train/ir.a.2026-10-21.x.jpg"),
        ("s0-ir.b.2026-10-2-2.jpg", "val/ir.b.2026-10-21.x.jpg"),
    ]);
    let e = check_unseen(&t.0, &[&ok]).unwrap_err().to_string();
    assert!(e.contains("ir.b.2026-10-21.x.jpg"), "{e}");
    // A listed image the record does not name, or no record: the check cannot be made.
    sources(&[("s0-ir.a.2026-10-21.x.jpg", "train/ir.a.2026-10-21.x.jpg")]);
    let e = check_unseen(&t.0, &[&frame("ir.c.2026-10-21.x.jpg")])
        .unwrap_err()
        .to_string();
    assert!(e.contains("cannot tell which frame"), "{e}");
    std::fs::remove_file(t.0.join(crate::training::dataset::IMAGE_SOURCES_FILE)).unwrap();
    std::fs::write(t.0.join("train.txt"), "./images/s0-ir.a.2026-10-21.x.jpg\n").unwrap();
    assert!(check_unseen(&t.0, &[&ok]).is_err());
    // A list that cannot be read fails too: no check, no prediction.
    sources(&[("s0-ir.a.2026-10-21.x.jpg", "train/ir.a.2026-10-21.x.jpg")]);
    check_unseen(&t.0, &[&ok]).unwrap();
    std::fs::remove_file(t.0.join("valid.txt")).unwrap();
    assert!(check_unseen(&t.0, &[&ok]).is_err());
}

#[test]
fn too_few_frames_or_camera_days_are_refused_before_training() {
    let t = tmp("few");
    let dir = t.0.join("dataset");
    put_frame_at(
        &dir,
        "solo",
        "a",
        (320, 240),
        None,
        vec![accepted("car", CAR)],
    );
    let predictor = Arc::new(PredictorFake::default());
    let job = plan(&t.0, &predictor);
    let fake = trainer();
    let e = run(&job, &fake).unwrap_err().to_string();
    assert!(e.contains("at least 4 reviewed frames"), "{e}");
    edit(&dir, "ir/solo/2026-10-21/a.jpg", |l| {
        l.status = Status::Reviewed
    });
    for i in 0..4 {
        let rel = put_frame_at(
            &dir,
            "solo",
            &format!("b{i}"),
            (320, 240),
            None,
            vec![accepted("car", CAR)],
        );
        edit(&dir, &rel, |l| l.status = Status::Reviewed);
    }
    let e = run(&plan(&t.0, &predictor), &fake).unwrap_err().to_string();
    assert!(e.contains("at least 2 camera days"), "{e}");
    assert_eq!(fake.trains.load(Ordering::SeqCst), 0);
}

#[test]
fn the_preview_counts_frames_folds_and_what_is_left_out() {
    let t = tmp("preview");
    let dir = t.0.join("dataset");
    dataset(&dir);
    // Reviewed with a pending box, reviewed with a label off the class list, unreviewed.
    let pending = put_frame_at(
        &dir,
        "echo",
        "p",
        (320, 240),
        None,
        vec![{
            let mut b = accepted("car", CAR);
            b.state = BoxState::Pending;
            b
        }],
    );
    edit(&dir, &pending, |l| l.status = Status::Reviewed);
    let zebra = put_frame_at(
        &dir,
        "echo",
        "z",
        (320, 240),
        None,
        vec![accepted("zebra", CAR)],
    );
    edit(&dir, &zebra, |l| l.status = Status::Reviewed);
    put_frame_at(
        &dir,
        "echo",
        "u",
        (320, 240),
        None,
        vec![accepted("car", CAR)],
    );
    let cfg = config(&t.0);
    let before = files(&t.0);
    let p = super::super::preview(&cfg, &RefineRequest::default(), &host()).unwrap();
    assert_eq!(files(&t.0), before, "the preview writes nothing");
    assert_eq!((p.frames, p.camera_days), (12, 4));
    assert_eq!((p.left_out.pending, p.left_out.off_class_list), (1, 1));
    assert_eq!(p.folds.len(), 4);
    assert!(p.folds.iter().all(|f| f.frames == 3 && f.camera_days == 1));
    assert_eq!(p.recipe.model, "rfdetr-nano");
    assert!(p.seconds.is_some_and(|s| s > 0.0));
    assert!(
        p.refused
            .as_deref()
            .is_some_and(|r| r.contains("at least 40")),
        "{p:?}"
    );
    assert!(
        p.warnings.iter().any(|w| w.contains("car (12)")),
        "{:?}",
        p.warnings
    );
    let text = super::super::preview_text(&p);
    assert!(text.contains("12 frames on 4 camera days"), "{text}");
    assert!(
        text.contains("1 with pending boxes, 1 with labels off the class list"),
        "{text}"
    );
    assert!(text.contains("fold 4: 3 frames on 1 camera days"), "{text}");
    // A filter narrows it; 2 folds over 2 cameras.
    let req = RefineRequest {
        folds: 8,
        frames: super::super::RefineFrames::parse("camera=alpha,camera=bravo").unwrap(),
        ..RefineRequest::default()
    };
    let p = super::super::preview(&cfg, &req, &host()).unwrap();
    assert_eq!((p.frames, p.left_out.filtered, p.folds.len()), (6, 6, 2));
    assert!(p.warnings.iter().any(|w| w.contains("8 folds asked for")));
    // No GPU: still a preview, without a time.
    let mut none = host();
    none.accelerator = None;
    let p = super::super::preview(&cfg, &RefineRequest::default(), &none).unwrap();
    assert!(p.seconds.is_none());
    assert!(p.time_basis.is_none());
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("NVIDIA GPU or Apple silicon"))
    );
    // No finished run: the time is the generic estimate, said to be a rough guess.
    let p = super::super::preview(&cfg, &RefineRequest::default(), &host()).unwrap();
    assert_eq!(p.time_basis, Some(super::super::TimeBasis::Guess));
    // A finished run on this GPU (its recipe is the folds'), 150 s per epoch of 1,000 + 100
    // images in its history: the folds' training at that pace, said to come from that run.
    let gpu = host().resolve_gpu(None).unwrap().unwrap();
    let run_dir = t.0.join("training").join("run-a");
    std::fs::create_dir_all(&run_dir).unwrap();
    let history: String = (1..=10_u32)
        .map(|e| {
            format!(
                "{{\"epoch\":{e},\"elapsed_s\":{},\"metrics\":{{}}}}\n",
                10.0 + 150.0 * f64::from(e)
            )
        })
        .collect();
    std::fs::write(run_dir.join("history.jsonl"), history).unwrap();
    let mut recipe = crate::training::recipe::TrainRecipe::new(
        crate::resources::catalog::TrainFamily::Rfdetr,
        "nano",
    )
    .unwrap();
    recipe.imgsz = 384;
    recipe.batch = 8;
    let runs = serde_json::json!([{
        "id": "run-a", "created": "2026-11-11T02:05:06Z", "state": "done",
        "request": {"recipe": recipe},
        "device": gpu.describe(),
        "progress": {"epoch": 10, "elapsed_s": 1520.0, "settings": {"gpu": gpu.name()}},
        "dataset": {"train_images": 1000, "valid_images": 100},
        "run_dir": run_dir,
    }]);
    let runs_file = crate::training::runs_file(&cfg);
    std::fs::create_dir_all(runs_file.parent().unwrap()).unwrap();
    std::fs::write(&runs_file, runs.to_string()).unwrap();
    let before = std::fs::read(&runs_file).unwrap();
    let p = super::super::preview(&cfg, &RefineRequest::default(), &host()).unwrap();
    assert_eq!(std::fs::read(&runs_file).unwrap(), before, "read only");
    assert_eq!(p.time_basis, Some(super::super::TimeBasis::LastRun));
    assert_eq!((p.recipe.imgsz, p.recipe.batch), (384, 8));
    // 12 frames in 4 folds of 3: each fold model trains on 9, 8 epochs.
    let want = 4.0 * (150.0 / 1100.0 * 9.0 * 8.0 + super::super::FOLD_OVERHEAD_S);
    let got = p.seconds.unwrap();
    assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    assert!(
        super::super::preview_text(&p).contains("(from your last training run)"),
        "{}",
        super::super::preview_text(&p)
    );
}

#[test]
fn planning_needs_a_gpu_and_a_download_allowed_and_stays_out_of_the_config() {
    let t = tmp("plan");
    let cfg = config(&t.0);
    let mut none = host();
    none.accelerator = None;
    let e = plan_refine(&cfg, RefineRequest::default(), &none)
        .unwrap_err()
        .to_string();
    assert!(e.contains("NVIDIA GPU or Apple silicon"), "{e}");
    let closed = Config {
        allow_large_downloads: false,
        ..cfg.clone()
    };
    let e = plan_refine(&closed, RefineRequest::default(), &host())
        .unwrap_err()
        .to_string();
    assert!(e.contains("--allow-large"), "{e}");
    let bad = RefineRequest {
        folds: 1,
        ..RefineRequest::default()
    };
    assert!(plan_refine(&cfg, bad, &host()).is_err());
    let job = plan_refine(&cfg, RefineRequest::default(), &host()).unwrap();
    // The fold run: in the work folder, installed there, registered nowhere.
    assert!(job.train.run_dir.starts_with(&job.work));
    assert!(job.train.install_dir().starts_with(&job.work));
    assert_eq!(job.train.request.recipe.patience, 0);
    assert!(job.train.request.benchmark_sets.is_empty());
    assert!(!job.train.request.include_public);
    assert!(job.train.request.balance.is_off());
    assert_eq!(
        job.config.models, cfg.models,
        "no fold model in the config copy"
    );
    assert!(!t.0.join("training").exists(), "planning writes nothing");
}

#[test]
fn the_queue_runs_it_like_any_other_job() {
    use crate::training::service::{RunState, TrainingService};
    let t = tmp("queue");
    let dir = t.0.join("dataset");
    let rels = dataset(&dir);
    let fake = Arc::new(trainer());
    let svc = TrainingService::start(
        fake.clone(),
        fake.clone(),
        Arc::new(crate::training::job::NoHooks),
        Some(t.0.join("runs.json")),
    );
    let predictor = Arc::new(PredictorFake::default());
    let id = svc.enqueue_task(Box::new(plan(&t.0, &predictor))).unwrap();
    let r = svc.wait_task(&id, Duration::from_secs(120)).unwrap();
    assert_eq!(r.state, RunState::Done, "{r:?}");
    assert_eq!(r.kind, TaskKind::Refine);
    assert_eq!(r.steps, 6);
    assert_eq!(r.percent, 100);
    assert!(svc.records().is_empty(), "no training run recorded");
    for rel in &rels {
        assert_eq!(labels_of(&dir, rel).refine.len(), 1, "{rel}");
    }
    svc.shutdown();
}

/// One detection of a [`DumpLine`] ([`Det`]'s fields).
#[derive(Serialize)]
struct DumpDet<'a> {
    label: &'a str,
    score: f32,
    bbox: [f32; 4],
}

/// One line of the detection dump: what a fold model found on one frame.
#[derive(Serialize)]
struct DumpLine<'a> {
    /// The fold model's name (`refine-fold-<n>`).
    model: &'a str,
    /// Its ONNX file.
    onnx: &'a Path,
    /// The device that ran ([`Detections::device`]).
    device: &'a str,
    /// The key in [`Detections::frames`] (`<scene>/<camera>/<date>/<stem>.jpg`).
    frame: &'a str,
    width: u32,
    height: u32,
    /// The threshold the predictor was asked for.
    threshold: f32,
    dets: Vec<DumpDet<'a>>,
}

/// A [`FoldPredictor`] that appends what `inner` found to `file` as JSONL, one line per frame,
/// flushed per call (a cancelled run keeps the finished folds). Calibration data of
/// [`refine_real_on_a_copy`] (`BOP_REFINE_DUMP`); test code only.
struct DumpPredictor {
    inner: Arc<dyn FoldPredictor>,
    file: PathBuf,
}

impl DumpPredictor {
    /// Lines of `found` for `images`: the frames the inner predictor answered; a frame it has
    /// no entry for is written with no detections unless the run was cancelled (then it was
    /// never looked at).
    fn lines(
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        found: &Detections,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for img in images {
            let dets = match found.frames.get(&img.frame) {
                Some(d) => d.as_slice(),
                None if found.cancelled => continue,
                None => &[],
            };
            let line = DumpLine {
                model: &model.name,
                onnx: &model.onnx,
                device: &found.device,
                frame: &img.frame,
                width: img.width,
                height: img.height,
                threshold,
                dets: dets
                    .iter()
                    .map(|d| DumpDet {
                        label: &d.label,
                        score: d.score,
                        bbox: d.bbox,
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut out, &line)?;
            out.push(b'\n');
        }
        Ok(out)
    }
}

impl FoldPredictor for DumpPredictor {
    fn predict(
        &self,
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections> {
        use std::io::Write as _;
        let found = self.inner.predict(model, images, threshold, cancel)?;
        let bytes = Self::lines(model, images, threshold, &found)?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)
            .with_context(|| format!("opening the dump {}", self.file.display()))?;
        f.write_all(&bytes)
            .and_then(|()| f.flush())
            .and_then(|()| f.sync_data())
            .with_context(|| format!("writing the dump {}", self.file.display()))?;
        Ok(found)
    }
}

#[test]
fn the_dump_writes_a_line_per_frame_and_appends() {
    let t = tmp("dump");
    let file = t.0.join("dets.jsonl");
    std::fs::write(&file, "").unwrap();
    let dump = DumpPredictor {
        inner: Arc::new(PredictorFake::default()),
        file: file.clone(),
    };
    let model = FoldModel {
        name: "refine-fold-1".into(),
        onnx: t.0.join("m.onnx"),
        yaml: t.0.join("m.yaml"),
        family: crate::model::ModelFamilyKind::RfDetr,
        classes: vec!["car".into(), "person".into()],
    };
    std::fs::write(&model.onnx, b"x").unwrap();
    std::fs::write(&model.yaml, b"x").unwrap();
    let image = |frame: &str| Image {
        frame: frame.into(),
        path: PathBuf::new(),
        width: 320,
        height: 240,
    };
    let cancel = AtomicBool::new(false);
    let a = [
        image("ir/alpha/2026-10-21/a.jpg"),
        image("ir/alpha/2026-10-21/b.jpg"),
    ];
    let found = dump
        .predict(&model, &a, judge::MATCH_MIN_SCORE, &cancel)
        .unwrap();
    assert_eq!(found.frames.len(), 2, "the inner answer is passed on");
    dump.predict(
        &model,
        &[image("day/bravo/2026-10-22/c.jpg")],
        judge::MATCH_MIN_SCORE,
        &cancel,
    )
    .unwrap();
    let text = std::fs::read_to_string(&file).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3, "{text}");
    assert_eq!(lines[0]["model"], "refine-fold-1");
    assert_eq!(lines[0]["frame"], "ir/alpha/2026-10-21/a.jpg");
    assert_eq!(lines[2]["frame"], "day/bravo/2026-10-22/c.jpg");
    assert_eq!(lines[0]["device"], "ort:cuda");
    let d = &lines[1]["dets"];
    assert_eq!(d.as_array().unwrap().len(), 2);
    assert_eq!(d[1]["label"], "person");
    assert_eq!(d[1]["score"].as_f64().unwrap() as f32, 0.82);
    let bbox: Vec<f32> = d[1]["bbox"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(bbox, FAR);
    // A cancelled answer writes only the frames it has; an unanswered frame of a finished one,
    // no detections.
    let found = Detections {
        frames: BTreeMap::from([("ir/alpha/2026-10-21/a.jpg".to_string(), Vec::new())]),
        device: String::new(),
        cancelled: true,
    };
    let b = DumpPredictor::lines(&model, &a, 0.05, &found).unwrap();
    assert_eq!(String::from_utf8(b).unwrap().lines().count(), 1);
    let found = Detections {
        cancelled: false,
        ..found
    };
    let b = String::from_utf8(DumpPredictor::lines(&model, &a, 0.05, &found).unwrap()).unwrap();
    assert_eq!(b.lines().count(), 2);
    assert!(b.lines().all(|l| l.contains("\"dets\":[]")), "{b}");
}

/// The whole check on a copy of a dataset with the real training environment, trainer and
/// detector: `BOP_REFINE_CONFIG=<config file whose collect.dir / review_dir is the copy>
/// cargo test --lib refine_real -- --ignored --nocapture`. Never point it at a dataset someone is
/// reviewing: it writes suggestions into the copy. `BOP_REFINE_DUMP=<file>` also writes what each
/// fold model found on each frame to that file as JSONL ([`DumpPredictor`]; emptied at the start).
#[test]
#[ignore = "needs a GPU, the training environment and a copy of a dataset"]
fn refine_real_on_a_copy() {
    let Some(path) = std::env::var_os("BOP_REFINE_CONFIG").map(PathBuf::from) else {
        println!("skipped: BOP_REFINE_CONFIG is not set");
        return;
    };
    let cfg = Config::load(&path).unwrap();
    let mut job = plan_refine(
        &cfg,
        RefineRequest::default(),
        &crate::training::capability_now(),
    )
    .unwrap();
    if let Some(file) = std::env::var_os("BOP_REFINE_DUMP").map(PathBuf::from) {
        std::fs::File::create(&file).unwrap();
        println!("detections go to {}", file.display());
        job.predictor = Some(Arc::new(DumpPredictor {
            inner: job.predictor(),
            file,
        }));
    }
    let runner = crate::resources::export::SystemRunner;
    let fetcher = crate::resources::export::cli_fetcher(&cfg.data_root());
    let started = Instant::now();
    let out = job
        .run(
            TaskContext {
                runner: &runner,
                fetcher: &fetcher,
            },
            &mut |r| match r {
                TaskReport::Stage {
                    stage,
                    percent,
                    detail,
                    ..
                } => println!("{percent}% {stage}: {detail}"),
                TaskReport::Line(l) if !l.trim().is_empty() => println!("{l}"),
                TaskReport::Line(_) => {}
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    println!("{:.1} s: {}", started.elapsed().as_secs_f64(), out.summary);
}

/// A detection as [`DumpPredictor`] wrote it.
#[derive(serde::Deserialize)]
struct ReplayDet {
    label: String,
    score: f32,
    bbox: [f32; 4],
}

/// The part of a [`DumpLine`] a replay reads.
#[derive(serde::Deserialize)]
struct ReplayLine {
    frame: String,
    #[serde(default)]
    dets: Vec<ReplayDet>,
}

/// The fold models' detections of a dump (`BOP_REFINE_DUMP`) as one [`Detections`], each frame's
/// filtered to [`judge::MATCH_MIN_SCORE`] as the predictor's threshold would have; how many lines
/// named a frame again (the later line wins).
fn read_dump(path: &Path) -> (Detections, usize) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading the dump {}: {e}", path.display()));
    let mut frames = BTreeMap::new();
    let mut again = 0;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let l: ReplayLine = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("{} line {}: {e}", path.display(), i + 1));
        let dets: Vec<Det> = l
            .dets
            .into_iter()
            .filter(|d| d.score >= judge::MATCH_MIN_SCORE)
            .map(|d| Det {
                label: d.label,
                score: d.score,
                bbox: d.bbox,
            })
            .collect();
        if frames.insert(l.frame, dets).is_some() {
            again += 1;
        }
    }
    (
        Detections {
            frames,
            device: "replay".into(),
            cancelled: false,
        },
        again,
    )
}

#[test]
fn a_dump_reads_back_filtered_to_the_noise_floor() {
    let t = tmp("replay-read");
    let file = t.0.join("dets.jsonl");
    let line = |frame: &str, scores: &[f32]| {
        let dets: Vec<serde_json::Value> = scores
            .iter()
            .map(|s| serde_json::json!({"label": "car", "score": s, "bbox": CAR}))
            .collect();
        serde_json::json!({"model": "refine-fold-1", "onnx": "m.onnx", "device": "ort:cuda",
            "frame": frame, "width": 320, "height": 240, "threshold": 0.05, "dets": dets})
        .to_string()
    };
    let text = [
        line(
            "ir/a/2026-10-21/x.jpg",
            &[0.05, 0.149, judge::MATCH_MIN_SCORE, 0.9],
        ),
        String::new(),
        line("ir/a/2026-10-21/y.jpg", &[0.1]),
        line("ir/a/2026-10-21/y.jpg", &[0.2]),
    ]
    .join("\n");
    std::fs::write(&file, text).unwrap();
    let (d, again) = read_dump(&file);
    assert_eq!(again, 1);
    assert!(!d.cancelled);
    let scores = |f: &str| -> Vec<f32> { d.frames[f].iter().map(|d| d.score).collect() };
    assert_eq!(
        scores("ir/a/2026-10-21/x.jpg"),
        [judge::MATCH_MIN_SCORE, 0.9]
    );
    assert_eq!(scores("ir/a/2026-10-21/y.jpg"), [0.2]);
    assert_eq!(d.frames["ir/a/2026-10-21/x.jpg"][0].bbox, CAR);
}

/// The judge and the writes of a check, replayed from a detection dump instead of fold models:
/// `BOP_REFINE_CONFIG=<config file> BOP_REFINE_REPLAY=<dump of BOP_REFINE_DUMP> cargo test --lib
/// refine_replay -- --ignored --nocapture`. Selects the reviewed frames and folds them as the job
/// does ([`folds::assign`](super::super::folds::assign) through the selection), judges each fold's
/// frames with that fold's context (the class counts of what its model trained on) and writes the
/// label files through the job's own [`RefineJob::write_fold_into`]; trains, predicts and
/// downloads nothing. It writes suggestions into the copy
/// (never point it at a dataset someone is reviewing) and no `report.json`, which would replace
/// the report of the real run that made the dump.
#[test]
#[ignore = "needs a copy of a dataset and a detection dump of refine_real_on_a_copy"]
fn refine_replay_on_a_copy() {
    let (Some(path), Some(dump)) = (
        std::env::var_os("BOP_REFINE_CONFIG").map(PathBuf::from),
        std::env::var_os("BOP_REFINE_REPLAY").map(PathBuf::from),
    ) else {
        println!("skipped: BOP_REFINE_CONFIG and BOP_REFINE_REPLAY must both be set");
        return;
    };
    let started = Instant::now();
    let cfg = Config::load(&path).unwrap();
    // Planned as the real test plans it: the same frames, classes, recipe (imgsz) and a new id.
    let job = plan_refine(
        &cfg,
        RefineRequest::default(),
        &crate::training::capability_now(),
    )
    .unwrap();
    let s = job.select();
    if let Some(why) = &s.refused {
        panic!("{why}");
    }
    let (found, again) = read_dump(&dump);
    let wanted = job.wanted();
    let answered_in = |files: &mut dyn Iterator<Item = LabelFile>| {
        let mut out: BTreeMap<String, usize> = BTreeMap::new();
        for l in files {
            for x in l.refine.iter().filter(|x| x.answered()) {
                *out.entry(format!("{} {:?}", x.kind.as_str(), x.answer))
                    .or_default() += 1;
            }
        }
        out
    };
    println!(
        "answered suggestions before: {:?}",
        answered_in(&mut s.frames.iter().map(|f| f.labels.clone()))
    );
    let mut acc = Acc::new(job.start_report(&s));
    let in_dump = s
        .frames
        .iter()
        .filter(|f| found.frames.contains_key(&f.rel.jpg()))
        .count();
    // Fold by fold as the job does: the fold's frames still on disk, judged with the class counts
    // of what its model trained on.
    for i in 0..s.k {
        let (held, train, gone) = job.fold_frames(&s, i);
        let class_boxes = super::super::class_boxes(train.iter().copied(), &wanted);
        // The selection's one spot table, as the job uses it.
        let jctx = job.judge_context(&wanted, &class_boxes, &s.spots);
        let rare: Vec<&String> = class_boxes
            .iter()
            .filter(|(_, n)| **n < judge::MIN_CLASS_BOXES)
            .map(|(c, _)| c)
            .collect();
        println!(
            "fold {} of {}: {} held out, {} trained on, {gone} gone; classes under {}: {rare:?}",
            i + 1,
            s.k,
            held.len(),
            train.len(),
            judge::MIN_CLASS_BOXES
        );
        job.write_fold_into(i + 1, s.k, &held, &found, &jctx, &mut acc, &mut |_, _| {})
            .unwrap();
        acc.rep.folds_done += 1;
    }
    acc.close_failures();
    let mut rep = acc.rep;
    rep.finish(&acc.tally);
    let r = &rep.recall;
    println!("review folder: {}", job.dataset_dir.display());
    println!(
        "run {}; imgsz {}; MATCH_MIN_SCORE {}",
        job.id,
        job.train.request.recipe.imgsz,
        judge::MATCH_MIN_SCORE
    );
    println!(
        "dump: {} frames ({again} named again), {in_dump} of {} selected frames in it",
        found.frames.len(),
        s.frames.len()
    );
    println!(
        "frames checked {}, frames changed {}",
        rep.frames_checked, rep.frames_written
    );
    println!(
        "found: add {} ({} at static spots, {} elsewhere), verify {}, relabel {}",
        rep.found.add,
        rep.adds_at_spots,
        rep.found.add - rep.adds_at_spots,
        rep.found.verify,
        rep.found.relabel
    );
    let spot_recall = |name: &str, r: &report::Recall| {
        println!(
            "recall {name}: {} of {} = {:?} (95% {:.3}-{:.3})",
            r.found, r.boxes, r.recall, r.low, r.high
        );
    };
    println!("static spots {}", rep.static_spots);
    spot_recall("at static spots", &rep.recall_at_spots);
    spot_recall("elsewhere", &rep.recall_elsewhere);
    // The answered suggestions of earlier checks survive a rerun.
    let mut now = s.frames.iter().filter_map(|f| {
        let fr = Frame::resolve(&job.dataset_dir, &f.rel.jpg())?;
        labels::read(&fr.labels).ok().flatten()
    });
    println!("answered suggestions after: {:?}", answered_in(&mut now));
    println!(
        "written: add {}, verify {}, relabel {}",
        rep.written.add, rep.written.verify, rep.written.relabel
    );
    println!("loose {}", rep.loose);
    println!(
        "recall {} of {} = {:?} (95% {:.3}-{:.3}) at score {} IoU {}",
        r.found, r.boxes, r.recall, r.low, r.high, rep.found_score, rep.found_iou
    );
    println!("warnings {}", rep.warnings.len());
    for w in &rep.warnings {
        println!("  warning: {w}");
    }
    println!("{:.1} s", started.elapsed().as_secs_f64());
}

/// The static spots of a copy of a dataset at several spot IoUs and dwell shares (the measurement
/// behind [`statics::SPOT_IOU`] and [`statics::MIN_DAY_SHARE`]): `BOP_REFINE_CONFIG=<config file>
/// cargo test --lib static_spots_measure -- --ignored --nocapture`. Reads the reviewed frames of
/// the folder under review as the check selects them; writes nothing. With `BOP_PLAN_EXPORT=<a
/// Dataset tab export>` it also plans a training run on that export (plan only, nothing written)
/// with the static cap off and at 20, and prints what each trains on.
#[test]
#[ignore = "needs a copy of a dataset"]
fn static_spots_measure_on_a_copy() {
    use crate::collect::labels::statics::{self, Table};
    let Some(path) = std::env::var_os("BOP_REFINE_CONFIG").map(PathBuf::from) else {
        println!("skipped: BOP_REFINE_CONFIG is not set");
        return;
    };
    let cfg = Config::load(&path).unwrap();
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    let (classes, hidden) = (&cfg.collect.classes, &cfg.collect.hidden_classes);
    let s = super::super::select(
        &dir,
        classes,
        hidden,
        &super::super::RefineFrames::default(),
        4,
        0,
    );
    let wanted = Wanted { classes, hidden };
    let obs = statics::observations(&s.frames, &wanted);
    println!(
        "{} reviewed frames, {} training boxes of wanted classes",
        s.frames.len(),
        obs.len()
    );
    let mut all: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_camera: BTreeMap<&str, usize> = BTreeMap::new();
    for o in &obs {
        *all.entry(o.label.clone()).or_default() += 1;
        if statics::may_be_static(&o.label) {
            *per_camera.entry(o.camera.as_str()).or_default() += 1;
        }
    }
    println!(
        "boxes that may be static per camera (clustered whole up to {}): {per_camera:?}",
        statics::MAX_OBS_PER_CAMERA
    );
    // The folder's spots as a "Static objects" card lists them (after a check: open adds).
    for x in statics::overview(&s.frames, &wanted) {
        println!(
            "  spot {}: {} on {} frames of {} days ({} stayed) {:?}; labelled frames {}, frames \
             with an open add {}",
            x.spot.id,
            x.spot.label,
            x.spot.frames,
            x.spot.days,
            x.spot.stay_days,
            x.spot.scenes,
            x.labelled,
            x.open_adds
        );
    }
    // Boxes at a spot per class, static-only frames and frames with two boxes at one spot.
    let describe = |table: &Table| {
        let mut at: BTreeMap<String, usize> = BTreeMap::new();
        for o in &obs {
            if table.spot_at(&o.camera, &o.label, &o.bbox).is_some() {
                *at.entry(o.label.clone()).or_default() += 1;
            }
        }
        for (c, n) in &all {
            println!(
                "  {c}: {} of {n} boxes at a spot",
                at.get(c).copied().unwrap_or(0)
            );
        }
        let mut static_only = 0;
        let mut positive = 0;
        let mut doubles = 0;
        for f in &s.frames {
            let boxes: Vec<(String, [f32; 4])> = f
                .labels
                .training()
                .filter(|b| !b.is_ignore())
                .filter_map(|b| {
                    let c = wanted.class_of(&b.label)?;
                    Some((
                        c.to_string(),
                        statics::normalised(&b.bbox, f.width, f.height)?,
                    ))
                })
                .collect();
            if boxes.is_empty() {
                continue;
            }
            positive += 1;
            let mut hit: BTreeMap<&str, usize> = BTreeMap::new();
            for (c, b) in &boxes {
                if let Some(sp) = table.spot_at(&f.rel.camera, c, b) {
                    *hit.entry(sp.id.as_str()).or_default() += 1;
                }
            }
            doubles += usize::from(hit.values().any(|n| *n > 1));
            if table.static_only(&f.rel.camera, boxes).is_some() {
                static_only += 1;
            }
        }
        println!(
            "  static-only frames: {static_only} of {positive} frames with boxes ({} frames); \
             frames with two boxes at one spot: {doubles}",
            s.frames.len()
        );
    };
    // The dwell rule: every spot with its smallest and median day share, at several shares.
    let base = Table::build_with(obs.clone(), statics::SPOT_IOU, 0.0);
    for share in [0.0f32, 0.2, 0.3, 0.5] {
        let started = Instant::now();
        let table = Table::build_with(obs.clone(), statics::SPOT_IOU, share);
        let ms = started.elapsed().as_millis();
        let by_cam: Vec<String> = table
            .cameras
            .iter()
            .map(|(c, v)| format!("{c} {}", v.len()))
            .collect();
        println!(
            "\n=== spot IoU {} dwell share {share}: {} spots ({}), {ms} ms ===",
            statics::SPOT_IOU,
            table.len(),
            by_cam.join(", ")
        );
        for sp in table.spots() {
            let mut shares: Vec<f64> = sp.day_shares().values().copied().collect();
            shares.sort_by(f64::total_cmp);
            let min = shares.first().copied().unwrap_or(0.0);
            let median = shares
                .get(shares.len().saturating_sub(1) / 2)
                .copied()
                .unwrap_or(0.0);
            println!(
                "  {} {}: {} frames, {} days ({} stayed), day share min {min:.2} median \
                 {median:.2}",
                sp.id, sp.label, sp.frames, sp.days, sp.stay_days
            );
        }
        // Spots of the table without the rule that this one does not have (by place).
        for sp in base.spots() {
            let kept = table
                .camera(&sp.camera)
                .iter()
                .any(|x| x.object == sp.object && labels::iou(&x.bbox, &sp.bbox) >= 0.9);
            if !kept {
                println!(
                    "  REMOVED {} {}: {} frames, {} days, {} of them at share >= {share}",
                    sp.id,
                    sp.label,
                    sp.frames,
                    sp.days,
                    sp.day_shares()
                        .values()
                        .filter(|v| **v >= f64::from(share))
                        .count()
                );
            }
        }
        describe(&table);
    }
    for t in [0.5f32, 0.7, 0.8] {
        let started = Instant::now();
        let table = Table::build_at(obs.clone(), t);
        let ms = started.elapsed().as_millis();
        println!(
            "\n=== spot IoU {t}, dwell share {} ({} spots, {ms} ms) ===",
            statics::MIN_DAY_SHARE,
            table.len()
        );
        for (cam, spots) in &table.cameras {
            let mut by_obj: BTreeMap<&str, usize> = BTreeMap::new();
            for sp in spots {
                *by_obj.entry(sp.label.as_str()).or_default() += 1;
            }
            // Neighbouring spots: the largest IoU between two representatives of one object.
            let mut max_pair = 0f32;
            let mut close = 0;
            for (i, a) in spots.iter().enumerate() {
                for b in spots.iter().skip(i + 1) {
                    if a.object == b.object {
                        let v = labels::iou(&a.bbox, &b.bbox);
                        max_pair = max_pair.max(v);
                        close += usize::from(v >= 0.3);
                    }
                }
            }
            println!(
                "  camera {cam}: {} spots {by_obj:?}; frames per spot {:?}; max rep-pair IoU \
                 {max_pair:.2}, pairs >= 0.3: {close}",
                spots.len(),
                spots.iter().map(|s| s.frames).collect::<Vec<_>>()
            );
        }
        describe(&table);
    }
    // The plan of a training run on an export, with the static cap off and on.
    let Some(export) = std::env::var_os("BOP_PLAN_EXPORT").map(PathBuf::from) else {
        return;
    };
    let source = crate::training::dataset::read_export(&export).unwrap();
    let sources = [source];
    for cap in [None, Some(20)] {
        let o = crate::training::dataset::Options {
            negative_share: cfg.training.negative_share,
            heldout_share: cfg.training.heldout_share,
            oversample: cfg.training.oversample,
            static_cap: cap,
            ..crate::training::dataset::Options::default()
        };
        let started = Instant::now();
        let p = crate::training::preflight::planned(&sources, &o).unwrap();
        println!(
            "\n=== plan of {} with static cap {cap:?} ({} ms) ===",
            export.display(),
            started.elapsed().as_millis()
        );
        println!(
            "  training frames {} (positives {}, negatives {}), validation {}, held out {}",
            p.train.len(),
            p.stats.train_positives,
            p.stats.train_negatives,
            p.valid.len(),
            p.heldout.len()
        );
        println!("  training boxes per class {:?}", p.stats.train_boxes);
        println!(
            "  static spots {}, frames capped {}, boxes capped {:?}",
            p.stats.static_spots, p.stats.static_capped, p.stats.static_capped_boxes
        );
        for c in &p.stats.static_caps {
            println!(
                "    {} {}: {} frames, {} kept",
                c.id, c.label, c.frames, c.kept
            );
        }
        for w in &p.warnings {
            println!("  warning: {w}");
        }
    }
}

/// What a size floor does to a copy of a dataset (the measurement behind `collect.size_floor`'s
/// default being off and the Refine tab's offer): `BOP_REFINE_CONFIG=<config file>
/// BOP_FLOOR_OUT=<an empty scratch folder outside the dataset> cargo test --lib size_floor_measure
/// -- --ignored --nocapture`. Reads the reviewed frames of the folder under review; counts the
/// boxes under the floor (`BOP_FLOOR=<px>,<input>`, default 10,384); then exports the folder into
/// `BOP_FLOOR_OUT` with the floor off and on (YOLO only, images copied or linked there: never into
/// the dataset) and plans a training run on each export in both ignore modes (plan only), and the
/// floor's export for runs at 384 and 640 px (background, static cap off: `floor_restored`). Never
/// point `BOP_FLOOR_OUT` at a folder someone uses.
#[test]
#[ignore = "needs a copy of a dataset"]
fn size_floor_measure_on_a_copy() {
    use crate::collect::labels::SizeFloor;
    use crate::collect::labels::export::{self as ex, ExportRequest, Format};
    use crate::training::dataset::{self, IgnoreMode};
    let (Some(path), Some(out)) = (
        std::env::var_os("BOP_REFINE_CONFIG").map(PathBuf::from),
        std::env::var_os("BOP_FLOOR_OUT").map(PathBuf::from),
    ) else {
        println!("skipped: BOP_REFINE_CONFIG or BOP_FLOOR_OUT is not set");
        return;
    };
    let floor = std::env::var("BOP_FLOOR")
        .ok()
        .and_then(|v| {
            let (px, input) = v.split_once(',')?;
            Some(SizeFloor {
                px: px.trim().parse().ok()?,
                input: input.trim().parse().ok()?,
            })
        })
        .unwrap_or(SizeFloor {
            px: 10.0,
            input: 384,
        });
    let cfg = Config::load(&path).unwrap();
    let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
    let classes = &cfg.collect.classes;
    let started = Instant::now();
    let c = ex::floor_counts(&dir, classes, floor);
    println!(
        "=== {} at floor {} px / input {} ({} ms) ===",
        dir.display(),
        floor.px,
        floor.input,
        started.elapsed().as_millis()
    );
    println!(
        "  {} reviewed frames; {} hold a training box under the floor, {} of them nothing else",
        c.frames, c.frames_with, c.frames_only
    );
    for (class, n) in &c.boxes {
        println!(
            "  {class}: {} of {n} training boxes under the floor",
            c.under.get(class).copied().unwrap_or(0)
        );
    }
    for w in c.warnings.iter().take(5) {
        println!("  warning: {w}");
    }
    let t = &cfg.training;
    for (name, f) in [("off", None), ("on", Some(floor))] {
        let req = ExportRequest {
            formats: vec![Format::Yolo],
            out_dir: out.join(format!("floor-{name}")),
            split: ex::SplitConfig::default(),
            benchmark_split: ex::BenchSplit::Val,
            include_unreviewed: false,
            hidden_classes: cfg.collect.hidden_classes.clone(),
            size_floor: f,
        };
        let started = Instant::now();
        let r = ex::run(&dir, classes, &req).unwrap();
        println!(
            "\n=== export, floor {name} ({} ms): {} frames ({} train, {} val), ignore regions {}, \
             under the floor {:?} ===",
            started.elapsed().as_millis(),
            r.frames,
            r.train,
            r.val,
            r.ignore_boxes,
            r.under_floor
        );
        let sources = [dataset::read_export(&r.out_dir).unwrap()];
        for (mode, cap) in [IgnoreMode::DropFrame, IgnoreMode::Background]
            .into_iter()
            .flat_map(|m| [(m, t.static_cap_option()), (m, None)])
        {
            let o = dataset::Options {
                negative_share: t.negative_share,
                heldout_share: t.heldout_share,
                oversample: t.oversample,
                static_cap: cap,
                ignore: mode,
                ..dataset::Options::default()
            };
            let p = crate::training::preflight::planned(&sources, &o).unwrap();
            let s = &p.stats;
            println!(
                "  {}, static cap {cap:?}: training {} (positives {}, negatives {}), validation \
                 {}, held out {}; boxes {:?}",
                mode.as_str(),
                p.train.len(),
                s.train_positives,
                s.train_negatives,
                p.valid.len(),
                p.heldout.len(),
                s.train_boxes
            );
            println!(
                "    frames with an ignore region: left out {} (only regions {}, validation days \
                 {}), trained with it as background {} {:?}; held-out fingerprint {}",
                s.ignore_frames,
                s.ignore_only,
                s.ignore_valid_day,
                s.ignore_kept,
                s.ignore_boxes_background,
                &p.heldout_fingerprint[..12]
            );
        }
        // The floor's export planned for runs of a known input size (background, no static cap):
        // its regions are judged again at the run's input.
        if f.is_none() {
            continue;
        }
        for imgsz in [384, 640] {
            let o = dataset::Options {
                negative_share: t.negative_share,
                heldout_share: t.heldout_share,
                oversample: t.oversample,
                static_cap: None,
                ignore: IgnoreMode::Background,
                imgsz: Some(imgsz),
                size_floor: Some(floor),
                ..dataset::Options::default()
            };
            let p = crate::training::preflight::planned(&sources, &o).unwrap();
            let s = &p.stats;
            println!(
                "  background at imgsz {imgsz}, static cap off: training {} (positives {}, \
                 negatives {}), validation {}, held out {}; boxes {:?}",
                p.train.len(),
                s.train_positives,
                s.train_negatives,
                p.valid.len(),
                p.heldout.len(),
                s.train_boxes
            );
            println!(
                "    trained with a region as background {} {:?}; left out {} (only regions {}, \
                 validation days {}); floor_restored {:?}; negatives trimmed {}; held-out \
                 fingerprint {}",
                s.ignore_kept,
                s.ignore_boxes_background,
                s.ignore_frames,
                s.ignore_only,
                s.ignore_valid_day,
                s.floor_restored,
                s.negatives_trimmed,
                &p.heldout_fingerprint[..12]
            );
            for w in p.warnings.iter().filter(|w| w.contains("size floor")) {
                println!("    warning: {w}");
            }
        }
    }
}
