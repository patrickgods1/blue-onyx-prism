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
    /// What each run trained on: `train.txt` + `valid.txt` names.
    trained: Mutex<Vec<String>>,
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
        for list in ["train.txt", "valid.txt"] {
            let t = std::fs::read_to_string(data.join(list)).unwrap();
            self.trained
                .lock()
                .unwrap()
                .extend(t.lines().map(str::to_string));
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

/// The fake fold model: on every frame the labelled car and an unlabelled person.
#[derive(Default)]
struct PredictorFake {
    calls: AtomicUsize,
    /// What each call was asked: (model, frames).
    asked: Mutex<Vec<(FoldModel, Vec<String>)>>,
    /// Called during the first call: a reviewer saving frames while the check runs.
    meanwhile: Mutex<Option<Meanwhile>>,
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
    std::fs::write(t.0.join("train.txt"), "./images/s0-ir.a.2026-10-21.x.jpg\n").unwrap();
    std::fs::write(t.0.join("valid.txt"), "").unwrap();
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
    };
    let ok = frame("ir.b.2026-10-21.x.jpg");
    let bad = frame("ir.a.2026-10-21.x.jpg");
    check_unseen(&t.0, &[&ok]).unwrap();
    let e = check_unseen(&t.0, &[&ok, &bad]).unwrap_err().to_string();
    assert!(e.contains("ir.a.2026-10-21.x.jpg"), "{e}");
    // A list that cannot be read fails too: no check, no prediction.
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
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("NVIDIA GPU or Apple silicon"))
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
/// refine_replay -- --ignored --nocapture`. Selects the reviewed frames as the job does, judges
/// them with the job's context and writes the label files through [`write::write_fold`] as a
/// real run does; trains, predicts and downloads nothing. It writes suggestions into the copy
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
    let jctx = job.judge_context(&wanted, &s);
    let mut rep = job.start_report(&s);
    let mut tally = RecallTally::default();
    let mut sample = Vec::new();
    let frames: Vec<&ExportFrame> = s.frames.iter().collect();
    let in_dump = frames
        .iter()
        .filter(|f| found.frames.contains_key(&f.rel.jpg()))
        .count();
    let w = write::write_fold(
        &job.dataset_dir,
        &frames,
        &found,
        &jctx,
        &mut tally,
        &mut rep.warnings,
        &mut |_, _| {},
    );
    count_written(&mut rep, &mut sample, &w);
    rep.folds_done = rep.folds;
    rep.finish(&tally);
    let r = &rep.recall;
    println!("review folder: {}", job.dataset_dir.display());
    println!(
        "run {}; imgsz {}; MATCH_MIN_SCORE {}",
        job.id,
        jctx.imgsz,
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
        "found: add {}, verify {}, relabel {}",
        rep.found.add, rep.found.verify, rep.found.relabel
    );
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
