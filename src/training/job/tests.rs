//! Tests of the training stage machine and the queue over fakes (no network, no Python): the
//! export's [`Fake`] for uv / downloads, plus a fake `train_model.py` that prints `@train`
//! events and writes the exported files.

use super::*;
use crate::backend::detect::ComputeCapability;
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::train::TrainLane;
use crate::resources::export::{ChildFailed, Cmd};
use crate::training::service::{self, RunState, TrainingService};
use std::sync::{Arc, Mutex};

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-train-job-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// What the fake training script does.
#[derive(Default)]
struct TrainFake {
    inner: Fake,
    cuda: bool,
    mps: bool,
    /// Block in training until cancelled.
    hang: bool,
    /// Fail the training with these output lines.
    fail_lines: Vec<&'static str>,
    /// Report these classes instead of the dataset's.
    wrong_classes: Option<Vec<String>>,
    train_cmds: Mutex<Vec<Cmd>>,
}

impl TrainFake {
    fn new() -> Self {
        Self {
            inner: Fake::new(),
            mps: true,
            ..Default::default()
        }
    }
}

impl CommandRunner for TrainFake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        let d = cmd.display();
        if !d.contains("train_model.py") {
            return self.inner.run(cmd, on_line, cancel);
        }
        self.train_cmds.lock().unwrap().push(cmd.clone());
        if d.contains("--self-check") {
            on_line("[train_model] self-check: importing torch");
            on_line(&format!(
                "@train {{\"event\":\"self_check\",\"cuda\":{},\"mps\":{},\"cuda_ok\":true,\"torch\":\"2.14.0\"}}",
                self.cuda, self.mps
            ));
            return Ok(());
        }
        on_line(
            r#"@train {"event":"start","device":"mps","train_images":20,"valid_images":2,"epochs":2}"#,
        );
        if self.hang {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            return Err(anyhow!(Cancelled));
        }
        if !self.fail_lines.is_empty() {
            for l in &self.fail_lines {
                on_line(l);
            }
            return Err(anyhow::Error::new(ChildFailed {
                program: "python".into(),
                status: "exit status: 1".into(),
                lines: self.fail_lines.iter().map(|l| l.to_string()).collect(),
            }));
        }
        for e in 1..=2 {
            on_line(&format!(
                "noise @train {{\"event\":\"batch\",\"epoch\":{e},\"epochs\":2,\"batch\":1,\"batches\":2}}"
            ));
            on_line(&format!(
                "@train {{\"event\":\"epoch\",\"epoch\":{e},\"epochs\":2,\"loss\":{}}}",
                3.0 - f64::from(e)
            ));
        }
        on_line(r#"@train {"event":"phase","phase":"export","detail":"x"}"#);
        let name = arg_after(cmd, "--name").unwrap().display().to_string();
        let out = arg_after(cmd, "--out-dir").unwrap();
        let data = arg_after(cmd, "--data").unwrap();
        let classes: Vec<String> = self.wrong_classes.clone().unwrap_or_else(|| {
            std::fs::read_to_string(data.join("classes.txt"))
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect()
        });
        let family = match arg_after(cmd, "--family").unwrap().to_str().unwrap() {
            "yolo26" => "yolo26",
            "yolo11" | "yolov8" => "yolo8",
            "rfdetr" => "rfdetr",
            _ => "detr",
        };
        std::fs::write(out.join(format!("{name}.onnx")), vec![1u8; 4096]).unwrap();
        let names: String = classes.iter().map(|c| format!("  - {c}\n")).collect();
        std::fs::write(out.join(format!("{name}.yaml")), format!("NAMES:\n{names}")).unwrap();
        on_line(&format!(
            "@train {}",
            serde_json::json!({"event": "done", "onnx": out.join(format!("{name}.onnx")), "yaml": "y",
                "family": family, "classes": classes, "imgsz": 640, "seconds": 12.5, "device": "mps"})
        ));
        Ok(())
    }
}

impl Fetcher for TrainFake {
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

/// Records the hook calls; the benchmark answers `bench`.
#[derive(Default)]
struct RecHooks {
    registered: Mutex<Vec<TrainOutput>>,
    benched: Mutex<Vec<(Option<PathBuf>, Vec<String>)>>,
    bench_fails: bool,
}

impl Hooks for RecHooks {
    fn register(&self, out: &TrainOutput) -> Result<String> {
        self.registered.lock().unwrap().push(out.clone());
        Ok(format!("added {}", out.name))
    }
    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        _: &AtomicBool,
    ) -> Result<Option<BenchSummary>> {
        self.benched
            .lock()
            .unwrap()
            .push((out.heldout.clone(), extra.to_vec()));
        if self.bench_fails {
            bail!("no runtime");
        }
        Ok(Some(BenchSummary {
            ap50: Some(0.61),
            ms_p50: Some(12.0),
            ..BenchSummary::default()
        }))
    }
}

fn jpg(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(40, 30, image::Rgb([50, 60, 70]))
        .save(path)
        .unwrap();
}

/// A review export: 2 cameras x 2 days, cat / raccoon boxes and empty frames.
fn export(dir: &Path) {
    std::fs::write(dir.join("classes.txt"), "cat\nraccoon\n").unwrap();
    for cam in ["yard", "gate"] {
        for day in ["2026-10-01", "2026-10-02"] {
            for i in 0..5 {
                let stem = format!("ir.{cam}.{day}.2026100{i}T000000000Z_x{i}");
                jpg(&dir.join("images").join(format!("{stem}.jpg")));
                let label = match i {
                    0 | 1 => "0 0.5 0.5 0.2 0.2\n",
                    2 | 3 => "1 0.4 0.4 0.3 0.2\n",
                    _ => "",
                };
                std::fs::create_dir_all(dir.join("labels")).unwrap();
                std::fs::write(dir.join("labels").join(format!("{stem}.txt")), label).unwrap();
            }
        }
    }
}

fn mps() -> Accelerator {
    Accelerator::Mps {
        name: "Apple M1 GPU".into(),
        memory_mb: 9000,
    }
}

fn cuda() -> Accelerator {
    Accelerator::Cuda {
        ordinal: 0,
        name: "NVIDIA GeForce GTX 1070".into(),
        compute_capability: ComputeCapability::new(6, 1),
        vram_mb: 8192,
        lane: TrainLane::Cu126,
    }
}

fn job(
    root: &Path,
    family: TrainFamily,
    size: &str,
    accel: Accelerator,
    platform: Platform,
) -> TrainJob {
    let exp = root.join("export");
    std::fs::create_dir_all(&exp).unwrap();
    if !exp.join("classes.txt").exists() {
        export(&exp);
    }
    let mut recipe = TrainRecipe::new(family, size, 2).unwrap();
    recipe.augment = crate::training::Augment::NIGHT;
    TrainJob {
        id: format!("run-{}", uuid::Uuid::new_v4().simple()),
        request: TrainRequest {
            recipe,
            datasets: vec![exp],
            include_public: false,
            scene: SceneFilter::NightIr,
            classes: None,
            rename: BTreeMap::new(),
            benchmark_sets: vec!["nir-night".into()],
            allow_large: false,
        },
        name: format!("night-{}-test", family.base_name(size)),
        paths: TrainPaths::new(root, root.join("models")),
        run_dir: root.join("training").join("run1"),
        platform,
        accelerator: accel,
        options: dataset::Options::default(),
        root: root.to_path_buf(),
        timeout: Duration::from_secs(60),
        workers: 0,
        keep_work: false,
    }
}

fn run(
    j: &TrainJob,
    fake: &TrainFake,
    hooks: &dyn Hooks,
    cancel: &AtomicBool,
) -> (Result<RunOutcome, TrainError>, Vec<(TrainStage, u8)>) {
    let mut stages = Vec::new();
    let r = run_training(
        j,
        fake,
        fake,
        hooks,
        &mut |p| {
            if let TrainReport::Stage { stage, percent, .. } = p {
                stages.push((stage, percent));
            }
        },
        cancel,
    );
    (r, stages)
}

#[test]
fn the_public_subset_is_not_downloaded_without_the_large_gate() {
    let t = tmp("public-gate");
    let mut j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    j.request.include_public = true;
    let fake = TrainFake::new();
    let hooks = RecHooks::default();
    let (r, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    let Err(e) = r else {
        panic!("a run with the public subset was not refused")
    };
    let e = format!("{e:#}");
    assert!(
        e.contains("large download") && e.contains("--allow-large"),
        "{e}"
    );
    let fetched = fake.inner.fetched.lock().unwrap().clone();
    assert!(
        !fetched.iter().any(|id| id == catalog::LILA_NIGHT_ID),
        "{fetched:?}"
    );
    assert!(hooks.registered.lock().unwrap().is_empty());
    // Allowed: the run downloads it.
    j.request.allow_large = true;
    let fake = TrainFake::new();
    let (r, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    if let Err(e) = &r {
        assert!(!format!("{e:#}").contains("--allow-large"), "{e:#}");
    }
    let fetched = fake.inner.fetched.lock().unwrap().clone();
    assert!(
        fetched.iter().any(|id| id == catalog::LILA_NIGHT_ID),
        "{fetched:?}"
    );
}

#[test]
fn a_run_trains_installs_registers_and_benchmarks() {
    let t = tmp("full");
    let j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    let fake = TrainFake::new();
    let hooks = RecHooks::default();
    let (r, stages) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    let out = r.unwrap();
    // Every stage, in order, ending at 100%.
    let mut order: Vec<TrainStage> = Vec::new();
    for (s, _) in &stages {
        if order.last() != Some(s) {
            order.push(*s);
        }
    }
    assert_eq!(order, TrainStage::ALL.to_vec());
    let train_pcts: Vec<u8> = stages
        .iter()
        .filter(|(s, _)| *s == TrainStage::Train)
        .map(|(_, p)| *p)
        .collect();
    assert!(
        train_pcts.windows(2).all(|w| w[0] <= w[1]),
        "{train_pcts:?}"
    );
    assert!(
        train_pcts.contains(&25) && train_pcts.contains(&100),
        "{train_pcts:?}"
    );
    // Installed with provenance.
    let models = t.0.join("models");
    let o = &out.output;
    assert_eq!(o.onnx, models.join("night-yolo26n-test.onnx"));
    assert!(o.onnx.is_file() && o.yaml.is_file());
    assert_eq!(o.family, ModelFamilyKind::Yolo26);
    assert_eq!(o.classes, ["cat", "raccoon"]);
    assert_eq!(o.license, "AGPL-3.0");
    let prov: Provenance = serde_json::from_slice(&std::fs::read(&o.provenance).unwrap()).unwrap();
    assert_eq!(prov.license, "AGPL-3.0");
    assert_eq!(prov.base_weights.id, "model:yolo26n");
    assert!(!prov.exdark_used);
    assert_eq!(prov.datasets.len(), 1);
    assert_eq!(prov.datasets[0].kind, "review");
    assert_eq!(prov.onnx_sha256, manager::sha256_file(&o.onnx).unwrap());
    assert!(!models.join(".night-yolo26n-test.train-staging").exists());
    // Held-out frames from one of the two camera/day groups x 2 cameras.
    assert!(out.prepared.heldout_images > 0);
    assert_eq!(out.prepared.classes, ["cat", "raccoon"]);
    // Hooks.
    assert_eq!(hooks.registered.lock().unwrap().len(), 1);
    let b = hooks.benched.lock().unwrap();
    assert_eq!(b[0].0.as_deref(), out.prepared.heldout_dir.as_deref());
    assert_eq!(b[0].1, ["nir-night"]);
    assert_eq!(out.benchmark.as_ref().unwrap().ap50, Some(0.61));
    // The commands.
    let cmds = fake.inner.cmds();
    let pip = cmds.iter().find(|c| c.contains(" pip install ")).unwrap();
    assert!(
        pip.contains("train-env") && !pip.contains("--torch-backend"),
        "{pip}"
    );
    let tc = fake.train_cmds.lock().unwrap();
    assert_eq!(tc.len(), 2, "self-check + training");
    let train = &tc[1];
    let d = train.display();
    for want in [
        "--trainer ultralytics",
        "--family yolo26",
        "--size n",
        "--device mps",
        "--aug-dark 0.5",
        "--aug-gray 0.5",
        "--epochs 2",
    ] {
        assert!(d.contains(want), "{want} in {d}");
    }
    assert!(
        arg_after(train, "--weights")
            .unwrap()
            .ends_with("export-weights/yolo26n/yolo26n.pt")
    );
    assert!(
        arg_after(train, "--export-script")
            .unwrap()
            .ends_with("export_model.py")
    );
    assert!(
        train
            .env
            .iter()
            .any(|(k, v)| k == "PYTORCH_ENABLE_MPS_FALLBACK" && v == "1")
    );
    let log = std::fs::read_to_string(j.run_dir.join(LOG_FILE)).unwrap();
    assert!(log.contains("\"event\":\"epoch\""), "{log}");
    // The work dir (checkpoints) is gone; the prepared data stays.
    assert!(!j.run_dir.join("work").exists());
    assert!(j.run_dir.join("data/dataset.json").is_file());
    // The same name again is refused before anything runs.
    let n = fake.inner.runs.load(Ordering::SeqCst);
    let (again, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    assert!(again.unwrap_err().message.contains("already exists"));
    assert_eq!(fake.inner.runs.load(Ordering::SeqCst), n);
}

#[test]
fn rfdetr_and_transformers_runs_pass_their_weights() {
    let t = tmp("weights");
    for (family, size, weights_end, trainer, service) in [
        (
            TrainFamily::Rfdetr,
            "nano",
            "train-weights/rfdetr-nano/rf-detr-nano.pth",
            "rfdetr",
            ModelFamilyKind::RfDetr,
        ),
        (
            TrainFamily::Rtdetrv2,
            "s",
            "train-weights/rtdetrv2-s",
            "transformers",
            ModelFamilyKind::Detr,
        ),
    ] {
        let j = job(&t.0, family, size, mps(), Platform::MACOS_ARM64);
        let fake = TrainFake::new();
        let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
        let out = r.unwrap();
        assert_eq!(out.output.family, service);
        assert_eq!(out.output.license, "Apache-2.0");
        let tc = fake.train_cmds.lock().unwrap();
        let train = tc.last().unwrap();
        assert!(
            arg_after(train, "--weights")
                .unwrap()
                .ends_with(weights_end),
            "{}",
            train.display()
        );
        assert!(train.display().contains(&format!("--trainer {trainer}")));
        assert_eq!(out.benchmark, None);
    }
}

#[test]
fn cancel_kills_the_training_and_installs_nothing() {
    let t = tmp("cancel");
    let j = job(&t.0, TrainFamily::Yolo26, "s", mps(), Platform::MACOS_ARM64);
    let fake = TrainFake {
        hang: true,
        ..TrainFake::new()
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        c2.store(true, Ordering::Relaxed);
    });
    let (r, _) = run(&j, &fake, &NoHooks, &cancel);
    stopper.join().unwrap();
    let e = r.unwrap_err();
    assert!(e.cancelled, "{e:?}");
    assert_eq!(e.stage, Some(TrainStage::Train));
    assert!(!t.0.join("models/night-yolo26s-test.onnx").exists());
    assert!(
        !t.0.join("models/.night-yolo26s-test.train-staging")
            .exists()
    );
}

#[test]
fn gpu_out_of_memory_says_to_lower_the_batch() {
    let t = tmp("oom");
    let j = job(
        &t.0,
        TrainFamily::Rfdetr,
        "small",
        cuda(),
        Platform::LINUX_X64,
    );
    let fake = TrainFake {
        cuda: true,
        mps: false,
        fail_lines: vec![
            "Traceback (most recent call last):",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        ],
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert_eq!(e.stage, Some(TrainStage::Train));
    assert!(e.message.contains("GPU ran out of memory"), "{}", e.message);
    assert!(e.message.contains("smaller batch (now 4)"), "{}", e.message);
    // The CUDA lock came from the cu126 index.
    let pip = fake
        .inner
        .cmds()
        .into_iter()
        .find(|c| c.contains(" pip install "))
        .unwrap();
    assert!(pip.contains("--torch-backend cu126"), "{pip}");
}

#[test]
fn a_self_check_without_the_gpu_stops_early() {
    let t = tmp("nogpu");
    let j = job(
        &t.0,
        TrainFamily::Yolo11,
        "n",
        cuda(),
        Platform::WINDOWS_X64,
    );
    let fake = TrainFake {
        cuda: false,
        mps: false,
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert_eq!(e.stage, Some(TrainStage::SelfCheck));
    assert!(
        e.message.contains("does not see the NVIDIA GPU"),
        "{}",
        e.message
    );
    assert_eq!(fake.train_cmds.lock().unwrap().len(), 1);
}

#[test]
fn a_class_mismatch_from_the_trainer_is_an_error() {
    let t = tmp("classes");
    let j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    let fake = TrainFake {
        wrong_classes: Some(vec!["cat".into()]),
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert!(e.message.contains("classes"), "{}", e.message);
    assert!(!t.0.join("models/night-yolo26n-test.onnx").exists());
}

#[test]
fn a_failed_benchmark_keeps_the_model() {
    let t = tmp("bench");
    let j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    let fake = TrainFake::new();
    let hooks = RecHooks {
        bench_fails: true,
        ..RecHooks::default()
    };
    let (r, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    let out = r.unwrap();
    assert!(out.output.onnx.is_file());
    assert_eq!(out.benchmark, None);
    assert!(
        out.benchmark_error
            .as_deref()
            .unwrap()
            .contains("no runtime")
    );
}

#[test]
fn classify_names_the_gpu_failures() {
    let r = TrainRecipe::new(TrainFamily::Yolo26, "m", 1).unwrap();
    let l = |s: &str| vec![s.to_string()];
    assert!(
        classify_train_failure(&l("RuntimeError: MPS backend out of memory"), &r)
            .unwrap()
            .contains("batch (now 8)")
    );
    assert!(
        classify_train_failure(
            &l("CUDA error: no kernel image is available for execution"),
            &r
        )
        .unwrap()
        .contains("no kernels")
    );
    assert!(classify_train_failure(&l("all fine"), &r).is_none());
}

#[test]
fn stage_spans_cover_the_bar() {
    assert_eq!(TrainStage::Setup.overall(0), 0);
    assert_eq!(TrainStage::Train.overall(50), 56);
    assert_eq!(TrainStage::Benchmark.overall(100), 100);
    let mut last = 0;
    for s in TrainStage::ALL {
        assert!(s.overall(0) >= last);
        last = s.overall(100);
    }
}

#[test]
fn the_queue_runs_one_at_a_time_cancels_and_persists() {
    let t = tmp("queue");
    let fake = Arc::new(TrainFake {
        hang: true,
        ..TrainFake::new()
    });
    let store = t.0.join("training").join("runs.json");
    let svc = TrainingService::start(
        fake.clone(),
        fake.clone(),
        Arc::new(NoHooks),
        Some(store.clone()),
    );
    let mut a = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    a.run_dir = t.0.join("training/a");
    let mut b = job(&t.0, TrainFamily::Yolo26, "s", mps(), Platform::MACOS_ARM64);
    b.run_dir = t.0.join("training/b");
    let mut dup = a.clone();
    dup.id = "other".into();
    let ida = svc.enqueue(a).unwrap();
    let idb = svc.enqueue(b).unwrap();
    assert!(svc.enqueue(dup).unwrap_err().contains("already queued"));
    // a runs (hangs in training), b waits.
    let deadline = Instant::now() + Duration::from_secs(10);
    while svc.get(&ida).unwrap().stage != Some(TrainStage::Train) {
        assert!(Instant::now() < deadline, "{:?}", svc.get(&ida));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(svc.get(&idb).unwrap().state, RunState::Queued);
    assert!(svc.busy());
    // Cancel the queued one, then the running one.
    assert!(svc.cancel(&idb));
    assert_eq!(svc.get(&idb).unwrap().state, RunState::Cancelled);
    assert!(svc.cancel(&ida));
    let ra = svc.wait(&ida, Duration::from_secs(10)).unwrap();
    assert_eq!(ra.state, RunState::Cancelled, "{}", ra.describe());
    assert!(!svc.busy());
    assert!(svc.remove(&idb).is_ok());
    svc.shutdown();
    let saved = service::load_records(&store);
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].id, ida);
    assert_eq!(saved[0].state, RunState::Cancelled);
    // A record left running by a stopped process reads back as interrupted.
    let mut stale = saved[0].clone();
    stale.state = RunState::Running;
    std::fs::write(&store, serde_json::to_vec(&vec![stale]).unwrap()).unwrap();
    let back = service::load_records(&store);
    assert_eq!(back[0].state, RunState::Failed);
    assert!(back[0].error.as_deref().unwrap().contains("interrupted"));
}
