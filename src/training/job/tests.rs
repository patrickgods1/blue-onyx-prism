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
    /// Run out of GPU memory when `--batch` is above this, after `oom_epochs` finished epochs.
    oom_above: Option<u32>,
    oom_epochs: u32,
    /// Run out of system memory (the incident's lines) when `--workers` is above this.
    host_oom_above: Option<u32>,
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
            // Like the script: an epoch, then waiting; the stop file (Finish now) ends training.
            on_line(r#"@train {"event":"epoch","epoch":1,"epochs":2,"loss":2.0,"map":0.3}"#);
            let stop = arg_after(cmd, "--stop-file").unwrap();
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Err(anyhow!(Cancelled));
                }
                if stop.exists() {
                    on_line(
                        r#"@train {"event":"stop","reason":"finish","epoch":1,"best_epoch":1,"best":0.3}"#,
                    );
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let batch: u32 = arg_after(cmd, "--batch")
            .and_then(|b| b.to_str().and_then(|b| b.parse().ok()))
            .unwrap_or(0);
        let workers: u32 = arg_after(cmd, "--workers")
            .and_then(|w| w.to_str().and_then(|w| w.parse().ok()))
            .unwrap_or(0);
        if self.host_oom_above.is_some_and(|max| workers > max) {
            let lines: Vec<String> = HOST_OOM.lines().map(str::to_string).collect();
            for l in &lines {
                on_line(l);
            }
            return Err(anyhow::Error::new(ChildFailed {
                program: "python".into(),
                status: "exit code: 1".into(),
                lines,
            }));
        }
        if self.oom_above.is_some_and(|max| batch > max) {
            for e in 1..=self.oom_epochs {
                on_line(&format!(
                    "@train {{\"event\":\"epoch\",\"epoch\":{e},\"epochs\":9,\"loss\":1.0}}"
                ));
            }
            let lines = vec![
                "Traceback (most recent call last):".to_string(),
                "RuntimeError: MPS backend out of memory (MPS allocated: 11.00 GiB)".to_string(),
            ];
            for l in &lines {
                on_line(l);
            }
            return Err(anyhow::Error::new(ChildFailed {
                program: "python".into(),
                status: "exit status: 1".into(),
                lines,
            }));
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
        _: &[String],
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
    let mut recipe = TrainRecipe::new(family, size).unwrap();
    recipe.epochs = 2;
    recipe.patience = 1;
    recipe.augment = crate::training::Augment::NIGHT;
    TrainJob {
        id: format!("run-{}", uuid::Uuid::new_v4().simple()),
        request: TrainRequest {
            recipe,
            datasets: vec![exp],
            include_public: false,
            scene: SceneFilter::NIGHT_IR,
            classes: None,
            rename: BTreeMap::new(),
            benchmark_sets: vec!["nir-night".into()],
            benchmark_devices: Vec::new(),
            allow_large: false,
            install_dir: None,
            gpu: None,
            excluded: Default::default(),
            balance: Default::default(),
            compare_group: None,
            // A repeat: the trainer's seed differs from the assembly seed (options.seed 0).
            train_seed: Some(2),
        },
        name: format!("night-{}-test", family.base_name(size)),
        paths: TrainPaths::new(root, root.join("models")),
        run_dir: root.join("training").join("run1"),
        platform,
        accelerator: accel,
        options: dataset::Options::default(),
        root: root.to_path_buf(),
        timeout: Duration::from_secs(60),
        workers: Workers::Fixed(0),
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
        "--seed 2",
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
    let (r, _) = std::thread::scope(|s| {
        // Cancel once the training command is running, however slow the earlier stages are.
        s.spawn(|| {
            while !fake
                .train_cmds
                .lock()
                .unwrap()
                .iter()
                .any(|c| !c.display().contains("--self-check"))
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            cancel.store(true, Ordering::Relaxed);
        });
        run(&j, &fake, &NoHooks, &cancel)
    });
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
    // Retried at half the batch down to 1, then the advice names the last batch.
    assert!(e.message.contains("smaller batch (now 1)"), "{}", e.message);
    let batches: Vec<String> = fake
        .train_cmds
        .lock()
        .unwrap()
        .iter()
        .filter(|c| !c.display().contains("--self-check"))
        .map(|c| arg_after(c, "--batch").unwrap().display().to_string())
        .collect();
    assert_eq!(batches, ["4", "2", "1"]);
    // The CUDA lock came from the cu126 index.
    let pip = fake
        .inner
        .cmds()
        .into_iter()
        .find(|c| c.contains(" pip install "))
        .unwrap();
    assert!(pip.contains("--torch-backend cu126"), "{pip}");
}

/// The batches of the training commands `fake` ran (not the self-check).
fn train_batches(fake: &TrainFake) -> Vec<u32> {
    fake.train_cmds
        .lock()
        .unwrap()
        .iter()
        .filter(|c| !c.display().contains("--self-check"))
        .map(|c| {
            arg_after(c, "--batch")
                .unwrap()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

#[test]
fn out_of_memory_early_retries_at_half_the_batch_and_keeps_the_batch_that_ran() {
    let t = tmp("oom-retry");
    let mut j = job(
        &t.0,
        TrainFamily::Rfdetr,
        "nano",
        mps(),
        Platform::MACOS_ARM64,
    );
    j.request.recipe.batch = 16;
    let fake = TrainFake {
        oom_above: Some(4),
        oom_epochs: 1,
        ..TrainFake::new()
    };
    let mut lines = Vec::new();
    let mut batches_reported = Vec::new();
    let mut details = Vec::new();
    let mut last = None;
    let r = run_training(
        &j,
        &fake,
        &fake,
        &NoHooks,
        &mut |p| match p {
            TrainReport::Line(l) => lines.push(l.to_string()),
            TrainReport::Batch(b) => batches_reported.push(b),
            TrainReport::Progress(tp) => {
                details.push(tp.describe());
                last = Some(tp.clone());
            }
            _ => {}
        },
        &AtomicBool::new(false),
    );
    let out = r.unwrap();
    assert_eq!(train_batches(&fake), [16, 8, 4]);
    assert_eq!(batches_reported, [8, 4]);
    assert!(
        lines
            .iter()
            .any(|l| l == "out of GPU memory at batch 16, retrying with batch 8"),
        "{lines:?}"
    );
    assert!(
        details
            .iter()
            .any(|d| d.contains("(out of GPU memory at batch 8, retrying with batch 4)")),
        "{details:?}"
    );
    // The provenance records the batch that trained.
    let prov: Provenance =
        serde_json::from_slice(&std::fs::read(&out.output.provenance).unwrap()).unwrap();
    assert_eq!(prov.recipe.batch, 4);
    // The progress carries the batch that trains (the fake's start event has none) and the
    // run's settings: RF-DETR's small-dataset rates, its cosine schedule to 0.1.
    let last = last.unwrap();
    assert_eq!(last.batch_size, Some(4));
    let s = last.settings.unwrap();
    assert_eq!(
        (s.lr, s.lr_encoder, s.lr_schedule, s.lr_min_factor),
        (
            Some(5e-5),
            Some(7.5e-5),
            crate::training::recipe::LrSchedule::Cosine,
            0.1
        )
    );
    assert_eq!((s.batch, s.epochs, s.seed), (16, 2, j.request.train_seed));
    assert_eq!(s.scene.as_deref(), Some(j.request.scene.as_str()));
    assert_eq!(s.gpu.as_deref(), Some(j.accelerator.name()));
    assert!(s.workers.is_some());
    // macOS: the MPS allocator is capped, RF-DETR gets its multi-scale choice (9000 MB: auto
    // keeps it on at batch 4).
    let cmds = fake.train_cmds.lock().unwrap();
    let train = cmds
        .iter()
        .rfind(|c| !c.display().contains("--self-check"))
        .unwrap();
    if std::env::var_os("PYTORCH_MPS_HIGH_WATERMARK_RATIO").is_none() {
        assert!(
            train
                .env
                .iter()
                .any(|(k, v)| k == "PYTORCH_MPS_HIGH_WATERMARK_RATIO" && v == "1.0"),
            "{:?}",
            train.env
        );
    }
    assert_eq!(
        arg_after(train, "--multi-scale").unwrap().to_str(),
        Some("on")
    );
    // A tiny training split: RF-DETR's small-dataset learning rates. Patience 2 of 2 epochs is
    // no early stop, so no anneal either.
    for (k, v) in [
        ("--lr", "0.00005"),
        ("--lr-encoder", "0.000075"),
        ("--anneal", "0"),
    ] {
        assert_eq!(arg_after(train, k).unwrap().to_str(), Some(v), "{k}");
    }
    let log = std::fs::read_to_string(j.run_dir.join(LOG_FILE)).unwrap();
    assert!(
        log.contains("# lr: 0.00005, encoder 0.000075 (")
            && log.contains("training images < 1000: lower than RF-DETR's default"),
        "{log}"
    );
}

#[test]
fn rfdetr_with_early_stopping_anneals_before_it_stops() {
    let t = tmp("anneal");
    let mut j = job(
        &t.0,
        TrainFamily::Rfdetr,
        "nano",
        cuda(),
        Platform::LINUX_X64,
    );
    j.request.recipe.epochs = 30;
    j.request.recipe.patience = 12;
    j.request.recipe.patience_recommended = false;
    j.request.recipe.lr = Some(2e-4);
    j.request.recipe.lr_schedule = Some(crate::training::recipe::LrSchedule::Step);
    j.request.recipe.lr_min_factor = Some(0.2);
    let fake = TrainFake {
        cuda: true,
        mps: false,
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    r.unwrap();
    let cmds = fake.train_cmds.lock().unwrap();
    let train = cmds
        .iter()
        .rfind(|c| !c.display().contains("--self-check"))
        .unwrap();
    for (k, v) in [
        ("--patience", "12"),
        ("--anneal", "6"),
        ("--lr", "0.0002"),
        ("--lr-encoder", "0.0003"),
        ("--lr-scheduler", "step"),
        ("--lr-min-factor", "0.2"),
    ] {
        assert_eq!(arg_after(train, k).unwrap().to_str(), Some(v), "{k}");
    }
    let log = std::fs::read_to_string(j.run_dir.join(LOG_FILE)).unwrap();
    assert!(
        log.contains(
            "patience 12 (set); mAP averaged over the last 3 epochs; at a stop point the \
             learning rate is lowered over 6 more epochs, then training stops"
        ),
        "{log}"
    );
    assert!(
        log.contains(
            "# lr: 0.0002, encoder 0.0003 (set); schedule step (set), anneal factor 0.2 (set)"
        ),
        "{log}"
    );
}

#[test]
fn out_of_memory_after_the_first_epochs_or_at_batch_1_is_not_retried() {
    let t = tmp("oom-late");
    let mut j = job(&t.0, TrainFamily::Yolo26, "n", cuda(), Platform::LINUX_X64);
    j.request.recipe.batch = 16;
    let fake = TrainFake {
        cuda: true,
        mps: false,
        oom_above: Some(4),
        oom_epochs: OOM_RETRY_EPOCHS as u32,
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert!(
        e.message.contains("smaller batch (now 16)"),
        "{}",
        e.message
    );
    assert_eq!(train_batches(&fake), [16]);
    // CUDA: no MPS settings, no RF-DETR option.
    let cmds = fake.train_cmds.lock().unwrap();
    let train = cmds
        .iter()
        .rfind(|c| !c.display().contains("--self-check"))
        .unwrap();
    assert!(
        !train
            .env
            .iter()
            .any(|(k, _)| k.starts_with("PYTORCH_MPS_HIGH"))
    );
    assert!(arg_after(train, "--multi-scale").is_none());
    drop(cmds);
    let t2 = tmp("oom-one");
    let mut j = job(&t2.0, TrainFamily::Yolo26, "n", cuda(), Platform::LINUX_X64);
    j.request.recipe.batch = 1;
    let fake = TrainFake {
        cuda: true,
        mps: false,
        oom_above: Some(0),
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    assert!(r.unwrap_err().message.contains("GPU ran out of memory"));
    assert_eq!(train_batches(&fake), [1]);
}

#[test]
fn multi_scale_auto_turns_off_when_batch_4_does_not_fit() {
    let mut r = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
    // 16 GB Mac share: batch 4 fits with multi-scale.
    assert!(r.multi_scale_on(9830));
    // 6 GB: batch 4 at 544 px does not (1500 + 4 x 650 x 2.0 = 6.7 GB > 5.4 GB).
    assert!(!r.multi_scale_on(6144));
    r.multi_scale = super::super::recipe::MultiScale::On;
    assert!(r.multi_scale_on(6144));
    r.multi_scale = super::super::recipe::MultiScale::Off;
    assert!(!r.multi_scale_on(100_000));
    let y = TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap();
    assert!(!y.multi_scale_on(100_000));
    // Stored only when not auto.
    let text =
        serde_json::to_string(&TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap()).unwrap();
    assert!(!text.contains("multi_scale"), "{text}");
    assert!(
        serde_json::to_string(&r)
            .unwrap()
            .contains(r#""multi_scale":"off""#)
    );
}

#[test]
fn a_chosen_gpu_trains_on_its_device_in_its_lanes_environment() {
    let t = tmp("gpu");
    // The second GPU of a machine, a Blackwell card: CUDA 13.0 build.
    let accel = Accelerator::Cuda {
        ordinal: 1,
        name: "NVIDIA GeForce RTX 5090".into(),
        compute_capability: ComputeCapability::new(12, 0),
        vram_mb: 32768,
        lane: TrainLane::Cu130,
    };
    let j = job(&t.0, TrainFamily::Yolo26, "n", accel, Platform::LINUX_X64);
    let fake = TrainFake {
        cuda: true,
        mps: false,
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    r.unwrap();
    let cmds = fake.train_cmds.lock().unwrap().clone();
    assert_eq!(cmds.len(), 2, "self-check, training");
    for c in &cmds {
        assert_eq!(
            arg_after(c, "--device").unwrap().to_str(),
            Some("cuda:1"),
            "{}",
            c.display()
        );
    }
    let pip = fake
        .inner
        .cmds()
        .into_iter()
        .find(|c| c.contains(" pip install "))
        .unwrap();
    assert!(pip.contains("--torch-backend cu130"), "{pip}");
    assert!(env::train_env_ready(&j.paths, j.platform, TrainLane::Cu130));
    assert!(!env::train_env_ready(
        &j.paths,
        j.platform,
        TrainLane::Cu126
    ));
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
fn a_run_installs_into_the_chosen_folder() {
    let t = tmp("install-dir");
    let mut j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    let elsewhere = t.0.join("elsewhere").join("models");
    j.request.install_dir = Some(elsewhere.clone());
    assert_eq!(j.install_dir(), elsewhere.as_path());
    let fake = TrainFake::new();
    let hooks = RecHooks::default();
    let (r, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    let o = r.unwrap().output;
    assert_eq!(o.onnx, elsewhere.join("night-yolo26n-test.onnx"));
    assert_eq!(o.yaml, elsewhere.join("night-yolo26n-test.yaml"));
    assert_eq!(
        o.provenance,
        elsewhere.join("night-yolo26n-test.provenance.json")
    );
    assert!(o.onnx.is_file() && o.yaml.is_file() && o.provenance.is_file());
    assert!(!t.0.join("models/night-yolo26n-test.onnx").exists());
    assert!(!elsewhere.join(".night-yolo26n-test.train-staging").exists());
    // The hooks get the installed paths (registered and benchmarked from there).
    assert_eq!(hooks.registered.lock().unwrap()[0].onnx, o.onnx);
    // The config entry points at the real file (absolute: outside the data root here).
    let mut c = crate::config::Config {
        download_dir: Some(t.0.join("root")),
        ..crate::config::Config::default()
    };
    crate::training::cli::add_trained_model(&mut c, &o).unwrap();
    let m = c
        .models
        .iter()
        .find(|m| m.effective_name() == "night-yolo26n-test")
        .unwrap();
    assert_eq!(
        c.data_path(&m.path).canonicalize().unwrap(),
        o.onnx.canonicalize().unwrap()
    );
    // The same name again in that folder is refused before anything runs.
    let (again, _) = run(&j, &fake, &hooks, &AtomicBool::new(false));
    assert!(again.unwrap_err().message.contains("already exists"));
}

#[test]
fn requests_round_trip_and_old_records_load() {
    let t = tmp("roundtrip");
    let mut j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    // Without an install folder the key is not written.
    let text = serde_json::to_string(&j.request).unwrap();
    assert!(!text.contains("install_dir"), "{text}");
    j.request.install_dir = Some(t.0.join("dst"));
    let text = serde_json::to_string(&j.request).unwrap();
    let back: TrainRequest = serde_json::from_str(&text).unwrap();
    assert_eq!(back, j.request);
    // A runs.json written before the field existed still loads.
    let old = r#"[{"id":"20260101-000000-abcd","name":"night-x","created":"2026-01-01T00:00:00Z",
        "request":{"recipe":RECIPE,"datasets":["/e"],"include_public":false,"scene":"all"},
        "device":"mps","state":"done","stage":null,"step":8,"steps":8,"percent":100,
        "detail":"","run_dir":"/t/run"}]"#
        .replace("RECIPE", &serde_json::to_string(&j.request.recipe).unwrap());
    let store = t.0.join("runs.json");
    std::fs::write(&store, old).unwrap();
    let recs = service::load_records(&store);
    assert_eq!(recs.len(), 1, "the old record loads");
    assert_eq!(recs[0].request.install_dir, None);
}

#[test]
fn the_queue_moves_to_another_training_folder_only_when_idle() {
    let t = tmp("store");
    let fake = Arc::new(TrainFake {
        hang: true,
        ..TrainFake::new()
    });
    let old = t.0.join("old").join("runs.json");
    let svc = TrainingService::start(
        fake.clone(),
        fake.clone(),
        Arc::new(NoHooks),
        Some(old.clone()),
    );
    let mut a = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    a.run_dir = t.0.join("old/a");
    let ida = svc.enqueue(a).unwrap();
    let new = t.0.join("new").join("runs.json");
    assert!(
        svc.set_store(&new)
            .unwrap_err()
            .contains("queued or running")
    );
    assert_eq!(svc.store().as_deref(), Some(old.as_path()));
    svc.cancel(&ida);
    svc.wait(&ida, Duration::from_secs(10)).unwrap();
    // Idle: the queue reads and writes the new folder; the old runs stay where they were.
    assert!(svc.set_store(&new).unwrap());
    assert!(!svc.set_store(&new).unwrap(), "already there");
    assert_eq!(svc.store().as_deref(), Some(new.as_path()));
    assert!(svc.runs().is_empty());
    assert!(svc.tasks().is_empty());
    assert_eq!(service::load_records(&old).len(), 1);
    // And back: the old folder's runs show again.
    assert!(svc.set_store(&old).unwrap());
    assert_eq!(svc.runs().len(), 1);
    svc.shutdown();
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

const LINUX_HOST: HostMemory = HostMemory {
    windows: false,
    workers: 2,
    commit: None,
};

#[test]
fn classify_names_the_gpu_failures() {
    let r = TrainRecipe::new(TrainFamily::Yolo26, "m").unwrap();
    let l = |s: &str| vec![s.to_string()];
    assert!(
        classify_train_failure(
            &l("RuntimeError: MPS backend out of memory"),
            &r,
            &LINUX_HOST
        )
        .unwrap()
        .contains("batch (now 8)")
    );
    assert!(
        classify_train_failure(
            &l("CUDA error: no kernel image is available for execution"),
            &r,
            &LINUX_HOST
        )
        .unwrap()
        .contains("no kernels")
    );
    assert!(classify_train_failure(&l("all fine"), &r, &LINUX_HOST).is_none());
}

/// Lines of the incident (RF-DETR nano on a GTX 1070 Ti, Windows, a 5.6 GB fixed paging file):
/// the WDDM CUDA allocation that fails with GPU memory free, then the CPU allocation.
const WDDM_OOM: &str = r"[W1029 16:54:57.000000000 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0 while trying to allocate 20971520 bytes (free: 6624903168, total: 8589803520).
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 6.17 GiB is free. Of the allocated memory 845.46 MiB is allocated by PyTorch, and 4.54 MiB is reserved by PyTorch but unallocated.
Traceback (most recent call last):
    return self._apply(lambda t: t.cpu())
RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 262144 bytes.";

/// The incident's last attempt: the validation workers fail to start at the end of epoch 1.
const HOST_OOM: &str = r#"Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:	\Lib\multiprocessing\spawn.py", line 122, in spawn_main
OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading "C:	\Lib\site-packages	orch\lib\cufft64_11.dll" or one of its dependencies.
OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: 'C:\t\DLLs'
RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3993600 bytes.
RuntimeError: can't start new thread"#;

#[test]
fn host_out_of_memory_is_not_a_gpu_failure() {
    let l = |s: &str| -> Vec<String> { s.lines().map(str::to_string).collect() };
    // WDDM: "CUDA out of memory" with the CPU allocator failing too = the host.
    let wddm = l(WDDM_OOM);
    assert!(
        host_out_of_memory_line(&wddm)
            .unwrap()
            .contains("DefaultCPUAllocator")
    );
    assert_eq!(out_of_memory_line(&wddm), None);
    // The paging file line wins over the later, vaguer ones.
    let spawn = l(HOST_OOM);
    assert!(
        host_out_of_memory_line(&spawn)
            .unwrap()
            .contains("[WinError 1455]")
    );
    // A real GPU OOM stays one; `torch.OutOfMemoryError` is not Python's `MemoryError`.
    let gpu = l("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB");
    assert_eq!(host_out_of_memory_line(&gpu), None);
    assert!(out_of_memory_line(&gpu).is_some());
    assert!(host_out_of_memory_line(&l("MemoryError: Unable to allocate 3.1 GiB")).is_some());
    assert!(
        host_out_of_memory_line(&l(
            "RuntimeError: DataLoader worker (pid 4242) is killed by signal: Killed."
        ))
        .is_some()
    );

    let mut r = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
    r.batch = 4;
    let windows = HostMemory {
        windows: true,
        workers: 5,
        commit: Some(crate::training::Commit {
            limit: 21_300 << 20,
            available: 2 << 30,
        }),
    };
    let m = classify_train_failure(&wddm, &r, &windows).unwrap();
    assert!(
        m.starts_with("training ran out of system memory (RuntimeError: [enforce fail"),
        "{m}"
    );
    assert!(
        m.contains("each of the 5 data loader workers loads PyTorch (~1 GB each); commit 18.8 of 20.8 GB was in use when training started."),
        "{m}"
    );
    assert!(m.contains("Let Windows manage the paging file size"), "{m}");
    assert!(m.contains("training.workers to 0"), "{m}");
    assert!(m.ends_with("then train again"), "{m}");
    assert!(!m.contains("GPU"), "{m}");
    // Down to no workers: the advice is the paging file alone.
    let m = classify_train_failure(
        &spawn,
        &r,
        &HostMemory {
            workers: 0,
            commit: None,
            ..windows
        },
    )
    .unwrap();
    assert!(m.contains("[WinError 1455]"), "{m}");
    assert!(m.contains("(no data loader workers)"), "{m}");
    assert!(
        !m.contains("training.workers") && !m.contains("commit"),
        "{m}"
    );
    let m = classify_train_failure(&spawn, &r, &LINUX_HOST).unwrap();
    assert!(m.contains("Close other programs or add swap"), "{m}");
}

/// The `--workers` of the training commands `fake` ran (not the self-check).
fn train_workers(fake: &TrainFake) -> Vec<u32> {
    fake.train_cmds
        .lock()
        .unwrap()
        .iter()
        .filter(|c| !c.display().contains("--self-check"))
        .map(|c| {
            arg_after(c, "--workers")
                .unwrap()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

#[test]
fn out_of_system_memory_halves_the_workers_and_keeps_the_batch() {
    let t = tmp("host-oom");
    let mut j = job(
        &t.0,
        TrainFamily::Rfdetr,
        "nano",
        cuda(),
        Platform::LINUX_X64,
    );
    j.request.recipe.batch = 4;
    j.workers = Workers::Fixed(5);
    let fake = TrainFake {
        cuda: true,
        mps: false,
        host_oom_above: Some(1),
        ..TrainFake::new()
    };
    let mut lines = Vec::new();
    let mut batches_reported = Vec::new();
    let r = run_training(
        &j,
        &fake,
        &fake,
        &NoHooks,
        &mut |p| match p {
            TrainReport::Line(l) => lines.push(l.to_string()),
            TrainReport::Batch(b) => batches_reported.push(b),
            _ => {}
        },
        &AtomicBool::new(false),
    );
    let out = r.unwrap();
    assert_eq!(train_workers(&fake), [5, 2, 1]);
    assert_eq!(train_batches(&fake), [4, 4, 4]);
    assert!(batches_reported.is_empty(), "{batches_reported:?}");
    assert!(
        lines
            .iter()
            .any(|l| l == "out of system memory with 5 data loader workers, retrying with 2"),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "out of system memory with 2 data loader workers, retrying with 1"),
        "{lines:?}"
    );
    let prov: Provenance =
        serde_json::from_slice(&std::fs::read(&out.output.provenance).unwrap()).unwrap();
    assert_eq!(prov.recipe.batch, 4);
    let log = std::fs::read_to_string(j.run_dir.join(LOG_FILE)).unwrap();
    assert!(
        log.contains("# data loader workers: 5 (training.workers)"),
        "{log}"
    );

    // Never enough: 5 -> 2 -> 1 -> 0, then the host message (not the GPU's).
    let t2 = tmp("host-oom-all");
    let mut j = job(
        &t2.0,
        TrainFamily::Rfdetr,
        "nano",
        cuda(),
        Platform::LINUX_X64,
    );
    j.request.recipe.batch = 4;
    j.workers = Workers::Fixed(5);
    let fake = TrainFake {
        cuda: true,
        mps: false,
        fail_lines: WDDM_OOM.lines().collect(),
        ..TrainFake::new()
    };
    let (r, _) = run(&j, &fake, &NoHooks, &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert_eq!(train_workers(&fake), [5, 2, 1, 0]);
    assert_eq!(train_batches(&fake), [4, 4, 4, 4]);
    assert!(
        e.message.starts_with("training ran out of system memory"),
        "{}",
        e.message
    );
    assert!(
        e.message.contains("(no data loader workers)"),
        "{}",
        e.message
    );
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

#[test]
fn finish_now_writes_the_stop_file_and_the_run_exports_its_best_epoch() {
    let t = tmp("finish");
    let fake = Arc::new(TrainFake {
        hang: true,
        ..TrainFake::new()
    });
    let svc = TrainingService::start(fake.clone(), fake.clone(), Arc::new(NoHooks), None);
    let mut a = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    a.run_dir = t.0.join("training/a");
    // A stale request of an earlier attempt does not end this one.
    std::fs::create_dir_all(a.run_dir.join("work")).unwrap();
    std::fs::write(stop_file(&a.run_dir), "old").unwrap();
    let mut b = job(&t.0, TrainFamily::Yolo26, "s", mps(), Platform::MACOS_ARM64);
    b.run_dir = t.0.join("training/b");
    let ida = svc.enqueue(a).unwrap();
    let idb = svc.enqueue(b).unwrap();
    assert!(svc.finish(&idb).unwrap_err().contains("only a running run"));
    let deadline = Instant::now() + Duration::from_secs(10);
    while svc
        .get(&ida)
        .unwrap()
        .progress
        .is_none_or(|p| p.history.is_empty())
    {
        assert!(Instant::now() < deadline, "{:?}", svc.get(&ida));
        std::thread::sleep(Duration::from_millis(10));
    }
    let args = fake.train_cmds.lock().unwrap().last().unwrap().clone();
    // The recommended-style patience 1 doubles on the tiny validation split (at most the 2
    // epochs); the mAP is averaged over 3 epochs.
    for (k, v) in [
        ("--patience", "2"),
        ("--min-epochs", "10"),
        ("--min-delta", "0.001"),
        ("--smooth", "3"),
        ("--anneal", "0"),
    ] {
        assert_eq!(arg_after(&args, k).unwrap(), PathBuf::from(v), "{k}");
    }
    // YOLO: no typed learning rate, none passed (no encoder rate for any YOLO).
    assert!(arg_after(&args, "--lr").is_none());
    assert!(arg_after(&args, "--lr-encoder").is_none());
    // No schedule set: the script picks the trainer's own.
    assert!(arg_after(&args, "--lr-scheduler").is_none());
    assert!(arg_after(&args, "--lr-min-factor").is_none());
    let stop = arg_after(&args, "--stop-file").unwrap();
    assert_eq!(PathBuf::from(&stop), stop_file(&t.0.join("training/a")));
    assert!(!stop.exists(), "the stale request was removed");
    // Paused: finishing resumes it.
    assert!(svc.pause(&ida));
    let msg = svc.finish(&ida).unwrap();
    assert!(msg.contains("after the current epoch"), "{msg}");
    assert!(stop.exists());
    let r = svc.wait(&ida, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Done, "{}", r.describe());
    let p = r.progress.unwrap();
    assert_eq!(
        p.stopped,
        Some(crate::training::progress::Stop {
            reason: "finish".into(),
            epoch: 1,
            best_epoch: 1
        })
    );
    let e = svc.finish(&ida).unwrap_err();
    assert!(e.contains("only a running run"), "{e}");
    let log = std::fs::read_to_string(t.0.join("training/a").join(LOG_FILE)).unwrap();
    assert!(
        log.contains(
            "validation images (< 50): patience 2 instead of 1; mAP averaged over the last 3 epochs"
        ),
        "{log}"
    );
    assert!(svc.cancel(&idb));
    svc.shutdown();
}
