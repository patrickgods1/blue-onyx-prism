//! In-app model training (docs/TRAINING.md): fine-tune YOLO26 / YOLO11 / YOLOv8 (Ultralytics,
//! AGPL-3.0), RF-DETR, RT-DETRv2 or D-FINE (Apache-2.0) on the frames reviewed on the Review page,
//! export a static ONNX, install it next to the other models, add it to the config and benchmark
//! it on held-out frames from cameras and days it did not train on.
//!
//! - [`capability`]: can this machine train, on which GPU, which families and sizes fit (pure).
//! - [`recipe`]: [`TrainRecipe`] (family, size, base weights, input size, epochs, batch, learning
//!   rate, night augmentation) and its validation.
//! - [`estimate`]: rough VRAM / time per epoch table.
//! - [`dataset`]: review exports (+ the optional public night subset) -> the trainer's input
//!   (YOLO + COCO) and a held-out benchmark `dir:` set.
//! - [`job`]: one run's stage machine over the model export's seams (`CommandRunner`, `Fetcher`;
//!   the environment is `tools/train-env`, [`crate::resources::export::train`]).
//! - [`service`]: the queue (one run at a time, cancellable, records in `runs.json`).
//! - [`finish`]: the steps after an import chain (reject low-scoring auto boxes, accept the
//!   settled frames, switch the Review folder).
//! - [`bench`]: the benchmark at the end of a run.
//! - [`cli`]: `blue-onyx-prism train ...`. The Training page is `crate::server` (`/training`).
//!
//! Like the export, training is user-triggered only (never by `auto_download`), runs downloaded
//! code only from pinned, hashed sources, and keeps everything under `<data root>/tools` (the
//! environment) and the training folder (`training.dir`, the runs).

pub mod autolabel;
pub mod bench;
pub mod capability;
pub mod cli;
pub mod compare;
pub mod config;
pub mod dataset;
pub mod estimate;
pub mod finish;
pub mod import;
pub mod job;
pub mod progress;
pub mod pyenv;
pub mod recipe;
pub mod service;
pub mod task;

pub use capability::{Accelerator, Capability};
pub use config::TrainingConfig;
pub use job::{TrainJob, TrainRequest};
pub use recipe::{Augment, TrainRecipe};

pub use crate::backend::memory::Commit;
use crate::config::Config;
use crate::resources::catalog::{self, Platform};
use crate::resources::export::train::TrainPaths;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// What this machine can train (detected hardware, cached).
pub fn capability_now() -> Capability {
    capability::capability(
        crate::backend::detect::hardware(),
        crate::system_info::total_memory_gb(),
    )
}

/// [`capability_now`] with the batch suggestions of earlier runs' measured peaks
/// ([`measured_peaks`] of `runs`).
pub fn capability_with_runs(runs: &[service::RunRecord]) -> Capability {
    let mut cap = capability_now();
    let mem = cap
        .accelerator
        .as_ref()
        .map_or(0, capability::Accelerator::memory_mb);
    cap.apply_measured(&measured_peaks(runs, mem));
    cap
}

/// The peak GPU memory of every run that reported `mem_gb` for a finished epoch, oldest first
/// (by `created`); `memory_mb` resolves RF-DETR's `auto` multi-scale as the run did.
pub fn measured_peaks(
    runs: &[service::RunRecord],
    memory_mb: u64,
) -> Vec<capability::MeasuredPeak> {
    let mut v: Vec<(&str, capability::MeasuredPeak)> = runs
        .iter()
        .filter_map(|r| {
            let peak = r
                .progress
                .as_ref()?
                .history
                .iter()
                .filter_map(|p| p.metrics.get("mem_gb").copied())
                .filter(|g| g.is_finite() && *g > 0.0)
                .fold(None, |m: Option<f64>, g| Some(m.map_or(g, |m| m.max(g))))?;
            let recipe = &r.request.recipe;
            Some((
                r.created.as_str(),
                capability::MeasuredPeak {
                    family: recipe.family,
                    size: recipe.size.clone(),
                    imgsz: recipe.imgsz,
                    multi_scale: recipe.multi_scale_on(memory_mb),
                    device: r.device.clone(),
                    batch: recipe.batch,
                    peak_mb: peak * 1024.0,
                },
            ))
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v.into_iter().map(|(_, p)| p).collect()
}

/// The training folder of `config`.
pub fn training_dir(config: &Config) -> PathBuf {
    config.training.dir_in(&config.data_root())
}

/// `runs.json` of `config`'s training folder.
pub fn runs_file(config: &Config) -> PathBuf {
    training_dir(config).join(service::RUNS_FILE)
}

/// Review exports found under `<training dir>/exports/` (the Review page's default export
/// folder), newest first, at most 20: folders with a `classes.txt` or `data.yaml`.
pub fn discover_exports(config: &Config) -> Vec<PathBuf> {
    let root = training_dir(config).join("exports");
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir() && (p.join("classes.txt").is_file() || p.join("data.yaml").is_file())
        })
        .collect();
    v.sort();
    v.reverse();
    v.truncate(20);
    v
}

/// Install folder of the public night subset (`train-data:lila-night`) under the download root.
pub fn public_subset_dir(root: &Path) -> PathBuf {
    root.join(crate::resources::catalog::LILA_NIGHT_DEST)
}

/// Whether the public night subset is downloaded and complete under `root`.
pub fn public_subset_installed(root: &Path) -> bool {
    crate::resources::catalog::lila_night()
        .is_some_and(|r| crate::resources::manager::is_installed(r, &public_subset_dir(root)))
}

/// The large-download gate of the public night subset (about 926 MB, over the 500 MB limit like
/// the training environment): refuse a run that would download it unless `allowed`
/// (`--allow-large`, the Training page's download confirmation, or `allow_large_downloads`).
/// Nothing to refuse when it is downloaded already.
pub fn check_public_download(root: &Path, allowed: bool) -> Result<()> {
    let Some(r) = catalog::lila_night() else {
        bail!(
            "the public night subset is not pinned in this build (its manifest lists no image); \
             train without it"
        );
    };
    if allowed || !r.is_large() || public_subset_installed(root) {
        return Ok(());
    }
    bail!(
        "the public night subset ({}, {} images) is a large download ({}, over the 500 MB \
         limit) and is not downloaded yet under {}: rerun with --allow-large (Training page: tick \
         \"Download what is missing\"), enable allow_large_downloads, or train without the \
         public subset",
        r.id,
        r.parts.len(),
        catalog::format_size(r.size()),
        public_subset_dir(root).display()
    )
}

/// Data loader workers the core count allows at most (CUDA GPUs outrun one decoding process; more
/// than 8 did not help).
pub const MAX_AUTO_WORKERS: u32 = 8;
/// Commit one data loader worker takes on Windows: each is a spawned Python that imports PyTorch
/// (measured ~0.9 GB commit for `import torch` alone), plus its prefetched batches (measured
/// ~1.4 GB per spawned worker during an RF-DETR nano run).
pub const WORKER_COMMIT_BYTES: u64 = 1400 << 20;
/// Commit kept free for the training process itself (model, optimizer, CUDA context, pinned
/// memory) before any worker is counted (measured 5.5 GB for the RF-DETR nano trainer process).
pub const MAIN_COMMIT_RESERVE_BYTES: u64 = 6 << 30;

/// Worker processes alive at once per `--workers` of `trainer`, read from the pinned packages:
/// - RF-DETR (rfdetr 1.11.2, `training/module_data.py` `train_dataloader` and `val_dataloader`):
///   both loaders get `num_workers` and `persistent_workers` (`train_model.py` sets it), so the
///   validation workers spawned at the end of epoch 1 join the training ones: 2.
/// - Ultralytics (8.4.163, `models/yolo/detect/train.py` `get_dataloader`): the validation
///   loader gets `workers * 2`, and both are `InfiniteDataLoader`s (`data/build.py`) that keep
///   their workers for the whole run: 1 + 2 = 3.
/// - transformers' `Trainer` (5.17.0, `trainer.py` `get_eval_dataloader`): with
///   `dataloader_persistent_workers` the evaluation loader is cached with its workers next to
///   the training one: 2.
pub const fn worker_loaders(trainer: catalog::Trainer) -> u32 {
    match trainer {
        catalog::Trainer::RfDetr | catalog::Trainer::Transformers => 2,
        catalog::Trainer::Ultralytics => 3,
    }
}

/// How many data loader workers a run gets, and why (logged at the start of training).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerChoice {
    /// `--workers` of the trainer (0 = load in the training process).
    pub workers: u32,
    /// What the cores alone allow.
    pub by_cores: u32,
    /// The commit seen when choosing (Windows only).
    pub commit: Option<Commit>,
}

impl WorkerChoice {
    /// The log line: "memory: commit 21.3 GB limit, 5.1 GB free -> 0 data loader workers (cores
    /// allow 5)".
    pub fn describe(&self) -> String {
        let n = workers_text(self.workers);
        match self.commit {
            Some(c) => format!(
                "memory: commit {:.1} GB limit, {:.1} GB free -> {n} (cores allow {})",
                gib(c.limit),
                gib(c.available),
                self.by_cores
            ),
            None => format!(
                "data loader workers: {} (cores allow {})",
                self.workers, self.by_cores
            ),
        }
    }
}

/// "1 data loader worker", "3 data loader workers".
pub fn workers_text(n: u32) -> String {
    format!("{n} data loader worker{}", if n == 1 { "" } else { "s" })
}

/// Bytes in GiB, for messages.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// Data loader workers on `os` with `cpus` logical cores, `commit` (Windows) and `loaders`
/// ([`worker_loaders`]): none on macOS (measured on an M1: 2 or 4 spawned workers did not speed
/// up RF-DETR nano, which is GPU-bound there, and Ultralytics uses none on MPS;
/// docs/TRAINING.md), else all cores but one, 1..=[`MAX_AUTO_WORKERS`]. On Windows every worker
/// is a fresh Python importing PyTorch, so a fixed paging file runs out of commit long before
/// RAM (WinError 1455): there also at most what the free commit holds after
/// [`MAIN_COMMIT_RESERVE_BYTES`], [`WORKER_COMMIT_BYTES`] per worker per loader, down to 0.
pub fn workers_for(os: &str, cpus: usize, commit: Option<Commit>, loaders: u32) -> WorkerChoice {
    let by_cores = if os == "macos" {
        0
    } else {
        u32::try_from(cpus.saturating_sub(1))
            .unwrap_or(u32::MAX)
            .clamp(1, MAX_AUTO_WORKERS)
    };
    let commit = commit.filter(|_| os == "windows");
    let by_memory = commit.map_or(u32::MAX, |c| {
        let per = WORKER_COMMIT_BYTES * u64::from(loaders.max(1));
        let n = c.available.saturating_sub(MAIN_COMMIT_RESERVE_BYTES) / per;
        u32::try_from(n).unwrap_or(u32::MAX)
    });
    WorkerChoice {
        workers: by_cores.min(by_memory),
        by_cores,
        commit,
    }
}

/// [`workers_for`] on this machine now, for `trainer`.
pub fn workers_now(trainer: catalog::Trainer) -> WorkerChoice {
    let cpus = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    workers_for(
        std::env::consts::OS,
        cpus,
        crate::system_info::commit(),
        worker_loaders(trainer),
    )
}

/// The dataset options of `request` on `config`: the `training` section's assembly defaults
/// with the request's scene, classes, renames, excluded mode and balance. The job
/// ([`plan_job`]) and [`preview_plan`] both build them here, so a preview is the run's plan.
pub fn options_for(config: &Config, request: &TrainRequest) -> dataset::Options {
    let t = &config.training;
    let base = dataset::Options {
        negative_share: t.negative_share,
        heldout_share: t.heldout_share,
        oversample: t.oversample,
        ..dataset::Options::default()
    };
    job::request_options(&base, request)
}

/// The plan a run of `request` would train on, without writing anything: the sources read like
/// the job reads them (relative dataset paths against the data root, `training.datasets` when
/// the request names none) and [`options_for`]. The public subset is read only when it is
/// installed; asked for but missing, the plan leaves it out and says so first in its warnings.
pub fn preview_plan(config: &Config, request: &TrainRequest) -> Result<dataset::Plan> {
    request.balance.validate()?;
    let datasets = if request.datasets.is_empty() {
        &config.training.datasets
    } else {
        &request.datasets
    };
    if datasets.is_empty() {
        bail!(
            "no dataset: choose a review export folder or set training.datasets (export reviewed \
             frames from the Review page in YOLO format first)"
        );
    }
    let datasets: Vec<PathBuf> = datasets.iter().map(|d| config.data_path(d)).collect();
    let root = config.data_root();
    let public_dir = public_subset_dir(&root);
    let public = request.include_public && public_subset_installed(&root);
    let sources = job::read_sources(&datasets, public.then_some(public_dir.as_path()))?;
    let mut plan = dataset::plan(&sources, &options_for(config, request))?;
    if request.include_public && !public {
        plan.warnings.insert(
            0,
            "the public night subset is not downloaded yet: this preview leaves it out (the run \
             downloads and mixes it in, repeating your frames training.oversample times)"
                .into(),
        );
    }
    Ok(plan)
}

/// Build the job of `request` on `config` and `cap` (see the checks below; `request.gpu` picks
/// one of `cap`'s GPUs, else the automatic pick). `name`: the model name, default
/// `<scene label>-<model>-<stamp>` (`custom-`, `day-`, `night-`, ...).
pub fn plan_job(
    config: &Config,
    mut request: TrainRequest,
    cap: &Capability,
    name: Option<String>,
) -> Result<TrainJob> {
    // A chosen GPU: its accelerator, lane and fits.
    let cap = &cap.for_gpu(request.gpu.as_deref())?;
    let Some(accel) = cap.accelerator.clone() else {
        bail!("{}", cap.summary());
    };
    // Stored as the device it resolved to (`auto` / empty = the automatic pick, None).
    request.gpu = capability::gpu_arg(request.gpu.as_deref())?.map(|_| accel.device_arg());
    let recipe = &request.recipe;
    recipe.validate()?;
    if let Some(f) = cap.family(recipe.family)
        && !f.available
    {
        bail!(
            "{} cannot train here: {}",
            recipe.family.title(),
            f.why.as_deref().unwrap_or("not supported")
        );
    }
    request.allow_large |= config.allow_large_downloads;
    if request.include_public {
        check_public_download(&config.data_root(), request.allow_large)?;
    }
    if request.datasets.is_empty() {
        request.datasets = config.training.datasets.clone();
    }
    if request.datasets.is_empty() {
        bail!(
            "no dataset: pass --dataset <review export folder> or set training.datasets (export \
             reviewed frames from the Review page in YOLO format first)"
        );
    }
    request.datasets = request
        .datasets
        .iter()
        .map(|d| config.data_path(d))
        .collect();
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M").to_string();
    let name = name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| request.recipe.default_name(request.scene.label(), &stamp));
    recipe::valid_model_name(&name)?;
    let paths = TrainPaths::for_config(config);
    request.install_dir = request
        .install_dir
        .take()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| config.data_path(&d))
        .filter(|d| *d != paths.export.models);
    if let Some(d) = &request.install_dir
        && d.exists()
        && !d.is_dir()
    {
        bail!(
            "install the model in: {} is not a folder; choose a folder or leave it empty for the \
             models folder",
            d.display()
        );
    }
    let install = request
        .install_dir
        .as_deref()
        .unwrap_or(&paths.export.models)
        .join(format!("{name}.onnx"));
    if install.exists() {
        bail!("{} exists; choose another model name", install.display());
    }
    if config
        .models
        .iter()
        .any(|m| m.effective_name().eq_ignore_ascii_case(&name))
    {
        bail!("the config already has a model named {name}; choose another name");
    }
    request.balance.validate()?;
    let id = service::new_run_id();
    let t = &config.training;
    let options = options_for(config, &request);
    Ok(TrainJob {
        run_dir: training_dir(config).join(&id),
        id,
        request,
        name,
        paths,
        platform: Platform::current(),
        accelerator: accel,
        options,
        root: config.data_root(),
        timeout: Duration::from_secs(u64::from(t.timeout_hours) * 3600),
        workers: t.workers.map_or(job::Workers::Auto, job::Workers::Fixed),
        keep_work: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{HardwareInfo, apple_gpu};
    use crate::resources::catalog::TrainFamily;

    fn mac() -> Capability {
        capability::capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        )
    }

    /// A run record of `recipe` on `device` whose epochs peaked at `mem_gb`.
    fn record(
        created: &str,
        recipe: &TrainRecipe,
        device: &str,
        mem_gb: &[f64],
    ) -> service::RunRecord {
        let history: Vec<serde_json::Value> = mem_gb
            .iter()
            .enumerate()
            .map(|(i, g)| serde_json::json!({"epoch": i + 1, "elapsed_s": 1.0, "metrics": {"mem_gb": g}}))
            .collect();
        serde_json::from_value(serde_json::json!({
            "id": created, "name": "x", "created": created,
            "request": {"recipe": recipe, "datasets": []},
            "device": device, "state": "done", "stage": null, "step": 8, "steps": 8, "percent": 100,
            "detail": "", "progress": {"history": history}, "run_dir": "/tmp/x"
        }))
        .unwrap()
    }

    #[test]
    fn measured_peaks_replace_the_table_batch() {
        let mut cap = mac();
        let accel = cap.accelerator.clone().unwrap();
        let dev = accel.describe();
        let size = |c: &Capability| {
            c.family(TrainFamily::Rfdetr)
                .unwrap()
                .sizes
                .iter()
                .find(|s| s.size == "nano")
                .unwrap()
                .clone()
        };
        assert_eq!(size(&cap).suggested_batch, Some(4));
        assert!(!size(&cap).measured);
        let mut r = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
        r.batch = 4;
        let mut other = r.clone();
        other.imgsz = 448;
        let runs = vec![
            // Newest of the matching ones wins: 3 GB at batch 4 -> 16.
            record("2026-10-27T10:00:00Z", &r, &dev, &[2.0, 3.0]),
            record("2026-10-26T10:00:00Z", &r, &dev, &[7.0]),
            // Other input size, other GPU, no memory reported: ignored.
            record("2026-10-28T10:00:00Z", &other, &dev, &[0.5]),
            record("2026-10-28T11:00:00Z", &r, "NVIDIA X (cuda:0)", &[0.5]),
            record("2026-10-28T12:00:00Z", &r, &dev, &[]),
        ];
        let peaks = measured_peaks(&runs, accel.memory_mb());
        assert_eq!(peaks.len(), 4);
        assert_eq!(peaks[0].peak_mb, 7.0 * 1024.0);
        assert!(peaks[0].multi_scale);
        cap.apply_measured(&peaks);
        let s = size(&cap);
        assert!(s.measured);
        assert_eq!(s.suggested_batch, Some(16));
        // Without runs the table stays.
        let mut cap = mac();
        cap.apply_measured(&[]);
        assert!(!size(&cap).measured);
    }

    fn request(family: TrainFamily, size: &str) -> TrainRequest {
        TrainRequest {
            recipe: TrainRecipe {
                epochs: 1,
                patience: 0,
                ..TrainRecipe::new(family, size).unwrap()
            },
            datasets: vec!["exports/a".into()],
            include_public: false,
            scene: dataset::SceneFilter::NIGHT_IR,
            classes: None,
            rename: Default::default(),
            benchmark_sets: Vec::new(),
            allow_large: false,
            install_dir: None,
            gpu: None,
            excluded: Default::default(),
            balance: Default::default(),
            compare_group: None,
        }
    }

    /// A review export of `n` frames per camera / day (flat names: scene, camera, date), every
    /// fourth one empty, one raccoon per IR group.
    fn tiny_export(dir: &Path) {
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::create_dir_all(dir.join("labels")).unwrap();
        std::fs::write(dir.join("classes.txt"), "person\nraccoon\n").unwrap();
        for cam in ["a", "b", "c"] {
            for (scene, date, n) in [("day", "2026-10-01", 9), ("ir", "2026-10-02", 3)] {
                for i in 0..n {
                    let stem = format!("{scene}.{cam}.{date}.f{i}");
                    std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
                    let label = match (scene, i % 4) {
                        (_, 3) => "",
                        ("ir", 0) => "1 0.5 0.5 0.2 0.2\n",
                        _ => "0 0.5 0.5 0.2 0.4\n",
                    };
                    std::fs::write(dir.join("labels").join(format!("{stem}.txt")), label).unwrap();
                }
            }
        }
    }

    #[test]
    fn preview_plan_is_the_jobs_plan() {
        let root = std::env::temp_dir().join(format!("bop-preview-{}", uuid::Uuid::new_v4()));
        tiny_export(&root.join("exports").join("a"));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let mut req = request(TrainFamily::Yolo26, "n");
        req.scene = dataset::SceneFilter::ALL_SCENES;
        req.balance = dataset::Balance {
            axes: dataset::Balance::parse_axes("class,scene").unwrap(),
            cap_dominant: Some(3.0),
            ..dataset::Balance::default()
        };
        let preview = preview_plan(&c, &req).unwrap();
        assert!(preview.balance.is_some());
        // What the job's dataset stage plans: its sources and options.
        let j = plan_job(&c, req.clone(), &mac(), None).unwrap();
        assert_eq!(j.options.balance, req.balance);
        assert_eq!(j.options, options_for(&c, &req));
        let sources = job::read_sources(&j.request.datasets, None).unwrap();
        let job_plan =
            dataset::plan(&sources, &job::request_options(&j.options, &j.request)).unwrap();
        assert_eq!(preview, job_plan);
        // training.datasets when the request names none.
        let mut cfg = c.clone();
        cfg.training.datasets = vec!["exports/a".into()];
        let mut bare = req.clone();
        bare.datasets.clear();
        assert_eq!(preview_plan(&cfg, &bare).unwrap(), preview);
        // The public subset, not downloaded: left out and said first.
        let mut public = req.clone();
        public.include_public = true;
        let p = preview_plan(&c, &public).unwrap();
        assert!(p.warnings[0].contains("not downloaded"), "{:?}", p.warnings);
        assert_eq!(p.train, preview.train);
        // Nothing was written.
        assert!(!training_dir(&c).exists());
        // A bad balance or no dataset: refused.
        let mut bad = req.clone();
        bad.balance.max_repeat = 0;
        assert!(preview_plan(&c, &bad).is_err());
        assert!(plan_job(&c, bad, &mac(), None).is_err());
        let e = preview_plan(&c, &bare).unwrap_err().to_string();
        assert!(e.contains("no dataset"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }

    const GB: u64 = 1 << 30;

    fn commit(limit_gb: f64, free_gb: f64) -> Option<Commit> {
        Some(Commit {
            limit: (limit_gb * GB as f64) as u64,
            available: (free_gb * GB as f64) as u64,
        })
    }

    #[test]
    fn workers_follow_the_cores_and_on_windows_the_free_commit() {
        let rf = worker_loaders(catalog::Trainer::RfDetr);
        let w = |os, cpus, c, loaders| workers_for(os, cpus, c, loaders).workers;
        // This box: 6 cores, 2 GB of commit free -> none (load in the training process).
        assert_eq!(w("windows", 6, commit(21.3, 2.0), rf), 0);
        // Train + val loaders: (9.1 - 6) / (2 x 1.37 GiB) = 1; (12 - 6) / 2.73 = 2.
        assert_eq!(w("windows", 6, commit(32.0, 9.1), 2), 1);
        assert_eq!(w("windows", 6, commit(32.0, 10.0), 2), 1);
        assert_eq!(w("windows", 6, commit(32.0, 12.0), 2), 2);
        // Plenty free: the cores rule.
        assert_eq!(w("windows", 6, commit(64.0, 60.0), rf), 5);
        // Ultralytics' validation loader doubles the count: 3 processes per worker.
        assert_eq!(worker_loaders(catalog::Trainer::Ultralytics), 3);
        assert_eq!(w("windows", 16, commit(32.0, 13.0), 3), 1);
        assert_eq!(w("windows", 16, commit(32.0, 16.0), 3), 2);
        assert_eq!(worker_loaders(catalog::Trainer::Transformers), 2);
        // Exactly at the reserve, or below it: none.
        assert_eq!(w("windows", 6, commit(8.0, 6.0), rf), 0);
        assert_eq!(w("windows", 6, commit(8.0, 0.5), rf), 0);
        // No commit reading: the cores rule (all but one, 1..=8).
        assert_eq!(w("windows", 6, None, rf), 5);
        assert_eq!(w("windows", 1, None, rf), 1);
        assert_eq!(w("windows", 64, None, rf), MAX_AUTO_WORKERS);
        // Linux overcommits: the commit is not a limit there; macOS never spawns workers.
        assert_eq!(w("linux", 6, commit(8.0, 0.5), rf), 5);
        assert_eq!(w("macos", 8, None, rf), 0);
        assert_eq!(w("macos", 8, commit(8.0, 8.0), rf), 0);
        // The log line.
        let c = workers_for("windows", 6, commit(21.3, 5.1), rf);
        assert_eq!(
            c.describe(),
            "memory: commit 21.3 GB limit, 5.1 GB free -> 0 data loader workers (cores allow 5)"
        );
        let c = workers_for("windows", 6, commit(21.3, 9.1), rf);
        assert_eq!(c.workers, 1);
        assert!(
            c.describe().contains("-> 1 data loader worker (cores"),
            "{}",
            c.describe()
        );
        assert_eq!(
            workers_for("linux", 6, commit(1.0, 1.0), rf).describe(),
            "data loader workers: 5 (cores allow 5)"
        );
    }

    #[test]
    fn requests_without_balance_read_as_off() {
        let req = request(TrainFamily::Yolo26, "n");
        let mut v = serde_json::to_value(&req).unwrap();
        assert!(v.get("balance").is_none(), "off is not written: {v}");
        v.as_object_mut().unwrap().remove("balance");
        let old: TrainRequest = serde_json::from_value(v).unwrap();
        assert!(old.balance.is_off());
        assert_eq!(old, req);
        let mut on = req.clone();
        on.balance.axes = dataset::Balance::parse_axes("camera-day").unwrap();
        let text = serde_json::to_string(&on).unwrap();
        assert!(
            text.contains(r#""balance":{"axes":["camera-day"]"#),
            "{text}"
        );
        assert_eq!(serde_json::from_str::<TrainRequest>(&text).unwrap(), on);
    }

    #[test]
    fn plan_job_names_and_places_the_run() {
        let c = Config {
            download_dir: Some("/data".into()),
            ..Config::default()
        };
        let j = plan_job(&c, request(TrainFamily::Yolo26, "n"), &mac(), None).unwrap();
        assert!(j.name.starts_with("night-yolo26n-"), "{}", j.name);
        // The data root as the config resolves it ("/data" is "D:/data" on Windows).
        let root = c.data_root();
        assert_eq!(j.request.datasets, [root.join("exports").join("a")]);
        assert!(
            j.run_dir.starts_with(root.join("training")),
            "{}",
            j.run_dir.display()
        );
        assert_eq!(j.accelerator.device_arg(), "mps");
        assert_eq!(j.options.negative_share, c.training.negative_share);
        // Workers: sized when training starts, unless training.workers forces a count.
        assert_eq!(j.workers, job::Workers::Auto);
        let mut forced = c.clone();
        forced.training.workers = Some(0);
        let j0 = plan_job(&forced, request(TrainFamily::Yolo26, "n"), &mac(), None).unwrap();
        assert_eq!(j0.workers, job::Workers::Fixed(0));
        let named = plan_job(
            &c,
            request(TrainFamily::Rfdetr, "nano"),
            &mac(),
            Some("raccoon-v1".into()),
        )
        .unwrap();
        assert_eq!(named.name, "raccoon-v1");
    }

    #[test]
    fn plan_job_resolves_the_install_folder() {
        let c = Config {
            download_dir: Some("/data".into()),
            ..Config::default()
        };
        let root = c.data_root();
        let models = TrainPaths::for_config(&c).export.models;
        // Default: the models folder.
        let j = plan_job(&c, request(TrainFamily::Yolo26, "n"), &mac(), None).unwrap();
        assert_eq!(j.request.install_dir, None);
        assert_eq!(j.install_dir(), models.as_path());
        // Relative: under the data root; absolute as given.
        let mut r = request(TrainFamily::Yolo26, "n");
        r.install_dir = Some("custom/models".into());
        let j = plan_job(&c, r, &mac(), None).unwrap();
        assert_eq!(j.install_dir(), root.join("custom").join("models"));
        let abs = std::env::temp_dir().join("bop-install-elsewhere");
        let mut r = request(TrainFamily::Yolo26, "n");
        r.install_dir = Some(abs.clone());
        assert_eq!(plan_job(&c, r, &mac(), None).unwrap().install_dir(), abs);
        // The models folder itself or an empty path: the default.
        for d in [models.clone(), PathBuf::new()] {
            let mut r = request(TrainFamily::Yolo26, "n");
            r.install_dir = Some(d);
            assert_eq!(
                plan_job(&c, r, &mac(), None).unwrap().request.install_dir,
                None
            );
        }
        // An existing model of that name in the chosen folder is refused; a file is no folder.
        let dir = std::env::temp_dir().join(format!("bop-install-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("taken.onnx"), b"x").unwrap();
        let mut r = request(TrainFamily::Yolo26, "n");
        r.install_dir = Some(dir.clone());
        let e = plan_job(&c, r, &mac(), Some("taken".into())).unwrap_err();
        assert!(e.to_string().contains("taken.onnx exists"), "{e}");
        let mut r = request(TrainFamily::Yolo26, "n");
        r.install_dir = Some(dir.join("taken.onnx"));
        let e = plan_job(&c, r, &mac(), None).unwrap_err();
        assert!(e.to_string().contains("is not a folder"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_public_subset_needs_the_large_download_gate() {
        use crate::resources::export::Fetcher;
        let Some(lila) = crate::resources::catalog::lila_night() else {
            panic!("the public night subset is pinned");
        };
        assert!(
            lila.is_large(),
            "the subset is over the large-download limit"
        );
        let root = std::env::temp_dir().join(format!("bop-train-public-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let mut req = request(TrainFamily::Yolo26, "n");
        req.include_public = true;
        // Not downloaded, not allowed: refused with the way out.
        let e = plan_job(&c, req.clone(), &mac(), None).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("large download"), "{msg}");
        assert!(msg.contains("--allow-large"), "{msg}");
        assert!(msg.contains("allow_large_downloads"), "{msg}");
        assert!(msg.contains("train-data:lila-night"), "{msg}");
        // Without the subset: no gate.
        let mut plain = req.clone();
        plain.include_public = false;
        assert!(
            !plan_job(&c, plain, &mac(), None)
                .unwrap()
                .request
                .allow_large
        );
        // --allow-large / the page's confirmation.
        let mut allowed = req.clone();
        allowed.allow_large = true;
        assert!(
            plan_job(&c, allowed, &mac(), None)
                .unwrap()
                .request
                .allow_large
        );
        // allow_large_downloads is folded into the request (the job re-checks it).
        let open = Config {
            allow_large_downloads: true,
            ..c.clone()
        };
        let j = plan_job(&open, req.clone(), &mac(), None).unwrap();
        assert!(j.request.allow_large);
        // Downloaded already: nothing to gate.
        let fake = crate::resources::export::fake::Fake::new();
        fake.fetch(
            lila,
            &public_subset_dir(&root),
            &mut |_, _| {},
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert!(public_subset_installed(&root));
        assert!(!plan_job(&c, req, &mac(), None).unwrap().request.allow_large);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn plan_job_trains_on_the_chosen_gpu() {
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
        let cap = capability::capability(
            &HardwareInfo::new(
                "linux",
                "x86_64",
                vec![
                    gpu("RTX 3090", (8, 6), 24576, 0),
                    gpu("RTX 5060", (12, 0), 8192, 1),
                ],
            )
            .with_cuda_driver(13, 0),
            None,
        );
        let c = Config::default();
        let j = plan_job(&c, request(TrainFamily::Yolo26, "n"), &cap, None).unwrap();
        assert_eq!(j.accelerator.device_arg(), "cuda:0");
        assert_eq!(j.request.gpu, None);
        let mut r = request(TrainFamily::Yolo26, "n");
        r.gpu = Some(" CUDA:1".into());
        let j = plan_job(&c, r, &cap, None).unwrap();
        assert_eq!(j.accelerator.device_arg(), "cuda:1");
        assert_eq!(
            j.accelerator.lane(),
            crate::resources::export::train::TrainLane::Cu130
        );
        assert_eq!(j.request.gpu.as_deref(), Some("cuda:1"), "canonical");
        let mut r = request(TrainFamily::Yolo26, "n");
        r.gpu = Some("cuda:4".into());
        let e = plan_job(&c, r, &cap, None).unwrap_err().to_string();
        assert!(
            e.contains("no such CUDA device") && e.contains("RTX 5060"),
            "{e}"
        );
        let mut r = request(TrainFamily::Yolo26, "n");
        r.gpu = Some("cuda:0".into());
        let e = plan_job(&c, r, &mac(), None).unwrap_err().to_string();
        assert!(e.contains("no GPU to choose"), "{e}");
    }

    #[test]
    fn plan_job_refuses_what_cannot_run() {
        let c = Config::default();
        let e = plan_job(&c, request(TrainFamily::Dfine, "s"), &mac(), None).unwrap_err();
        assert!(e.to_string().contains("MPS"), "{e}");
        let none = capability::capability(&HardwareInfo::new("windows", "x86_64", vec![]), None);
        let e = plan_job(&c, request(TrainFamily::Yolo26, "n"), &none, None).unwrap_err();
        assert!(e.to_string().contains("not available"), "{e}");
        let mut empty = request(TrainFamily::Yolo26, "n");
        empty.datasets.clear();
        let e = plan_job(&c, empty, &mac(), None).unwrap_err();
        assert!(e.to_string().contains("no dataset"), "{e}");
        let e = plan_job(
            &c,
            request(TrainFamily::Yolo26, "n"),
            &mac(),
            Some("bad/name".into()),
        )
        .unwrap_err();
        assert!(e.to_string().contains("model name"), "{e}");
        let mut taken = Config::default();
        taken.seed_default_model();
        let e = plan_job(
            &taken,
            request(TrainFamily::Yolo26, "n"),
            &mac(),
            Some("IPcam-general".into()),
        )
        .unwrap_err();
        assert!(e.to_string().contains("already has a model"), "{e}");
    }
}
