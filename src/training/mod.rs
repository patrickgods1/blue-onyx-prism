//! In-app model training (docs/TRAINING.md): fine-tune YOLO26 / YOLO11 / YOLOv8 (Ultralytics,
//! AGPL-3.0), RF-DETR, RT-DETRv2 or D-FINE (Apache-2.0) on the frames reviewed on the Review tab,
//! export a static ONNX, install it next to the other models, add it to the config and benchmark
//! it on held-out frames it did not train on (whole camera days when there are several, a share of
//! the frames when there is one; other days of a held-out camera may train).
//!
//! - [`capability`]: can this machine train, on which GPU, which families and sizes fit (pure).
//! - [`recipe`]: [`TrainRecipe`] (family, size, base weights, input size, epochs, batch, learning
//!   rate, night augmentation) and its validation.
//! - [`estimate`]: rough VRAM / time per epoch table.
//! - [`dataset`]: review exports (+ the optional public night subset) -> the trainer's input
//!   (YOLO + COCO) and a held-out benchmark `dir:` set.
//! - [`job`]: one run's stage machine over the model export's seams (`CommandRunner`, `Fetcher`;
//!   the environment is `tools/train-env`, [`crate::resources::export::train`]).
//! - [`preflight`]: the report shown before a run, the refusals and the disk check.
//! - [`commit`]: the memory commit check before a job starts a Python model process (Windows).
//! - [`retry`]: a failed run's kept checkpoints: Retry export, disk use, deleting run files.
//! - [`service`]: the queue (one run at a time, cancellable, records in `runs.json`).
//! - [`overlaps`]: the overlap check, suggestions for nested and partly overlapping boxes.
//! - [`refine`]: the dataset check (`train refine`), short fold models that never saw a camera
//!   day suggest missing boxes, labels to verify and other classes.
//! - [`finish`]: the steps after an import chain (reject low-scoring auto boxes, accept the
//!   settled frames, switch the Review folder).
//! - [`bench`]: the benchmark at the end of a run, and the current default on the same frames.
//! - [`switch`]: making a trained model the default safely: the diff, copied settings, the
//!   suggested threshold, Revert.
//! - [`cli`]: `blue-onyx-prism train ...`. The Train tab is `crate::server` (`/training`).
//!
//! Like the export, training is user-triggered only (never by `auto_download`), runs downloaded
//! code only from pinned, hashed sources, and keeps everything under `<data root>/tools` (the
//! environment) and the training folder (`training.dir`, the runs).

pub mod autolabel;
pub mod bench;
pub mod capability;
pub mod cli;
pub mod commit;
pub mod compare;
pub mod config;
pub mod dataset;
pub mod estimate;
pub mod eta;
pub mod finish;
pub mod import;
pub mod job;
pub mod overlaps;
pub mod preflight;
pub mod progress;
pub mod pyenv;
pub mod recipe;
pub mod refine;
pub mod retry;
pub mod service;
pub mod switch;
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
    if let Some(a) = cap.accelerator.clone() {
        cap.apply_measured(&measured_peaks(runs, &a));
    }
    cap
}

/// The peak GPU memory of every run that reported `mem_gb` for a finished epoch, oldest first
/// (by `created`); `accel` resolves RF-DETR's `auto` multi-scale as the run did.
pub fn measured_peaks(
    runs: &[service::RunRecord],
    accel: &capability::Accelerator,
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
                    multi_scale: recipe.multi_scale_on(accel.memory_mb(), accel.target()),
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

/// Review exports found under `<training dir>/exports/` (the Review tab's default export
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
/// (`--allow-large`, the Train tab's download confirmation, or `allow_large_downloads`).
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
         limit) and is not downloaded yet under {}: rerun with --allow-large (Train area: tick \
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
/// ~1.4 GB per spawned worker during an RF-DETR nano run; re-measured 1.43 GB per worker process
/// on a GTX 1070 Ti / i5-8600K box, RF-DETR nano 384, batch 8).
pub const WORKER_COMMIT_BYTES: u64 = 1400 << 20;
/// Commit kept free for the training process itself (model, optimizer, CUDA context, pinned
/// memory) before any worker is counted: measured 8.0 GB private commit for the RF-DETR nano 384,
/// batch 8 trainer process on a GTX 1070 Ti / i5-8600K box (5.5 GB in an earlier, smaller run).
pub const MAIN_COMMIT_RESERVE_BYTES: u64 = 8 << 30;
/// RAM the training process itself keeps in its working set on Windows (measured ~2.6 GB for the
/// same RF-DETR nano 384, batch 8 trainer); the rest of its commit may sit in the paging file.
pub const MAIN_RAM_BYTES: u64 = (26 << 30) / 10;
/// Available RAM kept after the trainer and its workers: below that the measured box paged in
/// 2.8k pages/s with 2 workers (2.06 GB left) and 12k/s with 4 (0.87 GB left, then out of memory).
pub const RAM_RESERVE_BYTES: u64 = 2 << 30;
/// RAM one data loader worker process keeps in its working set on Windows (measured ~0.75 GB per
/// spawned worker, RF-DETR nano 384, batch 8).
pub const WORKER_RAM_BYTES: u64 = (3 << 30) / 4;

/// What [`workers_for`] reads of the host's memory (Windows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerMemory {
    /// The commit limit and what is free of it now.
    pub commit: Commit,
    /// Bytes the paging files can still grow by ([`crate::system_info::paging_growth`]; 0 for a
    /// fixed paging file, a full disk or a failed read).
    pub growable: u64,
    /// Physical RAM available now (`ullAvailPhys`); None when it was not read (no RAM bound).
    pub ram_available: Option<u64>,
}

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
    /// Bytes the paging files could still grow by then (0 without a commit reading).
    pub growable: u64,
    /// What the commit (free + growable) allows (Windows only).
    pub by_commit: Option<u32>,
    /// Physical RAM available then (Windows only).
    pub ram_available: Option<u64>,
    /// What the available RAM allows (Windows only).
    pub by_ram: Option<u32>,
}

impl WorkerChoice {
    /// The log line: "memory: commit 23.3 GB limit (can grow to 38.3 GB), 5.4 GB free, RAM 7.6 GB
    /// free -> 2 data loader workers (cores allow 5, commit 4, RAM 2)"; without growth no
    /// "(can grow ...)", without a RAM reading no RAM parts.
    pub fn describe(&self) -> String {
        let n = workers_text(self.workers);
        let Some(c) = self.commit else {
            return format!(
                "data loader workers: {} (cores allow {})",
                self.workers, self.by_cores
            );
        };
        let grow = if self.growable > 0 {
            format!(
                " (can grow to {:.1} GB)",
                gib(c.limit.saturating_add(self.growable))
            )
        } else {
            String::new()
        };
        let ram = self
            .ram_available
            .map(|b| format!(", RAM {:.1} GB free", gib(b)))
            .unwrap_or_default();
        let mut bounds = format!("cores allow {}", self.by_cores);
        if let Some(k) = self.by_commit {
            bounds.push_str(&format!(", commit {k}"));
        }
        if let Some(r) = self.by_ram {
            bounds.push_str(&format!(", RAM {r}"));
        }
        format!(
            "memory: commit {:.1} GB limit{grow}, {:.1} GB free{ram} -> {n} ({bounds})",
            gib(c.limit),
            gib(c.available),
        )
    }
}

/// The host memory commit (Windows) of one training run of `family` with `workers`, for the check
/// before its first Python process ([`commit`], in [`job`]): [`MAIN_COMMIT_RESERVE_BYTES`] (the
/// measured trainer process, 8.0 GB) plus, only when `training.workers` forces a count,
/// [`WORKER_COMMIT_BYTES`] per worker per loader ([`worker_loaders`]). Automatic workers add
/// nothing: [`workers_for`] gives a run only the workers the commit holds beyond the trainer's
/// reserve when it starts (0 when none). [`commit::COMMIT_SLACK_BYTES`] comes on top in the check.
pub fn trainer_commit_need(family: catalog::TrainFamily, workers: job::Workers) -> commit::Need {
    let trainer = MAIN_COMMIT_RESERVE_BYTES;
    let n = match workers {
        job::Workers::Auto => 0,
        job::Workers::Fixed(n) => n,
    };
    let per = WORKER_COMMIT_BYTES * u64::from(worker_loaders(family.trainer()));
    let extra = per.saturating_mul(u64::from(n));
    let mut detail = format!("the training process {:.1} GB", gib(trainer));
    if extra > 0 {
        detail.push_str(&format!(
            " + {} {:.1} GB (training.workers)",
            workers_text(n),
            gib(extra)
        ));
    }
    commit::Need {
        what: "this training run".into(),
        bytes: trainer.saturating_add(extra),
        detail,
        lighter: (extra > 0).then(|| {
            "let training choose its data loader workers (remove training.workers)".to_string()
        }),
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

/// Data loader workers on `os` with `cpus` logical cores, `memory` (Windows) and `loaders`
/// ([`worker_loaders`]): none on macOS (measured on an M1: 2 or 4 spawned workers did not speed
/// up RF-DETR nano, which is GPU-bound there, and Ultralytics uses none on MPS;
/// docs/TRAINING.md), else all cores but one, 1..=[`MAX_AUTO_WORKERS`].
///
/// On Windows every worker is a fresh (spawned) Python importing PyTorch, so there also the
/// smaller of
/// - commit: `(free commit + growable paging file - MAIN_COMMIT_RESERVE_BYTES) / (WORKER_COMMIT_BYTES
///   x loaders)`; a fixed paging file runs out of commit long before RAM (WinError 1455), a growable
///   one is grown by Windows on demand;
/// - RAM: `(available RAM - MAIN_RAM_BYTES - RAM_RESERVE_BYTES) / (WORKER_RAM_BYTES x loaders)`;
///   a commit the paging file can back still pages: 4 workers on a 16 GB box left 0.87 GB of RAM,
///   paged in 12k pages/s and ran out of memory, 2 were as fast;
///
/// each down to 0. Elsewhere neither: Linux overcommits and forks its workers (copy-on-write, so
/// the spawned workers' RAM figures do not apply), and macOS uses none.
pub fn workers_for(
    os: &str,
    cpus: usize,
    memory: Option<WorkerMemory>,
    loaders: u32,
) -> WorkerChoice {
    let by_cores = if os == "macos" {
        0
    } else {
        u32::try_from(cpus.saturating_sub(1))
            .unwrap_or(u32::MAX)
            .clamp(1, MAX_AUTO_WORKERS)
    };
    let memory = memory.filter(|_| os == "windows");
    let loaders = u64::from(loaders.max(1));
    let count = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
    let by_commit = memory.map(|m| {
        let free = m.commit.available.saturating_add(m.growable);
        count(free.saturating_sub(MAIN_COMMIT_RESERVE_BYTES) / (WORKER_COMMIT_BYTES * loaders))
    });
    let ram_available = memory.and_then(|m| m.ram_available);
    let by_ram = ram_available.map(|b| {
        let left = b.saturating_sub(MAIN_RAM_BYTES + RAM_RESERVE_BYTES);
        count(left / (WORKER_RAM_BYTES * loaders))
    });
    let workers = [Some(by_cores), by_commit, by_ram]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(by_cores);
    WorkerChoice {
        workers,
        by_cores,
        commit: memory.map(|m| m.commit),
        growable: memory.map_or(0, |m| m.growable),
        by_commit,
        ram_available,
        by_ram,
    }
}

/// [`workers_for`] on this machine now, for `trainer`: the commit, the available RAM and how far
/// the paging files can grow, read when training starts.
pub fn workers_now(trainer: catalog::Trainer) -> WorkerChoice {
    let cpus = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    let memory = crate::system_info::commit().map(|commit| {
        let physical = crate::system_info::physical();
        WorkerMemory {
            commit,
            growable: physical.map_or(0, |p| {
                crate::system_info::paging_growth(commit.limit, p.total)
            }),
            ram_available: physical.map(|p| p.available),
        }
    });
    workers_for(std::env::consts::OS, cpus, memory, worker_loaders(trainer))
}

/// The dataset options of `request` on `config`: the `training` section's assembly defaults
/// (shares, oversampling, static cap, ignore mode), `collect.size_floor` (the exports' floors are
/// compared with it), the camera days held out for good ([`frozen_heldout`]) and the request's
/// scene, classes, renames, excluded mode, balance and input size. The job ([`plan_job`]) and
/// [`preview_plan`] both build them here, so a preview is the run's plan. Fails when the list of
/// days held out for good does not read: a run could then train on one of them.
pub fn options_for(config: &Config, request: &TrainRequest) -> Result<dataset::Options> {
    Ok(options_with(config, request, frozen_heldout(config)?))
}

/// [`options_for`] with the camera days held out for good given.
pub(crate) fn options_with(
    config: &Config,
    request: &TrainRequest,
    frozen_heldout: std::collections::BTreeSet<String>,
) -> dataset::Options {
    let t = &config.training;
    let base = dataset::Options {
        negative_share: t.negative_share,
        heldout_share: t.heldout_share,
        oversample: t.oversample,
        static_cap: t.static_cap_option(),
        ignore: t.ignore,
        size_floor: config.collect.size_floor,
        frozen_heldout,
        ..dataset::Options::default()
    };
    job::request_options(&base, request)
}

/// The `camera/day` keys of the folder under review's camera days held out for good
/// (`collect.review_dir`, else the dataset folder;
/// [`crate::collect::labels::heldout`]): [`dataset::Options::frozen_heldout`]. Empty without the
/// file; an error when it is newer, damaged or does not read.
pub fn frozen_heldout(config: &Config) -> Result<std::collections::BTreeSet<String>> {
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let file = crate::collect::labels::heldout::load(&dir).map_err(|e| {
        anyhow::anyhow!(
            "{e:#}. A training run reads it to keep those camera days out of training, so none \
             starts until it reads"
        )
    })?;
    Ok(file.keys())
}

/// The camera days (`camera`, `date`) any training run trained or validated on: the frames of
/// every `<training dir>/<run>/data/train.txt` and `valid.txt` (names as the run's dataset writes
/// them, `s<N>-<scene>.<camera>.<date>.<stem>.jpg`; public images name no day). A run folder
/// without a list (a run that never wrote its dataset) adds nothing. None when the training folder
/// or a list does not read: a day may then have been trained on unseen
/// ([`crate::collect::labels::heldout::HeldoutPolicy::seen`]).
pub fn seen_days(config: &Config) -> Option<std::collections::BTreeSet<(String, String)>> {
    seen_days_in(&training_dir(config))
}

fn seen_days_in(dir: &Path) -> Option<std::collections::BTreeSet<(String, String)>> {
    let mut days = std::collections::BTreeSet::new();
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(days),
        Err(e) => {
            tracing::warn!("{}: {e}", dir.display());
            return None;
        }
    };
    for entry in rd {
        let data = match entry {
            Ok(e) => e.path().join("data"),
            Err(e) => {
                tracing::warn!("{}: {e}", dir.display());
                return None;
            }
        };
        if !data.is_dir() {
            continue;
        }
        for list in ["train.txt", "valid.txt"] {
            let path = data.join(list);
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    tracing::warn!("{}: {e}", path.display());
                    return None;
                }
            };
            for line in text.lines() {
                let Some(rel) = autolabel::trained_frame(line)
                    .and_then(|f| crate::collect::layout::parse_rel(&f))
                else {
                    continue;
                };
                days.insert((rel.camera, rel.date));
            }
        }
    }
    Some(days)
}

/// The plan a run of `request` would train on, without writing anything: the sources read like
/// the job reads them (relative dataset paths against the data root, `training.datasets` when
/// the request names none) and [`options_for`]. The public subset is read only when it is
/// installed; asked for but missing, the plan leaves it out and says so first in its warnings.
pub fn preview_plan(config: &Config, request: &TrainRequest) -> Result<dataset::Plan> {
    Ok(preview_parts(config, request)?.plan)
}

/// A preview: the sources read, the options and the plan.
#[derive(Debug)]
pub struct Preview {
    pub sources: Vec<dataset::Source>,
    pub options: dataset::Options,
    pub plan: dataset::Plan,
}

/// [`preview_plan`] with what it read, for [`preflight::report`] and [`preflight::run_needs`].
pub fn preview_parts(config: &Config, request: &TrainRequest) -> Result<Preview> {
    preview_with(config, request, options_for(config, request)?)
}

/// [`preview_parts`] with `options` of the caller's: `train prepare --dry-run` has no model, so no
/// input size ([`dataset::Options::imgsz`] None).
pub fn preview_with(
    config: &Config,
    request: &TrainRequest,
    options: dataset::Options,
) -> Result<Preview> {
    request.balance.validate()?;
    let datasets = if request.datasets.is_empty() {
        &config.training.datasets
    } else {
        &request.datasets
    };
    if datasets.is_empty() {
        bail!(
            "no dataset: choose a review export folder or set training.datasets (export reviewed \
             frames from the Dataset tab in YOLO format first)"
        );
    }
    let datasets: Vec<PathBuf> = datasets.iter().map(|d| config.data_path(d)).collect();
    let root = config.data_root();
    let public_dir = public_subset_dir(&root);
    let public = request.include_public && public_subset_installed(&root);
    let sources = job::read_sources(&datasets, public.then_some(public_dir.as_path()))?;
    let mut plan = preflight::planned(&sources, &options)?;
    if request.include_public && !public {
        plan.warnings.insert(
            0,
            "the public night subset is not downloaded yet: this preview leaves it out (the run \
             downloads and mixes it in, repeating your frames training.oversample times)"
                .into(),
        );
    }
    Ok(Preview {
        sources,
        options,
        plan,
    })
}

/// Frames of the folder under review that are not reviewed yet (`collect.review_dir`, or the
/// dataset folder), for the pre-flight report. Reads every label file; None when the folder is not
/// there.
pub fn unreviewed_frames(config: &Config) -> Option<u64> {
    use crate::collect::labels::queue;
    let dir = config.collect.resolved_review_dir(&config.data_root());
    dir.is_dir().then(|| {
        let f = queue::Filters {
            source: queue::QueueSource::All,
            ..queue::Filters::default()
        };
        queue::build(&dir, &f).counts.remaining
    })
}

/// Disk the training environment and base weights still need under the tools folder, for
/// `recipes` (one download per weights) on `lane` (None = no GPU, nothing can run).
pub fn tools_needs(
    config: &Config,
    recipes: &[TrainRecipe],
    lane: Option<crate::resources::export::train::TrainLane>,
) -> Vec<preflight::VolumeNeed> {
    let paths = TrainPaths::for_config(config);
    let mut bytes = lane
        .filter(|l| {
            !crate::resources::export::train::train_env_ready(&paths, Platform::current(), *l)
        })
        .map_or(0, |l| l.estimate().1);
    let mut seen: Vec<&str> = Vec::new();
    for base in recipes.iter().filter_map(|r| r.base().ok()) {
        if !seen.contains(&base.id)
            && !crate::resources::manager::is_installed(base, &paths.weights_dir(base))
        {
            seen.push(base.id);
            bytes = bytes.saturating_add(base.size());
        }
    }
    vec![preflight::VolumeNeed {
        dir: paths.export.tools,
        bytes,
        advice: preflight::TOOLS_ADVICE,
    }]
}

/// Refuse `jobs` before they queue: data no run can use ([`preflight::check_plan`], naming the run
/// when there are several) and the disk they need together. The run checks again when it starts,
/// the disk may have filled up by then.
pub fn check_jobs(config: &Config, jobs: &[TrainJob]) -> Result<()> {
    let mut needs = Vec::new();
    for j in jobs {
        // Not downloaded yet: the plan without it may look unusable, and the run downloads it.
        if j.request.include_public && !public_subset_installed(&config.data_root()) {
            continue;
        }
        let p = preview_parts(config, &j.request)?;
        preflight::check_plan(&p.plan, &p.sources, &p.options).map_err(|e| {
            if jobs.len() > 1 {
                e.context(format!("run {}", j.name))
            } else {
                e
            }
        })?;
        needs.extend(preflight::run_needs(
            &p.plan,
            &p.sources,
            &j.request.recipe,
            &j.run_dir,
            j.install_dir(),
        )?);
    }
    let recipes: Vec<TrainRecipe> = jobs.iter().map(|j| j.request.recipe.clone()).collect();
    needs.extend(tools_needs(
        config,
        &recipes,
        jobs.first().map(|j| j.accelerator.lane()),
    ));
    preflight::check_disk(needs)
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
             reviewed frames from the Dataset tab in YOLO format first)"
        );
    }
    request.datasets = request
        .datasets
        .iter()
        .map(|d| config.data_path(d))
        .collect();
    let stamp = crate::clock::zone()
        .wall(std::time::SystemTime::now())
        .format("%Y%m%d-%H%M")
        .to_string();
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
    let options = options_for(config, &request)?;
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
        export_only: None,
        headroom: commit::now,
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
        let peaks = measured_peaks(&runs, &accel);
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
            benchmark_devices: Vec::new(),
            allow_large: false,
            install_dir: None,
            gpu: None,
            excluded: Default::default(),
            balance: Default::default(),
            compare_group: None,
            train_seed: None,
            ignore_memory_check: false,
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
        assert_eq!(j.options, options_for(&c, &req).unwrap());
        let sources = job::read_sources(&j.request.datasets, None).unwrap();
        let job_plan =
            preflight::planned(&sources, &job::request_options(&j.options, &j.request)).unwrap();
        assert_eq!(preview, job_plan);
        // A repeat with another training seed: the same frames.
        let mut seeded = req.clone();
        seeded.train_seed = Some(2);
        assert_eq!(
            options_for(&c, &seeded).unwrap(),
            options_for(&c, &req).unwrap()
        );
        assert_eq!(preview_plan(&c, &seeded).unwrap(), preview);
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

    fn gb(v: f64) -> u64 {
        (v * GB as f64) as u64
    }

    /// A commit reading with `growable_gb` of paging file growth and `ram_gb` of RAM available.
    fn mem(
        limit_gb: f64,
        free_gb: f64,
        growable_gb: f64,
        ram_gb: Option<f64>,
    ) -> Option<WorkerMemory> {
        Some(WorkerMemory {
            commit: Commit {
                limit: gb(limit_gb),
                available: gb(free_gb),
            },
            growable: gb(growable_gb),
            ram_available: ram_gb.map(gb),
        })
    }

    /// A fixed paging file, RAM not read: the commit bound alone.
    fn commit(limit_gb: f64, free_gb: f64) -> Option<WorkerMemory> {
        mem(limit_gb, free_gb, 0.0, None)
    }

    /// Fixed paging files. The reserve went from 6 to the measured 8 GB, so every free commit
    /// below moved up by 2 GB to keep its result (10.0 now gives 0 where it gave 1).
    #[test]
    fn workers_follow_the_cores_and_on_windows_the_free_commit() {
        let rf = worker_loaders(catalog::Trainer::RfDetr);
        let w = |os, cpus, c, loaders| workers_for(os, cpus, c, loaders).workers;
        // This box: 6 cores, 2 GB of commit free -> none (load in the training process).
        assert_eq!(w("windows", 6, commit(21.3, 2.0), rf), 0);
        // Train + val loaders: (10.8 - 8) / (2 x 1.37 GiB) = 1; (13.5 - 8) / 2.73 = 2.
        assert_eq!(w("windows", 6, commit(32.0, 10.0), 2), 0);
        assert_eq!(w("windows", 6, commit(32.0, 10.8), 2), 1);
        assert_eq!(w("windows", 6, commit(32.0, 13.4), 2), 1);
        assert_eq!(w("windows", 6, commit(32.0, 13.5), 2), 2);
        // Plenty free: the cores rule.
        assert_eq!(w("windows", 6, commit(64.0, 60.0), rf), 5);
        // Ultralytics' validation loader doubles the count: 3 processes per worker.
        assert_eq!(worker_loaders(catalog::Trainer::Ultralytics), 3);
        assert_eq!(w("windows", 16, commit(32.0, 13.0), 3), 1);
        assert_eq!(w("windows", 16, commit(32.0, 16.0), 3), 1);
        assert_eq!(w("windows", 16, commit(32.0, 16.5), 3), 2);
        assert_eq!(worker_loaders(catalog::Trainer::Transformers), 2);
        // Exactly at the reserve, or below it: none.
        assert_eq!(w("windows", 6, commit(10.0, 8.0), rf), 0);
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
            "memory: commit 21.3 GB limit, 5.1 GB free -> 0 data loader workers (cores allow 5, \
             commit 0)"
        );
        let c = workers_for("windows", 6, commit(21.3, 10.8), rf);
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

    /// The measured box (GTX 1070 Ti, i5-8600K: 6 cores, 15.9 GB RAM, `c:\pagefile.sys 2935
    /// 24576`, 19 GB free on C:): commit 23.3 GB limit, 5.4 GB free, 7.6 GB RAM available before a
    /// run. 2 workers trained 32% faster than none; 4 were no faster and ran out of memory. The
    /// main case gives C: 31 GB free, so the paging file's growth is not capped below 15 GB.
    #[test]
    fn a_growable_paging_file_allows_workers_and_the_ram_caps_them() {
        use crate::backend::memory::{growable_bytes, parse_paging_files, volume_root};
        let rf = worker_loaders(catalog::Trainer::RfDetr);
        let ram = gb(15.9);
        let files = parse_paging_files(&[r"c:\pagefile.sys 2935 24576".to_string()]);
        let grow = |limit: u64, disk_free: u64| {
            growable_bytes(
                &files,
                limit,
                ram,
                |f| volume_root(f, &[], Some("C:")),
                |root| (root == r"C:\").then_some(disk_free),
            )
        };
        // 24 GB max less 7.4 GB now = 16.6, but 31 GB free on C: keeps 16: 15.
        let growable = grow(gb(23.3), gb(31.0));
        assert_eq!(growable, 15 * GB);
        // With the 19 GB C: had when measured, the 16 GB kept leaves 3 GB of growth: no worker
        // (commit (5.4 + 3 - 8) / 2.8 < 1), which keeps the recordings' drive from filling.
        let tight = WorkerMemory {
            commit: Commit {
                limit: gb(23.3),
                available: gb(5.4),
            },
            growable: grow(gb(23.3), gb(19.0)),
            ram_available: Some(gb(7.6)),
        };
        assert_eq!(tight.growable, 3 * GB);
        assert_eq!(workers_for("windows", 6, Some(tight), rf).workers, 0);
        let measured = WorkerMemory {
            commit: Commit {
                limit: gb(23.3),
                available: gb(5.4),
            },
            growable,
            ram_available: Some(gb(7.6)),
        };
        let c = workers_for("windows", 6, Some(measured), rf);
        // Commit: (5.4 + 15 - 8) / 2.73 = 4; RAM: (7.6 - 2.6 - 2) / (2 x 0.75) = 2, exactly.
        assert_eq!(
            (c.workers, c.by_cores, c.by_commit, c.by_ram),
            (2, 5, Some(4), Some(2))
        );
        assert_eq!(
            c.describe(),
            "memory: commit 23.3 GB limit (can grow to 38.3 GB), 5.4 GB free, RAM 7.6 GB free -> \
             2 data loader workers (cores allow 5, commit 4, RAM 2)"
        );
        // A fixed file (the free commit only, as before) gives this box none.
        let fixed = mem(23.3, 5.4, 0.0, Some(7.6));
        assert_eq!(workers_for("windows", 6, fixed, rf).workers, 0);
        // Without the RAM reading the growable file alone allows the 4 that ran out of memory.
        let no_ram = mem(23.3, 5.4, 15.0, None);
        assert_eq!(workers_for("windows", 6, no_ram, rf).workers, 4);
        // The RAM bound: 4 workers need 10.6 GB available, 1 needs 6.1.
        let by_ram = |ram_gb| workers_for("windows", 6, mem(64.0, 60.0, 0.0, Some(ram_gb)), rf);
        assert_eq!(by_ram(7.5).workers, 1);
        assert_eq!(by_ram(9.0).workers, 2);
        assert_eq!(by_ram(10.7).workers, 4);
        assert_eq!(by_ram(6.0).workers, 0);
        assert_eq!(by_ram(4.0).workers, 0);
        assert_eq!(by_ram(1.0).workers, 0);
        assert_eq!(by_ram(64.0).workers, 5, "the cores still cap");
        // Ultralytics: 3 worker processes per worker.
        let ul = worker_loaders(catalog::Trainer::Ultralytics);
        assert_eq!(
            workers_for("windows", 6, mem(64.0, 60.0, 0.0, Some(9.2)), ul).workers,
            2
        );
        // A full disk: no growth, so the free commit alone (none here).
        assert_eq!(grow(gb(23.3), gb(3.0)), 0);
        let full = Some(WorkerMemory {
            growable: grow(gb(23.3), gb(3.0)),
            ..measured
        });
        assert_eq!(workers_for("windows", 6, full, rf).workers, 0);
        // The file at its maximum: no growth either.
        assert_eq!(grow(gb(15.9 + 25.0), gb(100.0)), 0);
        // Elsewhere neither bound applies (Linux forks, macOS uses none).
        assert_eq!(workers_for("linux", 6, Some(measured), rf).workers, 5);
        assert_eq!(workers_for("linux", 6, Some(measured), rf).by_ram, None);
        assert_eq!(workers_for("macos", 6, Some(measured), rf).workers, 0);
    }

    /// This machine's reading (shown with `--nocapture`): every bound present on Windows only,
    /// and the choice is their minimum.
    #[test]
    fn workers_now_reads_every_bound_on_windows() {
        let c = workers_now(catalog::Trainer::RfDetr);
        println!("{}", c.describe());
        let windows = cfg!(windows);
        assert_eq!(c.commit.is_some(), windows, "{c:?}");
        assert_eq!(c.by_commit.is_some(), windows, "{c:?}");
        assert_eq!(c.by_ram.is_some(), windows, "{c:?}");
        let min = [Some(c.by_cores), c.by_commit, c.by_ram]
            .into_iter()
            .flatten()
            .min();
        assert_eq!(Some(c.workers), min, "{c:?}");
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

    #[test]
    fn seen_days_are_every_runs_train_and_valid_days() {
        let root = std::env::temp_dir().join(format!("bop-seen-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let dir = training_dir(&c);
        // Nothing trained yet: no day seen.
        assert_eq!(seen_days(&c), Some(std::collections::BTreeSet::new()));
        let write = |run: &str, list: &str, text: &str| {
            let data = dir.join(run).join("data");
            std::fs::create_dir_all(&data).unwrap();
            std::fs::write(data.join(list), text).unwrap();
        };
        write(
            "20261101-1",
            "train.txt",
            "./images/s0-day.back.2026-10-25.20261025T010203004Z_ab.jpg\n\
             ./images/s0-day.back.2026-10-25.20261025T010203004Z_ab.jpg\n\
             ./images/s1-night.drive.2026-10-26.x.jpg\n\
             ./images/s2-lila_0001.jpg\n\n",
        );
        write(
            "20261101-1",
            "valid.txt",
            "./images/s0-ir.nook.2026-10-27.y.jpg\n",
        );
        write(
            "20261102-1",
            "train.txt",
            "./images/s3-day.alley.2026-11-01.z.jpg\n",
        );
        // Not runs: the exports folder, a run that never wrote its lists, a file.
        std::fs::create_dir_all(dir.join("exports").join("x")).unwrap();
        std::fs::create_dir_all(dir.join("20261103-1").join("data")).unwrap();
        std::fs::write(dir.join("runs.json"), "[]").unwrap();
        let day = |c: &str, d: &str| (c.to_string(), d.to_string());
        assert_eq!(
            seen_days(&c),
            Some(
                [
                    day("back", "2026-10-25"),
                    day("drive", "2026-10-26"),
                    day("nook", "2026-10-27"),
                    day("alley", "2026-11-01"),
                ]
                .into()
            )
        );
        // A list that does not read: nothing can be called unseen.
        let bad = dir.join("20261104-1").join("data");
        std::fs::create_dir_all(bad.join("train.txt")).unwrap();
        assert_eq!(seen_days(&c), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runs_options_hold_the_frozen_days_and_a_damaged_list_stops_them() {
        let root = std::env::temp_dir().join(format!("bop-frozen-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let req = request(TrainFamily::Yolo26, "n");
        assert!(options_for(&c, &req).unwrap().frozen_heldout.is_empty());
        let review = c.collect.resolved_review_dir(&c.data_root());
        std::fs::create_dir_all(&review).unwrap();
        let file = review.join(crate::collect::labels::heldout::FILE);
        std::fs::write(
            &file,
            r#"{"version":1,"days":[{"camera":"nook","date":"2026-10-27","added":"2026-11-18T00:00:00.000Z","reason":"camera-minimum"}]}"#,
        )
        .unwrap();
        let o = options_for(&c, &req).unwrap();
        assert_eq!(o.frozen_heldout, ["nook/2026-10-27".to_string()].into());
        // The service's job and the preview share them.
        assert_eq!(plan_job(&c, req.clone(), &mac(), None).unwrap().options, o);
        std::fs::write(&file, "{").unwrap();
        let e = options_for(&c, &req).unwrap_err().to_string();
        assert!(e.contains("damaged") && e.contains("none starts"), "{e}");
        assert!(plan_job(&c, req.clone(), &mac(), None).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"{");
        let _ = std::fs::remove_dir_all(&root);
    }
}
