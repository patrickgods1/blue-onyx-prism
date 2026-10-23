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
//! - [`bench`]: the benchmark at the end of a run.
//! - [`cli`]: `blue-onyx-prism train ...`. The Training page is `crate::server` (`/training`).
//!
//! Like the export, training is user-triggered only (never by `auto_download`), runs downloaded
//! code only from pinned, hashed sources, and keeps everything under `<data root>/tools` (the
//! environment) and the training folder (`training.dir`, the runs).

pub mod bench;
pub mod capability;
pub mod cli;
pub mod config;
pub mod dataset;
pub mod estimate;
pub mod job;
pub mod progress;
pub mod recipe;
pub mod service;

pub use capability::{Accelerator, Capability};
pub use config::TrainingConfig;
pub use job::{TrainJob, TrainRequest};
pub use recipe::{Augment, TrainRecipe};

use crate::config::Config;
use crate::resources::catalog::Platform;
use crate::resources::export::train::TrainPaths;
use anyhow::{Result, bail};
use std::path::PathBuf;
use std::time::Duration;

/// What this machine can train (detected hardware, cached).
pub fn capability_now() -> Capability {
    capability::capability(
        crate::backend::detect::hardware(),
        crate::system_info::total_memory_gb(),
    )
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

/// Data loader workers: none on macOS (spawned workers re-import the script; MPS runs were
/// smoke-tested without), else half the cores, at most 4.
fn workers() -> u32 {
    if cfg!(target_os = "macos") {
        return 0;
    }
    let cpus = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    u32::try_from((cpus / 2).clamp(1, 4)).unwrap_or(2)
}

/// Build the job of `request` on `config` and `cap` (see the checks below). `name`: the model
/// name, default `night-<model>-<stamp>` / `custom-<model>-<stamp>`.
pub fn plan_job(
    config: &Config,
    mut request: TrainRequest,
    cap: &Capability,
    name: Option<String>,
) -> Result<TrainJob> {
    let Some(accel) = cap.accelerator.clone() else {
        bail!("{}", cap.summary());
    };
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
    let night = request.scene == dataset::SceneFilter::NightIr;
    let name = name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| request.recipe.default_name(night, &stamp));
    recipe::valid_model_name(&name)?;
    let paths = TrainPaths::for_config(config);
    if paths.export.models.join(format!("{name}.onnx")).exists() {
        bail!(
            "{} exists; choose another model name",
            paths.export.models.join(format!("{name}.onnx")).display()
        );
    }
    if config
        .models
        .iter()
        .any(|m| m.effective_name().eq_ignore_ascii_case(&name))
    {
        bail!("the config already has a model named {name}; choose another name");
    }
    let id = service::new_run_id();
    let t = &config.training;
    Ok(TrainJob {
        run_dir: training_dir(config).join(&id),
        id,
        request,
        name,
        paths,
        platform: Platform::current(),
        accelerator: accel,
        options: dataset::Options {
            negative_share: t.negative_share,
            heldout_share: t.heldout_share,
            oversample: t.oversample,
            ..dataset::Options::default()
        },
        root: config.data_root(),
        timeout: Duration::from_secs(u64::from(t.timeout_hours) * 3600),
        workers: workers(),
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

    fn request(family: TrainFamily, size: &str) -> TrainRequest {
        TrainRequest {
            recipe: TrainRecipe::new(family, size, 1).unwrap(),
            datasets: vec!["exports/a".into()],
            include_public: false,
            scene: dataset::SceneFilter::NightIr,
            classes: None,
            rename: Default::default(),
            benchmark_sets: Vec::new(),
        }
    }

    #[test]
    fn plan_job_names_and_places_the_run() {
        let c = Config {
            download_dir: Some("/data".into()),
            ..Config::default()
        };
        let j = plan_job(&c, request(TrainFamily::Yolo26, "n"), &mac(), None).unwrap();
        assert!(j.name.starts_with("night-yolo26n-"), "{}", j.name);
        assert_eq!(j.request.datasets, [PathBuf::from("/data/exports/a")]);
        assert!(
            j.run_dir.starts_with("/data/training"),
            "{}",
            j.run_dir.display()
        );
        assert_eq!(j.accelerator.device_arg(), "mps");
        assert_eq!(j.options.negative_share, c.training.negative_share);
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
