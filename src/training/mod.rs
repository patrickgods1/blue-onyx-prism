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

pub mod autolabel;
pub mod bench;
pub mod capability;
pub mod cli;
pub mod config;
pub mod dataset;
pub mod estimate;
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

use crate::config::Config;
use crate::resources::catalog::Platform;
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
    use crate::resources::catalog;
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
    let night = request.scene == dataset::SceneFilter::NightIr;
    let name = name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| request.recipe.default_name(night, &stamp));
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
            allow_large: false,
            install_dir: None,
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
