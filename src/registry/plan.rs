//! Planning the workers of a registry generation: for each enabled model, where it is headed
//! and whether it can be spawned at all. Pure decisions over the config, the provisioning plan
//! and what the runtimes offer; no threads are started here ([`super::Spawner::spawn_model`]
//! turns a [`ModelPlan`] into a worker).

use super::{load_request, try_lock_runtimes, validate_names};
use crate::backend::spec::Device;
use crate::backend::{DeviceSpec, LoadRequest, Runtimes};
use crate::config::{Config, ModelConfig};
use crate::model::classes::resolve_class_names;
use crate::resources::provision::{Provision, Wait};
use crate::sync::LockExt;
use crate::worker::LiveSettings;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::info;

/// What the planner asks the runtimes. [`Runtimes`] answers it directly; [`LockedRuntimes`]
/// answers through the generation's mutex (reloads); tests use a fake.
pub trait PlanRuntimes {
    /// The first runnable load candidate of `spec` for `model` and its option label; None when
    /// nothing can run, or when the runtimes are busy compiling and the answer is not worth
    /// waiting for.
    fn first_runnable(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)>;

    /// As [`Self::first_runnable`], but waits for busy runtimes.
    fn first_runnable_wait(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)> {
        self.first_runnable(spec, model)
    }
}

impl PlanRuntimes for Runtimes {
    fn first_runnable(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)> {
        Runtimes::first_runnable(self, spec, model)
    }
}

/// The generation's shared runtimes, for planning a reload while other models may compile:
/// the label query only tries the lock, the interim-device query waits for it.
pub(super) struct LockedRuntimes<'a>(pub &'a Mutex<Runtimes>);

impl PlanRuntimes for LockedRuntimes<'_> {
    fn first_runnable(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)> {
        try_lock_runtimes(self.0)?.first_runnable(spec, model)
    }

    fn first_runnable_wait(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)> {
        let rt = self.0.lock_recover();
        rt.first_runnable(spec, model)
    }
}

/// Whether a planned model can be spawned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanOutcome {
    /// Spawn a worker (it may still wait for downloads: see [`ModelPlan::wait`]).
    Spawn,
    /// No worker can be set up: the message becomes the model's `Failed` state (missing file,
    /// pending export, bad classes, downloads that must be fetched by hand).
    Failed(String),
}

/// Everything [`super::Spawner::spawn_model`] needs to start the worker of one model.
pub struct ModelPlan {
    pub name: String,
    /// Absolute model path.
    pub path: PathBuf,
    /// Where to run (for a failed plan: the best guess, unused).
    pub device: DeviceSpec,
    pub load: LoadRequest,
    /// Empty while `wait` is set (read from disk once the files are there).
    pub classes: Vec<String>,
    pub live: LiveSettings,
    pub lazy: bool,
    /// Downloads the worker waits for before loading; None when the model can load now.
    pub wait: Option<Wait>,
    /// The config entry with its paths resolved.
    pub model: ModelConfig,
    /// Where the model is headed (its first runnable load candidate) and that option's label.
    pub planned: Option<(Device, String)>,
    pub outcome: PlanOutcome,
}

/// Plans for every enabled model of `config`, in config order. Errors for duplicate names.
pub fn plan_models<R: PlanRuntimes + ?Sized>(
    config: &Config,
    runtimes: &R,
    provision: Option<&Provision>,
    best_cpu: DeviceSpec,
) -> Result<Vec<ModelPlan>> {
    validate_names(config)?;
    Ok(config
        .enabled_models()
        .map(|m| plan_model(config, m, runtimes, provision, best_cpu))
        .collect())
}

/// The plan of one model (`m` is an entry of `config`).
pub fn plan_model<R: PlanRuntimes + ?Sized>(
    config: &Config,
    m: &ModelConfig,
    runtimes: &R,
    provision: Option<&Provision>,
    best_cpu: DeviceSpec,
) -> ModelPlan {
    let name = m.effective_name();
    let path = config.data_path(&m.path);
    let load = load_request(config, m, path.clone());
    let model = ModelConfig {
        path: path.clone(),
        classes: m.classes.as_deref().map(|c| config.data_path(c)),
        ..m.clone()
    };

    // Where it is headed, shown as "<label> (loading)".
    let planned = if config.force_cpu {
        Some(best_cpu)
    } else {
        config.device_spec_for(m).ok()
    }
    .and_then(|device| runtimes.first_runnable(&device, &path));

    let mut plan = ModelPlan {
        name: name.clone(),
        path: path.clone(),
        device: best_cpu,
        load,
        classes: Vec::new(),
        live: LiveSettings::for_model(config, m),
        lazy: m.lazy,
        wait: None,
        model,
        planned,
        outcome: PlanOutcome::Spawn,
    };

    // Downloads this model waits for (phase 7.3).
    if let Some(p) = provision
        && let Some(msg) = p.manual_failure(&name)
    {
        plan.outcome = PlanOutcome::Failed(msg);
        return plan;
    }
    let mut wait = provision.and_then(|p| p.wait_for(&name));
    if let Some(w) = &wait {
        let only_runtimes = w.blocks().iter().all(|b| b.is_runtime());
        if only_runtimes && path.is_file() {
            let device = config.device_spec_for(m).unwrap_or(best_cpu);
            if let Some((interim, _)) = runtimes.first_runnable_wait(&device, &path) {
                let ids: Vec<&str> = w.blocks().iter().map(|b| b.resource.id).collect();
                info!(
                    model = %name,
                    "loading on {} while {} downloads; switching after it is installed",
                    interim,
                    ids.join(", ")
                );
                wait = None;
            }
        }
    }

    let setup = (|| -> Result<(DeviceSpec, Vec<String>)> {
        // `force_cpu`: the best CPU option (OpenVINO, else ONNX Runtime).
        let device = if config.force_cpu {
            best_cpu
        } else {
            config.device_spec_for(m)?
        };
        if wait.is_some() {
            // Class names are read after the files arrive (in the worker).
            return Ok((device, Vec::new()));
        }
        // An exported model's file is made by a user-started export, never downloaded;
        // a file from another version (the old RF-DETR large download) is not used.
        if let Some(r) = crate::resources::export::pending_export(&path) {
            bail!(
                "{}",
                crate::resources::export::needs_export_message(r, &m.path, path.is_file())
            );
        }
        if !path.is_file() {
            bail!(
                "model file not found: {} (download it with `download-models` or fix `path`)",
                path.display()
            );
        }
        let classes_path = m.classes.as_deref().map(|c| config.data_path(c));
        Ok((device, resolve_class_names(&path, classes_path.as_deref())?))
    })();
    match setup {
        Ok((device, classes)) => {
            plan.device = device;
            plan.classes = classes;
            plan.wait = wait;
        }
        Err(e) => plan.outcome = PlanOutcome::Failed(format!("{e:#}")),
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::{OpenVinoProbe, OrtProbe, OvDevice, RuntimeProbe, select};
    use crate::backend::{HardwareInfo, Selection, plan_for};

    /// A machine with OpenVINO CPU only, planned the way [`Runtimes`] does it.
    struct Fake {
        sel: Selection,
        ov_devices: Vec<String>,
    }

    impl Fake {
        fn cpu_only() -> Self {
            let probe = RuntimeProbe {
                openvino: OpenVinoProbe {
                    error: None,
                    devices: vec![OvDevice {
                        name: "CPU".into(),
                        full_name: "Test CPU".into(),
                        device_type: None,
                    }],
                },
                ort: OrtProbe::not_in_build(),
            };
            let hw = HardwareInfo::new("linux", "x86_64", vec![]);
            Self {
                sel: select(&hw, &probe, true),
                ov_devices: vec!["CPU".into()],
            }
        }
    }

    impl PlanRuntimes for Fake {
        fn first_runnable(&self, spec: &DeviceSpec, _model: &Path) -> Option<(Device, String)> {
            plan_for(spec, &self.sel, &self.ov_devices)
                .into_iter()
                .find_map(|c| {
                    self.sel
                        .option(&c.device)
                        .filter(|o| o.runnable)
                        .map(|o| (c.device, o.label.clone()))
                })
        }
    }

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_plan_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn config_with(models: Vec<ModelConfig>) -> Config {
        Config {
            models,
            ..Default::default()
        }
    }

    fn entry(path: PathBuf) -> ModelConfig {
        ModelConfig {
            path,
            ..Default::default()
        }
    }

    fn plan_one(config: &Config) -> ModelPlan {
        let best = DeviceSpec::OPENVINO_CPU;
        let mut plans = plan_models(config, &Fake::cpu_only(), None, best).unwrap();
        assert_eq!(plans.len(), 1);
        plans.remove(0)
    }

    #[test]
    fn present_model_is_spawned_with_its_device_and_classes() {
        let dir = temp_dir();
        let file = dir.join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        let plan = plan_one(&config_with(vec![entry(file.clone())]));
        assert_eq!(plan.outcome, PlanOutcome::Spawn);
        assert_eq!(plan.name, "m");
        assert_eq!(plan.path, file);
        assert_eq!(plan.device, DeviceSpec::Auto);
        assert_eq!(plan.classes, crate::model::classes::coco80());
        assert!(plan.wait.is_none());
        let (device, label) = plan.planned.unwrap();
        assert_eq!(device.to_string(), "openvino:cpu");
        assert!(label.contains("OpenVINO"), "{label}");
    }

    #[test]
    fn force_cpu_uses_the_best_cpu_device() {
        let dir = temp_dir();
        let file = dir.join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        let mut config = config_with(vec![entry(file)]);
        config.force_cpu = true;
        let plan = plan_one(&config);
        assert_eq!(plan.outcome, PlanOutcome::Spawn);
        assert_eq!(plan.device, DeviceSpec::OPENVINO_CPU);
    }

    #[test]
    fn missing_file_fails_with_the_path() {
        let file = temp_dir().join("absent.onnx");
        let plan = plan_one(&config_with(vec![entry(file.clone())]));
        let PlanOutcome::Failed(msg) = &plan.outcome else {
            panic!("expected Failed");
        };
        assert!(msg.contains("model file not found"), "{msg}");
        assert!(msg.contains("absent.onnx"), "{msg}");
        assert!(plan.classes.is_empty());
    }

    #[test]
    fn exportable_model_without_a_file_needs_an_export() {
        let file = temp_dir().join("yolo26s.onnx");
        let plan = plan_one(&config_with(vec![entry(file)]));
        let PlanOutcome::Failed(msg) = &plan.outcome else {
            panic!("expected Failed");
        };
        assert!(!msg.contains("model file not found"), "{msg}");
    }

    #[test]
    fn unreadable_classes_fail() {
        let dir = temp_dir();
        let file = dir.join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        let mut m = entry(file);
        m.classes = Some(dir.join("missing.yaml"));
        let plan = plan_one(&config_with(vec![m]));
        let PlanOutcome::Failed(msg) = &plan.outcome else {
            panic!("expected Failed");
        };
        assert!(msg.contains("loading class names"), "{msg}");
    }

    #[test]
    fn invalid_device_spec_fails() {
        let dir = temp_dir();
        let file = dir.join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        let mut m = entry(file);
        m.device = Some("bogus:thing".into());
        let plan = plan_one(&config_with(vec![m]));
        let PlanOutcome::Failed(msg) = &plan.outcome else {
            panic!("expected Failed");
        };
        assert!(msg.contains("device"), "{msg}");
        assert!(plan.planned.is_none());
    }

    #[test]
    fn disabled_models_are_not_planned_and_duplicates_are_rejected() {
        let dir = temp_dir();
        let a = dir.join("a.onnx");
        let b = dir.join("b.onnx");
        std::fs::write(&a, b"x").unwrap();
        let mut off = entry(b);
        off.enabled = false;
        let config = config_with(vec![entry(a), off]);
        let plans =
            plan_models(&config, &Fake::cpu_only(), None, DeviceSpec::OPENVINO_CPU).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].name, "a");

        let dup = config_with(vec![entry(dir.join("a.onnx")), entry(dir.join("a.onnx"))]);
        assert!(plan_models(&dup, &Fake::cpu_only(), None, DeviceSpec::OPENVINO_CPU).is_err());
    }

    #[test]
    fn provision_without_needs_changes_nothing() {
        let dir = temp_dir();
        let file = dir.join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        let config = config_with(vec![entry(file)]);
        let provision = Provision::none(&config);
        let plans = plan_models(
            &config,
            &Fake::cpu_only(),
            Some(&provision),
            DeviceSpec::OPENVINO_CPU,
        )
        .unwrap();
        assert_eq!(plans[0].outcome, PlanOutcome::Spawn);
        assert!(plans[0].wait.is_none());
    }
}
