//! Loads every enabled model through the shared [`Runtimes`] (one OpenVINO `Core`, one ONNX
//! Runtime library) and owns their worker threads.
//! Disabled models (`enabled: false`) stay in the config but get no worker and are not served.
//!
//! Workers are spawned in config order; each waits for its predecessor's [`LoadGate`] before
//! locking the core, so compile order is deterministic and only one model compiles at a time.
//!
//! Config changes apply per model (docs/REVIEW_PLAN.md, phase 3): [`ModelRegistry::apply_live`]
//! swaps the per-request settings of every worker, [`ModelRegistry::reload_model`] replaces,
//! starts or stops one model's worker without touching the others (no `LoadGate`: a reload
//! compiles as soon as the runtimes lock is free). The decode + preprocess pool and the runtimes
//! belong to the generation: a reload uses them as they are.

use crate::backend::{CoreOptions, DeviceSpec, LoadRequest, OrtOptions, Runtimes, Selection};
use crate::config::{Config, ModelConfig};
use crate::metrics::{Metrics, ModelGauges};
use crate::profile::Profile;
use crate::resources::provision::Provision;
use crate::sync::{LockExt, RwLockExt};
use crate::worker::{LiveSettings, LoadGate, WorkerConfig, WorkerHandle, spawn_worker_after};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

mod plan;

pub use crate::backend::RuntimeInfo;
pub use plan::{ModelPlan, PlanOutcome, PlanRuntimes, plan_model, plan_models};

/// Lowercase and strip a trailing `.onnx` / `.xml` (Blue Iris sends file stems, users may not).
pub fn normalize_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    for ext in [".onnx", ".xml"] {
        if let Some(stripped) = lower.strip_suffix(ext)
            && !stripped.is_empty()
        {
            return stripped.to_string();
        }
    }
    lower
}

/// Build the per-model `LoadRequest` (the device spec comes from `Config::device_spec_for`).
pub fn load_request(config: &Config, m: &ModelConfig, path: PathBuf) -> LoadRequest {
    LoadRequest {
        path,
        requested: config.device_for(m),
        gpu_precision: m.gpu_precision.clone(),
    }
}

/// Index into the enabled models of the model serving `/v1/vision/detection`: `default_model`
/// matched with the same normalization as `/v1/vision/custom/{model}` (case-insensitive,
/// `.onnx`/`.xml` ignored), else the first enabled model. None when no model is enabled.
fn resolve_default(config: &Config) -> Option<usize> {
    let idx = config.effective_default_index()?;
    if let Some(name) = config
        .default_model
        .as_deref()
        .filter(|n| !n.trim().is_empty())
    {
        match config.default_model_index() {
            None => {
                warn!("default_model '{name}' is not configured; using the first enabled model")
            }
            Some(i) if !config.models[i].enabled => {
                warn!("default_model '{name}' is disabled; using the first enabled model")
            }
            Some(_) => {}
        }
    }
    Some(config.models[..idx].iter().filter(|m| m.enabled).count())
}

/// What [`ModelRegistry::apply_live`] changed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LiveApplied {
    /// Served models whose live settings changed.
    pub models: Vec<String>,
    /// The default model changed (to `default_model`; None: nothing is served).
    pub default_changed: bool,
    pub default_model: Option<String>,
}

/// The served model (index into the workers, looked up by normalized name in `by_name`) that
/// `config` makes the default: its effective default when that model is served, else the
/// first served model.
fn default_by_name(
    config: &Config,
    by_name: &HashMap<String, usize>,
    served: usize,
) -> Option<usize> {
    config
        .effective_default_index()
        .and_then(|i| {
            by_name
                .get(&normalize_name(&config.models[i].effective_name()))
                .copied()
        })
        .or((served > 0).then_some(0))
}

/// Decode + preprocess threads of a generation: the configured `preprocess_threads`, else the
/// profile's, capped when every model is headed for a CPU device (`planned`: the first runnable
/// device of each model, None when nothing can run).
pub fn preprocess_threads(
    config: &Config,
    profile: &Profile,
    planned: &[Option<crate::backend::spec::Device>],
) -> usize {
    if config.preprocess_threads > 0 {
        return config
            .preprocess_threads
            .min(crate::profile::MAX_PREPROCESS_THREADS * 4);
    }
    let mut known = planned.iter().flatten().peekable();
    let all_cpu =
        known.peek().is_some() && known.all(|d| d.target == crate::backend::spec::Target::Cpu);
    profile.preprocess_threads_for(all_cpu)
}

/// One served model: its handle (shared with the HTTP handlers that are using it), its own stop
/// token (a child of the generation's worker token) and its thread.
struct WorkerSlot {
    handle: Arc<WorkerHandle>,
    stop: CancellationToken,
    join: Option<JoinHandle<()>>,
}

/// The served models. `by_name` and the default index always match `workers` (they change
/// together under one lock).
#[derive(Default)]
struct Slots {
    workers: Vec<WorkerSlot>,
    /// Lowercase name -> index into `workers`.
    by_name: HashMap<String, usize>,
    default_idx: Option<usize>,
    /// Normalized name -> reloads of that model in this generation.
    reloads: HashMap<String, u64>,
}

impl Slots {
    fn from_slots(workers: Vec<WorkerSlot>, default_idx: Option<usize>) -> Self {
        let mut s = Self {
            workers,
            ..Self::default()
        };
        s.rebuild_names();
        s.default_idx = default_idx.filter(|&i| i < s.workers.len());
        s
    }

    fn rebuild_names(&mut self) {
        self.by_name = self
            .workers
            .iter()
            .enumerate()
            .map(|(i, w)| (normalize_name(&w.handle.name), i))
            .collect();
    }

    /// Put the workers in `config` order (models it does not list keep their order at the end),
    /// rebuild the name map and resolve the default model.
    fn reindex(&mut self, config: &Config) {
        let pos = |name: &str| {
            let key = normalize_name(name);
            config
                .models
                .iter()
                .position(|m| normalize_name(&m.effective_name()) == key)
                .unwrap_or(usize::MAX)
        };
        self.workers.sort_by_key(|w| pos(&w.handle.name));
        self.rebuild_names();
        self.default_idx = default_by_name(config, &self.by_name, self.workers.len());
    }
}

/// What a registry started from a config needs to spawn (and respawn) workers.
struct Spawner {
    runtimes: Arc<Mutex<Runtimes>>,
    metrics: Arc<Metrics>,
    /// The generation's worker token (each worker gets a child of it).
    shutdown: CancellationToken,
    profile: Profile,
    /// The generation's decode + preprocess pool (shared by every worker, reloads included).
    prep: Arc<crate::prep::PrepPool>,
    prep_threads: usize,
    best_cpu: DeviceSpec,
    can_use_gpu: bool,
    /// The provisioning plan made for the latest reload (None: the generation's plan holds).
    provision: RwLock<Option<Arc<Provision>>>,
    /// One reload at a time.
    reload: Mutex<()>,
}

pub struct ModelRegistry {
    slots: RwLock<Slots>,
    pub runtime_info: RuntimeInfo,
    /// None for registries built from handles (they cannot spawn workers).
    spawner: Option<Spawner>,
    /// The `stop-{name}` helper threads of workers removed by a disable or reload
    /// ([`retire_worker`]); pruned of finished ones on each retire, joined by [`Self::shutdown`].
    retiring: Mutex<Vec<JoinHandle<()>>>,
}

/// What [`ModelRegistry::reload_model`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reload {
    /// A new worker was spawned (replacing the old one, if any); the text names where it is
    /// headed ("OpenVINO CPU", "auto").
    Spawned {
        name: String,
        device: String,
        /// An old worker was replaced (false: the model was not served before).
        replaced: bool,
    },
    /// The model is no longer served (disabled or removed); its worker stops once its requests
    /// are answered.
    Stopped { name: String },
    /// Not served before and not enabled now: nothing to do.
    Unchanged { name: String },
}

impl Reload {
    /// "reloading IPcam-general on OpenVINO CPU", "loading x on auto", "stopping yolo26s".
    pub fn describe(&self) -> String {
        match self {
            Reload::Spawned {
                name,
                device,
                replaced: true,
            } => format!("reloading {name} on {device}"),
            Reload::Spawned { name, device, .. } => format!("loading {name} on {device}"),
            Reload::Stopped { name } => format!("stopping {name}"),
            Reload::Unchanged { name } => format!("{name} unchanged"),
        }
    }
}

/// The spec, label and planned devices helpers need the runtimes lock; a reload must not wait
/// behind a model that is compiling, so it only tries.
fn try_lock_runtimes(rt: &Mutex<Runtimes>) -> Option<std::sync::MutexGuard<'_, Runtimes>> {
    rt.try_lock_recover()
}

impl Spawner {
    /// Spawn the worker of a planned model (or a `Failed` handle when the plan failed: missing
    /// file, pending export, bad classes, downloads that must be fetched by hand). `after` is
    /// the startup [`LoadGate`] to wait for (reloads pass None); `stop` its own token. The
    /// returned handle owns the thread's join handle (`join`), None for a failed handle.
    fn spawn_model(
        &self,
        config: &Config,
        plan: ModelPlan,
        after: Option<LoadGate>,
        stop: CancellationToken,
    ) -> WorkerHandle {
        let ModelPlan {
            name,
            device,
            load,
            classes,
            live,
            lazy,
            wait,
            model,
            planned,
            outcome,
            ..
        } = plan;
        // Counters live for the whole process; the device/provider restart on each load.
        let mm = self.metrics.model(&name);
        mm.begin_load(load.requested.clone());
        if let PlanOutcome::Failed(msg) = outcome {
            warn!(model = %name, "{msg}");
            return WorkerHandle::failed(name, msg, mm);
        }

        let cfg = WorkerConfig {
            name,
            device,
            load,
            classes,
            live: live.shared(),
            queue_size: config.worker_queue_size,
            lazy,
            can_use_gpu: self.can_use_gpu,
            concurrency: Some(crate::worker::Concurrency {
                profile: self.profile.clone(),
                overrides: config.concurrency_overrides(&model),
            }),
            model,
            wait,
            prep: Some(self.prep.clone()),
        };
        let mut handle = spawn_worker_after(self.runtimes.clone(), cfg, mm, stop, after);
        handle.set_planned(device, planned.map(|(_, label)| label));
        handle
    }

    /// Where each enabled model of `config` is headed (its first runnable load candidate);
    /// None when the runtimes are busy compiling.
    fn planned_devices(
        &self,
        config: &Config,
    ) -> Option<Vec<Option<crate::backend::spec::Device>>> {
        let rt = try_lock_runtimes(&self.runtimes)?;
        Some(
            config
                .enabled_models()
                .map(|m| {
                    let device = if config.force_cpu {
                        self.best_cpu
                    } else {
                        config.device_spec_for(m).ok()?
                    };
                    rt.first_runnable(&device, &config.data_path(&m.path))
                        .map(|(d, _)| d)
                })
                .collect(),
        )
    }
}

/// Stop a worker that left the registry without blocking the caller: on a helper thread, let
/// its in-flight requests finish (the handle's sender goes away with the last HTTP handler that
/// uses it, and the worker exits once its channel is empty), at most the request timeout plus
/// a second, then cancel its token and join it. Returns the helper thread (None: the worker had
/// no thread, or the helper could not be spawned).
fn retire_worker(slot: WorkerSlot) -> Option<JoinHandle<()>> {
    let WorkerSlot { handle, stop, join } = slot;
    let name = handle.name.clone();
    let drain = handle.request_timeout() + Duration::from_secs(1);
    drop(handle);
    let Some(join) = join else {
        stop.cancel();
        return None;
    };
    let spawned = std::thread::Builder::new()
        .name(format!("stop-{name}"))
        .spawn(move || {
            let deadline = Instant::now() + drain;
            while !join.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            stop.cancel();
            if join.join().is_err() {
                error!(model = %name, "worker thread panicked");
            } else {
                info!(model = %name, "previous worker stopped");
            }
        });
    match spawned {
        Ok(helper) => Some(helper),
        Err(e) => {
            // The thread still stops (its token is cancelled); it is just not joined.
            warn!("could not spawn a thread to stop a worker: {e}");
            None
        }
    }
}

impl ModelRegistry {
    /// Create the `Core` and spawn one worker per enabled model. Returns as soon as the
    /// workers are spawned; models compile in the background (see `WorkerHandle::state`).
    /// An empty `models` list is an error; models that are all disabled give an empty registry
    /// (the web UI stays up so a model can be enabled).
    pub fn start(
        config: &Config,
        metrics: &Arc<Metrics>,
        shutdown: CancellationToken,
    ) -> Result<Self> {
        Self::start_with(config, metrics, shutdown, None)
    }

    /// [`Self::start`] for a generation planned by the [`Provisioner`]
    /// (`crate::resources::provision`): a model blocked on a download either loads on what is
    /// runnable now (its file is there and only a runtime is missing) or waits with progress in
    /// its state; with `auto_download: false` it is `Failed` with the command to run. Missing
    /// runtimes are not fatal while something is being provisioned.
    ///
    /// [`Provisioner`]: crate::resources::provision::Provisioner
    pub fn start_with(
        config: &Config,
        metrics: &Arc<Metrics>,
        shutdown: CancellationToken,
        provision: Option<&Provision>,
    ) -> Result<Self> {
        Self::start_with_benchmark(config, metrics, shutdown, provision, None)
    }

    /// [`Self::start_with`], ordering each `auto` model's devices by the benchmark results at
    /// `benchmark` (`benchmark.json` beside the config, read once per registry): the measured
    /// devices fastest first, then the rest of the static ranking (see
    /// `Selection::apply_measured_order`). A missing or unreadable file changes nothing.
    pub fn start_with_benchmark(
        config: &Config,
        metrics: &Arc<Metrics>,
        shutdown: CancellationToken,
        provision: Option<&Provision>,
        benchmark: Option<&Path>,
    ) -> Result<Self> {
        if config.models.is_empty() {
            bail!(
                "no models configured: pass `--model <path> [--family yolo5|yolo8|yolo26|rtdetr]`, \
                 add entries to `models` in the config file, or fetch one with \
                 `blue-onyx-prism download-models --name IPcam-general`"
            );
        }
        validate_names(config)?;
        let enabled: Vec<&ModelConfig> = config.enabled_models().collect();
        let disabled = config.models.len() - enabled.len();
        if enabled.is_empty() {
            warn!(
                "all {disabled} configured models are disabled; nothing will be served until one \
                 is enabled on the Config page (or with `\"enabled\": true` in the config file)"
            );
        } else if disabled > 0 {
            info!(disabled, "skipping disabled models");
        }

        let opts = CoreOptions {
            cache_dir: config.cache_dir_path(),
            intra_threads: config.intra_threads,
            openvino_dir: config.openvino_dir_effective(),
        };
        let ort_opts: OrtOptions = config.ort_options();
        let mut runtimes = Runtimes::new_with(&opts, &ort_opts);
        if let Some(path) = benchmark {
            let results = crate::benchmark::BenchmarkResults::load_or_warn(path);
            let orders = crate::benchmark::auto_orders(
                results.as_ref(),
                config,
                enabled.iter().copied(),
                runtimes.hardware(),
            );
            runtimes.set_measured_orders(orders);
        }
        if let Err(e) = runtimes.require_any() {
            match provision.filter(|p| p.has_needs()) {
                // Fresh install: the runtimes are being downloaded (or must be fetched); serve
                // HTTP with the models waiting / failed instead of exiting.
                Some(_) => warn!("{e:#} (the missing runtime is being provisioned)"),
                None => return Err(e),
            }
        }
        let runtime_info = runtimes.info();
        if runtimes.openvino().is_some() {
            info!(
                version = %runtime_info.openvino_version,
                devices = ?runtime_info.available_devices,
                cache_dir = ?opts.cache_dir,
                "OpenVINO core ready"
            );
        } else {
            warn!(
                "OpenVINO unavailable: {}",
                runtimes.openvino_error().unwrap_or("not initialized")
            );
        }
        match runtimes.onnxruntime_version() {
            Some(v) => info!(version = %v, "ONNX Runtime ready"),
            None => info!(
                "ONNX Runtime unavailable: {}",
                runtimes.onnxruntime_error().unwrap_or("not initialized")
            ),
        }
        let selection = runtimes.selection(None);
        info!(
            auto = %crate::backend::select::format_auto(&selection),
            can_use_gpu = runtime_info.has_gpu,
            "device options probed"
        );
        let best_cpu = runtimes.best_cpu();
        for m in &enabled {
            let path = config.data_path(&m.path);
            if runtimes.measured_orders().contains_key(&path)
                && matches!(config.device_spec_for(m), Ok(DeviceSpec::Auto))
                && !config.force_cpu
            {
                let sel = runtimes.selection(Some(&path));
                info!(
                    model = %m.effective_name(),
                    auto = %crate::backend::select::format_auto(&sel),
                    "device order from benchmark results"
                );
            }
        }
        // Where each enabled model is headed and whether it can be spawned, decided before the
        // workers can hold the runtimes lock.
        let plans = plan_models(config, &runtimes, provision, best_cpu)?;
        let profile = config.profile();
        let planned_devices: Vec<_> = plans
            .iter()
            .map(|p| p.planned.as_ref().map(|(d, _)| *d))
            .collect();
        let prep_threads = preprocess_threads(config, &profile, &planned_devices);
        info!(
            profile = %profile.summary(prep_threads),
            "hardware profile"
        );
        let spawner = Spawner {
            runtimes: Arc::new(Mutex::new(runtimes)),
            metrics: metrics.clone(),
            shutdown,
            profile,
            prep: crate::prep::PrepPool::new(prep_threads),
            prep_threads,
            best_cpu,
            can_use_gpu: runtime_info.has_gpu,
            provision: RwLock::new(None),
            reload: Mutex::new(()),
        };

        let mut workers = Vec::with_capacity(plans.len());
        // Startup order: each spawned worker waits for the previous one's gate.
        let mut prev: Option<LoadGate> = None;
        for plan in plans {
            let stop = spawner.shutdown.child_token();
            let mut handle = spawner.spawn_model(config, plan, prev.clone(), stop.clone());
            let join = handle.join.take();
            if join.is_some() {
                prev = Some(handle.load_gate());
            }
            workers.push(WorkerSlot {
                handle: Arc::new(handle),
                stop,
                join,
            });
        }

        let default_idx = resolve_default(config);
        if let Some(i) = default_idx {
            info!(model = %workers[i].handle.name, "default model for /v1/vision/detection");
        }

        Ok(Self {
            slots: RwLock::new(Slots::from_slots(workers, default_idx)),
            runtime_info,
            spawner: Some(spawner),
            retiring: Mutex::new(Vec::new()),
        })
    }

    /// Registry over already constructed handles (no OpenVINO involved). Used by tests that
    /// exercise the HTTP layer with e.g. [`WorkerHandle::failed`] handles. It cannot reload.
    pub fn from_handles(
        workers: Vec<WorkerHandle>,
        default_idx: Option<usize>,
        runtime_info: RuntimeInfo,
    ) -> Self {
        let workers = workers
            .into_iter()
            .map(|mut h| {
                let join = h.join.take();
                WorkerSlot {
                    handle: Arc::new(h),
                    stop: CancellationToken::new(),
                    join,
                }
            })
            .collect();
        Self {
            slots: RwLock::new(Slots::from_slots(workers, default_idx)),
            runtime_info,
            spawner: None,
            retiring: Mutex::new(Vec::new()),
        }
    }

    /// Stop a worker that left the registry without waiting for it ([`retire_worker`]); its
    /// helper thread is kept for [`Self::shutdown`], and finished helpers are dropped.
    fn retire(&self, slot: WorkerSlot) {
        let helper = retire_worker(slot);
        let mut retiring = self.retiring.lock_recover();
        retiring.retain(|h| !h.is_finished());
        retiring.extend(helper);
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Slots> {
        self.slots.read_recover()
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Slots> {
        self.slots.write_recover()
    }

    /// The shared runtimes (None for registries built with [`Self::from_handles`]). Lock briefly:
    /// workers hold the lock while compiling.
    pub fn runtimes(&self) -> Option<&Arc<Mutex<Runtimes>>> {
        self.spawner.as_ref().map(|s| &s.runtimes)
    }

    /// The hardware profile of this generation and its decode + preprocess thread count (None
    /// for registries built with [`Self::from_handles`]). Reloads keep the pool size.
    pub fn profile(&self) -> Option<(&Profile, usize)> {
        self.spawner.as_ref().map(|s| (&s.profile, s.prep_threads))
    }

    /// The provisioning plan made by the latest reload that involved downloads (None: the
    /// generation's own plan is current).
    pub fn latest_provision(&self) -> Option<Arc<Provision>> {
        let s = self.spawner.as_ref()?;
        s.provision.read_recover().clone()
    }

    /// Record the plan reloads are made with (see [`Self::latest_provision`]).
    pub fn set_provision(&self, provision: Arc<Provision>) {
        if let Some(s) = &self.spawner {
            *s.provision.write_recover() = Some(provision);
        }
    }

    /// True when the registry can spawn workers (not built from handles).
    pub fn can_reload(&self) -> bool {
        self.spawner.is_some()
    }

    /// Device options and the `auto` ranking for `model` (None: any model), from the shared
    /// runtimes. None for registries built with [`Self::from_handles`].
    pub fn selection(&self, model: Option<&Path>) -> Option<Selection> {
        let rt = self.runtimes()?;
        let rt = rt.lock_recover();
        Some(rt.selection(model))
    }

    /// Live per-model gauges (state, device, queue) for `/prometheus`, in config order.
    pub fn gauges(&self) -> Vec<ModelGauges> {
        self.read()
            .workers
            .iter()
            .map(|w| w.handle.gauges())
            .collect()
    }

    /// The model serving `/v1/vision/detection`.
    pub fn default_model(&self) -> Option<Arc<WorkerHandle>> {
        let s = self.read();
        s.default_idx
            .and_then(|i| s.workers.get(i))
            .map(|w| w.handle.clone())
    }

    /// Case-insensitive lookup; a trailing `.onnx` / `.xml` is ignored.
    pub fn by_name(&self, name: &str) -> Option<Arc<WorkerHandle>> {
        let s = self.read();
        s.by_name
            .get(&normalize_name(name))
            .and_then(|&i| s.workers.get(i))
            .map(|w| w.handle.clone())
    }

    /// Names of the served (enabled) models in config order.
    pub fn names(&self) -> Vec<String> {
        self.read()
            .workers
            .iter()
            .map(|w| w.handle.name.clone())
            .collect()
    }

    /// The served models' handles in config order (a snapshot: a reload replaces handles).
    pub fn workers(&self) -> Vec<Arc<WorkerHandle>> {
        self.read()
            .workers
            .iter()
            .map(|w| w.handle.clone())
            .collect()
    }

    /// How often `name` was reloaded in this generation.
    pub fn reloads(&self, name: &str) -> u64 {
        self.read()
            .reloads
            .get(&normalize_name(name))
            .copied()
            .unwrap_or(0)
    }

    /// Apply the live part of `config` without reloading anything: every served model's
    /// [`LiveSettings`] (threshold, NMS IoU, object filter, request timeout, image saving) and
    /// the default model. Models of `config` that are not served (disabled, or not loaded yet)
    /// are skipped. Returns what changed.
    pub fn apply_live(&self, config: &Config) -> LiveApplied {
        let mut out = LiveApplied::default();
        let mut s = self.write();
        for w in &s.workers {
            let key = normalize_name(&w.handle.name);
            let Some(m) = config
                .models
                .iter()
                .find(|m| normalize_name(&m.effective_name()) == key)
            else {
                continue;
            };
            if w.handle.set_live(LiveSettings::for_model(config, m)) {
                out.models.push(w.handle.name.clone());
            }
        }
        let new = default_by_name(config, &s.by_name, s.workers.len());
        if s.default_idx != new {
            s.default_idx = new;
            out.default_model = new.map(|i| s.workers[i].handle.name.clone());
            out.default_changed = true;
            info!(
                model = out.default_model.as_deref().unwrap_or("none"),
                "default model for /v1/vision/detection changed"
            );
        }
        if !out.models.is_empty() {
            info!(models = ?out.models, "live settings applied");
        }
        out
    }

    /// Reload model `name` from `config` without touching the others: a served model gets a
    /// new worker (the old one is taken out of routing at once and stops on a helper thread
    /// once its requests are answered); a model that is enabled but not served is spawned; a
    /// served model that is disabled or gone from `config` is stopped. The new worker loads
    /// right away (no startup [`LoadGate`]), through this generation's runtimes and prep pool;
    /// `provision` is the plan for downloads it may wait for. Runtime-level settings come from
    /// `config` as given (the caller keeps the running ones; see
    /// `config_merge::runtime_overlay`). Errors for registries built from handles.
    pub fn reload_model(
        &self,
        name: &str,
        config: &Config,
        provision: Option<&Provision>,
    ) -> Result<Reload> {
        let Some(sp) = &self.spawner else {
            bail!("this registry cannot reload models");
        };
        let _one = sp.reload.lock_recover();
        validate_names(config)?;
        let key = normalize_name(name);
        let entry = config
            .models
            .iter()
            .find(|m| normalize_name(&m.effective_name()) == key);
        let served = self.read().by_name.contains_key(&key);
        let Some(m) = entry.filter(|m| m.enabled) else {
            if !served {
                return Ok(Reload::Unchanged {
                    name: name.to_string(),
                });
            }
            let old = {
                let mut s = self.write();
                let i = s.by_name[&key];
                let old = s.workers.remove(i);
                s.reindex(config);
                old
            };
            let name = old.handle.name.clone();
            info!(model = %name, "model disabled; stopping its worker");
            self.retire(old);
            return Ok(Reload::Stopped { name });
        };

        let plan = plan_model(
            config,
            m,
            &plan::LockedRuntimes(&sp.runtimes),
            provision,
            sp.best_cpu,
        );
        let stop = sp.shutdown.child_token();
        let mut handle = sp.spawn_model(config, plan, None, stop.clone());
        let join = handle.join.take();
        let device_text = handle.planned_device_text();
        let new_name = handle.name.clone();
        let slot = WorkerSlot {
            handle: Arc::new(handle),
            stop,
            join,
        };
        let old = {
            let mut s = self.write();
            let old = s.by_name.get(&key).copied().map(|i| s.workers.remove(i));
            s.workers.push(slot);
            *s.reloads.entry(key.clone()).or_default() += u64::from(old.is_some());
            s.reindex(config);
            old
        };
        let replaced = old.is_some();
        match old {
            Some(old) => {
                info!(model = %new_name, device = %device_text, "reloading model");
                self.retire(old);
            }
            None => info!(model = %new_name, device = %device_text, "model enabled; loading"),
        }
        // The decode + preprocess pool is sized once per generation.
        if config.preprocess_threads == 0
            && let Some(planned) = sp.planned_devices(config)
        {
            let want = preprocess_threads(config, &sp.profile, &planned);
            if want != sp.prep_threads {
                info!(
                    threads = sp.prep_threads,
                    wanted = want,
                    "the models' devices now call for another preprocess pool size; it keeps \
                     its size until the server restarts"
                );
            }
        }
        Ok(Reload::Spawned {
            name: new_name,
            device: device_text,
            replaced,
        })
    }

    /// Stop every worker and join the threads (the registry is left empty, so a straggling
    /// request of a retired generation gets "no model"). Workers no other thread holds a handle of exit
    /// once their channel is empty; the others are cancelled. Cancel the shutdown token first
    /// for a prompt exit (it also cuts short the drain of workers retired by a reload). Joins
    /// the live workers, then the helper threads stopping retired ones.
    pub fn shutdown(&self) {
        // Taken out of the registry (not consumed): a handler of a retired generation may still
        // hold an `Arc` of it while its connection winds down; it then sees no models.
        let slots = std::mem::take(&mut *self.slots.write_recover());
        for WorkerSlot { handle, stop, join } in slots.workers {
            let name = handle.name.clone();
            if Arc::try_unwrap(handle).is_err() {
                // Someone still holds the sender: do not wait for the disconnect.
                stop.cancel();
            }
            if let Some(j) = join
                && j.join().is_err()
            {
                error!(model = %name, "worker thread panicked");
            }
            info!(model = %name, "worker stopped");
        }
        let retiring = std::mem::take(&mut *self.retiring.lock_recover());
        for helper in retiring {
            if helper.join().is_err() {
                error!("thread stopping a retired worker panicked");
            }
        }
    }
}

/// Names of all entries (enabled or not) must be distinct after normalization.
fn validate_names(config: &Config) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for m in &config.models {
        let key = normalize_name(&m.effective_name());
        if !seen.insert(key.clone()) {
            bail!("duplicate model name '{key}' in config; set a distinct `name` for each entry");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_normalization() {
        assert_eq!(normalize_name("IPcam-General"), "ipcam-general");
        assert_eq!(normalize_name("IPcam-general.onnx"), "ipcam-general");
        assert_eq!(normalize_name("yolo26s.XML"), "yolo26s");
        assert_eq!(normalize_name(".onnx"), ".onnx");
        assert_eq!(normalize_name(" a.b "), "a.b");
    }

    #[test]
    fn load_request_fallback_rules() {
        use crate::backend::plan_candidates;
        let gpu = ["CPU".to_string(), "GPU".to_string()];
        let plan = |c: &Config, m: &ModelConfig| -> Vec<String> {
            plan_candidates(&c.device_spec_for(m).unwrap(), &gpu)
                .iter()
                .map(|c| c.device.to_string())
                .collect()
        };
        let mut c = Config {
            device: "GPU".into(),
            ..Config::default()
        };
        let m = ModelConfig::default();
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.requested, "GPU");
        assert_eq!(plan(&c, &m), ["openvino:gpu", "openvino:cpu"]);
        c.force_cpu = true;
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.requested, "CPU");
        assert_eq!(plan(&c, &m), ["openvino:cpu"]);
    }

    #[test]
    fn empty_config_is_an_error() {
        let c = Config::default();
        let m = Arc::new(Metrics::new("0"));
        let err = ModelRegistry::start(&c, &m, CancellationToken::new())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("--model"));
    }

    #[test]
    fn duplicate_names_rejected() {
        let mut c = Config::default();
        for p in ["a/x.onnx", "b/X.xml"] {
            c.models.push(ModelConfig {
                path: p.into(),
                ..Default::default()
            });
        }
        let m = Arc::new(Metrics::new("0"));
        let err = ModelRegistry::start(&c, &m, CancellationToken::new())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("duplicate"));
    }

    #[test]
    fn default_model_resolution() {
        let mut c = Config::default();
        for p in ["models/yolo26s.xml", "models/IPcam-general.onnx"] {
            c.models.push(ModelConfig {
                path: p.into(),
                ..Default::default()
            });
        }
        assert_eq!(resolve_default(&c), Some(0));
        for name in ["IPcam-general", "ipcam-GENERAL.onnx", " ipcam-general "] {
            c.default_model = Some(name.into());
            assert_eq!(resolve_default(&c), Some(1), "{name}");
        }
        c.default_model = Some("yolo26s.XML".into());
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some("nope".into());
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some(String::new());
        assert_eq!(resolve_default(&c), Some(0));
        c.models.clear();
        assert_eq!(resolve_default(&c), None);
    }

    #[test]
    fn default_model_resolution_over_enabled_models() {
        let mut c = Config::default();
        for (p, enabled) in [("a.onnx", false), ("b.onnx", true), ("c.onnx", true)] {
            c.models.push(ModelConfig {
                path: p.into(),
                enabled,
                ..Default::default()
            });
        }
        // Indices are into the enabled models (= the workers): b -> 0, c -> 1.
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some("c".into());
        assert_eq!(resolve_default(&c), Some(1));
        // Default names a disabled model: first enabled one.
        c.default_model = Some("a".into());
        assert_eq!(resolve_default(&c), Some(0));
        for m in &mut c.models {
            m.enabled = false;
        }
        assert_eq!(resolve_default(&c), None);
    }

    #[test]
    fn all_disabled_still_validates_names() {
        let mut c = Config::default();
        for p in ["a/x.onnx", "b/X.xml"] {
            c.models.push(ModelConfig {
                path: p.into(),
                enabled: false,
                ..Default::default()
            });
        }
        let m = Arc::new(Metrics::new("0"));
        let err = ModelRegistry::start(&c, &m, CancellationToken::new())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("duplicate"));
    }

    #[test]
    fn pool_size_from_profile_and_planned_devices() {
        use crate::backend::spec::{Device, Runtime, Target};
        use crate::backend::{GpuAdapter, GpuVendor, HardwareInfo};
        use crate::profile::{Facts, classify};
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![GpuAdapter {
                vendor: GpuVendor::Intel,
                name: "Intel(R) UHD Graphics 630".into(),
                vram_mb: 0,
                index: 0,
                discrete: false,
                cuda: None,
            }],
        );
        let profile = classify(&Facts::from_hardware(6, Some(16.0), &hw));
        let cpu = Device::new(Runtime::OpenVino, Target::Cpu, None).unwrap();
        let ort_cpu = Device::new(Runtime::Ort, Target::Cpu, None).unwrap();
        let gpu = Device::new(Runtime::OpenVino, Target::Gpu, None).unwrap();
        let mut c = Config::default();
        assert_eq!(preprocess_threads(&c, &profile, &[]), 3);
        assert_eq!(preprocess_threads(&c, &profile, &[Some(gpu), Some(cpu)]), 3);
        assert_eq!(
            preprocess_threads(&c, &profile, &[Some(cpu), Some(ort_cpu), None]),
            2,
            "every model on a CPU device"
        );
        assert_eq!(preprocess_threads(&c, &profile, &[None]), 3);
        c.preprocess_threads = 6;
        assert_eq!(preprocess_threads(&c, &profile, &[Some(cpu)]), 6);
    }

    #[test]
    fn apply_live_swaps_settings_and_default() {
        let metrics = Metrics::new("0");
        let workers = ["a", "b"]
            .iter()
            .map(|n| WorkerHandle::failed(*n, "test", metrics.model(n)))
            .collect();
        let reg = ModelRegistry::from_handles(workers, Some(0), RuntimeInfo::default());
        let mut c = Config::default();
        for n in ["a", "b", "off"] {
            c.models.push(ModelConfig {
                name: Some(n.into()),
                path: format!("{n}.onnx").into(),
                enabled: n != "off",
                ..Default::default()
            });
        }
        // Defaults everywhere: nothing changes.
        assert_eq!(reg.apply_live(&c), LiveApplied::default());
        c.confidence_threshold = 0.3;
        c.models[1].confidence_threshold = Some(0.8);
        c.models[1].object_filter = Some(vec!["person".into()]);
        c.request_timeout_secs = 3;
        c.default_model = Some("B".into());
        let applied = reg.apply_live(&c);
        assert_eq!(applied.models, ["a", "b"]);
        assert!(applied.default_changed);
        assert_eq!(reg.default_model().unwrap().name, "b");
        let a = reg.by_name("a").unwrap();
        assert_eq!(a.live().confidence_threshold, 0.3);
        assert_eq!(a.request_timeout(), Duration::from_secs(3));
        let b = reg.by_name("b").unwrap().live();
        assert_eq!(b.confidence_threshold, 0.8);
        assert_eq!(b.object_filter, ["person"]);
        // Idempotent.
        assert_eq!(reg.apply_live(&c), LiveApplied::default());
        // A default that is not served falls back to the first served model.
        c.default_model = Some("off".into());
        assert!(reg.apply_live(&c).default_changed);
        assert_eq!(reg.default_model().unwrap().name, "a");
    }

    #[test]
    fn registries_from_handles_cannot_reload() {
        let metrics = Metrics::new("0");
        let reg = ModelRegistry::from_handles(
            vec![WorkerHandle::failed("a", "test", metrics.model("a"))],
            Some(0),
            RuntimeInfo::default(),
        );
        assert!(!reg.can_reload());
        let c = Config::default();
        assert!(reg.reload_model("a", &c, None).is_err());
        assert_eq!(reg.reloads("a"), 0);
        assert_eq!(reg.names(), ["a"]);
    }

    #[test]
    fn shutdown_joins_retired_workers() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let metrics = Metrics::new("0");
        let reg = ModelRegistry::from_handles(Vec::new(), None, RuntimeInfo::default());
        let generation = CancellationToken::new();
        let stopped = Arc::new(AtomicBool::new(false));
        // A worker that only exits once its token is cancelled (its sender is still held).
        let stop = generation.child_token();
        let (token, flag) = (stop.clone(), stopped.clone());
        let join = std::thread::spawn(move || {
            while !token.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            std::thread::sleep(Duration::from_millis(50));
            flag.store(true, Ordering::SeqCst);
        });
        reg.retire(WorkerSlot {
            handle: Arc::new(WorkerHandle::failed("a", "test", metrics.model("a"))),
            stop,
            join: Some(join),
        });
        // A handle without a thread is cancelled in place; no helper is kept.
        let b_stop = generation.child_token();
        reg.retire(WorkerSlot {
            handle: Arc::new(WorkerHandle::failed("b", "test", metrics.model("b"))),
            stop: b_stop.clone(),
            join: None,
        });
        assert!(b_stop.is_cancelled());
        assert_eq!(reg.retiring.lock_recover().len(), 1);
        assert!(!stopped.load(Ordering::SeqCst));
        // Cancelling the generation token first (as the runner does) cuts the drain short; the
        // helper then joins the worker, and shutdown joins the helper.
        let t = Instant::now();
        generation.cancel();
        reg.shutdown();
        assert!(
            stopped.load(Ordering::SeqCst),
            "shutdown returned before the worker"
        );
        assert!(t.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn reload_texts() {
        let r = Reload::Spawned {
            name: "m".into(),
            device: "OpenVINO CPU".into(),
            replaced: true,
        };
        assert_eq!(r.describe(), "reloading m on OpenVINO CPU");
        let r = Reload::Spawned {
            name: "m".into(),
            device: "auto".into(),
            replaced: false,
        };
        assert_eq!(r.describe(), "loading m on auto");
        assert_eq!(
            Reload::Stopped { name: "m".into() }.describe(),
            "stopping m"
        );
    }
}
