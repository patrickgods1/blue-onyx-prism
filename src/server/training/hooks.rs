//! What the training queue acts on: the run that started it ([`Target`], followed across
//! generation restarts until that run stops), its downloads through that run's download manager
//! ([`RunFetcher`]) and its [`Hooks`] ([`ServerHooks`]: add a finished model to the config with
//! [`save_config`], benchmark it and the default model on its held-out frames).

use super::records;
use crate::server::{AppState, Shared, save_config};
use crate::sync::LockExt;
use crate::training::bench;
use crate::training::job::{Baseline, BenchSummary, Hooks, TrainOutput};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

/// What the queue's hooks and downloads act on: the [`Shared`] of the run that started the queue
/// (so a generation restart is followed), until that run stops ([`Self::detach`]). A finished
/// run's [`AppState`] (its registry and runtimes, provisioner and download manager, benchmark
/// runner, config store) is never kept by the queue: a hook of a detached target fails with a
/// message instead of acting on a dead run (audit RA-006 / RA-008 / RA-021).
pub(super) struct Target {
    shared: std::sync::Mutex<Option<Shared>>,
}

impl Target {
    pub(super) fn new(shared: &Shared) -> Arc<Self> {
        Arc::new(Self {
            shared: std::sync::Mutex::new(Some(shared.clone())),
        })
    }

    /// The run's current generation, while the run serves.
    pub(super) fn state(&self) -> Option<Arc<AppState>> {
        self.shared.lock_recover().as_ref().map(Shared::current)
    }

    /// [`Self::state`], or why `what` cannot be done.
    pub(super) fn require(&self, what: &str) -> anyhow::Result<Arc<AppState>> {
        self.state().ok_or_else(|| {
            anyhow::anyhow!(
                "cannot {what}: the server is stopping or restarting. The run keeps its files; \
                 once the service is back, use Retry export on its row, or queue it again"
            )
        })
    }

    pub(super) fn shared(&self) -> Option<Shared> {
        self.shared.lock_recover().clone()
    }

    /// Let go of the run (its stop).
    pub(super) fn detach(&self) {
        self.shared.lock_recover().take();
    }
}

/// The queue's downloads: through the serving run's download manager, looked up per download and
/// held only while it runs, so a finished run's manager is never kept (and stops with its run).
/// Without a resources context (tests), one manager of the queue's own on the data root.
pub(super) struct RunFetcher {
    target: Arc<Target>,
    own: OnceLock<crate::resources::export::ManagerFetcher>,
}

impl RunFetcher {
    pub(super) fn new(target: Arc<Target>) -> Self {
        Self {
            target,
            own: OnceLock::new(),
        }
    }
}

impl crate::resources::export::Fetcher for RunFetcher {
    fn fetch(
        &self,
        res: &'static crate::resources::catalog::Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> anyhow::Result<()> {
        let state = self.target.require(&format!("download {}", res.title))?;
        let config = state.config_read().clone();
        let manager = state
            .resources
            .as_ref()
            .map(|ctx| ctx.provisioner.manager_for(&config));
        drop(state);
        match manager {
            Some(m) => {
                crate::resources::export::ManagerFetcher::new(m).fetch(res, dir, progress, cancel)
            }
            None => self
                .own
                .get_or_init(|| crate::resources::export::cli_fetcher(&config.data_root()))
                .fetch(res, dir, progress, cancel),
        }
    }
}

/// The queue's [`Hooks`] on the [`Target`] run.
pub(super) struct ServerHooks {
    pub(super) target: Arc<Target>,
}

impl ServerHooks {
    /// The benchmark context of the serving generation at each call (`held`, the generation the
    /// hook started on, once the run stopped: the hook ends on the job's cancel then).
    fn context(
        &self,
        held: &Arc<AppState>,
        results_path: Option<PathBuf>,
    ) -> impl Fn() -> crate::benchmark::service::RunContext + use<> {
        let (target, held) = (self.target.clone(), held.clone());
        move || {
            let st = target.state().unwrap_or_else(|| held.clone());
            crate::benchmark::service::RunContext {
                config: st.config_read().clone(),
                results_path: results_path
                    .clone()
                    .unwrap_or_else(|| crate::benchmark::results_path(&st.config_path)),
                runtimes: st.registry.runtimes().cloned(),
                loaded: crate::server::benchmark::loaded_devices(&st),
                resources: st.resources.clone(),
                profile: st.registry.profile().map(|(p, _)| p.clone()),
                metrics: Some(st.metrics.clone()),
            }
        }
    }
}

impl Hooks for ServerHooks {
    fn register(&self, out: &TrainOutput) -> anyhow::Result<String> {
        let state = self
            .target
            .require(&format!("add {} to the config", out.name))?;
        let mut msg = String::new();
        let saved = save_config(&state, "training", |c| {
            msg = crate::training::cli::add_trained_model(c, out)?;
            Ok(())
        })?;
        Ok(format!("{msg} {saved}"))
    }

    fn config_has_model(&self, name: &str) -> bool {
        // QA-101: the run-start re-check against the running config (a stopped service knows
        // none, and `register` names that problem).
        self.target.require("check the model name").is_ok_and(|s| {
            s.config_read()
                .models
                .iter()
                .any(|m| crate::model::name::same_model_name(&m.effective_name(), name))
        })
    }

    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        devices: &[String],
        cancel: &AtomicBool,
    ) -> anyhow::Result<Option<BenchSummary>> {
        let state = self.target.require(&format!("benchmark {}", out.name))?;
        let ctx = self.context(&state, None);
        bench::run(
            &state.benchmark,
            &ctx,
            &out.name,
            bench::datasets(&out.name, out.heldout.as_deref(), extra),
            devices,
            cancel,
        )
    }

    fn baseline(
        &self,
        out: &TrainOutput,
        extra: &[String],
        devices: &[String],
        run_dir: &Path,
        cancel: &AtomicBool,
    ) -> anyhow::Result<Option<Baseline>> {
        let sets = bench::datasets(&out.name, out.heldout.as_deref(), extra);
        if sets.is_empty() {
            return Ok(None);
        }
        let state = self.target.require("benchmark the default model")?;
        let config = state.config_read().clone();
        let default = crate::training::switch::current_default(&config);
        let name = default.clone().unwrap_or_default();
        let key = crate::registry::normalize_name(&name);
        let loaded = crate::server::benchmark::loaded_devices(&state).contains_key(&key);
        let classes = crate::training::autolabel::catalog::model_classes(&config)
            .into_iter()
            .find(|m| crate::registry::normalize_name(&m.name) == key)
            .map(|m| {
                crate::training::switch::reported(
                    &m.classes,
                    m.object_filter.as_deref(),
                    &config.object_filter,
                )
            });
        let skip = bench::baseline_skip(
            default.as_deref(),
            &out.name,
            &out.classes,
            classes.as_deref(),
            loaded,
        );
        if let Some(why) = skip {
            return Ok(Some(Baseline {
                model: name,
                note: Some(why),
                ..Baseline::default()
            }));
        }
        // Its own results file: see `bench`'s module docs.
        let dir = run_dir.join("baseline");
        let results_path = dir.join(crate::benchmark::report::RESULTS_FILE);
        let ctx = self.context(&state, Some(results_path));
        // What it would measure (None: cannot tell, measured as before); a baseline of the same
        // inputs measured for another run of this comparison group is taken over.
        let resolved = bench::resolve_devices(&mut ctx(), &name, devices).ok();
        let reuse_key = resolved
            .clone()
            .and_then(|d| bench::baseline_inputs(&config, &name, &sets, d))
            .map(|i| bench::baseline_key(&i));
        if let Some(k) = &reuse_key
            && let Some(b) = bench::reusable_baseline(&records(&state), run_dir, k)
        {
            return Ok(Some(b));
        }
        std::fs::create_dir_all(&dir)?;
        let result = bench::run(&state.benchmark, &ctx, &name, sets, devices, cancel);
        let _ = std::fs::remove_dir_all(&dir);
        match result {
            // Every device failed: a result of nothing is no comparison.
            Ok(Some(b)) if b.device.is_none() => Ok(Some(Baseline {
                note: Some(format!(
                    "{name} did not run on the benchmark devices; try it on the Benchmark page"
                )),
                model: name,
                ..Baseline::default()
            })),
            Ok(summary) => {
                // The key only stands for a result measured on the devices it names.
                let ran_on_key = summary
                    .as_ref()
                    .and_then(|s| s.device.as_ref())
                    .zip(resolved.as_ref())
                    .is_some_and(|(d, r)| {
                        let canon = |x: &str| crate::backend::spec::parse(x).map(|s| s.to_string());
                        r.iter().any(|x| {
                            x == d || matches!((canon(x), canon(d)), (Ok(a), Ok(b)) if a == b)
                        })
                    });
                Ok(Some(Baseline {
                    model: name,
                    summary,
                    key: reuse_key.filter(|_| ran_on_key),
                    ..Baseline::default()
                }))
            }
            Err(e)
                if e.downcast_ref::<crate::resources::export::Cancelled>()
                    .is_some() =>
            {
                Err(e)
            }
            Err(e) => Ok(Some(Baseline {
                model: name,
                note: Some(format!("{e:#}")),
                ..Baseline::default()
            })),
        }
    }
}
