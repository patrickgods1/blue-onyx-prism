//! The server run loop shared by the CLI binary and the Windows service.
//!
//! One *generation* is one [`ModelRegistry`] plus its [`AppState`]. `POST /config/restart`
//! cancels the generation token: the loop reloads the config file, applies a changed log level,
//! and starts the next generation. If the new config fails to start, the previous one is
//! restored. Cancelling the `shutdown` token (Ctrl-C/SIGTERM in the CLI, Stop in the service)
//! ends the loop.
//!
//! On-demand resources (phase 7.3): before each generation the [`Provisioner`] (which owns the
//! download manager for the whole run) works out what the config is missing and queues it.
//! When a runtime the generation can use gets installed, a watcher thread restarts the
//! generation the same way `/config/restart` does, but keeps the in-memory config (runtime
//! libraries load once per generation); an installed model file reloads only the models that
//! failed without it.
//!
//! The HTTP listener is bound once per run and serves [`server::Shared`]: a restart builds the
//! next generation while the old one keeps answering, swaps it in, and stops the old workers
//! once its in-flight requests have finished (at most [`DRAIN`]).

use crate::cli::LogReloadHandle;
use crate::config::Config;
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;
use crate::resources::provision::{Provision, Provisioner};
use crate::server::{self, AppState};
use crate::sync::LockExt;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

/// Error for a config without models (the server has nothing to serve).
pub fn ensure_models(config: &Config, config_path: &std::path::Path) -> Result<()> {
    if config.models.is_empty() {
        anyhow::bail!(
            "no models configured in {}. Start with `--model <path/to/model.onnx|.xml> --family yolo5` \
             or fetch one first with `blue-onyx-prism download-models --name IPcam-general`",
            config_path.display()
        );
    }
    Ok(())
}

/// Build the tokio runtime the server runs on: two worker threads (handlers are short; the point
/// is that one slow handler cannot stall the detection routes) plus the blocking pool that
/// [`server::blocking`] work runs on.
pub fn build_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("building tokio runtime")
}

/// Run the server until `shutdown` is cancelled, on a runtime built for this call.
/// See [`run_server_on`].
pub fn run_server(
    config: Config,
    config_path: PathBuf,
    log: LogReloadHandle,
    shutdown: CancellationToken,
) -> Result<()> {
    let rt = build_runtime()?;
    run_server_on(&rt, config, config_path, log, shutdown)
}

/// Run registry + HTTP server generations on `rt` until `shutdown` is cancelled (returns the
/// last server result, normally `Ok`). Returns an error when the first generation cannot start
/// (no models, OpenVINO init failure, port in use) and there is no previous config to restore.
pub fn run_server_on(
    rt: &tokio::runtime::Runtime,
    config: Config,
    config_path: PathBuf,
    log: LogReloadHandle,
    shutdown: CancellationToken,
) -> Result<()> {
    run_server_with(rt, config, config_path, log, shutdown, Provisioner::new())
}

/// How long a retired generation's in-flight requests may take before its workers are stopped.
const DRAIN: Duration = Duration::from_secs(5);

/// How long shutdown waits for the HTTP side to let go of the last generation's state after
/// the server task has returned. The server task can return a moment before every connection
/// task has dropped its copy of the router (axum 0.8 releases a connection's "still open" guard
/// before dropping the connection and the service it owns), so the state, and through it the
/// registry, can be shared for a few more microseconds; without this wait `finish` would skip
/// `ModelRegistry::shutdown` (no worker join) whenever it lost that race. Normally the wait
/// ends at once.
const SHUTDOWN_LINGER: Duration = Duration::from_secs(1);

/// A generation that is (or was) current: what has to be stopped once it is replaced.
struct Generation {
    state: Arc<AppState>,
    registry: Arc<ModelRegistry>,
    workers: CancellationToken,
    watcher: Option<std::thread::JoinHandle<()>>,
}

impl Generation {
    /// Stop a generation that is no longer current: wait until its in-flight requests let go of
    /// its state (at most `drain`), then stop its workers and watcher.
    fn retire(self, drain: Duration) {
        let Generation {
            state,
            registry,
            workers,
            watcher,
        } = self;
        state.restart.cancel();
        let deadline = Instant::now() + drain;
        while Arc::strong_count(&state) > 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if Arc::strong_count(&state) > 1 {
            warn!(
                "requests of the previous generation still running after {} s; stopping its workers",
                drain.as_secs()
            );
        }
        drop(state);
        workers.cancel();
        if let Some(w) = watcher
            && w.join().is_err()
        {
            warn!("resource watcher panicked");
        }
        // Shut down in place even when a connection task still holds the generation's state
        // (axum drops a connection's router clone after signalling the server future, see
        // `SHUTDOWN_LINGER`): the workers are always joined.
        if Arc::strong_count(&registry) > 1 {
            debug!(
                "registry still referenced by a winding-down request; shutting it down in place"
            );
        }
        registry.shutdown();
    }
}

/// The retention sweeper thread ([`crate::retention::spawn_sweeper`]) of one run: cancelled and
/// joined on drop, so a server that stops for any reason takes it along.
struct Sweeper {
    stop: CancellationToken,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Sweeper {
    fn spawn(store: Arc<crate::config_store::ConfigStore>, shutdown: &CancellationToken) -> Self {
        let stop = shutdown.child_token();
        let join = crate::retention::spawn_sweeper(
            Arc::new(move || {
                let (_, c) = store.snapshot();
                crate::retention::SweeperConfig {
                    save_image_dir: c.save_image_path.as_deref().map(crate::resolve_path),
                    max_age_days: c.save_image_max_age_days,
                    max_mb: c.save_image_max_mb,
                    data_root: c.data_root(),
                    min_free_mb: crate::retention::MIN_FREE_MB,
                }
            }),
            stop.clone(),
        );
        Self {
            stop,
            join: Some(join),
        }
    }
}

impl Drop for Sweeper {
    fn drop(&mut self) {
        self.stop.cancel();
        if let Some(j) = self.join.take()
            && j.join().is_err()
        {
            warn!("retention sweeper panicked");
        }
    }
}

/// What lives for the whole run (every generation shares it).
struct Process {
    started: Instant,
    config_path: PathBuf,
    log: LogReloadHandle,
    provisioner: Arc<Provisioner>,
    /// The address bound at process start (a changed port or address shows as pending until
    /// the process restarts).
    bound_address: std::net::IpAddr,
    bound_port: u16,
    /// One benchmark runner for the process, so its last results survive a restart.
    benchmark: Arc<crate::benchmark::service::BenchmarkService>,
    /// The live config (revision, change log) is shared by every generation; a restart that
    /// reloads the file (or restores the previous config) adopts it as a logged change.
    store: Arc<crate::config_store::ConfigStore>,
    /// Process-lifetime metrics: counters and histograms survive generation restarts.
    metrics: Arc<Metrics>,
}

impl Process {
    /// Start the registry of a generation (models compile in the background).
    fn start_registry(
        &self,
        config: &Config,
        workers: &CancellationToken,
        provision: &Arc<Provision>,
    ) -> Result<ModelRegistry> {
        ModelRegistry::start_with_benchmark(
            config,
            &self.metrics,
            workers.clone(),
            Some(provision),
            Some(&crate::benchmark::results_path(&self.config_path)),
        )
    }

    /// The `AppState` of a generation running `config` on `registry`.
    fn state_for(
        &self,
        config: &Config,
        registry: &Arc<ModelRegistry>,
        provision: &Arc<Provision>,
        generation: &CancellationToken,
        loaded_ort: Option<crate::resources::catalog::Flavor>,
    ) -> Arc<AppState> {
        Arc::new(AppState {
            registry: registry.clone(),
            metrics: self.metrics.clone(),
            config: self.store.clone(),
            running: arc_swap::ArcSwap::from_pointee({
                let mut c = config.clone();
                c.normalize();
                // What is running is the address bound at process start (a changed port or
                // address shows as pending until the process restarts).
                c.port = self.bound_port;
                c.bind_address = self.bound_address;
                c
            }),
            apply_lock: std::sync::Mutex::new(()),
            applied_revision: std::sync::atomic::AtomicU64::new(0),
            started: self.started,
            config_path: self.config_path.clone(),
            log_reload: Some(self.log.clone()),
            restart: generation.clone(),
            resources: Some(crate::resources::status::ResourcesCtx {
                provisioner: self.provisioner.clone(),
                provision: provision.clone(),
                openvino_loaded: registry
                    .runtimes()
                    .is_some_and(|rt| rt.lock_recover().openvino().is_some()),
                ort_loaded: loaded_ort,
            }),
            benchmark: self.benchmark.clone(),
        })
    }

    /// The config of the next generation after `POST /config/restart`: the config file again,
    /// or `current` when it cannot be read or has no models. Applies a changed log level and
    /// warns about the fields that need a process restart.
    fn reload_config(&self, current: &Config) -> Config {
        info!(config = %self.config_path.display(), "restarting: reloading the config file");
        let next = match Config::load(&self.config_path) {
            Ok(c) if !c.models.is_empty() => c,
            Ok(_) => {
                warn!("reloaded config has no models; keeping the current config");
                current.clone()
            }
            Err(e) => {
                warn!("{e:#}; keeping the current config");
                current.clone()
            }
        };
        if next.log_level != current.log_level
            && let Err(e) = self.log.set_level(next.log_level)
        {
            warn!("{e:#}");
        }
        if next.log_path != current.log_path {
            warn!("log_path changes take effect after the process is restarted");
        }
        if (next.port, next.bind_address) != (current.port, current.bind_address) {
            warn!(
                port = next.port,
                bind_address = %next.bind_address,
                "port and bind_address changes take effect after the process is restarted"
            );
        }
        next
    }
}

/// The HTTP side of a run: the listener (until the server starts), the server task, the
/// swappable current state, and the generation being replaced.
struct Run<'r> {
    rt: &'r tokio::runtime::Runtime,
    /// Bound once; taken when the server starts (it survives generation restarts).
    listeners: Option<server::Listeners>,
    /// Stops the HTTP server: on `shutdown`, or when the loop gives up.
    server_stop: CancellationToken,
    server_task: Option<tokio::task::JoinHandle<Result<()>>>,
    shared: Option<server::Shared>,
    /// The generation being replaced (still current until the next one is swapped in), or, on
    /// the way out, the last one.
    old: Option<Generation>,
}

impl Run<'_> {
    /// Make `state` current: swap it into the running server, or start the server with it.
    /// Errors only when the server has to start and the listener was already used (it is
    /// taken once, by the first generation).
    fn serve(&mut self, state: Arc<AppState>) -> Result<()> {
        match &self.shared {
            Some(sh) => {
                drop(sh.swap(state));
                info!("new generation is serving");
            }
            None => {
                let sh = server::Shared::new(state);
                let listeners = self
                    .listeners
                    .take()
                    .context("the HTTP listener was already used")?;
                self.server_task = Some(self.rt.spawn(server::serve_on(
                    listeners,
                    sh.clone(),
                    self.server_stop.clone(),
                )));
                self.shared = Some(sh);
            }
        }
        Ok(())
    }

    /// The replaced generation: in-flight requests finish (up to `DRAIN`), then its workers
    /// stop.
    fn retire_old(&mut self) {
        if let Some(g) = self.old.take() {
            g.retire(DRAIN);
        }
    }

    /// Serve until a restart (None: `generation` was cancelled) or the server task ending
    /// (Some: its result; the task is gone afterwards).
    fn serve_until_event(&mut self, generation: &CancellationToken) -> Option<Result<()>> {
        let Some(task) = self.server_task.as_mut() else {
            return Some(Err(anyhow::anyhow!("the HTTP server is not running")));
        };
        let ended = self.rt.block_on(async {
            tokio::select! {
                biased;
                _ = generation.cancelled() => None,
                r = task => Some(r.unwrap_or_else(|e| Err(anyhow::anyhow!("HTTP server task: {e}")))),
            }
        });
        if ended.is_some() {
            // A finished `JoinHandle` must not be awaited again.
            self.server_task = None;
        }
        ended
    }

    /// Stop the server and the last generation, and return `result`.
    fn finish(self, result: Result<()>) -> Result<()> {
        let Run {
            rt,
            server_stop,
            server_task,
            shared,
            old,
            ..
        } = self;
        server_stop.cancel();
        let served = server_task.map(|t| {
            rt.block_on(t)
                .unwrap_or_else(|e| Err(anyhow::anyhow!("HTTP server task: {e}")))
        });
        // The server (and its router) is gone; this was the last long-lived other holder of the
        // state (connection tasks may still be dropping theirs, see `SHUTDOWN_LINGER`).
        drop(shared);
        if let Some(g) = old {
            g.retire(SHUTDOWN_LINGER);
        }
        result.and(served.unwrap_or(Ok(())))
    }
}

/// [`run_server_on`] with a given [`Provisioner`] (tests inject catalog entries through it).
pub fn run_server_with(
    rt: &tokio::runtime::Runtime,
    mut config: Config,
    config_path: PathBuf,
    log: LogReloadHandle,
    shutdown: CancellationToken,
    provisioner: Provisioner,
) -> Result<()> {
    let provisioner = Arc::new(provisioner);
    ensure_models(&config, &config_path)?;
    let started = Instant::now();
    // The listener is bound once and survives generation restarts (a port change in the config
    // applies when the process restarts).
    let listeners = rt.block_on(server::bind(config.bind_address, config.port))?;
    // systemd `Type=notify`: the listener is up (models compile in the background; `/v1/status`
    // reports readiness). The watchdog pings while the listener accepts connections, so a hung
    // process is restarted but a model that compiles for minutes is not.
    crate::sdnotify::notify("READY=1");
    {
        let port = listeners.port();
        let ip = match config.bind_address {
            std::net::IpAddr::V4(a) if a.is_unspecified() => std::net::IpAddr::from([127, 0, 0, 1]),
            std::net::IpAddr::V6(a) if a.is_unspecified() => {
                std::net::IpAddr::from(std::net::Ipv6Addr::LOCALHOST)
            }
            other => other,
        };
        crate::sdnotify::spawn_watchdog(
            move || {
                std::net::TcpStream::connect_timeout(
                    &std::net::SocketAddr::new(ip, port),
                    Duration::from_secs(5),
                )
                .is_ok()
            },
            shutdown.clone(),
        );
    }
    let process = Process {
        started,
        bound_address: config.bind_address,
        bound_port: listeners.port(),
        store: Arc::new(crate::config_store::ConfigStore::new(
            config.clone(),
            config_path.clone(),
        )),
        config_path,
        log,
        provisioner,
        benchmark: crate::benchmark::service::BenchmarkService::new(),
        metrics: Arc::new(Metrics::new(crate::VERSION)),
    };
    // Disk retention (saved images, free-space warning): one sweeper per process, reading the
    // current settings on every tick; stopped and joined when this function returns.
    let _sweeper = Sweeper::spawn(process.store.clone(), &shutdown);
    let mut run = Run {
        rt,
        listeners: Some(listeners),
        server_stop: shutdown.child_token(),
        server_task: None,
        shared: None,
        old: None,
    };
    // Config to fall back to when a restart with the edited config fails to start.
    let mut previous: Option<Config> = None;
    // ONNX Runtime flavor loaded by an earlier generation (fixed for the process).
    let mut loaded_ort: Option<crate::resources::catalog::Flavor> = None;
    let mut adopt: Option<&'static str> = None;
    let mut first = true;
    loop {
        if !first {
            process.metrics.restarts.fetch_add(1, Ordering::Relaxed);
        }
        // One generation = one registry + one `AppState`, served through `shared`.
        // `POST /config/restart` (or an installed runtime) cancels the generation token; the
        // next generation is built while the old one keeps answering, then swapped in, and the
        // old workers stop once its requests have drained. Cancelling `shutdown` stops all.
        // The decode ceiling follows the generation's config (and every later config change,
        // see `ConfigStore`).
        crate::image::set_max_pixels(config.max_image_pixels);
        let generation = shutdown.child_token();
        let workers = shutdown.child_token();
        let provision = Arc::new(process.provisioner.prepare(&config, loaded_ort));
        let registry = match process.start_registry(&config, &workers, &provision) {
            Ok(r) => Arc::new(r),
            Err(e) => match previous.take() {
                Some(prev) => {
                    // The previous generation (if any) stays current meanwhile.
                    error!(
                        "restart with the new config failed: {e:#}; restoring the previous config"
                    );
                    config = prev;
                    adopt = Some("restart failed: previous config restored");
                    continue;
                }
                None => return run.finish(Err(e)),
            },
        };
        info!(models = ?registry.names(), "models registered (compiling in the background)");
        if loaded_ort.is_none() {
            loaded_ort = registry_ort_flavor(&registry);
        }
        let state = process.state_for(&config, &registry, &provision, &generation, loaded_ort);
        let resources_restart = Arc::new(AtomicBool::new(false));
        let watcher = spawn_watcher(
            provision.clone(),
            Arc::downgrade(&state),
            generation.clone(),
            resources_restart.clone(),
        );
        // The store's generation (and adopted config) changes together with the swap, so pages
        // waiting for the restart see the new generation's models.
        if !first {
            process
                .store
                .adopt(adopt.map(|_| config.clone()), adopt.unwrap_or("restart"));
        }
        first = false;
        adopt = None;
        if let Err(e) = run.serve(state.clone()) {
            run.retire_old();
            run.old = Some(Generation {
                state,
                registry,
                workers,
                watcher,
            });
            return run.finish(Err(e));
        }
        run.retire_old();
        let current = Generation {
            state,
            registry,
            workers,
            watcher,
        };

        // Serve until a restart, a shutdown, or the server failing.
        let server_ended = run.serve_until_event(&generation);
        // From here on `current` is the generation to retire (by the next one, or on the way
        // out).
        run.old = Some(current);
        if let Some(r) = server_ended {
            if shutdown.is_cancelled() {
                let r = run.finish(r);
                info!("bye");
                return r;
            }
            // The listener is gone: nothing left to serve on.
            let r = r.and(Err(anyhow::anyhow!("HTTP server stopped unexpectedly")));
            return run.finish(r);
        }
        // A benchmark still running uses this generation's runtimes: stop it.
        if process.benchmark.cancel() {
            info!("benchmark cancelled by the restart");
            process.benchmark.wait_idle(Duration::from_secs(30));
        }

        if shutdown.is_cancelled() {
            let r = run.finish(Ok(()));
            info!("bye");
            return r;
        }

        if resources_restart.load(Ordering::SeqCst) {
            info!("restarting the registry to load downloaded resources");
            previous = None;
            continue;
        }
        let next = process.reload_config(&config);
        previous = Some(std::mem::replace(&mut config, next));
        adopt = Some("restart (config file reloaded)");
    }
}

/// The ONNX Runtime flavor `registry` loaded, if any.
fn registry_ort_flavor(registry: &ModelRegistry) -> Option<crate::resources::catalog::Flavor> {
    let rt = registry.runtimes()?;
    let rt = rt.lock_recover();
    let ort = &rt.probe().ort;
    if !ort.is_available() {
        return None;
    }
    Some(
        ort.flavor
            .as_deref()
            .and_then(crate::resources::catalog::Flavor::parse)
            .unwrap_or(crate::resources::catalog::Flavor::Cpu),
    )
}

/// Watch the download manager during one generation and log every outcome. When a model file
/// is installed, the models of this generation that failed for want of it reload on their own
/// (per-model reload, [`server::reload_models`]); models waiting for it load by themselves.
/// When a runtime that this generation can use is installed, the generation restarts (sets
/// `restart`, cancels `generation`): runtime libraries are loaded once per generation, when
/// its `Runtimes` is built. The plan of the latest per-model reload (if any) supersedes
/// `provision`. None when nothing is being downloaded.
fn spawn_watcher(
    provision: Arc<Provision>,
    state: std::sync::Weak<AppState>,
    generation: CancellationToken,
    restart: Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<()>> {
    let manager = provision.manager()?.clone();
    let events = manager.subscribe();
    std::thread::Builder::new()
        .name("resources".into())
        .spawn(move || {
            loop {
                if generation.is_cancelled() {
                    return;
                }
                let ev = match events.recv_timeout(std::time::Duration::from_millis(250)) {
                    Ok(ev) => ev,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                };
                let text = crate::resources::provision::describe_event(&ev);
                let crate::resources::manager::Event::Installed { id, .. } = ev else {
                    warn!("{text}");
                    continue;
                };
                info!("{text}");
                let Some(state) = state.upgrade() else {
                    return;
                };
                let registry = state.registry.clone();
                let plan = registry
                    .latest_provision()
                    .unwrap_or_else(|| provision.clone());
                let Some(res) = plan
                    .resolution
                    .needs
                    .iter()
                    .map(|n| n.resource)
                    .find(|r| r.id == id)
                else {
                    continue;
                };
                if res.kind.is_model() {
                    // Failed models that wanted this file (e.g. fetched by hand with
                    // `auto_download` off): reload just them.
                    let failed: Vec<String> = registry
                        .workers()
                        .iter()
                        .filter(|w| matches!(w.state.get(), crate::startup::ModelState::Failed(_)))
                        .filter(|w| plan.blocks(&w.name).iter().any(|b| b.resource.id == id))
                        .map(|w| w.name.clone())
                        .collect();
                    if !failed.is_empty() {
                        info!(resource = id, models = ?failed, "model file installed; reloading");
                        server::reload_models(&state, &failed);
                    }
                    continue;
                }
                // Models not served on their planned device yet: waiting, failed, or loaded on
                // an interim device while their runtime downloaded.
                let not_ready: Vec<String> = registry
                    .workers()
                    .iter()
                    .filter(|w| !w.state.is_ready() || plan.is_blocked_on_runtime(&w.name))
                    .map(|w| w.name.clone())
                    .collect();
                drop(state);
                match plan.restart_for(res, &not_ready) {
                    Ok(()) => {
                        info!(
                            resource = id,
                            "runtime installed; starting a new registry generation to load it"
                        );
                        restart.store(true, Ordering::SeqCst);
                        generation.cancel();
                        return;
                    }
                    Err(why) => info!(resource = id, "no restart needed: {why}"),
                }
            }
        })
        .map_err(|e| warn!("could not spawn the resource watcher: {e}"))
        .ok()
}
