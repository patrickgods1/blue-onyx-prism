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
//! generation the same way `/config/restart` does, but keeps the in-memory config.
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
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

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
        match Arc::try_unwrap(registry) {
            Ok(reg) => reg.shutdown(),
            Err(_) => warn!("registry still referenced; workers stop via the shutdown token"),
        }
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
    let (bound_address, bound_port) = (config.bind_address, listeners.port());
    let mut listeners = Some(listeners);
    // Stops the HTTP server: on `shutdown`, or when the loop gives up.
    let server_stop = shutdown.child_token();
    let mut shared: Option<server::Shared> = None;
    let mut server_task: Option<tokio::task::JoinHandle<Result<()>>> = None;
    // The generation being replaced (still current until the next one is swapped in).
    let mut old: Option<Generation> = None;
    // Stop the server and the last generation, and return `result`.
    let finish = |server_task: Option<tokio::task::JoinHandle<Result<()>>>,
                  shared: Option<server::Shared>,
                  old: Option<Generation>,
                  result: Result<()>|
     -> Result<()> {
        server_stop.cancel();
        let served = server_task.map(|t| {
            rt.block_on(t)
                .unwrap_or_else(|e| Err(anyhow::anyhow!("HTTP server task: {e}")))
        });
        // The server (and its router) is gone; this was the last other holder of the state.
        drop(shared);
        if let Some(g) = old {
            g.retire(Duration::ZERO);
        }
        result.and(served.unwrap_or(Ok(())))
    };
    // Config to fall back to when a restart with the edited config fails to start.
    let mut previous: Option<Config> = None;
    // ONNX Runtime flavor loaded by an earlier generation (fixed for the process).
    let mut loaded_ort: Option<crate::resources::catalog::Flavor> = None;
    // One benchmark runner for the process, so its last results survive a restart.
    let benchmark = crate::benchmark::service::BenchmarkService::new();
    // The live config (revision, change log) is shared by every generation; a restart that
    // reloads the file (or restores the previous config) adopts it as a logged change.
    let store = Arc::new(crate::config_store::ConfigStore::new(
        config.clone(),
        config_path.clone(),
    ));
    let mut adopt: Option<&'static str> = None;
    let mut first = true;
    // Process-lifetime metrics: counters and histograms survive generation restarts.
    let metrics = Arc::new(Metrics::new(crate::VERSION));
    loop {
        if !first {
            metrics
                .restarts
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // One generation = one registry + one `AppState`, served through `shared`.
        // `POST /config/restart` (or an installed runtime) cancels the generation token; the
        // next generation is built while the old one keeps answering, then swapped in, and the
        // old workers stop once its requests have drained. Cancelling `shutdown` stops all.
        let generation = shutdown.child_token();
        let workers = shutdown.child_token();
        let provision = Arc::new(provisioner.prepare(&config, loaded_ort));
        let registry = match ModelRegistry::start_with_benchmark(
            &config,
            &metrics,
            workers.clone(),
            Some(&provision),
            Some(&crate::benchmark::results_path(&config_path)),
        ) {
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
                None => return finish(server_task, shared, old.take(), Err(e)),
            },
        };
        info!(models = ?registry.names(), "models registered (compiling in the background)");
        if loaded_ort.is_none() {
            loaded_ort = registry_ort_flavor(&registry);
        }
        let resources_restart = Arc::new(AtomicBool::new(false));
        let watcher = spawn_watcher(
            provision.clone(),
            registry.clone(),
            generation.clone(),
            resources_restart.clone(),
        );

        let state = Arc::new(AppState {
            registry: registry.clone(),
            metrics: metrics.clone(),
            config: store.clone(),
            running: Arc::new({
                let mut c = config.clone();
                c.normalize();
                // What is running is the address bound at process start (a changed port or
                // address shows as pending until the process restarts).
                c.port = bound_port;
                c.bind_address = bound_address;
                c
            }),
            started,
            config_path: config_path.clone(),
            log_reload: Some(log.clone()),
            restart: generation.clone(),
            resources: Some(crate::resources::status::ResourcesCtx {
                provisioner: provisioner.clone(),
                provision: provision.clone(),
                openvino_loaded: registry.runtimes().is_some_and(|rt| {
                    rt.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .openvino()
                        .is_some()
                }),
                ort_loaded: loaded_ort,
            }),
            benchmark: benchmark.clone(),
        });
        // The store's generation (and adopted config) changes together with the swap, so pages
        // waiting for the restart see the new generation's models.
        if !first {
            store.adopt(adopt.map(|_| config.clone()), adopt.unwrap_or("restart"));
        }
        first = false;
        adopt = None;
        match &shared {
            Some(sh) => {
                drop(sh.swap(state.clone()));
                info!("new generation is serving");
            }
            None => {
                let sh = server::Shared::new(state.clone());
                let listeners = listeners.take().expect("listeners are used once");
                server_task =
                    Some(rt.spawn(server::serve_on(listeners, sh.clone(), server_stop.clone())));
                shared = Some(sh);
            }
        }
        // The replaced generation: in-flight requests finish (up to `DRAIN`), then its workers
        // stop.
        if let Some(g) = old.take() {
            g.retire(DRAIN);
        }
        let current = Generation {
            state,
            registry,
            workers,
            watcher,
        };

        // Serve until a restart, a shutdown, or the server failing.
        let server_failed = rt.block_on(async {
            let task = server_task.as_mut().expect("server task is running");
            tokio::select! {
                biased;
                _ = generation.cancelled() => None,
                r = task => Some(r.unwrap_or_else(|e| Err(anyhow::anyhow!("HTTP server task: {e}")))),
            }
        });
        if let Some(r) = server_failed {
            if shutdown.is_cancelled() {
                let r = finish(None, shared, Some(current), r);
                info!("bye");
                return r;
            }
            // The listener is gone: nothing left to serve on.
            let r = r.and(Err(anyhow::anyhow!("HTTP server stopped unexpectedly")));
            return finish(None, shared, Some(current), r);
        }
        // A benchmark still running uses this generation's runtimes: stop it.
        if benchmark.cancel() {
            info!("benchmark cancelled by the restart");
            benchmark.wait_idle(Duration::from_secs(30));
        }

        if shutdown.is_cancelled() {
            let r = finish(server_task.take(), shared, Some(current), Ok(()));
            info!("bye");
            return r;
        }
        old = Some(current);

        if resources_restart.load(Ordering::SeqCst) {
            info!("restarting the registry to load downloaded resources");
            previous = None;
            continue;
        }
        info!(config = %config_path.display(), "restarting: reloading the config file");
        let next = match Config::load(&config_path) {
            Ok(c) if !c.models.is_empty() => c,
            Ok(_) => {
                warn!("reloaded config has no models; keeping the current config");
                config.clone()
            }
            Err(e) => {
                warn!("{e:#}; keeping the current config");
                config.clone()
            }
        };
        if next.log_level != config.log_level
            && let Err(e) = log.set_level(next.log_level)
        {
            warn!("{e:#}");
        }
        if next.log_path != config.log_path {
            warn!("log_path changes take effect after the process is restarted");
        }
        if (next.port, next.bind_address) != (config.port, config.bind_address) {
            warn!(
                port = next.port,
                bind_address = %next.bind_address,
                "port and bind_address changes take effect after the process is restarted"
            );
        }
        previous = Some(std::mem::replace(&mut config, next));
        adopt = Some("restart (config file reloaded)");
    }
}

/// The ONNX Runtime flavor `registry` loaded, if any.
fn registry_ort_flavor(registry: &ModelRegistry) -> Option<crate::resources::catalog::Flavor> {
    let rt = registry.runtimes()?;
    let rt = rt.lock().unwrap_or_else(|e| e.into_inner());
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

/// Watch the download manager during one generation: log every outcome and, when a runtime
/// that this generation can use is installed, restart it (sets `restart`, cancels
/// `generation`). None when nothing is being downloaded.
fn spawn_watcher(
    provision: Arc<Provision>,
    registry: Arc<ModelRegistry>,
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
                let Some(res) = provision
                    .resolution
                    .needs
                    .iter()
                    .map(|n| n.resource)
                    .find(|r| r.id == id)
                else {
                    continue;
                };
                // Models not served on their planned device yet: waiting, failed, or loaded on
                // an interim device while their runtime downloaded.
                let not_ready: Vec<String> = registry
                    .workers()
                    .iter()
                    .filter(|w| !w.state.is_ready() || provision.is_blocked_on_runtime(&w.name))
                    .map(|w| w.name.clone())
                    .collect();
                match provision.restart_for(res, &not_ready) {
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
