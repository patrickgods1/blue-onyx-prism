//! Windows service host for Blue Onyx Prism (`BlueOnyxPrismService`).
//!
//! Install with `scripts/install_service.ps1`. The service reads
//! `blue_onyx_prism_config_service.json` next to the executable and runs the same server loop
//! as the CLI binary ([`blue_onyx_prism::runner`]). Controls: Stop / Shutdown, Interrogate,
//! and the user-defined codes 130 (stop) and 131 (restart: cancel the running server, reload the
//! service config, start again), e.g. `sc.exe control BlueOnyxPrismService 131`.
//!
//! Logs go to the configured `log_path` (daily rolling file; stdout is discarded for services)
//! and, at INFO and above, to the Application event log under the source `BlueOnyxPrism`.
//! Modeled on the blue-onyx service binary (MIT).

#![cfg_attr(
    not(test),
    warn(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::large_futures,
        clippy::unused_async,
        clippy::dbg_macro,
        missing_debug_implementations
    )
)]
#![deny(unsafe_code)]

/// `--version` / `-V`: version, git sha and build date (the binary takes no other arguments).
fn version_requested() -> bool {
    std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
}

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    if version_requested() {
        println!(
            "blue-onyx-prism-service {}",
            blue_onyx_prism::cli::long_version_str()
        );
        return std::process::ExitCode::SUCCESS;
    }
    service::main()
}

#[cfg(not(windows))]
fn main() -> std::process::ExitCode {
    if version_requested() {
        println!(
            "blue-onyx-prism-service {}",
            blue_onyx_prism::cli::long_version_str()
        );
        return std::process::ExitCode::SUCCESS;
    }
    eprintln!(
        "blue-onyx-prism-service is a Windows service binary and only runs on Windows. \
         On Linux/macOS run `blue-onyx-prism` under systemd or launchd (see deploy/)."
    );
    std::process::ExitCode::FAILURE
}

#[cfg(windows)]
mod service {
    use anyhow::Result;
    use blue_onyx_prism::backend::libs;
    use blue_onyx_prism::cli::{self, ExtraLogLayer, LogReloadHandle};
    use blue_onyx_prism::config::{Config, LogFormat, LogLevel};
    use blue_onyx_prism::sync::LockExt;
    use blue_onyx_prism::{resolve_path, runner, system_info};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use tracing::{debug, error, info, warn};
    use tracing_subscriber::Layer;
    use tracing_subscriber::filter::LevelFilter;
    use windows_service::service::{
        ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
        ServiceType,
    };
    use windows_service::service_control_handler::{
        self, ServiceControlHandlerResult, ServiceStatusHandle,
    };
    use windows_service::{define_windows_service, service_dispatcher};

    pub const SERVICE_NAME: &str = "BlueOnyxPrismService";
    /// Application event log source; created by `scripts/install_service.ps1`.
    pub const EVENT_SOURCE: &str = "BlueOnyxPrism";
    const SERVICE_TYPE: ServiceType = ServiceType::OWN_PROCESS;
    /// User-defined control code: stop the service.
    const CONTROL_STOP: u32 = 130;
    /// User-defined control code: restart the server with a freshly loaded service config.
    const CONTROL_RESTART: u32 = 131;
    /// Generous: the first GPU compile of several models can take minutes.
    const START_WAIT_HINT: Duration = Duration::from_secs(600);
    const STOP_WAIT_HINT: Duration = Duration::from_secs(30);
    const RETRY_DELAY: Duration = Duration::from_secs(5);
    /// Service-specific exit code: the server did not stop within `runner::SHUTDOWN_DEADLINE`.
    const EXIT_STOP_TIMED_OUT: u32 = 2;

    define_windows_service!(ffi_service_main, service_main);

    pub fn main() -> ExitCode {
        match service_dispatcher::start(SERVICE_NAME, ffi_service_main) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!(
                    "could not connect to the Service Control Manager: {e}. This binary is started \
                     by Windows as the '{SERVICE_NAME}' service (install it with \
                     scripts/install_service.ps1); for a console server run blue-onyx-prism.exe."
                );
                ExitCode::FAILURE
            }
        }
    }

    /// Signals from the control handler (and the server thread) to the service main thread.
    enum Event {
        Stop,
        ServerExited,
    }

    fn status(state: ServiceState, wait_hint: Duration, checkpoint: u32) -> ServiceStatus {
        let controls_accepted = match state {
            ServiceState::Running => ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
            _ => ServiceControlAccept::empty(),
        };
        ServiceStatus {
            service_type: SERVICE_TYPE,
            current_state: state,
            controls_accepted,
            exit_code: ServiceExitCode::NO_ERROR,
            checkpoint,
            wait_hint,
            process_id: None,
        }
    }

    fn set_status(handle: &ServiceStatusHandle, s: ServiceStatus) {
        if let Err(e) = handle.set_service_status(s) {
            warn!("SetServiceStatus failed: {e}");
        }
    }

    /// The event log layer, or the reason it could not be opened (logged once the file log is
    /// up: a service's stderr is discarded, and without this nobody learns why the event log
    /// stays empty).
    fn event_log_layer() -> Result<ExtraLogLayer, String> {
        match tracing_layer_win_eventlog::EventLogLayer::new(EVENT_SOURCE) {
            Ok(layer) => Ok(Box::new(layer.with_filter(LevelFilter::INFO))),
            Err(e) => Err(format!(
                "could not open the event log source {EVENT_SOURCE}: {e}; the Application event \
                 log gets nothing from this service (run scripts/install_service.ps1 to create \
                 the source); the file log is unaffected"
            )),
        }
    }

    /// File log (when configured, in `format`) + event log. A log directory that cannot be
    /// created falls back to stdout (discarded for a service) with a warning that also reaches
    /// the event log (`cli::init_logging_opts`).
    fn init_logging(
        level: LogLevel,
        log_path: Option<PathBuf>,
        format: LogFormat,
    ) -> Result<LogReloadHandle> {
        let (layer, problem) = match event_log_layer() {
            Ok(l) => (Some(l), None),
            Err(why) => {
                eprintln!("{why}");
                (None, Some(why))
            }
        };
        let log = cli::init_logging_opts(level, log_path.as_deref(), format, layer)?;
        if let Some(why) = problem {
            warn!("{why}");
        }
        Ok(log)
    }

    fn service_main(_arguments: Vec<OsString>) {
        // Service working directory is System32: every path is resolved against the exe dir.
        let first = cli::for_service();
        let (level, log_path, format) = match &first {
            Ok((c, _)) => (
                c.log_level,
                c.log_path.as_deref().map(resolve_path),
                c.log_format,
            ),
            Err(_) => (LogLevel::Info, None, LogFormat::Text),
        };
        let log = match init_logging(level, log_path, format) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{e:#}");
                return;
            }
        };
        if let Ok((config, _)) = &first {
            cli::log_startup_warnings(config);
        }
        if let Err(e) = run_service(first, log) {
            error!("service failed: {e:#}");
        }
    }

    fn run_service(
        first: Result<(Config, PathBuf)>,
        log: LogReloadHandle,
    ) -> windows_service::Result<()> {
        let stop = CancellationToken::new();
        // Token of the running server attempt; restart (131) cancels just this one.
        let current = Arc::new(Mutex::new(stop.child_token()));
        let restart_requested = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Event>();

        let handler = {
            let stop = stop.clone();
            let current = current.clone();
            let restart_requested = restart_requested.clone();
            let tx = tx.clone();
            move |control: ServiceControl| -> ServiceControlHandlerResult {
                match control {
                    ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
                    ServiceControl::Stop | ServiceControl::Shutdown => {
                        info!("stop requested by the service control manager");
                        stop.cancel();
                        let _ = tx.send(Event::Stop);
                        ServiceControlHandlerResult::NoError
                    }
                    ServiceControl::UserEvent(code) if code.to_raw() == CONTROL_STOP => {
                        info!("stop requested (control code {CONTROL_STOP})");
                        stop.cancel();
                        let _ = tx.send(Event::Stop);
                        ServiceControlHandlerResult::NoError
                    }
                    ServiceControl::UserEvent(code) if code.to_raw() == CONTROL_RESTART => {
                        info!("restart requested (control code {CONTROL_RESTART})");
                        restart_requested.store(true, Ordering::SeqCst);
                        current.lock_recover().cancel();
                        ServiceControlHandlerResult::NoError
                    }
                    _ => ServiceControlHandlerResult::NotImplemented,
                }
            }
        };
        let handle = service_control_handler::register(SERVICE_NAME, handler)?;
        set_status(
            &handle,
            status(ServiceState::StartPending, START_WAIT_HINT, 1),
        );

        info!(
            version = blue_onyx_prism::VERSION,
            git_sha = blue_onyx_prism::GIT_SHA,
            build_date = blue_onyx_prism::BUILD_DATE,
            os = %system_info::os_description(),
            cpu = %system_info::cpu_name(),
            memory_gb = ?system_info::total_memory_gb(),
            config = %Config::service_config_path().display(),
            "starting Blue Onyx Prism service {}",
            blue_onyx_prism::long_version()
        );
        // Set OPENVINO_INSTALL_DIR (and the DLL search path) before any runtime/worker thread
        // exists; `OvCore::new` calls it again per generation and then leaves it untouched.
        // `openvino_dir`, else `<download_dir>/openvino` (installs go under the install dir or
        // `download_dir`, never %TEMP%).
        let openvino_dir = first
            .as_ref()
            .ok()
            .and_then(|(c, _)| c.openvino_dir_effective());
        libs::prepare_environment(openvino_dir.as_deref());

        let server = {
            let stop = stop.clone();
            let tx = tx.clone();
            std::thread::Builder::new()
                .name("server".into())
                .spawn(move || {
                    server_loop(first, log, stop, current, restart_requested);
                    let _ = tx.send(Event::ServerExited);
                })
        };
        let server = match server {
            Ok(s) => s,
            Err(e) => {
                error!("could not spawn the server thread: {e}");
                let mut s = status(ServiceState::Stopped, Duration::ZERO, 0);
                s.exit_code = ServiceExitCode::ServiceSpecific(1);
                set_status(&handle, s);
                return Ok(());
            }
        };

        // HTTP is served while models compile, so the service counts as running right away.
        set_status(&handle, status(ServiceState::Running, Duration::ZERO, 0));
        let exit = match rx.recv() {
            Ok(Event::Stop) | Err(_) => ServiceExitCode::NO_ERROR,
            Ok(Event::ServerExited) => {
                error!("server thread exited unexpectedly");
                ServiceExitCode::ServiceSpecific(1)
            }
        };
        stop.cancel();

        // Workers finish their current inference (or compile) before exiting; keep the SCM
        // informed while we wait, at most `runner::SHUTDOWN_DEADLINE` (CO-12: a GPU compile
        // can take minutes). Past it the service reports Stopped with
        // `EXIT_STOP_TIMED_OUT` and the process ends with the server thread still running.
        let deadline = std::time::Instant::now() + runner::SHUTDOWN_DEADLINE;
        let mut checkpoint = 1;
        while !server.is_finished() && std::time::Instant::now() < deadline {
            set_status(
                &handle,
                status(ServiceState::StopPending, STOP_WAIT_HINT, checkpoint),
            );
            checkpoint += 1;
            std::thread::sleep(Duration::from_millis(500));
        }
        let exit = if server.is_finished() {
            if server.join().is_err() {
                error!("server thread panicked");
            }
            info!("service stopped");
            exit
        } else {
            error!(
                "the server did not stop within {} s (a model compile or a request is still \
                 running); reporting the service as stopped anyway. If this repeats, check the \
                 log for the model that was loading",
                runner::SHUTDOWN_DEADLINE.as_secs()
            );
            ServiceExitCode::ServiceSpecific(EXIT_STOP_TIMED_OUT)
        };
        let mut s = status(ServiceState::Stopped, Duration::ZERO, 0);
        s.exit_code = exit;
        set_status(&handle, s);
        Ok(())
    }

    /// Run the server until `stop`; on failure log and retry after [`RETRY_DELAY`]; on restart
    /// (131) reload the service config and start again immediately.
    fn server_loop(
        first: Result<(Config, PathBuf)>,
        log: LogReloadHandle,
        stop: CancellationToken,
        current: Arc<Mutex<CancellationToken>>,
        restart_requested: Arc<AtomicBool>,
    ) {
        let rt = match runner::build_runtime() {
            Ok(rt) => rt,
            Err(e) => {
                error!("{e:#}");
                return;
            }
        };
        let mut loaded = Some(first);
        let mut active_level: Option<LogLevel> = None;
        let mut last_error: Option<String> = None;
        while !stop.is_cancelled() {
            let attempt = stop.child_token();
            *current.lock_recover() = attempt.clone();
            restart_requested.store(false, Ordering::SeqCst);

            let result = match loaded.take().unwrap_or_else(cli::for_service) {
                Ok((config, path)) => {
                    if active_level.is_some_and(|l| l != config.log_level)
                        && let Err(e) = log.set_level(config.log_level)
                    {
                        warn!("{e:#}");
                    }
                    active_level = Some(config.log_level);
                    runner::run_server_on(&rt, config, path, log.clone(), attempt.clone())
                }
                Err(e) => Err(e),
            };
            if stop.is_cancelled() {
                break;
            }
            match result {
                Ok(()) => {
                    // Only a cancelled attempt ends the server loop without error.
                    info!("restarting: reloading the service config");
                    last_error = None;
                    continue;
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    // Avoid flooding the event log with the same error every few seconds.
                    if last_error.as_deref() == Some(msg.as_str()) {
                        debug!("server still failing: {msg}");
                    } else {
                        error!(
                            "server failed: {msg}; retrying every {} s",
                            RETRY_DELAY.as_secs()
                        );
                        last_error = Some(msg);
                    }
                }
            }
            if restart_requested.load(Ordering::SeqCst) {
                continue;
            }
            // Wait before retrying; a stop or restart request ends the wait early.
            rt.block_on(async {
                tokio::select! {
                    _ = attempt.cancelled() => {}
                    _ = tokio::time::sleep(RETRY_DELAY) => {}
                }
            });
        }
        // Blocking tasks get a bounded grace period (CO-12).
        rt.shutdown_timeout(runner::RUNTIME_SHUTDOWN_GRACE);
    }
}
