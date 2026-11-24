//! Windows service host for Blue Onyx Prism (`BlueOnyxPrismService`).
//!
//! Installed by the MSI or `scripts/install_service.ps1`. The service reads the config named by
//! `--config <path>` (or `--config=<path>`) on its command line (the service ImagePath; a
//! relative path resolves against the exe dir), else `blue_onyx_prism_config_service.json` next
//! to the executable, and runs the same server loop as the CLI binary
//! ([`blue_onyx_prism::runner`]). A config created outside the exe dir keeps its downloads, cache
//! and logs next to it ([`blue_onyx_prism::cli::service_default_config`]).
//!
//! Controls: Stop / Preshutdown, Interrogate, and the user-defined codes 130 (stop) and 131 (restart: cancel the running server, reload the
//! service config, start again), e.g. `sc.exe control BlueOnyxPrismService 131`.
//!
//! Stop: the server follows the runner's stop timeline (`runner`, "Stopping") while this thread
//! reports StopPending every 500 ms, for at most `runner::SHUTDOWN_DEADLINE` (60 s); past it the
//! service ends the child processes it recorded, writes the log's tail and reports Stopped.
//! The service accepts PRESHUTDOWN (not SHUTDOWN, which gets only the ~5 s
//! WaitToKillServiceTimeout at an OS restart), so an OS shutdown gets the same stop: the SCM waits
//! up to the service's preshutdown timeout, which `blue-onyx-prism-service.exe
//! --set-preshutdown-timeout` (run by `scripts/install_service.ps1`, elevated) sets to
//! `runner::SERVICE_MANAGER_STOP` (75 s; Windows' default is 10 s).
//!
//! Start: StartPending (the checkpoint moves every second) until the HTTP listener is bound, then
//! Running (models compile while the service runs). When no listener could be bound for
//! `runner::BIND_GIVE_UP` (60 s; another program holds the port, or the config cannot be read)
//! the service stops with a service-specific exit code (3: port in use, 4: other start failure),
//! logging what holds the port, so the SCM's recovery actions apply. A panicking server thread
//! stops it with exit code 1 (the panic hook logs the message, [`blue_onyx_prism::panic`]).
//!
//! Logs go to the configured `log_path` (daily rolling file; stdout is discarded for services)
//! and, at INFO and above, to the Application event log under the source `BlueOnyxPrism`
//! ([`event_log_gate`]: a model export's child-process output never goes there, and an INFO or
//! WARN line reaches it at most once per [`event_log_gate::EVERY`] per call site; ERROR always
//! does; the file log keeps every line).
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

/// `--version` / `-V`: version, git sha and build date. The other arguments are
/// `--config <path>` / `--config=<path>` ([`blue_onyx_prism::cli::service_config_arg`]) and
/// [`SET_PRESHUTDOWN`].
fn version_requested() -> bool {
    std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
}

/// Install-time argument (elevated, not as the service): set the installed service's
/// preshutdown timeout to `runner::SERVICE_MANAGER_STOP` and exit.
#[cfg(windows)]
const SET_PRESHUTDOWN: &str = "--set-preshutdown-timeout";

#[cfg(windows)]
fn set_preshutdown_requested() -> bool {
    std::env::args().skip(1).any(|a| a == SET_PRESHUTDOWN)
}

/// What of the service's log reaches the Application event log (audit RA-118). Each line costs
/// a synchronous `ReportEventW` and lands in the log every other program shares, so lines that
/// repeat (a lasting fault, a child process's progress) must not flood it: the file log keeps
/// all of them. Portable so it is tested on every OS; only the Windows service uses it.
#[cfg(any(windows, test))]
mod event_log_gate {
    use blue_onyx_prism::logthrottle::Throttle;
    use blue_onyx_prism::sync::LockExt;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;
    use tracing::callsite::Identifier;
    use tracing::subscriber::Interest;
    use tracing::{Event, Level, Metadata};
    use tracing_subscriber::filter::{LevelFilter, Targets};
    use tracing_subscriber::layer::Context;

    /// Target of a model export's child-process output lines (`resources::export::exporter`):
    /// a uv install and the Python export print hundreds, each `\r` redraw of a progress bar
    /// counting as one. Kept in the export's in-memory tail and the file log at debug.
    pub const EXPORT_CHILD_TARGET: &str = "blue_onyx_prism::resources::export::child";

    /// Smallest gap between two INFO / WARN event log records of one call site. The crate's
    /// throttle interval for per-request lines ([`blue_onyx_prism::logthrottle::DEFAULT_EVERY`]).
    pub const EVERY: Duration = blue_onyx_prism::logthrottle::DEFAULT_EVERY;

    /// Call sites throttled one by one. Call sites are static (a few thousand in the binary),
    /// so the map is bounded by the code; past this the rest share one throttle.
    const MAX_CALLSITES: usize = 4096;

    /// The event log layer's filter: INFO and above, nothing of [`EXPORT_CHILD_TARGET`], and
    /// per call site at most one INFO / WARN record per [`EVERY`]. ERROR always passes (one
    /// shared `error!` site can carry distinct failures, e.g. one per model).
    #[derive(Debug)]
    pub struct Gate {
        targets: Targets,
        sites: Mutex<HashMap<Identifier, Throttle>>,
        overflow: Throttle,
        /// Milliseconds on a monotonic clock (the process clock; tests step a fake).
        clock: fn() -> u64,
    }

    impl Gate {
        /// The service's gate, on the process clock.
        #[cfg(windows)]
        pub fn new() -> Self {
            Self::with_clock(blue_onyx_prism::logthrottle::process_ms)
        }

        pub fn with_clock(clock: fn() -> u64) -> Self {
            Self {
                targets: Targets::new()
                    .with_default(LevelFilter::INFO)
                    .with_target(EXPORT_CHILD_TARGET, LevelFilter::OFF),
                sites: Mutex::new(HashMap::new()),
                overflow: Throttle::new(EVERY),
                clock,
            }
        }

        /// The static part: level and target.
        pub fn wants(&self, meta: &Metadata<'_>) -> bool {
            self.targets.would_enable(meta.target(), meta.level())
        }

        /// Whether this event of `meta`'s call site goes to the event log at `now_ms`.
        pub fn passes(&self, meta: &Metadata<'_>, now_ms: u64) -> bool {
            if !self.wants(meta) {
                return false;
            }
            if *meta.level() == Level::ERROR {
                return true;
            }
            let mut sites = self.sites.lock_recover();
            let full = sites.len() >= MAX_CALLSITES;
            let id = meta.callsite();
            if let Some(t) = sites.get(&id) {
                return t.allow(now_ms).is_some();
            }
            if full {
                return self.overflow.allow(now_ms).is_some();
            }
            let t = Throttle::new(EVERY);
            let pass = t.allow(now_ms).is_some();
            sites.insert(id, t);
            pass
        }
    }

    impl<S> tracing_subscriber::layer::Filter<S> for Gate {
        fn enabled(&self, meta: &Metadata<'_>, _: &Context<'_, S>) -> bool {
            self.wants(meta)
        }

        fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> Interest {
            // `sometimes`, not `always`: the throttle decides per event in `event_enabled`, and
            // tracing-subscriber expects a per-layer filter's `enabled` pass before it (with
            // `always` it skips that pass and its filter state check fails).
            if self.wants(meta) {
                Interest::sometimes()
            } else {
                Interest::never()
            }
        }

        fn event_enabled(&self, event: &Event<'_>, _: &Context<'_, S>) -> bool {
            self.passes(event.metadata(), (self.clock)())
        }

        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(LevelFilter::INFO)
        }
    }
}

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    blue_onyx_prism::panic::install();
    // Before any thread starts (Linux glibc: arena cap, fixed mmap threshold).
    blue_onyx_prism::backend::memory::tune_allocator();
    if version_requested() {
        println!(
            "blue-onyx-prism-service {}",
            blue_onyx_prism::cli::long_version_str()
        );
        return std::process::ExitCode::SUCCESS;
    }
    if set_preshutdown_requested() {
        return service::set_preshutdown_timeout();
    }
    service::main()
}

#[cfg(not(windows))]
fn main() -> std::process::ExitCode {
    blue_onyx_prism::panic::install();
    // Before any thread starts (Linux glibc: arena cap, fixed mmap threshold).
    blue_onyx_prism::backend::memory::tune_allocator();
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
    use blue_onyx_prism::panic::ExitSignal;
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
    /// Wait hint of each StartPending report. The checkpoint moves every [`START_TICK`] until
    /// the HTTP listener is bound (models compile after that, while the service is Running).
    const START_WAIT_HINT: Duration = Duration::from_secs(10);
    /// How often StartPending is reported again with the next checkpoint.
    const START_TICK: Duration = Duration::from_secs(1);
    /// Wait hint of each StopPending report (the checkpoint moves every 500 ms; the whole stop is
    /// bounded by `runner::SHUTDOWN_DEADLINE`).
    const STOP_WAIT_HINT: Duration = Duration::from_secs(30);
    const RETRY_DELAY: Duration = Duration::from_secs(5);
    /// Service-specific exit code: the server thread ended unexpectedly (a panic, logged with
    /// its message by the panic hook, or a runtime that could not be built).
    const EXIT_SERVER_EXITED: u32 = 1;
    /// Service-specific exit code: the server did not stop within `runner::SHUTDOWN_DEADLINE`.
    const EXIT_STOP_TIMED_OUT: u32 = 2;
    /// Service-specific exit code: the HTTP port could not be bound for
    /// `runner::BIND_GIVE_UP` (another program holds it).
    const EXIT_PORT_IN_USE: u32 = 3;
    /// Service-specific exit code: the server could not start for `runner::BIND_GIVE_UP`
    /// (e.g. an unreadable config) and never got to listen.
    const EXIT_START_FAILED: u32 = 4;

    define_windows_service!(ffi_service_main, service_main);

    pub fn main() -> ExitCode {
        match service_dispatcher::start(SERVICE_NAME, ffi_service_main) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!(
                    "could not connect to the Service Control Manager: {e}. This binary is started \
                     by Windows as the '{SERVICE_NAME}' service (install it with the Blue Onyx \
                     Prism MSI or scripts/install_service.ps1); for a console server run \
                     blue-onyx-prism.exe."
                );
                ExitCode::FAILURE
            }
        }
    }

    /// Signals from the control handler (and the server thread) to the service main thread.
    enum Event {
        Stop,
        /// The HTTP listener is bound: the service is Running.
        Bound,
        /// The server thread ended (sent on return and, by [`ExitSignal`], on a panic).
        ServerExited(ServerExit),
    }

    /// Why the server thread ended.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ServerExit {
        /// A stop was requested.
        Stopped,
        /// It panicked (the message is in the log, written by the panic hook) or could not
        /// build its runtime.
        Unexpected,
        /// The port could not be bound for `runner::BIND_GIVE_UP`.
        PortInUse,
        /// Startup kept failing for `runner::BIND_GIVE_UP` before the port was bound.
        StartFailed,
    }

    impl ServerExit {
        fn exit_code(self) -> ServiceExitCode {
            match self {
                ServerExit::Stopped => ServiceExitCode::NO_ERROR,
                ServerExit::Unexpected => ServiceExitCode::ServiceSpecific(EXIT_SERVER_EXITED),
                ServerExit::PortInUse => ServiceExitCode::ServiceSpecific(EXIT_PORT_IN_USE),
                ServerExit::StartFailed => ServiceExitCode::ServiceSpecific(EXIT_START_FAILED),
            }
        }
    }

    /// `--set-preshutdown-timeout`: the SCM waits this long at an OS shutdown for the service to
    /// stop (ChangeServiceConfig2, takes effect at once). Run elevated by the installer.
    pub fn set_preshutdown_timeout() -> ExitCode {
        use windows_service::service::ServiceAccess;
        use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};
        let timeout = runner::SERVICE_MANAGER_STOP;
        let done = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .and_then(|m| m.open_service(SERVICE_NAME, ServiceAccess::CHANGE_CONFIG))
            .and_then(|svc| svc.set_preshutdown_timeout(timeout));
        match done {
            Ok(()) => {
                println!(
                    "{SERVICE_NAME}: preshutdown timeout set to {} s",
                    timeout.as_secs()
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!(
                    "could not set the preshutdown timeout of {SERVICE_NAME}: {e}. Run this from \
                     an administrator prompt after the service is installed; without it an OS \
                     shutdown gives the service Windows' default 10 s to stop"
                );
                ExitCode::FAILURE
            }
        }
    }

    fn status(state: ServiceState, wait_hint: Duration, checkpoint: u32) -> ServiceStatus {
        // PRESHUTDOWN and SHUTDOWN are mutually exclusive: preshutdown gives the stop the
        // service's preshutdown timeout instead of the ~5 s an OS shutdown allows otherwise.
        let controls_accepted = match state {
            ServiceState::Running => ServiceControlAccept::STOP | ServiceControlAccept::PRESHUTDOWN,
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
            Ok(layer) => Ok(Box::new(
                layer.with_filter(crate::event_log_gate::Gate::new()),
            )),
            Err(e) => Err(format!(
                "could not open the event log source {EVENT_SOURCE}: {e}; the Application event \
                 log gets nothing from this service (reinstall with the MSI or run \
                 scripts/install_service.ps1 to create the source); the file log is unaffected"
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

    /// Load (or create) the service config at `path`; a bad `--config` argument fails every
    /// attempt with the same error (the retry loop logs it once).
    fn load_config(path: &std::result::Result<PathBuf, String>) -> Result<(Config, PathBuf)> {
        match path {
            Ok(p) => cli::for_service_at(p).map(|c| (c, p.clone())),
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    }

    fn service_main(_arguments: Vec<OsString>) {
        // Service working directory is System32: every path is resolved against the exe dir.
        // `--config` comes from the process args (the ImagePath); SCM start arguments are ignored.
        let config_path = cli::service_config_path_from_args().map_err(|e| format!("{e:#}"));
        let first = load_config(&config_path);
        let (level, log_path, format) = match &first {
            Ok((c, _)) => {
                // Log lines, file names and pages follow the config's zone from the first line.
                blue_onyx_prism::clock::set(c.timezone);
                (
                    c.log_level,
                    c.log_path.as_deref().map(resolve_path),
                    c.log_format,
                )
            }
            Err(_) => (LogLevel::Info, None, LogFormat::Text),
        };
        let log = match init_logging(level, log_path, format) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{e:#}");
                return;
            }
        };
        if let Ok((config, path)) = &first {
            cli::log_startup_warnings(config, path);
        }
        if let Err(e) = run_service(first, config_path, log) {
            error!("service failed: {e:#}");
        }
        // The file log's last lines (RA-024): the process may end right after this returns.
        cli::flush_logs();
    }

    fn run_service(
        first: Result<(Config, PathBuf)>,
        config_path: std::result::Result<PathBuf, String>,
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
                    ServiceControl::Stop
                    | ServiceControl::Preshutdown
                    | ServiceControl::Shutdown => {
                        info!("stop requested by the service control manager ({control:?})");
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
            config = %config_path
                .as_ref()
                .map_or_else(|e| format!("invalid --config ({e})"), |p| p.display().to_string()),
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
            // The CLI's stack for the same server path (`main_thread`): a debug build needs
            // more than the 1 MiB that `service_main`'s dispatcher thread has, which its own
            // config and log setup (about 256 KiB measured) fit in.
            std::thread::Builder::new()
                .name("server".into())
                .stack_size(blue_onyx_prism::main_thread::STACK_BYTES)
                .spawn(move || {
                    // Sent on return and on a panic (unwinding drops it).
                    let signal =
                        ExitSignal::new(tx.clone(), Event::ServerExited(ServerExit::Unexpected));
                    let exit = server_loop(
                        first,
                        config_path,
                        log,
                        stop,
                        current,
                        restart_requested,
                        tx,
                    );
                    signal.finish(Event::ServerExited(exit));
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

        // StartPending (checkpoint moving every second) until the HTTP listener is bound, then
        // Running: HTTP is served while models compile. A port that stays taken ends the
        // server thread after `runner::BIND_GIVE_UP` (`ServerExit::PortInUse`), so the SCM sees
        // a failed start and its recovery actions apply.
        let mut running = false;
        let mut checkpoint = 1;
        let exit = loop {
            match rx.recv_timeout(START_TICK) {
                Ok(Event::Bound) => {
                    if !running {
                        set_status(&handle, status(ServiceState::Running, Duration::ZERO, 0));
                        info!("service running");
                        running = true;
                    }
                }
                Ok(Event::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break ServiceExitCode::NO_ERROR;
                }
                Ok(Event::ServerExited(why)) => {
                    if why == ServerExit::Unexpected {
                        error!(
                            "the server thread ended unexpectedly (the error or panic before \
                             this line says why); stopping the service with exit code \
                             {EXIT_SERVER_EXITED} so its recovery actions apply"
                        );
                    }
                    break why.exit_code();
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if !running {
                        checkpoint += 1;
                        set_status(
                            &handle,
                            status(ServiceState::StartPending, START_WAIT_HINT, checkpoint),
                        );
                    }
                }
            }
        };
        stop.cancel();

        // Workers finish their current inference (or compile) before exiting; keep the SCM
        // informed while we wait, at most `runner::SHUTDOWN_DEADLINE` (CO-12: a GPU compile
        // can take minutes). Past it the service reports Stopped with
        // the failure's own code (a requested stop: no error) and the process ends with the server
        // thread still running.
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
                 running); ending its child processes and reporting the service as stopped \
                 anyway. If this repeats, check the log for the model that was loading",
                runner::SHUTDOWN_DEADLINE.as_secs()
            );
            // The process ends right after Stopped is reported, with the server thread still
            // running: no destructor runs (RA-029 / RA-024).
            runner::before_forced_exit();
            // A stop someone asked for stays a clean stop: with the failure flag set
            // (`sc failureflag`, so a start that gives up is restarted), a non-zero code here
            // would restart a service its user just stopped. A stop the server's own failure
            // started keeps that failure's code.
            if matches!(exit, ServiceExitCode::Win32(0)) {
                ServiceExitCode::Win32(0)
            } else if matches!(exit, ServiceExitCode::ServiceSpecific(_)) {
                exit
            } else {
                ServiceExitCode::ServiceSpecific(EXIT_STOP_TIMED_OUT)
            }
        };
        // Before Stopped: at an OS shutdown the process may be ended right after it.
        cli::flush_logs();
        let mut s = status(ServiceState::Stopped, Duration::ZERO, 0);
        s.exit_code = exit;
        set_status(&handle, s);
        Ok(())
    }

    /// Run the server until `stop`; on failure log and retry after [`RETRY_DELAY`]; on restart
    /// (131) reload the service config and start again immediately. Sends [`Event::Bound`]
    /// each time the listener is bound. Gives up when attempts fail for
    /// [`runner::BIND_GIVE_UP`] with no bind between ([`runner::BindRetry`]).
    fn server_loop(
        first: Result<(Config, PathBuf)>,
        config_path: std::result::Result<PathBuf, String>,
        log: LogReloadHandle,
        stop: CancellationToken,
        current: Arc<Mutex<CancellationToken>>,
        restart_requested: Arc<AtomicBool>,
        events: mpsc::Sender<Event>,
    ) -> ServerExit {
        let rt = match runner::build_runtime() {
            Ok(rt) => rt,
            Err(e) => {
                error!("{e:#}");
                return ServerExit::Unexpected;
            }
        };
        let mut loaded = Some(first);
        let mut active_level: Option<LogLevel> = None;
        let mut last_error: Option<String> = None;
        let mut retry = runner::BindRetry::new(runner::BIND_GIVE_UP);
        let mut exit = ServerExit::Stopped;
        while !stop.is_cancelled() {
            let attempt = stop.child_token();
            *current.lock_recover() = attempt.clone();
            restart_requested.store(false, Ordering::SeqCst);
            let bound = Arc::new(AtomicBool::new(false));

            let result = match loaded.take().unwrap_or_else(|| load_config(&config_path)) {
                Ok((config, path)) => {
                    if active_level.is_some_and(|l| l != config.log_level)
                        && let Err(e) = log.set_level(config.log_level)
                    {
                        warn!("{e:#}");
                    }
                    blue_onyx_prism::clock::change(config.timezone);
                    active_level = Some(config.log_level);
                    let on_bound = {
                        let (bound, events) = (bound.clone(), events.clone());
                        Box::new(move |_port: u16| {
                            bound.store(true, Ordering::SeqCst);
                            let _ = events.send(Event::Bound);
                        })
                    };
                    runner::run_server_notifying(
                        &rt,
                        config,
                        path,
                        log.clone(),
                        attempt.clone(),
                        on_bound,
                    )
                }
                Err(e) => Err(e),
            };
            if bound.load(Ordering::SeqCst) {
                retry.bound();
            }
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
                    if retry.failed(std::time::Instant::now()) {
                        let port = e.downcast_ref::<runner::BindFailed>();
                        error!(
                            "giving up after {} s without an HTTP listener: {msg}. Stopping the \
                             service with exit code {} so its recovery actions apply (they \
                             restart it); fix the cause above to stop the restarts",
                            runner::BIND_GIVE_UP.as_secs(),
                            if port.is_some() {
                                EXIT_PORT_IN_USE
                            } else {
                                EXIT_START_FAILED
                            },
                        );
                        exit = if port.is_some() {
                            ServerExit::PortInUse
                        } else {
                            ServerExit::StartFailed
                        };
                        break;
                    }
                    // Avoid flooding the event log with the same error every few seconds.
                    if last_error.as_deref() == Some(msg.as_str()) {
                        debug!("server still failing: {msg}");
                    } else {
                        error!(
                            "server failed: {msg}; retrying every {} s (after {} s without an \
                             HTTP listener the service stops with an error)",
                            RETRY_DELAY.as_secs(),
                            runner::BIND_GIVE_UP.as_secs()
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
        exit
    }
}

#[cfg(test)]
mod tests {
    use super::event_log_gate::{EVERY, EXPORT_CHILD_TARGET, Gate};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;

    static NOW_MS: AtomicU64 = AtomicU64::new(1_000);

    fn fake_now() -> u64 {
        NOW_MS.load(Ordering::SeqCst)
    }

    /// Counts what reaches it, standing in for the event log layer.
    struct Counter(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for Counter {
        fn on_event(&self, _: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// One call site twice within [`EVERY`]: the second is held back, then one passes again
    /// once the interval is over; ERROR always passes; DEBUG and the export child target never;
    /// another call site has its own throttle.
    #[test]
    fn event_log_gate_throttles_per_call_site() {
        let n = Arc::new(AtomicUsize::new(0));
        // Beside an unfiltered layer, as the file log is: it still gets every line.
        let all = Arc::new(AtomicUsize::new(0));
        let sub = tracing_subscriber::registry()
            .with(Counter(n.clone()).with_filter(Gate::with_clock(fake_now)))
            .with(Counter(all.clone()));
        let every = u64::try_from(EVERY.as_millis()).unwrap();
        tracing::subscriber::with_default(sub, || {
            let warn_site = || tracing::warn!("the same warning");
            let got = |want: usize| assert_eq!(n.swap(0, Ordering::SeqCst), want);
            warn_site();
            warn_site();
            got(1);
            NOW_MS.fetch_add(every - 1, Ordering::SeqCst);
            warn_site();
            got(0);
            NOW_MS.fetch_add(1, Ordering::SeqCst);
            warn_site();
            got(1);
            // Another call site is not held back by the first.
            tracing::info!("another line");
            got(1);
            for _ in 0..3 {
                tracing::error!("a failure");
            }
            got(3);
            tracing::debug!("debug never");
            tracing::info!(target: "blue_onyx_prism::resources::export::child", "Collecting torch");
            tracing::error!(target: "blue_onyx_prism::resources::export::child", "child error");
            got(0);
        });
        assert_eq!(
            all.load(Ordering::SeqCst),
            11,
            "the other layer saw every line"
        );
    }

    /// The exporter still logs its child lines under the target the gate turns off.
    #[test]
    fn the_export_child_target_matches_the_exporter() {
        let src = include_str!("../resources/export/exporter.rs");
        assert!(src.contains(&format!("target: \"{EXPORT_CHILD_TARGET}\"")));
    }
}
