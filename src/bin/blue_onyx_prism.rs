//! Main binary: CLI subcommands (setup-openvino, setup-onnxruntime, download-models, list-models,
//! fetch, list-resources, list-devices, benchmark, diagnostics) or the HTTP service.

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

use anyhow::Result;
use blue_onyx_prism::backend::{CoreOptions, DeviceSpec, OrtOptions, Runtimes, select};
use blue_onyx_prism::cli::{self, Cli, Command};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::{
    benchmark, config_file, download, resources, runner, setup_onnxruntime, setup_openvino,
    system_info,
};
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

fn main() -> ExitCode {
    blue_onyx_prism::panic::install();
    // Before any thread starts (Linux glibc: arena cap, fixed mmap threshold).
    blue_onyx_prism::backend::memory::tune_allocator();
    // A debug build's clap parsing alone needs more than Windows' 1 MiB main-thread stack.
    blue_onyx_prism::main_thread::run(cli_main)
}

fn cli_main() -> ExitCode {
    // Paths typed on the command line are relative to the cwd (config-file paths are
    // relative to the exe dir), so make them absolute before anything uses them.
    let cli = Cli::parse();
    let cli = match std::env::current_dir() {
        Ok(cwd) => cli.absolutized(&cwd),
        Err(_) => cli,
    };
    let code = match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    };
    // Commands that never set up logging still show what config handling had to say (CO-11).
    for (level, msg) in blue_onyx_prism::logbuf::take_early() {
        eprintln!("{level}: {msg}");
    }
    // The file log's last lines (RA-024): nothing else drops its writer guard.
    cli::flush_logs();
    code
}

/// `std::process::exit(code)` after the file log's last lines are written (RA-024).
fn exit_flushed(code: i32) -> ! {
    cli::flush_logs();
    std::process::exit(code);
}

/// `service start|stop|restart|status`: prints the resulting state. The exit code (`Ok`) is
/// 0, or for `status` 3 when the service is not running (like `systemctl status`); start and
/// restart that did not reach "running" in time also give 3, stop that did not stop gives 1.
fn service_command(cmd: cli::ServiceCommand) -> Result<i32> {
    use blue_onyx_prism::service_ctl::{self, Action, ServiceState};
    let action = match cmd {
        cli::ServiceCommand::Status => {
            let state = service_ctl::query()?;
            println!("Blue Onyx Prism service: {}", state.label());
            return Ok(if state == ServiceState::Running { 0 } else { 3 });
        }
        cli::ServiceCommand::Start => Action::Start,
        cli::ServiceCommand::Stop => Action::Stop,
        cli::ServiceCommand::Restart => Action::Restart,
    };
    let state = service_ctl::control(action)?;
    println!("Blue Onyx Prism service: {}", state.label());
    let reached = match action {
        Action::Stop => state == ServiceState::Stopped,
        Action::Start | Action::Restart => state == ServiceState::Running,
    };
    if !reached {
        eprintln!(
            "warning: the service did not reach the expected state yet; check again with \
             `blue-onyx-prism service status` and see the service log if it stays {}",
            state.label()
        );
        return Ok(if action == Action::Stop { 1 } else { 3 });
    }
    Ok(0)
}

/// `admin-token`: the token on stdout, nothing else there. No config file means this is not the
/// service's data root, and a token made here would be one nobody checks, so that is an error.
/// The service's config for a command that works on its data root: the subcommand's
/// `--config`, else the global one, else the default path; it must exist (`what` names what
/// could not be found otherwise).
fn service_config(cli: &Cli, config: Option<PathBuf>, what: &str) -> Result<Config> {
    let cli = Cli {
        config: config.or_else(|| cli.config.clone()),
        ..cli.clone()
    };
    let file = config_path(&cli);
    if !file.is_file() {
        anyhow::bail!(
            "there is no config file at {}, so the service's {what} cannot be found; pass the \
             service's config with --config <file> (it is named in the service's start command)",
            file.display()
        );
    }
    read_config(&cli)
}

fn admin_token_command(cli: &Cli, rotate: bool, config: Option<PathBuf>) -> Result<()> {
    let config = service_config(cli, config, "token")?;
    let token = blue_onyx_prism::admin_token::for_command(&config, rotate)?;
    if rotate {
        eprintln!(
            "Replaced {}. A running service uses the new token at once; signed-in browsers have \
             to sign in again.",
            blue_onyx_prism::admin_token::path(&config).display()
        );
    }
    println!("{}", token.reveal());
    Ok(())
}

/// The `train` subcommands that run jobs (spawn Python or uv trees, take the training or
/// export lock, trust busy markers, remove the training environment): the orphan sweep runs
/// before them. Exhaustive on purpose, so a new subcommand has to choose.
fn train_sweeps_orphans(cmd: &blue_onyx_prism::training::cli::TrainCommand) -> bool {
    use blue_onyx_prism::training::cli::TrainCommand as T;
    match cmd {
        T::Run { .. }
        | T::Import(_)
        | T::Autolabel(_)
        | T::Overlaps(_)
        | T::Refine(_)
        | T::RetryExport { .. }
        | T::RemoveEnv => true,
        T::Capability { .. } | T::Estimate { .. } | T::Prepare { .. } | T::Classes(_) | T::List => {
            false
        }
    }
}

/// RA-011: end the children an earlier, now dead, process left in this config's data root
/// ([`blue_onyx_prism::children::sweep_orphans_once`]) before a command that takes a job lock
/// or trusts a busy marker: those locks die with a crashed process while its children may still
/// write. Said on stderr (these commands have no info-level log). A config that cannot be read
/// is left to the command to report.
fn sweep_orphans(cli: &Cli) {
    let Ok(cfg) = read_config(cli) else {
        return;
    };
    let Some(report) = blue_onyx_prism::children::sweep_orphans_once(&cfg.data_root()) else {
        return;
    };
    for k in &report.killed {
        match &k.error {
            None => eprintln!(
                "ended {} process {} ({}), left running by an earlier run",
                k.kind,
                k.pid,
                k.exe.display()
            ),
            Some(e) => eprintln!(
                "warning: could not end {} process {} ({}), left running by an earlier run: {e}; \
                 end it by hand",
                k.kind,
                k.pid,
                k.exe.display()
            ),
        }
    }
    if let Some(e) = &report.error {
        eprintln!(
            "warning: the check for processes an earlier run left running could not run: {e}; \
             if a training, export or auto-label process from a crashed run still runs, end it \
             by hand"
        );
    }
}

/// `--config` or the default config path.
fn config_path(cli: &Cli) -> PathBuf {
    cli.config
        .clone()
        .unwrap_or_else(Config::default_config_path)
}

/// The config file a command works with: its own `--config` (the commands acting on a service),
/// else the global one, else the default path.
fn command_config_path(cli: &Cli) -> PathBuf {
    let own = match &cli.command {
        Some(
            Command::AdminToken { config, .. }
            | Command::User { config, .. }
            | Command::SetupCode { config }
            | Command::TlsFingerprint { config }
            | Command::TlsRegenerate { config },
        ) => config.clone(),
        _ => None,
    };
    own.unwrap_or_else(|| config_path(cli))
}

/// The config file (`--config` or the default path) if it exists, else defaults. Read-only.
/// A file that exists but cannot be read or parsed is an error (the message names the path).
fn read_config(cli: &Cli) -> Result<Config> {
    Config::load_or_default(&config_path(cli))
}

/// `--dir` if given, else the configured `models_dir` (from the config file when it exists),
/// resolved against the exe dir.
fn models_dir(cli: &Cli, dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d);
    }
    let cfg = read_config(cli)?;
    Ok(cfg.data_path(&cfg.models_dir))
}

/// `list-devices`: hardware, runtimes, the option table, the `auto` pick, and the load plan of each
/// model (`auto` ordered by `benchmark.json` beside the config when it has results for the
/// model, as the server does). Works without OpenVINO (its options are shown as not runnable).
fn list_devices(cli: &Cli, models: &[PathBuf]) -> Result<()> {
    let mut cfg = read_config(cli)?;
    if let Some(d) = &cli.openvino_dir {
        cfg.openvino_dir = Some(d.clone());
    }
    if let Some(d) = &cli.device {
        cfg.device = d.clone();
    }
    if cli.force_cpu {
        cfg.force_cpu = true;
    }
    let ort_opts: OrtOptions = cfg.ort_options();
    let mut rt = Runtimes::new_with(
        &CoreOptions {
            cache_dir: None,
            intra_threads: 0,
            openvino_dir: cfg.openvino_dir_effective(),
        },
        &ort_opts,
    );
    let hw = rt.hardware();
    println!("Platform: {} {}", hw.os, hw.arch);
    println!("GPUs:");
    if hw.gpus.is_empty() {
        println!("  (none detected)");
    }
    for g in &hw.gpus {
        println!("  {g}");
    }
    if let Some(v) = &hw.cuda_driver {
        println!("  NVIDIA driver: supports CUDA {v}");
    }
    println!("Runtimes:");
    match rt.openvino() {
        Some(core) => println!(
            "  OpenVINO {}: {}",
            core.openvino_version(),
            core.available_devices().join(", ")
        ),
        None => println!(
            "  OpenVINO: not available ({})",
            rt.openvino_error().unwrap_or("not initialized")
        ),
    }
    match &rt.probe().ort.error {
        None => println!("  ONNX Runtime: available"),
        Some(e) => println!("  ONNX Runtime: not available ({e})"),
    }
    let mut sel = rt.selection(None);
    let installed = resources::detect_installed(&cfg, &cfg.data_root(), Some(rt.probe().clone()));
    resources::resolve::annotate_selection(&mut sel, &cfg, hw, &installed, true);
    println!("Device options:");
    print!("{}", select::format_options(&sel));
    let pick = sel
        .auto_pick()
        .map(|o| format!("{} ({})", o.spec, o.label))
        .unwrap_or_else(|| "nothing runnable".to_string());
    println!("auto: {pick}");
    println!("auto order: {}", select::format_auto(&sel));

    let entries: Vec<ModelConfig> = if models.is_empty() {
        cfg.enabled_models().cloned().collect()
    } else {
        models
            .iter()
            .map(|p| ModelConfig {
                path: p.clone(),
                ..ModelConfig::default()
            })
            .collect()
    };
    let results =
        benchmark::BenchmarkResults::load_or_warn(&benchmark::results_path(&config_path(cli)));
    let orders = benchmark::auto_orders(results.as_ref(), &cfg, &entries, rt.hardware());
    rt.set_measured_orders(orders);
    // Same pool size as the server: configured, else the profile's (capped when every model is
    // headed for a CPU device).
    let planned: Vec<_> = entries
        .iter()
        .map(|m| {
            let spec = cfg.device_spec_for(m).ok()?;
            rt.first_runnable(&spec, &cfg.data_path(&m.path))
                .map(|(d, _)| d)
        })
        .collect();
    let profile = cfg.profile();
    let threads = blue_onyx_prism::registry::preprocess_threads(&cfg, &profile, &planned);
    println!("Hardware profile: {}", profile.summary(threads));
    if entries.is_empty() {
        return Ok(());
    }
    println!("Models:");
    for m in &entries {
        let path = cfg.data_path(&m.path);
        let name = m.effective_name();
        let spec = match cfg.device_spec_for(m) {
            Ok(s) => s,
            Err(e) => {
                println!("  {name}: {e:#}");
                continue;
            }
        };
        let plan: Vec<String> = rt
            .plan(&spec, &path)
            .iter()
            .map(|c| c.device.to_string())
            .collect();
        let measured =
            if matches!(spec, DeviceSpec::Auto) && rt.selection(Some(&path)).auto_measured {
                " (benchmark order)"
            } else {
                ""
            };
        let onnx = match select::onnx_path_for(&path) {
            Some(_) => "",
            None => ", no .onnx for ONNX Runtime",
        };
        println!(
            "  {name} ({}{onnx}): device {spec} -> {}{measured}",
            path.display(),
            plan.join(" -> ")
        );
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    // Every command (training, benchmark, accounts, diagnostics, the server) shows times and
    // names files in the config's `timezone`, from its first log line on. A file that does not
    // parse leaves `local`; the command itself reports it when it reads the config.
    if let Some(c) = Config::parse_file(&command_config_path(&cli)) {
        blue_onyx_prism::clock::set(c.timezone);
    }
    match cli.command.clone() {
        Some(Command::SetupOpenvino {
            dest,
            version,
            archive,
        }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            let opts = setup_openvino::SetupOptions {
                dest,
                version,
                archive,
                keep_archive: false,
                root: Some(resources::download_root(&read_config(&cli)?)),
            };
            let dir = setup_openvino::run(&opts)?;
            println!("OpenVINO runtime installed in {}", dir.display());
            return Ok(());
        }
        Some(Command::SetupOnnxruntime { flavor, dir }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            let opts = setup_onnxruntime::SetupOptions {
                dest: dir,
                flavor,
                keep_archive: false,
                root: Some(resources::download_root(&read_config(&cli)?)),
            };
            let (dir, flavor) = setup_onnxruntime::run(&opts)?;
            println!(
                "ONNX Runtime ({}) installed in {}",
                flavor.as_str(),
                dir.display()
            );
            return Ok(());
        }
        Some(Command::DownloadModels {
            all,
            name,
            dir,
            add_to_config,
        }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            if !all && name.is_empty() {
                anyhow::bail!("specify --name <model> (repeatable) or --all; see `list-models`");
            }
            let dir = models_dir(&cli, dir)?;
            let root = resources::download_root(&read_config(&cli)?);
            let files = download::download_with_root(&name, all, &dir, &root)?;
            for f in &files {
                println!("{}", f.display());
            }
            if add_to_config {
                let path = cli
                    .config
                    .clone()
                    .unwrap_or_else(Config::default_config_path);
                let models = download::model_configs(&files, &blue_onyx_prism::exe_dir());
                // Under the config lock: a page save of a running service is never undone.
                let edited = config_file::edit_file(&path, config_file::Missing::Default, |cfg| {
                    Ok(download::add_to_config(cfg, models))
                })?;
                for o in &edited.value {
                    println!("{}", o.describe(&path));
                }
            }
            return Ok(());
        }
        Some(Command::AddModel {
            path: model_path,
            name,
            family,
            classes,
        }) => {
            let path = cli
                .config
                .clone()
                .unwrap_or_else(Config::default_config_path);
            let req = resources::actions::CustomModel {
                path: model_path.to_string_lossy().to_string(),
                name,
                family,
                classes: classes.map(|c| c.to_string_lossy().to_string()),
            };
            // Under the config lock: a page save of a running service is never undone.
            let edited = config_file::edit_file(&path, config_file::Missing::Default, |cfg| {
                resources::actions::add_custom_model(cfg, &req)
                    .map_err(|e| anyhow::anyhow!("{}", e.message()))
            })?;
            println!("{}", edited.value);
            println!("config: {}", path.display());
            return Ok(());
        }
        Some(Command::ListModels { dir }) => {
            download::print_list(&models_dir(&cli, dir)?);
            return Ok(());
        }
        Some(Command::Fetch {
            for_config,
            resource,
            all_for_platform,
            allow_large,
        }) => {
            // Exports take the export lock and replace tools: orphans first (RA-011).
            sweep_orphans(&cli);
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let req = resources::commands::FetchRequest {
                for_config,
                resources: resource,
                all_for_platform,
                allow_large,
            };
            return resources::commands::fetch(&read_config(&cli)?, &req);
        }
        Some(Command::ListResources { check_urls: true }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            return resources::commands::check_urls();
        }
        Some(Command::ListResources { check_urls: false }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let cfg = read_config(&cli)?;
            let root = resources::download_root(&cfg);
            let installed = resources::detect_installed(&cfg, &root, None);
            print!(
                "{}",
                resources::commands::list_resources(
                    &cfg,
                    blue_onyx_prism::backend::detect::hardware(),
                    &installed
                )
            );
            return Ok(());
        }
        Some(Command::Benchmark(args)) => {
            // Sets up its own stderr logging (warn, or info with --verbose).
            let code = blue_onyx_prism::benchmark::cli::main(*args, cli.config.clone());
            if code != ExitCode::SUCCESS {
                exit_flushed(1);
            }
            return Ok(());
        }
        Some(Command::Train(args)) => {
            if train_sweeps_orphans(&args.command) {
                sweep_orphans(&cli);
            }
            let code = blue_onyx_prism::training::cli::main(*args, cli.config.clone());
            if code != ExitCode::SUCCESS {
                exit_flushed(1);
            }
            return Ok(());
        }
        Some(Command::ListDevices { model }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            list_devices(&cli, &model)?;
            return Ok(());
        }
        Some(Command::HfLogin) => return resources::hf_token::login(&read_config(&cli)?),
        Some(Command::HfLogout) => return resources::hf_token::logout(&read_config(&cli)?),
        Some(Command::AdminToken { rotate, config }) => {
            return admin_token_command(&cli, rotate, config);
        }
        Some(Command::User { action, config }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let config = service_config(&cli, config, "accounts")?;
            return blue_onyx_prism::accounts::commands::user(
                &config,
                &action,
                &mut blue_onyx_prism::accounts::commands::Terminal,
                &mut std::io::stdout(),
            );
        }
        Some(Command::SetupCode { config }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let config = service_config(&cli, config, "accounts")?;
            print!(
                "{}",
                blue_onyx_prism::accounts::commands::setup_code(&config)?
            );
            return Ok(());
        }
        Some(Command::TlsFingerprint { config }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let config = service_config(&cli, config, "HTTPS certificate")?;
            print!("{}", blue_onyx_prism::tls::fingerprint_report(&config)?);
            return Ok(());
        }
        Some(Command::TlsRegenerate { config }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let config = service_config(&cli, config, "HTTPS certificate")?;
            print!("{}", blue_onyx_prism::tls::regenerate_report(&config)?);
            return Ok(());
        }
        Some(Command::Diagnostics { out }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let path = blue_onyx_prism::diagnostics::command(&config_path(&cli), out.as_deref())?;
            println!("{}", path.display());
            eprintln!(
                "Wrote the diagnostics zip. It holds no frames, labels, datasets or tokens, but \
                 camera names and file paths are in it: look through it before sharing it \
                 publicly."
            );
            return Ok(());
        }
        Some(Command::Service { action }) => {
            let code = service_command(action)?;
            if code != 0 {
                exit_flushed(code);
            }
            return Ok(());
        }
        None => {}
    }

    let (mut config, config_path) = cli::resolve_config(&cli)?;
    blue_onyx_prism::clock::set(config.timezone);
    // CO-05: no models -> the default catalog model, so the UI comes up (and it downloads).
    cli::seed_default_model(&mut config, &config_path);
    // Initialized once; later level changes go through the reload handle (web UI, restart).
    // A relative `log_path` from the file resolves against the exe dir, as in the service
    // (CO-10; a `--log-path` typed on the command line is already absolute). A log dir that
    // cannot be created falls back to stdout with a warning (CO-11).
    let log_path = config
        .log_path
        .as_deref()
        .map(blue_onyx_prism::resolve_path);
    let log = cli::init_logging_opts(
        config.log_level,
        log_path.as_deref(),
        config.log_format,
        None,
    )?;
    cli::log_startup_warnings(&config, &config_path);
    info!(
        version = blue_onyx_prism::VERSION,
        git_sha = blue_onyx_prism::GIT_SHA,
        build_date = blue_onyx_prism::BUILD_DATE,
        os = %system_info::os_description(),
        cpu = %system_info::cpu_name(),
        memory_gb = ?system_info::total_memory_gb(),
        config = %config_path.display(),
        "starting Blue Onyx Prism {}",
        blue_onyx_prism::long_version()
    );
    runner::ensure_models(&config, &config_path)?;

    let rt = runner::build_runtime()?;
    let shutdown = CancellationToken::new();
    {
        let token = shutdown.clone();
        rt.spawn(async move {
            let first = wait_for_signal().await;
            match first {
                Signal::Interrupt => info!(
                    "shutdown requested; waiting for running work to finish, at most {} s \
                     (press Ctrl-C again to exit at once)",
                    runner::SHUTDOWN_DEADLINE.as_secs()
                ),
                Signal::Closing(what) => warn!(
                    "{what}: stopping now; Windows ends the process within seconds, so the \
                     stop is cut short after {} s",
                    URGENT_DEADLINE.as_secs()
                ),
            }
            token.cancel();
            spawn_shutdown_deadline(match first {
                Signal::Interrupt => runner::SHUTDOWN_DEADLINE,
                Signal::Closing(_) => URGENT_DEADLINE,
            });
            // CO-12: a second signal exits at once (after ending this process's child trees and
            // writing the log's tail, RA-029 / RA-024).
            wait_for_signal().await;
            eprintln!("second interrupt: exiting without waiting for running work");
            info!("second interrupt: exiting without waiting for running work");
            runner::before_forced_exit();
            std::process::exit(130);
        });
    }
    let result = runner::run_server_on(&rt, config, config_path, log, shutdown);
    // Blocking tasks (file browsing, a cancelled benchmark or export step) get a bounded
    // grace period instead of holding the process.
    rt.shutdown_timeout(runner::RUNTIME_SHUTDOWN_GRACE);
    result
}

/// How long the stop may take when Windows is closing the console (window closed, logoff,
/// shutdown): the OS ends the process about 5 s after the event, so the forced exit (child trees
/// ended, log flushed) comes first.
const URGENT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(4);

/// CO-12: after a shutdown request the process exits within `deadline`
/// ([`runner::SHUTDOWN_DEADLINE`]) even when a worker is stuck (e.g. a GPU compile that takes
/// minutes). Before it exits it ends the child trees this process recorded and writes the file
/// log's tail ([`runner::before_forced_exit`]).
fn spawn_shutdown_deadline(deadline: std::time::Duration) {
    let spawned = std::thread::Builder::new()
        .name("shutdown-deadline".into())
        .spawn(move || {
            std::thread::sleep(deadline);
            error!(
                "shutdown did not finish within {} s (a model compile or a request is still \
                 running); ending this process's child processes and exiting anyway",
                deadline.as_secs()
            );
            eprintln!(
                "shutdown did not finish within {} s; exiting anyway",
                deadline.as_secs()
            );
            runner::before_forced_exit();
            std::process::exit(1);
        });
    if let Err(e) = spawned {
        error!("could not start the shutdown deadline thread: {e}");
    }
}

/// What asked the process to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    /// Ctrl-C, SIGTERM, Ctrl-Break: the normal bounded stop.
    Interrupt,
    /// Windows is closing the console (window closed, logoff, shutdown) and ends the process
    /// within seconds.
    #[cfg_attr(not(windows), allow(dead_code, reason = "Windows console events only"))]
    Closing(&'static str),
}

async fn wait_for_signal() -> Signal {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
        Signal::Interrupt
    }
    #[cfg(windows)]
    {
        windows_signal().await
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
        Signal::Interrupt
    }
}

/// Ctrl-C and Ctrl-Break, and the console's close, logoff and shutdown events (RA-027). A handler
/// the OS refuses is left out (that event then ends the process the OS's way).
#[cfg(windows)]
async fn windows_signal() -> Signal {
    use tokio::signal::windows;
    let mut c = windows::ctrl_c().ok();
    let mut brk = windows::ctrl_break().ok();
    let mut close = windows::ctrl_close().ok();
    let mut logoff = windows::ctrl_logoff().ok();
    let mut shutdown = windows::ctrl_shutdown().ok();
    // A receiver that could not be registered is `None`: its branch is disabled at once.
    tokio::select! {
        Some(()) = async { c.as_mut()?.recv().await } => Signal::Interrupt,
        Some(()) = async { brk.as_mut()?.recv().await } => Signal::Interrupt,
        Some(()) = async { close.as_mut()?.recv().await } => {
            Signal::Closing("the console window is closing")
        }
        Some(()) = async { logoff.as_mut()?.recv().await } => {
            Signal::Closing("the user is logging off")
        }
        Some(()) = async { shutdown.as_mut()?.recv().await } => {
            Signal::Closing("Windows is shutting down")
        }
        else => std::future::pending::<Signal>().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(args: &[&str]) -> blue_onyx_prism::training::cli::TrainCommand {
        let mut argv = vec!["blue-onyx-prism", "train"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv).expect("parses").command {
            Some(Command::Train(a)) => a.command,
            other => panic!("not a train command: {other:?}"),
        }
    }

    /// RA-011: the job-running `train` subcommands sweep orphans first; the read-only ones do
    /// not.
    #[test]
    fn job_running_train_commands_sweep_orphans_first() {
        for args in [
            &["run", "--model", "yolo26n"][..],
            &["import", "--path", "clips"],
            &["autolabel"],
            &["overlaps"],
            &["refine"],
            &["remove-env"],
            &["retry-export", "run-1"],
        ] {
            assert!(train_sweeps_orphans(&train(args)), "{args:?}");
        }
        for args in [&["capability"][..], &["list"]] {
            assert!(!train_sweeps_orphans(&train(args)), "{args:?}");
        }
    }
}
