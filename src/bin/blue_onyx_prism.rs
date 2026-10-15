//! Main binary: CLI subcommands (setup-openvino, setup-onnxruntime, download-models, list-models,
//! fetch, list-resources, list-devices, benchmark) or the HTTP service.

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

use anyhow::{Context, Result};
use blue_onyx_prism::backend::{CoreOptions, DeviceSpec, OrtOptions, Runtimes, select};
use blue_onyx_prism::cli::{self, Cli, Command};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::{
    benchmark, download, resources, runner, setup_onnxruntime, setup_openvino, system_info,
};
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

fn main() -> ExitCode {
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
    code
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

/// `--config` or the default config path.
fn config_path(cli: &Cli) -> PathBuf {
    cli.config
        .clone()
        .unwrap_or_else(Config::default_config_path)
}

/// The config file (`--config` or the default path) if it exists, else defaults. Read-only.
/// A file that exists but cannot be read or parsed is an error (the message names the path).
fn read_config(cli: &Cli) -> Result<Config> {
    let path = config_path(cli);
    if path.exists() {
        Config::load(&path).context("fix the config file, or pass --config <another file>")
    } else {
        Ok(Config::default())
    }
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
                let mut cfg = if path.exists() {
                    Config::load(&path)?
                } else {
                    Config::default()
                };
                let models = download::model_configs(&files, &blue_onyx_prism::exe_dir());
                let outcomes = download::add_to_config(&mut cfg, models);
                for o in &outcomes {
                    println!("{}", o.describe(&path));
                }
                if outcomes
                    .iter()
                    .any(|o| matches!(o, download::AddOutcome::Added { .. }))
                {
                    cfg.save(&path)?;
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
            let mut cfg = if path.exists() {
                Config::load(&path)?
            } else {
                Config::default()
            };
            let req = resources::actions::CustomModel {
                path: model_path.to_string_lossy().to_string(),
                name,
                family,
                classes: classes.map(|c| c.to_string_lossy().to_string()),
            };
            let msg = resources::actions::add_custom_model(&mut cfg, &req)
                .map_err(|e| anyhow::anyhow!("{}", e.message()))?;
            cfg.save(&path)?;
            println!("{msg}");
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
                std::process::exit(1);
            }
            return Ok(());
        }
        Some(Command::ListDevices { model }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            list_devices(&cli, &model)?;
            return Ok(());
        }
        Some(Command::Service { action }) => {
            let code = service_command(action)?;
            if code != 0 {
                std::process::exit(code);
            }
            return Ok(());
        }
        None => {}
    }

    let (mut config, config_path) = cli::resolve_config(&cli)?;
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
    cli::log_startup_warnings(&config);
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
            wait_for_signal().await;
            info!(
                "shutdown requested; waiting for running work to finish (press Ctrl-C again \
                 to exit at once)"
            );
            token.cancel();
            spawn_shutdown_deadline();
            // CO-12: a second signal exits at once.
            wait_for_signal().await;
            eprintln!("second interrupt: exiting without waiting for running work");
            std::process::exit(130);
        });
    }
    let result = runner::run_server_on(&rt, config, config_path, log, shutdown);
    // Blocking tasks (file browsing, a cancelled benchmark or export step) get a bounded
    // grace period instead of holding the process.
    rt.shutdown_timeout(runner::RUNTIME_SHUTDOWN_GRACE);
    result
}

/// CO-12: after a shutdown request the process exits within [`runner::SHUTDOWN_DEADLINE`]
/// even when a worker is stuck (e.g. a GPU compile that takes minutes).
fn spawn_shutdown_deadline() {
    let spawned = std::thread::Builder::new()
        .name("shutdown-deadline".into())
        .spawn(|| {
            std::thread::sleep(runner::SHUTDOWN_DEADLINE);
            error!(
                "shutdown did not finish within {} s (a model compile or a request is still \
                 running); exiting anyway",
                runner::SHUTDOWN_DEADLINE.as_secs()
            );
            eprintln!(
                "shutdown did not finish within {} s; exiting anyway",
                runner::SHUTDOWN_DEADLINE.as_secs()
            );
            std::process::exit(1);
        });
    if let Err(e) = spawned {
        error!("could not start the shutdown deadline thread: {e}");
    }
}

async fn wait_for_signal() {
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
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
