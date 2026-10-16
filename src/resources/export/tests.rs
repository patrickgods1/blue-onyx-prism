//! Unit tests of the export module (stage machine, installs, exporter queue) over the fakes.

use super::fake::{Fake, arg_after};
use super::*;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bop-export-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn yolo26n() -> &'static Resource {
    catalog::export_model("yolo26n").unwrap()
}

/// RS-09: `install_outputs` moves `.yaml`, then `.onnx`, then the manifest; a failure at the
/// last step restores the previous files and leaves no new manifest.
#[test]
fn install_outputs_is_all_or_nothing() {
    let root = tmp("install-tx");
    let (staging, models) = (root.join("staging"), root.join("models"));
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::create_dir_all(&models).unwrap();
    let res = yolo26n();
    let (onnx, yaml) = catalog::export_files(res).unwrap();
    let (onnx, yaml) = (onnx.as_str(), yaml.as_str());
    let names: String = (0..80).map(|i| format!("  - c{i}\n")).collect();
    let stage = |tag: &str| {
        std::fs::write(staging.join(onnx), vec![tag.as_bytes()[0]; 2048]).unwrap();
        std::fs::write(staging.join(yaml), format!("NAMES:\n{names}# {tag}\n")).unwrap();
    };
    // A previous export by hand: old files, no manifest.
    std::fs::write(models.join(onnx), b"old onnx").unwrap();
    std::fs::write(models.join(yaml), b"old yaml").unwrap();
    let manifest = manager::manifest_path(res, &models);

    stage("new");
    FAIL_MANIFEST_RENAME.with(|f| f.set(true));
    let err = install_outputs(res, &staging, &models, onnx, yaml).unwrap_err();
    FAIL_MANIFEST_RENAME.with(|f| f.set(false));
    assert!(format!("{err:#}").contains("rolled back"), "{err:#}");
    assert_eq!(std::fs::read(models.join(onnx)).unwrap(), b"old onnx");
    assert_eq!(std::fs::read(models.join(yaml)).unwrap(), b"old yaml");
    assert!(!manifest.exists());
    let leftovers: Vec<String> = std::fs::read_dir(&models)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    stage("new");
    let out = install_outputs(res, &staging, &models, onnx, yaml).unwrap();
    assert_eq!(out.classes, 80);
    assert_eq!(std::fs::read(models.join(onnx)).unwrap(), vec![b'n'; 2048]);
    assert!(
        std::fs::read_to_string(models.join(yaml))
            .unwrap()
            .contains("# new")
    );
    let m = manager::read_manifest(&manifest).unwrap();
    assert_eq!(m.files, [onnx.to_string(), yaml.to_string()]);
    assert!(is_exported(res, &models));
    let leftovers = std::fs::read_dir(&models)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".old-"))
        .count();
    assert_eq!(leftovers, 0);
    let _ = std::fs::remove_dir_all(&root);
}

/// RS-11: the export lock wait is bounded and says why.
#[test]
fn export_lock_wait_is_bounded() {
    let root = tmp("lock-wait");
    let path = root.join("export.lock");
    let never = AtomicBool::new(false);
    let held = ExportLock::acquire(&path, &never, Duration::from_secs(1), || {}).unwrap();
    let mut told = 0;
    let e = ExportLock::acquire(&path, &never, Duration::from_millis(300), || told += 1)
        .err()
        .unwrap();
    assert_eq!(told, 1);
    assert!(format!("{e:#}").contains("another process"), "{e:#}");
    drop(held);
    ExportLock::acquire(&path, &never, Duration::from_millis(300), || {}).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

const MAC: Platform = Platform::MACOS_ARM64;
const LINUX: Platform = Platform::LINUX_X64;

fn job(root: &Path, platform: Platform) -> ExportJob {
    ExportJob {
        resource: yolo26n(),
        paths: Paths::new(root, root.join("models")),
        platform,
        timeouts: Timeouts::default(),
    }
}

fn run(fake: &Fake, j: &ExportJob) -> (Result<ExportOutput, ExportError>, Vec<(Stage, u8)>) {
    let mut stages = Vec::new();
    let cancel = AtomicBool::new(false);
    let r = run_export(
        j,
        fake,
        fake,
        &mut |p| {
            if let Progress::Stage { stage, percent, .. } = p {
                stages.push((stage, stage.overall(percent)));
            }
        },
        &cancel,
    );
    (r, stages)
}

#[test]
fn full_export_runs_every_stage_in_order() {
    let root = tmp("full");
    let fake = Fake::new();
    let j = job(&root, MAC);
    let (r, stages) = run(&fake, &j);
    let out = r.unwrap();
    assert_eq!(out.classes, 80);
    assert_eq!(out.onnx, root.join("models/yolo26n.onnx"));
    assert!(out.onnx.is_file() && out.yaml.is_file());
    // Stages appear in order and the overall percent never goes back.
    let order: Vec<Stage> = stages.iter().map(|s| s.0).fold(Vec::new(), |mut v, s| {
        if v.last() != Some(&s) {
            v.push(s);
        }
        v
    });
    assert_eq!(order, Stage::ALL);
    assert!(stages.windows(2).all(|w| w[0].1 <= w[1].1), "{stages:?}");
    assert_eq!(stages.last().unwrap().1, 100);
    // Downloads went through the fetcher: uv and the weights.
    assert_eq!(*fake.fetched.lock().unwrap(), ["tool:uv", "model:yolo26n"]);
    let cmds = fake.cmds();
    assert_eq!(cmds.len(), 4, "{cmds:#?}");
    assert!(
        cmds[0].contains(" python install 3.11 --install-dir "),
        "{}",
        cmds[0]
    );
    assert!(cmds[0].contains("--no-bin") && !cmds[0].contains("--no-registry"));
    assert!(cmds[0].contains("--no-config"));
    assert!(cmds[1].contains(" venv --python 3.11 --managed-python --clear "));
    assert!(cmds[2].contains(" pip install --python "));
    assert!(cmds[2].contains("--require-hashes --no-deps --only-binary :all: -r "));
    assert!(!cmds[2].contains("--torch-backend"), "macOS uses PyPI");
    assert!(cmds[3].contains(" -u -I "));
    assert!(
        cmds[3].contains("--recipe ultralytics --name yolo26n --family yolo26 --weights "),
        "{}",
        cmds[3]
    );
    assert!(cmds[3].contains("--imgsz 640 --no-openvino --out-dir "));
    assert!(!cmds[3].contains("--source"));
    let all = fake.commands.lock().unwrap().clone();
    let export = &all[3];
    assert_eq!(export.program, j.paths.env_python(false));
    for (k, v) in [
        ("YOLO_OFFLINE", "1"),
        ("YOLO_AUTOINSTALL", "false"),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
    ] {
        assert!(export.env.iter().any(|(a, b)| a == k && b == v), "{k}");
    }
    assert_eq!(
        arg_after(export, "--weights").unwrap(),
        root.join("tools/export-weights/yolo26n/yolo26n.pt")
    );
    assert_eq!(
        arg_after(export, "--work").unwrap(),
        root.join("tools/export-work/yolo26n")
    );
    assert!(
        !root.join("tools/export-work/yolo26n").exists(),
        "work files removed"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("tools/export-work/export_model.py")).unwrap(),
        EXPORT_SCRIPT
    );
    // Manifest, marker, staging gone, cache gone.
    let m = manifest(yolo26n(), &root.join("models")).unwrap();
    assert_eq!(m.version, catalog::ULTRALYTICS_EXPORT_VERSION);
    assert_eq!(m.files, ["yolo26n.onnx", "yolo26n.yaml"]);
    assert_eq!(m.sha256.len(), 2);
    assert!(is_exported(yolo26n(), &root.join("models")));
    assert!(!j.paths.staging("yolo26n").exists());
    assert!(toolchain_state(&j.paths, MAC).ready());
    assert!(
        std::fs::read_to_string(root.join("tools/model-export-requirements.lock.txt"))
            .unwrap()
            .contains("ultralytics==8.4.163")
    );

    // Second export: toolchain reused; only the export runs (weights are kept too).
    let fake2 = Fake::new();
    let (r, _) = run(&fake2, &j);
    r.unwrap();
    assert!(fake2.fetched.lock().unwrap().is_empty());
    let cmds = fake2.cmds();
    assert_eq!(cmds.len(), 1);
    assert!(cmds[0].contains("export_model.py"));

    // Remove: model files and toolchain.
    remove_export(yolo26n(), &j.paths, None).unwrap();
    assert!(!is_exported(yolo26n(), &root.join("models")));
    assert!(manifest(yolo26n(), &root.join("models")).is_none());
    remove_toolchain(&j.paths).unwrap();
    assert!(!toolchain_state(&j.paths, MAC).present);
    assert!(!root.join("tools").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// Remove deletes everything of one export (IR of earlier versions, weights, work and staging
/// files, set-aside files, CoreML rewrites) and nothing of other models or the toolchain.
#[test]
fn remove_export_leaves_nothing_behind() {
    let root = tmp("remove");
    let j = job(&root, MAC);
    let p = &j.paths;
    let cache = root.join("cache");
    let put = |f: &Path| {
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, b"x").unwrap();
    };
    let ours = [
        p.models.join("yolo26n.onnx"),
        p.models.join("yolo26n.yaml"),
        p.models.join("yolo26n.xml"),
        p.models.join("yolo26n.bin"),
        manager::manifest_path(yolo26n(), &p.models),
        p.models.join(".yolo26n.onnx.old-1234"),
        p.staging("yolo26n").join("yolo26n.onnx"),
        p.weights_of("yolo26n").join("yolo26n.pt"),
        p.work.join("yolo26n").join("x.tmp"),
        p.tools.join("yolo26-weights/yolo26n.pt"),
        cache.join("coreml/yolo26n-0123456789abcdef-v3.onnx"),
    ];
    let others = [
        p.models.join("yolo26s.onnx"),
        p.models.join("yolo26s.xml"),
        p.models.join("yolo26nano.onnx"),
        p.weights_of("yolo26s").join("yolo26s.pt"),
        p.uv_exe(false),
        cache.join("coreml/yolo26s-0123456789abcdef-v3.onnx"),
        cache.join("coreml/yolo26n-extra-v3.onnx"),
        cache.join("0123.blob"),
    ];
    for f in ours.iter().chain(&others) {
        put(f);
    }
    assert_eq!(
        export_leftovers(yolo26n(), p, Some(&cache)).len(),
        // The staging, weights and work directories count once each.
        ours.len(),
    );
    remove_export(yolo26n(), p, Some(&cache)).unwrap();
    for f in &ours {
        assert!(!f.exists(), "{} left behind", f.display());
    }
    for d in [
        p.staging("yolo26n"),
        p.weights_of("yolo26n"),
        p.work.clone(),
    ] {
        assert!(!d.exists(), "{} left behind", d.display());
    }
    for f in &others {
        assert!(f.exists(), "{} removed", f.display());
    }
    assert!(export_leftovers(yolo26n(), p, Some(&cache)).is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn linux_uses_the_torch_cpu_index_and_windows_paths() {
    let root = tmp("linux");
    let fake = Fake::new();
    run(&fake, &job(&root, LINUX)).0.unwrap();
    let cmds = fake.cmds();
    assert!(cmds[2].contains("--torch-backend cpu"), "{}", cmds[2]);

    let root = tmp("win");
    let fake = Fake::new();
    let j = job(&root, Platform::WINDOWS_X64);
    run(&fake, &j).0.unwrap();
    let all = fake.commands.lock().unwrap().clone();
    assert!(all[0].display().contains("--no-registry"));
    assert_eq!(all[0].program, root.join("tools/uv/uv.exe"));
    assert_eq!(
        all[3].program,
        root.join("tools/export-env/Scripts/python.exe")
    );
}

#[test]
fn unsupported_platform_fails_before_doing_anything() {
    let root = tmp("unsup");
    let fake = Fake::new();
    let (r, _) = run(&fake, &job(&root, Platform::new("macos", "x86_64")));
    let e = r.unwrap_err();
    assert!(
        e.message.contains("not available on macos/x86_64"),
        "{}",
        e.message
    );
    assert!(!e.cancelled);
    assert!(fake.cmds().is_empty() && fake.fetched.lock().unwrap().is_empty());
    assert!(unsupported_reason("macos", "aarch64").is_none());
    assert!(unsupported_reason("linux", "riscv64").is_some());
}

#[test]
fn failures_name_the_stage_and_leave_no_partial_install() {
    // Package install fails: no marker, so the next export rebuilds the environment.
    let root = tmp("fail-pkgs");
    let fake = Fake {
        fail_on: Some(" pip install "),
        ..Fake::new()
    };
    let j = job(&root, MAC);
    let e = run(&fake, &j).0.unwrap_err();
    assert_eq!(e.stage, Some(Stage::Packages));
    assert!(e.message.contains("something broke"), "{}", e.message);
    assert!(!toolchain_state(&j.paths, MAC).env_ready);
    assert!(!is_exported(yolo26n(), &root.join("models")));

    // Exporter fails: staging removed, nothing installed.
    let fake = Fake {
        fail_on: Some("export_model.py"),
        ..Fake::new()
    };
    let e = run(&fake, &j).0.unwrap_err();
    assert_eq!(e.stage, Some(Stage::Export));
    assert!(!j.paths.staging("yolo26n").exists());
    assert!(!is_exported(yolo26n(), &root.join("models")));

    // Wrong output: refused at install.
    let fake = Fake {
        classes: 3,
        ..Fake::new()
    };
    let e = run(&fake, &j).0.unwrap_err();
    assert_eq!(e.stage, Some(Stage::Install));
    assert!(e.message.contains("3 classes"), "{}", e.message);
    assert!(!is_exported(yolo26n(), &root.join("models")));

    // Weights hash mismatch from the fetcher (fresh root: no weights yet).
    let root = tmp("fail-weights");
    let j = job(&root, MAC);
    let fake = Fake {
        fetch_fails: Some("model:yolo26n"),
        ..Fake::new()
    };
    let e = run(&fake, &j).0.unwrap_err();
    assert_eq!(e.stage, Some(Stage::Weights));
    assert!(e.message.contains("hash mismatch"), "{}", e.message);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn exporter_queues_cancels_and_reports() {
    let root = tmp("exporter");
    let fake = Arc::new(Fake {
        hang_on: Some("export_model.py"),
        ..Fake::new()
    });
    let ex = Exporter::start(fake.clone(), fake.clone());
    let mut j = job(&root, MAC);
    ex.enqueue(ExportRequest {
        job: j.clone(),
        on_done: None,
    })
    .unwrap();
    // Same model again while busy: refused.
    assert!(
        ex.enqueue(ExportRequest {
            job: j.clone(),
            on_done: None
        })
        .is_err()
    );
    // A second model queues behind it.
    j.resource = catalog::export_model("yolo26s").unwrap();
    ex.enqueue(ExportRequest {
        job: j.clone(),
        on_done: None,
    })
    .unwrap();
    assert!(ex.busy());
    // Wait until the first export hangs in the exporter.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let st = ex.status("model:yolo26n").unwrap();
        if let ExportState::Running {
            stage: Stage::Export,
            step,
            steps,
            ..
        } = &st.state
        {
            assert_eq!((*step, *steps), (5, 6));
            assert!(
                st.state
                    .describe()
                    .starts_with("exporting: exporting to ONNX (5/6)")
            );
            assert!(
                st.log.iter().any(|l| l.contains("Downloaded torch")),
                "{:?}",
                st.log
            );
            break;
        }
        assert!(Instant::now() < deadline, "{st:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        ex.status("model:yolo26s").unwrap().state,
        ExportState::Queued
    );
    // Cancel the queued one, then the running one.
    assert!(ex.cancel("model:yolo26s"));
    assert_eq!(
        ex.status("model:yolo26s").unwrap().state,
        ExportState::Cancelled
    );
    assert!(ex.cancel("MODEL:yolo26n"));
    let st = ex
        .wait("model:yolo26n", Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(st.state, ExportState::Cancelled);
    assert!(!ex.busy());
    assert!(!is_exported(yolo26n(), &root.join("models")));
    assert!(!j.paths.staging("yolo26n").exists());
    // Nothing to cancel any more.
    assert!(!ex.cancel("model:yolo26n"));
    ex.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn exporter_failure_and_success_with_callback() {
    let root = tmp("exporter2");
    let fake = Arc::new(Fake {
        fetch_fails: Some("model:yolo26n"),
        ..Fake::new()
    });
    let ex = Exporter::start(fake.clone(), fake.clone());
    let j = job(&root, MAC);
    ex.enqueue(ExportRequest {
        job: j.clone(),
        on_done: None,
    })
    .unwrap();
    let st = ex
        .wait("model:yolo26n", Some(Duration::from_secs(10)))
        .unwrap();
    match &st.state {
        ExportState::Failed { stage, error } => {
            assert_eq!(*stage, Some(Stage::Weights));
            assert!(error.contains("hash mismatch"));
        }
        other => panic!("{other:?}"),
    }
    assert!(
        st.state
            .describe()
            .starts_with("export failed while downloading the weights")
    );
    ex.shutdown();

    // Retry with a working fetcher: success and the callback runs.
    let fake = Arc::new(Fake::new());
    let ex = Exporter::start(fake.clone(), fake);
    let (tx, rx) = crossbeam_channel::bounded(1);
    ex.enqueue(ExportRequest {
        job: j,
        on_done: Some(Box::new(move |o: &ExportOutput| {
            let _ = tx.send(o.onnx.clone());
        })),
    })
    .unwrap();
    let st = ex
        .wait("model:yolo26n", Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(st.state, ExportState::Installed);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        root.join("models/yolo26n.onnx")
    );
    ex.forget("model:yolo26n");
    assert!(ex.status("model:yolo26n").is_none());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn system_runner_streams_fails_times_out_and_cancels() {
    #[cfg(unix)]
    let (sh, flag) = ("/bin/sh", "-c");
    #[cfg(windows)]
    let (sh, flag) = ("cmd", "/C");
    let dir = tmp("runner");
    let cmd = |script: &str, timeout: Duration| Cmd {
        stage: Stage::Export,
        program: sh.into(),
        args: vec![flag.into(), script.into()],
        env: vec![("BOP_TEST".into(), "hello".into())],
        cwd: dir.clone(),
        timeout,
    };
    let never = AtomicBool::new(false);
    let mut lines = Vec::new();
    #[cfg(unix)]
    let ok = "echo one; echo two 1>&2; echo $BOP_TEST";
    #[cfg(windows)]
    let ok = "echo one& echo two 1>&2& echo %BOP_TEST%";
    SystemRunner
        .run(
            &cmd(ok, Duration::from_secs(30)),
            &mut |l| lines.push(l.trim().to_string()),
            &never,
        )
        .unwrap();
    lines.sort();
    assert_eq!(lines, ["hello", "one", "two"]);
    // RS-07: the child's environment is the allowlist plus `Cmd::env`, whatever the
    // service inherited. RS-08: a line longer than MAX_OUTPUT_LINE is cut and marked.
    #[cfg(unix)]
    {
        let mut lines = Vec::new();
        let mut c = Cmd {
            program: "env".into(),
            args: vec![],
            ..cmd("", Duration::from_secs(30))
        };
        c.env.push(("PYTHONPATH".into(), "/explicit".into()));
        SystemRunner
            .run(&c, &mut |l| lines.push(l.to_string()), &never)
            .unwrap();
        assert!(lines.iter().any(|l| l == "BOP_TEST=hello"), "{lines:?}");
        assert!(
            lines.iter().any(|l| l == "PYTHONPATH=/explicit"),
            "{lines:?}"
        );
        assert!(lines.iter().any(|l| l.starts_with("PATH=")), "{lines:?}");
        for l in &lines {
            let key = l.split('=').next().unwrap_or_default();
            assert!(
                inherited(key) || key == "BOP_TEST" || key == "PYTHONPATH",
                "leaked {l}"
            );
        }
        let mut lines = Vec::new();
        let script = format!(
            "head -c {} /dev/zero | tr '\\0' 'x'; echo; echo tail",
            MAX_OUTPUT_LINE * 3
        );
        SystemRunner
            .run(
                &cmd(&script, Duration::from_secs(30)),
                &mut |l| lines.push(l.to_string()),
                &never,
            )
            .unwrap();
        assert_eq!(lines.len(), 2, "{}", lines.len());
        assert_eq!(lines[0].len(), MAX_OUTPUT_LINE + 3);
        assert!(lines[0].ends_with("xxx..."));
        assert_eq!(lines[1], "tail");
    }
    #[cfg(unix)]
    let bad = "echo bad output; exit 3";
    #[cfg(windows)]
    let bad = "echo bad output& exit 3";
    let e = SystemRunner
        .run(&cmd(bad, Duration::from_secs(30)), &mut |_| {}, &never)
        .unwrap_err();
    assert!(format!("{e:#}").contains("bad output"), "{e:#}");
    #[cfg(unix)]
    {
        let started = Instant::now();
        let e = SystemRunner
            .run(
                &cmd("sleep 30", Duration::from_millis(300)),
                &mut |_| {},
                &never,
            )
            .unwrap_err();
        assert!(format!("{e:#}").contains("timed out"), "{e:#}");
        assert!(started.elapsed() < Duration::from_secs(10));
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            c2.store(true, Ordering::Relaxed);
        });
        let e = SystemRunner
            .run(
                &cmd("sleep 30", Duration::from_secs(60)),
                &mut |_| {},
                &cancel,
            )
            .unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        assert!(started.elapsed() < Duration::from_secs(10));

        // RS-02: a grandchild (`sleep 30 &`, whose parent `sh` then waits on another sleep)
        // dies with the process group on cancel.
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            c2.store(true, Ordering::Relaxed);
        });
        let mut lines = Vec::new();
        let e = SystemRunner
            .run(
                &cmd("sleep 30 & echo $!; sleep 30", Duration::from_secs(60)),
                &mut |l| lines.push(l.to_string()),
                &cancel,
            )
            .unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
        let grandchild = lines.first().expect("pid line").trim().to_string();
        assert!(grandchild.parse::<u32>().is_ok(), "{grandchild}");
        let alive = || {
            std::process::Command::new("kill")
                .args(["-0", &grandchild])
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        let until = Instant::now() + Duration::from_secs(5);
        while alive() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!alive(), "grandchild {grandchild} survived the cancel");

        // A helper that outlives a successful child (holding the pipes) is stopped after
        // the grace period instead of hanging the runner.
        let started = Instant::now();
        let mut lines = Vec::new();
        SystemRunner
            .run(
                &cmd("sleep 30 & echo $!; exit 0", Duration::from_secs(60)),
                &mut |l| lines.push(l.to_string()),
                &never,
            )
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));
        let orphan = lines.first().expect("pid line").trim().to_string();
        let alive = || {
            std::process::Command::new("kill")
                .args(["-0", &orphan])
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        let until = Instant::now() + Duration::from_secs(5);
        while alive() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!alive(), "orphan {orphan} survived");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sanitized_variables_and_package_progress() {
    // RS-07: the child starts from an empty environment plus this allowlist.
    for k in [
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "YOLO_CONFIG_DIR",
        "pythonpath",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "XDG_CONFIG_HOME",
        "OMP_NUM_THREADS",
        "HF_HOME",
        "TORCH_HOME",
        "",
    ] {
        assert!(!inherited(k), "{k}");
    }
    for k in [
        "PATH",
        "HOME",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
        "SystemRoot",
        "LANG",
        "LC_ALL",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ] {
        assert!(inherited(k), "{k}");
    }
    let mut p = PackagesProgress::default();
    assert_eq!(p.line("Resolved 48 packages in 3ms"), None);
    assert_eq!(p.line("Downloading torch (75.3MiB)").unwrap().0, 0);
    assert_eq!(p.line("Downloading opencv-python (40MiB)").unwrap().0, 0);
    assert_eq!(
        p.line(" Downloaded torch").unwrap(),
        (42, "downloaded torch".to_string())
    );
    assert_eq!(p.line("Prepared 48 packages in 40s").unwrap().0, 90);
    assert_eq!(p.line("Installed 48 packages in 2s").unwrap().0, 100);
    assert_eq!(Stage::Uv.overall(None), 0);
    assert_eq!(Stage::Packages.overall(Some(50)), 41);
    assert_eq!(Stage::Install.overall(Some(100)), 100);
    assert_eq!(class_count("NAMES:\n  - a\n  - b\n").unwrap(), 2);
    assert_eq!(
        strip_ansi("\u{1b}[34m\u{1b}[1mONNX:\u{1b}[0m export success \u{2705} 3.4s"),
        "ONNX: export success \u{2705} 3.4s"
    );
    assert_eq!(strip_ansi("plain"), "plain");
    assert!(class_count("nope: 1").is_err());
}

#[test]
fn locks_match_the_requirements_and_platforms() {
    let reqs = include_str!("../../../scripts/requirements-model-export.txt");
    for lock in [LOCK_MACOS_ARM64, LOCK_TORCH_CPU_INDEX] {
        for line in reqs.lines().filter(|l| l.contains("==")) {
            let (pkg, ver) = line.split_once("==").unwrap();
            let pinned = lock
                .lines()
                .any(|l| l.starts_with(&format!("{pkg}=={ver}")));
            assert!(pinned, "{pkg}=={ver} missing from a lock");
        }
        // Every requirement is hashed.
        let reqs_in_lock = lock
            .lines()
            .filter(|l| l.contains("==") && !l.trim_start().starts_with('#'))
            .count();
        let hashed = lock.split("==").count() - 1;
        assert_eq!(reqs_in_lock, hashed);
        assert!(lock.contains("--hash=sha256:"));
        assert!(!lock.contains("openvino"), "no OpenVINO Python package");
    }
    assert!(LOCK_TORCH_CPU_INDEX.contains("torch==2.14.0+cpu"));
    assert!(LOCK_MACOS_ARM64.contains("torch==2.14.0 "));
    assert!(reqs.contains(&format!("ultralytics=={}", catalog::ULTRALYTICS_VERSION)));
    assert!(reqs.contains(&format!("transformers=={}", catalog::TRANSFORMERS_VERSION)));
    // Supported platforms agree across the lock, uv and the estimates.
    for p in crate::resources::catalog::SHIPPED_PLATFORMS {
        assert!(package_lock(p.os, p.arch).is_some(), "{p}");
        assert!(unsupported_reason(p.os, p.arch).is_none(), "{p}");
    }
    for e in catalog::TOOLCHAIN_ESTIMATES {
        assert!(package_lock(e.platform.os, e.platform.arch).is_some());
    }
    // The script has the flags the exporter passes.
    for f in [
        "--recipe",
        "--name",
        "--family",
        "--weights",
        "--imgsz",
        "--no-openvino",
        "--out-dir",
        "--work",
        "--source",
        "HF_HUB_OFFLINE",
    ] {
        assert!(EXPORT_SCRIPT.contains(f), "{f}");
    }
}

#[test]
fn needs_export_text() {
    let t = needs_export_message(yolo26n(), Path::new("models/yolo26n.onnx"), false);
    assert!(t.starts_with("needs export: models/yolo26n.onnx is not there yet"));
    assert!(t.contains("fetch --resource model:yolo26n"));
    assert!(t.contains("AGPL-3.0"));
    let rf = catalog::export_model("rfdetr-large").unwrap();
    let t = needs_export_message(rf, Path::new("models/rfdetr-large.onnx"), true);
    assert!(t.contains("older version"), "{t}");
    assert!(t.contains("Apache-2.0"));
}

#[test]
fn deimv2_export_fetches_the_source_and_passes_it() {
    let root = tmp("deimv2");
    let fake = Fake::new();
    let mut j = job(&root, LINUX);
    j.resource = catalog::export_model("deimv2-atto").unwrap();
    let (r, _) = run(&fake, &j);
    let out = r.unwrap();
    assert_eq!(out.onnx, root.join("models/deimv2-atto.onnx"));
    assert_eq!(
        *fake.fetched.lock().unwrap(),
        ["tool:uv", "model:deimv2-atto", "tool:deimv2-src"]
    );
    let all = fake.commands.lock().unwrap().clone();
    let export = all.last().unwrap();
    let d = export.display();
    assert!(
        d.contains("--recipe deimv2 --name deimv2-atto --family rtdetr"),
        "{d}"
    );
    assert!(d.contains("--imgsz 320"), "{d}");
    assert_eq!(
        arg_after(export, "--weights").unwrap(),
        root.join("tools/export-weights/deimv2-atto")
    );
    assert_eq!(
        arg_after(export, "--source").unwrap(),
        root.join("tools/deimv2-src")
    );
    assert!(
        root.join("tools/export-weights/deimv2-atto/model.safetensors")
            .is_file()
    );
    assert!(
        root.join("tools/export-weights/deimv2-atto/config.json")
            .is_file()
    );
    let m = manifest(j.resource, &root.join("models")).unwrap();
    assert_eq!(m.version, j.resource.version);

    // A second DEIMv2 export reuses the source; RF-DETR needs none and passes its directory.
    let fake = Fake::new();
    j.resource = catalog::export_model("deimv2-n").unwrap();
    run(&fake, &j).0.unwrap();
    assert_eq!(*fake.fetched.lock().unwrap(), ["model:deimv2-n"]);
    let fake = Fake::new();
    j.resource = catalog::export_model("rfdetr-large").unwrap();
    run(&fake, &j).0.unwrap();
    assert_eq!(*fake.fetched.lock().unwrap(), ["model:rfdetr-large"]);
    let all = fake.commands.lock().unwrap().clone();
    let d = all.last().unwrap().display();
    assert!(
        d.contains("--recipe rfdetr --name rfdetr-large --family rfdetr"),
        "{d}"
    );
    assert!(d.contains("--imgsz 704") && !d.contains("--source"), "{d}");

    // Remove toolchain: the source goes with it.
    remove_toolchain(&j.paths).unwrap();
    assert!(!root.join("tools").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn outdated_manifest_means_needs_export() {
    let root = tmp("outdated");
    let models = root.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let rf = catalog::export_model("rfdetr-large").unwrap();
    let onnx = models.join("rfdetr-large.onnx");
    // Exported by hand (no manifest): counts.
    std::fs::write(&onnx, b"x").unwrap();
    assert!(is_exported(rf, &models));
    assert!(pending_export(&onnx).is_none());
    // The old onnx-community download of the same name: outdated.
    let old = Manifest {
        id: "model:rfdetr-large".into(),
        version: "4988fbacee4aa815e43c2e0666377ab26bddf1e8".into(),
        sha256: vec!["ab".into()],
        files: vec!["rfdetr-large.onnx".into()],
    };
    std::fs::write(
        manager::manifest_path(rf, &models),
        serde_json::to_vec(&old).unwrap(),
    )
    .unwrap();
    assert!(is_outdated(rf, &models));
    assert!(!is_exported(rf, &models));
    assert_eq!(pending_export(&onnx).unwrap().id, "model:rfdetr-large");
    // Missing file: pending; not an export output (IR, unknown name): none.
    assert!(pending_export(&models.join("yolov8n.onnx")).is_some());
    assert!(pending_export(&models.join("yolov8n.xml")).is_none());
    assert!(pending_export(&models.join("custom.onnx")).is_none());
    // Exporting replaces it.
    let fake = Fake::new();
    let mut j = job(&root, MAC);
    j.resource = rf;
    run(&fake, &j).0.unwrap();
    assert!(is_exported(rf, &models));
    assert!(pending_export(&onnx).is_none());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn legacy_toolchain_is_replaced() {
    let root = tmp("legacy");
    for d in [
        "yolo26-env/bin",
        "yolo26-weights",
        "yolo26-work",
        "yolo26-config",
    ] {
        std::fs::create_dir_all(root.join("tools").join(d)).unwrap();
    }
    std::fs::write(root.join("tools/yolo26-requirements.lock.txt"), b"x").unwrap();
    let fake = Fake::new();
    let j = job(&root, MAC);
    run(&fake, &j).0.unwrap();
    for d in [
        "yolo26-env",
        "yolo26-weights",
        "yolo26-work",
        "yolo26-config",
    ] {
        assert!(!root.join("tools").join(d).exists(), "{d}");
    }
    assert!(!root.join("tools/yolo26-requirements.lock.txt").exists());
    let _ = std::fs::remove_dir_all(&root);
}
