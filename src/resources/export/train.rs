//! The training environment: the export's trust model ([`super`]: pinned uv, Python 3.11 and
//! hashed wheels, minimal child environment, process-tree kill, bounded output) applied to a
//! second virtual environment, `tools/train-env`, with GPU builds of PyTorch and the trainers
//! (`scripts/requirements-model-train.txt`, locks `scripts/model-export/locks/train-*.txt`).
//! [`crate::training`] runs the stages; this module only knows the files, the locks and the
//! commands.
//!
//! - **Locks**: macOS arm64 (PyPI, Apple MPS); Windows / Linux x86_64 with the PyTorch CUDA 12.6
//!   builds (compute capability 6.1 to 9.0) or CUDA 13.0 builds (Blackwell, 10.0 / 12.0, needs a
//!   CUDA 13 driver). [`TrainLane`] picks one; [`crate::training::capability`] picks the lane.
//! - **Shared with the export**: `tool:uv`, `tools/python` (uv's Python installs), the uv cache
//!   and the Ultralytics base weights (`tools/export-weights/<name>`). The environment is built
//!   by [`super::prepare_env`] under the export lock, so an export and a training setup never
//!   run uv at the same time; the training itself holds its own lock (`tools/.train.lock`) so
//!   one training runs at a time across processes while exports go on.
//! - **Extending**: the import / auto-label stages add their packages to
//!   `scripts/requirements-model-train.txt` and regenerate the locks; the environment marker
//!   (lock name + SHA-256) makes every existing environment rebuild once.

use super::*;

/// The training script (`scripts/train_model.py`), written into the work directory before it runs.
pub(crate) static TRAIN_SCRIPT: &str = include_str!("../../../scripts/train_model.py");
/// File name of the training script in the work directory.
pub(crate) const TRAIN_SCRIPT_NAME: &str = "train_model.py";

static LOCK_TRAIN_MACOS_ARM64: &str =
    include_str!("../../../scripts/model-export/locks/train-macos-arm64.txt");
static LOCK_TRAIN_CU126_WINDOWS: &str =
    include_str!("../../../scripts/model-export/locks/train-cu126-windows-x86_64.txt");
static LOCK_TRAIN_CU126_LINUX: &str =
    include_str!("../../../scripts/model-export/locks/train-cu126-linux-x86_64.txt");
static LOCK_TRAIN_CU130_WINDOWS: &str =
    include_str!("../../../scripts/model-export/locks/train-cu130-windows-x86_64.txt");
static LOCK_TRAIN_CU130_LINUX: &str =
    include_str!("../../../scripts/model-export/locks/train-cu130-linux-x86_64.txt");

/// Which PyTorch build the training environment gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainLane {
    /// macOS arm64, PyTorch from PyPI (Metal Performance Shaders).
    Mps,
    /// PyTorch CUDA 12.6 builds: kernels for compute capability 5.0-9.0 (Windows also 6.1);
    /// any driver with CUDA 12.
    Cu126,
    /// PyTorch CUDA 13.0 builds: 7.5-12.0 (Blackwell); a CUDA 13 driver.
    Cu130,
}

impl TrainLane {
    pub const fn as_str(self) -> &'static str {
        match self {
            TrainLane::Mps => "mps",
            TrainLane::Cu126 => "cu126",
            TrainLane::Cu130 => "cu130",
        }
    }

    /// Download of the environment (uv, Python and the wheels; rough, from the lock's wheel
    /// sizes) and disk use afterwards.
    pub const fn estimate(self) -> (u64, u64) {
        match self {
            TrainLane::Mps => (450_000_000, 1_500_000_000),
            TrainLane::Cu126 | TrainLane::Cu130 => (3_200_000_000, 6_500_000_000),
        }
    }
}

/// The lock of `lane` on `(os, arch)`; None = no training environment there.
pub(crate) fn train_lock(os: &str, arch: &str, lane: TrainLane) -> Option<PackageLock> {
    let (name, text, backend) = match (os, arch, lane) {
        ("macos", "aarch64", TrainLane::Mps) => {
            ("train-macos-arm64.txt", LOCK_TRAIN_MACOS_ARM64, None)
        }
        ("windows", "x86_64", TrainLane::Cu126) => (
            "train-cu126-windows-x86_64.txt",
            LOCK_TRAIN_CU126_WINDOWS,
            Some("cu126"),
        ),
        ("linux", "x86_64", TrainLane::Cu126) => (
            "train-cu126-linux-x86_64.txt",
            LOCK_TRAIN_CU126_LINUX,
            Some("cu126"),
        ),
        ("windows", "x86_64", TrainLane::Cu130) => (
            "train-cu130-windows-x86_64.txt",
            LOCK_TRAIN_CU130_WINDOWS,
            Some("cu130"),
        ),
        ("linux", "x86_64", TrainLane::Cu130) => (
            "train-cu130-linux-x86_64.txt",
            LOCK_TRAIN_CU130_LINUX,
            Some("cu130"),
        ),
        _ => return None,
    };
    Some(PackageLock {
        name,
        text,
        torch_backend: backend,
    })
}

/// Where the training environment and its files live (all under `<data root>/tools`, except the
/// runs, which go to the training folder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainPaths {
    /// The export's paths with `env` = `tools/train-env` (what [`super::prepare_env`] builds).
    pub export: Paths,
    /// `tools/train-weights`: base weights other than the Ultralytics `.pt` files.
    pub weights: PathBuf,
    /// `tools/train-config`: `YOLO_CONFIG_DIR`, `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`.
    pub config: PathBuf,
    /// `tools/train-work`: the scripts written before a run.
    pub scripts: PathBuf,
}

impl TrainPaths {
    pub fn new(root: &Path, models: PathBuf) -> Self {
        let mut export = Paths::new(root, models);
        export.env = export.tools.join("train-env");
        Self {
            weights: root.join(catalog::TRAIN_WEIGHTS_DEST),
            config: export.tools.join("train-config"),
            scripts: export.tools.join("train-work"),
            export,
        }
    }

    pub fn for_config(config: &crate::config::Config) -> Self {
        Self::new(&config.data_root(), config.data_path(&config.models_dir))
    }

    /// Directory the base weights `res` are installed in: the export's weights directory for the
    /// Ultralytics pins (shared with exports), `tools/train-weights/<key>` otherwise.
    pub fn weights_dir(&self, res: &Resource) -> PathBuf {
        match res.model_name() {
            Some(name) => self.export.weights_of(name),
            None => self.weights.join(
                res.id
                    .strip_prefix(catalog::TRAIN_BASE_PREFIX)
                    .unwrap_or(res.id),
            ),
        }
    }

    /// Cross-process training lock (one training at a time).
    pub fn train_lock_file(&self) -> PathBuf {
        self.export.tools.join(".train.lock")
    }

    /// What "Remove training environment" deletes (the shared uv / Python stay for the export).
    pub fn removable(&self) -> Vec<PathBuf> {
        vec![
            self.export.env.clone(),
            self.weights.clone(),
            self.config.clone(),
            self.scripts.clone(),
        ]
    }
}

/// The training environment is complete for `lock`.
pub fn train_env_ready(paths: &TrainPaths, platform: Platform, lane: TrainLane) -> bool {
    train_lock(platform.os, platform.arch, lane)
        .is_some_and(|l| env_ready(&paths.export, platform, &l))
}

/// Delete the training environment, its base weights and settings (not uv or Python, which the
/// export shares, nor the training runs).
pub fn remove_train_env(paths: &TrainPaths) -> Result<()> {
    for d in paths.removable() {
        manager::remove_any(&d)?;
    }
    let _ = std::fs::remove_file(paths.train_lock_file());
    Ok(())
}

/// An [`ExportJob`] for the environment stages of a training (`resource`: the run's base
/// weights; [`super::prepare_env`] does not use it).
pub fn env_job(paths: &TrainPaths, resource: &'static Resource, platform: Platform) -> ExportJob {
    ExportJob {
        resource,
        paths: paths.export.clone(),
        platform,
        timeouts: Timeouts::default(),
        system_dir: vc_runtime::system_dir(),
    }
}

/// Set up `tools/train-env` for `lane` (uv, Python, packages, Visual C++ runtime) under the export
/// lock. Reports the export's [`Stage`]s. Returns System32's `msvcp140.dll` version once probed.
pub(crate) fn prepare_train_env(
    job: &ExportJob,
    lane: TrainLane,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
) -> Result<Option<vc_runtime::FileVersion>, ExportError> {
    let mut current = None;
    let (os, arch) = (job.platform.os, job.platform.arch);
    let result = (|| -> Result<Option<vc_runtime::FileVersion>> {
        let lock = train_lock(os, arch, lane).ok_or_else(|| {
            anyhow!(
                "no training environment for {os}/{arch} with {}: training needs an NVIDIA GPU on \
                 Windows or Linux x86_64, or Apple silicon",
                lane.as_str()
            )
        })?;
        std::fs::create_dir_all(&job.paths.tools)
            .with_context(|| format!("creating {}", job.paths.tools.display()))?;
        current = Some(Stage::Uv);
        report(Progress::Stage {
            stage: Stage::Uv,
            percent: None,
            detail: "checking the toolchain",
        });
        let _lock = ExportLock::acquire(
            &job.paths.lock_file(),
            cancel,
            job.timeouts.lock_wait,
            || {
                tracing::info!("a model export is running; the training setup waits for it");
            },
        )?;
        prepare_env(
            job,
            &lock,
            false,
            runner,
            fetcher,
            report,
            cancel,
            &mut current,
        )
    })();
    result.map_err(|e| {
        let cancelled = e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed);
        ExportError {
            stage: current,
            message: if cancelled {
                "cancelled".to_string()
            } else {
                format!(
                    "{:#}",
                    failure::explain(
                        e,
                        &failure::FailureContext {
                            windows: os == "windows",
                            system_msvcp140: None,
                            tools: &job.paths.tools,
                            python_script: false,
                        },
                    )
                )
            },
            cancelled,
        }
    })
}

/// Exclusive cross-process training lock; released on drop.
#[derive(Debug)]
pub struct TrainLock {
    _file: std::fs::File,
}

impl TrainLock {
    /// Take the lock, or fail at once naming the holder's likely identity.
    pub fn try_acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(std::fs::TryLockError::WouldBlock) => bail!(
                "another process (the service or a `blue-onyx-prism train` command) is training \
                 a model and holds {}; wait for it to finish or cancel it, then try again",
                path.display()
            ),
            Err(std::fs::TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("locking {}", path.display()))
            }
        }
    }
}

/// The training environment's Python running `args` as `python -X faulthandler -u -I <args>`,
/// offline (`YOLO_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`), with its settings and
/// caches in `tools/train-config` and the MPS CPU fallback on (ops MPS lacks run on the CPU).
pub fn train_python_cmd(
    paths: &TrainPaths,
    platform: Platform,
    stage: Stage,
    cwd: &Path,
    timeout: Duration,
    args: Vec<OsString>,
) -> Cmd {
    let cfg = |sub: &str| paths.config.join(sub).into_os_string();
    let mut all: Vec<OsString> = vec!["-X".into(), "faulthandler".into(), "-u".into(), "-I".into()];
    all.extend(args);
    Cmd {
        stage,
        program: paths.export.env_python(platform.os == "windows"),
        args: all,
        env: vec![
            ("YOLO_OFFLINE".into(), "1".into()),
            ("YOLO_AUTOINSTALL".into(), "false".into()),
            ("YOLO_CONFIG_DIR".into(), cfg("ultralytics")),
            ("HF_HUB_OFFLINE".into(), "1".into()),
            ("TRANSFORMERS_OFFLINE".into(), "1".into()),
            ("HF_HUB_DISABLE_TELEMETRY".into(), "1".into()),
            ("HF_HOME".into(), cfg("huggingface")),
            ("TORCH_HOME".into(), cfg("torch")),
            ("MPLCONFIGDIR".into(), cfg("matplotlib")),
            ("MPLBACKEND".into(), "Agg".into()),
            ("PYTORCH_ENABLE_MPS_FALLBACK".into(), "1".into()),
            // rfdetr / Lightning telemetry and update checks.
            ("RF_HOME".into(), cfg("rfdetr")),
            ("WANDB_MODE".into(), "disabled".into()),
            ("PL_DISABLE_FORK".into(), "1".into()),
        ],
        cwd: cwd.to_path_buf(),
        timeout,
    }
}

/// Write the training and export scripts into `tools/train-work`; returns their paths.
pub fn write_scripts(paths: &TrainPaths) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(&paths.scripts)
        .with_context(|| format!("creating {}", paths.scripts.display()))?;
    std::fs::create_dir_all(&paths.config)
        .with_context(|| format!("creating {}", paths.config.display()))?;
    let train = paths.scripts.join(TRAIN_SCRIPT_NAME);
    let export = paths.scripts.join(SCRIPT_NAME);
    std::fs::write(&train, TRAIN_SCRIPT).with_context(|| format!("writing {}", train.display()))?;
    std::fs::write(&export, EXPORT_SCRIPT)
        .with_context(|| format!("writing {}", export.display()))?;
    Ok((train, export))
}

/// `e` with the likely cause and next action of a recognized child failure (out of memory, a
/// missing DLL, disk full, ...), for a Python script of the training environment.
pub fn explain_train_failure(
    e: anyhow::Error,
    paths: &TrainPaths,
    platform: Platform,
) -> anyhow::Error {
    failure::explain(
        e,
        &failure::FailureContext {
            windows: platform.os == "windows",
            system_msvcp140: None,
            tools: &paths.export.tools,
            python_script: true,
        },
    )
}

#[cfg(test)]
mod train_env_tests {
    use super::super::fake::Fake;
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-train-env-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn locks_exist_per_lane_and_are_hashed() {
        for (os, arch, lane, backend) in [
            ("macos", "aarch64", TrainLane::Mps, None),
            ("windows", "x86_64", TrainLane::Cu126, Some("cu126")),
            ("linux", "x86_64", TrainLane::Cu126, Some("cu126")),
            ("windows", "x86_64", TrainLane::Cu130, Some("cu130")),
            ("linux", "x86_64", TrainLane::Cu130, Some("cu130")),
        ] {
            let l = train_lock(os, arch, lane).unwrap();
            assert_eq!(l.torch_backend, backend);
            assert!(l.text.contains("--hash=sha256:"), "{}", l.name);
            for pkg in [
                "ultralytics==",
                "rfdetr==",
                "transformers==",
                "accelerate==",
                "onnxruntime==",
            ] {
                assert!(l.text.contains(pkg), "{} lacks {pkg}", l.name);
            }
            // One OpenCV build only (a second would overwrite cv2).
            assert!(!l.text.contains("opencv-python-headless=="), "{}", l.name);
            if let Some(b) = backend {
                assert!(l.text.contains(&format!("torch==2.14.0+{b}")), "{}", l.name);
            }
        }
        assert!(train_lock("macos", "aarch64", TrainLane::Cu126).is_none());
        assert!(train_lock("linux", "aarch64", TrainLane::Cu126).is_none());
        assert!(train_lock("windows", "x86_64", TrainLane::Mps).is_none());
    }

    #[test]
    fn paths_share_uv_and_python_with_the_export() {
        let root = Path::new("/data");
        let p = TrainPaths::new(root, root.join("models"));
        let e = Paths::new(root, root.join("models"));
        assert_eq!(p.export.env, root.join("tools/train-env"));
        assert_eq!(p.export.uv_dir, e.uv_dir);
        assert_eq!(p.export.python_installs, e.python_installs);
        assert_eq!(p.export.lock_file(), e.lock_file());
        let yolo = catalog::export_model("yolo26n").unwrap();
        assert_eq!(p.weights_dir(yolo), e.weights_of("yolo26n"));
        let rf = catalog::base_weights(catalog::TrainFamily::Rfdetr, "nano").unwrap();
        assert_eq!(
            p.weights_dir(rf),
            root.join("tools/train-weights/rfdetr-nano")
        );
        assert!(!p.removable().contains(&e.uv_dir));
        assert!(!p.removable().contains(&e.python_installs));
    }

    #[test]
    fn env_setup_installs_the_cuda_lock_with_its_torch_index() {
        let root = tmp("setup");
        let p = TrainPaths::new(&root, root.join("models"));
        let fake = Fake::new();
        let res = catalog::export_model("yolo26n").unwrap();
        let job = env_job(&p, res, Platform::LINUX_X64);
        let cancel = AtomicBool::new(false);
        let mut stages = Vec::new();
        prepare_train_env(
            &job,
            TrainLane::Cu126,
            &fake,
            &fake,
            &mut |pr| {
                if let Progress::Stage { stage, .. } = pr {
                    stages.push(stage);
                }
            },
            &cancel,
        )
        .unwrap();
        let cmds = fake.cmds();
        let pip = cmds.iter().find(|c| c.contains(" pip install ")).unwrap();
        assert!(pip.contains("--torch-backend cu126"), "{pip}");
        assert!(pip.contains("--require-hashes"), "{pip}");
        assert!(
            pip.contains(&root.join("tools/train-env").display().to_string()),
            "{pip}"
        );
        assert!(train_env_ready(&p, Platform::LINUX_X64, TrainLane::Cu126));
        assert!(!train_env_ready(&p, Platform::LINUX_X64, TrainLane::Cu130));
        assert!(stages.contains(&Stage::Packages));
        // The export environment is untouched.
        assert!(!Paths::new(&root, root.join("models")).env.exists());
        // A second setup reuses it.
        let n = fake.runs.load(Ordering::SeqCst);
        prepare_train_env(&job, TrainLane::Cu126, &fake, &fake, &mut |_| {}, &cancel).unwrap();
        assert_eq!(fake.runs.load(Ordering::SeqCst), n);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn train_lock_is_exclusive() {
        let root = tmp("lock");
        let path = root.join(".train.lock");
        let held = TrainLock::try_acquire(&path).unwrap();
        let e = TrainLock::try_acquire(&path).unwrap_err();
        assert!(format!("{e:#}").contains("another process"), "{e:#}");
        drop(held);
        TrainLock::try_acquire(&path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn python_cmd_is_offline_with_the_mps_fallback() {
        let p = TrainPaths::new(Path::new("/d"), PathBuf::from("/d/models"));
        let c = train_python_cmd(
            &p,
            Platform::MACOS_ARM64,
            Stage::Export,
            Path::new("/w"),
            Duration::from_secs(5),
            vec!["x.py".into()],
        );
        assert_eq!(c.program, PathBuf::from("/d/tools/train-env/bin/python"));
        assert_eq!(
            c.args[..4],
            ["-X", "faulthandler", "-u", "-I"].map(OsString::from)
        );
        let get = |k: &str| c.env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("HF_HUB_OFFLINE"), Some("1".into()));
        assert_eq!(get("YOLO_OFFLINE"), Some("1".into()));
        assert_eq!(get("PYTORCH_ENABLE_MPS_FALLBACK"), Some("1".into()));
        assert_eq!(
            get("HF_HOME"),
            Some(OsString::from("/d/tools/train-config/huggingface"))
        );
    }
}
