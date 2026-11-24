//! On-demand model export: **the one place in Blue Onyx Prism that executes downloaded code.**
//!
//! Everything else the service downloads is data (libraries are loaded, never run as programs;
//! archives are extracted by whitelist). Some models are only published as PyTorch weights:
//! Ultralytics YOLO (AGPL-3.0, cannot be redistributed), RF-DETR L and DEIMv2 (safetensors). The
//! only way to offer them is to fetch the official weights and export them to ONNX on the user's
//! machine ([`catalog::ExportRecipe`]: Ultralytics' exporter, transformers + `torch.onnx`, or the
//! official DEIMv2 code + `torch.onnx`). That is a deliberate exception with these limits:
//!
//! - **User-triggered only**: the Config page's Export button (with a confirm naming the model and
//!   its license) or `blue-onyx-prism fetch --resource model:<name>`. A config entry pointing at a
//!   missing (or outdated) exported file is reported as "needs export"; `auto_download` never
//!   exports.
//! - **Pinned toolchain**: `tool:uv` (catalog, SHA-256 checked), Python 3.11 installed by that uv,
//!   and the packages of `scripts/model-export/locks/*.txt`, installed with
//!   `--require-hashes --no-deps --only-binary :all:` (every wheel hash-checked, nothing built from
//!   source). On Linux and Windows torch comes from the PyTorch CPU index. On Windows the export
//!   also needs a current Microsoft Visual C++ runtime (PyTorch: `msvcp140.dll` 14.40+); when the
//!   system's is older, `tool:vc-runtime` (Microsoft-signed DLLs from the `msvc-runtime` wheel,
//!   each SHA-256 checked) is copied next to the export's own Python (see [`vc_runtime`]).
//! - **Pinned weights and code**: the catalog's parts (`.pt`, or `model.safetensors` +
//!   `config.json`) downloaded and verified by the download manager into
//!   `tools/export-weights/<name>/`; for DEIMv2 also `tool:deimv2-src`, the official source at a
//!   pinned commit (SHA-256 checked GitHub archive, only `engine/**/*.py` and `configs/**/*.yml`
//!   extracted).
//! - **Pinned script**: `scripts/export_model.py`, embedded in the binary, run as
//!   `python -X faulthandler -X utf8 -u -I` (a native crash prints its stack) offline: `YOLO_OFFLINE=1`,
//!   `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (nothing reaches the network or pip-installs anything), with its
//!   settings and caches in our own `tools/export-config`.
//! - **Contained**: everything lives under `<data root>/tools/` (see [`Paths`]); "Remove export
//!   toolchain" deletes it but the zero-byte lock files, and only while no export, training
//!   setup or training holds `tools/.model-export.lock` or `tools/.train.lock` (`hold_tools`,
//!   [`ToolsBusy`]). Each export runs in a fresh work directory under `tools/work/`.
//!
//! # Trust model of the child processes
//!
//! What runs, in order, and what each step trusts:
//!
//! 1. **Nothing executes before its hash is checked.** The `uv` binary (`tool:uv`), the
//!    Python build uv installs (uv's own verification), every wheel (`--require-hashes` against
//!    the embedded lock), the weights (catalog SHA-256 through the download manager) and, for
//!    DEIMv2, the source archive (`tool:deimv2-src`, SHA-256, whitelist-extracted) are verified
//!    before the stage that uses them starts. The export script is embedded in the binary. On
//!    Windows the Visual C++ runtime DLLs (`tool:vc-runtime`) are loaded, never run: the wheel is
//!    SHA-256 checked, only the pinned DLLs are extracted, each against its own SHA-256 (all
//!    Microsoft-signed, unmodified), and they are copied only into our own `tools/python/<python>`
//!    directory (the venv's `pyvenv.cfg` `home`, refused when it points outside `tools/python`),
//!    each through a temporary file and a rename. System32 is only read.
//! 2. **Minimal environment** (RS-07): the child starts from `env_clear()`; it inherits only
//!    `PATH`, `HOME`/`USERPROFILE`, `SystemRoot` and the other Windows basics (`COMSPEC`,
//!    `PATHEXT`, `APPDATA`, `LOCALAPPDATA`, `PROGRAMDATA`, `PROCESSOR_ARCHITECTURE`/`W6432`, which
//!    Python's `platform.machine()` reads), `TEMP`/`TMP`/`TMPDIR`, `LANG`/`LC_*`,
//!    the proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `ALL_PROXY`, any case) and
//!    `SSL_CERT_FILE`/`SSL_CERT_DIR` when the operator set them (a private CA for that proxy);
//!    see `inherited`. Everything else (`PYTHON*`, `UV_*`, `PIP_*`, `CONDA_*`, `LD_*`,
//!    `DYLD_*`, `REQUESTS_CA_BUNDLE`, `YOLO_*`, `HF_*`, `TORCH_*`, ...) never reaches it. On top
//!    come the exporter's own [`Cmd::env`] entries: `UV_NO_CONFIG`, `UV_PYTHON_INSTALL_DIR`,
//!    the offline switches and the cache dirs under `tools/`; uv also runs `--no-config`.
//! 3. **Whole process tree is killed** (RS-02) on cancel, stage timeout and service stop: on
//!    Unix the child leads its own process group and the group gets `SIGTERM`, then `SIGCONT` +
//!    `SIGKILL` after 1.5 s (through `killpg(2)`, never an external `kill` program); on Windows
//!    the child is started suspended, put in a Job Object with kill-on-close
//!    (`crate::backend::job`) and only then resumed, so every helper uv or PyTorch spawns is in
//!    it and dies with it, also when the service process itself dies (a job that cannot be made
//!    fails the stage instead of running an unprotected tree). The tree also dies with the
//!    service on Unix (audit 2026-11, RA resources-1, recovery-1): Linux sends the child
//!    `SIGKILL` when the service dies (`PR_SET_PDEATHSIG`), every child's stdin is a pipe the
//!    service never writes, whose end of file the scripts' `watch_parent()` turns into a kill of
//!    their process group, and each child is listed in `<data root>/.children.json` for the next
//!    start's sweep. The run's stop cancels exports explicitly (`Provisioner::stop_for_exit`).
//! 4. **Bounded output** (RS-08): stdout and stderr are read line by line (`\n` or `\r` ends a
//!    line), each line is cut at `MAX_OUTPUT_LINE` bytes, and at most `MAX_OUTPUT_LINES`
//!    lines wait between the reader threads and the log; a child that outruns the log blocks
//!    on its pipe. A failed child keeps its last `FAILURE_LINES` (200) lines, which `failure`
//!    classifies (outdated Visual C++ runtime, missing DLL or module, out of memory, disk full,
//!    a silent native crash) into the likely cause and next action; the message quotes the last
//!    8 lines.
//! 5. One export at a time ([`Exporter`] queues the rest; a lock file serializes processes).
//!
//! Stages ([`Stage`]): uv -> Python -> packages -> Visual C++ runtime (Windows; probes System32
//! and the base Python's directory, places the private runtime when needed, never touches the
//! network otherwise) -> self-check (`export_model.py --self-check` imports what the recipe needs
//! and prints the versions, so a broken environment fails in seconds, before the weights) ->
//! weights -> export -> install. The result, `models/<name>.onnx` + `.yaml`, is moved into place
//! from a staging directory and recorded in `models/.<name>.installed.json` like a downloaded
//! model (an older version's manifest makes the file "needs export" again). The staging folder is
//! held busy (`storage::busy`) and removed however the export ends; what a crash or kill leaves
//! (staging and work folders, set-aside files, the manifest temp) is cleaned by the storage sweep
//! (`leftovers`).
//!
//! `run_export` is the stage machine over two seams, [`CommandRunner`] (processes) and
//! [`Fetcher`] (verified downloads), so it is tested without network or Python.

use super::catalog::{self, Platform, Resource, ResourceKind};
use super::manager::{self, Job, Manager, Manifest, State as DownloadState};
use crate::sync::LockExt;
use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The export script (`scripts/export_model.py`), written into the work directory before it runs.
pub(crate) static EXPORT_SCRIPT: &str = include_str!("../../../scripts/export_model.py");
/// File name of the export script in the work directory.
pub(crate) const SCRIPT_NAME: &str = "export_model.py";
/// Hashed package lock for macOS arm64 (PyPI).
pub(crate) static LOCK_MACOS_ARM64: &str =
    include_str!("../../../scripts/model-export/locks/macos-arm64.txt");
/// Hashed package lock for Linux x86_64 / aarch64 and Windows x86_64 (torch from the PyTorch CPU
/// index).
pub(crate) static LOCK_TORCH_CPU_INDEX: &str =
    include_str!("../../../scripts/model-export/locks/torch-cpu-index.txt");

/// Python version of the export environment.
pub const PYTHON_VERSION: &str = "3.11";
/// Directory under the data root that holds the whole toolchain.
pub const TOOLS_DIR: &str = "tools";
/// Marker written into the environment once every package is installed.
pub(crate) const ENV_MARKER: &str = ".blue-onyx-prism-env.json";

/// How long an export waits for another process's export to finish (see [`Timeouts::lock_wait`]).
pub(crate) const EXPORT_LOCK_WAIT: Duration = Duration::from_secs(10 * 60);

/// Stage timeouts (the child process is killed when one passes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    pub python: Duration,
    pub packages: Duration,
    pub export: Duration,
    /// Longest wait for another process's export lock before failing (RS-11).
    pub lock_wait: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            python: Duration::from_secs(15 * 60),
            packages: Duration::from_secs(60 * 60),
            export: Duration::from_secs(60 * 60),
            lock_wait: EXPORT_LOCK_WAIT,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Paths, locks, toolchain state
// ---------------------------------------------------------------------------------------------

/// Where the export toolchain and its outputs live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// `<data root>/tools`.
    pub tools: PathBuf,
    /// `tools/uv` (the `tool:uv` install).
    pub uv_dir: PathBuf,
    /// `tools/export-env`: the Python virtual environment.
    pub env: PathBuf,
    /// `tools/python`: Python installations made by uv.
    pub python_installs: PathBuf,
    /// `tools/uv-cache` (deleted after the packages are installed).
    pub uv_cache: PathBuf,
    /// `tools/export-weights`: verified weights, one directory per model (`<name>/`).
    pub weights: PathBuf,
    /// `tools/export-work`: the script and the exporters' intermediate files (`<name>/`).
    pub work: PathBuf,
    /// `tools/export-config`: `YOLO_CONFIG_DIR` (Ultralytics settings), `HF_HOME`, `TORCH_HOME`.
    pub yolo_config: PathBuf,
    /// `tools/deimv2-src`: the DEIMv2 source code (`tool:deimv2-src`).
    pub deimv2_src: PathBuf,
    /// `tools/vc-runtime`: the private Visual C++ runtime (`tool:vc-runtime`, Windows).
    pub vc_runtime: PathBuf,
    /// The configured models directory (where the `.onnx` / `.yaml` go).
    pub models: PathBuf,
}

impl Paths {
    pub fn new(root: &Path, models: PathBuf) -> Self {
        let tools = root.join(TOOLS_DIR);
        Self {
            uv_dir: root.join(catalog::UV_DEST),
            env: tools.join("export-env"),
            python_installs: tools.join("python"),
            uv_cache: tools.join("uv-cache"),
            weights: tools.join("export-weights"),
            work: tools.join("export-work"),
            yolo_config: tools.join("export-config"),
            deimv2_src: root.join(catalog::DEIMV2_SRC_DEST),
            vc_runtime: root.join(catalog::VC_RUNTIME_DEST),
            tools,
            models,
        }
    }

    /// Paths for `config`: data root and `models_dir`.
    pub fn for_config(config: &crate::config::Config) -> Self {
        Self::new(&config.data_root(), config.data_path(&config.models_dir))
    }

    pub fn uv_exe(&self, windows: bool) -> PathBuf {
        self.uv_dir.join(super::extract::uv_binary_name(windows))
    }

    /// The environment's interpreter.
    pub fn env_python(&self, windows: bool) -> PathBuf {
        if windows {
            self.env.join("Scripts").join("python.exe")
        } else {
            self.env.join("bin").join("python")
        }
    }

    /// Cross-process export lock.
    pub fn lock_file(&self) -> PathBuf {
        self.tools.join(".model-export.lock")
    }

    /// Verified weights of the model `name` (`tools/export-weights/<name>`).
    pub fn weights_of(&self, name: &str) -> PathBuf {
        self.weights.join(name)
    }

    /// Directories of the YOLO26-only export of earlier versions (same layout, older names);
    /// deleted when the current environment is built, and by "Remove export toolchain".
    pub fn legacy_dirs(&self) -> Vec<PathBuf> {
        [
            "yolo26-env",
            "yolo26-weights",
            "yolo26-work",
            "yolo26-config",
        ]
        .iter()
        .map(|d| self.tools.join(d))
        .chain([
            self.tools.join(".yolo26-export.lock"),
            self.tools.join("yolo26-requirements.lock.txt"),
        ])
        .collect()
    }

    /// Staging directory of one export, next to the models (same filesystem: atomic renames).
    pub fn staging(&self, name: &str) -> PathBuf {
        self.models.join(format!(".{name}.export-staging"))
    }

    /// Everything "Remove export toolchain" deletes.
    pub fn toolchain_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.uv_dir.clone(),
            self.env.clone(),
            self.python_installs.clone(),
            self.uv_cache.clone(),
            self.weights.clone(),
            self.work.clone(),
            self.yolo_config.clone(),
            self.deimv2_src.clone(),
            self.vc_runtime.clone(),
        ]
        .into_iter()
        .chain(self.legacy_dirs())
        .collect()
    }
}

/// The package lock for a platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackageLock {
    /// File name under `scripts/model-export/locks/`.
    pub name: &'static str,
    pub text: &'static str,
    /// torch / torchvision come from this PyTorch index (`--torch-backend cpu` for the export on
    /// Linux and Windows, `cu126` / `cu130` for training there); None = PyPI.
    pub torch_backend: Option<&'static str>,
}

/// The package lock for `(os, arch)`; None = models cannot be exported there (no PyTorch wheels).
pub(crate) fn package_lock(os: &str, arch: &str) -> Option<PackageLock> {
    match (os, arch) {
        ("macos", "aarch64") => Some(PackageLock {
            name: "macos-arm64.txt",
            text: LOCK_MACOS_ARM64,
            torch_backend: None,
        }),
        ("linux", "x86_64") | ("linux", "aarch64") | ("windows", "x86_64") => Some(PackageLock {
            name: "torch-cpu-index.txt",
            text: LOCK_TORCH_CPU_INDEX,
            torch_backend: Some("cpu"),
        }),
        _ => None,
    }
}

/// Why models cannot be exported on `(os, arch)`, if they cannot.
pub fn unsupported_reason(os: &str, arch: &str) -> Option<String> {
    let ok = package_lock(os, arch).is_some()
        && catalog::uv_for(os, arch).is_some()
        && catalog::toolchain_estimate(os, arch).is_some();
    (!ok).then(|| {
        format!(
            "model export is not available on {os}/{arch} (no PyTorch 2.14 CPU wheels); export \
             on another machine with scripts/export_model.py and copy the .onnx and .yaml"
        )
    })
}

/// `env/.blue-onyx-prism-env.json`: what the environment was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EnvMarker {
    lock: String,
    lock_sha256: String,
    python: String,
    uv: String,
}

fn marker_for(lock: &PackageLock) -> EnvMarker {
    EnvMarker {
        lock: lock.name.to_string(),
        lock_sha256: manager::sha256_hex(lock.text.as_bytes()),
        python: PYTHON_VERSION.to_string(),
        uv: catalog::UV_VERSION.to_string(),
    }
}

/// The environment is complete and built from this binary's lock, and its base interpreter (the
/// `home` of its `pyvenv.cfg`, under `tools/python`) is still there. A venv's interpreter is a
/// link to, or on Windows a launcher of, that base: without it every Python step fails ("No
/// Python at ..."), so such an environment is rebuilt rather than used (audit 2026-11: "Remove
/// export toolchain" deletes `tools/python`, which `tools/train-env` shares).
fn env_ready(paths: &Paths, platform: Platform, lock: &PackageLock) -> bool {
    let windows = platform.os == "windows";
    let marker: Option<EnvMarker> = std::fs::read(paths.env.join(ENV_MARKER))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    marker.is_some_and(|m| m == marker_for(lock))
        && paths.env_python(windows).is_file()
        && base_python_present(&paths.env)
}

/// The base interpreter's directory named by `env/pyvenv.cfg` (`home = <dir>`) exists.
fn base_python_present(env: &Path) -> bool {
    std::fs::read_to_string(env.join("pyvenv.cfg"))
        .ok()
        .and_then(|text| vc_runtime::pyvenv_home(&text))
        .is_some_and(|home| home.is_dir())
}

/// What of the toolchain is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolchainState {
    /// `tool:uv` is installed (manifest).
    pub uv: bool,
    /// The Python environment is complete for this binary's lock.
    pub env_ready: bool,
    /// Any toolchain directory exists (something to remove).
    pub present: bool,
}

impl ToolchainState {
    /// An export needs no large download (only the weights).
    pub fn ready(&self) -> bool {
        self.uv && self.env_ready
    }
}

pub fn toolchain_state(paths: &Paths, platform: Platform) -> ToolchainState {
    let uv = catalog::uv_for(platform.os, platform.arch)
        .is_some_and(|r| manager::is_installed(r, &paths.uv_dir))
        && paths.uv_exe(platform.os == "windows").is_file();
    let env_ready =
        package_lock(platform.os, platform.arch).is_some_and(|l| env_ready(paths, platform, &l));
    let present = paths.toolchain_dirs().iter().any(|d| d.exists());
    ToolchainState {
        uv,
        env_ready,
        present,
    }
}

/// `res` is exported into `models`: its `.onnx` is there (with our manifest of this version, or
/// exported by hand with the script, i.e. without a manifest).
pub fn is_exported(res: &Resource, models: &Path) -> bool {
    catalog::export_files(res).is_some_and(|(onnx, _)| models.join(onnx).is_file())
        && !is_outdated(res, models)
}

/// A manifest of `res` in `models` records another version: the file there is from an older
/// export or an earlier download under the same name (the onnx-community RF-DETR large v1). An
/// export by an earlier exporter of the same recipe is not outdated
/// ([`catalog::EARLIER_EXPORTERS`]).
pub fn is_outdated(res: &Resource, models: &Path) -> bool {
    manifest(res, models).is_some_and(|m| !catalog::export_version_matches(res.version, &m.version))
}

/// The exportable model whose output `path` (absolute, as the service reads it) is and that still
/// has to be exported: the `.onnx` is missing, or is from another version (see [`is_outdated`]).
pub fn pending_export(path: &Path) -> Option<&'static Resource> {
    let onnx = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    let res = catalog::export_model(&path.to_string_lossy()).filter(|_| onnx)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    (!path.is_file() || is_outdated(res, dir)).then_some(res)
}

/// Our manifest of an export of `res` in `models`.
pub fn manifest(res: &Resource, models: &Path) -> Option<Manifest> {
    manager::read_manifest(&manager::manifest_path(res, models)).filter(|m| m.id == res.id)
}

/// Delete the export toolchain (uv, Python, the environment, weights, work files, the DEIMv2
/// source, the private Visual C++ runtime), holding both tool locks (`hold_tools`); refused
/// with [`ToolsBusy`] while an export, a training environment setup or a training holds one.
/// The training environment's ready marker goes too: `tools/python`, deleted here, is its base
/// interpreter, so the next training rebuilds it. The lock files stay (see `hold_tools`).
pub fn remove_toolchain(paths: &Paths) -> Result<()> {
    let Some(_held) = hold_tools(paths)? else {
        return Ok(());
    };
    for d in paths.toolchain_dirs() {
        manager::remove_any(&d)?;
    }
    manager::remove_any(&paths.tools.join(LOCKFILE_NAME))?;
    manager::remove_any(&paths.tools.join(train::TRAIN_ENV_DIR).join(ENV_MARKER))?;
    Ok(())
}

/// The toolchain or the training environment is in use: an export, a training environment setup
/// or a training (of this process or another) holds `tools/.model-export.lock` or
/// `tools/.train.lock`. Removals refuse with it (HTTP 409); callers tell it apart with
/// `anyhow::Error::is::<ToolsBusy>()`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ToolsBusy(pub String);

fn tools_busy(lock: &Path) -> ToolsBusy {
    ToolsBusy(format!(
        "an export, a training setup or a training is running (in this service or a \
         `blue-onyx-prism fetch` / `train` command) and holds {}; wait for it to finish or \
         cancel it, then remove again",
        lock.display()
    ))
}

/// Both tool locks, held by a removal under `tools/` for its whole length.
#[derive(Debug)]
pub(crate) struct ToolsHeld {
    _train: train::TrainLock,
    _export: ExportLock,
}

/// Take `tools/.train.lock`, then `tools/.model-export.lock` (a training's own order), each with
/// [`train::LOCK_GRACE`] of retries; [`ToolsBusy`] when either stays held. None when there is no
/// `tools` folder (nothing to remove, no lock to create). The lock files are never deleted: a
/// holder keeps its lock on the unlinked file while the next taker creates a new one and also
/// "gets" it.
pub(crate) fn hold_tools(paths: &Paths) -> Result<Option<ToolsHeld>> {
    if !paths.tools.is_dir() {
        return Ok(None);
    }
    let train_lock = paths.tools.join(train::TRAIN_LOCK_NAME);
    let Some(train) = train::TrainLock::try_hold(&train_lock)
        .with_context(|| format!("locking {}", train_lock.display()))?
    else {
        return Err(anyhow!(tools_busy(&train_lock)));
    };
    let export_lock = paths.lock_file();
    let Some(export) = ExportLock::try_acquire(&export_lock)
        .with_context(|| format!("locking {}", export_lock.display()))?
    else {
        return Err(anyhow!(tools_busy(&export_lock)));
    };
    Ok(Some(ToolsHeld {
        _train: train,
        _export: export,
    }))
}

/// Why the toolchain cannot be removed now, from one look at its locks without waiting (for the
/// Resources card and the Remove button's refusal; the removal itself takes them,
/// `hold_tools`). Opens existing lock files only, creates nothing.
pub fn tools_in_use(paths: &Paths) -> Option<String> {
    [paths.tools.join(train::TRAIN_LOCK_NAME), paths.lock_file()]
        .into_iter()
        .find(|p| lock_held_now(p))
        .map(|p| tools_busy(&p).0)
}

/// Another handle holds the fs4 lock on the existing file `path` (one try, no wait).
fn lock_held_now(path: &Path) -> bool {
    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    else {
        return false;
    };
    matches!(
        crate::fsx::lock_wait_with(
            &file,
            crate::fsx::LockKind::Exclusive,
            None,
            crate::fsx::Pauses::fixed(Duration::ZERO),
            || false,
        ),
        Ok(crate::fsx::Waited::Stopped)
    )
}

/// OpenVINO IR of an export of `res` (`<name>.xml`, `<name>.bin`): written into the models
/// directory by earlier versions and by `scripts/export_model.py` without `--no-openvino`.
pub fn ir_files(res: &Resource) -> Option<(String, String)> {
    catalog::export_files(res)?;
    res.model_name()
        .map(|n| (format!("{n}.xml"), format!("{n}.bin")))
}

/// `name` is a CoreML rewrite of the model `stem` (`<stem>-<16 hex>-v<N>.onnx`, any rewriter
/// version; see `onnx_rewrite::cache_file_name`).
fn is_coreml_copy_of(name: &str, stem: &str) -> bool {
    let Some(rest) = name
        .strip_prefix(stem)
        .and_then(|r| r.strip_prefix('-'))
        .and_then(|r| r.strip_suffix(".onnx"))
    else {
        return false;
    };
    let Some((sha, v)) = rest.split_once("-v") else {
        return false;
    };
    sha.len() == 16
        && sha.bytes().all(|b| b.is_ascii_hexdigit())
        && !v.is_empty()
        && v.bytes().all(|b| b.is_ascii_digit())
}

/// Everything on disk that belongs to the export of `res` alone and exists: the `.onnx`, `.yaml`,
/// manifest and IR in the models directory, files an interrupted install set aside, the staging
/// and work directories, the downloaded weights (current and legacy layout) and the CoreML
/// rewrites in `cache_dir`. The shared toolchain is not part of it.
pub fn export_leftovers(res: &Resource, paths: &Paths, cache_dir: Option<&Path>) -> Vec<PathBuf> {
    let (Some((onnx, yaml)), Some(name)) = (catalog::export_files(res), res.model_name()) else {
        return Vec::new();
    };
    let models = &paths.models;
    let mut files = vec![
        models.join(&onnx),
        models.join(&yaml),
        manager::manifest_path(res, models),
        paths.staging(name),
        paths.weights_of(name),
        paths.work.join(name),
        paths
            .tools
            .join("yolo26-weights")
            .join(format!("{name}.pt")),
    ];
    if let Some((xml, bin)) = ir_files(res) {
        files.push(models.join(xml));
        files.push(models.join(bin));
    }
    // `.<file>.old-<pid>` from `InstallTx::put` and the manifest's `.json.tmp`.
    let aside = [format!(".{onnx}.old-"), format!(".{yaml}.old-")];
    let tmp = manager::manifest_path(res, models).with_extension("json.tmp");
    files.push(tmp);
    if let Ok(rd) = std::fs::read_dir(models) {
        files.extend(
            rd.flatten()
                .filter(|e| {
                    let n = e.file_name();
                    let n = n.to_string_lossy();
                    aside.iter().any(|a| n.starts_with(a.as_str()))
                })
                .map(|e| e.path()),
        );
    }
    if let Some(Ok(rd)) = cache_dir.map(|c| std::fs::read_dir(c.join("coreml"))) {
        files.extend(
            rd.flatten()
                .filter(|e| is_coreml_copy_of(&e.file_name().to_string_lossy(), name))
                .map(|e| e.path()),
        );
    }
    files.retain(|p| std::fs::symlink_metadata(p).is_ok());
    files
}

/// Delete everything of the export of `res` ([`export_leftovers`]), then the per-model parent
/// directories that are left empty.
pub fn remove_export(res: &Resource, paths: &Paths, cache_dir: Option<&Path>) -> Result<()> {
    if catalog::export_files(res).is_none() {
        bail!("{} is not exportable", res.id);
    }
    for p in export_leftovers(res, paths, cache_dir) {
        manager::remove_any(&p)?;
    }
    // Only when nothing else is in there.
    for d in [
        &paths.weights,
        &paths.work,
        &paths.tools.join("yolo26-weights"),
    ] {
        let _ = std::fs::remove_dir(d);
    }
    Ok(())
}

/// Exclusive cross-process export lock; released on drop.
#[derive(Debug)]
struct ExportLock {
    _file: std::fs::File,
}

impl ExportLock {
    /// Take the lock if it is free; None when an export or a training environment setup still
    /// holds it after [`train::LOCK_GRACE`] of retries (a child spawned while a lock file is open
    /// shares it until it execs, so a lock just released can look held for a moment; a real
    /// holder keeps it for the whole export). The lock file is created in an existing folder only.
    fn try_acquire(path: &Path) -> std::io::Result<Option<Self>> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let waited = crate::fsx::lock_wait_with(
            &file,
            crate::fsx::LockKind::Exclusive,
            crate::fsx::deadline_after(train::LOCK_GRACE),
            crate::fsx::Pauses::fixed(Duration::from_millis(20)),
            || true,
        )?;
        Ok(matches!(waited, crate::fsx::Waited::Held).then_some(Self { _file: file }))
    }

    fn acquire(
        path: &Path,
        cancel: &AtomicBool,
        wait: Duration,
        mut waiting: impl FnMut(),
    ) -> Result<Self> {
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
        let mut told = false;
        let waited = crate::fsx::lock_wait_with(
            &file,
            crate::fsx::LockKind::Exclusive,
            crate::fsx::deadline_after(wait),
            crate::fsx::Pauses::fixed(Duration::from_millis(250)),
            || {
                if !told {
                    waiting();
                    told = true;
                }
                !cancel.load(Ordering::Relaxed)
            },
        )
        .with_context(|| format!("locking {}", path.display()))?;
        match waited {
            crate::fsx::Waited::Held => Ok(Self { _file: file }),
            crate::fsx::Waited::Stopped => Err(anyhow!(Cancelled)),
            crate::fsx::Waited::TimedOut => bail!(
                "another process (the service or a `fetch` command) has been exporting a \
                 model for over {} min and holds {}; wait for it to finish (the Config \
                 page shows its progress) or cancel it, then try again",
                wait.as_secs() / 60,
                path.display()
            ),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stages and progress
// ---------------------------------------------------------------------------------------------

/// Export stages, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Download and verify `tool:uv`.
    Uv,
    /// `uv python install` + `uv venv`.
    Python,
    /// `uv pip install --require-hashes` of the lock.
    Packages,
    /// Windows: make sure the Visual C++ runtime PyTorch loads is new enough (places the private
    /// `tool:vc-runtime` next to the export's Python when not); nothing elsewhere.
    Runtime,
    /// `export_model.py --self-check`: import what the recipe needs and print the versions.
    SelfCheck,
    /// Download and verify the weights (and the DEIMv2 source).
    Weights,
    /// Run the export script.
    Export,
    /// Check the output and move it into the models directory.
    Install,
}

impl Stage {
    pub const ALL: [Stage; 8] = [
        Stage::Uv,
        Stage::Python,
        Stage::Packages,
        Stage::Runtime,
        Stage::SelfCheck,
        Stage::Weights,
        Stage::Export,
        Stage::Install,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Stage::Uv => "downloading uv",
            Stage::Python => "installing Python 3.11",
            Stage::Packages => "installing the pinned Python packages",
            Stage::Runtime => "checking the Visual C++ runtime",
            Stage::SelfCheck => "checking the Python environment",
            Stage::Weights => "downloading the weights",
            Stage::Export => "exporting to ONNX",
            Stage::Install => "installing the model files",
        }
    }

    /// 1-based position.
    pub fn number(self) -> usize {
        Stage::ALL.iter().position(|s| *s == self).unwrap_or(0) + 1
    }

    /// Share of the overall progress bar: `[start, end)` percent.
    fn span(self) -> (u8, u8) {
        match self {
            Stage::Uv => (0, 4),
            Stage::Python => (4, 12),
            Stage::Packages => (12, 70),
            Stage::Runtime => (70, 71),
            Stage::SelfCheck => (71, 73),
            Stage::Weights => (73, 79),
            Stage::Export => (79, 97),
            Stage::Install => (97, 100),
        }
    }

    /// Overall percent when this stage is `pct` percent done (None = just started).
    pub fn overall(self, pct: Option<u8>) -> u8 {
        let (a, b) = self.span();
        let p = u32::from(pct.unwrap_or(0).min(100));
        a + u8::try_from((u32::from(b - a) * p) / 100).unwrap_or(b - a)
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// What `run_export` reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress<'a> {
    /// Entered or advanced a stage (`percent` of the stage when known).
    Stage {
        stage: Stage,
        percent: Option<u8>,
        detail: &'a str,
    },
    /// One line of a child process's output.
    Line { stage: Stage, line: &'a str },
}

/// The export was cancelled.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Why an export failed, and in which stage.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub(crate) struct ExportError {
    pub stage: Option<Stage>,
    pub message: String,
    pub cancelled: bool,
}

/// Result of a successful export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOutput {
    pub onnx: PathBuf,
    pub yaml: PathBuf,
    /// Number of class names written.
    pub classes: usize,
}

mod exporter;
mod failure;
#[cfg(test)]
pub(crate) mod fake;
mod leftovers;
mod seams;
mod stages;
#[cfg(test)]
mod tests;
pub mod train;
pub mod vc_runtime;

pub use exporter::*;
pub(crate) use leftovers::{export_staging_stem, sweep_leftovers};
pub use seams::*;
pub use stages::*;
