//! ONNX Runtime backend (Cargo feature `onnxruntime`), built on the `ort` 2.x crate with
//! `load-dynamic`: the ONNX Runtime shared library is opened at startup from config
//! `onnxruntime_dir`, `ORT_DYLIB_PATH` or `<exe_dir>/onnxruntime` (see
//! `libs::find_onnxruntime`), never linked. All `ort` crate code lives in this file.
//!
//! - [`OrtRuntime::new`] loads the library once per process (a process can hold only one ONNX
//!   Runtime) and probes its execution providers into an [`OrtProbe`]: `is_available()` per EP,
//!   plus for CUDA/TensorRT whether the provider library, CUDA runtime and cuDNN can be loaded
//!   (they stay loaded, BE-02) and whether TensorRT's `nvinfer` exists, so the reason names what
//!   is missing. `nvinfer`, the largest image, is loaded by path only when a TensorRT session is
//!   built, just before its provider registers (`DepCheck`, RA-086).
//! - [`OrtRuntime::compile`] creates a session on exactly one EP (registration errors are
//!   errors, not a silent CPU fallback; the CPU fallback is the caller's candidate loop):
//!   CUDA `device_id`; TensorRT fp16 with its engine and timing cache in the model's own
//!   `cache/tensorrt/<stem>-<key16>/` ([`crate::storage::caches::model_cache_dir`]: a retrained
//!   file at the same path gets a new folder, since ONNX Runtime keys an engine by file and graph
//!   names, not weights; CUDA behind it for unsupported nodes); DirectML `device_id`, memory
//!   pattern off, sequential execution; CoreML MLProgram, compute units ALL, cache in
//!   `cache/coreml`; `intra_threads` everywhere (see [`super::cpu_threads`]) and thread spinning
//!   off (`BOP_ORT_SPINNING=1` turns it on). CPU sessions keep ONNX Runtime's CPU arena;
//!   `BOP_ORT_CPU_ARENA=0` turns it off, for measurement only (`ENV_ORT_CPU_ARENA`).
//! - The image input size is checked at load (`super::device::image_input_size`) on the
//!   pinned shape before any accelerator session is built, so an oversized model fails to load
//!   instead of allocating a huge frame later.
//! - [`LoadRequest::no_cache`] (a throwaway model such as a dataset check's fold model): no
//!   TensorRT engine or timing cache, no CoreML model cache, and the CoreML rewrite below is kept
//!   in memory (the session loads from its bytes), so nothing of it reaches `cache/`.
//! - CoreML loads a rewritten copy of the model when it has ops the CoreML EP cannot take but
//!   that have exact equivalents (HardSigmoid/HardSwish, opset < 13 Split; see
//!   [`super::onnx_rewrite`]), cached as `cache/coreml/<stem>-<sha256/16>-v<N>.onnx`. YOLOv5 then
//!   runs as one CoreML partition instead of ~50 (6x faster). `BOP_COREML_REWRITE=0` turns it off.
//!   [`ModelInfo::path`] stays the original file. A CoreML compile ends with
//!   [`super::memory::trim_soon`] (the rewrite and the conversion free a lot of heap).
//! - Named dynamic dimensions of the image input and RT-DETR's `orig_target_sizes` are pinned
//!   to `[1,3,640,640]` / `[1,2]` with free-dimension overrides (the same defaults as the
//!   OpenVINO reshape), so the ports reported in [`ModelInfo`] are static and `make_family`
//!   auto-detection is unchanged. Pinning is also what lets RT-DETR run on CoreML: an RT-DETR
//!   model that keeps an unpinnable dynamic input dimension is refused there (see
//!   `unsupported_on`).
//! - [`OrtBackend`] runs each session (`ort_sessions`, one model copy each) on its own thread,
//!   one inference in flight per session (`Session::run` takes `&mut self`): `submit` copies the
//!   CHW image into the slot's buffer and hands it over, the thread converts extra inputs to the
//!   port's element type (i64/i32/f32), runs, copies every output into the slot's reused
//!   [`NamedOutput`] buffers and signals a [`Completion`]. The model's [`vram::Hold`] is shared
//!   by the backend and every session thread, so a thread detached while hung keeps the model
//!   listed as a holder of the GPU memory its session still has.

use super::detect::{GpuVendor, HardwareInfo};
#[cfg(test)]
use super::device::DEFAULT_IMAGE_SHAPE;
use super::device::{
    DEFAULT_TARGET_SIZES_SHAPE, dynamic_image_shape, extra_input_index, find_image_input,
    image_input_size,
};
use super::select::{EpStatus, NEEDS_ONNX, OrtProbe, cuda_ordinals, onnx_path_for};
use super::spec::{Runtime, Target};
use super::vram;
use super::{
    Candidate, Completion, DeviceInfo, LoadRequest, ModelInfo, SessionDead, libs, onnx_rewrite,
};
use crate::model::{ExtraData, ExtraInput, NamedOutput, PortElem, PortSpec};
use crate::sync::LockExt;
use ::ort::ep::{self, ExecutionProvider};
use ::ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ::ort::session::{Session, SessionInputValue};
use ::ort::value::{DynValue, Tensor, TensorElementType, TensorRef, ValueType};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The loaded ONNX Runtime library (process-wide).
#[derive(Debug, Clone)]
struct LoadedLib {
    path: PathBuf,
    version: String,
    env: ::ort::environment::Environment,
}

/// Set once the library is loaded and the `ort` environment committed. Failures are not cached,
/// so a later generation (e.g. after `setup-onnxruntime`) can retry.
static LOADED: Mutex<Option<LoadedLib>> = Mutex::new(None);

fn ort_err<R>(e: ::ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// `OrtGetApiBase()->GetVersionString()` of the library at `path`. The handle is leaked on
/// purpose: ONNX Runtime stays loaded for the process (`ort` keeps its own handle too) and
/// libraries with Objective-C code cannot be unloaded on macOS anyway.
fn library_version(path: &Path) -> Result<String> {
    type GetApiBase = unsafe extern "system" fn() -> *const ::ort::sys::OrtApiBase;
    // SAFETY: `path` is an ONNX Runtime shared library chosen by the user/installer; loading it
    // runs its initializers exactly as `ort::init_from` would right after. `OrtGetApiBase` has
    // the declared signature in every ONNX Runtime release (C API, `onnxruntime_c_api.h`), the
    // returned struct is static and `GetVersionString` returns a static NUL-terminated string.
    unsafe {
        let lib = crate::backend::libs::open_path(path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;
        let get =
            crate::backend::libs::symbol::<GetApiBase>(&lib, b"OrtGetApiBase\0").map_err(|e| {
                anyhow::anyhow!("{} does not export OrtGetApiBase: {e}", path.display())
            })?;
        let base = get();
        if base.is_null() {
            bail!("{}: OrtGetApiBase returned null", path.display());
        }
        let v = ((*base).GetVersionString)();
        let version = if v.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(v).to_string_lossy().into_owned()
        };
        std::mem::forget(lib);
        Ok(version)
    }
}

/// Before this process first loads ONNX Runtime from `path`: apply an update of that flavor that
/// was downloaded while it was in use, and mark the folder in use for the life of the process
/// (`resources::manager::in_use`). Runs before anything touches the folder (the DLL search
/// registration included). The swap may rename a versioned library file
/// (`libonnxruntime.so.1.24.4`), so the library is looked up again in the same folder.
fn before_first_load(path: &Path) -> PathBuf {
    let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) else {
        return path.to_path_buf();
    };
    if LOADED.lock_recover().is_some() {
        return path.to_path_buf();
    }
    crate::resources::manager::before_load(dir);
    if path.is_file() {
        return path.to_path_buf();
    }
    crate::backend::libs::find_ort_library_in(dir).unwrap_or_else(|| path.to_path_buf())
}

/// `(major, minor)` of a version string like "1.24.4".
pub(crate) fn parse_version(v: &str) -> Option<(u32, u32)> {
    let mut it = v.trim().split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

/// Whether a library of `version` can serve the `ort` crate's API level (1.`min_minor`+).
pub(crate) fn version_supported(version: &str, min_minor: u32) -> bool {
    matches!(parse_version(version), Some((1, m)) if m >= min_minor)
}

fn load_library(path: &Path) -> Result<LoadedLib> {
    let mut slot = LOADED.lock_recover();
    if let Some(lib) = slot.as_ref() {
        if lib.path != path {
            tracing::warn!(
                "ONNX Runtime is already loaded from {}; {} is ignored because a process can load \
                 only one ONNX Runtime flavor; restart the service to switch to it",
                lib.path.display(),
                path.display()
            );
        }
        return Ok(lib.clone());
    }
    let version = library_version(path)?;
    if !version_supported(&version, ::ort::MINOR_VERSION) {
        bail!(
            "{} is ONNX Runtime {version}; 1.{}.0 or newer is required (run `setup-onnxruntime`)",
            path.display(),
            ::ort::MINOR_VERSION
        );
    }
    // Windows: DirectML.dll shipped next to onnxruntime.dll must win over the older copy in
    // System32, so load it by absolute path first.
    #[cfg(windows)]
    if let Some(dir) = path.parent() {
        let dml = dir.join("DirectML.dll");
        if dml.is_file() {
            match ::ort::util::preload_dylib(&dml) {
                Ok(()) => tracing::debug!("preloaded {}", dml.display()),
                Err(e) => tracing::warn!(
                    "could not preload {}: {}",
                    dml.display(),
                    crate::backend::libs::dl_error(&e)
                ),
            }
        }
    }
    let env = ::ort::init_from(path)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_name("blue-onyx-prism")
        .with_telemetry(false)
        .build()
        .map_err(|e| anyhow::anyhow!("creating the ONNX Runtime environment: {e}"))?;
    // `ort` forwards every ONNX Runtime message to `tracing`, including the verbose/info ones it
    // prints per session; keep warnings and errors unless `ORT_LOG` asks for more.
    let level = match std::env::var("ORT_LOG").as_deref() {
        Ok("verbose") => ::ort::logging::LogLevel::Verbose,
        Ok("info") => ::ort::logging::LogLevel::Info,
        Ok("error") => ::ort::logging::LogLevel::Error,
        Ok("fatal") => ::ort::logging::LogLevel::Fatal,
        _ => ::ort::logging::LogLevel::Warning,
    };
    env.set_log_level(level);
    let lib = LoadedLib {
        path: path.to_path_buf(),
        version,
        env,
    };
    *slot = Some(lib.clone());
    Ok(lib)
}

// ---- CUDA / TensorRT dependency checks ---------------------------------------------------------

/// Library names tried for each dependency, in order (any one loading is enough).
#[cfg(windows)]
mod dep_names {
    pub const CUDART: &[&str] = &["cudart64_12.dll", "cudart64_13.dll"];
    pub const CUDNN: &[&str] = &["cudnn64_9.dll"];
    pub const NVINFER: &[&str] = &["nvinfer_10.dll"];
    pub const CUDA_PROVIDER: &str = "onnxruntime_providers_cuda.dll";
    pub const TRT_PROVIDER: &str = "onnxruntime_providers_tensorrt.dll";
}
#[cfg(not(windows))]
mod dep_names {
    pub const CUDART: &[&str] = &["libcudart.so.12", "libcudart.so.13"];
    pub const CUDNN: &[&str] = &["libcudnn.so.9"];
    pub const NVINFER: &[&str] = &["libnvinfer.so.10"];
    pub const CUDA_PROVIDER: &str = "libonnxruntime_providers_cuda.so";
    pub const TRT_PROVIDER: &str = "libonnxruntime_providers_tensorrt.so";
}

/// One dependency of an NVIDIA execution provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dep {
    /// "CUDA 12 runtime", "cuDNN 9", ...
    pub label: &'static str,
    /// File names tried (the first is named in messages).
    pub names: &'static [&'static str],
    pub loadable: bool,
    /// Part of `nvidia-cuda-libs` (downloadable), not the provider or TensorRT.
    pub cuda_lib: bool,
}

/// Why an NVIDIA EP cannot run, naming every missing piece; None when all deps load.
#[cfg(test)]
pub(crate) fn missing_deps_reason(ep: &str, deps: &[Dep]) -> Option<String> {
    missing_deps_reason_with(ep, deps, None)
}

/// [`missing_deps_reason`]; when a missing CUDA library (runtime, cuDNN) could be downloaded,
/// `download_hint` says how (see [`cuda_download_hint`]).
pub(crate) fn missing_deps_reason_with(
    ep: &str,
    deps: &[Dep],
    download_hint: Option<&str>,
) -> Option<String> {
    let missing: Vec<&Dep> = deps.iter().filter(|d| !d.loadable).collect();
    if missing.is_empty() {
        return None;
    }
    let list: Vec<String> = missing
        .iter()
        .map(|d| format!("{} ({})", d.label, d.names.join(" / ")))
        .collect();
    let fix = match download_hint {
        Some(h) if missing.iter().any(|d| d.cuda_lib) => h.to_string(),
        _ => format!(
            "install it or add its directory to {}",
            if cfg!(windows) {
                "PATH"
            } else {
                "LD_LIBRARY_PATH"
            }
        ),
    };
    Some(format!(
        "{ep} execution provider needs {} which could not be loaded; {fix}",
        list.join(" and ")
    ))
}

/// "CUDA libraries missing — enable allow_large_downloads or run `blue-onyx-prism fetch
/// --resource nvidia-cuda-libs` (1.9 GB), or install CUDA 12.8+ and cuDNN 9 ..." when the catalog has
/// them for this platform; None otherwise.
pub(crate) fn cuda_download_hint(os: &str, arch: &str) -> Option<String> {
    let r = crate::resources::catalog::cuda_libs_for(os, arch)?;
    Some(format!(
        "CUDA libraries missing \u{2014} enable allow_large_downloads or run `blue-onyx-prism \
         fetch --resource {}` ({}), or install CUDA 12.8+ and cuDNN 9 (9.10 or older for GTX \
         10-series cards)",
        r.id,
        crate::resources::catalog::format_size(r.size())
    ))
}

/// Whether any of `names` loads by absolute path from one of `dirs` (the ONNX Runtime dir,
/// `cuda-libs`) or from the enumerated system library directories
/// ([`libs::cuda_search_dirs`]); never by bare name, so the working directory is never
/// consulted (BE-01). A library that loads stays loaded (see [`libs::probe_cuda_library`]).
fn any_loadable(names: &[&str], dirs: &[&Path]) -> bool {
    names
        .iter()
        .any(|n| match libs::probe_cuda_library(n, dirs) {
            Ok(_) => true,
            Err(e) => {
                tracing::debug!("NVIDIA dependency probe: {e}");
                false
            }
        })
}

/// The directory `LD_LIBRARY_PATH`-style variable [`libs::cuda_search_dirs`] reads (the same
/// name as `libs`' private `ENV_LIBRARY_PATH`).
#[cfg(target_os = "windows")]
const LIBRARY_PATH_VAR: &str = "PATH";
#[cfg(target_os = "macos")]
const LIBRARY_PATH_VAR: &str = "DYLD_LIBRARY_PATH";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const LIBRARY_PATH_VAR: &str = "LD_LIBRARY_PATH";

/// The first of `names` that exists as a file in `dirs` (the ONNX Runtime dir, `cuda-libs`)
/// or the system CUDA directories ([`libs::cuda_search_dirs`], the directories
/// [`libs::probe_cuda_library`] loads from), without loading it.
fn find_cuda_file(names: &[&str], dirs: &[&Path]) -> Option<PathBuf> {
    let home = std::env::var_os("CUDA_PATH").or_else(|| std::env::var_os("CUDA_HOME"));
    let lp = std::env::var_os(LIBRARY_PATH_VAR);
    let search = libs::cuda_search_dirs(
        dirs,
        (home.as_deref(), lp.as_deref()),
        cfg!(windows),
        cfg!(target_os = "linux"),
    );
    names
        .iter()
        .flat_map(|n| search.iter().map(move |d| d.join(n)))
        .find(|p| p.is_file())
}

/// Whether any of `names` exists ([`find_cuda_file`]); nothing is loaded.
fn any_present(names: &[&str], dirs: &[&Path]) -> bool {
    find_cuda_file(names, dirs).is_some()
}

/// How the probe decides whether a dependency of an NVIDIA execution provider is there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepCheck {
    /// Loaded by absolute path and kept loaded ([`any_loadable`]): the CUDA runtime and cuDNN,
    /// so a missing dependency of theirs shows in the reason (BE-02), and the CUDA provider's
    /// by-name imports resolve to these copies.
    Load,
    /// The file exists ([`any_present`]); loaded only when a session needs it
    /// ([`preload_for_session`]): TensorRT's `nvinfer`, the largest image by far, which `auto`
    /// never picks, so a probe must not map it into every process with the cuda flavor (RA-086).
    Exists,
}

/// The check each probed dependency gets (pure): TensorRT by file, the rest by loading.
pub(crate) fn dep_check(names: &[&str]) -> DepCheck {
    if names == dep_names::NVINFER {
        DepCheck::Exists
    } else {
        DepCheck::Load
    }
}

/// The libraries a `target` session loads by path just before its provider registers (pure):
/// TensorRT's `nvinfer` (probed by file only, [`DepCheck::Exists`]); nothing for the others.
pub(crate) fn session_preloads(target: Target) -> &'static [&'static str] {
    match target {
        Target::TensorRt => dep_names::NVINFER,
        _ => &[],
    }
}

/// Loads the first of [`session_preloads`]`(target)` that loads by absolute path from `dirs`
/// or the system CUDA directories ([`libs::probe_cuda_library`]; kept loaded, so the
/// provider's by-name import resolves to that copy). Once one loaded, later sessions skip the
/// lookup. A failure is the session's error (the probe only saw that the file exists).
fn preload_for_session(target: Target, dirs: &[&Path]) -> Result<()> {
    static LOADED_FOR_SESSIONS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let names = session_preloads(target);
    let mut loaded = LOADED_FOR_SESSIONS.lock_recover();
    if names.is_empty() || names.iter().any(|n| loaded.contains(n)) {
        return Ok(());
    }
    let mut errors = Vec::new();
    for &n in names {
        match libs::probe_cuda_library(n, dirs) {
            Ok(p) => {
                tracing::info!(
                    "{} session: loaded {n} from {}",
                    target.display_name(),
                    p.display()
                );
                loaded.push(n);
                return Ok(());
            }
            Err(e) => errors.push(e),
        }
    }
    bail!(
        "{} needs {} which could not be loaded ({}); reinstall TensorRT 10 or add its \
         directory to {LIBRARY_PATH_VAR}",
        target.display_name(),
        names.join(" / "),
        errors.join("; ")
    )
}

fn nvidia_deps(target: Target, ort_dir: &Path, cuda_libs: Option<&Path>) -> Vec<Dep> {
    nvidia_deps_with(
        target,
        ort_dir,
        cuda_libs,
        &|names, dirs| match dep_check(names) {
            DepCheck::Load => any_loadable(names, dirs),
            DepCheck::Exists => any_present(names, dirs),
        },
    )
}

/// [`nvidia_deps`] with `check` deciding each library dependency (the provider files are
/// always checked by existence in the ONNX Runtime dir).
fn nvidia_deps_with(
    target: Target,
    ort_dir: &Path,
    cuda_libs: Option<&Path>,
    check: &dyn Fn(&[&str], &[&Path]) -> bool,
) -> Vec<Dep> {
    let dirs: Vec<&Path> = std::iter::once(ort_dir).chain(cuda_libs).collect();
    let mut deps = vec![
        Dep {
            label: "the ONNX Runtime CUDA provider",
            names: &[dep_names::CUDA_PROVIDER],
            loadable: ort_dir.join(dep_names::CUDA_PROVIDER).is_file(),
            cuda_lib: false,
        },
        Dep {
            label: "the CUDA runtime",
            names: dep_names::CUDART,
            loadable: check(dep_names::CUDART, &dirs),
            cuda_lib: true,
        },
        Dep {
            label: "cuDNN 9",
            names: dep_names::CUDNN,
            loadable: check(dep_names::CUDNN, &dirs),
            cuda_lib: true,
        },
    ];
    if target == Target::TensorRt {
        deps.push(Dep {
            label: "the ONNX Runtime TensorRT provider",
            names: &[dep_names::TRT_PROVIDER],
            loadable: ort_dir.join(dep_names::TRT_PROVIDER).is_file(),
            cuda_lib: false,
        });
        deps.push(Dep {
            label: "TensorRT 10",
            names: dep_names::NVINFER,
            loadable: check(dep_names::NVINFER, &dirs),
            cuda_lib: false,
        });
    }
    deps
}

// ---- runtime ---------------------------------------------------------------------------------

/// The ONNX Runtime library of this process plus the options sessions are created with.
pub struct OrtRuntime {
    lib: LoadedLib,
    cache_dir: Option<PathBuf>,
    intra_threads: usize,
    probe: OrtProbe,
    hw: HardwareInfo,
    /// The `cuda-libs` directory given to [`Self::new`] when it exists: searched again for the
    /// libraries a session loads just before its provider registers ([`session_preloads`]).
    cuda_libs: Option<PathBuf>,
}

impl std::fmt::Debug for OrtRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtRuntime")
            .field("path", &self.lib.path)
            .field("version", &self.lib.version)
            .field("probe", &self.probe)
            .finish()
    }
}

impl OrtRuntime {
    /// Load the library at `path` (from `libs::find_onnxruntime`), then probe its execution
    /// providers.
    ///
    /// `cuda_libs`: the `onnxruntime/cuda-libs` directory of `nvidia-cuda-libs`. When it exists,
    /// an NVIDIA GPU is present and the CUDA provider is installed, its libraries are preloaded
    /// first (see [`libs::preload_cuda_libs`]) so the CUDA provider finds them.
    pub fn new(
        path: &Path,
        opts: &super::CoreOptions,
        hw: &HardwareInfo,
        cuda_libs: Option<&Path>,
    ) -> Result<Self> {
        let path = &before_first_load(path);
        // The flavor dir holds the library's own imports (`onnxruntime_providers_shared`,
        // `DirectML.dll`): make it part of the DLL search (Windows; a no-op elsewhere) since
        // `PATH` and the working directory are not searched (BE-01).
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            libs::register_dll_dir(dir);
        }
        let lib = load_library(path)
            .with_context(|| format!("loading ONNX Runtime from {}", path.display()))?;
        let ort_dir = lib.path.parent().unwrap_or(Path::new("."));
        let cuda_libs = cuda_libs.filter(|d| d.is_dir());
        if let Some(dir) = cuda_libs
            && hw.has_vendor(GpuVendor::Nvidia)
            && ort_dir.join(dep_names::CUDA_PROVIDER).is_file()
        {
            let r = libs::preload_cuda_libs(dir);
            tracing::info!(
                "preloaded {} CUDA libraries from {}",
                r.loaded.len(),
                dir.display()
            );
            for (p, e) in &r.failed {
                tracing::warn!("could not preload {}: {e}", p.display());
            }
        }
        let probe = probe_providers(&lib, hw, cuda_libs);
        tracing::info!(
            "ONNX Runtime {} loaded from {} (flavor {}); execution providers: {}",
            lib.version,
            lib.path.display(),
            probe.flavor.as_deref().unwrap_or("unknown"),
            probe
                .providers
                .iter()
                .map(|p| match &p.error {
                    None => p.target.display_name().to_string(),
                    Some(e) => format!("{} (unusable: {e})", p.target.display_name()),
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(Self {
            lib,
            cache_dir: opts.cache_dir.clone(),
            intra_threads: super::cpu_threads::effective(opts.intra_threads),
            probe,
            hw: hw.clone(),
            cuda_libs: cuda_libs.map(Path::to_path_buf),
        })
    }

    pub fn probe(&self) -> &OrtProbe {
        &self.probe
    }

    /// Library version, e.g. "1.24.4".
    pub fn version(&self) -> &str {
        &self.lib.version
    }

    pub fn library_path(&self) -> &Path {
        &self.lib.path
    }

    /// Create `sessions` (at least 1) sessions for `req` on exactly `cand.device` (an ONNX
    /// Runtime target). Each session runs one inference at a time on its own thread
    /// ([`OrtBackend`]) and holds its own copy of the model.
    pub fn compile(
        &self,
        cand: &Candidate,
        req: &LoadRequest,
        sessions: usize,
    ) -> Result<OrtCompiled> {
        let target = cand.device.target;
        if cand.device.runtime != Runtime::Ort {
            bail!("{} is not an ONNX Runtime device", cand.device);
        }
        if let Some(reason) = self.probe.ep_error(target) {
            bail!("{} cannot run: {reason}", cand.device);
        }
        let path = onnx_path_for(&req.path).with_context(|| {
            format!(
                "{} cannot load {}: {NEEDS_ONNX}",
                cand.device,
                req.path.display()
            )
        })?;
        let t0 = Instant::now();
        let index = cand.device.index.unwrap_or(0);
        // The card's free memory before the sessions exist, for the load line.
        let probe = vram::LoadProbe::start(&path, &cand.device);
        // Ports first. On the CPU the real session doubles as the metadata source (rebuilt only
        // when dynamic dims need pinning); for accelerators an unoptimized CPU session reads
        // them, so the EP compiles once, already with the pinned shapes.
        let original = ModelSource::File(path.clone());
        let no_cache = req.no_cache;
        let session = if target == Target::Cpu {
            Some(self.build_session(target, index, &original, &[], no_cache)?)
        } else {
            None
        };
        let ((mut inputs, syms), meta_outputs) = {
            let meta;
            let s = match &session {
                Some(s) => s,
                None => {
                    meta = self.metadata_session(&path)?;
                    &meta
                }
            };
            (ports(s.inputs()), ports(s.outputs()).0)
        };
        let image_idx = find_image_input(&inputs).with_context(|| {
            format!(
                "model {}: no image input (named 'images'/'input' or 4-D) among {:?}",
                path.display(),
                inputs
                    .iter()
                    .map(|p| (&p.name, &p.shape))
                    .collect::<Vec<_>>()
            )
        })?;
        let extra_idx = extra_input_index(&inputs, image_idx, "orig_target_sizes");
        let image_default = dynamic_image_shape(&path, &meta_outputs);
        // The input size the sessions will be pinned to, checked before any accelerator session
        // (or a second CPU session) is built for it (QA-026).
        if let Some(img) = inputs.get(image_idx) {
            let pinned = PortSpec {
                shape: fixed_shape(&img.shape, &image_default),
                ..img.clone()
            };
            image_input_size(&path, &pinned)?;
        }
        let overrides = dimension_overrides(&inputs, &syms, image_idx, extra_idx, &image_default);
        if let Some(reason) = unsupported_on(target, &inputs, &syms, &overrides) {
            bail!(
                "{} cannot run model {}: {reason}",
                cand.device,
                path.display()
            );
        }
        if !overrides.is_empty() {
            tracing::info!(
                "model {}: pinning dynamic dimensions {overrides:?}",
                path.display()
            );
        }
        // CoreML: a copy without the ops that would split the graph into many partitions.
        let source = if target == Target::CoreMl {
            self.coreml_model(&path, no_cache)
        } else {
            original
        };
        let session = match session {
            Some(s) if overrides.is_empty() => s,
            _ => {
                let s = self.build_session(target, index, &source, &overrides, no_cache)?;
                inputs = ports(s.inputs()).0;
                s
            }
        };
        // Report (and feed) static shapes for the image and `orig_target_sizes` inputs.
        let img = inputs
            .get_mut(image_idx)
            .with_context(|| format!("model {}: image input index out of range", path.display()))?;
        if img.shape.len() != 4 {
            bail!(
                "model {}: image input '{}' has rank {} (expected 4-D NCHW, shape {:?})",
                path.display(),
                img.name,
                img.shape.len(),
                img.shape
            );
        }
        img.shape = fixed_shape(&img.shape, &image_default);
        if img.elem != PortElem::F32 {
            bail!(
                "model {}: image input '{}' has element type {:?}; only f32 is supported",
                path.display(),
                img.name,
                img.elem
            );
        }
        let input_size = image_input_size(&path, img)?;
        let image_input = img.name.clone();
        if let Some(i) = extra_idx
            && let Some(p) = inputs.get_mut(i)
        {
            if p.shape.is_empty() {
                p.shape = DEFAULT_TARGET_SIZES_SHAPE.to_vec();
            } else {
                p.shape = fixed_shape(&p.shape, &DEFAULT_TARGET_SIZES_SHAPE);
            }
        }
        let (outputs, _) = ports(session.outputs());
        let mut all = vec![session];
        for _ in 1..sessions.max(1) {
            all.push(self.build_session(target, index, &source, &overrides, no_cache)?);
        }
        let compile_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
        let device = DeviceInfo {
            requested: req.requested.clone(),
            actual: device_actual(cand),
            full_name: self.full_name(cand),
            fell_back: cand.fell_back,
            runtime: Runtime::Ort,
            spec: cand.device.to_string(),
        };
        tracing::info!(
            "model {} loaded on {} in {compile_ms} ms ({} session{}); inputs {:?}, outputs {:?}",
            path.display(),
            device.execution_provider(),
            all.len(),
            if all.len() == 1 { "" } else { "s" },
            inputs
                .iter()
                .map(|p| (&p.name, &p.shape))
                .collect::<Vec<_>>(),
            outputs
                .iter()
                .map(|p| (&p.name, &p.shape))
                .collect::<Vec<_>>()
        );
        let vram_hold = probe.finish();
        if target == Target::CoreMl {
            // The rewrite (parsed graph, a rewritten copy kept in memory for a `no_cache` load)
            // and the CoreML conversion leave freed heap behind: hand it back to the OS.
            drop(source);
            super::memory::trim_soon("coreml rewrite");
        }
        Ok(OrtCompiled {
            sessions: all,
            vram_hold,
            info: ModelInfo {
                path,
                inputs,
                outputs,
                device,
                image_input,
                input_size,
                compile_ms,
            },
        })
    }

    /// GPU name for CUDA/TensorRT (NVIDIA ordinal) and DirectML (DXGI index); empty otherwise.
    fn full_name(&self, cand: &Candidate) -> String {
        gpu_full_name(&self.hw, cand.device.target, cand.device.index.unwrap_or(0))
    }

    /// What a CoreML session loads: a cached rewrite of `path` (see
    /// [`onnx_rewrite::rewrite_cached`]) when it has rewritable ops, else `path`. With `no_cache`
    /// the rewrite stays in memory ([`onnx_rewrite::rewrite_file`]). Failures only log a warning
    /// and fall back to the original.
    fn coreml_model(&self, path: &Path, no_cache: bool) -> ModelSource {
        let original = || ModelSource::File(path.to_path_buf());
        if !coreml_rewrite_enabled(std::env::var(ENV_COREML_REWRITE).ok().as_deref()) {
            tracing::info!(
                "model {}: CoreML graph rewrite disabled ({ENV_COREML_REWRITE})",
                path.display()
            );
            return original();
        }
        if no_cache {
            return coreml_model_in_memory(path);
        }
        // Only the configured cache dir: a predictable path in the shared temp dir could be
        // pre-created by another local user (BE-04).
        let Some(dir) = self.cache_dir.as_ref().map(|c| c.join("coreml")) else {
            tracing::warn!(
                "model {}: CoreML graph rewrite skipped because cache_dir is empty (set it to \
                 enable the rewritten copy; unsupported ops fall back to the CPU provider)",
                path.display()
            );
            return original();
        };
        let t0 = Instant::now();
        match onnx_rewrite::rewrite_cached(path, &dir, &onnx_rewrite::RewriteOptions::COREML) {
            Ok(onnx_rewrite::Cached::Rewritten {
                path: rewritten,
                counts,
                skipped,
                created,
            }) => {
                tracing::info!(
                    "model {}: CoreML loads a rewritten copy ({}; {} in {} ms): {}",
                    path.display(),
                    onnx_rewrite::summary(&counts),
                    if created { "written" } else { "cached" },
                    t0.elapsed().as_millis(),
                    rewritten.display()
                );
                if !skipped.is_empty() {
                    tracing::info!(
                        "model {}: not rewritten for CoreML: {}",
                        path.display(),
                        skipped.join("; ")
                    );
                }
                ModelSource::File(rewritten)
            }
            Ok(onnx_rewrite::Cached::Original { skipped }) => {
                if !skipped.is_empty() {
                    tracing::info!(
                        "model {}: not rewritten for CoreML: {}",
                        path.display(),
                        skipped.join("; ")
                    );
                }
                original()
            }
            Ok(onnx_rewrite::Cached::ExternalData) => {
                tracing::info!(
                    "model {}: has external data; CoreML loads it without the graph rewrite",
                    path.display()
                );
                original()
            }
            Err(e) => {
                tracing::warn!(
                    "model {}: CoreML graph rewrite failed, loading the original (ops CoreML \
                     cannot run fall back to the CPU provider, slower): {e:#}; if it persists, \
                     delete cache_dir/coreml or set BOP_COREML_REWRITE=0",
                    path.display()
                );
                original()
            }
        }
    }

    /// `cache/<name>/<stem>-<key16>/` for the model file `model`
    /// ([`crate::storage::caches::model_cache_dir`], which records the use); None (logged) when
    /// there is no cache dir or it cannot be made: the session then builds without a cache.
    fn model_cache_subdir(&self, name: &str, model: &Path) -> Option<String> {
        let folder = self.cache_dir.as_ref()?.join(name);
        match crate::storage::caches::model_cache_dir(&folder, model) {
            Ok(dir) => dir.to_str().map(str::to_string),
            Err(e) => {
                tracing::warn!(
                    "model {}: could not make its cache folder in {} ({e}); building without a \
                     cache (check that the cache dir is writable)",
                    model.display(),
                    folder.display()
                );
                None
            }
        }
    }

    fn cache_subdir(&self, name: &str) -> Option<String> {
        let dir = self.cache_dir.as_ref()?.join(name);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("could not create cache dir {}: {e}", dir.display());
            return None;
        }
        // The storage sweep removes cache folders unused for 60 days.
        crate::storage::touch(&dir);
        dir.to_str().map(str::to_string)
    }

    /// A session builder on the process-wide environment (the library must be loaded).
    fn session_builder() -> Result<SessionBuilder> {
        let env = LOADED
            .lock_recover()
            .as_ref()
            .map(|lib| lib.env.clone())
            .ok_or_else(|| anyhow::anyhow!("ONNX Runtime is not loaded"))?;
        Session::builder(&env).map_err(ort_err)
    }

    /// An unoptimized CPU session, only to read the model's ports.
    fn metadata_session(&self, path: &Path) -> Result<Session> {
        Self::session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
    }

    /// A session on `target` for `source`. `no_cache` ([`LoadRequest::no_cache`]): no TensorRT
    /// engine or timing cache and no CoreML model cache.
    fn build_session(
        &self,
        target: Target,
        index: u32,
        source: &ModelSource,
        overrides: &[(String, i64)],
        no_cache: bool,
    ) -> Result<Session> {
        let path = source.path();
        let mut b: SessionBuilder = Self::session_builder()?;
        // Every session, GPU EPs included (their CPU fallback ops run on this pool).
        b = b.with_intra_threads(self.intra_threads).map_err(ort_err)?;
        let spin = ort_spinning_enabled(std::env::var(ENV_ORT_SPINNING).ok().as_deref());
        tracing::debug!(
            "ONNX Runtime {} session for {}: {} intra-op threads, spinning {}",
            target.display_name(),
            path.display(),
            self.intra_threads,
            if spin { "on" } else { "off" }
        );
        b = b
            .with_intra_op_spinning(spin)
            .map_err(ort_err)?
            .with_inter_op_spinning(spin)
            .map_err(ort_err)?;
        for (name, value) in overrides {
            b = b.with_dimension_override(name, *value).map_err(ort_err)?;
        }
        // `spec::parse` caps indexes at `MAX_DEVICE_INDEX`; never wrap a larger one negative.
        let device_id = i32::try_from(index).map_err(|_| {
            anyhow::anyhow!(
                "device index {index} is out of range for ONNX Runtime; it is probably mistyped; \
                 run `list-devices` and use one of the indexes it shows"
            )
        })?;
        let eps = match target {
            Target::Cpu if !cpu_arena_enabled(std::env::var(ENV_ORT_CPU_ARENA).ok().as_deref()) => {
                log_cpu_arena_off();
                vec![
                    ep::CPU::default()
                        .with_arena_allocator(false)
                        .build()
                        .error_on_failure(),
                ]
            }
            Target::Cpu => Vec::new(),
            Target::Cuda => vec![
                cuda_provider(device_id, vram::ort_session_limit_mb())
                    .build()
                    .error_on_failure(),
            ],
            Target::TensorRt => {
                // TensorRT is probed by file only (RA-086): load it now, by path, so the
                // provider's import resolves to this copy and a broken install fails here.
                let ort_dir = self.lib.path.parent().unwrap_or(Path::new("."));
                let dirs: Vec<&Path> = std::iter::once(ort_dir)
                    .chain(self.cuda_libs.as_deref())
                    .collect();
                preload_for_session(target, &dirs)?;
                let mut trt = ep::TensorRT::default()
                    .with_device_id(device_id)
                    .with_fp16(true);
                // Engines of the model's own file (see the module docs); a bytes source never
                // reaches TensorRT (only the CoreML rewrite is one).
                let cached = match source {
                    ModelSource::File(file) if !no_cache => {
                        self.model_cache_subdir("tensorrt", file)
                    }
                    _ => None,
                };
                if let Some(dir) = cached {
                    trt = trt
                        .with_engine_cache(true)
                        .with_engine_cache_path(&dir)
                        .with_timing_cache(true)
                        .with_timing_cache_path(&dir);
                }
                vec![
                    trt.build().error_on_failure(),
                    // Nodes TensorRT does not take run on CUDA rather than the CPU.
                    cuda_provider(device_id, vram::ort_session_limit_mb())
                        .build()
                        .fail_silently(),
                ]
            }
            Target::DirectMl => {
                b = b
                    .with_memory_pattern(false)
                    .map_err(ort_err)?
                    .with_parallel_execution(false)
                    .map_err(ort_err)?;
                vec![
                    ep::DirectML::default()
                        .with_device_id(device_id)
                        .build()
                        .error_on_failure(),
                ]
            }
            Target::CoreMl => {
                let mut coreml = ep::CoreML::default()
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .with_compute_units(ep::coreml::ComputeUnits::All);
                if !no_cache && let Some(dir) = self.cache_subdir("coreml") {
                    coreml = coreml.with_model_cache_dir(dir);
                }
                vec![coreml.build().error_on_failure()]
            }
            Target::Gpu | Target::Npu => bail!(
                "ONNX Runtime has no '{}' execution provider",
                target.as_str()
            ),
        };
        if !eps.is_empty() {
            b = b.with_execution_providers(eps).map_err(|e| {
                anyhow::anyhow!(
                    "registering the {} execution provider: {e}",
                    target.display_name()
                )
            })?;
        }
        match source {
            ModelSource::File(file) => b.commit_from_file(file),
            ModelSource::Memory { bytes, .. } => b.commit_from_memory(bytes),
        }
        .map_err(|e| {
            anyhow::anyhow!(
                "creating ONNX Runtime {} session for {}: {e}",
                target.display_name(),
                path.display()
            )
        })
    }
}

/// What a session loads: a model file, or the bytes of a rewrite of one kept in memory
/// ([`LoadRequest::no_cache`] on CoreML).
#[derive(Debug)]
enum ModelSource {
    File(PathBuf),
    Memory {
        bytes: Vec<u8>,
        /// The model file the bytes are a rewrite of (for messages).
        of: PathBuf,
    },
}

impl ModelSource {
    /// The model file, for messages.
    fn path(&self) -> &Path {
        match self {
            ModelSource::File(p) => p,
            ModelSource::Memory { of, .. } => of,
        }
    }
}

/// The CoreML rewrite of `path` kept in memory ([`onnx_rewrite::rewrite_file`]), else the file
/// itself (nothing to rewrite, external data, too large, or a failure, logged).
fn coreml_model_in_memory(path: &Path) -> ModelSource {
    let t0 = Instant::now();
    match onnx_rewrite::rewrite_file(path, &onnx_rewrite::RewriteOptions::COREML) {
        Ok(onnx_rewrite::Outcome::Rewritten(r)) => {
            tracing::info!(
                "model {}: CoreML loads a rewritten copy from memory ({}; in {} ms; no cache: a \
                 throwaway model)",
                path.display(),
                onnx_rewrite::summary(&r.counts),
                t0.elapsed().as_millis()
            );
            ModelSource::Memory {
                bytes: r.bytes,
                of: path.to_path_buf(),
            }
        }
        Ok(onnx_rewrite::Outcome::Unchanged { .. } | onnx_rewrite::Outcome::ExternalData) => {
            ModelSource::File(path.to_path_buf())
        }
        Err(e) => {
            tracing::warn!(
                "model {}: CoreML graph rewrite failed, loading the original (ops CoreML cannot \
                 run fall back to the CPU provider, slower): {e:#}",
                path.display()
            );
            ModelSource::File(path.to_path_buf())
        }
    }
}

/// The CUDA execution provider of a CUDA session, and the one behind TensorRT:
/// - `arena_extend_strategy` `kSameAsRequested`: the arena grows by what a run asks for; ONNX
///   Runtime's default (`kNextPowerOfTwo`) rounds every extension up, so a model can hold up to
///   about twice what it uses, on a card a training or auto-label job may share;
/// - `cudnn_conv_algo_search` `HEURISTIC`: the default exhaustive search benchmarks every
///   convolution algorithm at the first run, each with its own workspace: a memory peak at load
///   that can fail on a nearly full card. Measured against EXHAUSTIVE on a GTX 1070 Ti (plan
///   phase 3.6, docs/PRODUCTION_REVIEW.md): the same inference time and memory peak for
///   rf-detr-nano and dfine-s; [`ENV_CUDNN_CONV_ALGO_SEARCH`] picks another search to measure;
/// - `gpu_mem_limit` = `limit_mb` (config `ort_gpu_mem_limit_mb`, per session) when set.
pub(crate) fn cuda_provider(device_id: i32, limit_mb: Option<u64>) -> ep::CUDA {
    cuda_provider_with(device_id, limit_mb, conv_search_from_env())
}

/// [`cuda_provider`] with the cuDNN convolution search given. Pure.
pub(crate) fn cuda_provider_with(
    device_id: i32,
    limit_mb: Option<u64>,
    search: ConvSearch,
) -> ep::CUDA {
    let mut cuda = ep::CUDA::default()
        .with_device_id(device_id)
        .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
        .with_conv_algorithm_search(search.to_ort());
    if let Some(mb) = limit_mb {
        let bytes = usize::try_from(mb.saturating_mul(1024 * 1024)).unwrap_or(usize::MAX);
        cuda = cuda.with_memory_limit(bytes);
    }
    cuda
}

/// Environment variable that replaces the CUDA provider's `cudnn_conv_algo_search`
/// (`exhaustive`, `heuristic` or `default`), for measuring the search against the shipped
/// HEURISTIC only (plan phase 3.6): unset, empty or unknown = HEURISTIC, unchanged. Read once per
/// process, logged once when set. Also reaches the CUDA provider behind `ort:tensorrt`.
pub(crate) const ENV_CUDNN_CONV_ALGO_SEARCH: &str = "BOP_CUDNN_CONV_ALGO_SEARCH";

/// cuDNN convolution algorithm search of the CUDA provider (`ep::cuda::ConvAlgorithmSearch`,
/// which is neither `Copy` nor comparable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvSearch {
    /// Benchmark every algorithm at the first run (ONNX Runtime's own default).
    Exhaustive,
    /// cuDNN's heuristic ranking (what this service uses).
    Heuristic,
    /// `IMPLICIT_PRECOMP_GEMM`, no search.
    Default,
}

impl ConvSearch {
    fn to_ort(self) -> ep::cuda::ConvAlgorithmSearch {
        match self {
            ConvSearch::Exhaustive => ep::cuda::ConvAlgorithmSearch::Exhaustive,
            ConvSearch::Heuristic => ep::cuda::ConvAlgorithmSearch::Heuristic,
            ConvSearch::Default => ep::cuda::ConvAlgorithmSearch::Default,
        }
    }

    /// The ONNX Runtime option value (`EXHAUSTIVE`, `HEURISTIC`, `DEFAULT`).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ConvSearch::Exhaustive => "EXHAUSTIVE",
            ConvSearch::Heuristic => "HEURISTIC",
            ConvSearch::Default => "DEFAULT",
        }
    }
}

/// The search [`ENV_CUDNN_CONV_ALGO_SEARCH`]'s value asks for: Ok(None) when unset or empty
/// (HEURISTIC), Err(value) when it names no search. Case and surrounding space are ignored.
pub(crate) fn parse_conv_search(env: Option<&str>) -> Result<Option<ConvSearch>, String> {
    let Some(v) = env.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    match v.to_ascii_lowercase().as_str() {
        "exhaustive" => Ok(Some(ConvSearch::Exhaustive)),
        "heuristic" => Ok(Some(ConvSearch::Heuristic)),
        "default" => Ok(Some(ConvSearch::Default)),
        _ => Err(v.to_string()),
    }
}

/// What a parsed [`ENV_CUDNN_CONV_ALGO_SEARCH`] means: the search, and the one log line to write
/// (`true` = a warning: the search differs from HEURISTIC or the value is unknown). Pure.
pub(crate) fn conv_search_choice(
    parsed: Result<Option<ConvSearch>, String>,
) -> (ConvSearch, Option<(bool, String)>) {
    match parsed {
        Ok(None) => (ConvSearch::Heuristic, None),
        Ok(Some(ConvSearch::Heuristic)) => (
            ConvSearch::Heuristic,
            Some((
                false,
                format!(
                    "{ENV_CUDNN_CONV_ALGO_SEARCH} is set to the default: CUDA sessions use \
                     cudnn_conv_algo_search=HEURISTIC"
                ),
            )),
        ),
        Ok(Some(s)) => (
            s,
            Some((
                true,
                format!(
                    "{ENV_CUDNN_CONV_ALGO_SEARCH} is set: CUDA sessions use \
                     cudnn_conv_algo_search={} instead of HEURISTIC (a measurement setting; \
                     unset it to return to the default)",
                    s.as_str()
                ),
            )),
        ),
        Err(v) => (
            ConvSearch::Heuristic,
            Some((
                true,
                format!(
                    "{ENV_CUDNN_CONV_ALGO_SEARCH}='{v}' is not exhaustive, heuristic or default; \
                     CUDA sessions keep cudnn_conv_algo_search=HEURISTIC"
                ),
            )),
        ),
    }
}

/// The search of every CUDA session of this process: [`ENV_CUDNN_CONV_ALGO_SEARCH`] read once
/// (one log line when it is set, [`conv_search_choice`]), HEURISTIC otherwise.
fn conv_search_from_env() -> ConvSearch {
    static SEARCH: std::sync::OnceLock<ConvSearch> = std::sync::OnceLock::new();
    *SEARCH.get_or_init(|| {
        let env = std::env::var(ENV_CUDNN_CONV_ALGO_SEARCH).ok();
        let (search, note) = conv_search_choice(parse_conv_search(env.as_deref()));
        match note {
            Some((true, msg)) => tracing::warn!("{msg}"),
            Some((false, msg)) => tracing::info!("{msg}"),
            None => {}
        }
        search
    })
}

/// Environment variable that lets ONNX Runtime's thread pools spin-wait (`1`, `true`, `on`,
/// `yes`). ORT's pools spin by default (`session.intra_op.allow_spinning`), which in a
/// long-lived service keeps every core busy between runs; off costs tens of microseconds of
/// wake-up latency per run.
pub(crate) const ENV_ORT_SPINNING: &str = "BOP_ORT_SPINNING";

/// Whether ORT's intra/inter-op pools spin, given the value of [`ENV_ORT_SPINNING`]
/// (default off).
pub(crate) fn ort_spinning_enabled(env: Option<&str>) -> bool {
    matches!(
        env.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// Environment variable that turns the CPU arena of `ort:cpu` sessions off (`0`, `false`, `off`,
/// `no`), for measurement only (RA-094). ONNX Runtime's default CPU arena (kept by default)
/// grows by powers of two, so a session can hold up to about twice its plan for its whole
/// life; with the arena off every run allocates and frees its plan instead (on Windows
/// VirtualAlloc, on glibc mmap above the mmap threshold), which can add page-fault time to
/// every detection. Compare private bytes and p50/p99 `inference_ms` with it on and off before
/// changing the default. Only `Target::Cpu` sessions are affected.
pub(crate) const ENV_ORT_CPU_ARENA: &str = "BOP_ORT_CPU_ARENA";

/// Whether `ort:cpu` sessions keep ONNX Runtime's CPU arena, given the value of
/// [`ENV_ORT_CPU_ARENA`] (default on).
pub(crate) fn cpu_arena_enabled(env: Option<&str>) -> bool {
    !matches!(
        env.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// One info line per process when [`ENV_ORT_CPU_ARENA`] turned the CPU arena off.
fn log_cpu_arena_off() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "{ENV_ORT_CPU_ARENA} is set: ort:cpu sessions run without ONNX Runtime's CPU arena \
             (a measurement setting; unset it to restore the default)"
        );
    });
}

/// Environment variable that turns the CoreML graph rewrite off (`0`, `false`, `off`, `no`).
pub(crate) const ENV_COREML_REWRITE: &str = "BOP_COREML_REWRITE";

/// Whether the CoreML graph rewrite runs, given the value of [`ENV_COREML_REWRITE`].
pub(crate) fn coreml_rewrite_enabled(env: Option<&str>) -> bool {
    !matches!(
        env.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// Input dimensions that stay dynamic after the free-dimension `overrides` (from
/// [`dimension_overrides`]), as "input[axis]". `symbols` is per input, as from [`ports`].
pub(crate) fn unpinned_dims(
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    overrides: &[(String, i64)],
) -> Vec<String> {
    let mut out = Vec::new();
    for (i, port) in inputs.iter().enumerate() {
        for (axis, &dim) in port.shape.iter().enumerate() {
            if dim >= 0 {
                continue;
            }
            let pinned = symbols
                .get(i)
                .and_then(|s| s.get(axis))
                .filter(|s| !s.is_empty())
                .is_some_and(|s| overrides.iter().any(|(name, _)| name == s));
            if !pinned {
                out.push(format!("{}[{axis}]", port.name));
            }
        }
    }
    out
}

/// Why a model with these inputs must not be run on `target`, if it must not. `symbols` and
/// `overrides` are the dimension names of `inputs` and the free-dimension overrides the session
/// will be created with.
///
/// RT-DETR (`orig_target_sizes` input) on CoreML (ONNX Runtime 1.24.4, the pinned version,
/// measured on an M1): with a *dynamic batch* the Core ML program has unbounded dimensions
/// (`_postprocessor_Tile_output_0`, the encoder reshapes), Core ML rejects it and MPSGraph
/// *aborts the process* while specializing the first partition (`'mps.concat' op invalid input
/// tensor shapes` -> `MPSGraphExecutable.mm: failed assertion 'original module failed
/// verification'`) instead of returning an error. With every input dimension pinned (the
/// xnorpx RT-DETRv2 exports name the batch `N`, which [`dimension_overrides`] pins to 1) the
/// graph is fully static, 673 of 691 nodes (rt-detrv2-s) go to Core ML in 7 partitions, and the
/// default options (MLProgram, compute units ALL: fp32 on the GPU) run it correctly (scores
/// within 2e-6 of the CPU provider) and ~1.6-1.9x faster than `openvino:cpu`. Measured
/// alternatives on rt-detrv2-s, infer p50: ALL / CPUAndGPU 51 ms; CPUAndNeuralEngine and
/// CPUOnly 105 ms; NeuralNetwork format 82 ms (scores off by 7e-3); low-precision GPU
/// accumulation and FastPrediction no change; a dynamic batch with RequireStaticInputShapes
/// avoids the abort but leaves only 163 of 925 nodes to Core ML (237 ms, slower than
/// `openvino:cpu`). So only an RT-DETR model whose inputs keep a dynamic dimension that cannot
/// be pinned (unnamed) is refused; the plan then falls back to the next candidate (CPU).
pub(crate) fn unsupported_on(
    target: Target,
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    overrides: &[(String, i64)],
) -> Option<String> {
    let rtdetr = inputs.iter().any(|p| p.name == "orig_target_sizes");
    if target != Target::CoreMl || !rtdetr {
        return None;
    }
    let dynamic = unpinned_dims(inputs, symbols, overrides);
    (!dynamic.is_empty()).then(|| {
        format!(
            "RT-DETR models with dynamic input dimensions ({}) are not supported by the CoreML \
             execution provider (MPSGraph aborts the process); export the model with a static \
             or named batch dimension",
            dynamic.join(", ")
        )
    })
}

/// `executionProvider`-independent device string: "cpu", "cuda:0", "coreml", ...
fn device_actual(cand: &Candidate) -> String {
    let t = cand.device.target.as_str();
    match cand.device.index {
        Some(i) => format!("{t}:{i}"),
        None => t.to_string(),
    }
}

fn probe_providers(lib: &LoadedLib, hw: &HardwareInfo, cuda_libs: Option<&Path>) -> OrtProbe {
    let dir = lib.path.parent().unwrap_or(Path::new("."));
    let flavor = libs::read_ort_flavor(dir);
    let mut providers = vec![EpStatus::usable(Target::Cpu)];
    type Check = (Target, &'static str, fn() -> ::ort::Result<bool>);
    let checks: [Check; 4] = [
        (Target::Cuda, ep::CUDA::NAME, ep::CUDA::is_available),
        (
            Target::TensorRt,
            ep::TensorRT::NAME,
            ep::TensorRT::is_available,
        ),
        (
            Target::DirectMl,
            ep::DirectML::NAME,
            ep::DirectML::is_available,
        ),
        (Target::CoreMl, ep::CoreML::NAME, ep::CoreML::is_available),
    ];
    for (target, name, is_available) in checks {
        match is_available() {
            Ok(false) => {}
            Err(e) => providers.push(EpStatus::broken(target, format!("querying {name}: {e}"))),
            Ok(true) => {
                let reason = match target {
                    Target::Cuda | Target::TensorRt if !hw.has_vendor(GpuVendor::Nvidia) => {
                        Some("no NVIDIA GPU detected".to_string())
                    }
                    Target::Cuda | Target::TensorRt => missing_deps_reason_with(
                        target.display_name(),
                        &nvidia_deps(target, dir, cuda_libs),
                        cuda_download_hint(&hw.os, &hw.arch).as_deref(),
                    ),
                    _ => None,
                };
                providers.push(EpStatus {
                    target,
                    error: reason,
                });
            }
        }
    }
    OrtProbe {
        error: None,
        flavor,
        version: Some(lib.version.clone()),
        providers,
    }
}

// ---- ports and shapes ------------------------------------------------------------------------

fn port_elem(t: TensorElementType) -> PortElem {
    match t {
        TensorElementType::Float32 => PortElem::F32,
        TensorElementType::Float16 => PortElem::F16,
        TensorElementType::Int64 => PortElem::I64,
        TensorElementType::Int32 => PortElem::I32,
        TensorElementType::Uint8 => PortElem::U8,
        _ => PortElem::Other,
    }
}

/// Port specs plus, per port, the symbolic names of its dimensions ("" when unnamed).
fn ports(outlets: &[::ort::value::Outlet]) -> (Vec<PortSpec>, Vec<Vec<String>>) {
    outlets
        .iter()
        .map(|o| match o.dtype() {
            ValueType::Tensor {
                ty,
                shape,
                dimension_symbols,
            } => (
                PortSpec {
                    name: o.name().to_string(),
                    shape: shape.to_vec(),
                    elem: port_elem(*ty),
                },
                dimension_symbols.to_vec(),
            ),
            _ => (
                PortSpec {
                    name: o.name().to_string(),
                    shape: Vec::new(),
                    elem: PortElem::Other,
                },
                Vec::new(),
            ),
        })
        .unzip()
}

/// `shape` with every dynamic (negative) dim replaced by the default at that position.
pub(crate) fn fixed_shape(shape: &[i64], defaults: &[i64]) -> Vec<i64> {
    shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d >= 0 {
                d
            } else {
                defaults.get(i).copied().unwrap_or(1)
            }
        })
        .collect()
}

/// Free-dimension overrides that pin the named dynamic dims of the image input (to
/// `image_default`, usually `[1,3,640,640]`) and of `orig_target_sizes` (to `[1,2]`). A symbol
/// that would need two different values is left dynamic.
pub(crate) fn dimension_overrides(
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    image_idx: usize,
    extra_idx: Option<usize>,
    image_default: &[i64; 4],
) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    let targets = std::iter::once((image_idx, &image_default[..]))
        .chain(extra_idx.map(|i| (i, &DEFAULT_TARGET_SIZES_SHAPE[..])));
    for (idx, defaults) in targets {
        let (Some(port), Some(syms)) = (inputs.get(idx), symbols.get(idx)) else {
            continue;
        };
        for (pos, &dim) in port.shape.iter().enumerate() {
            let Some(sym) = syms.get(pos).filter(|s| !s.is_empty()) else {
                continue;
            };
            if dim >= 0 {
                continue;
            }
            let value = defaults.get(pos).copied().unwrap_or(1);
            match out.iter().find(|(s, _)| s == sym) {
                Some((_, v)) if *v != value => conflicts.push(sym.clone()),
                Some(_) => {}
                None => out.push((sym.clone(), value)),
            }
        }
    }
    out.retain(|(s, _)| !conflicts.contains(s));
    out
}

// ---- compiled model and inference ------------------------------------------------------------

/// ONNX Runtime sessions of one model plus its metadata. `Session` is `Send`, so this moves
/// into the worker.
pub struct OrtCompiled {
    sessions: Vec<Session>,
    /// This model as a holder of its GPU's memory ([`vram::hold`]) while its sessions live.
    vram_hold: Option<vram::Hold>,
    pub info: ModelInfo,
}

impl std::fmt::Debug for OrtCompiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtCompiled")
            .field("sessions", &self.sessions.len())
            .field("info", &self.info)
            .finish()
    }
}

impl OrtCompiled {
    /// Start one thread per session ([`OrtBackend`]).
    pub(crate) fn into_backend(self) -> Result<OrtBackend> {
        let image_idx = self
            .info
            .inputs
            .iter()
            .position(|p| p.name == self.info.image_input)
            .with_context(|| {
                format!(
                    "model {}: image input '{}' not among inputs",
                    self.info.path.display(),
                    self.info.image_input
                )
            })?;
        let image_shape = self
            .info
            .inputs
            .get(image_idx)
            .with_context(|| {
                format!(
                    "model {}: image input index out of range",
                    self.info.path.display()
                )
            })?
            .shape
            .clone();
        let run = Arc::new(RunCtx {
            info: self.info.clone(),
            image_idx,
            image_shape,
        });
        let (done_tx, done_rx) = crossbeam_channel::unbounded();
        let hold = self.vram_hold.map(Arc::new);
        let mut slots = Vec::with_capacity(self.sessions.len());
        for (slot, session) in self.sessions.into_iter().enumerate() {
            let (job_tx, job_rx) = crossbeam_channel::bounded::<OrtJob>(1);
            let (res_tx, res_rx) = crossbeam_channel::bounded::<OrtResult>(1);
            let (ctx, done) = (run.clone(), done_tx.clone());
            let thread = spawn_session(slot, hold.clone(), move || {
                session_thread(session, &ctx, slot, &job_rx, &res_tx, &done);
            })
            .context("spawning an ONNX Runtime session thread")?;
            slots.push(OrtSlot {
                jobs: Some(job_tx),
                results: res_rx,
                thread: Some(thread),
                chw: Vec::new(),
                outs: Vec::new(),
                busy: false,
                seq: 0,
                dead: None,
            });
        }
        Ok(OrtBackend {
            info: self.info,
            slots,
            _vram_hold: hold,
            done_rx,
            abort_timeout: ABORT_TIMEOUT,
        })
    }
}

/// What a session thread needs to feed its session.
struct RunCtx {
    info: ModelInfo,
    image_idx: usize,
    image_shape: Vec<i64>,
}

/// One inference for a session thread; the buffers travel back in [`OrtResult`].
struct OrtJob {
    seq: u64,
    chw: OrtChw,
    extra: Vec<ExtraInput>,
    outs: Vec<NamedOutput>,
}

struct OrtResult {
    chw: OrtChw,
    outs: Vec<NamedOutput>,
    result: Result<()>,
}

/// The image tensor of an [`OrtJob`]: the slot's own buffer holding a copy
/// ([`OrtBackend::submit`]), or the caller's pooled tensor handed over without a copy
/// ([`OrtBackend::submit_pooled`]), which goes back to its pool when the slot is collected.
enum OrtChw {
    Slot(Vec<f32>),
    Pooled(crate::prep::PooledBuf),
}

impl std::ops::Deref for OrtChw {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            OrtChw::Slot(v) => v,
            OrtChw::Pooled(b) => b,
        }
    }
}

/// A session thread and the buffers of its slot while idle.
struct OrtSlot {
    jobs: Option<crossbeam_channel::Sender<OrtJob>>,
    results: crossbeam_channel::Receiver<OrtResult>,
    thread: Option<std::thread::JoinHandle<()>>,
    chw: Vec<f32>,
    outs: Vec<NamedOutput>,
    busy: bool,
    seq: u64,
    /// Why the slot's session thread is gone (panicked, exited, or hung past
    /// [`ABORT_TIMEOUT`]); every later `submit` fails with [`SessionDead`] (BE-10).
    dead: Option<String>,
}

impl OrtSlot {
    /// Mark the slot dead (first reason wins) and return the matching error.
    fn kill(&mut self, slot: usize, reason: String) -> anyhow::Error {
        let reason = self.dead.get_or_insert(reason).clone();
        self.jobs = None;
        self.busy = false;
        SessionDead { slot, reason }.into()
    }
}

/// How long [`OrtBackend::abort`] (and so `Drop`) waits for an inference in flight before it
/// gives the session up as hung. A session wedged in `Session::run` (a driver hang) cannot be
/// interrupted: its thread is detached and keeps the session (and its GPU memory) until the
/// call returns, possibly for the rest of the process (BE-10).
pub(crate) const ABORT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Per-worker ONNX Runtime inference state: one session per slot, each on its own thread
/// (`Session::run` takes `&mut self`, so one inference in flight per session).
pub struct OrtBackend {
    info: ModelInfo,
    slots: Vec<OrtSlot>,
    /// [`OrtCompiled`]'s registration, shared with every session thread ([`spawn_session`]):
    /// it ends when the backend and every session thread are gone, so a thread detached while
    /// hung in `Session::run` (whose session keeps its GPU memory) still names the model in
    /// [`vram::holders`].
    _vram_hold: Option<Arc<vram::Hold>>,
    done_rx: crossbeam_channel::Receiver<Completion>,
    /// [`ABORT_TIMEOUT`]; shorter in tests.
    abort_timeout: std::time::Duration,
}

impl std::fmt::Debug for OrtBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtBackend")
            .field("info", &self.info)
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

/// Spawns session thread `slot` running `body`. `hold` (the model's [`vram::Hold`]) is dropped
/// only after `body` returned, that is after the session it owns was dropped: a thread that
/// `Drop` detaches while hung in the driver keeps the registration for exactly as long as the
/// session's GPU memory stays in use (RA-084).
fn spawn_session(
    slot: usize,
    hold: Option<Arc<vram::Hold>>,
    body: impl FnOnce() + Send + 'static,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("ort-session-{slot}"))
        .spawn(move || {
            body();
            drop(hold);
        })
}

fn session_thread(
    mut session: Session,
    ctx: &RunCtx,
    slot: usize,
    jobs: &crossbeam_channel::Receiver<OrtJob>,
    results: &crossbeam_channel::Sender<OrtResult>,
    done: &crossbeam_channel::Sender<Completion>,
) {
    serve_jobs(slot, jobs, results, done, |job| {
        run_session(&mut session, ctx, &job.chw, &job.extra, &mut job.outs)
    });
}

/// The job loop of a session thread, with `run` doing the inference.
fn serve_jobs(
    slot: usize,
    jobs: &crossbeam_channel::Receiver<OrtJob>,
    results: &crossbeam_channel::Sender<OrtResult>,
    done: &crossbeam_channel::Sender<Completion>,
    mut run: impl FnMut(&mut OrtJob) -> Result<()>,
) {
    while let Ok(mut job) = jobs.recv() {
        // A panic inside `ort` (e.g. an assertion) must not leave the job without an answer
        // or a completion: catch it, answer this job with `SessionDead` and stop the thread,
        // since the session's state after a panic is unknown (BE-10).
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut job)));
        let (result, panicked) = match ran {
            Ok(r) => (r, false),
            Err(payload) => {
                let reason = format!(
                    "session thread {slot} panicked during inference ({})",
                    panic_message(payload.as_ref())
                );
                (Err(SessionDead { slot, reason }.into()), true)
            }
        };
        let at = Instant::now();
        let r = OrtResult {
            chw: job.chw,
            outs: job.outs,
            result,
        };
        if results.send(r).is_err() {
            break;
        }
        let _ = done.send(Completion {
            slot,
            seq: job.seq,
            at,
        });
        if panicked {
            break;
        }
    }
}

/// The text of a panic payload (`&str` or `String`), else a placeholder.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// Run `session` on one image (+ extra inputs) and copy its outputs into `outs`.
fn run_session(
    session: &mut Session,
    ctx: &RunCtx,
    chw: &[f32],
    extra: &[ExtraInput],
    outs: &mut Vec<NamedOutput>,
) -> Result<()> {
    let info = &ctx.info;
    let where_ = || format!("model {} on {}", info.path.display(), info.device.spec);
    let expected: i64 = ctx.image_shape.iter().product();
    if chw.len() as i64 != expected {
        bail!(
            "{}: input '{}' expects {expected} f32 values ({:?}), got {}",
            where_(),
            info.image_input,
            ctx.image_shape,
            chw.len()
        );
    }
    let image = TensorRef::from_array_view((ctx.image_shape.clone(), chw))
        .map_err(ort_err)
        .with_context(|| format!("{}: creating the image tensor", where_()))?;
    let mut inputs: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(1 + extra.len());
    inputs.push((info.image_input.clone(), image.into()));
    for e in extra {
        let i = extra_input_index(&info.inputs, ctx.image_idx, &e.name).with_context(|| {
            format!(
                "{}: extra input '{}' not among model inputs {:?}",
                where_(),
                e.name,
                info.inputs.iter().map(|p| &p.name).collect::<Vec<_>>()
            )
        })?;
        let port = info
            .inputs
            .get(i)
            .with_context(|| format!("{}: input index {i} out of range", where_()))?;
        let value = extra_value(port, e).with_context(where_)?;
        inputs.push((port.name.clone(), value.into()));
    }
    let outputs = session
        .run(inputs)
        .map_err(|e| anyhow::anyhow!("{}: inference failed: {e}", where_()))?;
    NamedOutput::resize_slots(outs, info.outputs.len());
    for (spec, out) in info.outputs.iter().zip(outs.iter_mut()) {
        let v = outputs
            .get(&spec.name)
            .with_context(|| format!("{}: missing output '{}'", where_(), spec.name))?;
        read_output_into(&spec.name, v, out).with_context(where_)?;
    }
    Ok(())
}

/// An owned input tensor of `port`'s element type built from `extra` (converted as needed).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "extra inputs are small image-size values built by the model family code; `as` keeps ONNX Cast semantics (saturating float->int, wrapping int64->int32) when the port type differs"
)]
pub(crate) fn extra_value(port: &PortSpec, extra: &ExtraInput) -> Result<DynValue> {
    let n: usize = extra.shape.iter().product();
    let len = match &extra.data {
        ExtraData::I64(v) => v.len(),
        ExtraData::I32(v) => v.len(),
        ExtraData::F32(v) => v.len(),
    };
    if len != n {
        bail!(
            "extra input '{}': {len} values for shape {:?}",
            extra.name,
            extra.shape
        );
    }
    let shape: Vec<i64> = extra.shape.iter().map(|&d| d as i64).collect();
    let ctx = || format!("creating input tensor '{}'", extra.name);
    Ok(match port.elem {
        PortElem::I64 => {
            let v: Vec<i64> = match &extra.data {
                ExtraData::I64(v) => v.clone(),
                ExtraData::I32(v) => v.iter().map(|&x| x as i64).collect(),
                ExtraData::F32(v) => v.iter().map(|&x| x as i64).collect(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        PortElem::I32 => {
            let v: Vec<i32> = match &extra.data {
                ExtraData::I64(v) => v.iter().map(|&x| x as i32).collect(),
                ExtraData::I32(v) => v.clone(),
                ExtraData::F32(v) => v.iter().map(|&x| x as i32).collect(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        PortElem::F32 => {
            let v: Vec<f32> = match &extra.data {
                ExtraData::I64(v) => v.iter().map(|&x| x as f32).collect(),
                ExtraData::I32(v) => v.iter().map(|&x| x as f32).collect(),
                ExtraData::F32(v) => v.clone(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        other => bail!(
            "extra input '{}': unsupported port element type {other:?}",
            extra.name
        ),
    })
}

/// Copy one output value into `out`, reusing its buffer.
fn read_output_into(name: &str, v: &DynValue, out: &mut NamedOutput) -> Result<()> {
    let ty = v.dtype().tensor_type();
    let shape: &[i64] = match ty {
        Some(TensorElementType::Float32) => {
            let (s, d) = v.try_extract_tensor::<f32>().map_err(ort_err)?;
            out.data.f32_mut().extend_from_slice(d);
            s
        }
        Some(TensorElementType::Int64) => {
            let (s, d) = v.try_extract_tensor::<i64>().map_err(ort_err)?;
            out.data.i64_mut().extend_from_slice(d);
            s
        }
        Some(TensorElementType::Int32) => {
            let (s, d) = v.try_extract_tensor::<i32>().map_err(ort_err)?;
            out.data.i32_mut().extend_from_slice(d);
            s
        }
        other => bail!("output '{name}': unsupported element type {other:?}"),
    };
    out.shape.clear();
    out.shape
        .extend(shape.iter().map(|&d| usize::try_from(d).unwrap_or(0)));
    // Element count must match the shape before any consumer indexes by it.
    let expected = out
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let len = out.data.len();
    if expected != Some(len) {
        bail!(
            "output '{name}': {len} values but shape {:?} ({})",
            out.shape,
            expected.map_or("element count overflows".to_string(), |n| format!(
                "{n} elements"
            ))
        );
    }
    if out.name != name {
        out.name = name.to_string();
    }
    Ok(())
}

impl OrtBackend {
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Number of sessions (slots).
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Completions of submitted inferences.
    pub(crate) fn completions(&self) -> &crossbeam_channel::Receiver<Completion> {
        &self.done_rx
    }

    pub(crate) fn is_busy(&self, slot: usize) -> bool {
        self.slots.get(slot).is_some_and(|s| s.busy)
    }

    /// False once any session thread is gone (panicked, exited or hung past
    /// [`ABORT_TIMEOUT`]): that slot answers every `submit` with [`SessionDead`] until the
    /// model is reloaded.
    pub(crate) fn is_healthy(&self) -> bool {
        self.slots.iter().all(|s| {
            s.dead.is_none()
                && s.jobs.is_some()
                && s.thread.as_ref().is_some_and(|t| !t.is_finished())
        })
    }

    /// Hand one inference to `slot`'s session thread (the image is copied into the slot's
    /// buffer). Returns its sequence number. A dead session thread fails with [`SessionDead`]
    /// (see [`super::is_session_dead`]).
    pub(crate) fn submit(&mut self, slot: usize, chw: &[f32], extra: &[ExtraInput]) -> Result<u64> {
        self.submit_input(
            slot,
            |s| {
                let mut buf = std::mem::take(&mut s.chw);
                buf.clear();
                buf.extend_from_slice(chw);
                OrtChw::Slot(buf)
            },
            extra,
        )
    }

    /// [`Self::submit`] without the copy: the session thread reads `chw` itself, and it goes back
    /// to its pool when the slot is collected (or with the thread, if that never returns).
    pub(crate) fn submit_pooled(
        &mut self,
        slot: usize,
        chw: crate::prep::PooledBuf,
        extra: &[ExtraInput],
    ) -> Result<u64> {
        self.submit_input(slot, |_| OrtChw::Pooled(chw), extra)
    }

    /// The checks and the hand-over of [`Self::submit`]; `input` makes the job's image tensor
    /// once the slot is known to be free and alive.
    fn submit_input(
        &mut self,
        slot: usize,
        input: impl FnOnce(&mut OrtSlot) -> OrtChw,
        extra: &[ExtraInput],
    ) -> Result<u64> {
        let n = self.slots.len();
        let where_ = format!(
            "model {} on {}",
            self.info.path.display(),
            self.info.device.spec
        );
        let s = self
            .slots
            .get_mut(slot)
            .with_context(|| format!("{where_}: no session {slot} (have {n})"))?;
        if s.busy {
            bail!("{where_}: session {slot} is busy");
        }
        if let Some(reason) = &s.dead {
            let reason = reason.clone();
            return Err(s.kill(slot, reason)).context(where_);
        }
        if s.thread.as_ref().is_none_or(|t| t.is_finished()) {
            return Err(s.kill(slot, format!("session thread {slot} exited"))).context(where_);
        }
        let chw = input(s);
        s.seq += 1;
        let job = OrtJob {
            seq: s.seq,
            chw,
            extra: extra.to_vec(),
            outs: std::mem::take(&mut s.outs),
        };
        let sent = match s.jobs.as_ref() {
            Some(tx) => tx.send(job).is_ok(),
            None => false,
        };
        if !sent {
            return Err(s.kill(slot, format!("session thread {slot} exited"))).context(where_);
        }
        s.busy = true;
        Ok(s.seq)
    }

    /// The outputs of `slot`'s inference (waits for it); give the buffers back with
    /// `Self::recycle`.
    pub fn collect(&mut self, slot: usize) -> Result<Vec<NamedOutput>> {
        self.collect_within(slot, None)
    }

    /// [`Self::collect`], giving up after `limit` (None = wait forever). On timeout the slot is
    /// marked dead (hung session) and [`SessionDead`] is returned.
    fn collect_within(
        &mut self,
        slot: usize,
        limit: Option<std::time::Duration>,
    ) -> Result<Vec<NamedOutput>> {
        let where_ = format!(
            "model {} on {}",
            self.info.path.display(),
            self.info.device.spec
        );
        let s = self
            .slots
            .get_mut(slot)
            .with_context(|| format!("{where_}: no session {slot}"))?;
        if !s.busy {
            bail!("{where_}: session {slot} has no inference in flight");
        }
        let received = match limit {
            None => s.results.recv().map_err(|_| None),
            Some(t) => s
                .results
                .recv_timeout(t)
                .map_err(|e| e.is_timeout().then_some(t)),
        };
        let r = match received {
            Ok(r) => r,
            Err(None) => {
                return Err(s.kill(slot, format!("session thread {slot} exited"))).context(where_);
            }
            Err(Some(t)) => {
                let reason = format!(
                    "session {slot} still running after {} s (hung in Session::run?); its thread \
                     is left behind",
                    t.as_secs()
                );
                return Err(s.kill(slot, reason)).context(where_);
            }
        };
        s.busy = false;
        // The slot's buffer is kept for the next copy; a pooled tensor goes back to its pool.
        if let OrtChw::Slot(buf) = r.chw {
            s.chw = buf;
        }
        match r.result {
            Ok(()) => Ok(r.outs),
            Err(e) => {
                s.outs = r.outs;
                if let Some(dead) = e.downcast_ref::<SessionDead>() {
                    s.dead.get_or_insert_with(|| dead.reason.clone());
                    s.jobs = None;
                }
                Err(e.context(where_))
            }
        }
    }

    /// Return output buffers lent by [`Self::collect`] to `slot`.
    pub(crate) fn recycle(&mut self, slot: usize, outs: Vec<NamedOutput>) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.outs = outs;
        }
    }

    /// Wait for every inference in flight and discard it (a running session cannot be
    /// interrupted). Waits at most `ABORT_TIMEOUT` in total (all slots together); a session still running then is
    /// marked dead and logged, so shutdown and reloads never block on a hung session.
    pub fn abort(&mut self) {
        let deadline = Instant::now() + self.abort_timeout;
        let busy: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots.get(i).is_some_and(|s| s.busy))
            .collect();
        for slot in busy {
            let left = deadline.saturating_duration_since(Instant::now());
            if let Err(e) = self.collect_within(slot, Some(left))
                && super::is_session_dead(&e)
            {
                tracing::warn!(
                    "{e:#}; the inference was abandoned; if this repeats, the GPU driver is \
                     probably hung: restart the service (or the machine)"
                );
            }
        }
    }

    /// Run one inference synchronously on session 0. `chw` must be `3 * w * h` f32 values in
    /// CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        let mut outs = Vec::new();
        self.infer_into(chw, extra, &mut outs)?;
        Ok(outs)
    }

    /// [`Self::infer`] into `outs`, reusing its buffers.
    pub fn infer_into(
        &mut self,
        chw: &[f32],
        extra: &[ExtraInput],
        outs: &mut Vec<NamedOutput>,
    ) -> Result<()> {
        // Completions of earlier synchronous calls are never consumed; drop them.
        while self.done_rx.try_recv().is_ok() {}
        self.recycle(0, std::mem::take(outs));
        self.submit(0, chw, extra)?;
        *outs = self.collect(0)?;
        Ok(())
    }
}

impl Drop for OrtBackend {
    /// Bounded: [`Self::abort`] waits at most `ABORT_TIMEOUT`, then the threads get a short
    /// grace period to exit; a thread still running (hung in `Session::run`) is detached with
    /// a warning instead of blocking the worker's shutdown or a config reload (BE-10).
    fn drop(&mut self) {
        self.abort();
        for s in &mut self.slots {
            s.jobs = None;
        }
        let deadline = Instant::now() + JOIN_GRACE;
        for (slot, s) in self.slots.iter_mut().enumerate() {
            let Some(t) = s.thread.take() else { continue };
            while !t.is_finished() && Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            if !t.is_finished() {
                tracing::warn!(
                    "model {}: ONNX Runtime session thread {slot} did not stop within {} ms \
                     (probably hung in the driver); detaching it, its memory stays in use \
                     (and the model stays listed as holding it) until it returns; restart the \
                     service if GPU memory runs short",
                    self.info.path.display(),
                    JOIN_GRACE.as_millis()
                );
                drop(t);
            } else if t.join().is_err() {
                tracing::warn!("ONNX Runtime session thread {slot} panicked");
            }
        }
    }
}

/// How long `Drop` waits for the session threads to exit after their job channels close.
const JOIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// GPU name for `target` at device `index`: CUDA/TensorRT index = CUDA ordinal (the driver's
/// order, [`cuda_ordinals`], not the platform order), DirectML index = DXGI adapter index.
/// Empty for other targets or an unknown index.
fn gpu_full_name(hw: &HardwareInfo, target: Target, index: u32) -> String {
    match target {
        Target::Cuda | Target::TensorRt => cuda_ordinals(hw)
            .into_iter()
            .find(|(o, _)| *o == index)
            .map(|(_, g)| g.name.clone())
            .unwrap_or_default(),
        Target::DirectMl => hw
            .gpus
            .iter()
            .find(|g| g.index == index)
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, shape: &[i64], elem: PortElem) -> PortSpec {
        PortSpec {
            name: name.into(),
            shape: shape.to_vec(),
            elem,
        }
    }

    fn syms(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn versions() {
        assert_eq!(parse_version("1.24.4"), Some((1, 24)));
        assert_eq!(parse_version(" 1.17.0\n"), Some((1, 17)));
        assert_eq!(parse_version("garbage"), None);
        assert!(version_supported("1.24.4", 17));
        assert!(version_supported("1.17.0", 17));
        assert!(!version_supported("1.16.3", 17));
        assert!(!version_supported("2.0.0", 17));
        assert!(!version_supported("", 17));
    }

    #[test]
    fn fixed_shapes() {
        assert_eq!(
            fixed_shape(&[-1, 3, -1, -1], &DEFAULT_IMAGE_SHAPE),
            vec![1, 3, 640, 640]
        );
        assert_eq!(
            fixed_shape(&[1, 3, 480, 640], &DEFAULT_IMAGE_SHAPE),
            vec![1, 3, 480, 640]
        );
        assert_eq!(
            fixed_shape(&[-1, 2], &DEFAULT_TARGET_SIZES_SHAPE),
            vec![1, 2]
        );
    }

    #[test]
    fn overrides_for_named_dynamic_dims() {
        // RT-DETR style: batch "N" shared by both inputs, image H/W named.
        let inputs = [
            port("images", &[-1, 3, -1, -1], PortElem::F32),
            port("orig_target_sizes", &[-1, 2], PortElem::I64),
        ];
        let s = [syms(&["N", "", "H", "W"]), syms(&["N", ""])];
        assert_eq!(
            dimension_overrides(&inputs, &s, 0, Some(1), &DEFAULT_IMAGE_SHAPE),
            vec![
                ("N".to_string(), 1),
                ("H".to_string(), 640),
                ("W".to_string(), 640)
            ]
        );
        // Static model: nothing to do.
        let st = [port("images", &[1, 3, 640, 640], PortElem::F32)];
        assert!(
            dimension_overrides(
                &st,
                &[syms(&["", "", "", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
        // Unnamed dynamic dims cannot be overridden.
        let un = [port("images", &[-1, 3, -1, -1], PortElem::F32)];
        assert!(
            dimension_overrides(
                &un,
                &[syms(&["", "", "", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
        // One symbol for H and the channel count would need 3 and 640: left dynamic.
        let bad = [port("images", &[1, -1, -1, 640], PortElem::F32)];
        assert!(
            dimension_overrides(
                &bad,
                &[syms(&["", "d", "d", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
    }

    #[test]
    fn coreml_rewrite_switch() {
        for on in [None, Some("1"), Some("true"), Some(""), Some("yes")] {
            assert!(coreml_rewrite_enabled(on), "{on:?}");
        }
        for off in ["0", "false", "OFF", " no "] {
            assert!(!coreml_rewrite_enabled(Some(off)), "{off}");
        }
    }

    #[test]
    fn cudnn_search_override_parses() {
        for unset in [None, Some(""), Some("  ")] {
            assert_eq!(parse_conv_search(unset), Ok(None), "{unset:?}");
        }
        for (v, want) in [
            ("exhaustive", ConvSearch::Exhaustive),
            (" HEURISTIC ", ConvSearch::Heuristic),
            ("Default", ConvSearch::Default),
        ] {
            assert_eq!(parse_conv_search(Some(v)), Ok(Some(want)), "{v}");
        }
        assert_eq!(parse_conv_search(Some(" fast ")), Err("fast".to_string()));
        // Each search reaches the provider options under ONNX Runtime's own name.
        for s in [
            ConvSearch::Exhaustive,
            ConvSearch::Heuristic,
            ConvSearch::Default,
        ] {
            let opts = format!("{:?}", cuda_provider_with(0, None, s));
            let want = format!("\"cudnn_conv_algo_search\": \"{}\"", s.as_str());
            assert!(opts.contains(&want), "{want} in {opts}");
        }
        assert_eq!(ENV_CUDNN_CONV_ALGO_SEARCH, "BOP_CUDNN_CONV_ALGO_SEARCH");
    }

    #[test]
    fn cudnn_search_override_logs_once_and_plainly() {
        assert_eq!(conv_search_choice(Ok(None)), (ConvSearch::Heuristic, None));
        // The default named explicitly: an info line, never "HEURISTIC instead of HEURISTIC".
        let (s, note) = conv_search_choice(Ok(Some(ConvSearch::Heuristic)));
        assert_eq!(s, ConvSearch::Heuristic);
        let (warn, msg) = note.unwrap();
        assert!(!warn && msg.contains("set to the default"), "{msg}");
        assert!(!msg.contains("instead of"), "{msg}");
        for other in [ConvSearch::Exhaustive, ConvSearch::Default] {
            let (s, note) = conv_search_choice(Ok(Some(other)));
            assert_eq!(s, other);
            let (warn, msg) = note.unwrap();
            assert!(warn, "{msg}");
            assert!(
                msg.contains(&format!("={} instead of HEURISTIC", other.as_str())),
                "{msg}"
            );
        }
        let (s, note) = conv_search_choice(Err("fast".into()));
        assert_eq!(s, ConvSearch::Heuristic);
        let (warn, msg) = note.unwrap();
        assert!(warn && msg.contains("'fast'"), "{msg}");
    }

    #[test]
    fn cuda_sessions_grow_their_arena_by_the_request_and_search_heuristically() {
        let plain = format!("{:?}", cuda_provider_with(1, None, ConvSearch::Heuristic));
        for want in [
            "\"device_id\": \"1\"",
            "\"arena_extend_strategy\": \"kSameAsRequested\"",
            "\"cudnn_conv_algo_search\": \"HEURISTIC\"",
        ] {
            assert!(plain.contains(want), "{want} in {plain}");
        }
        assert!(!plain.contains("gpu_mem_limit"), "{plain}");
        let capped = format!("{:?}", cuda_provider(0, Some(1536)));
        // Without the measurement override every session searches heuristically.
        if std::env::var_os(ENV_CUDNN_CONV_ALGO_SEARCH).is_none() {
            assert!(capped.contains("\"cudnn_conv_algo_search\": \"HEURISTIC\""));
        }
        assert!(
            capped.contains("\"gpu_mem_limit\": \"1610612736\""),
            "{capped}"
        );
    }

    #[test]
    fn ort_spinning_switch() {
        for on in [Some("1"), Some("true"), Some(" ON "), Some("yes")] {
            assert!(ort_spinning_enabled(on), "{on:?}");
        }
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("no"),
            Some("2"),
        ] {
            assert!(!ort_spinning_enabled(off), "{off:?}");
        }
    }

    /// RA-094: the CPU arena stays on unless the measurement switch turns it off.
    #[test]
    fn cpu_arena_switch() {
        assert_eq!(ENV_ORT_CPU_ARENA, "BOP_ORT_CPU_ARENA");
        for on in [
            None,
            Some(""),
            Some("1"),
            Some("on"),
            Some("yes"),
            Some("2"),
        ] {
            assert!(cpu_arena_enabled(on), "{on:?}");
        }
        for off in [Some("0"), Some("false"), Some(" OFF "), Some("no")] {
            assert!(!cpu_arena_enabled(off), "{off:?}");
        }
    }

    /// RA-086: the probe loads the CUDA runtime and cuDNN (BE-02) but only looks for TensorRT's
    /// `nvinfer`, which a TensorRT session loads by path right before its provider registers.
    #[test]
    fn tensorrt_is_probed_by_file_and_loaded_by_its_session() {
        assert_eq!(dep_check(dep_names::CUDART), DepCheck::Load);
        assert_eq!(dep_check(dep_names::CUDNN), DepCheck::Load);
        assert_eq!(dep_check(dep_names::NVINFER), DepCheck::Exists);
        assert_eq!(session_preloads(Target::TensorRt), dep_names::NVINFER);
        for t in [Target::Cpu, Target::Cuda, Target::DirectMl, Target::CoreMl] {
            assert!(session_preloads(t).is_empty(), "{t:?}");
        }
        // What `nvidia_deps` asks its check for, and how: nvinfer only for TensorRT.
        let asked = Mutex::new(Vec::new());
        let check = |names: &[&str], _: &[&Path]| {
            asked
                .lock()
                .unwrap()
                .push((names[0].to_string(), dep_check(names)));
            names != dep_names::NVINFER
        };
        let dir = Path::new("ort-dir");
        let cuda = nvidia_deps_with(Target::Cuda, dir, None, &check);
        assert_eq!(cuda.len(), 3);
        let trt = nvidia_deps_with(Target::TensorRt, dir, None, &check);
        let nv = trt.iter().find(|d| d.label == "TensorRT 10").unwrap();
        assert!(!nv.loadable && !nv.cuda_lib);
        let asked = asked.into_inner().unwrap();
        let loaded: Vec<&str> = asked
            .iter()
            .filter(|(_, c)| *c == DepCheck::Load)
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(!loaded.contains(&dep_names::NVINFER[0]), "{asked:?}");
        assert_eq!(
            asked
                .iter()
                .filter(|(n, _)| n == dep_names::NVINFER[0])
                .count(),
            1,
            "{asked:?}"
        );
        // A TensorRT install that is not there: the reason names it, nothing was loaded.
        let empty = std::env::temp_dir().join(format!("bop-no-trt-{}", uuid::Uuid::new_v4()));
        assert!(!any_present(&["bop-not-a-library.so.10"], &[&empty]));
    }

    #[test]
    fn rtdetr_on_coreml_needs_pinned_dims() {
        let inputs = [
            port("images", &[-1, 3, 640, 640], PortElem::F32),
            port("orig_target_sizes", &[-1, 2], PortElem::I64),
        ];
        let named = [syms(&["N", "", "", ""]), syms(&["N", ""])];
        let ov = dimension_overrides(&inputs, &named, 0, Some(1), &DEFAULT_IMAGE_SHAPE);
        assert_eq!(ov, vec![("N".to_string(), 1)]);
        assert!(unpinned_dims(&inputs, &named, &ov).is_empty());
        // Named batch, pinned: runs on CoreML.
        assert_eq!(unsupported_on(Target::CoreMl, &inputs, &named, &ov), None);
        // Without the override (or with unnamed dims) the batch stays dynamic: refused.
        assert_eq!(
            unpinned_dims(&inputs, &named, &[]),
            vec!["images[0]", "orig_target_sizes[0]"]
        );
        let r = unsupported_on(Target::CoreMl, &inputs, &named, &[]).unwrap();
        assert!(r.contains("images[0], orig_target_sizes[0]"), "{r}");
        let unnamed = [syms(&["", "", "", ""]), syms(&["", ""])];
        let ov = dimension_overrides(&inputs, &unnamed, 0, Some(1), &DEFAULT_IMAGE_SHAPE);
        assert!(unsupported_on(Target::CoreMl, &inputs, &unnamed, &ov).is_some());
        // Other targets and non-RT-DETR models are never refused.
        assert_eq!(unsupported_on(Target::Cpu, &inputs, &unnamed, &[]), None);
        let yolo = [port("images", &[-1, 3, -1, -1], PortElem::F32)];
        assert_eq!(
            unsupported_on(Target::CoreMl, &yolo, &[syms(&["", "", "", ""])], &[]),
            None
        );
        // Fully static RT-DETR: nothing to pin, runs.
        let st = [
            port("images", &[1, 3, 640, 640], PortElem::F32),
            port("orig_target_sizes", &[1, 2], PortElem::I64),
        ];
        assert_eq!(unsupported_on(Target::CoreMl, &st, &[], &[]), None);
    }

    #[test]
    fn nvidia_reasons_name_what_is_missing() {
        let dep = |label, names, loadable| Dep {
            label,
            names,
            loadable,
            cuda_lib: label != "TensorRT 10",
        };
        assert_eq!(
            missing_deps_reason(
                "CUDA",
                &[
                    dep("the CUDA runtime", &["cudart64_12.dll"], true),
                    dep("cuDNN 9", &["cudnn64_9.dll"], true)
                ]
            ),
            None
        );
        let r = missing_deps_reason(
            "CUDA",
            &[
                dep(
                    "the CUDA runtime",
                    &["libcudart.so.12", "libcudart.so.13"],
                    true,
                ),
                dep("cuDNN 9", &["libcudnn.so.9"], false),
            ],
        )
        .unwrap();
        assert!(
            r.starts_with("CUDA execution provider needs cuDNN 9 (libcudnn.so.9)"),
            "{r}"
        );
        assert!(!r.contains("cudart"), "{r}");
        let r = missing_deps_reason(
            "TensorRT",
            &[
                dep("the CUDA runtime", &["a", "b"], false),
                dep("TensorRT 10", &["c"], false),
            ],
        )
        .unwrap();
        assert!(
            r.contains("the CUDA runtime (a / b) and TensorRT 10 (c)"),
            "{r}"
        );

        // Downloadable CUDA libraries: the reason says how to get them.
        let hint = cuda_download_hint("linux", "x86_64").unwrap();
        assert!(hint.contains("fetch --resource nvidia-cuda-libs"), "{hint}");
        assert!(
            hint.contains("allow_large_downloads") && hint.contains("GB"),
            "{hint}"
        );
        assert!(cuda_download_hint("macos", "aarch64").is_none());
        let r = missing_deps_reason_with(
            "CUDA",
            &[dep("cuDNN 9", &["libcudnn.so.9"], false)],
            Some(&hint),
        )
        .unwrap();
        assert!(r.ends_with(&hint), "{r}");
        // Only TensorRT missing: not fixed by the download.
        let r = missing_deps_reason_with(
            "TensorRT",
            &[dep("TensorRT 10", &["c"], false)],
            Some(&hint),
        )
        .unwrap();
        assert!(r.contains("install it"), "{r}");
    }

    #[test]
    fn extra_values_are_validated() {
        let e = ExtraInput {
            name: "orig_target_sizes".into(),
            shape: vec![1, 2],
            data: ExtraData::I64(vec![640]),
        };
        let p = port("orig_target_sizes", &[1, 2], PortElem::I64);
        let err = extra_value(&p, &e).unwrap_err();
        assert!(format!("{err:#}").contains("1 values for shape [1, 2]"));
        let e = ExtraInput {
            data: ExtraData::I64(vec![640, 640]),
            ..e
        };
        let p = port("orig_target_sizes", &[1, 2], PortElem::U8);
        assert!(extra_value(&p, &e).is_err());
    }

    fn nvidia(name: &str, index: u32, ordinal: Option<u32>) -> super::super::detect::GpuAdapter {
        use super::super::detect::{ComputeCapability, CudaInfo, GpuAdapter};
        GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: name.into(),
            vram_mb: 8060,
            index,
            discrete: true,
            cuda: ordinal.map(|ordinal| CudaInfo {
                ordinal,
                compute_capability: ComputeCapability::new(8, 6),
            }),
        }
    }

    #[test]
    fn full_name_follows_cuda_ordinals() {
        use super::super::detect::GpuAdapter;
        // DXGI lists the 3060 first (index 0), the driver the 4090 (ordinal 0, index 2).
        let igpu = GpuAdapter {
            vendor: GpuVendor::Intel,
            name: "Intel(R) UHD Graphics 630".into(),
            vram_mb: 128,
            index: 1,
            discrete: false,
            cuda: None,
        };
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![
                nvidia("NVIDIA GeForce RTX 3060", 0, Some(1)),
                igpu,
                nvidia("NVIDIA GeForce RTX 4090", 2, Some(0)),
            ],
        );
        assert_eq!(
            gpu_full_name(&hw, Target::Cuda, 0),
            "NVIDIA GeForce RTX 4090"
        );
        assert_eq!(
            gpu_full_name(&hw, Target::TensorRt, 1),
            "NVIDIA GeForce RTX 3060"
        );
        assert_eq!(gpu_full_name(&hw, Target::Cuda, 2), "");
        // DirectML: the DXGI index.
        assert_eq!(
            gpu_full_name(&hw, Target::DirectMl, 0),
            "NVIDIA GeForce RTX 3060"
        );
        assert_eq!(
            gpu_full_name(&hw, Target::DirectMl, 1),
            "Intel(R) UHD Graphics 630"
        );
        assert_eq!(gpu_full_name(&hw, Target::Cpu, 0), "");
    }

    /// An `OrtBackend` whose slots run `run` instead of a session.
    fn fake_backend(
        slots: usize,
        run: impl Fn(&mut OrtJob) -> Result<()> + Send + Sync + Clone + 'static,
    ) -> OrtBackend {
        fake_backend_holding(slots, None, run)
    }

    /// [`fake_backend`] whose threads share `hold` as [`OrtCompiled::into_backend`]'s do.
    fn fake_backend_holding(
        slots: usize,
        hold: Option<vram::Hold>,
        run: impl Fn(&mut OrtJob) -> Result<()> + Send + Sync + Clone + 'static,
    ) -> OrtBackend {
        let (done_tx, done_rx) = crossbeam_channel::unbounded();
        let hold = hold.map(Arc::new);
        let slots = (0..slots)
            .map(|slot| {
                let (job_tx, job_rx) = crossbeam_channel::bounded::<OrtJob>(1);
                let (res_tx, res_rx) = crossbeam_channel::bounded::<OrtResult>(1);
                let (done, run) = (done_tx.clone(), run.clone());
                let thread = spawn_session(slot, hold.clone(), move || {
                    serve_jobs(slot, &job_rx, &res_tx, &done, run);
                })
                .unwrap();
                OrtSlot {
                    jobs: Some(job_tx),
                    results: res_rx,
                    thread: Some(thread),
                    chw: Vec::new(),
                    outs: Vec::new(),
                    busy: false,
                    seq: 0,
                    dead: None,
                }
            })
            .collect();
        OrtBackend {
            info: ModelInfo {
                path: PathBuf::from("fake.onnx"),
                inputs: Vec::new(),
                outputs: Vec::new(),
                device: DeviceInfo {
                    requested: "ort:cpu".into(),
                    actual: "CPU".into(),
                    full_name: String::new(),
                    fell_back: false,
                    runtime: Runtime::Ort,
                    spec: "ort:cpu".into(),
                },
                image_input: "images".into(),
                input_size: (1, 1),
                compile_ms: 0,
            },
            slots,
            _vram_hold: hold,
            done_rx,
            abort_timeout: std::time::Duration::from_millis(100),
        }
    }

    /// A pooled tensor reaches the session thread as it is (the same allocation, no copy into
    /// the slot) and goes back to its pool once the slot is collected; a slice is still copied.
    #[test]
    fn pooled_tensors_are_handed_over_without_a_copy() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let mut b = fake_backend(1, move |job: &mut OrtJob| {
            let pooled = matches!(job.chw, OrtChw::Pooled(_));
            log.lock()
                .unwrap()
                .push((job.chw.as_ptr() as usize, job.chw.to_vec(), pooled));
            Ok(())
        });
        let pool = crate::prep::TensorPool::new();
        let mut chw = pool.take(3);
        chw.vec_mut().extend_from_slice(&[1.0, 2.0, 3.0]);
        let ptr = chw.as_ptr() as usize;
        b.submit_pooled(0, chw, &[]).unwrap();
        assert_eq!(pool.free_count(3), 0, "held while the inference runs");
        b.collect(0).unwrap();
        assert_eq!(pool.free_count(3), 1, "back in its pool once collected");
        b.submit(0, &[4.0, 5.0, 6.0], &[]).unwrap();
        b.collect(0).unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0], (ptr, vec![1.0, 2.0, 3.0], true));
        assert_eq!((seen[1].1.clone(), seen[1].2), (vec![4.0, 5.0, 6.0], false));
        assert_ne!(seen[1].0, ptr, "a slice goes into the slot's own buffer");
    }

    #[test]
    fn panicking_session_is_reported_dead() {
        let mut b = fake_backend(2, |job: &mut OrtJob| {
            assert!(job.chw.first() != Some(&13.0), "ort assertion");
            Ok(())
        });
        assert!(b.is_healthy());
        let seq = b.submit(0, &[1.0], &[]).unwrap();
        assert!(b.collect(0).is_ok());
        assert_eq!(b.completions().recv().unwrap().seq, seq);
        // The panic answers the job (and its completion) with SessionDead.
        let seq = b.submit(0, &[13.0], &[]).unwrap();
        let err = b.collect(0).unwrap_err();
        assert!(super::super::is_session_dead(&err), "{err:#}");
        assert!(format!("{err:#}").contains("panicked"), "{err:#}");
        assert_eq!(b.completions().recv().unwrap().seq, seq);
        assert!(!b.is_healthy());
        // The slot stays dead; the other one still works.
        let err = b.submit(0, &[1.0], &[]).unwrap_err();
        assert!(super::super::is_session_dead(&err), "{err:#}");
        assert!(!b.is_busy(0));
        b.submit(1, &[1.0], &[]).unwrap();
        assert!(b.collect(1).is_ok());
    }

    #[test]
    fn hung_session_does_not_block_abort_or_drop() {
        let (release_tx, release_rx) = crossbeam_channel::bounded::<()>(0);
        let mut b = fake_backend(1, move |_job: &mut OrtJob| {
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
            Ok(())
        });
        b.submit(0, &[1.0], &[]).unwrap();
        let t0 = Instant::now();
        b.abort();
        assert!(t0.elapsed() < std::time::Duration::from_secs(5));
        assert!(!b.is_busy(0));
        assert!(!b.is_healthy());
        let err = b.submit(0, &[1.0], &[]).unwrap_err();
        assert!(format!("{err:#}").contains("still running"), "{err:#}");
        let t0 = Instant::now();
        drop(b);
        // Bounded: abort (nothing busy any more) + JOIN_GRACE, then detached.
        assert!(t0.elapsed() < JOIN_GRACE + std::time::Duration::from_secs(2));
        drop(release_tx);
    }

    /// RA-084: a session thread detached while hung keeps the model registered as a holder of
    /// its GPU memory until the call returns; a backend whose threads stopped drops it at once.
    #[test]
    fn detached_session_keeps_its_vram_hold() {
        let held = |model: &str| vram::holders().iter().any(|h| h.model == model);
        let holder = |model: &str| vram::Holder {
            model: model.to_string(),
            device: "ort:cuda:0".into(),
            adapter: 0,
            held_mb: Some(1),
        };
        // A healthy backend: the registration ends with it.
        let quiet = format!("bop-quiet-{}", uuid::Uuid::new_v4());
        let mut b = fake_backend_holding(2, Some(vram::hold(holder(&quiet))), |_| Ok(()));
        b.submit(0, &[1.0], &[]).unwrap();
        b.collect(0).unwrap();
        assert!(held(&quiet));
        drop(b);
        assert!(
            !held(&quiet),
            "every thread joined: nothing holds the memory any more"
        );

        // A hung one: still listed after the drop, gone once the call returns.
        let hung = format!("bop-hung-{}", uuid::Uuid::new_v4());
        let (release_tx, release_rx) = crossbeam_channel::bounded::<()>(0);
        let mut b = fake_backend_holding(2, Some(vram::hold(holder(&hung))), move |_job| {
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
            Ok(())
        });
        b.submit(1, &[1.0], &[]).unwrap();
        drop(b);
        assert!(
            held(&hung),
            "the detached thread's session still holds its memory"
        );
        drop(release_tx);
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while held(&hung) && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!held(&hung), "released with the thread");
    }

    /// A `no_cache` CoreML load (a fold model) leaves nothing in `cache/` and nothing beside the
    /// model (the rewrite loads from memory) and computes what the cached load computes. Needs
    /// `models/IPcam-general.onnx` and an ONNX Runtime with CoreML in
    /// `target/release/onnxruntime/coreml` (or `ORT_DYLIB_PATH`):
    /// `cargo test --lib -- --ignored backend::ort::tests::no_cache_coreml`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs models/IPcam-general.onnx and an ONNX Runtime CoreML library"]
    fn no_cache_coreml_leaves_nothing_behind() {
        use super::super::{CoreOptions, OrtOptions, Runtimes, spec};
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let lib_dir = std::env::var_os("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| repo.join("target/release/onnxruntime/coreml"));
        let src = repo.join("models").join("IPcam-general.onnx");
        assert!(src.is_file() && lib_dir.is_dir(), "fixtures missing");
        let dir = std::env::temp_dir().join(format!("bop-ort-nocache-{}", uuid::Uuid::new_v4()));
        let fold = dir.join("fold-0").join("model");
        std::fs::create_dir_all(&fold).unwrap();
        let model = fold.join("refine-fold-0.onnx");
        std::fs::copy(&src, &model).unwrap();
        let cache = dir.join("cache");
        let mut rt = Runtimes::new_with(
            &CoreOptions {
                cache_dir: Some(cache.clone()),
                intra_threads: 0,
                openvino_dir: None,
            },
            &OrtOptions {
                onnxruntime_dir: Some(lib_dir),
                ..OrtOptions::default()
            },
        );
        assert!(rt.has_ort(), "{:?}", rt.onnxruntime_error());
        let coreml = spec::parse("ort:coreml").unwrap();
        let req = |no_cache| LoadRequest {
            path: model.clone(),
            requested: "ort:coreml".into(),
            gpu_precision: None,
            no_cache,
        };
        let chw: Vec<f32> = (0..3 * 640 * 640u64)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();
        let mut run = |no_cache| {
            let mut b = rt
                .load(&coreml, &req(no_cache))
                .expect("load on ort:coreml");
            let ms = b.info().compile_ms;
            let out = b.infer(&chw, &[]).expect("infer")[0]
                .as_f32()
                .expect("f32 output")
                .to_vec();
            (out, ms)
        };
        let files = |d: &Path| -> Vec<PathBuf> {
            let mut v: Vec<PathBuf> = walk(d);
            v.sort();
            v
        };
        fn walk(d: &Path) -> Vec<PathBuf> {
            let mut out = Vec::new();
            for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.extend(walk(&p));
                } else {
                    out.push(p);
                }
            }
            out
        }
        let (thrown, ms) = run(true);
        println!("no_cache CoreML load {ms} ms; cache {:?}", files(&cache));
        assert!(
            files(&cache.join("coreml")).is_empty(),
            "{:?}",
            files(&cache)
        );
        assert_eq!(
            files(&fold),
            std::slice::from_ref(&model),
            "nothing beside the model"
        );
        let (cached, ms) = run(false);
        println!("cached CoreML load {ms} ms; cache {:?}", files(&cache));
        assert!(
            !files(&cache.join("coreml")).is_empty(),
            "the ordinary load caches"
        );
        let diff = thrown
            .iter()
            .zip(&cached)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "max |no_cache - cached| over {} outputs: {diff}",
            cached.len()
        );
        assert_eq!(thrown.len(), cached.len());
        assert!(diff < 1e-2, "{diff}");
        drop(rt);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
