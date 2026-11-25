//! The `ort`-crate-free rules of the ONNX Runtime backend ([`super::ort`]): the CUDA / TensorRT
//! dependency checks (what an NVIDIA execution provider needs, the reason naming what is
//! missing, TensorRT's `nvinfer` loaded by its session only), the environment switches
//! (`BOP_CUDNN_CONV_ALGO_SEARCH`, `BOP_ORT_SPINNING`, `BOP_ORT_CPU_ARENA`, `BOP_COREML_REWRITE`),
//! the per-session thread budget and the shape pinning of dynamic inputs (free-dimension
//! overrides, and RT-DETR refused on CoreML while a dimension stays dynamic). Same feature gate
//! as `ort.rs`, which holds every call into the `ort` crate and uses these.

use super::device::DEFAULT_TARGET_SIZES_SHAPE;
use super::libs;
use super::spec::Target;
use crate::model::PortSpec;
use crate::sync::LockExt;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ---- CUDA / TensorRT dependency checks ---------------------------------------------------------

/// Library names tried for each dependency, in order (any one loading is enough).
#[cfg(windows)]
pub(super) mod dep_names {
    pub const CUDART: &[&str] = &["cudart64_12.dll", "cudart64_13.dll"];
    pub const CUDNN: &[&str] = &["cudnn64_9.dll"];
    pub const NVINFER: &[&str] = &["nvinfer_10.dll"];
    pub const CUDA_PROVIDER: &str = "onnxruntime_providers_cuda.dll";
    pub const TRT_PROVIDER: &str = "onnxruntime_providers_tensorrt.dll";
}
#[cfg(not(windows))]
pub(super) mod dep_names {
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
pub(super) fn preload_for_session(target: Target, dirs: &[&Path]) -> Result<()> {
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

pub(super) fn nvidia_deps(target: Target, ort_dir: &Path, cuda_libs: Option<&Path>) -> Vec<Dep> {
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

// ---- environment switches ---------------------------------------------------------------------

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
pub(super) fn conv_search_from_env() -> ConvSearch {
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

/// Intra-op threads for each of `sessions` sessions on `target`. Every session runs one
/// inference at a time on its own thread, so on the CPU `sessions` of them compute at once and
/// must share the budget (`intra_threads`, physical cores minus one) instead of each taking all
/// of it; an accelerator's pool only runs its CPU-fallback ops, so it keeps the full count.
pub(super) fn session_threads(target: Target, intra_threads: usize, sessions: usize) -> usize {
    if target == Target::Cpu && sessions > 1 {
        (intra_threads / sessions).max(1)
    } else {
        intra_threads
    }
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
pub(super) fn log_cpu_arena_off() {
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

// ---- shape pinning ---------------------------------------------------------------------------

/// Input dimensions that stay dynamic after the free-dimension `overrides` (from
/// [`dimension_overrides`]), as "input[axis]". `symbols` is per input, as from `ort::ports`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device::DEFAULT_IMAGE_SHAPE;
    use crate::model::PortElem;

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
    fn cpu_sessions_share_the_thread_budget() {
        assert_eq!(session_threads(Target::Cpu, 5, 1), 5);
        assert_eq!(session_threads(Target::Cpu, 5, 2), 2);
        assert_eq!(session_threads(Target::Cpu, 5, 4), 1);
        assert_eq!(session_threads(Target::Cpu, 1, 4), 1);
        assert_eq!(session_threads(Target::Cpu, 5, 0), 5);
        assert_eq!(session_threads(Target::Cuda, 5, 2), 5);
    }
}
