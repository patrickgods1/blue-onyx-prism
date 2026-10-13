//! Resource catalog (docs/PLAN.md, "Resource catalog"): every file the service can download,
//! pinned by URL, SHA-256 and size. Pure data, no I/O.
//!
//! One table per resource kind:
//! - `OPENVINO`: `openvino-runtime` per `(os, arch)`, provides `openvino:*`.
//! - `ONNXRUNTIME`: `onnxruntime-{cpu,cuda,directml,coreml}` per `(os, arch)`. `cuda` includes
//!   TensorRT's provider library (TensorRT itself is not downloadable), `directml` is two NuGet
//!   packages (ONNX Runtime + `DirectML.dll`), `coreml` is the macOS package (which also serves
//!   `cpu` there).
//! - `CUDA_LIBS`: `nvidia-cuda-libs`, NVIDIA's CUDA 12.8 / cuDNN 9 redistributable wheels from
//!   PyPI (Windows x64, Linux x64). Large and opt-in.
//! - [`MODELS`]: the Hugging Face models of `download.rs` (`.onnx` + `.yaml` for YOLOv5 and
//!   RT-DETRv2; `.onnx` only for D-FINE and RF-DETR), pinned to a repo commit so the hashes cannot
//!   drift.
//! - [`EXPORT_MODELS`]: models exported on this machine by [`super::export`] when the user asks
//!   for it ([`ExportRecipe`]): Ultralytics YOLOv8/9/10/11/12/26 (AGPL-3.0, official `.pt` release
//!   assets), RF-DETR L (Roboflow's safetensors + config) and DEIMv2 (Intellindust's safetensors +
//!   config, exported with the official code, `DEIMV2_SRC_ID`). Never downloaded or exported
//!   automatically.
//! - [`TOOLS`]: `tool:uv`, Astral's uv per platform (it installs Python and the pinned packages of
//!   `scripts/requirements-model-export.txt`), and `tool:deimv2-src`, the DEIMv2 source archive;
//!   both used only by the model export.
//!
//! This is the single source of truth for the pinned versions and URLs: `setup_openvino` and
//! `setup_onnxruntime` read their packages from here.
//!
//! Where the hashes come from: OpenVINO `<archive>.sha256` files on storage.openvinotoolkit.org;
//! the GitHub release API `digest` for ONNX Runtime archives; the PyPI JSON API for the NVIDIA
//! wheels; Hugging Face LFS `oid`s for `.onnx` files; NuGet packages and `.yaml` files were
//! downloaded and hashed (the yaml git blob ids match the Hugging Face tree); the GitHub release
//! API `digest` for the Ultralytics weights (downloaded and hashed), Hugging Face LFS `oid`s for
//! the RF-DETR / DEIMv2 safetensors (downloaded and hashed; their `config.json` files downloaded and
//! hashed), the GitHub archive of the DEIMv2 commit (downloaded and hashed) and the release's
//! `<archive>.sha256` files for uv (archives downloaded and checked against them).

use crate::backend::detect::ComputeCapability;
use crate::backend::spec::{Runtime, Target};
use crate::model::ModelFamilyKind;
use serde::{Serialize, Serializer};
mod models;
mod runtimes;
mod tools;

pub use models::*;
pub(crate) use runtimes::*;
pub use tools::*;

/// Pinned OpenVINO runtime version.
pub const OPENVINO_VERSION: &str = "2026.4.0";
/// Pinned ONNX Runtime version. The `ort` crate is built without `api-NN` features, so any ONNX
/// Runtime 1.17+ library loads; 1.24.4 is the newest release that ships every package we use
/// (the DirectML NuGet package stops at 1.24.x, and 1.25+ renamed the GPU archives).
pub const ONNXRUNTIME_VERSION: &str = "1.24.4";
/// GPU code in a CUDA build: which compute capabilities its kernels run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaArchs {
    /// Real (SASS) architectures: run on GPUs of the same major version and a minor version at
    /// least this one (an `sm_86` cubin runs on 8.6 and 8.9, not on 8.0 or 9.0).
    pub sass: &'static [ComputeCapability],
    /// Virtual (PTX) architectures: JIT-compiled by the driver for any GPU at or above them.
    pub ptx: &'static [ComputeCapability],
}

impl CudaArchs {
    /// Whether a GPU of compute capability `cc` has kernels in this build.
    pub fn runs_on(&self, cc: ComputeCapability) -> bool {
        self.sass
            .iter()
            .any(|a| a.major == cc.major && a.minor <= cc.minor)
            || self.ptx.iter().any(|a| *a <= cc)
    }

    /// Lowest compute capability with kernels.
    pub fn min(&self) -> Option<ComputeCapability> {
        self.sass.iter().chain(self.ptx).min().copied()
    }

    /// "7.5, 8.6, 8.9 (PTX 9.0)".
    pub fn describe(&self) -> String {
        let list = |v: &[ComputeCapability]| {
            v.iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match (self.sass.is_empty(), self.ptx.is_empty()) {
            (_, true) => list(self.sass),
            (true, false) => format!("PTX {}", list(self.ptx)),
            (false, false) => format!("{} (PTX {})", list(self.sass), list(self.ptx)),
        }
    }
}

const fn cc(major: u32, minor: u32) -> ComputeCapability {
    ComputeCapability::new(major, minor)
}

/// GPU code of the pinned ONNX Runtime 1.24.4 CUDA 12 build for Windows x64
/// (`onnxruntime-win-x64-gpu-1.24.4.zip`): `CMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;
/// 90-virtual` (`CudaArchs` for CUDA 12.8 in `tools/ci_build/github/azure-pipelines/
/// cuda-packaging-pipeline.yml` on branch rel-1.24.4). Checked on the shipped
/// `onnxruntime_providers_cuda.dll`: its 200 embedded fatbinaries hold SASS for sm_75, sm_86 and
/// sm_89 and PTX for compute_90 only. So 7.5 is the minimum, and 6.x (Pascal), 7.0/7.2 (Volta)
/// and 8.0 (A100) fail at run time with `cudaErrorNoKernelImageForDevice`.
pub(crate) const ORT_CUDA_ARCHS_WINDOWS: CudaArchs = CudaArchs {
    sass: &[cc(7, 5), cc(8, 6), cc(8, 9)],
    ptx: &[cc(9, 0)],
};
/// GPU code of the pinned ONNX Runtime 1.24.4 CUDA 12 build for Linux x64
/// (`onnxruntime-linux-x64-gpu-1.24.4.tgz`): `CUDA_ARCHS="60-real;70-real;75-real;80-real;
/// 90a-real;90-virtual"` for CUDA 12.8 in `tools/ci_build/github/linux/
/// build_cuda_c_api_package.sh` on branch rel-1.24.4 (`sm_90a` runs on 9.0 only, as `9.0` here).
/// Minimum 6.0.
pub(crate) const ORT_CUDA_ARCHS_LINUX: CudaArchs = CudaArchs {
    sass: &[cc(6, 0), cc(7, 0), cc(7, 5), cc(8, 0), cc(9, 0)],
    ptx: &[cc(9, 0)],
};

/// GPU code of the pinned ONNX Runtime CUDA build for `os` (None: no CUDA package there).
pub(crate) fn ort_cuda_archs(os: &str) -> Option<&'static CudaArchs> {
    match os {
        "windows" => Some(&ORT_CUDA_ARCHS_WINDOWS),
        "linux" => Some(&ORT_CUDA_ARCHS_LINUX),
        _ => None,
    }
}

/// Pinned DirectML redistributable (NuGet `Microsoft.AI.DirectML`).
pub const DIRECTML_VERSION: &str = "1.15.4";
/// Version label of `nvidia-cuda-libs`: CUDA 12.8 Update 1 components (ONNX Runtime 1.24's CUDA
/// 12 build needs CUDA >= 12.8) and cuDNN 9.8.
pub(crate) const CUDA_LIBS_VERSION: &str = "cuda-12.8.1+cudnn-9.8.0.87";

/// Downloads larger than this need `allow_large_downloads` (or a click in the UI).
pub const LARGE_DOWNLOAD_BYTES: u64 = 500 * 1024 * 1024;

/// Resource ids that are not per-flavor or per-model.
pub const OPENVINO_RUNTIME_ID: &str = "openvino-runtime";
pub const CUDA_LIBS_ID: &str = "nvidia-cuda-libs";
/// Install directory of `nvidia-cuda-libs` under the download root.
pub(crate) const CUDA_LIBS_DEST: &str = "onnxruntime/cuda-libs";
/// Model resource ids are `model:<name>` (catalog spelling, matched case-insensitively).
pub(crate) const MODEL_ID_PREFIX: &str = "model:";
/// Id prefix of benchmark image sets: `bench:coco-cctv`.
pub(crate) const BENCH_ID_PREFIX: &str = "bench:";
/// Id prefix of tools: `tool:uv`.
pub(crate) const TOOL_ID_PREFIX: &str = "tool:";
/// The uv tool resource (model export toolchain).
pub const UV_ID: &str = "tool:uv";
/// Install directory of `tool:uv` under the download root.
pub const UV_DEST: &str = "tools/uv";
/// Pinned uv release (2026-09-25).
pub const UV_VERSION: &str = "0.12.19";
/// `(os, arch)` as in `std::env::consts::{OS, ARCH}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Platform {
    pub os: &'static str,
    pub arch: &'static str,
}

impl Platform {
    pub const WINDOWS_X64: Platform = Platform::new("windows", "x86_64");
    pub const LINUX_X64: Platform = Platform::new("linux", "x86_64");
    pub const LINUX_ARM64: Platform = Platform::new("linux", "aarch64");
    pub const MACOS_ARM64: Platform = Platform::new("macos", "aarch64");

    pub const fn new(os: &'static str, arch: &'static str) -> Self {
        Self { os, arch }
    }

    /// The platform this binary was built for.
    pub fn current() -> Self {
        Self::new(std::env::consts::OS, std::env::consts::ARCH)
    }

    pub fn matches(&self, os: &str, arch: &str) -> bool {
        self.os == os && self.arch == arch
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.os, self.arch)
    }
}

/// Platforms release binaries are built for. Each has an OpenVINO runtime and at least the CPU
/// flavor of ONNX Runtime in the catalog (checked by a test).
pub const SHIPPED_PLATFORMS: &[Platform] = &[
    Platform::WINDOWS_X64,
    Platform::LINUX_X64,
    Platform::LINUX_ARM64,
    Platform::MACOS_ARM64,
];

/// Installed ONNX Runtime flavor (one per process; written to `flavor.txt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Flavor {
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
}

impl Flavor {
    pub const ALL: [Flavor; 4] = [Flavor::Cpu, Flavor::Cuda, Flavor::DirectMl, Flavor::CoreMl];

    pub const fn as_str(self) -> &'static str {
        match self {
            Flavor::Cpu => "cpu",
            Flavor::Cuda => "cuda",
            Flavor::DirectMl => "directml",
            Flavor::CoreMl => "coreml",
        }
    }

    /// Parse a flavor name (`flavor.txt`, `--flavor`). `gpu` (the GPU package's name) means
    /// `cuda`.
    pub fn parse(s: &str) -> Option<Flavor> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Flavor::Cpu,
            "cuda" | "gpu" => Flavor::Cuda,
            "directml" => Flavor::DirectMl,
            "coreml" => Flavor::CoreMl,
            _ => return None,
        })
    }

    /// Resource id: `onnxruntime-<flavor>`.
    pub const fn resource_id(self) -> &'static str {
        match self {
            Flavor::Cpu => "onnxruntime-cpu",
            Flavor::Cuda => "onnxruntime-cuda",
            Flavor::DirectMl => "onnxruntime-directml",
            Flavor::CoreMl => "onnxruntime-coreml",
        }
    }

    /// Install directory under the download root: `onnxruntime/<flavor>`.
    pub const fn dest(self) -> &'static str {
        match self {
            Flavor::Cpu => "onnxruntime/cpu",
            Flavor::Cuda => "onnxruntime/cuda",
            Flavor::DirectMl => "onnxruntime/directml",
            Flavor::CoreMl => "onnxruntime/coreml",
        }
    }

    /// Execution providers the flavor's package contains (CPU is in every flavor).
    pub const fn targets(self) -> &'static [Target] {
        match self {
            Flavor::Cpu => &[Target::Cpu],
            Flavor::Cuda => &[Target::Cuda, Target::TensorRt, Target::Cpu],
            Flavor::DirectMl => &[Target::DirectMl, Target::Cpu],
            Flavor::CoreMl => &[Target::CoreMl, Target::Cpu],
        }
    }

    /// The flavor whose package has `target`'s execution provider; None for CPU (every flavor
    /// has it) and for non-ORT targets.
    pub fn providing(target: Target) -> Option<Flavor> {
        match target {
            Target::Cuda | Target::TensorRt => Some(Flavor::Cuda),
            Target::DirectMl => Some(Flavor::DirectMl),
            Target::CoreMl => Some(Flavor::CoreMl),
            _ => None,
        }
    }
}

impl std::fmt::Display for Flavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a resource is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "flavor")]
pub enum ResourceKind {
    OpenVinoRuntime,
    OnnxRuntime(Flavor),
    /// NVIDIA CUDA/cuDNN shared libraries for the CUDA execution provider.
    CudaLibs,
    Model,
    /// A benchmark image set (plain image files, `bench/<set>/`), from `assets/bench/<set>.json`.
    BenchImages,
    /// A model exported on this machine: the parts are the verified weights (`.pt`, or
    /// `model.safetensors` + `config.json`, downloaded into the export toolchain's weights
    /// directory); the model files are produced by [`super::export`].
    ExportModel,
    /// A helper of the model export (`tool:uv`, `tool:deimv2-src`), extracted from an archive.
    Tool,
}

impl ResourceKind {
    /// Plain verified files placed as is (models, benchmark images, export weights), not an
    /// extracted archive.
    pub fn is_plain_files(self) -> bool {
        matches!(
            self,
            ResourceKind::Model | ResourceKind::BenchImages | ResourceKind::ExportModel
        )
    }

    /// Model files: downloaded ([`ResourceKind::Model`]) or exported locally
    /// ([`ResourceKind::ExportModel`]).
    pub fn is_model(self) -> bool {
        matches!(self, ResourceKind::Model | ResourceKind::ExportModel)
    }
}

/// Container format of a downloaded part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ArchiveKind {
    /// `.zip`, `.nupkg` and `.whl` files.
    Zip,
    /// `.tgz` / `.tar.gz`.
    TarGz,
}

/// Which entries of a part are kept, and where they go (whitelists live in the extractors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    /// OpenVINO toolkit archive: `<top>/runtime/...` libraries, layout preserved.
    OpenVino,
    /// ONNX Runtime GitHub release archive: `<top>/lib/<libs>`, copied flat.
    OrtRelease,
    /// `Microsoft.ML.OnnxRuntime.DirectML` NuGet package: `runtimes/win-x64/native/<dlls>`.
    NugetOrt,
    /// `Microsoft.AI.DirectML` NuGet package: `bin/x64-win/DirectML.dll`.
    NugetDirectMl,
    /// NVIDIA PyPI wheel: `nvidia/<component>/{bin,lib}/<libs>`, copied flat.
    NvidiaWheel,
    /// uv release archive: only `<top>/uv` (`uv.exe` on Windows), copied flat.
    Uv,
    /// Python source archive (DEIMv2): `<top>/engine/**/*.py` and the license, paths kept.
    SourceTree,
    /// A plain file saved as [`Part::file_name`].
    File,
}

/// One file to download.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Part {
    /// HTTPS URL, pinned to a version (and, for Hugging Face, a commit).
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: &'static str,
    /// Exact size in bytes.
    pub size: u64,
    /// File name to download to (the final name for [`Layout::File`] parts).
    pub file_name: &'static str,
    /// None for plain files.
    pub archive: Option<ArchiveKind>,
    pub layout: Layout,
}

/// What a resource makes possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provides {
    /// Device options of a runtime (`openvino:gpu`, `ort:cuda`, ...).
    Runtime {
        runtime: Runtime,
        targets: &'static [Target],
    },
    /// Libraries that make `ort:cuda` usable (with the `cuda` flavor).
    CudaLibs,
    /// A model servable under `name`.
    Model {
        name: &'static str,
        family: ModelFamilyKind,
    },
    /// Benchmark images of dataset `set`.
    BenchImages { set: &'static str },
    /// A helper program (`tool:<name>`).
    Tool { name: &'static str },
}

impl Provides {
    /// Human-readable list: `["openvino:gpu", "openvino:cpu"]`, `["model:IPcam-general"]`.
    pub fn labels(&self) -> Vec<String> {
        match self {
            Provides::Runtime { runtime, targets } => targets
                .iter()
                .map(|t| format!("{}:{}", runtime.as_str(), t.as_str()))
                .collect(),
            Provides::CudaLibs => vec!["ort:cuda".to_string(), "ort:tensorrt".to_string()],
            Provides::Model { name, .. } => vec![format!("{MODEL_ID_PREFIX}{name}")],
            Provides::BenchImages { set } => vec![format!("{BENCH_ID_PREFIX}{set}")],
            Provides::Tool { name } => vec![format!("{TOOL_ID_PREFIX}{name}")],
        }
    }

    /// Whether this resource provides `runtime:target`.
    pub fn provides_target(&self, runtime: Runtime, target: Target) -> bool {
        matches!(self, Provides::Runtime { runtime: r, targets } if *r == runtime && targets.contains(&target))
    }
}

impl Serialize for Provides {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.labels())
    }
}

/// One downloadable resource (possibly several files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Resource {
    /// `openvino-runtime`, `onnxruntime-cuda`, `nvidia-cuda-libs`, `model:IPcam-general`. Unique
    /// per platform (models are platform-independent).
    pub id: &'static str,
    pub kind: ResourceKind,
    pub version: &'static str,
    /// None = any platform (models).
    pub platform: Option<Platform>,
    pub parts: &'static [Part],
    /// Install directory relative to the download root (`download_dir`, default the exe dir):
    /// `openvino`, `onnxruntime/<flavor>`, `onnxruntime/cuda-libs`. Models use `models`, but the
    /// resolver places them where the config's model path points.
    pub dest: &'static str,
    pub provides: Provides,
    /// Human-readable name for logs and the UI.
    pub title: &'static str,
    /// One-line description (`list-models`, `list-resources`).
    pub description: &'static str,
    /// License of the downloaded content (shown before opt-in downloads).
    pub license: &'static str,
}

impl Resource {
    /// Total download size in bytes.
    pub fn size(&self) -> u64 {
        self.parts.iter().map(|p| p.size).sum()
    }

    /// Larger than [`LARGE_DOWNLOAD_BYTES`]: needs `allow_large_downloads`.
    pub fn is_large(&self) -> bool {
        is_large(self.size())
    }

    /// Available on `(os, arch)` (always true for platform-independent resources).
    pub fn available_on(&self, os: &str, arch: &str) -> bool {
        self.platform.is_none_or(|p| p.matches(os, arch))
    }

    /// The ONNX Runtime flavor of an `onnxruntime-*` resource.
    pub fn flavor(&self) -> Option<Flavor> {
        match self.kind {
            ResourceKind::OnnxRuntime(f) => Some(f),
            _ => None,
        }
    }

    /// Model name of a model resource.
    pub fn model_name(&self) -> Option<&'static str> {
        match self.provides {
            Provides::Model { name, .. } => Some(name),
            _ => None,
        }
    }

    /// "ONNX Runtime CUDA 1.24.4 (281 MB)".
    pub fn describe(&self) -> String {
        format!(
            "{} {} ({})",
            self.title,
            self.version,
            format_size(self.size())
        )
    }
}

/// Larger than [`LARGE_DOWNLOAD_BYTES`].
pub fn is_large(bytes: u64) -> bool {
    bytes > LARGE_DOWNLOAD_BYTES
}

/// "83 MB", "1.7 GB" (decimal units, as download sizes are usually shown).
pub(crate) fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------------------------

/// Every resource of every platform.
pub fn all() -> impl Iterator<Item = &'static Resource> {
    OPENVINO
        .iter()
        .chain(ONNXRUNTIME)
        .chain(CUDA_LIBS)
        .chain(MODELS)
        .chain(EXPORT_MODELS)
        .chain(TOOLS)
        .chain(DEIMV2_SOURCE)
        .chain(bench_sets())
}

/// A `bench-images` resource for a benchmark set manifest: one plain-file part per image
/// (URL, SHA-256, size and file name from the manifest; images without them are skipped),
/// installed into `bench/<id>/`. Built once per manifest and kept for the process (`'static`
/// like the rest of the catalog).
pub(crate) fn bench_resource(m: &crate::benchmark::images::Manifest) -> Resource {
    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }
    let parts: Vec<Part> = m
        .images
        .iter()
        .filter(|i| crate::benchmark::images::safe_file_name(&i.file))
        .filter_map(|i| {
            Some(Part {
                url: leak(i.url.clone()?),
                sha256: leak(i.sha256.clone()?.to_ascii_lowercase()),
                size: i.size?,
                file_name: leak(i.file.clone()),
                archive: None,
                layout: Layout::File,
            })
        })
        .collect();
    let id = leak(format!("{BENCH_ID_PREFIX}{}", m.id));
    let set = leak(m.id.clone());
    Resource {
        id,
        kind: ResourceKind::BenchImages,
        version: leak(if m.revision.is_empty() {
            "1".to_string()
        } else {
            m.revision.clone()
        }),
        platform: None,
        parts: Box::leak(parts.into_boxed_slice()),
        dest: leak(format!("{}/{}", crate::benchmark::images::BENCH_DIR, m.id)),
        provides: Provides::BenchImages { set },
        title: leak(if m.title.is_empty() {
            format!("Benchmark images {}", m.id)
        } else {
            m.title.clone()
        }),
        description: leak(if m.description.is_empty() {
            format!("{} benchmark images", m.images.len())
        } else {
            m.description.clone()
        }),
        license: leak(m.license.clone()),
    }
}

/// The benchmark image sets embedded in this build (`assets/bench/*.json`).
pub(crate) fn bench_sets() -> &'static [Resource] {
    static SETS: std::sync::OnceLock<Vec<Resource>> = std::sync::OnceLock::new();
    SETS.get_or_init(|| {
        crate::benchmark::images::builtin_manifests()
            .iter()
            .map(bench_resource)
            .collect()
    })
}

/// The `bench:<set>` resource.
pub(crate) fn bench_set(set: &str) -> Option<&'static Resource> {
    bench_sets()
        .iter()
        .find(|r| matches!(r.provides, Provides::BenchImages { set: s } if s.eq_ignore_ascii_case(set.trim())))
}

/// Resources usable on `(os, arch)` (`fetch --all-for-platform`).
pub(crate) fn for_platform<'a>(
    os: &'a str,
    arch: &'a str,
) -> impl Iterator<Item = &'static Resource> + 'a {
    all().filter(move |r| r.available_on(os, arch))
}

/// Resource `id` for `(os, arch)`; model ids match case-insensitively.
pub fn find(id: &str, os: &str, arch: &str) -> Option<&'static Resource> {
    let id = id.trim();
    if let Some(name) = strip_prefix_ci(id, MODEL_ID_PREFIX) {
        return model(name).or_else(|| export_model(name));
    }
    for_platform(os, arch).find(|r| r.id.eq_ignore_ascii_case(id))
}

/// `s` without `prefix`, compared ASCII case-insensitively.
pub(crate) fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    // `get` (not slicing) so a multibyte character straddling the prefix length is a plain
    // mismatch rather than a panic (user-supplied ids reach this).
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &s[prefix.len()..])
}

/// `openvino-runtime` for `(os, arch)`.
pub fn openvino_runtime(os: &str, arch: &str) -> Option<&'static Resource> {
    OPENVINO.iter().find(|r| r.available_on(os, arch))
}

/// `onnxruntime-<flavor>` for `(os, arch)`.
pub fn onnxruntime(os: &str, arch: &str, flavor: Flavor) -> Option<&'static Resource> {
    ONNXRUNTIME
        .iter()
        .find(|r| r.available_on(os, arch) && r.flavor() == Some(flavor))
}

/// ONNX Runtime flavors downloadable for `(os, arch)`.
pub fn ort_flavors(os: &str, arch: &str) -> Vec<Flavor> {
    Flavor::ALL
        .into_iter()
        .filter(|f| onnxruntime(os, arch, *f).is_some())
        .collect()
}

/// `nvidia-cuda-libs` for `(os, arch)`.
pub fn cuda_libs_for(os: &str, arch: &str) -> Option<&'static Resource> {
    CUDA_LIBS.iter().find(|r| r.available_on(os, arch))
}

/// Model resource by name (case-insensitive; a trailing `.onnx` is accepted), like
/// `download::find`.
pub fn model(name: &str) -> Option<&'static Resource> {
    let n = name.trim();
    let n = n
        .len()
        .checked_sub(5)
        .filter(|&i| n.is_char_boundary(i) && n[i..].eq_ignore_ascii_case(".onnx"))
        .map_or(n, |i| &n[..i]);
    MODELS
        .iter()
        .find(|r| r.model_name().is_some_and(|m| m.eq_ignore_ascii_case(n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_prefix_ci_is_char_boundary_safe() {
        assert_eq!(strip_prefix_ci("MODEL:x", "model:"), Some("x"));
        assert_eq!(strip_prefix_ci("abcdeé", "abcdef"), None);
        assert_eq!(strip_prefix_ci("abcdé", "abcdef"), None);
        assert_eq!(strip_prefix_ci("é", "ab"), None);
        assert_eq!(strip_prefix_ci("ab", "abc"), None);
    }

    #[test]
    fn ort_cuda_compute_capabilities() {
        let win = ort_cuda_archs("windows").unwrap();
        let linux = ort_cuda_archs("linux").unwrap();
        assert!(ort_cuda_archs("macos").is_none());
        assert_eq!(win.min(), Some(cc(7, 5)));
        assert_eq!(linux.min(), Some(cc(6, 0)));
        let runs = |a: &CudaArchs, v: &[(u32, u32)]| -> Vec<bool> {
            v.iter().map(|&(m, n)| a.runs_on(cc(m, n))).collect()
        };
        let probe = [
            (5, 2),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 2),
            (7, 5),
            (8, 0),
            (8, 6),
            (8, 7),
            (8, 9),
            (9, 0),
            (10, 0),
            (12, 0),
        ];
        assert_eq!(
            runs(win, &probe),
            [
                false, false, false, false, false, true, false, true, true, true, true, true, true
            ]
        );
        assert_eq!(
            runs(linux, &probe),
            [
                false, true, true, true, true, true, true, true, true, true, true, true, true
            ]
        );
        assert_eq!(win.describe(), "7.5, 8.6, 8.9 (PTX 9.0)");
    }

    #[test]
    fn wheel_names_and_model_lookup() {
        assert_eq!(
            CUDA_LIBS[0].parts[0].file_name,
            "nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl"
        );
        assert_eq!(
            model("ipcam-general.ONNX").unwrap().id,
            "model:IPcam-general"
        );
        assert_eq!(model("RT-DETRV2-S").unwrap().id, "model:rt-detrv2-s");
        let rf = model("rfdetr-base.onnx").unwrap();
        assert_eq!(rf.parts.len(), 1);
        assert_eq!(rf.parts[0].file_name, "rfdetr-base.onnx");
        assert_eq!(
            rf.parts[0].url,
            "https://huggingface.co/onnx-community/rfdetr_base-ONNX/resolve/\
             7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f/onnx/model.onnx"
        );
        assert_eq!(rf.version, "7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f");
        assert!(model("yolo26s").is_none());
        assert_eq!(
            find("MODEL:ipcam-dark", "linux", "x86_64").unwrap().id,
            "model:IPcam-dark"
        );
        assert_eq!(
            find("onnxruntime-cuda", "windows", "x86_64").unwrap().parts[0].file_name,
            "onnxruntime-win-x64-gpu-1.24.4.zip"
        );
        assert!(find("onnxruntime-cuda", "macos", "aarch64").is_none());
    }

    #[test]
    fn detr_models() {
        let detr: Vec<_> = MODELS
            .iter()
            .filter_map(|r| match r.provides {
                Provides::Model { name, family }
                    if matches!(family, ModelFamilyKind::Detr | ModelFamilyKind::RfDetr) =>
                {
                    Some((r, name, family))
                }
                _ => None,
            })
            .collect();
        assert_eq!(detr.len(), 8);
        for (r, name, family) in detr {
            assert_eq!(r.license, "Apache-2.0");
            assert_eq!(r.parts.len(), 1, "{name}: no class file part");
            let p = &r.parts[0];
            assert_eq!(p.file_name, format!("{name}.onnx"));
            assert!(p.url.starts_with("https://huggingface.co/onnx-community/"));
            assert!(
                p.url
                    .ends_with(&format!("/resolve/{}/onnx/model.onnx", r.version))
            );
            assert_eq!(r.version.len(), 40);
            assert_eq!(p.sha256.len(), 64);
            assert!(p.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            let want = if name.starts_with("rfdetr") {
                ModelFamilyKind::RfDetr
            } else {
                ModelFamilyKind::Detr
            };
            assert_eq!(family, want, "{name}");
            assert!(model_input_size(name).is_some(), "{name} has no input size");
        }
        assert_eq!(model_input_size("RFDETR-NANO.onnx"), Some(384));
        assert_eq!(model_input_size("dfine-s"), Some(640));
        assert_eq!(model_input_size("IPcam-general.onnx"), None);
        assert_eq!(model_input_size("unknown"), None);
        // Every size entry names a catalog model (downloaded or exported).
        for (n, _) in MODEL_INPUT_SIZES {
            assert!(model(n).or_else(|| export_model(n)).is_some(), "{n}");
        }
        // RF-DETR L is exported now (704 px); the old onnx-community v1 download is gone.
        assert!(model("rfdetr-large").is_none());
        assert_eq!(model_input_size("rfdetr-large.onnx"), Some(704));
        assert_eq!(model_input_size("DEIMV2-ATTO"), Some(320));
        assert_eq!(model_input_size("deimv2-femto.onnx"), Some(416));
        assert_eq!(model_input_size("deimv2-n"), None);
    }

    #[test]
    fn export_models() {
        use std::collections::HashSet;
        let mut names = HashSet::new();
        for r in EXPORT_MODELS {
            let name = r.model_name().unwrap();
            assert!(names.insert(name), "duplicate {name}");
            assert_eq!(r.id, format!("model:{name}"));
            assert_eq!(r.kind, ResourceKind::ExportModel);
            assert!(r.platform.is_none());
            let recipe = export_recipe(r).unwrap_or_else(|| panic!("{name} has no recipe"));
            let Provides::Model { family, .. } = r.provides else {
                panic!("{name}")
            };
            for p in r.parts {
                assert_eq!(p.sha256.len(), 64, "{name}");
                assert!(p.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(p.url.starts_with("https://"), "{name}");
                assert!(p.url.ends_with(p.file_name), "{name}");
                assert_eq!(p.layout, Layout::File);
                assert!(p.size > 0);
            }
            assert!(!r.is_large(), "{name}");
            match recipe {
                ExportRecipe::Ultralytics => {
                    assert_eq!(r.parts.len(), 1);
                    assert_eq!(r.parts[0].file_name, format!("{name}.pt"));
                    assert_eq!(r.version, ULTRALYTICS_EXPORT_VERSION);
                    assert_eq!(r.license, ULTRALYTICS_LICENSE);
                    // End-to-end heads (v10, 26) vs raw heads with NMS.
                    let e2e = name.starts_with("yolov10") || name.starts_with("yolo26");
                    let want = if e2e {
                        ModelFamilyKind::Yolo26
                    } else {
                        ModelFamilyKind::Yolo8
                    };
                    assert_eq!(family, want, "{name}");
                    assert!(r.parts[0].url.contains(ULTRALYTICS_ASSETS_RELEASE));
                    assert_eq!(export_input_size(r), 640);
                }
                ExportRecipe::RfDetr => {
                    assert_eq!(family, ModelFamilyKind::RfDetr);
                    assert!(r.version.contains(RFDETR_LARGE_REVISION));
                    assert!(r.version.contains(TRANSFORMERS_VERSION));
                    assert_eq!(export_input_size(r), 704);
                }
                ExportRecipe::DeimV2 => {
                    assert_eq!(family, ModelFamilyKind::RtDetr);
                    assert!(
                        r.parts
                            .iter()
                            .any(|p| p.url.contains("huggingface.co/Intellindust/DEIMv2_"))
                    );
                    assert_eq!(export_extras(r), [deimv2_source().unwrap()]);
                }
            }
            let files: Vec<_> = r.parts.iter().map(|p| p.file_name).collect();
            if recipe != ExportRecipe::Ultralytics {
                assert_eq!(files, ["model.safetensors", "config.json"], "{name}");
                if recipe == ExportRecipe::DeimV2 {
                    assert!(r.license.starts_with("Non-commercial"), "{name}");
                } else {
                    assert!(r.license.starts_with("Apache-2.0"), "{name}");
                }
            }
        }
        // Every size the request asked for.
        for n in [
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolov9t",
            "yolov9s",
            "yolov9m",
            "yolov9c",
            "yolov9e",
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolo12n",
            "yolo12s",
            "yolo12m",
            "yolo12l",
            "yolo12x",
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
            "rfdetr-large",
            "deimv2-atto",
            "deimv2-femto",
            "deimv2-pico",
            "deimv2-n",
            "deimv2-s",
            "deimv2-m",
            "deimv2-l",
            "deimv2-x",
        ] {
            assert!(names.contains(n), "{n}");
        }
        assert_eq!(names.len(), 40);
        assert_eq!(export_model("YOLOV8N.onnx").unwrap().id, "model:yolov8n");
        assert_eq!(
            find("model:deimv2-s", "linux", "x86_64").unwrap().id,
            "model:deimv2-s"
        );
        assert_eq!(
            find("model:rfdetr-large", "linux", "x86_64").unwrap().kind,
            ResourceKind::ExportModel
        );
        let src = find(DEIMV2_SRC_ID, "windows", "x86_64").unwrap();
        assert_eq!(src.parts[0].layout, Layout::SourceTree);
        assert!(src.parts[0].url.contains(DEIMV2_COMMIT));
        assert_eq!(src.version, DEIMV2_COMMIT);
        assert!(needs_f32("DEIMV2-L.onnx") && needs_f32("rfdetr-large"));
        assert!(!needs_f32("deimv2-s.onnx") && !needs_f32("rfdetr-base.onnx"));
        for m in F32_MODELS {
            assert!(export_model(m).is_some(), "{m}");
        }
        // YOLO26 export version unchanged (earlier exports stay valid).
        assert_eq!(
            ULTRALYTICS_EXPORT_VERSION,
            "ultralytics-8.4.163+assets-v8.4.0"
        );
    }

    #[test]
    fn sizes_and_large_flag() {
        assert_eq!(format_size(29465778), "29 MB");
        assert_eq!(format_size(1_800_000_000), "1.8 GB");
        assert_eq!(format_size(52), "52 B");
        assert!(!is_large(LARGE_DOWNLOAD_BYTES));
        assert!(is_large(LARGE_DOWNLOAD_BYTES + 1));
        for r in CUDA_LIBS {
            assert!(r.is_large(), "{} {}", r.id, r.size());
        }
        for r in OPENVINO.iter().chain(ONNXRUNTIME).chain(MODELS) {
            assert!(!r.is_large(), "{} {:?}", r.id, r.platform);
        }
    }

    #[test]
    fn provides_labels() {
        let ov = openvino_runtime("windows", "x86_64").unwrap();
        assert_eq!(ov.provides.labels(), ["openvino:gpu", "openvino:cpu"]);
        assert!(ov.provides.provides_target(Runtime::OpenVino, Target::Gpu));
        let mac = openvino_runtime("macos", "aarch64").unwrap();
        assert!(!mac.provides.provides_target(Runtime::OpenVino, Target::Gpu));
        let json =
            serde_json::to_value(onnxruntime("macos", "aarch64", Flavor::CoreMl).unwrap()).unwrap();
        assert_eq!(
            json["provides"],
            serde_json::json!(["ort:coreml", "ort:cpu"])
        );
        assert_eq!(
            json["kind"],
            serde_json::json!({"type": "onnx-runtime", "flavor": "coreml"})
        );
        for f in Flavor::ALL {
            assert_eq!(Flavor::parse(f.as_str()), Some(f));
        }
        assert_eq!(Flavor::parse(" GPU "), Some(Flavor::Cuda));
    }
}
