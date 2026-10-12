//! Resource catalog (docs/PLAN.md, "Resource catalog"): every file the service can download,
//! pinned by URL, SHA-256 and size. Pure data, no I/O.
//!
//! One table per resource kind:
//! - [`OPENVINO`]: `openvino-runtime` per `(os, arch)`, provides `openvino:*`.
//! - [`ONNXRUNTIME`]: `onnxruntime-{cpu,cuda,directml,coreml}` per `(os, arch)`. `cuda` includes
//!   TensorRT's provider library (TensorRT itself is not downloadable), `directml` is two NuGet
//!   packages (ONNX Runtime + `DirectML.dll`), `coreml` is the macOS package (which also serves
//!   `cpu` there).
//! - [`CUDA_LIBS`]: `nvidia-cuda-libs`, NVIDIA's CUDA 12.8 / cuDNN 9 redistributable wheels from
//!   PyPI (Windows x64, Linux x64). Large and opt-in.
//! - [`MODELS`]: the Hugging Face models of `download.rs` (`.onnx` + `.yaml` for YOLOv5 and
//!   RT-DETRv2; `.onnx` only for D-FINE and RF-DETR), pinned to a repo commit so the hashes cannot
//!   drift.
//! - [`EXPORT_MODELS`]: models exported on this machine by [`super::export`] when the user asks
//!   for it ([`ExportRecipe`]): Ultralytics YOLOv8/9/10/11/12/26 (AGPL-3.0, official `.pt` release
//!   assets), RF-DETR L (Roboflow's safetensors + config) and DEIMv2 (Intellindust's safetensors +
//!   config, exported with the official code, [`DEIMV2_SRC_ID`]). Never downloaded or exported
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

/// Pinned OpenVINO runtime version.
pub const OPENVINO_VERSION: &str = "2026.4.0";
/// Pinned ONNX Runtime version. The `ort` crate is built without `api-NN` features, so any ONNX
/// Runtime 1.17+ library loads; 1.24.4 is the newest release that ships every package we use
/// (the DirectML NuGet package stops at 1.24.x, and 1.25+ renamed the GPU archives).
pub const ONNXRUNTIME_VERSION: &str = "1.24.4";
/// GPU code in a CUDA build: which compute capabilities its kernels run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaArchs {
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
pub const ORT_CUDA_ARCHS_WINDOWS: CudaArchs = CudaArchs {
    sass: &[cc(7, 5), cc(8, 6), cc(8, 9)],
    ptx: &[cc(9, 0)],
};
/// GPU code of the pinned ONNX Runtime 1.24.4 CUDA 12 build for Linux x64
/// (`onnxruntime-linux-x64-gpu-1.24.4.tgz`): `CUDA_ARCHS="60-real;70-real;75-real;80-real;
/// 90a-real;90-virtual"` for CUDA 12.8 in `tools/ci_build/github/linux/
/// build_cuda_c_api_package.sh` on branch rel-1.24.4 (`sm_90a` runs on 9.0 only, as `9.0` here).
/// Minimum 6.0.
pub const ORT_CUDA_ARCHS_LINUX: CudaArchs = CudaArchs {
    sass: &[cc(6, 0), cc(7, 0), cc(7, 5), cc(8, 0), cc(9, 0)],
    ptx: &[cc(9, 0)],
};

/// GPU code of the pinned ONNX Runtime CUDA build for `os` (None: no CUDA package there).
pub fn ort_cuda_archs(os: &str) -> Option<&'static CudaArchs> {
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
pub const CUDA_LIBS_VERSION: &str = "cuda-12.8.1+cudnn-9.8.0.87";

/// Downloads larger than this need `allow_large_downloads` (or a click in the UI).
pub const LARGE_DOWNLOAD_BYTES: u64 = 500 * 1024 * 1024;

/// Resource ids that are not per-flavor or per-model.
pub const OPENVINO_RUNTIME_ID: &str = "openvino-runtime";
pub const CUDA_LIBS_ID: &str = "nvidia-cuda-libs";
/// Install directory of `nvidia-cuda-libs` under the download root.
pub const CUDA_LIBS_DEST: &str = "onnxruntime/cuda-libs";
/// Model resource ids are `model:<name>` (catalog spelling, matched case-insensitively).
pub const MODEL_ID_PREFIX: &str = "model:";
/// Id prefix of benchmark image sets: `bench:coco-cctv`.
pub const BENCH_ID_PREFIX: &str = "bench:";
/// Id prefix of tools: `tool:uv`.
pub const TOOL_ID_PREFIX: &str = "tool:";
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
pub fn format_size(bytes: u64) -> String {
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
// OpenVINO runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ov_url {
    ($dir:literal, $file:literal) => {
        concat!(
            "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/",
            $dir,
            "/",
            $file
        )
    };
}

macro_rules! ov_part {
    ($dir:literal, $file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: ov_url!($dir, $file),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OpenVino,
        }
    };
}

const OV_GPU_CPU: &[Target] = &[Target::Gpu, Target::Cpu];
const OV_CPU: &[Target] = &[Target::Cpu];

const fn openvino(
    platform: Platform,
    part: &'static [Part],
    targets: &'static [Target],
) -> Resource {
    Resource {
        id: OPENVINO_RUNTIME_ID,
        kind: ResourceKind::OpenVinoRuntime,
        version: OPENVINO_VERSION,
        platform: Some(platform),
        parts: part,
        dest: "openvino",
        provides: Provides::Runtime {
            runtime: Runtime::OpenVino,
            targets,
        },
        title: "OpenVINO runtime",
        description: "Intel OpenVINO runtime libraries (openvino:* devices)",
        license: "Apache-2.0",
    }
}

/// `openvino-runtime` per platform. The Intel GPU plugin ships only for x86_64.
pub static OPENVINO: &[Resource] = &[
    openvino(
        Platform::WINDOWS_X64,
        &[ov_part!(
            "windows_vc_mt",
            "openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64.zip",
            ArchiveKind::Zip,
            "82946205ac9f9f8113acf5f7fb5c705ef06a666aa15a8acf52cf2b4e1be90063",
            235127912
        )],
        OV_GPU_CPU,
    ),
    openvino(
        Platform::LINUX_X64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_ubuntu24_2026.4.0.22959.99c81491cc3_x86_64.tgz",
            ArchiveKind::TarGz,
            "0bd86d578beb1e8805655f593315c69bf909008e212f801ed76e3a350927736f",
            120557925
        )],
        OV_GPU_CPU,
    ),
    openvino(
        Platform::LINUX_ARM64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_ubuntu22_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "ed3f103ae8dd7a509c23645f0d34983378d3e609711a82d65301ac77d4a34800",
            43359520
        )],
        OV_CPU,
    ),
    openvino(
        Platform::MACOS_ARM64,
        &[ov_part!(
            "macos",
            "openvino_toolkit_macos_12_6_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "1e67db1fa25e8d736f35f5b3d663d2fd669d51c0d2892f682d0facc9c22b019c",
            44413951
        )],
        OV_CPU,
    ),
];

// ---------------------------------------------------------------------------------------------
// ONNX Runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ort_release {
    ($file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OrtRelease,
        }
    };
}

macro_rules! nuget {
    ($id:literal, $ver:literal, $layout:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://api.nuget.org/v3-flatcontainer/",
                $id,
                "/",
                $ver,
                "/",
                $id,
                ".",
                $ver,
                ".nupkg"
            ),
            sha256: $sha,
            size: $size,
            file_name: concat!($id, ".", $ver, ".nupkg"),
            archive: Some(ArchiveKind::Zip),
            layout: $layout,
        }
    };
}

const fn ort(
    platform: Platform,
    flavor: Flavor,
    parts: &'static [Part],
    title: &'static str,
) -> Resource {
    Resource {
        id: flavor.resource_id(),
        kind: ResourceKind::OnnxRuntime(flavor),
        version: ONNXRUNTIME_VERSION,
        platform: Some(platform),
        parts,
        dest: flavor.dest(),
        provides: Provides::Runtime {
            runtime: Runtime::Ort,
            targets: flavor.targets(),
        },
        title,
        description: "ONNX Runtime shared libraries (ort:* devices); one flavor is active per process",
        license: "MIT",
    }
}

const ORT_OSX_ARM64: Part = ort_release!(
    "onnxruntime-osx-arm64-1.24.4.tgz",
    ArchiveKind::TarGz,
    "93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989",
    30937282
);

/// `onnxruntime-<flavor>` per platform (the combinations `setup-onnxruntime` supports).
pub static ONNXRUNTIME: &[Resource] = &[
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-win-x64-1.24.4.zip",
            ArchiveKind::Zip,
            "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357",
            74442783
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cuda,
        &[ort_release!(
            "onnxruntime-win-x64-gpu-1.24.4.zip",
            ArchiveKind::Zip,
            "ef3337a0b8184eb8beec310f7c83bd50376b3eefc43aab84ac8e452f6987df0a",
            280958859
        )],
        "ONNX Runtime CUDA/TensorRT",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::DirectMl,
        &[
            nuget!(
                "microsoft.ml.onnxruntime.directml",
                "1.24.4",
                Layout::NugetOrt,
                "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
                12458649
            ),
            nuget!(
                "microsoft.ai.directml",
                "1.15.4",
                Layout::NugetDirectMl,
                "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
                202292617
            ),
        ],
        "ONNX Runtime DirectML",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-linux-x64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
            8155822
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cuda,
        &[ort_release!(
            "onnxruntime-linux-x64-gpu-1.24.4.tgz",
            ArchiveKind::TarGz,
            "c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808",
            205429115
        )],
        "ONNX Runtime CUDA/TensorRT",
    ),
    ort(
        Platform::LINUX_ARM64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-linux-aarch64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e",
            7181958
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::MACOS_ARM64,
        Flavor::CoreMl,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CoreML",
    ),
    // The macOS package has both EPs; `cpu` is the same archive.
    ort(
        Platform::MACOS_ARM64,
        Flavor::Cpu,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CPU",
    ),
];

// ---------------------------------------------------------------------------------------------
// NVIDIA CUDA / cuDNN libraries (PyPI wheels)
// ---------------------------------------------------------------------------------------------

macro_rules! wheel {
    ($url:literal, $sha:literal, $size:literal) => {
        Part {
            url: $url,
            sha256: $sha,
            size: $size,
            file_name: wheel_file_name($url),
            archive: Some(ArchiveKind::Zip),
            layout: Layout::NvidiaWheel,
        }
    };
}

/// Last path segment of a URL (const, for the wheel table).
const fn wheel_file_name(url: &'static str) -> &'static str {
    let bytes = url.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] != b'/' {
        i -= 1;
    }
    let (_, tail) = bytes.split_at(i);
    match std::str::from_utf8(tail) {
        Ok(s) => s,
        Err(_) => panic!("URL is not UTF-8"),
    }
}

const fn cuda_libs(platform: Platform, parts: &'static [Part]) -> Resource {
    Resource {
        id: CUDA_LIBS_ID,
        kind: ResourceKind::CudaLibs,
        version: CUDA_LIBS_VERSION,
        platform: Some(platform),
        parts,
        dest: CUDA_LIBS_DEST,
        provides: Provides::CudaLibs,
        title: "NVIDIA CUDA 12 + cuDNN 9 libraries",
        description: "CUDA runtime, cuBLAS, cuDNN, cuFFT, cuRAND for ort:cuda (large, opt-in)",
        license: "NVIDIA Software License Agreement (CUDA, cuDNN redistributables)",
    }
}

/// `nvidia-cuda-libs`: what ONNX Runtime's CUDA provider links (cudart, cuBLAS/cuBLASLt, cuDNN 9,
/// cuFFT, cuRAND; checked against `libonnxruntime_providers_cuda.so` 1.24.4), plus nvJitLink
/// (cuFFT's dependency) and NVRTC (cuDNN's runtime-compiled engines). The driver's `libcuda`
/// / `nvcuda.dll` is not included.
pub static CUDA_LIBS: &[Resource] = &[
    cuda_libs(
        Platform::WINDOWS_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/30/a5/a515b7600ad361ea14bfa13fb4d6687abf500adc270f19e89849c0590492/nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl",
                "c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8",
                944318
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/70/61/7d7b3c70186fb651d0fbd35b01dbfc8e755f69fd58f817f3d0f642df20c3/nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl",
                "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af",
                567544208
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/39/6a/5e9910b2b2c9dcddee9aaef372b3db0f08b9f7eaf1d462f859461a79caf9/nvidia_cudnn_cu12-9.8.0.87-py3-none-win_amd64.whl",
                "b4b5cfddc32aa4180f9d390ee99e9a9f55a89e7087329b41aba4319327e22466",
                684630375
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/7d/ec/ce1629f1e478bb5ccd208986b5f9e0316a78538dd6ab1d0484f012f8e2a1/nvidia_cufft_cu12-11.3.3.83-py3-none-win_amd64.whl",
                "7a64a98ef2a7c47f905aaf8931b69a3a43f27c55530c698bb2ed7c75c0b42cb7",
                192216559
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/b9/75/70c05b2f3ed5be3bb30b7102b6eb78e100da4bbf6944fd6725c012831cab/nvidia_curand_cu12-10.3.9.90-py3-none-win_amd64.whl",
                "f149a8ca457277da854f89cf282d6ef43176861926c7ac85b2a0fbd237c587ec",
                62765309
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/ed/d7/34f02dad2e30c31b10a51f6b04e025e5dd60e5f936af9045a9b858a05383/nvidia_nvjitlink_cu12-12.8.93-py3-none-win_amd64.whl",
                "bd93fbeeee850917903583587f4fc3a4eafa022e34572251368238ab5e6bd67f",
                268553710
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/45/51/52a3d84baa2136cc8df15500ad731d74d3a1114d4c123e043cb608d4a32b/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl",
                "7a4b6b2904850fe78e0bd179c4b655c404d4bb799ef03ddc60804247099ae909",
                73586838
            ),
        ],
    ),
    cuda_libs(
        Platform::LINUX_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90",
                954765
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
                "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142",
                594346921
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/77/f0/8236c886a061d203e51247aec2b8e3a8f5350178251ab57237daf2140680/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_x86_64.whl",
                "d6b02cd0e3e24aa31d0193a8c39fec239354360d7d81055edddb69f35d53a4c8",
                697999707
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74",
                193118695
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
                "b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9",
                63619976
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88",
                39254836
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994",
                88040029
            ),
        ],
    ),
];

// ---------------------------------------------------------------------------------------------
// Models (Hugging Face, pinned commits)
// ---------------------------------------------------------------------------------------------

/// `xnorpx/rt-detr2-onnx` commit the RT-DETRv2 hashes belong to.
pub const RTDETR_REVISION: &str = "c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795";
/// `xnorpx/blue-onyx-yolo5` commit the YOLOv5 hashes belong to.
pub const YOLO5_REVISION: &str = "c0c3ff5178f5692236e4c1552d6a040135204f98";

macro_rules! hf_file {
    ($repo_rev:literal, $name:literal, $ext:literal, $sha:expr, $size:expr) => {
        Part {
            url: concat!("https://huggingface.co/", $repo_rev, "/", $name, $ext),
            sha256: $sha,
            size: $size,
            file_name: concat!($name, $ext),
            archive: None,
            layout: Layout::File,
        }
    };
}

/// Every RT-DETRv2 model uses the same COCO class list.
const RTDETR_YAML_SHA: &str = "ed029ee87019fa769269fcff15eda24a15c17dcea42b70f4dd0e3ab744503e7d";
const RTDETR_YAML_SIZE: u64 = 1193;

macro_rules! rtdetr {
    ($name:literal, $desc:literal, $onnx_sha:literal, $onnx_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: RTDETR_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".yaml",
                    RTDETR_YAML_SHA,
                    RTDETR_YAML_SIZE
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::RtDetr,
            },
            title: concat!("Model ", $name),
            description: $desc,
            license: "Apache-2.0",
        }
    };
}

/// A Hugging Face `onnx-community/*-ONNX` export (D-FINE, RF-DETR): the fp32 `onnx/model.onnx`
/// of `$repo` at commit `$rev`, saved as `models/<name>.onnx`. No class file: both use the COCO-80
/// fallback (`family::detr` maps RF-DETR's COCO category ids).
macro_rules! hf_detr {
    ($family:expr, $name:literal, $repo:literal, $rev:literal, $title:literal, $desc:literal, $sha:literal, $size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: $rev,
            platform: None,
            parts: &[Part {
                url: concat!(
                    "https://huggingface.co/",
                    $repo,
                    "/resolve/",
                    $rev,
                    "/onnx/model.onnx"
                ),
                sha256: $sha,
                size: $size,
                file_name: concat!($name, ".onnx"),
                archive: None,
                layout: Layout::File,
            }],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: $family,
            },
            title: $title,
            description: $desc,
            license: "Apache-2.0",
        }
    };
}

/// D-FINE (`onnx-community/dfine_*-ONNX`, 640x640 input), family `detr`. D-FINE nano
/// (`dfine_n_coco-ONNX`) is left out: its export finds almost nothing on real photos (best score
/// 0.65 for an obvious bus, no people) on both OpenVINO and ONNX Runtime.
macro_rules! dfine {
    ($($args:tt)*) => {
        hf_detr!(ModelFamilyKind::Detr, $($args)*)
    };
}

/// RF-DETR (`onnx-community/rfdetr_*-ONNX`), family `rfdetr` (ImageNet normalization). Input
/// sizes are in [`MODEL_INPUT_SIZES`].
macro_rules! rfdetr {
    ($($args:tt)*) => {
        hf_detr!(ModelFamilyKind::RfDetr, $($args)*)
    };
}

/// Square input size of catalog models whose ONNX input is dynamic (from each Hugging Face
/// repo's `preprocessor_config.json`), and of exported models that do not use 640 (the export's
/// `--imgsz`; DEIMv2 sizes from each `config.json` `eval_spatial_size`). Models not listed use 640.
pub const MODEL_INPUT_SIZES: &[(&str, u32)] = &[
    ("dfine-s", 640),
    ("dfine-m", 640),
    ("dfine-l", 640),
    ("dfine-x", 640),
    ("rfdetr-nano", 384),
    ("rfdetr-small", 512),
    ("rfdetr-base", 560),
    ("rfdetr-medium", 576),
    ("rfdetr-large", 704),
    ("deimv2-atto", 320),
    ("deimv2-femto", 416),
];

/// Models whose activations overflow half precision: OpenVINO's f16 inference (the GPU default,
/// and the CPU default on ARM) gives NaN (DEIMv2 L/X, DINOv3 ViT-S backbones) or garbage (RF-DETR
/// L) where f32 matches PyTorch. The OpenVINO backend compiles them with
/// `INFERENCE_PRECISION_HINT=f32` (a model's explicit `gpu_precision` still wins on GPU).
pub const F32_MODELS: &[&str] = &["rfdetr-large", "deimv2-l", "deimv2-x"];

/// The model at file name / name `name` (`deimv2-l.onnx`) needs f32 inference ([`F32_MODELS`]).
pub fn needs_f32(name: &str) -> bool {
    let n = name.trim();
    let stem = std::path::Path::new(n)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(n);
    F32_MODELS.iter().any(|m| m.eq_ignore_ascii_case(stem))
}

/// Input size for a catalog model (downloaded or exported), by name or file name (`rfdetr-base`,
/// `rfdetr-base.onnx`, `deimv2-atto.onnx`).
pub fn model_input_size(name: &str) -> Option<u32> {
    let name = model(name).or_else(|| export_model(name))?.model_name()?;
    MODEL_INPUT_SIZES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|&(_, s)| s)
}

macro_rules! yolo5 {
    ($name:literal, $desc:literal, $onnx_sha:literal, $onnx_size:literal, $yaml_sha:literal, $yaml_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: YOLO5_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".yaml",
                    $yaml_sha,
                    $yaml_size
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::Yolo5,
            },
            title: concat!("Model ", $name),
            description: $desc,
            license: "AGPL-3.0",
        }
    };
}

/// Downloadable models (the `download.rs` catalog). Models published only as PyTorch weights
/// (Ultralytics YOLO, RF-DETR L, DEIMv2) are exported locally on request ([`EXPORT_MODELS`]).
pub static MODELS: &[Resource] = &[
    rtdetr!(
        "rt-detrv2-s",
        "RT-DETRv2 small, general COCO",
        "c849b46a43925d0c24d88c7cab096fc8158ade265b7be1ae433e3effe5758a6a",
        80530290
    ),
    rtdetr!(
        "rt-detrv2-ms",
        "RT-DETRv2 medium-small, general COCO",
        "c3315cec9f1ce83956ae1600b6224ead49bf1e5f347fb6a9a2da9740beb50a64",
        125541681
    ),
    rtdetr!(
        "rt-detrv2-m",
        "RT-DETRv2 medium, general COCO",
        "a3a7bc94da99678f788f35aeb6078e1045f4f038b266316e769c09fb51abc62d",
        132661010
    ),
    rtdetr!(
        "rt-detrv2-l",
        "RT-DETRv2 large, general COCO",
        "4c580c4f579a0372c6bea171b0ff3ccea707146a34dc5cea1b817e3c0f4fc0ba",
        169218591
    ),
    rtdetr!(
        "rt-detrv2-x",
        "RT-DETRv2 extra large, general COCO",
        "810a18839401187bd94004bd722d602158302e3f35400840e7cbcfacb71d61c2",
        300394655
    ),
    dfine!(
        "dfine-s",
        "onnx-community/dfine_s_obj2coco-ONNX",
        "f69c4ca98cba7ca58aa15b3d4600867808fecf1b",
        "D-FINE small",
        "D-FINE small (Objects365+COCO), COCO 80 classes; fast, good default for CPU/iGPU",
        "b9e2e76610053aeeac3b2f1f685d8f9a1182a93a338f624b6c8cb7fb390cb532",
        41535197
    ),
    dfine!(
        "dfine-m",
        "onnx-community/dfine_m_obj2coco-ONNX",
        "a2e1133c83887da7d5a000a025594b9d69fef5c2",
        "D-FINE medium",
        "D-FINE medium (Objects365+COCO), COCO 80 classes; balanced accuracy and speed",
        "347f2faba93248c2e7500c9e604317fb391706c58a04802dd908573376dc1323",
        78624257
    ),
    dfine!(
        "dfine-l",
        "onnx-community/dfine_l_obj2coco_e25-ONNX",
        "60e6123fe13da26a6e1dafc9d608ddeb5c393dcc",
        "D-FINE large",
        "D-FINE large (Objects365+COCO), COCO 80 classes; accurate, best on a GPU",
        "76e521ed66874590456f9eeb89c2adf33f0599778cd33b1b614f3090f95845b9",
        125348332
    ),
    dfine!(
        "dfine-x",
        "onnx-community/dfine_x_obj2coco-ONNX",
        "4d3a85a6c29f55a5346fabaf9728f5f3751f947a",
        "D-FINE extra large",
        "D-FINE extra large (Objects365+COCO), COCO 80 classes; most accurate, slow without a GPU",
        "486ee1cba40b25f3ea8d6783f664eca88b7c12d01078984834d36834d9b2c802",
        251138448
    ),
    rfdetr!(
        "rfdetr-nano",
        "onnx-community/rfdetr_nano-ONNX",
        "eae21cee0687a91bcf9fa071605c48d7705d2d91",
        "RF-DETR nano",
        "RF-DETR nano, COCO 80 classes, 384x384 input; fastest RF-DETR",
        "9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee",
        108074865
    ),
    rfdetr!(
        "rfdetr-small",
        "onnx-community/rfdetr_small-ONNX",
        "63463b68b200177d1fea7015f11f3cebb0ba4eeb",
        "RF-DETR small",
        "RF-DETR small, COCO 80 classes, 512x512 input; fast",
        "121cc1476a7b69d865ca4bdc2bca59a3239020d227e4c3f35a811bef81aeb7f1",
        114680416
    ),
    rfdetr!(
        "rfdetr-base",
        "onnx-community/rfdetr_base-ONNX",
        "7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f",
        "RF-DETR base",
        "RF-DETR base, COCO 80 classes, 560x560 input; balanced accuracy and speed",
        "48b9502c04be895713efc0579c8904e69a97b3e03413a2f5aab1bdecda7e36ff",
        107773649
    ),
    rfdetr!(
        "rfdetr-medium",
        "onnx-community/rfdetr_medium-ONNX",
        "8553cefdd984bbaeb9d9ca8743d6a6a1c3e29dce",
        "RF-DETR medium",
        "RF-DETR medium, COCO 80 classes, 576x576 input; more accurate, slower",
        "768928acf337ac86d069c9d55a929e11e563c1e1d272a3ffc9366248d66b2654",
        121015411
    ),
    yolo5!(
        "delivery",
        "YOLOv5 delivery (vehicles, people, packages)",
        "1e66d71f3d5bbd6140e15716a43ca0e25fc5132af0a959c2696c11c1bf643e56",
        29571692,
        "17e0d52b7df884f834b9894395847319c0e872725327076df8684ae2045e962b",
        155
    ),
    yolo5!(
        "IPcam-animal",
        "YOLOv5 IP camera animals",
        "e62a9f8df5299476cf77e2351c711f7554e64b1200cca7756d7b58bbb9f32085",
        29616938,
        "9ca7e805c2c70013311688ebe1b339dfcb38f17216d5d499cc8195cc6b83ca61",
        227
    ),
    yolo5!(
        "ipcam-bird",
        "YOLOv5 IP camera birds",
        "265363bff2eda770463be1a1054eaaf722dec3703ec5b672a155fd1315a486be",
        84407925,
        "d79b8168621ca94221010b958bdebc08ce899ff68d4d03dc60937685f110b847",
        851
    ),
    yolo5!(
        "IPcam-combined",
        "YOLOv5 IP camera combined",
        "8f9f39828f979a4440151be2d32660e97f7a41e981af26002c5b638c4db9236a",
        29681724,
        "5224a2b0d32a0ea6ef9bb9a03ed54087280bffd0ee8228af3c46eb800b4a8315",
        308
    ),
    yolo5!(
        "IPcam-dark",
        "YOLOv5 IP camera night/dark scenes",
        "ca460c36ff279f0344756b3cadf936323fe01b28ef67b226cf00ca86b05808b7",
        29519768,
        "1e148765187f81e4fd931975810602eb5b7f976fb44ef4e679c578036e4b6558",
        114
    ),
    yolo5!(
        "IPcam-general",
        "YOLOv5 IP camera general purpose",
        "3b338be21f271a2924051fc237086a03c49daecd7e0fb2a8e97411ea5a33c6dc",
        29465778,
        "4070e087499ab1bf111efc58d98e6703b9457ce06ff0c22fb79d548074afb9c4",
        52
    ),
    yolo5!(
        "package",
        "YOLOv5 package detection",
        "b233f5a622a59f5a7ac7b48fbed79b5989ad6225a9569b3a8866a0bd75d9a9d7",
        29452878,
        "6187ff891bb16e442c0ce36593c04be118ab613299ea6060c4b1fd8c2855fe81",
        23
    ),
];

// ---------------------------------------------------------------------------------------------
// Models exported locally (Ultralytics YOLO, RF-DETR L, DEIMv2)
// ---------------------------------------------------------------------------------------------

/// How an exportable model is turned into `<name>.onnx` (`scripts/export_model.py --recipe`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExportRecipe {
    /// Ultralytics' exporter on an official `.pt` (YOLOv8/9/10/11/12/26). AGPL-3.0.
    Ultralytics,
    /// Hugging Face transformers `RfDetrForObjectDetection` + `torch.onnx` (static input,
    /// `pixel_values` -> `logits`, `pred_boxes`). Apache-2.0.
    RfDetr,
    /// The official DEIMv2 code ([`DEIMV2_SRC_ID`]) + `torch.onnx`, deploy mode (`images`,
    /// `orig_target_sizes` -> `labels`, `boxes`, `scores`).
    DeimV2,
}

impl ExportRecipe {
    /// `--recipe` value of the export script.
    pub const fn as_str(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => "ultralytics",
            ExportRecipe::RfDetr => "rfdetr",
            ExportRecipe::DeimV2 => "deimv2",
        }
    }

    /// What runs on this machine, for confirmations and logs.
    pub const fn exporter(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => "Ultralytics' exporter",
            ExportRecipe::RfDetr => "Hugging Face transformers and torch.onnx",
            ExportRecipe::DeimV2 => "the official DEIMv2 code (pinned commit) and torch.onnx",
        }
    }

    /// Shown before an export and in the Resources card.
    pub const fn notice(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => ULTRALYTICS_NOTICE,
            ExportRecipe::RfDetr => RFDETR_NOTICE,
            ExportRecipe::DeimV2 => DEIMV2_NOTICE,
        }
    }
}

/// Ultralytics package version the export environment pins (`requirements-model-export.txt`).
pub const ULTRALYTICS_VERSION: &str = "8.4.163";
/// Ultralytics `assets` release with the YOLO weights (v8 to 26 are all in it).
pub const ULTRALYTICS_ASSETS_RELEASE: &str = "v8.4.0";
/// Version recorded for an Ultralytics export: exporter version + weights release (unchanged since
/// the YOLO26-only export, so earlier YOLO26 exports stay valid).
pub const ULTRALYTICS_EXPORT_VERSION: &str = "ultralytics-8.4.163+assets-v8.4.0";
/// License of the Ultralytics weights and exporter.
pub const ULTRALYTICS_LICENSE: &str = "AGPL-3.0";
const ULTRALYTICS_NOTICE: &str = "Ultralytics YOLO weights and the Ultralytics exporter are \
    licensed AGPL-3.0. Blue Onyx Prism does not redistribute them: on your request it downloads the \
    official weights from Ultralytics' GitHub release (SHA-256 pinned) and runs Ultralytics' \
    exporter on this machine in its own Python environment.";
/// transformers version of the export environment (RF-DETR).
pub const TRANSFORMERS_VERSION: &str = "5.17.0";
/// `Roboflow/rf-detr-large` commit of the RF-DETR L weights.
pub const RFDETR_LARGE_REVISION: &str = "f62f7dd5252b61097cbace33886045816dadbde9";
const RFDETR_NOTICE: &str = "RF-DETR (Apache-2.0) is published as safetensors only: on your \
    request Blue Onyx Prism downloads Roboflow's official weights from Hugging Face (SHA-256 \
    pinned) and exports them to ONNX on this machine with transformers and torch.onnx.";
/// The DEIMv2 source code resource (`Intellindust-AI-Lab/DEIMv2`, GitHub archive).
pub const DEIMV2_SRC_ID: &str = "tool:deimv2-src";
/// Install directory of [`DEIMV2_SRC_ID`] under the download root.
pub const DEIMV2_SRC_DEST: &str = "tools/deimv2-src";
/// DEIMv2 commit the source archive is pinned to (2026-08-14): the last one released under
/// Apache-2.0 (the repository switched to a non-commercial license on 2026-08-24 with docs-only
/// commits; the model code is the same).
pub const DEIMV2_COMMIT: &str = "add5bcdb499bf7b8a366bfeac1a47d3dc278de27";
const DEIMV2_NOTICE: &str = "DEIMv2 is exported on this machine from Intellindust's official \
    weights (Hugging Face, SHA-256 pinned, published 2025-10 under Apache-2.0) with the official \
    DEIMv2 code pinned to commit add5bcd, the last Apache-2.0 release; the repository has since \
    moved to a non-commercial license. The DINOv3 variants (S/M/L/X) also carry Meta's DINOv3 \
    License for the backbone.";

macro_rules! ultralytics {
    ($name:literal, $family:expr, $title:literal, $desc:literal, $sha:literal, $bytes:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::ExportModel,
            version: ULTRALYTICS_EXPORT_VERSION,
            platform: None,
            parts: &[Part {
                url: concat!(
                    "https://github.com/ultralytics/assets/releases/download/v8.4.0/",
                    $name,
                    ".pt"
                ),
                sha256: $sha,
                size: $bytes,
                file_name: concat!($name, ".pt"),
                archive: None,
                layout: Layout::File,
            }],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: $family,
            },
            title: $title,
            description: $desc,
            license: ULTRALYTICS_LICENSE,
        }
    };
}

/// YOLOv8 / v9 / 11 / 12: raw `[1, 84, 8400]` head, family `yolo8` (NMS in the service).
macro_rules! yolo_raw {
    ($($args:tt)*) => {
        ultralytics!($($args)*)
    };
}

/// A plain file of a Hugging Face repo at a commit.
macro_rules! hf_part {
    ($repo:expr, $rev:expr, $file:literal, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://huggingface.co/",
                $repo,
                "/resolve/",
                $rev,
                "/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: None,
            layout: Layout::File,
        }
    };
}

/// A DEIMv2 COCO model (`Intellindust/DEIMv2_<variant>_COCO`): `model.safetensors` +
/// `config.json` (PyTorchModelHubMixin), family `rtdetr`.
macro_rules! deimv2 {
    ($name:literal, $variant:literal, $rev:literal, $title:literal, $desc:literal, $license:expr,
     $st_sha:literal, $st_size:literal, $cfg_sha:literal, $cfg_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::ExportModel,
            version: concat!("deimv2-add5bcd+hf-", $rev),
            platform: None,
            parts: &[
                hf_part!(
                    concat!("Intellindust/DEIMv2_", $variant, "_COCO"),
                    $rev,
                    "model.safetensors",
                    $st_sha,
                    $st_size
                ),
                hf_part!(
                    concat!("Intellindust/DEIMv2_", $variant, "_COCO"),
                    $rev,
                    "config.json",
                    $cfg_sha,
                    $cfg_size
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::RtDetr,
            },
            title: $title,
            description: $desc,
            license: $license,
        }
    };
}

/// DEIMv2 is marked non-commercial / personal use: the pinned source (add5bcd) and the weights
/// predate the move to the non-commercial DEIMv2 License, but that license now claims the weights.
const DEIMV2_HGNET_LICENSE: &str = "Non-commercial / personal use only (DEIMv2 License)";
const DEIMV2_DINOV3_LICENSE: &str =
    "Non-commercial / personal use only (DEIMv2 License) + DINOv3 License (backbone)";

/// Models exported on this machine on request ([`super::export`]): the parts are the official
/// weights (and configs); the export writes `models/<name>.onnx` + `models/<name>.yaml` (80 COCO
/// names). All COCO 80 classes. YOLOv10 and YOLO26 are end-to-end (`[1,300,6]`, family
/// `yolo26`); the other YOLOs are raw heads (family `yolo8`). Never downloaded or exported
/// automatically.
pub static EXPORT_MODELS: &[Resource] = &[
    // YOLOv8
    yolo_raw!(
        "yolov8n",
        ModelFamilyKind::Yolo8,
        "YOLOv8 nano",
        "YOLOv8 nano, COCO 80 classes; fast and small, a good CPU/iGPU baseline",
        "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        6549796
    ),
    yolo_raw!(
        "yolov8s",
        ModelFamilyKind::Yolo8,
        "YOLOv8 small",
        "YOLOv8 small, COCO 80 classes; more accurate than nano, still light",
        "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
        22588772
    ),
    yolo_raw!(
        "yolov8m",
        ModelFamilyKind::Yolo8,
        "YOLOv8 medium",
        "YOLOv8 medium, COCO 80 classes; accurate, best with a GPU",
        "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
        52136884
    ),
    yolo_raw!(
        "yolov8l",
        ModelFamilyKind::Yolo8,
        "YOLOv8 large",
        "YOLOv8 large, COCO 80 classes; slow without a GPU",
        "64c9115303f6a25575f82200d1b22ec409fa6bd7d08d0313884fc20d919478cd",
        87792836
    ),
    yolo_raw!(
        "yolov8x",
        ModelFamilyKind::Yolo8,
        "YOLOv8 extra large",
        "YOLOv8 extra large, COCO 80 classes; needs a fast GPU",
        "3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8",
        136890692
    ),
    // YOLOv9
    yolo_raw!(
        "yolov9t",
        ModelFamilyKind::Yolo8,
        "YOLOv9 tiny",
        "YOLOv9 tiny, COCO 80 classes; smallest YOLOv9",
        "410e3caca789aa8fec19971c05577616cbb963a416bcd0228171b2dff7f718f7",
        4969803
    ),
    yolo_raw!(
        "yolov9s",
        ModelFamilyKind::Yolo8,
        "YOLOv9 small",
        "YOLOv9 small, COCO 80 classes; light and accurate for its size",
        "67731fc88bb6782fbb4f32703a2cac052acd65148b65c9e67161fe13536c2ffc",
        15389259
    ),
    yolo_raw!(
        "yolov9m",
        ModelFamilyKind::Yolo8,
        "YOLOv9 medium",
        "YOLOv9 medium, COCO 80 classes; accurate, best with a GPU",
        "be7ac91362bd81dfa901077706d634e571c1ff7f3886b3aac6a0e97eff2f5440",
        40993492
    ),
    yolo_raw!(
        "yolov9c",
        ModelFamilyKind::Yolo8,
        "YOLOv9 compact",
        "YOLOv9 compact (C), COCO 80 classes; large, needs a GPU",
        "876eb84f515d40c34a3b111f8fc1077d3aee59d3a243afd1cc5b77d520f237c7",
        51794840
    ),
    yolo_raw!(
        "yolov9e",
        ModelFamilyKind::Yolo8,
        "YOLOv9 extended",
        "YOLOv9 extended (E), COCO 80 classes; most accurate YOLOv9, needs a fast GPU",
        "0d2270e0c794bdaa059738d32c1781cdec4857b3931d5503e13ae92e8517e7bd",
        117521038
    ),
    // YOLOv10 (end-to-end, no NMS)
    ultralytics!(
        "yolov10n",
        ModelFamilyKind::Yolo26,
        "YOLOv10 nano",
        "YOLOv10 nano, COCO 80 classes, NMS-free; fast",
        "11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f",
        5860383
    ),
    ultralytics!(
        "yolov10s",
        ModelFamilyKind::Yolo26,
        "YOLOv10 small",
        "YOLOv10 small, COCO 80 classes, NMS-free; more accurate, still light",
        "65fa8332c38a7189597a268897bce6c1026f9499711b9761fb0c2c639b91d4d6",
        16623111
    ),
    ultralytics!(
        "yolov10m",
        ModelFamilyKind::Yolo26,
        "YOLOv10 medium",
        "YOLOv10 medium, COCO 80 classes, NMS-free; accurate, best with a GPU",
        "6dc78f7a88591cec1e8716b8f5c7e3aefa9206684f025d202be34439ccb329a0",
        33643667
    ),
    ultralytics!(
        "yolov10b",
        ModelFamilyKind::Yolo26,
        "YOLOv10 balanced",
        "YOLOv10 balanced (B), COCO 80 classes, NMS-free; needs a GPU",
        "31ef0baecfa8010ebb4b53b0624836f30d73702b04710c23945782dba5d258c2",
        41679663
    ),
    ultralytics!(
        "yolov10l",
        ModelFamilyKind::Yolo26,
        "YOLOv10 large",
        "YOLOv10 large, COCO 80 classes, NMS-free; needs a GPU",
        "d0685e1e597f73aba8e56bc9651b6bb939c5b2f201dff5de0b9ef2a6207ae301",
        52425230
    ),
    ultralytics!(
        "yolov10x",
        ModelFamilyKind::Yolo26,
        "YOLOv10 extra large",
        "YOLOv10 extra large, COCO 80 classes, NMS-free; needs a fast GPU",
        "605982e0b2c2ff070351afc8df3ecb38c54cfd7f133813acb366f454eb79b9a6",
        64395854
    ),
    // YOLO11
    yolo_raw!(
        "yolo11n",
        ModelFamilyKind::Yolo8,
        "YOLO11 nano",
        "YOLO11 nano, COCO 80 classes; fast, good on CPUs and iGPUs",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        5613764
    ),
    yolo_raw!(
        "yolo11s",
        ModelFamilyKind::Yolo8,
        "YOLO11 small",
        "YOLO11 small, COCO 80 classes; more accurate, about 2x nano's time",
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        19313732
    ),
    yolo_raw!(
        "yolo11m",
        ModelFamilyKind::Yolo8,
        "YOLO11 medium",
        "YOLO11 medium, COCO 80 classes; accurate, best with a GPU",
        "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
        40684120
    ),
    yolo_raw!(
        "yolo11l",
        ModelFamilyKind::Yolo8,
        "YOLO11 large",
        "YOLO11 large, COCO 80 classes; slow without a GPU",
        "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
        51387343
    ),
    yolo_raw!(
        "yolo11x",
        ModelFamilyKind::Yolo8,
        "YOLO11 extra large",
        "YOLO11 extra large, COCO 80 classes; needs a fast GPU",
        "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        114636239
    ),
    // YOLO12
    yolo_raw!(
        "yolo12n",
        ModelFamilyKind::Yolo8,
        "YOLO12 nano",
        "YOLO12 nano, COCO 80 classes (attention-based); fast",
        "419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6",
        5595063
    ),
    yolo_raw!(
        "yolo12s",
        ModelFamilyKind::Yolo8,
        "YOLO12 small",
        "YOLO12 small, COCO 80 classes (attention-based); more accurate",
        "e915c2c4286e3f6f8610ef106fa3f94a7b8c19b30eccede5887e22c33ef75f58",
        19006455
    ),
    yolo_raw!(
        "yolo12m",
        ModelFamilyKind::Yolo8,
        "YOLO12 medium",
        "YOLO12 medium, COCO 80 classes (attention-based); best with a GPU",
        "4c6d179786eddf6134ee469ae2f4ce04cbe4e9d1a47d6b669d9cd6b9c6c513d8",
        40904955
    ),
    yolo_raw!(
        "yolo12l",
        ModelFamilyKind::Yolo8,
        "YOLO12 large",
        "YOLO12 large, COCO 80 classes (attention-based); needs a GPU",
        "0babd8dc8f775bb64bb052debdff3d8b9e9b57efa9d7bfa11c84bb82c3fec336",
        53699086
    ),
    yolo_raw!(
        "yolo12x",
        ModelFamilyKind::Yolo8,
        "YOLO12 extra large",
        "YOLO12 extra large, COCO 80 classes (attention-based); needs a fast GPU",
        "682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f",
        119322638
    ),
    // YOLO26 (end-to-end, no NMS). Speeds: ONNX Runtime CoreML on an Apple M1.
    ultralytics!(
        "yolo26n",
        ModelFamilyKind::Yolo26,
        "YOLO26 nano",
        "YOLO26 nano, COCO 80 classes; fastest (about 15 ms on an M1 with CoreML), good on CPUs and iGPUs",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        5544453
    ),
    ultralytics!(
        "yolo26s",
        ModelFamilyKind::Yolo26,
        "YOLO26 small",
        "YOLO26 small, COCO 80 classes; more accurate, about 2x nano's time (29 ms on an M1 with CoreML)",
        "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
        20422725
    ),
    ultralytics!(
        "yolo26m",
        ModelFamilyKind::Yolo26,
        "YOLO26 medium",
        "YOLO26 medium, COCO 80 classes; accurate, best with a GPU",
        "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
        44255705
    ),
    ultralytics!(
        "yolo26l",
        ModelFamilyKind::Yolo26,
        "YOLO26 large",
        "YOLO26 large, COCO 80 classes; slower, needs a GPU for camera streams",
        "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
        53211173
    ),
    ultralytics!(
        "yolo26x",
        ModelFamilyKind::Yolo26,
        "YOLO26 extra large",
        "YOLO26 extra large, COCO 80 classes; most accurate and slowest, needs a fast GPU",
        "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
        118667365
    ),
    // RF-DETR L (current release, DINOv2 windowed backbone, 704x704), family `rfdetr`.
    Resource {
        id: "model:rfdetr-large",
        kind: ResourceKind::ExportModel,
        version: "transformers-5.17.0+rf-detr-large-f62f7dd5252b61097cbace33886045816dadbde9",
        platform: None,
        parts: &[
            hf_part!(
                "Roboflow/rf-detr-large",
                "f62f7dd5252b61097cbace33886045816dadbde9",
                "model.safetensors",
                "1ec604598aea748627eb5755a9989f3c80839fc7fb977a91bfba107ee76fcd7d",
                135795376
            ),
            hf_part!(
                "Roboflow/rf-detr-large",
                "f62f7dd5252b61097cbace33886045816dadbde9",
                "config.json",
                "6630951d97cc8dd2f32da43df7c79af153b032f6df196bbe9bf9e47bd7be0c0c",
                5771
            ),
        ],
        dest: "models",
        provides: Provides::Model {
            name: "rfdetr-large",
            family: ModelFamilyKind::RfDetr,
        },
        title: "RF-DETR large",
        description: "RF-DETR large (current, 33.9M params), COCO 80 classes, 704x704 input; most accurate RF-DETR, needs a GPU",
        license: "Apache-2.0",
    },
    // DEIMv2 (HGNetv2 backbones: ImageNet-free input; DINOv3: normalization baked into the graph).
    deimv2!(
        "deimv2-atto",
        "HGNetv2_ATTO",
        "21fbe0e24abeb35ed216dbfd8a32855dbc03214f",
        "DEIMv2 atto",
        "DEIMv2 atto (0.5M params), COCO 80 classes, 320x320 input; tiny, for very weak CPUs",
        DEIMV2_HGNET_LICENSE,
        "774dd877ef7b90e10d4136f31555f34627a62dc00a0d6908d9b1a48574002730",
        2120688,
        "51bcdedb67ac127b6f955c1f919c7836db3eb358465e2c7057bba8c270368081",
        822
    ),
    deimv2!(
        "deimv2-femto",
        "HGNetv2_FEMTO",
        "ccbd03616ce5373c32622c605fb29bbb7dec4b52",
        "DEIMv2 femto",
        "DEIMv2 femto (1.0M params), COCO 80 classes, 416x416 input; very small",
        DEIMV2_HGNET_LICENSE,
        "db7026cc3a7c308934a8cf3fa8244f9967078d7cc8ec20da8e5250ef8842e75f",
        4033545,
        "b37530572fb615601f18f7876e1619d00b73cf9e71c3a25c0b79ba03c7eb3717",
        823
    ),
    deimv2!(
        "deimv2-pico",
        "HGNetv2_PICO",
        "849bb61f94f3d42a8977cabb38b998fa91fed44a",
        "DEIMv2 pico",
        "DEIMv2 pico (1.5M params), COCO 80 classes, 640x640 input; small and fast",
        DEIMV2_HGNET_LICENSE,
        "788b3a42297333ae3d47c337a89d859187f9000a7303f748773e88db8a454e0f",
        6293772,
        "ae9c327e8d281e626e324660d8c71480abe8ac6c91dd4fe57645480d82e6b84f",
        826
    ),
    deimv2!(
        "deimv2-n",
        "HGNetv2_N",
        "47f40396e40453719386cd5c238a57aae954e83f",
        "DEIMv2 nano",
        "DEIMv2 nano (3.6M params), COCO 80 classes, 640x640 input; fast, good on CPUs and iGPUs",
        DEIMV2_HGNET_LICENSE,
        "0de14388a703d86c95200589b32a4497b0b00dceb1d2d381c6a14fbd958856cb",
        14574360,
        "6eb57450791c8ada4b76811ad4fbcf040a58911ecf3dff5a785487389d4443f8",
        1326
    ),
    deimv2!(
        "deimv2-s",
        "DINOv3_S",
        "cf0540f3f319bb8ecbe132624358a35a7beb86d5",
        "DEIMv2 small",
        "DEIMv2 small (DINOv3-distilled ViT, 9.7M params), COCO 80 classes; accurate for its size",
        DEIMV2_DINOV3_LICENSE,
        "fe545b150766b8761a696f7b1d92ea138fafe8e02a7398823e0c62d2a5867956",
        39386484,
        "c815791122b6be862d5c5902b3557771bda640acd0ca21ca9999c540eb6e6853",
        1321
    ),
    deimv2!(
        "deimv2-m",
        "DINOv3_M",
        "c5db2fc6f4d9e59635cfc2effaf097bc9a293beb",
        "DEIMv2 medium",
        "DEIMv2 medium (DINOv3-distilled ViT, 18.1M params), COCO 80 classes; accurate, best with a GPU",
        DEIMV2_DINOV3_LICENSE,
        "f5d6beb9c032b7cdd24da6eebf6966da6c7d2d49b312ac6c0198f112170c0af6",
        73752660,
        "468aa716b1cec24298dfdba4c07c07eab1ff4b6e7a3a64c7396a367a93d7a67b",
        1322
    ),
    deimv2!(
        "deimv2-l",
        "DINOv3_L",
        "6ce90ce18c14271c8ce145c35626d9a6544f881b",
        "DEIMv2 large",
        "DEIMv2 large (DINOv3 ViT-S, 32.2M params), COCO 80 classes; needs a GPU",
        DEIMV2_DINOV3_LICENSE,
        "c0d0f45fa62b785126db5b40f5530866fecaab2d340cea812d2ba5f0f5646163",
        130596044,
        "9800c67c83501a1f5159b5b372c7f96b367b329d3713ca6d7309c86b2e7a0319",
        1373
    ),
    deimv2!(
        "deimv2-x",
        "DINOv3_X",
        "b1d81486dce5686b2afcd38a231d086d142c996d",
        "DEIMv2 extra large",
        "DEIMv2 extra large (DINOv3 ViT-S+, 50.3M params), COCO 80 classes; most accurate DEIMv2, needs a fast GPU",
        DEIMV2_DINOV3_LICENSE,
        "c15e1860118e6841f8f803219507dbfb630c2aa0164ab723c339b94e554bae7f",
        205313972,
        "456e2f740828eb52bc3dcc789de28af67f366cdf2b415f2338bc3b4e2f28fae4",
        1382
    ),
];

/// The recipe of an exportable model (keyed on its name; every [`EXPORT_MODELS`] entry has one,
/// checked by a test).
pub fn export_recipe(res: &Resource) -> Option<ExportRecipe> {
    if res.kind != ResourceKind::ExportModel {
        return None;
    }
    let name = res.model_name()?;
    if name.starts_with("yolo") {
        Some(ExportRecipe::Ultralytics)
    } else if name.starts_with("rfdetr") {
        Some(ExportRecipe::RfDetr)
    } else if name.starts_with("deimv2") {
        Some(ExportRecipe::DeimV2)
    } else {
        None
    }
}

/// Square input side an export of `res` uses (`--imgsz`): [`MODEL_INPUT_SIZES`], else 640.
pub fn export_input_size(res: &Resource) -> u32 {
    res.model_name()
        .and_then(model_input_size)
        .unwrap_or(crate::model::DEFAULT_INPUT_SIZE)
}

/// Other resources an export of `res` needs besides its weights (the DEIMv2 source code).
pub fn export_extras(res: &Resource) -> Vec<&'static Resource> {
    match export_recipe(res) {
        Some(ExportRecipe::DeimV2) => deimv2_source().into_iter().collect(),
        _ => Vec::new(),
    }
}

/// What setting up the model export toolchain costs on one platform (uv, Python 3.11 and the
/// pinned packages; the weights are extra). Download sizes are the sums of the locked wheels (see
/// `scripts/model-export/gen_locks.sh`; macOS 334 MB, Linux x64 443 MB, Linux arm64 379 MB, Windows
/// 321 MB) plus uv and Python; disk is `tools/` afterwards (measured on macOS arm64; the others
/// scaled by their wheel sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolchainEstimate {
    pub platform: Platform,
    /// Bytes downloaded once (uv archive + Python + wheels).
    pub download: u64,
    /// Bytes on disk afterwards (`tools/`).
    pub disk: u64,
}

/// Platforms the model export supports (PyTorch 2.14 CPU wheels exist), with their costs.
pub const TOOLCHAIN_ESTIMATES: &[ToolchainEstimate] = &[
    ToolchainEstimate {
        platform: Platform::MACOS_ARM64,
        download: 375_000_000,
        disk: 1_300_000_000,
    },
    ToolchainEstimate {
        platform: Platform::LINUX_X64,
        download: 495_000_000,
        disk: 1_700_000_000,
    },
    ToolchainEstimate {
        platform: Platform::LINUX_ARM64,
        download: 425_000_000,
        disk: 1_500_000_000,
    },
    ToolchainEstimate {
        platform: Platform::WINDOWS_X64,
        download: 385_000_000,
        disk: 1_400_000_000,
    },
];

/// The export toolchain estimate for `(os, arch)`; None = models cannot be exported there.
pub fn toolchain_estimate(os: &str, arch: &str) -> Option<&'static ToolchainEstimate> {
    TOOLCHAIN_ESTIMATES
        .iter()
        .find(|e| e.platform.matches(os, arch))
}

/// Exportable model by name or file name (`yolo26s`, `YOLO26S.onnx`, `yolo26s.xml`,
/// `model:deimv2-n`).
pub fn export_model(name: &str) -> Option<&'static Resource> {
    let n = name.trim();
    let stem = std::path::Path::new(n)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(n);
    let stem = strip_prefix_ci(stem, MODEL_ID_PREFIX).unwrap_or(stem);
    EXPORT_MODELS
        .iter()
        .find(|r| r.model_name().is_some_and(|m| m.eq_ignore_ascii_case(stem)))
}

/// File names an export of `res` writes into the models directory: `(<name>.onnx, <name>.yaml)`.
pub fn export_files(res: &Resource) -> Option<(String, String)> {
    (res.kind == ResourceKind::ExportModel)
        .then(|| res.model_name())
        .flatten()
        .map(|n| (format!("{n}.onnx"), format!("{n}.yaml")))
}

// ---------------------------------------------------------------------------------------------
// Tools (uv and the DEIMv2 source, for the model export)
// ---------------------------------------------------------------------------------------------

macro_rules! uv_part {
    ($file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://github.com/astral-sh/uv/releases/download/0.12.19/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::Uv,
        }
    };
}

const fn uv(platform: Platform, parts: &'static [Part]) -> Resource {
    Resource {
        id: UV_ID,
        kind: ResourceKind::Tool,
        version: UV_VERSION,
        platform: Some(platform),
        parts,
        dest: UV_DEST,
        provides: Provides::Tool { name: "uv" },
        title: "uv (model export toolchain)",
        description: "Astral's uv: sets up Python 3.11 and the pinned export packages in \
                      tools/export-env; only runs when you export a model",
        license: "MIT OR Apache-2.0",
    }
}

/// `tool:uv` per platform: the standalone uv binary from Astral's GitHub release (only `uv` /
/// `uv.exe` is extracted).
pub static TOOLS: &[Resource] = &[
    uv(
        Platform::MACOS_ARM64,
        &[uv_part!(
            "uv-aarch64-apple-darwin.tar.gz",
            ArchiveKind::TarGz,
            "a9a8df1eedeb192f2e47e40e2faabfb387db4b850209118786d42f89dde3e0ba",
            16988553
        )],
    ),
    uv(
        Platform::new("macos", "x86_64"),
        &[uv_part!(
            "uv-x86_64-apple-darwin.tar.gz",
            ArchiveKind::TarGz,
            "cb5fa57bafe68fc0fb94b17f06bee0b0b9a7feb94ccbd110445afa0696e39273",
            20718482
        )],
    ),
    uv(
        Platform::LINUX_X64,
        &[uv_part!(
            "uv-x86_64-unknown-linux-gnu.tar.gz",
            ArchiveKind::TarGz,
            "23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8",
            19831732
        )],
    ),
    uv(
        Platform::LINUX_ARM64,
        &[uv_part!(
            "uv-aarch64-unknown-linux-gnu.tar.gz",
            ArchiveKind::TarGz,
            "0804e9b164c64b6914182d5920c08551958a095986f10a3731056df701126436",
            18916891
        )],
    ),
    uv(
        Platform::WINDOWS_X64,
        &[uv_part!(
            "uv-x86_64-pc-windows-msvc.zip",
            ArchiveKind::Zip,
            "6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0",
            17955780
        )],
    ),
];

/// `tool:deimv2-src`: the DEIMv2 source code at [`DEIMV2_COMMIT`] (GitHub archive). Only
/// `engine/**/*.py`, `configs/**/*.yml` and the license are extracted; the export script imports
/// the few modules the model needs and builds it from the training config. Code that runs, so only fetched by a user-started DEIMv2 export.
pub static DEIMV2_SOURCE: &[Resource] = &[Resource {
    id: DEIMV2_SRC_ID,
    kind: ResourceKind::Tool,
    version: DEIMV2_COMMIT,
    platform: None,
    parts: &[Part {
        url: "https://github.com/Intellindust-AI-Lab/DEIMv2/archive/add5bcdb499bf7b8a366bfeac1a47d3dc278de27.tar.gz",
        sha256: "62dc953f01b9189922f846f048d03967daa33d6f97596a9e8e71d58430db2216",
        size: 839973,
        file_name: "add5bcdb499bf7b8a366bfeac1a47d3dc278de27.tar.gz",
        archive: Some(ArchiveKind::TarGz),
        layout: Layout::SourceTree,
    }],
    dest: DEIMV2_SRC_DEST,
    provides: Provides::Tool { name: "deimv2-src" },
    title: "DEIMv2 source (model export)",
    description: "Intellindust's DEIMv2 model code at commit add5bcd; only used when you export a DEIMv2 model",
    license: "Apache-2.0",
}];

/// The DEIMv2 source resource.
pub fn deimv2_source() -> Option<&'static Resource> {
    DEIMV2_SOURCE.first()
}

/// `tool:uv` for `(os, arch)`.
pub fn uv_for(os: &str, arch: &str) -> Option<&'static Resource> {
    TOOLS
        .iter()
        .find(|r| r.id == UV_ID && r.available_on(os, arch))
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
pub fn bench_resource(m: &crate::benchmark::images::Manifest) -> Resource {
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
pub fn bench_sets() -> &'static [Resource] {
    static SETS: std::sync::OnceLock<Vec<Resource>> = std::sync::OnceLock::new();
    SETS.get_or_init(|| {
        crate::benchmark::images::builtin_manifests()
            .iter()
            .map(bench_resource)
            .collect()
    })
}

/// The `bench:<set>` resource.
pub fn bench_set(set: &str) -> Option<&'static Resource> {
    bench_sets()
        .iter()
        .find(|r| matches!(r.provides, Provides::BenchImages { set: s } if s.eq_ignore_ascii_case(set.trim())))
}

/// Resources usable on `(os, arch)` (`fetch --all-for-platform`).
pub fn for_platform<'a>(
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
