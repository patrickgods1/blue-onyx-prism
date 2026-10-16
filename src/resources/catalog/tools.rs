//! Tool pins for the model export: Astral's uv per platform ([`TOOLS`]), the DEIMv2 source
//! archive ([`DEIMV2_SOURCE`]) and the Windows Visual C++ runtime ([`VC_RUNTIME`]). Never run here;
//! only `super::super::export` uses them, when the user starts an export.

use super::*;

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
pub(crate) static DEIMV2_SOURCE: &[Resource] = &[Resource {
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
pub(crate) fn deimv2_source() -> Option<&'static Resource> {
    DEIMV2_SOURCE.first()
}

/// `tool:uv` for `(os, arch)`.
pub fn uv_for(os: &str, arch: &str) -> Option<&'static Resource> {
    TOOLS
        .iter()
        .find(|r| r.id == UV_ID && r.available_on(os, arch))
}

// ---------------------------------------------------------------------------------------------
// Microsoft Visual C++ runtime (Windows, for the model export's PyTorch)
// ---------------------------------------------------------------------------------------------

/// The private Visual C++ runtime of the model export.
pub const VC_RUNTIME_ID: &str = "tool:vc-runtime";
/// Install directory of [`VC_RUNTIME_ID`] under the download root.
pub const VC_RUNTIME_DEST: &str = "tools/vc-runtime";
/// Version of the `msvc-runtime` wheel (Visual Studio 2022 17.14 redistributable).
pub const VC_RUNTIME_VERSION: &str = "14.44.35112";
/// `FileVersion` of every DLL in [`VC_RUNTIME_DLLS`] (Microsoft-signed, unmodified).
pub const VC_RUNTIME_FILE_VERSION: [u16; 4] = [14, 44, 35211, 0];
/// Oldest Visual C++ runtime the pinned PyTorch loads with (built with Visual Studio 2022
/// 17.10+, which needs `msvcp140.dll` 14.40 and its `vcruntime140_threads.dll` /
/// `msvcp140_atomic_wait.dll`).
pub const VC_RUNTIME_TORCH_MIN: [u16; 2] = [14, 40];

/// The DLLs extracted from [`VC_RUNTIME`]: `(file name, entry path in the wheel, SHA-256)`.
/// Every one is checked after extraction; nothing else of the wheel is written.
pub const VC_RUNTIME_DLLS: &[(&str, &str, &str)] = &[
    (
        "concrt140.dll",
        "msvc_runtime-14.44.35112.data/data/concrt140.dll",
        "2405355f0a58067b258f8df33c327e3a3d716eaac5a3a5aebb757842d85bd376",
    ),
    (
        "msvcp140.dll",
        "msvc_runtime-14.44.35112.data/data/msvcp140.dll",
        "0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9",
    ),
    (
        "msvcp140_1.dll",
        "msvc_runtime-14.44.35112.data/data/msvcp140_1.dll",
        "bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d",
    ),
    (
        "msvcp140_2.dll",
        "msvc_runtime-14.44.35112.data/data/msvcp140_2.dll",
        "3ea06f0ee098b4823cb79599df3780e7f23cce52c19aac31d2a0d47efe33a5e9",
    ),
    (
        "msvcp140_atomic_wait.dll",
        "msvc_runtime-14.44.35112.data/data/msvcp140_atomic_wait.dll",
        "640b2aefced484d0368eea5bdd06addd0658a3a70a49256e560d6923b404a479",
    ),
    (
        "msvcp140_codecvt_ids.dll",
        "msvc_runtime-14.44.35112.data/data/msvcp140_codecvt_ids.dll",
        "f2069a52880ec885ee7f0511186100eb7fada0411a2b4948fafea7735b878a18",
    ),
    (
        "vcomp140.dll",
        "msvc_runtime-14.44.35112.data/data/vcomp140.dll",
        "55aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164",
    ),
    (
        "vcruntime140_threads.dll",
        "msvc_runtime-14.44.35112.data/data/vcruntime140_threads.dll",
        "219915cf20822f34d5e7c1fdd4e21ae7f3396881096c51036225fb8f84b47afa",
    ),
    (
        "vcruntime140.dll",
        "msvc_runtime-14.44.35112.data/data/Scripts/vcruntime140.dll",
        "d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066",
    ),
    (
        "vcruntime140_1.dll",
        "msvc_runtime-14.44.35112.data/data/Scripts/vcruntime140_1.dll",
        "1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f",
    ),
];

/// `tool:vc-runtime` (Windows x64 only): Christoph Gohlke's `msvc-runtime` wheel from PyPI, which
/// carries Microsoft's Visual C++ 2015-2022 runtime DLLs unmodified (Authenticode-signed by
/// Microsoft, `FileVersion` 14.44.35211.0) and the "Distributable Code for Visual Studio 2022"
/// terms. Only the [`VC_RUNTIME_DLLS`] are extracted (each SHA-256 checked). A model export copies
/// them next to its own Python (`tools/python/...`) when the machine's runtime is older than
/// PyTorch needs; nothing else uses them and nothing installs them system-wide.
pub static VC_RUNTIME: &[Resource] = &[Resource {
    id: VC_RUNTIME_ID,
    kind: ResourceKind::Tool,
    version: VC_RUNTIME_VERSION,
    platform: Some(Platform::WINDOWS_X64),
    parts: &[Part {
        url: "https://files.pythonhosted.org/packages/ce/92/5a10262c2a489d5854f96d69e287923d6f720c4935dd26634deb7a5426e9/msvc_runtime-14.44.35112-cp311-cp311-win_amd64.whl",
        sha256: "aba7fbe71897d25ed53fbb7f391e9f50289378a8a9ae218ba18530c663448391",
        size: 1921911,
        file_name: "msvc_runtime-14.44.35112-cp311-cp311-win_amd64.whl",
        archive: Some(ArchiveKind::Zip),
        layout: Layout::VcRuntime,
    }],
    dest: VC_RUNTIME_DEST,
    provides: Provides::Tool { name: "vc-runtime" },
    title: "Visual C++ runtime (model export)",
    description: "Microsoft Visual C++ 2015-2022 runtime DLLs 14.44 (unmodified); copied next to \
                  the export's own Python only when this machine's runtime is too old for PyTorch",
    license: "Microsoft Visual C++ Redistributable (Distributable Code for Visual Studio 2022), unmodified",
}];

/// `tool:vc-runtime` for `(os, arch)` (Windows x64 only).
pub fn vc_runtime_for(os: &str, arch: &str) -> Option<&'static Resource> {
    VC_RUNTIME.iter().find(|r| r.available_on(os, arch))
}
