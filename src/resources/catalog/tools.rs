//! Tool pins for the model export: Astral's uv per platform ([`TOOLS`]) and the DEIMv2 source
//! archive ([`DEIMV2_SOURCE`]). Never run here; only `super::super::export` runs them, when the
//! user starts an export.

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
