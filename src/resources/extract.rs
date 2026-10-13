//! Whitelist-only extraction of downloaded archives, shared by the download manager and the
//! `setup-*` commands. Dispatches on the catalog [`Layout`] of each part.
//!
//! # Trust model
//!
//! An archive reaches this module only after the download manager checked it against the
//! catalog pin (`https://` URL, exact size, SHA-256; see `super::manager`), or when the user
//! passed a local archive to a `setup-*` command. Its entries are still treated as untrusted
//! data:
//! - only whitelisted file names are written, to destination paths built from our own
//!   whitelist (never taken verbatim from the entry path);
//! - an entry whose path is absolute (`/x`, `C:/x`, `\\x`) or has a `..` component fails the
//!   whole extraction, even when it is not whitelisted; a wanted entry carrying the setuid or
//!   setgid bit fails it too;
//! - symlinks and hard links are only recreated (as same-directory links) when both ends are
//!   whitelisted files in the same directory, and never followed when writing;
//! - decompressed data is counted while streaming (never trusted from headers), against
//!   [`MAX_ENTRY_BYTES`] per file and [`MAX_TOTAL_BYTES`] per archive;
//! - every archive is extracted into a staging directory (`super::manager::run_job`) that is
//!   renamed into place only after every entry passed and the install was checked
//!   ([`finalize`]); a failure removes the staging directory;
//! - nothing from an archive is executed here. The `uv` binary and the DEIMv2 sources are run
//!   only by a model export the user starts explicitly (`super::export`).
//!
//! Layouts:
//! - [`Layout::OpenVino`]: the toolkit's `runtime/...` libraries, layout preserved (what
//!   `openvino-finder` expects under `OPENVINO_INSTALL_DIR`).
//! - [`Layout::OrtRelease`], [`Layout::NugetOrt`], [`Layout::NugetDirectMl`]: ONNX Runtime and
//!   DirectML shared libraries, flat.
//! - [`Layout::NvidiaWheel`]: CUDA/cuDNN shared libraries from `nvidia/<component>/{bin,lib}/`,
//!   flat.
//! - [`Layout::Uv`]: only the `uv` / `uv.exe` binary of a uv release archive (made executable;
//!   it is run only by a model export the user starts, see `super::export`).
//! - [`Layout::SourceTree`]: the DEIMv2 source archive: `engine/**/*.py`, `configs/**/*.yml` and
//!   the license, paths kept (plain path components only). Imported / read only by a DEIMv2
//!   export the user starts.
//! - [`Layout::File`]: not an archive; the manager moves the verified file into place.

use super::catalog::{ArchiveKind, Flavor, Layout, Resource, ResourceKind};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;

/// Target platform of an extraction (the resource's platform, normally this machine's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target<'a> {
    pub os: &'a str,
    pub arch: &'a str,
}

impl Target<'_> {
    fn windows(&self) -> bool {
        self.os == "windows"
    }
}

/// Most decompressed bytes one extracted file may have (4 GiB). The largest real entry is a
/// cuDNN engine library of about 1 GiB.
pub const MAX_ENTRY_BYTES: u64 = 4 << 30;
/// Most decompressed bytes one archive may produce in total (8 GiB). The largest real archive
/// (the cuDNN wheel) extracts to about 1.5 GiB of whitelisted libraries.
pub const MAX_TOTAL_BYTES: u64 = 8 << 30;

/// Decompressed-size ceilings of one extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Per extracted file.
    pub entry_bytes: u64,
    /// Over every extracted file of the archive.
    pub total_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            entry_bytes: MAX_ENTRY_BYTES,
            total_bytes: MAX_TOTAL_BYTES,
        }
    }
}

/// Bytes written so far against [`Limits`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget {
    limits: Limits,
    used: u64,
}

/// An archive exceeded a decompressed-size ceiling.
#[derive(Debug, thiserror::Error)]
#[error("{entry}: extracted data exceeds {limit} bytes ({what}); the archive is refused")]
pub struct OverLimit {
    pub entry: String,
    pub limit: u64,
    pub what: &'static str,
}

/// An archive entry whose path or mode is not acceptable.
#[derive(Debug, thiserror::Error)]
#[error("archive entry {entry:?} is refused: {why}")]
pub struct HostileEntry {
    pub entry: String,
    pub why: &'static str,
}

impl Budget {
    fn new(limits: Limits) -> Self {
        Self { limits, used: 0 }
    }

    /// Bytes one more entry may still write: the smaller of the per-file ceiling and what is
    /// left of the total.
    fn allowance(&self) -> u64 {
        self.limits
            .entry_bytes
            .min(self.limits.total_bytes.saturating_sub(self.used))
    }

    fn over(&self, entry: &str) -> OverLimit {
        if self.limits.total_bytes.saturating_sub(self.used) < self.limits.entry_bytes {
            OverLimit {
                entry: entry.to_string(),
                limit: self.limits.total_bytes,
                what: "archive total",
            }
        } else {
            OverLimit {
                entry: entry.to_string(),
                limit: self.limits.entry_bytes,
                what: "per file",
            }
        }
    }
}

/// Refuse an entry path that is absolute or climbs out of the archive, whatever the
/// whitelist says about it: a real release archive never has one.
fn check_entry_path(entry: &str) -> Result<()> {
    let e = entry.replace('\\', "/");
    let why = if e.starts_with('/') {
        Some("absolute path")
    } else if e
        .split('/')
        .next()
        .is_some_and(|first| first.len() == 2 && first.ends_with(':'))
    {
        Some("absolute path (drive letter)")
    } else if e.split('/').any(|p| p == "..") {
        Some("`..` path component")
    } else {
        None
    };
    match why {
        Some(why) => Err(anyhow::Error::new(HostileEntry {
            entry: entry.to_string(),
            why,
        })),
        None => Ok(()),
    }
}

/// Refuse a wanted file with the setuid or setgid bit. (Entry modes are never applied to
/// written files anyway; `uv` is made 0o755 explicitly. The zip reader strips these bits
/// before they reach us, so this bites on tar headers.)
fn check_entry_mode(entry: &str, mode: Option<u32>) -> Result<()> {
    if mode.is_some_and(|m| m & 0o6000 != 0) {
        return Err(anyhow::Error::new(HostileEntry {
            entry: entry.to_string(),
            why: "setuid/setgid bit set",
        }));
    }
    Ok(())
}

/// Extract the whitelisted entries of one downloaded `archive` (a part with `layout`) into
/// `dest`. Returns the written paths relative to `dest` (`/`-separated).
pub fn extract_part(
    archive: &Path,
    kind: ArchiveKind,
    layout: Layout,
    flavor: Option<Flavor>,
    target: Target<'_>,
    dest: &Path,
) -> Result<BTreeSet<String>> {
    extract_part_with(
        archive,
        kind,
        layout,
        flavor,
        target,
        dest,
        Limits::default(),
    )
}

/// [`extract_part`] with explicit decompressed-size [`Limits`].
pub fn extract_part_with(
    archive: &Path,
    kind: ArchiveKind,
    layout: Layout,
    flavor: Option<Flavor>,
    target: Target<'_>,
    dest: &Path,
    limits: Limits,
) -> Result<BTreeSet<String>> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let windows = target.windows();
    let flavor = flavor.unwrap_or(Flavor::Cpu);
    let mut budget = Budget::new(limits);
    let b = &mut budget;
    match layout {
        Layout::OpenVino => {
            let sel = selection_for(target.os)?;
            let wanted = |e: &str| strip_top(e).and_then(|r| sel.wanted(&r));
            let link = |rel: &str, t: &str, hard: bool| {
                let name = same_dir_link_target(rel, t, hard)?;
                let dir = rel.rsplit_once('/').map_or("", |(d, _)| d);
                Some(format!("{dir}/{name}"))
            };
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted, b),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, link, b),
            }
        }
        Layout::OrtRelease | Layout::NugetOrt | Layout::NugetDirectMl => {
            let wanted = |e: &str| ort_wanted(layout, flavor, windows, e);
            // Flat destination: the link must name a same-directory, whitelisted library.
            let link = |rel: &str, t: &str, hard: bool| {
                let tn = same_dir_link_target(&format!("x/{rel}"), t, hard)?;
                lib_stem(&tn, windows)
                    .is_some_and(|s| stem_allowed(s, flavor))
                    .then_some(tn)
            };
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted, b),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, link, b),
            }
        }
        Layout::NvidiaWheel => {
            let wanted = |e: &str| wheel_wanted(windows, e);
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted, b),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, |_, _, _| None, b),
            }
        }
        Layout::Uv => {
            let wanted = |e: &str| uv_wanted(windows, e);
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted, b),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, |_, _, _| None, b),
            }
        }
        Layout::SourceTree => match kind {
            ArchiveKind::Zip => extract_zip(archive, dest, source_wanted, b),
            ArchiveKind::TarGz => extract_tgz(archive, dest, source_wanted, |_, _, _| None, b),
        },
        Layout::File => anyhow::bail!("{} is not an archive", archive.display()),
    }
}

/// Source archives (`<top>/...`, as GitHub makes them): `engine/**/*.py`, `configs/**/*.yml`
/// (the training configs the model is built from) and the top-level `LICENSE` / `LICENSE.md`,
/// paths kept below the top folder. Every component must be a plain
/// name (ASCII letters, digits, `_`, `-`, `.`; not starting with `.`).
pub fn source_wanted(entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let rest = parts.get(1..)?;
    let plain = |p: &String| {
        !p.starts_with('.')
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    };
    if rest.is_empty() || !rest.iter().all(plain) {
        return None;
    }
    match rest {
        [f] if f == "LICENSE" || f == "LICENSE.md" => Some(f.clone()),
        [first, .., last] if first == "engine" && last.ends_with(".py") => Some(rest.join("/")),
        [first, .., last] if first == "configs" && last.ends_with(".yml") => Some(rest.join("/")),
        _ => None,
    }
}

/// A source archive must have given us the package to import.
fn source_check(got: &BTreeSet<String>) -> Result<Vec<String>> {
    if !got.iter().any(|f| f.starts_with("engine/"))
        || !got.iter().any(|f| f.starts_with("configs/"))
    {
        anyhow::bail!("the source archive has no engine/ package or configs/");
    }
    Ok(Vec::new())
}

/// File name of the uv binary.
pub fn uv_binary_name(windows: bool) -> &'static str {
    if windows { "uv.exe" } else { "uv" }
}

/// uv release archives: `uv-<triple>/uv` (tar.gz) or `uv.exe` at the top (Windows zip). Only the
/// binary itself is kept (not `uvx` / `uvw`).
pub fn uv_wanted(windows: bool, entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let name = uv_binary_name(windows);
    match parts.as_slice() {
        [file] | [_, file] if file == name => Some(name.to_string()),
        _ => None,
    }
}

/// Make the extracted uv binary executable and check it is there.
fn uv_check(windows: bool, dest: &Path, got: &BTreeSet<String>) -> Result<Vec<String>> {
    let name = uv_binary_name(windows);
    if !got.contains(name) {
        anyhow::bail!("the uv archive has no {name}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = dest.join(name);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("making {} executable", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = dest;
    Ok(Vec::new())
}

/// After every part of `resource` is extracted into `dest`: platform fix-ups, sanity checks
/// (errors when the core library is missing, warnings otherwise) and the `VERSION` /
/// `flavor.txt` marker files. Returns the warnings; `got` gains the files written here.
pub fn finalize(
    resource: &Resource,
    version: &str,
    target: Target<'_>,
    dest: &Path,
    got: &mut BTreeSet<String>,
) -> Result<Vec<String>> {
    let warnings = match resource.kind {
        ResourceKind::OpenVinoRuntime => {
            let sel = selection_for(target.os)?;
            mirror_tbb_into_lib_dir(&sel, dest, got)?;
            sel.check(target.arch, got)?
        }
        ResourceKind::OnnxRuntime(flavor) => {
            let w = ort_check(flavor, target.windows(), got)?;
            write_marker(dest, ORT_FLAVOR_FILE, flavor.as_str(), got)?;
            w
        }
        ResourceKind::CudaLibs => wheel_check(target.windows(), dest, got)?,
        ResourceKind::Tool
            if resource
                .parts
                .iter()
                .all(|p| p.layout == Layout::SourceTree) =>
        {
            source_check(got)?
        }
        ResourceKind::Tool => uv_check(target.windows(), dest, got)?,
        ResourceKind::Model | ResourceKind::BenchImages | ResourceKind::ExportModel => Vec::new(),
    };
    if !resource.kind.is_plain_files() {
        write_marker(dest, "VERSION", version, got)?;
    }
    Ok(warnings)
}

fn write_marker(dest: &Path, name: &str, value: &str, got: &mut BTreeSet<String>) -> Result<()> {
    std::fs::write(dest.join(name), format!("{value}\n"))
        .with_context(|| format!("writing {}", dest.join(name).display()))?;
    got.insert(name.to_string());
    Ok(())
}

/// What the user should know after installing `resource`.
pub fn install_note(resource: &Resource) -> Option<&'static str> {
    match resource.kind {
        ResourceKind::OnnxRuntime(Flavor::Cuda) => Some(
            "CUDA 12.x and cuDNN 9 runtimes are not part of this package; install them, or \
             fetch `nvidia-cuda-libs` (large, opt-in), or use the OpenVINO runtime.",
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Strip the archive's single top-level folder (`openvino_toolkit_.../`) and normalise slashes.
/// Returns None for entries with suspicious components.
pub(crate) fn strip_top(entry: &str) -> Option<String> {
    let e = entry.replace('\\', "/");
    let mut parts = e.split('/').filter(|p| !p.is_empty() && *p != ".");
    let first = parts.next()?;
    let rest: Vec<&str> = if first == "runtime" {
        std::iter::once(first).chain(parts).collect()
    } else {
        parts.collect()
    };
    if rest.is_empty() || rest.iter().any(|p| *p == ".." || p.contains(':')) {
        return None;
    }
    Some(rest.join("/"))
}

/// File name of the target of a tar link entry at `rel` (top folder already stripped) when the
/// target lives in the same directory; None otherwise. Symlink targets are relative to the
/// link's directory (`libfoo.so -> libfoo.so.2026.4.0`); hard-link targets are full archive
/// paths (`openvino_toolkit_.../runtime/lib/.../libfoo.2026.4.0.dylib`) and go through
/// [`strip_top`] like regular entries.
pub(crate) fn same_dir_link_target(rel: &str, target: &str, hard: bool) -> Option<String> {
    let (dir, _) = rel.rsplit_once('/')?;
    let t = target.replace('\\', "/");
    let t = t.strip_prefix("./").unwrap_or(&t);
    let name = if !t.contains('/') {
        t.to_string()
    } else if hard {
        let stripped = strip_top(t)?;
        let (tdir, name) = stripped.rsplit_once('/')?;
        if tdir != dir {
            return None;
        }
        name.to_string()
    } else {
        return None;
    };
    if name.is_empty() || name.starts_with('.') || name.contains(':') {
        return None;
    }
    Some(name)
}

/// Write `reader` to `dest/rel` through a `.partial` file, counting the bytes against
/// `budget` while streaming (the archive's own size fields are not trusted). An entry that
/// goes over the allowance is removed and fails with [`OverLimit`].
pub(crate) fn write_entry(
    dest: &Path,
    rel: &str,
    reader: &mut dyn Read,
    budget: &mut Budget,
) -> Result<u64> {
    let out = dest.join(rel);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let tmp = out.with_extension("partial");
    let mut f =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let allowance = budget.allowance();
    // One byte more than allowed: reaching it proves the entry is too large.
    let mut capped = reader.take(allowance.saturating_add(1));
    let n =
        std::io::copy(&mut capped, &mut f).with_context(|| format!("writing {}", out.display()));
    let n = match n {
        Ok(n) if n <= allowance => n,
        Ok(_) => {
            drop(f);
            let _ = std::fs::remove_file(&tmp);
            return Err(anyhow::Error::new(budget.over(rel)));
        }
        Err(e) => {
            drop(f);
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    budget.used = budget.used.saturating_add(n);
    f.flush()?;
    drop(f);
    if out.symlink_metadata().is_ok() {
        std::fs::remove_file(&out).with_context(|| format!("replacing {}", out.display()))?;
    }
    std::fs::rename(&tmp, &out).with_context(|| format!("renaming to {}", out.display()))?;
    Ok(n)
}

/// Split an archive entry name into clean components; None for suspicious ones.
fn components(entry: &str) -> Option<Vec<String>> {
    let e = entry.replace('\\', "/");
    let parts: Vec<String> = e
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .map(str::to_string)
        .collect();
    if parts.is_empty() || parts.iter().any(|p| p == ".." || p.contains(':')) {
        return None;
    }
    Some(parts)
}

/// Zip driver: regular files whose name `wanted` maps to a destination path. Symlink entries
/// are never recreated (no zip in the catalog has any); directories are implied by the files.
fn extract_zip(
    archive: &Path,
    dest: &Path,
    wanted: impl Fn(&str) -> Option<String>,
    budget: &mut Budget,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading zip {}", archive.display()))?;
    let mut got = BTreeSet::new();
    for i in 0..zip.len() {
        // Filter on the central-directory metadata; only wanted entries are opened (seek,
        // local header, decompressor).
        let (name, rel) = {
            let meta = zip
                .by_index_data(i)
                .with_context(|| format!("reading zip entry #{i} of {}", archive.display()))?;
            let name = meta.name().with_context(|| {
                format!(
                    "reading the name of zip entry #{i} of {}",
                    archive.display()
                )
            })?;
            check_entry_path(&name)?;
            if meta.is_dir() || meta.is_symlink() {
                continue;
            }
            let Some(rel) = wanted(&name) else {
                continue;
            };
            check_entry_mode(&name, meta.unix_mode())?;
            (name.into_owned(), rel)
        };
        let mut entry = zip
            .by_index(i)
            .with_context(|| format!("opening zip entry #{i} ({name}) of {}", archive.display()))?;
        write_entry(dest, &rel, &mut entry, budget)
            .with_context(|| format!("extracting {name}"))?;
        got.insert(rel);
    }
    Ok(got)
}

/// Tar.gz driver: regular files mapped by `wanted`; links are recreated when `link(rel,
/// target, hard)` maps them to a same-directory destination path that was extracted.
fn extract_tgz(
    archive: &Path,
    dest: &Path,
    wanted: impl Fn(&str) -> Option<String>,
    link: impl Fn(&str, &str, bool) -> Option<String>,
    budget: &mut Budget,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut tar = tar::Archive::new(gz);
    let mut got = BTreeSet::new();
    // (link rel path, target rel path in the same directory)
    let mut links: Vec<(String, String)> = Vec::new();
    for entry in tar
        .entries()
        .with_context(|| format!("reading tar {}", archive.display()))?
    {
        let mut entry =
            entry.with_context(|| format!("reading tar entry in {}", archive.display()))?;
        // `Entry::path` already refuses `..` and absolute paths on GNU/ustar headers; the
        // raw bytes are checked too so a PAX `path` record cannot sidestep that.
        let raw = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        check_entry_path(&raw)?;
        let path = entry.path()?.to_string_lossy().into_owned();
        check_entry_path(&path)?;
        let Some(rel) = wanted(&path) else {
            continue;
        };
        let ty = entry.header().entry_type();
        if ty.is_symlink() || ty.is_hard_link() {
            if let Some(target) = entry.link_name()? {
                let t = target.to_string_lossy().into_owned();
                match link(&rel, &t, ty.is_hard_link()) {
                    Some(target_rel) => links.push((rel, target_rel)),
                    None => tracing::debug!("skipping link {path} -> {t}"),
                }
            }
            continue;
        }
        if !ty.is_file() {
            continue;
        }
        check_entry_mode(&path, entry.header().mode().ok())?;
        write_entry(dest, &rel, &mut entry, budget)
            .with_context(|| format!("extracting {path}"))?;
        got.insert(rel);
    }
    // Resolve link chains (a -> b -> real file) by repeated passes.
    for _ in 0..4 {
        let mut progressed = false;
        for (rel, target_rel) in &links {
            if got.contains(rel) || !got.contains(target_rel) {
                continue;
            }
            let link_path = dest.join(rel);
            if link_path.symlink_metadata().is_ok() {
                std::fs::remove_file(&link_path).ok();
            }
            #[cfg(unix)]
            {
                let name = target_rel.rsplit('/').next().unwrap_or(target_rel);
                std::os::unix::fs::symlink(name, &link_path)
                    .with_context(|| format!("creating symlink {}", link_path.display()))?;
            }
            #[cfg(not(unix))]
            std::fs::copy(dest.join(target_rel), &link_path)
                .with_context(|| format!("copying {}", link_path.display()))?;
            got.insert(rel.clone());
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    Ok(got)
}

// ---------------------------------------------------------------------------------------------
// OpenVINO
// ---------------------------------------------------------------------------------------------

/// Which files of the archive make up the runtime on one OS.
pub(crate) struct Selection {
    os: &'static str,
    /// Directory of the OpenVINO libs inside the archive (relative to the top folder).
    lib_dirs: &'static [&'static str],
    tbb_dir: &'static str,
}

/// Libraries copied on every OS (base names without prefix/suffix) when the archive has them.
/// Plugins on macOS use a `.so` suffix while the other libs are `.dylib`; both are matched.
const CORE_LIBS: &[&str] = &[
    "openvino",
    "openvino_c",
    CPU_PLUGIN_X86,
    CPU_PLUGIN_ARM,
    GPU_PLUGIN,
    "openvino_auto_plugin",
    "openvino_hetero_plugin",
    "openvino_ir_frontend",
    "openvino_onnx_frontend",
];
/// x86 CPU plugin (Windows, Linux x86_64).
const CPU_PLUGIN_X86: &str = "openvino_intel_cpu_plugin";
/// ARM CPU plugin (macOS arm64, Linux aarch64).
const CPU_PLUGIN_ARM: &str = "openvino_arm_cpu_plugin";
/// Intel GPU plugin (only shipped for Windows and Linux x86_64).
pub(crate) const GPU_PLUGIN: &str = "openvino_intel_gpu_plugin";
/// Libraries whose absence only deserves a warning (the CPU and GPU plugins are checked
/// separately because which ones exist depends on the platform).
const EXPECTED_LIBS: &[&str] = &[
    "openvino",
    "openvino_auto_plugin",
    "openvino_hetero_plugin",
    "openvino_ir_frontend",
    "openvino_onnx_frontend",
];
/// Plugin registry files. Shipped in the Windows archive; 2026.x macOS/Linux archives do not
/// have them (OpenVINO then discovers plugins by file name), so they are optional there.
const EXTRA_FILES: &[&str] = &["plugins.xml", "cache.json"];
const WINDOWS_TBB: &[&str] = &["tbb12.dll", "tbbbind_2_5.dll"];

pub(crate) fn selection_for(os: &str) -> Result<Selection> {
    Ok(match os {
        "windows" => Selection {
            os: "windows",
            lib_dirs: &["runtime/bin/intel64/Release"],
            tbb_dir: "runtime/3rdparty/tbb/bin",
        },
        "linux" => Selection {
            os: "linux",
            lib_dirs: &[
                "runtime/lib/intel64",
                "runtime/lib/aarch64",
                "runtime/lib/arm64",
            ],
            tbb_dir: "runtime/3rdparty/tbb/lib",
        },
        "macos" => Selection {
            os: "macos",
            lib_dirs: &["runtime/lib/arm64/Release", "runtime/lib/intel64/Release"],
            tbb_dir: "runtime/3rdparty/tbb/lib",
        },
        other => anyhow::bail!("unsupported OS '{other}' for OpenVINO runtime setup"),
    })
}

/// Base library name of a unix shared-library file name: `libopenvino_c.so.2026.4.0` ->
/// `openvino_c`, `libopenvino.2026.4.0.dylib` -> `openvino`. None if not a shared library.
pub(crate) fn unix_lib_base(name: &str) -> Option<&str> {
    let is_lib = name.contains(".so") || name.ends_with(".dylib");
    if !is_lib {
        return None;
    }
    let stem = name.strip_prefix("lib")?;
    stem.split('.').next()
}

impl Selection {
    /// Decide whether `rel` (path inside the archive with the top folder stripped, `/`-separated)
    /// belongs to the runtime subset. Returns the (validated) relative destination path.
    pub(crate) fn wanted(&self, rel: &str) -> Option<String> {
        let (dir, name) = rel.rsplit_once('/')?;
        if name.is_empty() || name.contains('\\') {
            return None;
        }
        if self.lib_dirs.contains(&dir) {
            let keep = if self.os == "windows" {
                EXTRA_FILES.contains(&name)
                    || name
                        .strip_suffix(".dll")
                        .is_some_and(|b| CORE_LIBS.contains(&b))
            } else {
                EXTRA_FILES.contains(&name)
                    || unix_lib_base(name).is_some_and(|b| CORE_LIBS.contains(&b))
            };
            return keep.then(|| format!("{dir}/{name}"));
        }
        if dir == self.tbb_dir {
            let keep = if self.os == "windows" {
                WINDOWS_TBB.contains(&name)
            } else {
                // libhwloc is a dependency of libtbbbind.
                (name.starts_with("libtbb") || name.starts_with("libhwloc"))
                    && !name.contains("debug")
                    && unix_lib_base(name).is_some()
            };
            return keep.then(|| format!("{dir}/{name}"));
        }
        None
    }

    /// Warn about missing expected files; fail if `openvino_c` is missing. Returns the
    /// warnings (also logged) so tests can inspect them.
    pub(crate) fn check(&self, arch: &str, got: &BTreeSet<String>) -> Result<Vec<String>> {
        let names: Vec<&str> = got
            .iter()
            .filter_map(|p| p.rsplit_once('/').map(|(_, n)| n))
            .collect();
        let has_lib = |base: &str| {
            names.iter().any(|n| {
                if self.os == "windows" {
                    n.strip_suffix(".dll") == Some(base)
                } else {
                    unix_lib_base(n) == Some(base)
                }
            })
        };
        if !has_lib("openvino_c") {
            anyhow::bail!(
                "the archive does not contain the openvino_c library under {:?}",
                self.lib_dirs
            );
        }
        let mut warnings = Vec::new();
        for base in EXPECTED_LIBS {
            if !has_lib(base) {
                warnings.push(format!("OpenVINO archive is missing library '{base}'"));
            }
        }
        if !has_lib(CPU_PLUGIN_X86) && !has_lib(CPU_PLUGIN_ARM) {
            warnings.push(format!(
                "OpenVINO archive has no CPU plugin ('{CPU_PLUGIN_X86}' or '{CPU_PLUGIN_ARM}'); \
                 CPU inference will not work"
            ));
        }
        let gpu_expected = matches!((self.os, arch), ("windows", "x86_64") | ("linux", "x86_64"));
        if gpu_expected && !has_lib(GPU_PLUGIN) {
            warnings.push(format!(
                "OpenVINO archive is missing library '{GPU_PLUGIN}'; GPU inference will not work"
            ));
        }
        let tbb: &[&str] = if self.os == "windows" {
            WINDOWS_TBB
        } else {
            &[]
        };
        for f in tbb {
            if !names.contains(f) {
                warnings.push(format!("OpenVINO archive is missing '{f}' (continuing)"));
            }
        }
        for f in EXTRA_FILES {
            if !names.contains(f) {
                if self.os == "windows" {
                    warnings.push(format!("OpenVINO archive is missing '{f}' (continuing)"));
                } else {
                    tracing::debug!("OpenVINO archive has no '{f}' (not needed on {})", self.os);
                }
            }
        }
        if self.os != "windows" && !names.iter().any(|n| n.starts_with("libtbb")) {
            warnings.push(
                "OpenVINO archive has no bundled TBB; the system libtbb will be used".to_string(),
            );
        }
        for w in &warnings {
            tracing::warn!("{w}");
        }
        Ok(warnings)
    }
}

/// On macOS, `(source rel, destination rel)` pairs that copy the bundled TBB/hwloc libraries
/// next to `libopenvino_c` (empty elsewhere).
///
/// `libopenvino.dylib` and the plugins reference `@rpath/libtbb.12.dylib`; the only LC_RPATH of
/// libopenvino is `@loader_path/../../../3rdparty/tbb` (not the `tbb/lib` folder the archive
/// actually uses), while `libopenvino_c` has `@loader_path/`. Having TBB in the lib dir makes
/// `@rpath` resolve through `libopenvino_c` (which is what we load first).
fn tbb_mirror_plan(sel: &Selection, got: &BTreeSet<String>) -> Vec<(String, String)> {
    if sel.os != "macos" {
        return Vec::new();
    }
    let Some(lib_dir) = got.iter().find_map(|rel| {
        let (dir, name) = rel.rsplit_once('/')?;
        (sel.lib_dirs.contains(&dir) && unix_lib_base(name) == Some("openvino_c")).then_some(dir)
    }) else {
        return Vec::new();
    };
    let prefix = format!("{}/", sel.tbb_dir);
    got.iter()
        .filter_map(|rel| {
            let name = rel.strip_prefix(&prefix)?;
            (!name.contains('/')).then(|| (rel.clone(), format!("{lib_dir}/{name}")))
        })
        .collect()
}

/// Apply [`tbb_mirror_plan`]: regular files are copied, symlinks recreated with the same
/// (same-directory) target.
fn mirror_tbb_into_lib_dir(sel: &Selection, dest: &Path, got: &mut BTreeSet<String>) -> Result<()> {
    let plan = tbb_mirror_plan(sel, got);
    // Regular files first so symlinks never dangle, even transiently.
    let is_link = |rel: &str| {
        dest.join(rel)
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
    };
    let (links, files): (Vec<_>, Vec<_>) = plan.into_iter().partition(|(src, _)| is_link(src));
    for (src, dst) in files.iter().chain(links.iter()) {
        let from = dest.join(src);
        let to = dest.join(dst);
        if to.symlink_metadata().is_ok() {
            std::fs::remove_file(&to).with_context(|| format!("replacing {}", to.display()))?;
        }
        let link_target = std::fs::read_link(&from).ok();
        match link_target {
            #[cfg(unix)]
            Some(target) => std::os::unix::fs::symlink(&target, &to)
                .with_context(|| format!("creating symlink {}", to.display()))?,
            _ => {
                std::fs::copy(&from, &to)
                    .with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
            }
        }
        got.insert(dst.clone());
    }
    if !files.is_empty() {
        tracing::debug!(
            "installed {} TBB libraries next to libopenvino for @rpath resolution",
            files.len() + links.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime / DirectML
// ---------------------------------------------------------------------------------------------

/// File naming the flavor in an ONNX Runtime install directory.
pub const ORT_FLAVOR_FILE: &str = "flavor.txt";

/// Library stem of a file name (`onnxruntime.dll` -> `onnxruntime`,
/// `libonnxruntime.so.1.24.4` -> `onnxruntime`, `libonnxruntime.1.24.4.dylib` -> `onnxruntime`).
/// None when the name is not a shared library of the target OS.
pub(crate) fn lib_stem(name: &str, windows: bool) -> Option<&str> {
    if windows {
        return name.strip_suffix(".dll");
    }
    if !(name.contains(".so") || name.ends_with(".dylib")) {
        return None;
    }
    name.strip_prefix("lib")?.split('.').next()
}

/// Libraries every flavor needs.
const CORE_STEMS: &[&str] = &["onnxruntime", "onnxruntime_providers_shared"];
/// Provider libraries of the CUDA/TensorRT package.
const GPU_STEMS: &[&str] = &[
    "onnxruntime_providers_cuda",
    "onnxruntime_providers_tensorrt",
];
const DIRECTML_DLL: &str = "DirectML.dll";

fn stem_allowed(stem: &str, flavor: Flavor) -> bool {
    CORE_STEMS.contains(&stem) || (flavor == Flavor::Cuda && GPU_STEMS.contains(&stem))
}

/// Decide whether an ONNX Runtime / DirectML archive entry belongs to the runtime subset.
/// Returns the flat destination file name (the entry's last component, itself whitelisted).
pub fn ort_wanted(layout: Layout, flavor: Flavor, windows: bool, entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let (name, dirs) = parts.split_last()?;
    let dirs: Vec<&str> = dirs.iter().map(String::as_str).collect();
    match layout {
        Layout::OrtRelease => {
            // `<top>/lib/<file>` (some Windows packages use `bin`).
            if dirs.len() != 2 || !matches!(dirs.get(1), Some(&("lib" | "bin"))) {
                return None;
            }
            let stem = lib_stem(name, windows)?;
            stem_allowed(stem, flavor).then(|| name.clone())
        }
        Layout::NugetOrt => {
            if dirs != ["runtimes", "win-x64", "native"] {
                return None;
            }
            let stem = lib_stem(name, true)?;
            stem_allowed(stem, flavor).then(|| name.clone())
        }
        Layout::NugetDirectMl => {
            (dirs == ["bin", "x64-win"] && name == DIRECTML_DLL).then(|| name.clone())
        }
        Layout::OpenVino | Layout::NvidiaWheel | Layout::Uv | Layout::SourceTree | Layout::File => {
            None
        }
    }
}

/// True for a file a previous ONNX Runtime install may have left in a directory (removed before
/// installing into a shared directory so flavors never mix).
pub(crate) fn is_ort_file(name: &str, windows: bool) -> bool {
    name == ORT_FLAVOR_FILE
        || name == "VERSION"
        || name == DIRECTML_DLL
        || lib_stem(name, windows)
            .is_some_and(|s| CORE_STEMS.contains(&s) || GPU_STEMS.contains(&s))
}

/// Warn about missing expected files; fail when the core library is missing.
pub(crate) fn ort_check(
    flavor: Flavor,
    windows: bool,
    got: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let has = |stem: &str| got.iter().any(|n| lib_stem(n, windows) == Some(stem));
    if !has("onnxruntime") {
        anyhow::bail!("the archive does not contain the onnxruntime shared library");
    }
    let mut warnings = Vec::new();
    if flavor == Flavor::Cuda {
        for stem in GPU_STEMS {
            if !has(stem) {
                warnings.push(format!("ONNX Runtime package is missing '{stem}'"));
            }
        }
    }
    if flavor == Flavor::DirectMl && !got.contains(DIRECTML_DLL) {
        warnings.push(format!("DirectML package is missing '{DIRECTML_DLL}'"));
    }
    for w in &warnings {
        tracing::warn!("{w}");
    }
    Ok(warnings)
}

// ---------------------------------------------------------------------------------------------
// NVIDIA wheels
// ---------------------------------------------------------------------------------------------

/// `nvidia/<component>/bin/<x>.dll` (Windows) or `nvidia/<component>/lib/lib<x>.so[.N]`
/// (Linux) -> the flat file name.
pub fn wheel_wanted(windows: bool, entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let (name, dirs) = parts.split_last()?;
    let [top, component, sub] = dirs else {
        return None;
    };
    let component_ok = !component.is_empty()
        && component
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if top != "nvidia" || !component_ok {
        return None;
    }
    let is_lib = if windows {
        sub == "bin" && name.to_ascii_lowercase().ends_with(".dll")
    } else {
        sub == "lib" && name.starts_with("lib") && name.contains(".so")
    };
    (is_lib && !name.starts_with('.')).then(|| name.clone())
}

/// Exact library names ONNX Runtime 1.24's CUDA provider links (`DT_NEEDED` / imports, checked
/// on `libonnxruntime_providers_cuda.so`): they must exist after extraction.
pub fn cuda_provider_needs(windows: bool) -> &'static [&'static str] {
    if windows {
        &[
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudnn64_9.dll",
            "cufft64_11.dll",
            "curand64_10.dll",
        ]
    } else {
        &[
            "libcudart.so.12",
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn.so.9",
            "libcufft.so.11",
            "libcurand.so.10",
        ]
    }
}

/// Soname links missing among extracted Linux libraries: `libfoo.so.12.8.90` without
/// `libfoo.so.12` gives `("libfoo.so.12", "libfoo.so.12.8.90")` (the shortest longer name wins
/// when there are several). The NVIDIA wheels normally ship the soname itself.
pub fn soname_links(names: &BTreeSet<String>) -> Vec<(String, String)> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for n in names {
        let Some((base, ver)) = n.split_once(".so.") else {
            continue;
        };
        let mut parts = ver.split('.');
        let Some(major) = parts
            .next()
            .filter(|m| m.chars().all(|c| c.is_ascii_digit()))
        else {
            continue;
        };
        if parts.next().is_none() || !base.starts_with("lib") || base.contains('/') {
            continue;
        }
        let link = format!("{base}.so.{major}");
        if names.contains(&link) {
            continue;
        }
        match out.get(&link) {
            Some(t) if t.len() <= n.len() => {}
            _ => {
                out.insert(link, n.clone());
            }
        }
    }
    out.into_iter().collect()
}

/// Create the [`soname_links`] in `dest` (symlinks on Unix, copies elsewhere).
fn add_soname_links(dest: &Path, got: &mut BTreeSet<String>) -> Result<()> {
    for (link, target) in soname_links(got) {
        let path = dest.join(&link);
        if path.symlink_metadata().is_ok() {
            std::fs::remove_file(&path).ok();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &path)
            .with_context(|| format!("creating symlink {}", path.display()))?;
        #[cfg(not(unix))]
        std::fs::copy(dest.join(&target), &path)
            .with_context(|| format!("copying {}", path.display()))?;
        got.insert(link);
    }
    Ok(())
}

/// Linux: add soname links; then every library the CUDA provider links must be there.
fn wheel_check(windows: bool, dest: &Path, got: &mut BTreeSet<String>) -> Result<Vec<String>> {
    if !windows {
        add_soname_links(dest, got)?;
    }
    let missing: Vec<&str> = cuda_provider_needs(windows)
        .iter()
        .copied()
        .filter(|n| !got.contains(*n))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("the CUDA wheels are missing {}", missing.join(", "));
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the OS temp dir, removed on drop (also when a test panics).
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("bop-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl std::ops::Deref for TempDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl AsRef<Path> for TempDir {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn windows_selection() {
        let s = selection_for("windows").unwrap();
        let top = "openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64/";
        let w = |p: &str| strip_top(&format!("{top}{p}")).and_then(|r| s.wanted(&r));
        assert_eq!(
            w("runtime/bin/intel64/Release/openvino_c.dll").as_deref(),
            Some("runtime/bin/intel64/Release/openvino_c.dll")
        );
        assert!(w("runtime/bin/intel64/Release/cache.json").is_some());
        assert!(w("runtime/3rdparty/tbb/bin/tbb12.dll").is_some());
        assert!(w("runtime/3rdparty/tbb/bin/tbb12_debug.dll").is_none());
        assert!(w("runtime/bin/intel64/Debug/openvino_cd.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/openvino_intel_npu_plugin.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/openvino_pytorch_frontend.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/../../../evil.dll").is_none());
    }

    #[test]
    fn unix_selection() {
        let s = selection_for("linux").unwrap();
        let w = |p: &str| strip_top(&format!("top/{p}")).and_then(|r| s.wanted(&r));
        assert!(w("runtime/lib/intel64/libopenvino_c.so.2026.4.0").is_some());
        assert!(w("runtime/lib/intel64/libopenvino_c.so").is_some());
        assert!(w("runtime/lib/intel64/libopenvino_intel_npu_plugin.so").is_none());
        assert!(w("runtime/3rdparty/tbb/lib/libtbb.so.12").is_some());
        assert!(w("runtime/3rdparty/tbb/lib/libtbb_debug.so.12").is_none());
        let m = selection_for("macos").unwrap();
        assert!(
            strip_top("top/runtime/lib/arm64/Release/libopenvino.2026.4.0.dylib")
                .and_then(|r| m.wanted(&r))
                .is_some()
        );
        assert_eq!(
            unix_lib_base("libopenvino.2026.4.0.dylib"),
            Some("openvino")
        );
        assert_eq!(unix_lib_base("libopenvino_c.so.2640"), Some("openvino_c"));
        // macOS plugins are `.so`, the ARM CPU plugin is selected; hwloc (tbbbind dep) too.
        let mw = |p: &str| strip_top(&format!("top/{p}")).and_then(|r| m.wanted(&r));
        assert!(mw("runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so").is_some());
        assert!(mw("runtime/lib/arm64/Release/libopenvino_auto_batch_plugin.so").is_none());
        assert!(mw("runtime/lib/arm64/Release/libopenvino_pytorch_frontend.2640.dylib").is_none());
        assert!(mw("runtime/3rdparty/tbb/lib/libhwloc.15.dylib").is_some());
        assert!(mw("runtime/3rdparty/tbb/lib/pkgconfig/tbb.pc").is_none());
        assert!(w("runtime/lib/aarch64/libopenvino_arm_cpu_plugin.so").is_some());
    }

    #[test]
    fn link_targets() {
        let rel = "runtime/lib/arm64/Release/libopenvino.2640.dylib";
        // Symlinks: bare same-dir names only.
        assert_eq!(
            same_dir_link_target(rel, "libopenvino.2026.4.0.dylib", false).as_deref(),
            Some("libopenvino.2026.4.0.dylib")
        );
        assert_eq!(same_dir_link_target(rel, "../x.dylib", false), None);
        assert_eq!(same_dir_link_target(rel, "/usr/lib/x.dylib", false), None);
        // Hard links: full archive path, mapped through strip_top and required in the same dir.
        assert_eq!(
            same_dir_link_target(
                rel,
                "openvino_toolkit_macos_12_6_2026.4.0_arm64/runtime/lib/arm64/Release/libopenvino.2026.4.0.dylib",
                true
            )
            .as_deref(),
            Some("libopenvino.2026.4.0.dylib")
        );
        assert_eq!(
            same_dir_link_target(rel, "top/runtime/3rdparty/tbb/lib/libtbb.12.dylib", true),
            None
        );
        assert_eq!(
            same_dir_link_target(rel, "top/runtime/lib/arm64/Release/../../../../evil", true),
            None
        );
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn check_warnings_per_platform() {
        let mac = set(&[
            "runtime/lib/arm64/Release/libopenvino.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_c.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_auto_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_hetero_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_ir_frontend.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_onnx_frontend.2640.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.dylib",
        ]);
        let m = selection_for("macos").unwrap();
        assert!(m.check("aarch64", &mac).unwrap().is_empty());
        // Without any CPU plugin there is a warning.
        let mut no_cpu = mac.clone();
        no_cpu.remove("runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so");
        let w = m.check("aarch64", &no_cpu).unwrap();
        assert!(w.len() == 1 && w[0].contains("CPU plugin"), "{w:?}");
        // openvino_c is mandatory.
        let mut no_c = mac.clone();
        no_c.remove("runtime/lib/arm64/Release/libopenvino_c.2640.dylib");
        assert!(m.check("aarch64", &no_c).is_err());

        // Linux x86_64 expects the GPU plugin; aarch64 does not.
        let lin = set(&[
            "runtime/lib/intel64/libopenvino.so.2640",
            "runtime/lib/intel64/libopenvino_c.so.2640",
            "runtime/lib/intel64/libopenvino_intel_cpu_plugin.so",
            "runtime/lib/intel64/libopenvino_auto_plugin.so",
            "runtime/lib/intel64/libopenvino_hetero_plugin.so",
            "runtime/lib/intel64/libopenvino_ir_frontend.so.2640",
            "runtime/lib/intel64/libopenvino_onnx_frontend.so.2640",
            "runtime/3rdparty/tbb/lib/libtbb.so.12",
        ]);
        let l = selection_for("linux").unwrap();
        let w = l.check("x86_64", &lin).unwrap();
        assert!(w.len() == 1 && w[0].contains(GPU_PLUGIN), "{w:?}");
        assert!(l.check("aarch64", &lin).unwrap().is_empty());
    }

    #[test]
    fn macos_tbb_is_mirrored_next_to_libopenvino() {
        let got = set(&[
            "runtime/lib/arm64/Release/libopenvino_c.2640.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.13.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.dylib",
        ]);
        let plan = tbb_mirror_plan(&selection_for("macos").unwrap(), &got);
        assert_eq!(
            plan,
            vec![
                (
                    "runtime/3rdparty/tbb/lib/libtbb.12.13.dylib".to_string(),
                    "runtime/lib/arm64/Release/libtbb.12.13.dylib".to_string()
                ),
                (
                    "runtime/3rdparty/tbb/lib/libtbb.12.dylib".to_string(),
                    "runtime/lib/arm64/Release/libtbb.12.dylib".to_string()
                ),
            ]
        );
        assert!(tbb_mirror_plan(&selection_for("linux").unwrap(), &got).is_empty());
    }

    #[test]
    fn ort_release_whitelist_unix() {
        let w = |f, e: &str| ort_wanted(Layout::OrtRelease, f, false, e);
        let top = "onnxruntime-linux-x64-gpu-1.24.4";
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so.1.24.4")).as_deref(),
            Some("libonnxruntime.so.1.24.4")
        );
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so")).as_deref(),
            Some("libonnxruntime.so")
        );
        for p in ["cuda", "tensorrt", "shared"] {
            let e = format!("{top}/lib/libonnxruntime_providers_{p}.so");
            assert!(w(Flavor::Cuda, &e).is_some(), "{e}");
        }
        // Providers only for the cuda flavor.
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/libonnxruntime_providers_cuda.so")
            )
            .is_none()
        );
        // Not libraries / wrong places.
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/pkgconfig/libonnxruntime.pc")
            )
            .is_none()
        );
        assert!(w(Flavor::Cpu, &format!("{top}/include/onnxruntime_c_api.h")).is_none());
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/testdata/libcustom_op_library.so")
            )
            .is_none()
        );
        assert!(w(Flavor::Cpu, "lib/libonnxruntime.so").is_none());
        assert!(w(Flavor::Cpu, &format!("{top}/lib/../../libonnxruntime.so")).is_none());
        assert!(w(Flavor::Cpu, &format!("{top}/lib/libcustom.so")).is_none());
    }

    #[test]
    fn ort_release_whitelist_mac_and_windows() {
        let top = "onnxruntime-osx-arm64-1.24.4";
        let m = |e: &str| ort_wanted(Layout::OrtRelease, Flavor::CoreMl, false, e);
        assert!(m(&format!("{top}/lib/libonnxruntime.1.24.4.dylib")).is_some());
        assert!(m(&format!("{top}/lib/libonnxruntime.dylib")).is_some());
        assert!(
            m(&format!(
                "{top}/lib/libonnxruntime.1.24.4.dylib.dSYM/Contents/Info.plist"
            ))
            .is_none()
        );
        let w = |f, e: &str| ort_wanted(Layout::OrtRelease, f, true, e);
        let top = "onnxruntime-win-x64-gpu-1.24.4";
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}\\lib\\onnxruntime.dll")).as_deref(),
            Some("onnxruntime.dll")
        );
        assert!(
            w(
                Flavor::Cuda,
                &format!("{top}/lib/onnxruntime_providers_cuda.dll")
            )
            .is_some()
        );
        assert!(w(Flavor::Cuda, &format!("{top}/lib/onnxruntime.lib")).is_none());
        assert!(w(Flavor::Cuda, &format!("{top}/lib/onnxruntime.pdb")).is_none());
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/onnxruntime_providers_cuda.dll")
            )
            .is_none()
        );
    }

    #[test]
    fn nuget_whitelist() {
        let o = |e: &str| ort_wanted(Layout::NugetOrt, Flavor::DirectMl, true, e);
        assert!(o("runtimes/win-x64/native/onnxruntime.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime_providers_shared.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime.lib").is_none());
        assert!(o("runtimes/win-arm64/native/onnxruntime.dll").is_none());
        let d = |e: &str| ort_wanted(Layout::NugetDirectMl, Flavor::DirectMl, true, e);
        assert_eq!(
            d("bin/x64-win/DirectML.dll").as_deref(),
            Some("DirectML.dll")
        );
        assert!(d("bin/x64-win/DirectML.Debug.dll").is_none());
        assert!(d("bin/arm64-win/DirectML.dll").is_none());
        assert!(d("../bin/x64-win/DirectML.dll").is_none());
    }

    #[test]
    fn ort_check_requires_core_library() {
        assert!(
            ort_check(Flavor::Cpu, false, &set(&["libonnxruntime.so.1"]))
                .unwrap()
                .is_empty()
        );
        assert!(ort_check(Flavor::Cpu, false, &set(&[])).is_err());
        let w = ort_check(Flavor::Cuda, false, &set(&["libonnxruntime.so.1"])).unwrap();
        assert_eq!(w.len(), 2);
        let w = ort_check(Flavor::DirectMl, true, &set(&["onnxruntime.dll"])).unwrap();
        assert_eq!(w.len(), 1);
        assert!(is_ort_file("libonnxruntime_providers_cuda.so", false));
        assert!(!is_ort_file("libcudnn.so.9", false));
    }

    #[test]
    fn nvidia_wheel_whitelist() {
        let w = |e: &str| wheel_wanted(true, e);
        assert_eq!(
            w("nvidia/cudnn/bin/cudnn64_9.dll").as_deref(),
            Some("cudnn64_9.dll")
        );
        assert!(w("nvidia/cublas/bin/cublasLt64_12.dll").is_some());
        assert!(w("nvidia/cudnn/include/cudnn.h").is_none());
        assert!(w("nvidia/cudnn/lib/x64/cudnn.lib").is_none());
        assert!(w("nvidia_cudnn_cu12-9.8.0.87.dist-info/RECORD").is_none());
        assert!(w("nvidia/../bin/evil.dll").is_none());
        let l = |e: &str| wheel_wanted(false, e);
        assert_eq!(
            l("nvidia/cudnn/lib/libcudnn.so.9").as_deref(),
            Some("libcudnn.so.9")
        );
        assert!(l("nvidia/cuda_runtime/lib/libcudart.so.12").is_some());
        assert!(l("nvidia/cuda_runtime/lib/libcudart_static.a").is_none());
        assert!(l("nvidia/cudnn/bin/libcudnn.so.9").is_none());
        let dir = TempDir::new("wheelchk");
        let mut got = set(&["libcudart.so.12", "libcublas.so.12"]);
        let err = wheel_check(false, &dir, &mut got).unwrap_err();
        assert!(format!("{err}").contains("libcudnn.so.9"), "{err}");
        let mut got: BTreeSet<String> = cuda_provider_needs(true)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(wheel_check(true, &dir, &mut got).is_ok());
    }

    #[test]
    fn soname_links_fill_in_missing_names() {
        let names = set(&[
            "libcudart.so.12.8.90",
            "libcudnn.so.9",
            "libcudnn.so.9.8.0",
            "libnvrtc.so.12.8.93",
            "libnvrtc.so.12.8",
            "libweird.so.x.1",
            "cudart64_12.dll",
        ]);
        assert_eq!(
            soname_links(&names),
            [
                (
                    "libcudart.so.12".to_string(),
                    "libcudart.so.12.8.90".to_string()
                ),
                ("libnvrtc.so.12".to_string(), "libnvrtc.so.12.8".to_string()),
            ]
        );
        assert!(soname_links(&set(&["libcudnn.so.9"])).is_empty());
    }

    /// Write a zip (a wheel) with `entries` (name, bytes).
    fn zip_with(path: &Path, entries: &[(&str, &[u8])]) {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        for (name, data) in entries {
            z.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            z.write_all(data).unwrap();
        }
        z.finish().unwrap();
    }

    /// Only wanted regular files are opened and extracted; directory entries and unwanted
    /// files are skipped on their metadata.
    #[test]
    fn zip_extracts_only_wanted_files() {
        let dir = TempDir::new("zip-filter");
        let archive = dir.join("a.zip");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let mut z = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default();
            z.add_directory("lib/", opts).unwrap();
            z.start_file("lib/skip.txt", opts).unwrap();
            z.write_all(b"skip").unwrap();
            z.start_file("lib/keep.bin", opts).unwrap();
            z.write_all(b"keep").unwrap();
            z.finish().unwrap();
        }
        let dest = dir.join("out");
        std::fs::create_dir_all(&dest).unwrap();
        let mut budget = Budget::new(Limits::default());
        let got = extract_zip(
            &archive,
            &dest,
            |name| {
                assert!(!name.ends_with('/'), "directory entry offered: {name}");
                name.strip_prefix("lib/")
                    .filter(|n| n.ends_with(".bin"))
                    .map(str::to_string)
            },
            &mut budget,
        )
        .unwrap();
        assert_eq!(budget.used, 4);
        assert_eq!(got, BTreeSet::from(["keep.bin".to_string()]));
        assert_eq!(std::fs::read(dest.join("keep.bin")).unwrap(), b"keep");
        assert!(!dest.join("skip.txt").exists());
        assert!(!dest.join("lib").exists());
    }

    /// Synthetic Linux wheels: only shared libraries land (flat), headers / static libs /
    /// metadata do not, and a missing soname gets its link.
    #[test]
    fn linux_wheels_extract_flat_with_soname_links() {
        let dir = TempDir::new("wheel-l");
        let w1 = dir.join("rt.whl");
        zip_with(
            &w1,
            &[
                ("nvidia/__init__.py", b""),
                ("nvidia/cuda_runtime/include/cuda.h", b"h"),
                ("nvidia/cuda_runtime/lib/libcudart.so.12.8.90", b"cudart"),
                ("nvidia/cuda_runtime/lib/libcudart_static.a", b"a"),
                ("nvidia_cuda_runtime_cu12-12.8.90.dist-info/RECORD", b"r"),
            ],
        );
        let w2 = dir.join("rest.whl");
        zip_with(
            &w2,
            &[
                ("nvidia/cublas/lib/libcublas.so.12", b"cublas"),
                ("nvidia/cublas/lib/libcublasLt.so.12", b"lt"),
                ("nvidia/cudnn/lib/libcudnn.so.9", b"cudnn"),
                ("nvidia/cudnn/lib/libcudnn_graph.so.9", b"graph"),
                ("nvidia/cudnn/include/cudnn.h", b"h"),
                ("nvidia/cufft/lib/libcufft.so.11", b"fft"),
                ("nvidia/curand/lib/libcurand.so.10", b"rand"),
                ("nvidia/evil/include/libevil.so.1", b"x"),
            ],
        );
        let dest = dir.join("cuda-libs");
        let t = Target {
            os: "linux",
            arch: "x86_64",
        };
        let mut got = BTreeSet::new();
        for w in [&w1, &w2] {
            got.extend(
                extract_part(w, ArchiveKind::Zip, Layout::NvidiaWheel, None, t, &dest).unwrap(),
            );
        }
        let res = crate::resources::catalog::cuda_libs_for("linux", "x86_64").unwrap();
        finalize(res, res.version, t, &dest, &mut got).unwrap();
        for n in cuda_provider_needs(false) {
            assert!(dest.join(n).exists(), "{n}");
        }
        assert_eq!(
            std::fs::read(dest.join("libcudart.so.12")).unwrap(),
            b"cudart"
        );
        #[cfg(unix)]
        assert!(
            dest.join("libcudart.so.12")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(dest.join("libcudnn_graph.so.9").is_file());
        assert!(dest.join("VERSION").is_file());
        for no in [
            "cuda.h",
            "cudnn.h",
            "libcudart_static.a",
            "RECORD",
            "libevil.so.1",
        ] {
            assert!(!dest.join(no).exists(), "{no}");
        }
    }

    /// Synthetic Windows wheels: DLLs from `bin/`, never `.lib`, headers or other dirs.
    #[test]
    fn windows_wheels_extract_only_dlls() {
        let dir = TempDir::new("wheel-w");
        let w = dir.join("all.whl");
        let mut entries: Vec<(String, Vec<u8>)> = cuda_provider_needs(true)
            .iter()
            .map(|n| (format!("nvidia/x/bin/{n}"), n.as_bytes().to_vec()))
            .collect();
        entries.push(("nvidia/cudnn/bin/cudnn_graph64_9.dll".into(), b"g".to_vec()));
        entries.push(("nvidia/cudnn/lib/x64/cudnn.lib".into(), b"l".to_vec()));
        entries.push(("nvidia/cudnn/include/cudnn.h".into(), b"h".to_vec()));
        entries.push((
            "nvidia/cudnn/lib/cudnn64_9.dll".into(),
            b"wrong dir".to_vec(),
        ));
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        zip_with(&w, &refs);
        let dest = dir.join("cuda-libs");
        let t = Target {
            os: "windows",
            arch: "x86_64",
        };
        let mut got =
            extract_part(&w, ArchiveKind::Zip, Layout::NvidiaWheel, None, t, &dest).unwrap();
        let res = crate::resources::catalog::cuda_libs_for("windows", "x86_64").unwrap();
        finalize(res, res.version, t, &dest, &mut got).unwrap();
        assert!(got.contains("cudnn_graph64_9.dll"));
        assert_eq!(
            std::fs::read(dest.join("cudnn64_9.dll")).unwrap(),
            b"cudnn64_9.dll"
        );
        for no in ["cudnn.lib", "cudnn.h"] {
            assert!(!dest.join(no).exists(), "{no}");
        }
    }

    /// Build a small tar.gz with a library, a symlink to it, a link escaping the directory and
    /// a non-whitelisted file, and extract it with the ORT release layout.
    #[test]
    fn tgz_links_and_whitelist() {
        let dir = TempDir::new("extract");
        let archive = dir.join("a.tgz");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            let mut add = |path: &str, data: &[u8]| {
                let mut h = tar::Header::new_gnu();
                h.set_size(data.len() as u64);
                h.set_mode(0o644);
                h.set_cksum();
                tar.append_data(&mut h, path, data).unwrap();
            };
            add("ort-1/lib/libonnxruntime.so.1.24.4", b"lib");
            add("ort-1/lib/libcustom.so", b"no");
            add("ort-1/include/onnxruntime_c_api.h", b"no");
            let mut link = |path: &str, target: &str| {
                let mut h = tar::Header::new_gnu();
                h.set_entry_type(tar::EntryType::Symlink);
                h.set_size(0);
                h.set_mode(0o777);
                tar.append_link(&mut h, path, target).unwrap();
            };
            link("ort-1/lib/libonnxruntime.so", "libonnxruntime.so.1.24.4");
            link(
                "ort-1/lib/libonnxruntime_providers_shared.so",
                "../../etc/passwd",
            );
            tar.into_inner().unwrap().finish().unwrap();
        }
        let dest = dir.join("out");
        let got = extract_part(
            &archive,
            ArchiveKind::TarGz,
            Layout::OrtRelease,
            Some(Flavor::Cpu),
            Target {
                os: "linux",
                arch: "x86_64",
            },
            &dest,
        )
        .unwrap();
        assert_eq!(got, set(&["libonnxruntime.so", "libonnxruntime.so.1.24.4"]));
        assert_eq!(
            std::fs::read(dest.join("libonnxruntime.so")).unwrap(),
            b"lib"
        );
        assert!(!dest.join("libcustom.so").exists());
        assert!(!dest.join("libonnxruntime_providers_shared.so").exists());
    }

    #[test]
    fn uv_archives_extract_only_the_binary() {
        assert_eq!(
            uv_wanted(false, "uv-aarch64-apple-darwin/uv").as_deref(),
            Some("uv")
        );
        assert_eq!(uv_wanted(false, "uv-aarch64-apple-darwin/uvx"), None);
        assert_eq!(
            uv_wanted(false, "a/b/uv"),
            None,
            "only top level or one folder deep"
        );
        assert_eq!(uv_wanted(false, "../uv"), None);
        assert_eq!(uv_wanted(true, "uv.exe").as_deref(), Some("uv.exe"));
        assert_eq!(uv_wanted(true, "uvw.exe"), None);
        assert_eq!(uv_wanted(true, "uv"), None);

        let dir = TempDir::new("uv");
        let mac = crate::resources::catalog::uv_for("macos", "aarch64").unwrap();
        // tar.gz with uv + uvx in a top folder.
        let archive = dir.join("uv.tgz");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            for name in ["uv-aarch64-apple-darwin/uv", "uv-aarch64-apple-darwin/uvx"] {
                let mut h = tar::Header::new_gnu();
                h.set_size(4);
                h.set_mode(0o644);
                h.set_cksum();
                tar.append_data(&mut h, name, &b"\x7fELF"[..]).unwrap();
            }
            tar.into_inner().unwrap().finish().unwrap();
        }
        let t = Target {
            os: "macos",
            arch: "aarch64",
        };
        let dest = dir.join("out");
        let mut got =
            extract_part(&archive, ArchiveKind::TarGz, Layout::Uv, None, t, &dest).unwrap();
        assert_eq!(got.iter().collect::<Vec<_>>(), ["uv"]);
        assert!(!dest.join("uvx").exists());
        finalize(mac, "0.12.19", t, &dest, &mut got).unwrap();
        assert!(got.contains("VERSION"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dest.join("uv"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o755);
        }
        // Windows zip: uv.exe at the top; an archive without it is refused.
        let win = crate::resources::catalog::uv_for("windows", "x86_64").unwrap();
        let wt = Target {
            os: "windows",
            arch: "x86_64",
        };
        let zip = dir.join("uv.zip");
        zip_with(
            &zip,
            &[("uv.exe", b"MZ"), ("uvx.exe", b"MZ"), ("uvw.exe", b"MZ")],
        );
        let wdest = dir.join("win");
        let mut got = extract_part(&zip, ArchiveKind::Zip, Layout::Uv, None, wt, &wdest).unwrap();
        assert_eq!(got.iter().collect::<Vec<_>>(), ["uv.exe"]);
        finalize(win, "0.12.19", wt, &wdest, &mut got).unwrap();
        zip_with(&zip, &[("uvx.exe", b"MZ")]);
        let empty = dir.join("empty");
        let mut got = extract_part(&zip, ArchiveKind::Zip, Layout::Uv, None, wt, &empty).unwrap();
        assert!(finalize(win, "0.12.19", wt, &empty, &mut got).is_err());
    }

    #[test]
    fn source_tree_whitelist() {
        let w = source_wanted;
        assert_eq!(
            w("DEIMv2-abc/engine/deim/postprocessor.py").as_deref(),
            Some("engine/deim/postprocessor.py")
        );
        assert_eq!(
            w("DEIMv2-abc/engine/backbone/dinov3/layers/block.py").as_deref(),
            Some("engine/backbone/dinov3/layers/block.py")
        );
        assert_eq!(w("DEIMv2-abc/LICENSE").as_deref(), Some("LICENSE"));
        assert_eq!(
            w("DEIMv2-abc/configs/deimv2/deimv2_hgnetv2_pico_coco.yml").as_deref(),
            Some("configs/deimv2/deimv2_hgnetv2_pico_coco.yml")
        );
        for bad in [
            "pax_global_header",
            "DEIMv2-abc/",
            "DEIMv2-abc/train.py",
            "DEIMv2-abc/tools/deployment/export_onnx.py",
            "DEIMv2-abc/engine/data/x.yml",
            "DEIMv2-abc/configs/x.py",
            "DEIMv2-abc/configs/deimv2/x.yaml.sh",
            "DEIMv2-abc/engine/../../x.py",
            "DEIMv2-abc/engine/.hidden/x.py",
            "DEIMv2-abc/engine/a b.py",
            "DEIMv2-abc/engine/x.py.sh",
            "DEIMv2-abc/example.jpg",
            "C:/x/engine/x.py",
        ] {
            assert_eq!(w(bad), None, "{bad}");
        }

        // A real tar.gz: only the whitelisted files land, paths kept.
        let dir = TempDir::new("src");
        let archive = dir.join("src.tar.gz");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            for (name, body) in [
                ("top/engine/__init__.py", "from . import data\n"),
                ("top/engine/deim/deim.py", "x = 1\n"),
                ("top/configs/base/deimv2.yml", "a: 1\n"),
                ("top/train.py", "import os\n"),
                ("top/LICENSE", "Apache\n"),
            ] {
                let mut h = tar::Header::new_gnu();
                h.set_size(body.len() as u64);
                h.set_mode(0o644);
                h.set_cksum();
                tar.append_data(&mut h, name, body.as_bytes()).unwrap();
            }
            tar.into_inner().unwrap().finish().unwrap();
        }
        let dest = dir.join("out");
        let t = Target {
            os: "linux",
            arch: "x86_64",
        };
        let mut got = extract_part(
            &archive,
            ArchiveKind::TarGz,
            Layout::SourceTree,
            None,
            t,
            &dest,
        )
        .unwrap();
        assert_eq!(
            got.iter().map(String::as_str).collect::<Vec<_>>(),
            [
                "LICENSE",
                "configs/base/deimv2.yml",
                "engine/__init__.py",
                "engine/deim/deim.py"
            ]
        );
        assert!(dest.join("engine/deim/deim.py").is_file());
        assert!(!dest.join("train.py").exists());
        let src = crate::resources::catalog::deimv2_source().unwrap();
        finalize(src, src.version, t, &dest, &mut got).unwrap();
        assert!(dest.join("VERSION").is_file());
        let mut empty = BTreeSet::new();
        assert!(finalize(src, src.version, t, &dir.join("none"), &mut empty).is_err());
    }

    // -----------------------------------------------------------------------------------------
    // Refusals: crafted archives
    // -----------------------------------------------------------------------------------------

    /// A tar.gz built from `entries`: (path, kind, data or link target, mode).
    enum TarEntry<'a> {
        File(&'a str, &'a [u8], u32),
        Symlink(&'a str, &'a str),
        Hardlink(&'a str, &'a str),
    }

    fn tgz_with(path: &Path, entries: &[TarEntry<'_>]) {
        let f = std::fs::File::create(path).unwrap();
        let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for e in entries {
            let mut h = tar::Header::new_gnu();
            match e {
                TarEntry::File(name, data, mode) => {
                    h.set_size(data.len() as u64);
                    h.set_mode(*mode);
                    // `set_path` refuses `..` and absolute names; a crafted archive has
                    // them in the raw header, so write the bytes directly.
                    let field = &mut h.as_gnu_mut().unwrap().name;
                    field[..name.len()].copy_from_slice(name.as_bytes());
                    h.set_cksum();
                    tar.append(&h, *data).unwrap();
                }
                TarEntry::Symlink(name, target) => {
                    h.set_entry_type(tar::EntryType::Symlink);
                    h.set_size(0);
                    h.set_mode(0o777);
                    tar.append_link(&mut h, name, target).unwrap();
                }
                TarEntry::Hardlink(name, target) => {
                    h.set_entry_type(tar::EntryType::Link);
                    h.set_size(0);
                    h.set_mode(0o644);
                    tar.append_link(&mut h, name, target).unwrap();
                }
            }
        }
        tar.into_inner().unwrap().finish().unwrap();
    }

    /// The zip writer normalises `..` and absolute names away, so a crafted name is written
    /// with a same-length placeholder (`..` -> `ZZ`, leading `/` -> `Z`) and patched into
    /// the local header and the central directory afterwards (the CRC covers data only).
    fn zip_with_modes(path: &Path, entries: &[(&str, &[u8], Option<u32>)]) {
        let placeholder = |n: &str| {
            n.replace("..", "ZZ")
                .replacen('/', "Z", usize::from(n.starts_with('/')))
        };
        {
            let f = std::fs::File::create(path).unwrap();
            let mut z = zip::ZipWriter::new(f);
            for (name, data, mode) in entries {
                let mut o = zip::write::SimpleFileOptions::default();
                if let Some(m) = mode {
                    // `unix_permissions` masks to 0o777; the raw attributes keep the file
                    // type and setuid/setgid bits.
                    o = o.system(zip::System::Unix).external_attributes(m << 16);
                }
                z.start_file(placeholder(name), o).unwrap();
                z.write_all(data).unwrap();
            }
            z.finish().unwrap();
        }
        let mut bytes = std::fs::read(path).unwrap();
        for (name, _, _) in entries {
            let from = placeholder(name);
            if from == *name {
                continue;
            }
            let mut at = 0;
            while let Some(i) = bytes[at..]
                .windows(from.len())
                .position(|w| w == from.as_bytes())
            {
                bytes[at + i..at + i + from.len()].copy_from_slice(name.as_bytes());
                at += i + from.len();
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    const LINUX: Target<'static> = Target {
        os: "linux",
        arch: "x86_64",
    };

    fn ort_tgz(archive: &Path, dest: &Path, limits: Limits) -> Result<BTreeSet<String>> {
        extract_part_with(
            archive,
            ArchiveKind::TarGz,
            Layout::OrtRelease,
            Some(Flavor::Cpu),
            LINUX,
            dest,
            limits,
        )
    }

    fn wheel_zip(archive: &Path, dest: &Path, limits: Limits) -> Result<BTreeSet<String>> {
        extract_part_with(
            archive,
            ArchiveKind::Zip,
            Layout::NvidiaWheel,
            None,
            LINUX,
            dest,
            limits,
        )
    }

    fn refused(r: Result<BTreeSet<String>>, why: &str) {
        let err = r.expect_err("the archive must be refused");
        let text = format!("{err:#}");
        assert!(text.contains(why), "expected {why:?} in: {text}");
    }

    #[test]
    fn hostile_paths_are_refused() {
        for (entry, why) in [
            ("/etc/passwd", "absolute path"),
            ("C:/Windows/evil.dll", "drive letter"),
            ("lib/../../x", "`..`"),
            ("..", "`..`"),
            ("nvidia/..\\evil/lib/libevil.so.1", "`..`"),
        ] {
            assert!(
                format!("{}", check_entry_path(entry).unwrap_err()).contains(why),
                "{entry}"
            );
        }
        assert!(check_entry_path("ort-1/lib/libonnxruntime.so").is_ok());
        assert!(check_entry_path("./lib/x").is_ok());
        assert!(check_entry_path("a..b/c...").is_ok());
        assert!(check_entry_mode("x", Some(0o755)).is_ok());
        assert!(check_entry_mode("x", None).is_ok());
        assert!(check_entry_mode("x", Some(0o4755)).is_err());
        assert!(check_entry_mode("x", Some(0o2644)).is_err());
    }

    #[test]
    fn tgz_refuses_dotdot_absolute_and_setuid_entries() {
        let dir = TempDir::new("tgz-refuse");
        let good = TarEntry::File("ort-1/lib/libonnxruntime.so", b"lib", 0o644);
        // `..` in any entry, wanted or not, fails the whole archive and leaves nothing.
        let a = dir.join("dotdot.tgz");
        tgz_with(
            &a,
            &[
                good,
                TarEntry::File("ort-1/lib/../../evil.txt", b"x", 0o644),
            ],
        );
        let dest = dir.join("dotdot");
        refused(ort_tgz(&a, &dest, Limits::default()), "`..`");
        assert!(!dir.join("evil.txt").exists());
        // Absolute path.
        let a = dir.join("abs.tgz");
        tgz_with(&a, &[TarEntry::File("/tmp/evil.txt", b"x", 0o644)]);
        refused(ort_tgz(&a, &dir.join("abs"), Limits::default()), "absolute");
        // setuid / setgid on a wanted file.
        for (name, mode) in [("suid.tgz", 0o4755), ("sgid.tgz", 0o2755)] {
            let a = dir.join(name);
            tgz_with(
                &a,
                &[TarEntry::File("ort-1/lib/libonnxruntime.so", b"lib", mode)],
            );
            let dest = dir.join(format!("{name}.out"));
            refused(ort_tgz(&a, &dest, Limits::default()), "setuid/setgid");
            assert!(!dest.join("libonnxruntime.so").exists());
        }
        // A plain archive still extracts, with the mode ignored.
        let a = dir.join("ok.tgz");
        tgz_with(
            &a,
            &[TarEntry::File("ort-1/lib/libonnxruntime.so", b"lib", 0o755)],
        );
        assert_eq!(
            ort_tgz(&a, &dir.join("ok"), Limits::default()).unwrap(),
            set(&["libonnxruntime.so"])
        );
    }

    #[test]
    fn tgz_links_outside_the_whitelist_are_not_created() {
        let dir = TempDir::new("tgz-links");
        let a = dir.join("links.tgz");
        tgz_with(
            &a,
            &[
                TarEntry::File("ort-1/lib/libonnxruntime.so.1", b"lib", 0o644),
                // Symlink out of the tree, symlink to an absolute path.
                TarEntry::Symlink("ort-1/lib/libonnxruntime.so", "../../../etc/passwd"),
                TarEntry::Symlink(
                    "ort-1/lib/libonnxruntime_providers_shared.so",
                    "/etc/passwd",
                ),
                // Hard link to a file in another directory of the archive.
                TarEntry::Hardlink(
                    "ort-1/lib/libonnxruntime_providers_cuda.so",
                    "ort-1/include/onnxruntime_c_api.h",
                ),
            ],
        );
        let dest = dir.join("out");
        let got = ort_tgz(&a, &dest, Limits::default()).unwrap();
        assert_eq!(got, set(&["libonnxruntime.so.1"]));
        for no in [
            "libonnxruntime.so",
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_cuda.so",
        ] {
            assert!(dest.join(no).symlink_metadata().is_err(), "{no}");
        }
        // A same-directory link to a whitelisted file is fine.
        let a = dir.join("ok.tgz");
        tgz_with(
            &a,
            &[
                TarEntry::File("ort-1/lib/libonnxruntime.so.1", b"lib", 0o644),
                TarEntry::Symlink("ort-1/lib/libonnxruntime.so", "libonnxruntime.so.1"),
                TarEntry::Hardlink(
                    "ort-1/lib/libonnxruntime_providers_shared.so",
                    "ort-1/lib/libonnxruntime.so.1",
                ),
            ],
        );
        let dest = dir.join("ok");
        let got = ort_tgz(&a, &dest, Limits::default()).unwrap();
        // The symlink is recreated; the flat ORT layout never recreates hard links (only the
        // OpenVINO layout maps them, within one directory).
        assert_eq!(got, set(&["libonnxruntime.so", "libonnxruntime.so.1"]));
        assert_eq!(
            std::fs::read(dest.join("libonnxruntime.so")).unwrap(),
            b"lib"
        );
        assert!(
            dest.join("libonnxruntime_providers_shared.so")
                .symlink_metadata()
                .is_err()
        );
    }

    #[test]
    fn zip_refuses_dotdot_absolute_and_setuid_entries() {
        let dir = TempDir::new("zip-refuse");
        let good: (&str, &[u8], Option<u32>) =
            ("nvidia/cudnn/lib/libcudnn.so.9", b"cudnn", Some(0o644));
        let a = dir.join("dotdot.whl");
        zip_with_modes(&a, &[good, ("nvidia/../evil/lib/libevil.so.1", b"x", None)]);
        let dest = dir.join("dotdot");
        refused(wheel_zip(&a, &dest, Limits::default()), "`..`");
        assert!(!dest.join("libcudnn.so.9").exists() || !dir.join("evil").exists());
        let a = dir.join("abs.whl");
        zip_with_modes(&a, &[("/nvidia/cudnn/lib/libcudnn.so.9", b"x", None)]);
        refused(
            wheel_zip(&a, &dir.join("abs"), Limits::default()),
            "absolute",
        );
        let a = dir.join("suid.whl");
        zip_with_modes(
            &a,
            &[("nvidia/cudnn/lib/libcudnn.so.9", b"x", Some(0o4755))],
        );
        // The zip reader itself strips the special bits from `unix_mode()`, and no entry
        // mode is ever applied to a written file: the file lands as a plain file.
        let dest = dir.join("suid");
        assert_eq!(
            wheel_zip(&a, &dest, Limits::default()).unwrap(),
            set(&["libcudnn.so.9"])
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dest.join("libcudnn.so.9"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o6000, 0, "{mode:o}");
        }
        // An unwanted entry's mode does not matter; a symlink entry is skipped.
        let a = dir.join("ok.whl");
        zip_with_modes(
            &a,
            &[
                good,
                ("nvidia/cudnn/include/tool", b"x", Some(0o4755)),
                (
                    "nvidia/cudnn/lib/libcudnn_link.so.9",
                    b"libcudnn.so.9",
                    Some(0o120777),
                ),
            ],
        );
        let dest = dir.join("ok");
        assert_eq!(
            wheel_zip(&a, &dest, Limits::default()).unwrap(),
            set(&["libcudnn.so.9"])
        );
        assert!(dest.join("libcudnn_link.so.9").symlink_metadata().is_err());
    }

    #[test]
    fn decompressed_size_ceilings_are_enforced_while_streaming() {
        let dir = TempDir::new("limits");
        let big = vec![0u8; 10_000];
        let small = Limits {
            entry_bytes: 4_000,
            total_bytes: 1 << 20,
        };
        // Per-file ceiling, tar.gz: the file is removed and nothing is installed.
        let a = dir.join("big.tgz");
        tgz_with(
            &a,
            &[TarEntry::File("ort-1/lib/libonnxruntime.so", &big, 0o644)],
        );
        let dest = dir.join("tgz");
        refused(ort_tgz(&a, &dest, small), "per file");
        assert!(std::fs::read_dir(&dest).unwrap().next().is_none());
        // Per-file ceiling, zip, with a header that lies (the zip writer stores the real
        // size; the check counts bytes while streaming regardless).
        let a = dir.join("big.whl");
        zip_with_modes(&a, &[("nvidia/cudnn/lib/libcudnn.so.9", &big, None)]);
        refused(wheel_zip(&a, &dir.join("zip"), small), "per file");
        // Exactly the ceiling is fine.
        let exact = Limits {
            entry_bytes: big.len() as u64,
            total_bytes: 1 << 20,
        };
        assert_eq!(
            wheel_zip(&a, &dir.join("exact"), exact).unwrap(),
            set(&["libcudnn.so.9"])
        );
        // Total ceiling across files.
        let a = dir.join("many.whl");
        zip_with_modes(
            &a,
            &[
                ("nvidia/cudnn/lib/libcudnn.so.9", &big, None),
                ("nvidia/cublas/lib/libcublas.so.12", &big, None),
                ("nvidia/cufft/lib/libcufft.so.11", &big, None),
            ],
        );
        let total = Limits {
            entry_bytes: 1 << 20,
            total_bytes: 25_000,
        };
        refused(wheel_zip(&a, &dir.join("total"), total), "archive total");
        let roomy = Limits {
            entry_bytes: 1 << 20,
            total_bytes: 30_000,
        };
        assert_eq!(wheel_zip(&a, &dir.join("roomy"), roomy).unwrap().len(), 3);
        // The defaults are generous enough for every pinned archive.
        const { assert!(MAX_ENTRY_BYTES >= 2 << 30 && MAX_TOTAL_BYTES >= MAX_ENTRY_BYTES) };
    }
}
