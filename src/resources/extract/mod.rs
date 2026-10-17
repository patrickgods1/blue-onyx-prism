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
//!   `MAX_ENTRY_BYTES` per file and `MAX_TOTAL_BYTES` per archive;
//! - every archive is extracted into a staging directory (`super::manager::run_job`) that is
//!   renamed into place only after every entry passed and the install was checked
//!   (`finalize`); a failure removes the staging directory;
//! - nothing from an archive is executed here. The `uv` binary and the DEIMv2 sources are run
//!   only by a model export the user starts explicitly (`super::export`). The Visual C++ runtime
//!   DLLs are only loaded by that export's Python.
//!
//! Layouts:
//! - [`Layout::OpenVino`]: the toolkit's `runtime/...` libraries, layout preserved (what
//!   `openvino-finder` expects under `OPENVINO_INSTALL_DIR`), plus its license notices
//!   (`docs/licensing/LICENSE`, the `*third-party-programs.txt` files of the runtime, oneDNN
//!   and oneTBB, `runtime/3rdparty/tbb/TBB-LICENSE`).
//! - [`Layout::OrtRelease`], [`Layout::NugetOrt`], [`Layout::NugetDirectMl`]: ONNX Runtime and
//!   DirectML shared libraries, flat, plus the license notices under `licenses/onnxruntime/`
//!   (`LICENSE`, `ThirdPartyNotices.txt`) and `licenses/directml/` (`LICENSE.txt`,
//!   `ThirdPartyNotices.txt`) so a redistributed install keeps them (NR-17).
//! - [`Layout::NvidiaWheel`]: CUDA/cuDNN shared libraries from `nvidia/<component>/{bin,lib}/`,
//!   flat.
//! - [`Layout::Uv`]: only the `uv` / `uv.exe` binary of a uv release archive (made executable;
//!   it is run only by a model export the user starts, see `super::export`).
//! - [`Layout::SourceTree`]: the DEIMv2 source archive: `engine/**/*.py`, `configs/**/*.yml` and
//!   the license, paths kept (plain path components only). Imported / read only by a DEIMv2
//!   export the user starts.
//! - [`Layout::VcRuntime`]: the `msvc-runtime` wheel: only the pinned Visual C++ runtime DLLs
//!   (`catalog::VC_RUNTIME_DLLS`, exact entry paths), flat, each checked against its own pinned
//!   SHA-256 after extraction. Copied next to the export's Python only by a model export.
//! - [`Layout::File`]: not an archive; the manager moves the verified file into place.

use super::catalog::{ArchiveKind, Flavor, Layout, Resource, ResourceKind, VC_RUNTIME_DLLS};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod fuzz_tests;
mod nvidia;
mod onnxruntime;
mod openvino;
#[cfg(test)]
mod test_util;

pub(crate) use nvidia::*;
pub(crate) use onnxruntime::*;
pub(crate) use openvino::*;

/// An extraction stopped because its stop flag was set (service shutdown).
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub(crate) struct Cancelled;

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
pub(crate) const MAX_ENTRY_BYTES: u64 = 4 << 30;
/// Most decompressed bytes one archive may produce in total (8 GiB). The largest real archive
/// (the cuDNN wheel) extracts to about 1.5 GiB of whitelisted libraries.
pub(crate) const MAX_TOTAL_BYTES: u64 = 8 << 30;

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

/// Bytes written so far against [`Limits`], and the stop flag checked while writing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget<'a> {
    limits: Limits,
    used: u64,
    stop: Option<&'a AtomicBool>,
}

/// An archive exceeded a decompressed-size ceiling.
#[derive(Debug, thiserror::Error)]
#[error("{entry}: extracted data exceeds {limit} bytes ({what}); the archive is refused")]
pub(crate) struct OverLimit {
    pub entry: String,
    pub limit: u64,
    pub what: &'static str,
}

/// An archive entry whose path or mode is not acceptable.
#[derive(Debug, thiserror::Error)]
#[error("archive entry {entry:?} is refused: {why}")]
pub(crate) struct HostileEntry {
    pub entry: String,
    pub why: &'static str,
}

impl<'a> Budget<'a> {
    fn new(limits: Limits, stop: Option<&'a AtomicBool>) -> Self {
        Self {
            limits,
            used: 0,
            stop,
        }
    }

    /// Fail with [`Cancelled`] once the stop flag is set.
    fn check_stop(&self) -> Result<()> {
        if self.stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            return Err(anyhow::Error::new(Cancelled));
        }
        Ok(())
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

/// [`extract_part`] that stops with [`Cancelled`] as soon as `stop` is set (checked between
/// entries and every 256 KiB inside one), so a service stop does not wait for a large archive.
pub(crate) fn extract_part_until(
    archive: &Path,
    kind: ArchiveKind,
    layout: Layout,
    flavor: Option<Flavor>,
    target: Target<'_>,
    dest: &Path,
    stop: &AtomicBool,
) -> Result<BTreeSet<String>> {
    extract_impl(
        archive,
        kind,
        layout,
        flavor,
        target,
        dest,
        Budget::new(Limits::default(), Some(stop)),
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
    extract_impl(
        archive,
        kind,
        layout,
        flavor,
        target,
        dest,
        Budget::new(limits, None),
    )
}

fn extract_impl(
    archive: &Path,
    kind: ArchiveKind,
    layout: Layout,
    flavor: Option<Flavor>,
    target: Target<'_>,
    dest: &Path,
    mut budget: Budget<'_>,
) -> Result<BTreeSet<String>> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let windows = target.windows();
    let flavor = flavor.unwrap_or(Flavor::Cpu);
    let b = &mut budget;
    match layout {
        Layout::OpenVino => {
            let sel = selection_for(target.os)?;
            let wanted = |e: &str| {
                strip_top(e).and_then(|r| sel.wanted(&r).or_else(|| openvino_license_wanted(&r)))
            };
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
        Layout::VcRuntime => match kind {
            ArchiveKind::Zip => extract_zip(archive, dest, vc_runtime_wanted, b),
            ArchiveKind::TarGz => extract_tgz(archive, dest, vc_runtime_wanted, |_, _, _| None, b),
        },
        Layout::File => anyhow::bail!("{} is not an archive", archive.display()),
    }
}

/// Source archives (`<top>/...`, as GitHub makes them): `engine/**/*.py`, `configs/**/*.yml`
/// (the training configs the model is built from) and the top-level `LICENSE` / `LICENSE.md`,
/// paths kept below the top folder. Every component must be a plain
/// name (ASCII letters, digits, `_`, `-`, `.`; not starting with `.`).
pub(crate) fn source_wanted(entry: &str) -> Option<String> {
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

/// `msvc-runtime` wheel entries: exactly the pinned paths of `catalog::VC_RUNTIME_DLLS` (the
/// `data/` DLLs and `data/Scripts/vcruntime140*.dll`), written flat under their file name.
pub(crate) fn vc_runtime_wanted(entry: &str) -> Option<String> {
    let e = entry.replace('\\', "/");
    VC_RUNTIME_DLLS
        .iter()
        .find(|(_, path, _)| *path == e)
        .map(|(name, _, _)| (*name).to_string())
}

/// Every pinned Visual C++ runtime DLL was extracted and has its pinned SHA-256.
fn vc_runtime_check(dest: &Path, got: &BTreeSet<String>) -> Result<Vec<String>> {
    for (name, _, sha) in VC_RUNTIME_DLLS {
        if !got.contains(*name) {
            anyhow::bail!("the Visual C++ runtime wheel has no {name}");
        }
        let actual = super::manager::sha256_file(&dest.join(name))?;
        if !actual.eq_ignore_ascii_case(sha) {
            anyhow::bail!(
                "{name} from the Visual C++ runtime wheel has SHA-256 {actual}, pinned {sha}; \
                 the download is refused"
            );
        }
    }
    Ok(Vec::new())
}

/// File name of the uv binary.
pub(crate) fn uv_binary_name(windows: bool) -> &'static str {
    if windows { "uv.exe" } else { "uv" }
}

/// uv release archives: `uv-<triple>/uv` (tar.gz) or `uv.exe` at the top (Windows zip). Only the
/// binary itself is kept (not `uvx` / `uvw`).
pub(crate) fn uv_wanted(windows: bool, entry: &str) -> Option<String> {
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
pub(crate) fn finalize(
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
        ResourceKind::Tool if resource.parts.iter().all(|p| p.layout == Layout::VcRuntime) => {
            vc_runtime_check(dest, got)?
        }
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
pub(crate) fn install_note(resource: &Resource) -> Option<&'static str> {
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
    budget: &mut Budget<'_>,
) -> Result<u64> {
    budget.check_stop()?;
    let out = dest.join(rel);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    // `<name>.partial` keeps the whole name (`with_extension` would turn
    // `libonnxruntime.so.1.24.4` into `libonnxruntime.so.1.24.partial`).
    let tmp = out.with_file_name({
        let mut name = out.file_name().unwrap_or_default().to_os_string();
        name.push(".partial");
        name
    });
    let mut f =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let allowance = budget.allowance();
    // One byte more than allowed: reaching it proves the entry is too large.
    let mut stoppable = StopRead {
        inner: reader,
        stop: budget.stop,
        since_check: 0,
    };
    let mut capped = (&mut stoppable).take(allowance.saturating_add(1));
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
            budget.check_stop()?;
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

/// A reader that fails with [`Cancelled`] once `stop` is set (checked every 256 KiB).
struct StopRead<'r, 's> {
    inner: &'r mut dyn Read,
    stop: Option<&'s AtomicBool>,
    since_check: usize,
}

impl Read for StopRead<'_, '_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.since_check >= 256 * 1024 || self.since_check == 0 {
            self.since_check = 0;
            if self.stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
                return Err(std::io::Error::other(Cancelled));
            }
        }
        let n = self.inner.read(buf)?;
        self.since_check = self.since_check.saturating_add(n.max(1));
        Ok(n)
    }
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
    budget: &mut Budget<'_>,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading zip {}", archive.display()))?;
    let mut got = BTreeSet::new();
    for i in 0..zip.len() {
        budget.check_stop()?;
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
    budget: &mut Budget<'_>,
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
        budget.check_stop()?;
        let mut entry =
            entry.with_context(|| format!("reading tar entry in {}", archive.display()))?;
        // `Entry::path` already refuses `..` and absolute paths on GNU/ustar headers; the
        // raw bytes are checked too so a PAX `path` record cannot sidestep that.
        let raw = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        check_entry_path(&raw)?;
        let path = entry
            .path()
            .with_context(|| format!("reading an entry name in {}", archive.display()))?
            .to_string_lossy()
            .into_owned();
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

#[cfg(test)]
mod tests {
    use super::test_util::*;
    use super::*;

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
        let mut budget = Budget::new(Limits::default(), None);
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

    /// The `msvc-runtime` wheel: only the pinned DLL paths are written (flat), and every one must
    /// match its pinned SHA-256.
    #[test]
    fn vc_runtime_wheel_extracts_only_pinned_dlls() {
        use crate::resources::catalog::{VC_RUNTIME_DLLS, vc_runtime_for};
        let data = "msvc_runtime-14.44.35112.data/data";
        assert_eq!(
            vc_runtime_wanted(&format!("{data}/msvcp140.dll")).as_deref(),
            Some("msvcp140.dll")
        );
        assert_eq!(
            vc_runtime_wanted(&format!("{data}\\Scripts\\vcruntime140_1.dll")).as_deref(),
            Some("vcruntime140_1.dll")
        );
        for refused in [
            format!("{data}/Scripts/msvcp140.dll"),
            format!("{data}/vcamp140.dll"),
            format!("{data}/vccorlib140.dll"),
            "msvc_runtime.cp311-win_amd64.pyd".to_string(),
            "msvc_runtime-14.44.35112.dist-info/licenses/LICENSE".to_string(),
            "msvcp140.dll".to_string(),
            format!("other/{data}/msvcp140.dll"),
        ] {
            assert_eq!(vc_runtime_wanted(&refused), None, "{refused}");
        }

        let dir = TempDir::new("vc-runtime");
        let wheel = dir.join("msvc.whl");
        let mut entries: Vec<(String, Vec<u8>)> = VC_RUNTIME_DLLS
            .iter()
            .map(|(name, path, _)| (path.to_string(), format!("MZ {name}").into_bytes()))
            .collect();
        for extra in [
            "msvc_runtime.cp311-win_amd64.pyd",
            "msvc_runtime-14.44.35112.data/data/vcamp140.dll",
            "msvc_runtime-14.44.35112.data/data/Scripts/msvcp140.dll",
        ] {
            entries.push((extra.to_string(), b"MZ extra".to_vec()));
        }
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        zip_with(&wheel, &refs);
        let wt = Target {
            os: "windows",
            arch: "x86_64",
        };
        let dest = dir.join("out");
        let mut got =
            extract_part(&wheel, ArchiveKind::Zip, Layout::VcRuntime, None, wt, &dest).unwrap();
        let names: BTreeSet<String> = VC_RUNTIME_DLLS
            .iter()
            .map(|(n, _, _)| n.to_string())
            .collect();
        assert_eq!(got, names);
        assert_eq!(
            std::fs::read(dest.join("msvcp140.dll")).unwrap(),
            b"MZ msvcp140.dll",
            "the data/ copy, not the Scripts/ duplicate"
        );
        assert!(!dest.join("vcamp140.dll").exists());
        assert!(!dest.join("Scripts").exists());
        // Synthetic bytes: the per-file pins refuse them.
        let res = vc_runtime_for("windows", "x86_64").unwrap();
        let e = finalize(res, res.version, wt, &dest, &mut got).unwrap_err();
        assert!(format!("{e:#}").contains("pinned"), "{e:#}");
        // A wheel without one of the DLLs is refused too.
        zip_with(&wheel, &refs[1..]);
        let short = dir.join("short");
        let mut got = extract_part(
            &wheel,
            ArchiveKind::Zip,
            Layout::VcRuntime,
            None,
            wt,
            &short,
        )
        .unwrap();
        let e = finalize(res, res.version, wt, &short, &mut got).unwrap_err();
        assert!(format!("{e:#}").contains("has no concrt140.dll"), "{e:#}");
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

    /// RS-13: the temp file is `<full name>.partial`, not `with_extension` (which would cut a
    /// versioned soname at its last dot), and it is gone once the entry is in place.
    #[test]
    fn write_entry_keeps_the_full_name_for_the_partial_file() {
        let dir = TempDir::new("partial-name");
        struct Probe<'a> {
            dir: &'a Path,
            seen: bool,
        }
        impl Read for Probe<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.seen {
                    return Ok(0);
                }
                self.seen = true;
                assert!(self.dir.join("libonnxruntime.so.1.24.4.partial").is_file());
                assert!(!self.dir.join("libonnxruntime.so.1.24.partial").exists());
                buf[..3].copy_from_slice(b"lib");
                Ok(3)
            }
        }
        let mut probe = Probe {
            dir: &dir,
            seen: false,
        };
        let mut budget = Budget::new(Limits::default(), None);
        let n = write_entry(&dir, "libonnxruntime.so.1.24.4", &mut probe, &mut budget).unwrap();
        assert!(probe.seen);
        assert_eq!(n, 3);
        assert_eq!(
            std::fs::read(dir.join("libonnxruntime.so.1.24.4")).unwrap(),
            b"lib"
        );
        let names: Vec<_> = std::fs::read_dir(&*dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["libonnxruntime.so.1.24.4"]);
    }
}
