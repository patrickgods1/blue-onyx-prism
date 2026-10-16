//! The Microsoft Visual C++ runtime of the model export (Windows): the pinned PyTorch is built
//! with Visual Studio 2022 17.10+ and needs `msvcp140.dll` 14.40 or newer (plus
//! `vcruntime140_threads.dll` / `msvcp140_atomic_wait.dll`, which older runtimes lack). A machine
//! with an older system runtime (System32) crashes in `import torch` (WinError 1114, access
//! violation). The export then places the private copy `tool:vc-runtime` (Microsoft-signed DLLs,
//! SHA-256 checked at extraction) next to its own base Python under `tools/python/`, which the
//! loader searches before System32; nothing is installed system-wide and no admin rights are
//! needed.
//!
//! Pure parts, tested on every OS: [`pe_file_version`] (the `VS_FIXEDFILEINFO` of a PE file, read
//! from its resource section), [`decide`] (whether to place the private runtime), [`pyvenv_home`].
//! Only [`system_dir`] (where System32 is) is Windows-specific; [`ensure`] is the export stage.

use super::*;
use catalog::{VC_RUNTIME_DLLS, VC_RUNTIME_FILE_VERSION, VC_RUNTIME_TORCH_MIN};
use std::fmt;

/// Microsoft's download of the current Visual C++ 2015-2022 x64 runtime (system-wide, admin).
pub const VC_REDIST_URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// `BOP_VC_RUNTIME=force` places the private runtime even when the system runtime is new enough
/// (a runtime that reports a current version but is damaged).
pub const FORCE_ENV: &str = "BOP_VC_RUNTIME";

/// The DLLs whose effective copy (next to Python, else in System32) must be at least the pin.
/// `msvcp140_codecvt_ids.dll` is placed with the others but not probed (nothing we load needs it).
pub const PROBED: [&str; 9] = [
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "vcruntime140_threads.dll",
    "concrt140.dll",
    "vcomp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
];

/// A PE `FileVersion` (`VS_FIXEDFILEINFO.dwFileVersionMS/LS`): `14.44.35211.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileVersion(pub [u16; 4]);

impl FileVersion {
    /// The version of every DLL in `tool:vc-runtime`.
    pub const PIN: FileVersion = FileVersion(VC_RUNTIME_FILE_VERSION);

    /// At least `major.minor`.
    pub fn at_least(self, [major, minor]: [u16; 2]) -> bool {
        (self.0[0], self.0[1]) >= (major, minor)
    }
}

impl fmt::Display for FileVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [a, b, c, d] = self.0;
        write!(f, "{a}.{b}.{c}.{d}")
    }
}

/// "14.40": the oldest runtime PyTorch loads with.
pub fn torch_min() -> String {
    let [a, b] = VC_RUNTIME_TORCH_MIN;
    format!("{a}.{b}")
}

// ---------------------------------------------------------------------------------------------
// PE FileVersion
// ---------------------------------------------------------------------------------------------

/// `VS_FIXEDFILEINFO.dwSignature`.
const FIXED_FILE_INFO_SIGNATURE: u32 = 0xFEEF_04BD;
/// `VS_FIXEDFILEINFO.dwStrucVersion` (1.0).
const FIXED_FILE_INFO_STRUC_VERSION: u32 = 0x0001_0000;
/// Largest file [`file_version`] reads (the runtime DLLs are well under 2 MB).
const MAX_PE_BYTES: u64 = 64 << 20;

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        b.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// The `FileVersion` of a PE image (`.dll` / `.exe`): the headers locate the resource directory
/// (data directory 2), its section's raw data is scanned for the `VS_FIXEDFILEINFO` signature
/// (`0xFEEF04BD`, followed by `dwStrucVersion` 1.0) and `dwFileVersionMS` / `dwFileVersionLS` are
/// read. None for anything that is not a PE image with a version resource.
pub fn pe_file_version(b: &[u8]) -> Option<FileVersion> {
    if b.get(..2)? != b"MZ" {
        return None;
    }
    let pe = usize::try_from(u32_at(b, 0x3C)?).ok()?;
    if b.get(pe..pe.checked_add(4)?)? != b"PE\0\0" {
        return None;
    }
    let coff = pe + 4;
    let sections = usize::from(u16_at(b, coff + 2)?);
    let opt_size = usize::from(u16_at(b, coff + 16)?);
    let opt = coff + 20;
    // Data directories: after 96 bytes (PE32) or 112 bytes (PE32+) of the optional header.
    let (count_at, dirs_at) = match u16_at(b, opt)? {
        0x10b => (opt + 92, opt + 96),
        0x20b => (opt + 108, opt + 112),
        _ => return None,
    };
    if u32_at(b, count_at)? < 3 {
        return None;
    }
    let rsrc_rva = u32_at(b, dirs_at + 2 * 8)?;
    if rsrc_rva == 0 {
        return None;
    }
    let table = opt.checked_add(opt_size)?;
    for i in 0..sections {
        let s = table.checked_add(i.checked_mul(40)?)?;
        let virtual_size = u32_at(b, s + 8)?;
        let va = u32_at(b, s + 12)?;
        let raw_size = u32_at(b, s + 16)?;
        let raw_at = u32_at(b, s + 20)?;
        let span = virtual_size.max(raw_size);
        if rsrc_rva < va || rsrc_rva - va >= span {
            continue;
        }
        let start = usize::try_from(raw_at).ok()?;
        let end = start
            .checked_add(usize::try_from(raw_size).ok()?)?
            .min(b.len());
        return scan_fixed_file_info(b.get(start..end)?);
    }
    None
}

/// The first `VS_FIXEDFILEINFO` in `data` (4-byte aligned, as the structure is in a version
/// resource).
fn scan_fixed_file_info(data: &[u8]) -> Option<FileVersion> {
    let mut at = 0usize;
    while at + 16 <= data.len() {
        if u32_at(data, at) == Some(FIXED_FILE_INFO_SIGNATURE)
            && u32_at(data, at + 4) == Some(FIXED_FILE_INFO_STRUC_VERSION)
        {
            let ms = u32_at(data, at + 8)?;
            let ls = u32_at(data, at + 12)?;
            let hi = |v: u32| u16::try_from(v >> 16).unwrap_or(0);
            let lo = |v: u32| u16::try_from(v & 0xFFFF).unwrap_or(0);
            return Some(FileVersion([hi(ms), lo(ms), hi(ls), lo(ls)]));
        }
        at += 4;
    }
    None
}

/// [`pe_file_version`] of the file at `path`; None when it is missing, unreadable, too large or
/// has no version resource.
pub fn file_version(path: &Path) -> Option<FileVersion> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_PE_BYTES {
        return None;
    }
    pe_file_version(&std::fs::read(path).ok()?)
}

// ---------------------------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------------------------

/// Versions of one runtime DLL: in System32 and next to the export's base Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Found {
    pub name: &'static str,
    pub system: Option<FileVersion>,
    pub python: Option<FileVersion>,
}

impl Found {
    /// The copy Windows loads for Python: the one next to `python.exe` when present (the
    /// application directory is searched first), else System32's.
    pub fn effective(&self) -> Option<(FileVersion, &'static str)> {
        self.python
            .map(|v| (v, "next to Python"))
            .or(self.system.map(|v| (v, "System32")))
    }
}

/// What [`decide`] concluded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// DLLs of `tool:vc-runtime` to copy next to Python (empty: nothing to do).
    pub place: Vec<&'static str>,
    /// Why ("msvcp140.dll 14.26.28720.3 in System32 is older than 14.44.35211.0").
    pub reasons: Vec<String>,
}

impl Plan {
    pub fn needed(&self) -> bool {
        !self.place.is_empty()
    }
}

/// Whether the private runtime must be placed: every [`PROBED`] DLL Windows would load for Python
/// (see [`Found::effective`]) must exist and be at least `pin`; otherwise (or with `force`) every
/// pinned DLL whose copy next to Python is missing or older than `pin` is placed (a newer copy
/// there is kept). `found` should list every DLL of `catalog::VC_RUNTIME_DLLS`.
pub fn decide(found: &[Found], pin: FileVersion, force: bool) -> Plan {
    let mut reasons = Vec::new();
    for name in PROBED {
        let f = found.iter().find(|f| f.name == name);
        match f.and_then(Found::effective) {
            None => reasons.push(format!("{name} is missing")),
            Some((v, at)) if v < pin => {
                reasons.push(format!("{name} {v} ({at}) is older than {pin}"));
            }
            Some(_) => {}
        }
    }
    if force {
        reasons.push(format!("{FORCE_ENV}=force is set"));
    }
    if reasons.is_empty() {
        return Plan::default();
    }
    let place = found
        .iter()
        .filter(|f| f.python.is_none_or(|v| v < pin))
        .map(|f| f.name)
        .collect();
    Plan { place, reasons }
}

/// "msvcp140.dll 14.26.28720.3 in System32" / "no msvcp140.dll in System32", for messages.
pub fn system_runtime(found: &[Found]) -> String {
    match found
        .iter()
        .find(|f| f.name == "msvcp140.dll")
        .and_then(|f| f.system)
    {
        Some(v) => format!("msvcp140.dll {v} in System32"),
        None => "no msvcp140.dll in System32".to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Filesystem
// ---------------------------------------------------------------------------------------------

/// The system DLL directory (`%SystemRoot%\System32`) on Windows; None elsewhere.
pub fn system_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        Some(PathBuf::from(root).join("System32"))
    }
    #[cfg(not(windows))]
    None
}

/// Versions of every pinned DLL in `system` (None: no system directory, all missing) and `python`.
pub fn probe(system: Option<&Path>, python: &Path) -> Vec<Found> {
    VC_RUNTIME_DLLS
        .iter()
        .map(|(name, _, _)| Found {
            name,
            system: system.and_then(|d| file_version(&d.join(name))),
            python: file_version(&python.join(name)),
        })
        .collect()
}

/// The `home = <dir>` of a `pyvenv.cfg`: the base interpreter's directory.
pub fn pyvenv_home(text: &str) -> Option<PathBuf> {
    text.lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        let v = v.trim();
        (k.trim().eq_ignore_ascii_case("home") && !v.is_empty()).then(|| PathBuf::from(v))
    })
}

/// The directory of the environment's base `python.exe` (from its `pyvenv.cfg`), which must be
/// inside our own `tools/python` (uv's install; checked on the resolved paths, links followed): the
/// only place DLLs are ever copied to. Returned as written in `pyvenv.cfg`.
pub fn base_python_dir(paths: &Paths) -> Result<PathBuf> {
    let cfg = paths.env.join("pyvenv.cfg");
    let incomplete = "the export environment is incomplete: click \"Remove export toolchain\" \
                      on the Config page and export again";
    let text = std::fs::read_to_string(&cfg)
        .with_context(|| format!("reading {} ({incomplete})", cfg.display()))?;
    let home = pyvenv_home(&text)
        .ok_or_else(|| anyhow!("{} has no `home` line; {incomplete}", cfg.display()))?;
    let (Ok(real_home), Ok(real_installs)) = (
        std::fs::canonicalize(&home),
        std::fs::canonicalize(&paths.python_installs),
    ) else {
        bail!(
            "the export environment's base Python {} does not exist; {incomplete}",
            home.display()
        );
    };
    if !real_home.starts_with(&real_installs) {
        bail!(
            "the export environment's base Python {} is not in {}; {incomplete}",
            home.display(),
            paths.python_installs.display()
        );
    }
    Ok(home)
}

/// Copy `names` from `src` (the verified `tool:vc-runtime` install) into `dst`, each through a
/// temporary file and a rename. Every source must carry the pinned version. Returns
/// "msvcp140.dll (was 14.26.28720.3)" / "(was missing)" lines.
pub fn place(src: &Path, dst: &Path, names: &[&str], found: &[Found]) -> Result<Vec<String>> {
    let mut placed = Vec::new();
    for name in names {
        let from = src.join(name);
        match file_version(&from) {
            Some(v) if v == FileVersion::PIN => {}
            other => bail!(
                "{} is version {}, not the pinned {}; click \"Remove export toolchain\" on the \
                 Config page and export again",
                from.display(),
                other.map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                FileVersion::PIN
            ),
        }
        let to = dst.join(name);
        let tmp = dst.join(format!(".{name}.bop-tmp"));
        let _ = std::fs::remove_file(&tmp);
        std::fs::copy(&from, &tmp)
            .with_context(|| format!("copying {} to {}", from.display(), tmp.display()))?;
        if let Err(e) = std::fs::rename(&tmp, &to) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| {
                format!(
                    "replacing {} (is another Python of the export still running? stop it, then \
                     export again)",
                    to.display()
                )
            });
        }
        let was = found
            .iter()
            .find(|f| f.name == *name)
            .and_then(|f| f.python)
            .map_or_else(|| "missing".to_string(), |v| v.to_string());
        placed.push(format!("{name} (was {was})"));
    }
    Ok(placed)
}

/// `BOP_VC_RUNTIME=force` is set.
fn forced() -> bool {
    std::env::var(FORCE_ENV).is_ok_and(|v| v.trim().eq_ignore_ascii_case("force"))
}

/// The "Visual C++ runtime" stage (Windows): probe System32 and the base Python dir; when the
/// runtime is too old, fetch `tool:vc-runtime` (never otherwise: a current system runtime means
/// no network) and copy its DLLs next to Python. Returns what was found (for failure messages).
pub(super) fn ensure(
    job: &ExportJob,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
) -> Result<Vec<Found>> {
    let p = &job.paths;
    let (os, arch) = (job.platform.os, job.platform.arch);
    let stage = |report: &mut dyn FnMut(Progress<'_>), pct: u8, detail: &str| {
        report(Progress::Stage {
            stage: Stage::Runtime,
            percent: Some(pct),
            detail,
        });
    };
    let line = |report: &mut dyn FnMut(Progress<'_>), l: &str| {
        tracing::info!("{l}");
        report(Progress::Line {
            stage: Stage::Runtime,
            line: l,
        });
    };
    let python = base_python_dir(p)?;
    let found = probe(job.system_dir.as_deref(), &python);
    let plan = decide(&found, FileVersion::PIN, forced());
    let system = system_runtime(&found);
    if !plan.needed() {
        let detail = match found
            .iter()
            .find(|f| f.name == "msvcp140.dll")
            .and_then(Found::effective)
        {
            Some((v, at)) => format!("msvcp140.dll {v} ({at}) is new enough"),
            None => "new enough".to_string(),
        };
        line(
            report,
            &format!(
                "Visual C++ runtime: {detail} (PyTorch needs {})",
                torch_min()
            ),
        );
        stage(report, 100, &detail);
        return Ok(found);
    }
    line(
        report,
        &format!(
            "Visual C++ runtime: placing the private copy {} next to Python in {}: {}",
            catalog::VC_RUNTIME_VERSION,
            python.display(),
            plan.reasons.join("; ")
        ),
    );
    let res = catalog::vc_runtime_for(os, arch)
        .ok_or_else(|| anyhow!("no pinned Visual C++ runtime for {os}/{arch}"))?;
    let installed = manager::is_installed(res, &p.vc_runtime)
        && plan.place.iter().all(|n| p.vc_runtime.join(n).is_file());
    if !installed {
        stage(report, 0, res.title);
        let fetched = fetcher.fetch(
            res,
            &p.vc_runtime,
            &mut |b, t| {
                let d = format!("{} of {} MB", stages::mb(b), stages::mb(t));
                stage(report, manager::percent(b, t).min(90), &d);
            },
            cancel,
        );
        if let Err(e) = fetched {
            if e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed) {
                return Err(e);
            }
            bail!(
                "downloading the private Visual C++ runtime ({}) failed: {e:#}. This machine's \
                 Visual C++ runtime is too old for PyTorch ({system}; PyTorch needs {} or \
                 newer). Check the network and proxy settings and export again, or install the \
                 current runtime system-wide as administrator from {VC_REDIST_URL} and export \
                 again",
                res.id,
                torch_min()
            );
        }
    }
    let placed = place(&p.vc_runtime, &python, &plan.place, &found)?;
    line(
        report,
        &format!(
            "Visual C++ runtime: placed {} {} into {}",
            placed.join(", "),
            FileVersion::PIN,
            python.display()
        ),
    );
    let after = probe(job.system_dir.as_deref(), &python);
    let still = decide(&after, FileVersion::PIN, false);
    if still.needed() {
        bail!(
            "the private Visual C++ runtime was copied into {} but is still not effective ({}); \
             install the current runtime system-wide as administrator from {VC_REDIST_URL} and \
             export again",
            python.display(),
            still.reasons.join("; ")
        );
    }
    stage(
        report,
        100,
        &format!("private runtime {} placed", FileVersion::PIN),
    );
    Ok(after)
}

/// A minimal PE32+ image with one `.rsrc` section holding a version resource of `v` (tests).
#[cfg(test)]
pub(crate) fn synthetic_dll(v: FileVersion) -> Vec<u8> {
    let mut b = vec![0u8; 0x400];
    b[..2].copy_from_slice(b"MZ");
    b[0x3C..0x40].copy_from_slice(&0x40u32.to_le_bytes());
    b[0x40..0x44].copy_from_slice(b"PE\0\0");
    let coff = 0x44;
    b[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes());
    b[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes());
    b[coff + 16..coff + 18].copy_from_slice(&240u16.to_le_bytes());
    let opt = coff + 20;
    b[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes());
    b[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes());
    let rsrc_dir = opt + 112 + 16;
    b[rsrc_dir..rsrc_dir + 4].copy_from_slice(&0x2000u32.to_le_bytes());
    b[rsrc_dir + 4..rsrc_dir + 8].copy_from_slice(&0x100u32.to_le_bytes());
    let sec = opt + 240;
    b[sec..sec + 8].copy_from_slice(b".rsrc\0\0\0");
    b[sec + 8..sec + 12].copy_from_slice(&0x100u32.to_le_bytes());
    b[sec + 12..sec + 16].copy_from_slice(&0x2000u32.to_le_bytes());
    b[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes());
    b[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes());
    // VS_VERSIONINFO header + "VS_VERSION_INFO" (UTF-16) + padding, then VS_FIXEDFILEINFO.
    let mut at = 0x200 + 6;
    for u in "VS_VERSION_INFO\0".encode_utf16() {
        b[at..at + 2].copy_from_slice(&u.to_le_bytes());
        at += 2;
    }
    at = at.next_multiple_of(4);
    let [a, bb, c, d] = v.0;
    let ms = (u32::from(a) << 16) | u32::from(bb);
    let ls = (u32::from(c) << 16) | u32::from(d);
    for (i, w) in [
        FIXED_FILE_INFO_SIGNATURE,
        FIXED_FILE_INFO_STRUC_VERSION,
        ms,
        ls,
    ]
    .iter()
    .enumerate()
    {
        b[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: FileVersion = FileVersion([14, 26, 28720, 3]);
    const NEW: FileVersion = FileVersion([14, 50, 35719, 0]);

    fn all(system: Option<FileVersion>, python: Option<FileVersion>) -> Vec<Found> {
        VC_RUNTIME_DLLS
            .iter()
            .map(|(name, _, _)| Found {
                name,
                system,
                python,
            })
            .collect()
    }

    /// uv's Python ships `vcruntime140*.dll` 14.44 next to `python.exe`; System32 has the rest.
    fn uv_python(system: Option<FileVersion>) -> Vec<Found> {
        let mut f = all(system, None);
        for x in f.iter_mut().filter(|f| f.name.starts_with("vcruntime140")) {
            if x.name != "vcruntime140_threads.dll" {
                x.python = Some(FileVersion::PIN);
            }
        }
        f
    }

    #[test]
    fn file_version_from_synthetic_and_bad_images() {
        let v = FileVersion([14, 44, 35211, 0]);
        assert_eq!(pe_file_version(&synthetic_dll(v)), Some(v));
        assert_eq!(v.to_string(), "14.44.35211.0");
        assert_eq!(FileVersion::PIN, v);
        // Not a PE, truncated headers, no signature.
        assert_eq!(pe_file_version(b""), None);
        assert_eq!(pe_file_version(b"MZ"), None);
        assert_eq!(pe_file_version(&[b'M', b'Z', 0, 0, 0xFF, 0xFF]), None);
        let mut b = synthetic_dll(v);
        b.truncate(0x22C);
        assert_eq!(pe_file_version(&b), None, "resource data cut off");
        let mut b = synthetic_dll(v);
        let sig = b
            .windows(4)
            .position(|w| w == FIXED_FILE_INFO_SIGNATURE.to_le_bytes())
            .unwrap();
        b[sig + 4] = 9; // wrong dwStrucVersion
        assert_eq!(pe_file_version(&b), None);
        let mut b = synthetic_dll(v);
        b[0x40] = b'X';
        assert_eq!(pe_file_version(&b), None, "no PE signature");
        // A huge e_lfanew does not panic.
        let mut b = synthetic_dll(v);
        b[0x3C..0x40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(pe_file_version(&b), None);
        // Ordering is numeric per field.
        assert!(FileVersion([14, 9, 0, 0]) < FileVersion([14, 40, 0, 0]));
        assert!(OLD < FileVersion::PIN && FileVersion::PIN < NEW);
        assert!(!OLD.at_least([14, 40]) && FileVersion::PIN.at_least([14, 40]));
        assert_eq!(torch_min(), "14.40");
    }

    /// Real system DLLs carry a version resource (Windows only).
    #[cfg(windows)]
    #[test]
    fn file_version_of_real_system_dlls() {
        let sys = system_dir().unwrap();
        let k = file_version(&sys.join("kernel32.dll")).expect("kernel32.dll has a FileVersion");
        assert_eq!(k.0[0], 10, "{k}");
        if let Some(v) = file_version(&sys.join("msvcp140.dll")) {
            assert_eq!(v.0[0], 14, "{v}");
        }
    }

    #[test]
    fn current_system_runtime_needs_nothing() {
        let plan = decide(&uv_python(Some(NEW)), FileVersion::PIN, false);
        assert_eq!(plan, Plan::default());
        assert!(!plan.needed());
        // Exactly the pin is enough too.
        assert!(!decide(&uv_python(Some(FileVersion::PIN)), FileVersion::PIN, false).needed());
    }

    /// The incident: System32 14.26 and no `vcruntime140_threads.dll` / `msvcp140_atomic_wait.dll`.
    #[test]
    fn old_system_runtime_places_all_but_current_python_copies() {
        let mut found = uv_python(Some(OLD));
        for f in found.iter_mut().filter(|f| {
            f.name == "vcruntime140_threads.dll" || f.name == "msvcp140_atomic_wait.dll"
        }) {
            f.system = None;
        }
        let plan = decide(&found, FileVersion::PIN, false);
        assert!(plan.needed());
        assert!(
            plan.reasons.contains(
                &"msvcp140.dll 14.26.28720.3 (System32) is older than 14.44.35211.0".into()
            ),
            "{:?}",
            plan.reasons
        );
        assert!(
            plan.reasons
                .contains(&"vcruntime140_threads.dll is missing".to_string())
        );
        // vcruntime140/_1 next to Python are current: kept, not counted as a reason.
        assert!(
            !plan
                .reasons
                .iter()
                .any(|r| r.starts_with("vcruntime140.dll"))
        );
        assert_eq!(
            plan.place,
            [
                "concrt140.dll",
                "msvcp140.dll",
                "msvcp140_1.dll",
                "msvcp140_2.dll",
                "msvcp140_atomic_wait.dll",
                "msvcp140_codecvt_ids.dll",
                "vcomp140.dll",
                "vcruntime140_threads.dll"
            ]
        );
        assert_eq!(
            system_runtime(&found),
            "msvcp140.dll 14.26.28720.3 in System32"
        );
        assert_eq!(system_runtime(&[]), "no msvcp140.dll in System32");
    }

    #[test]
    fn python_copies_win_over_system32() {
        // Placed earlier: the copies next to Python decide, whatever System32 has.
        assert!(
            !decide(
                &all(Some(OLD), Some(FileVersion::PIN)),
                FileVersion::PIN,
                false
            )
            .needed()
        );
        assert!(!decide(&all(None, Some(NEW)), FileVersion::PIN, false).needed());
        // An old copy next to Python shadows a current System32.
        let mut found = all(Some(NEW), None);
        found[1].python = Some(OLD); // msvcp140.dll
        let plan = decide(&found, FileVersion::PIN, false);
        assert_eq!(
            plan.reasons,
            ["msvcp140.dll 14.26.28720.3 (next to Python) is older than 14.44.35211.0"]
        );
        assert_eq!(plan.place.len(), VC_RUNTIME_DLLS.len());
        // Old vcruntime140 next to Python is replaced; a newer one is kept.
        let mut found = uv_python(Some(NEW));
        found
            .iter_mut()
            .find(|f| f.name == "vcruntime140.dll")
            .unwrap()
            .python = Some(OLD);
        found
            .iter_mut()
            .find(|f| f.name == "vcruntime140_1.dll")
            .unwrap()
            .python = Some(NEW);
        let plan = decide(&found, FileVersion::PIN, false);
        assert!(plan.place.contains(&"vcruntime140.dll"));
        assert!(!plan.place.contains(&"vcruntime140_1.dll"));
        // No System32 at all (a test platform): everything is missing.
        let plan = decide(&all(None, None), FileVersion::PIN, false);
        assert_eq!(plan.reasons.len(), PROBED.len());
        assert_eq!(plan.place.len(), VC_RUNTIME_DLLS.len());
    }

    #[test]
    fn force_places_what_is_not_current() {
        let plan = decide(&uv_python(Some(NEW)), FileVersion::PIN, true);
        assert_eq!(plan.reasons, ["BOP_VC_RUNTIME=force is set"]);
        assert_eq!(plan.place.len(), VC_RUNTIME_DLLS.len() - 2);
        assert!(!decide(&all(None, Some(FileVersion::PIN)), FileVersion::PIN, true).needed());
    }

    #[test]
    fn pyvenv_home_line() {
        let cfg = "home = C:\\data\\tools\\python\\cpython-3.11-windows-x86_64-none\n\
                   implementation = CPython\nuv = 0.12.19\nversion_info = 3.11.16\n";
        assert_eq!(
            pyvenv_home(cfg),
            Some(PathBuf::from(
                "C:\\data\\tools\\python\\cpython-3.11-windows-x86_64-none"
            ))
        );
        assert_eq!(
            pyvenv_home("HOME=/opt/py/bin"),
            Some(PathBuf::from("/opt/py/bin"))
        );
        assert_eq!(pyvenv_home("home =\nversion = 3.11"), None);
        assert_eq!(pyvenv_home("homes = x"), None);
    }

    #[test]
    fn base_python_dir_must_be_ours_and_place_copies() {
        let root = std::env::temp_dir().join(format!("bop-vcrt-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root, root.join("models"));
        assert!(base_python_dir(&paths).is_err(), "no pyvenv.cfg");
        let base = paths.python_installs.join("cpython-3.11.16-x");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&paths.env).unwrap();
        let outside = root.join("elsewhere");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(
            paths.env.join("pyvenv.cfg"),
            format!("home = {}\n", outside.display()),
        )
        .unwrap();
        let e = base_python_dir(&paths).unwrap_err();
        assert!(
            format!("{e:#}").contains("Remove export toolchain"),
            "{e:#}"
        );
        std::fs::write(
            paths.env.join("pyvenv.cfg"),
            format!("home = {}\n", base.display()),
        )
        .unwrap();
        let dir = base_python_dir(&paths).unwrap();
        assert_eq!(dir, base);

        // place: only pinned-version sources, through a temp file, reporting what was there.
        std::fs::create_dir_all(&paths.vc_runtime).unwrap();
        std::fs::write(
            paths.vc_runtime.join("msvcp140.dll"),
            synthetic_dll(FileVersion::PIN),
        )
        .unwrap();
        std::fs::write(paths.vc_runtime.join("vcomp140.dll"), synthetic_dll(OLD)).unwrap();
        std::fs::write(dir.join("msvcp140.dll"), synthetic_dll(OLD)).unwrap();
        let found = probe(None, &dir);
        let placed = place(&paths.vc_runtime, &dir, &["msvcp140.dll"], &found).unwrap();
        assert_eq!(placed, ["msvcp140.dll (was 14.26.28720.3)"]);
        assert_eq!(
            file_version(&dir.join("msvcp140.dll")),
            Some(FileVersion::PIN)
        );
        let e = place(&paths.vc_runtime, &dir, &["vcomp140.dll"], &found).unwrap_err();
        assert!(format!("{e:#}").contains("not the pinned"), "{e:#}");
        assert!(!dir.join("vcomp140.dll").exists());
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("bop-tmp"))
            .collect();
        assert!(leftovers.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
