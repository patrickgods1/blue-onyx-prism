//! OpenVINO runtime archives ([`Layout::OpenVino`]): which `runtime/...` libraries and license
//! notices are kept (layout preserved, per-OS plugin whitelist), the post-extraction install
//! check, and the macOS TBB mirror next to `libopenvino`. Entry paths reach this file already
//! checked by the drivers in `mod.rs`.

use super::*;

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

/// License notices of the OpenVINO archive (path below the top folder), kept in place (NR-17).
pub(crate) const OPENVINO_LICENSE_FILES: &[&str] = &[
    "docs/licensing/LICENSE",
    "docs/licensing/runtime-third-party-programs.txt",
    "docs/licensing/onednn_third-party-programs.txt",
    "docs/licensing/onetbb_third-party-programs.txt",
    "runtime/3rdparty/tbb/TBB-LICENSE",
];

/// `rel` (top folder stripped) is one of [`OPENVINO_LICENSE_FILES`].
pub(crate) fn openvino_license_wanted(rel: &str) -> Option<String> {
    OPENVINO_LICENSE_FILES
        .contains(&rel)
        .then(|| rel.to_string())
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
pub(super) fn mirror_tbb_into_lib_dir(
    sel: &Selection,
    dest: &Path,
    got: &mut BTreeSet<String>,
) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::super::test_util::*;
    use super::*;

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
}
