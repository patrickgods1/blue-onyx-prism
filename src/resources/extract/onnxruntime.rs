//! ONNX Runtime and DirectML packages ([`Layout::OrtRelease`], [`Layout::NugetOrt`],
//! [`Layout::NugetDirectMl`]): the per-flavor shared-library whitelist (written flat), the
//! license notices under `licenses/`, and the check that the core library landed.

use super::*;

// ---------------------------------------------------------------------------------------------
// ONNX Runtime / DirectML
// ---------------------------------------------------------------------------------------------

/// File naming the flavor in an ONNX Runtime install directory.
pub(crate) const ORT_FLAVOR_FILE: &str = "flavor.txt";

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

pub(super) fn stem_allowed(stem: &str, flavor: Flavor) -> bool {
    CORE_STEMS.contains(&stem) || (flavor == Flavor::Cuda && GPU_STEMS.contains(&stem))
}

/// Where the ONNX Runtime license notices go in an install directory.
pub(crate) const ORT_LICENSE_DIR: &str = "licenses/onnxruntime";
/// Where the DirectML license notices go in an install directory.
pub(crate) const DIRECTML_LICENSE_DIR: &str = "licenses/directml";

/// License notices of the ONNX Runtime / DirectML archives (NR-17): the release archives and the
/// ONNX Runtime NuGet have `LICENSE` + `ThirdPartyNotices.txt` (below the top folder, or at the
/// root of the package), the DirectML NuGet `LICENSE.txt` + `ThirdPartyNotices.txt` at its root.
/// Returns the destination below `licenses/` (never flat, so the two packages of the DirectML
/// flavor cannot overwrite each other's notices).
fn ort_license_wanted(layout: Layout, dirs: &[&str], name: &str) -> Option<String> {
    let (at_place, dir, names): (bool, &str, &[&str]) = match layout {
        Layout::OrtRelease => (
            dirs.len() == 1,
            ORT_LICENSE_DIR,
            &["LICENSE", "ThirdPartyNotices.txt"],
        ),
        Layout::NugetOrt => (
            dirs.is_empty(),
            ORT_LICENSE_DIR,
            &["LICENSE", "ThirdPartyNotices.txt"],
        ),
        Layout::NugetDirectMl => (
            dirs.is_empty(),
            DIRECTML_LICENSE_DIR,
            &["LICENSE.txt", "ThirdPartyNotices.txt"],
        ),
        _ => return None,
    };
    (at_place && names.contains(&name)).then(|| format!("{dir}/{name}"))
}

/// Decide whether an ONNX Runtime / DirectML archive entry belongs to the runtime subset.
/// Returns the flat destination file name (the entry's last component, itself whitelisted), or
/// `licenses/<component>/<name>` for a license notice.
pub(crate) fn ort_wanted(
    layout: Layout,
    flavor: Flavor,
    windows: bool,
    entry: &str,
) -> Option<String> {
    let parts = components(entry)?;
    let (name, dirs) = parts.split_last()?;
    let dirs: Vec<&str> = dirs.iter().map(String::as_str).collect();
    if let Some(license) = ort_license_wanted(layout, &dirs, name) {
        return Some(license);
    }
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
        Layout::OpenVino
        | Layout::NvidiaWheel
        | Layout::Uv
        | Layout::SourceTree
        | Layout::VcRuntime
        | Layout::File => None,
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

#[cfg(test)]
mod tests {
    use super::super::test_util::*;
    use super::*;

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

    /// NR-17: each runtime keeps its license notices (names as in the pinned 1.24.4 / 2026.4
    /// archives and the DirectML 1.15.4 NuGet), under `licenses/` so the two DirectML packages
    /// cannot collide; other top-level files are still dropped.
    #[test]
    fn license_notices_are_kept() {
        for (layout, top) in [
            (Layout::OrtRelease, "onnxruntime-linux-x64-1.24.4/"),
            (Layout::OrtRelease, "onnxruntime-win-x64-1.24.4/"),
            (Layout::NugetOrt, ""),
        ] {
            let w = |e: &str| ort_wanted(layout, Flavor::DirectMl, true, &format!("{top}{e}"));
            assert_eq!(
                w("LICENSE").as_deref(),
                Some("licenses/onnxruntime/LICENSE")
            );
            assert_eq!(
                w("ThirdPartyNotices.txt").as_deref(),
                Some("licenses/onnxruntime/ThirdPartyNotices.txt")
            );
            assert!(w("README.md").is_none() && w("Privacy.md").is_none());
            assert!(w("lib/LICENSE").is_none() && w("runtimes/LICENSE").is_none());
        }
        let d = |e: &str| ort_wanted(Layout::NugetDirectMl, Flavor::DirectMl, true, e);
        assert_eq!(
            d("LICENSE.txt").as_deref(),
            Some("licenses/directml/LICENSE.txt")
        );
        assert_eq!(
            d("ThirdPartyNotices.txt").as_deref(),
            Some("licenses/directml/ThirdPartyNotices.txt")
        );
        assert!(d("LICENSE").is_none() && d("bin/LICENSE.txt").is_none());
        // OpenVINO: layout kept below the top folder.
        for rel in OPENVINO_LICENSE_FILES {
            assert_eq!(openvino_license_wanted(rel).as_deref(), Some(*rel));
        }
        for rel in [
            "docs/licensing/EULA.txt",
            "docs/licensing/documentation-third-party-programs.txt",
            "LICENSE",
            "docs/licensing/../LICENSE",
        ] {
            assert!(openvino_license_wanted(rel).is_none(), "{rel}");
        }
    }

    /// An ONNX Runtime release tarball with its notices extracts them under `licenses/` and the
    /// install check still finds the library; a set stop flag cancels the extraction.
    #[test]
    fn ort_tgz_keeps_licenses_and_honors_the_stop_flag() {
        let dir = TempDir::new("licenses");
        let archive = dir.join("ort.tgz");
        tgz_with(
            &archive,
            &[
                TarEntry::File("ort-1/LICENSE", b"MIT", 0o644),
                TarEntry::File("ort-1/ThirdPartyNotices.txt", b"notices", 0o644),
                TarEntry::File("ort-1/README.md", b"no", 0o644),
                TarEntry::File("ort-1/lib/libonnxruntime.so.1.24.4", b"lib", 0o644),
            ],
        );
        let t = Target {
            os: "linux",
            arch: "x86_64",
        };
        let dest = dir.join("out");
        let stop = AtomicBool::new(false);
        let got = extract_part_until(
            &archive,
            ArchiveKind::TarGz,
            Layout::OrtRelease,
            Some(Flavor::Cpu),
            t,
            &dest,
            &stop,
        )
        .unwrap();
        assert_eq!(
            got,
            set(&[
                "libonnxruntime.so.1.24.4",
                "licenses/onnxruntime/LICENSE",
                "licenses/onnxruntime/ThirdPartyNotices.txt",
            ])
        );
        assert_eq!(
            std::fs::read(dest.join("licenses/onnxruntime/LICENSE")).unwrap(),
            b"MIT"
        );
        assert!(!dest.join("README.md").exists());
        ort_check(Flavor::Cpu, false, &got).unwrap();

        stop.store(true, Ordering::SeqCst);
        let e = extract_part_until(
            &archive,
            ArchiveKind::TarGz,
            Layout::OrtRelease,
            Some(Flavor::Cpu),
            t,
            &dir.join("out2"),
            &stop,
        )
        .unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
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
}
