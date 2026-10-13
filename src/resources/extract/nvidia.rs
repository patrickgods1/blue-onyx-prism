//! NVIDIA CUDA / cuDNN wheels ([`Layout::NvidiaWheel`]): shared libraries from
//! `nvidia/<component>/{bin,lib}/` written flat, the soname links a wheel leaves out, and the
//! check that the CUDA provider's dependencies are all present.

use super::*;

// ---------------------------------------------------------------------------------------------
// NVIDIA wheels
// ---------------------------------------------------------------------------------------------

/// `nvidia/<component>/bin/<x>.dll` (Windows) or `nvidia/<component>/lib/lib<x>.so[.N]`
/// (Linux) -> the flat file name.
pub(crate) fn wheel_wanted(windows: bool, entry: &str) -> Option<String> {
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
pub(crate) fn cuda_provider_needs(windows: bool) -> &'static [&'static str] {
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
pub(crate) fn soname_links(names: &BTreeSet<String>) -> Vec<(String, String)> {
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
pub(super) fn wheel_check(
    windows: bool,
    dest: &Path,
    got: &mut BTreeSet<String>,
) -> Result<Vec<String>> {
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
    use super::super::test_util::*;
    use super::*;

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
}
