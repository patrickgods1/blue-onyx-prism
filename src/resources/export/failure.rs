//! Classifies a failed child process of the export (the env self-check, the export script, uv)
//! from its last output lines ([`ChildFailed::lines`]) into a message with the likely cause and
//! the next action. Pure; an unclassified failure keeps the plain "failed; last output" text.

use super::vc_runtime::{FileVersion, VC_REDIST_URL, torch_min};
use super::*;

/// What the classifier knows besides the output.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FailureContext<'a> {
    /// The export runs on Windows.
    pub windows: bool,
    /// System32's `msvcp140.dll` version (None: missing or not probed).
    pub system_msvcp140: Option<FileVersion>,
    /// `<data root>/tools`.
    pub tools: &'a Path,
    /// The child is our Python script (self-check or export), whose own lines start with
    /// `[export_model]`.
    pub python_script: bool,
}

/// Prefix of the export script's own log lines.
const OWN_PREFIX: &str = "[export_model]";

const REMOVE_TOOLCHAIN: &str =
    "click \"Remove export toolchain\" on the Config page and export again";

/// The first line containing any of `needles` (ASCII case-insensitive).
fn find<'a>(lines: &'a [String], needles: &[&str]) -> Option<&'a str> {
    lines.iter().map(String::as_str).find(|l| {
        let low = l.to_ascii_lowercase();
        needles
            .iter()
            .any(|n| low.contains(&n.to_ascii_lowercase()))
    })
}

/// `x.dll` of `Error loading "C:\...\x.dll" or one of its dependencies.`
fn error_loading(line: &str) -> Option<&str> {
    let rest = &line[line.find("Error loading \"")? + "Error loading \"".len()..];
    let path = &rest[..rest.find('"')?];
    path.rsplit(['\\', '/']).next().filter(|s| !s.is_empty())
}

/// `X` of `... while importing X: ...`.
fn while_importing(line: &str) -> Option<&str> {
    let rest = &line[line.find("while importing ")? + "while importing ".len()..];
    let end = rest.find([':', ' ']).unwrap_or(rest.len());
    Some(&rest[..end]).filter(|s| !s.is_empty())
}

/// `x` of `No module named 'x'` / `cannot import name 'y' from 'x'`.
fn missing_module(line: &str) -> Option<&str> {
    let quoted = |s: &'_ str| -> Option<(usize, usize)> {
        let a = s.find('\'')? + 1;
        Some((a, a + s[a..].find('\'')?))
    };
    if let Some(i) = line.find("No module named ") {
        let rest = &line[i..];
        let (a, b) = quoted(rest)?;
        return Some(&rest[a..b]);
    }
    if let Some(i) = line.find(" from '") {
        let rest = &line[i..];
        let (a, b) = quoted(rest)?;
        return Some(&rest[a..b]);
    }
    None
}

/// Cut a quoted output line for a message.
fn quote(line: &str) -> String {
    let l = line.trim();
    let cut: String = l.chars().take(200).collect();
    if cut.len() < l.len() {
        format!("\"{cut}...\"")
    } else {
        format!("\"{cut}\"")
    }
}

/// The likely cause and next action of a failed child from its output, None when nothing
/// matches (see the module docs).
pub(crate) fn classify(lines: &[String], ctx: &FailureContext<'_>) -> Option<String> {
    if let Some(l) = find(
        lines,
        &[
            "No space left on device",
            "[WinError 112]",
            "[Errno 28]",
            "not enough space on the disk",
        ],
    ) {
        return Some(format!(
            "the disk holding {} is full ({}); free a few GB there (or set download_dir to a \
             larger disk), then export again",
            ctx.tools.display(),
            quote(l)
        ));
    }
    if let Some(l) = find(
        lines,
        &[
            "MemoryError",
            "[WinError 1455]",
            "paging file is too small",
            "out of memory",
            "Cannot allocate memory",
            "not enough memory",
        ],
    ) {
        let pagefile = if ctx.windows {
            " and a paging file large enough to map its libraries (WinError 1455 means the \
             paging file is too small: let Windows manage its size under System Properties > \
             Advanced > Performance > Virtual memory)"
        } else {
            ""
        };
        return Some(format!(
            "the export ran out of memory ({}): PyTorch needs a few GB of free RAM{pagefile}; \
             close other programs and export again",
            quote(l)
        ));
    }
    if ctx.windows {
        let av_in_import = find(lines, &["Windows fatal exception: access violation"]).is_some()
            && find(lines, &["importlib._bootstrap", "_load_dll_libraries"]).is_some();
        let runtime = find(
            lines,
            &[
                "[WinError 1114]",
                "DLL initialization routine failed",
                "[WinError 127]",
            ],
        )
        .or(av_in_import
            .then(|| find(lines, &["Windows fatal exception: access violation"]))
            .flatten());
        if let Some(l) = runtime {
            let lib = lines
                .iter()
                .find_map(|l| error_loading(l))
                .map(|d| format!(" ({d})"))
                .unwrap_or_default();
            let found = ctx.system_msvcp140.map_or_else(
                || "found no msvcp140.dll in System32".to_string(),
                |v| format!("found msvcp140.dll {v} in System32"),
            );
            return Some(format!(
                "Python could not load PyTorch's native libraries{lib} ({}): the Microsoft Visual \
                 C++ runtime on this machine is too old or broken for PyTorch ({found}, needs {} \
                 or newer). Blue Onyx Prism installs a private copy for the export; if this still \
                 fails, install the current runtime system-wide as administrator from \
                 {VC_REDIST_URL} and export again",
                quote(l),
                torch_min()
            ));
        }
    }
    if let Some(l) = find(lines, &["[WinError 126]", "DLL load failed"]) {
        let what = while_importing(l)
            .map(|m| format!("the Python module {m}"))
            .or_else(|| error_loading(l).map(|d| d.to_string()))
            .unwrap_or_else(|| "a Python module".to_string());
        return Some(format!(
            "a library needed by {what} could not be loaded ({}): the export environment is \
             incomplete or a system library is missing; {REMOVE_TOOLCHAIN}. If it still fails, \
             install the current Microsoft Visual C++ runtime as administrator from \
             {VC_REDIST_URL}",
            quote(l)
        ));
    }
    if let Some(l) = find(lines, &["ModuleNotFoundError", "ImportError"]) {
        let module = missing_module(l)
            .map(|m| format!(" (module {m})"))
            .unwrap_or_default();
        return Some(format!(
            "the export environment is incomplete{module} ({}); {REMOVE_TOOLCHAIN} (the pinned \
             packages are installed again)",
            quote(l)
        ));
    }
    if ctx.python_script {
        if let Some(l) = find(lines, &["Fatal Python error", "Windows fatal exception"]) {
            return Some(format!(
                "Python crashed in native code ({}); the log above has its faulthandler stack. \
                 Export again; if it keeps crashing, {REMOVE_TOOLCHAIN}",
                quote(l)
            ));
        }
        let own_only = lines.iter().all(|l| l.trim_start().starts_with(OWN_PREFIX));
        if own_only {
            let after = lines
                .last()
                .map(|l| format!(" after {}", quote(l)))
                .unwrap_or_default();
            return Some(format!(
                "Python exited without an error message{after} (likely a native crash in a \
                 library); the log above has a faulthandler stack if any. Export again; if it \
                 keeps failing, {REMOVE_TOOLCHAIN}"
            ));
        }
    }
    None
}

/// `e` with the classified cause in front when it is a [`ChildFailed`] that [`classify`]
/// recognizes; otherwise unchanged.
pub(crate) fn explain(e: anyhow::Error, ctx: &FailureContext<'_>) -> anyhow::Error {
    let Some(child) = e.downcast_ref::<ChildFailed>() else {
        return e;
    };
    match classify(&child.lines, ctx) {
        Some(msg) => anyhow!("{msg} [{child}]"),
        None => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(str::to_string).collect()
    }

    fn ctx(windows: bool) -> FailureContext<'static> {
        FailureContext {
            windows,
            system_msvcp140: Some(FileVersion([14, 26, 28720, 3])),
            tools: Path::new("D:/bop/tools"),
            python_script: true,
        }
    }

    /// The incident's output, run with `-X faulthandler`.
    const INCIDENT: &str = r#"[export_model] yolo26n: exporting ONNX (opset 17)
Windows fatal exception: access violation

Current thread 0x00001a2c (most recent call first):
  File "C:\ProgramData\BlueOnyxPrism\tools\export-env\Lib\site-packages\torch\__init__.py", line 281 in _load_dll_libraries
  File "C:\ProgramData\BlueOnyxPrism\tools\export-env\Lib\site-packages\torch\__init__.py", line 296 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
Traceback (most recent call last):
  File "C:\ProgramData\BlueOnyxPrism\tools\export-work\export_model.py", line 322, in ultralytics_model
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\ProgramData\BlueOnyxPrism\tools\export-env\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies."#;

    #[test]
    fn vc_runtime_failures() {
        let m = classify(&lines(INCIDENT), &ctx(true)).unwrap();
        assert!(
            m.starts_with("Python could not load PyTorch's native libraries (c10.dll)"),
            "{m}"
        );
        assert!(
            m.contains("too old or broken for PyTorch (found msvcp140.dll 14.26.28720.3 in System32, needs 14.40 or newer)"),
            "{m}"
        );
        assert!(
            m.contains("https://aka.ms/vs/17/release/vc_redist.x64.exe"),
            "{m}"
        );
        // The access violation alone (no OSError yet) during an import.
        let av: String = INCIDENT.lines().take(7).collect::<Vec<_>>().join("\n");
        let m = classify(&lines(&av), &ctx(true)).unwrap();
        assert!(m.contains("Visual C++ runtime"), "{m}");
        // Not on Linux/macOS: there it is a generic native crash.
        let m = classify(&lines(&av), &ctx(false)).unwrap();
        assert!(m.starts_with("Python crashed in native code"), "{m}");
        let c = FailureContext {
            system_msvcp140: None,
            ..ctx(true)
        };
        let m = classify(&lines("OSError: [WinError 127] The specified procedure could not be found. Error loading \"C:\\x\\torch\\lib\\torch_cpu.dll\" or one of its dependencies."), &c).unwrap();
        assert!(
            m.contains("(torch_cpu.dll)") && m.contains("found no msvcp140.dll"),
            "{m}"
        );
    }

    #[test]
    fn missing_dll_and_modules() {
        let m = classify(
            &lines("Traceback (most recent call last):\nImportError: DLL load failed while importing _C: The specified module could not be found."),
            &ctx(true),
        )
        .unwrap();
        assert!(
            m.starts_with("a library needed by the Python module _C could not be loaded"),
            "{m}"
        );
        assert!(m.contains("Remove export toolchain"), "{m}");
        let m = classify(
            &lines("OSError: [WinError 126] The specified module could not be found. Error loading \"C:\\e\\torch\\lib\\fbgemm.dll\" or one of its dependencies."),
            &ctx(true),
        )
        .unwrap();
        assert!(m.starts_with("a library needed by fbgemm.dll"), "{m}");
        let m = classify(
            &lines("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nModuleNotFoundError: No module named 'onnxruntime'"),
            &ctx(false),
        )
        .unwrap();
        assert!(
            m.starts_with("the export environment is incomplete (module onnxruntime)"),
            "{m}"
        );
        assert!(m.contains("Remove export toolchain"), "{m}");
        let m = classify(
            &lines("ImportError: cannot import name 'RfDetrForObjectDetection' from 'transformers' (x)"),
            &ctx(false),
        )
        .unwrap();
        assert!(m.contains("(module transformers)"), "{m}");
    }

    #[test]
    fn memory_and_disk() {
        let m = classify(
            &lines("OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading \"C:\\e\\torch\\lib\\c10.dll\""),
            &ctx(true),
        )
        .unwrap();
        assert!(m.starts_with("the export ran out of memory"), "{m}");
        assert!(m.contains("paging file"), "{m}");
        let m = classify(&lines("MemoryError"), &ctx(false)).unwrap();
        assert!(
            m.starts_with("the export ran out of memory") && !m.contains("paging"),
            "{m}"
        );
        let m = classify(
            &lines("RuntimeError: [enforce fail at alloc_cpu.cpp:115] data. DefaultCPUAllocator: not enough memory: you tried to allocate 123 bytes."),
            &ctx(true),
        )
        .unwrap();
        assert!(m.contains("out of memory"), "{m}");
        let m = classify(
            &lines("OSError: [Errno 28] No space left on device: '/data/tools/export-work/x'"),
            &ctx(false),
        )
        .unwrap();
        assert!(
            m.starts_with("the disk holding D:/bop/tools is full"),
            "{m}"
        );
        let m = classify(
            &lines("error: Failed to write: [WinError 112] There is not enough space on the disk."),
            &FailureContext {
                python_script: false,
                ..ctx(true)
            },
        )
        .unwrap();
        assert!(m.contains("is full"), "{m}");
    }

    #[test]
    fn silent_exit_and_unclassified() {
        let m = classify(
            &lines("[export_model] yolo26n: exporting ONNX (opset 17)"),
            &ctx(true),
        )
        .unwrap();
        assert!(
            m.starts_with("Python exited without an error message after \"[export_model] yolo26n: exporting ONNX (opset 17)\" (likely a native crash in a library)"),
            "{m}"
        );
        assert!(
            classify(&[], &ctx(true))
                .unwrap()
                .starts_with("Python exited without")
        );
        // uv's own failures are not "silent Python exits".
        let uv = FailureContext {
            python_script: false,
            ..ctx(true)
        };
        assert_eq!(classify(&[], &uv), None);
        assert_eq!(
            classify(
                &lines("error: Failed to fetch: `https://x`\n  Caused by: timed out"),
                &uv
            ),
            None
        );
        // Some other Python error stays as it is.
        assert_eq!(
            classify(
                &lines("[export_model] x\nValueError: bad shape"),
                &ctx(true)
            ),
            None
        );
    }

    #[test]
    fn explain_wraps_only_recognized_child_failures() {
        let child = ChildFailed {
            program: "python.exe".into(),
            status: "exit code: 255".into(),
            lines: lines(INCIDENT),
        };
        let e = explain(anyhow::Error::new(child), &ctx(true));
        let s = format!("{e:#}");
        assert!(
            s.starts_with("Python could not load PyTorch's native libraries"),
            "{s}"
        );
        assert!(
            s.contains("[python.exe failed (exit code: 255); last output: "),
            "{s}"
        );
        let e = explain(anyhow!("plain"), &ctx(true));
        assert_eq!(format!("{e:#}"), "plain");
        let child = ChildFailed {
            program: "uv.exe".into(),
            status: "exit code: 2".into(),
            lines: lines("error: nope"),
        };
        let e = explain(
            anyhow::Error::new(child),
            &FailureContext {
                python_script: false,
                ..ctx(true)
            },
        );
        assert_eq!(
            format!("{e:#}"),
            "uv.exe failed (exit code: 2); last output: error: nope"
        );
    }
}
