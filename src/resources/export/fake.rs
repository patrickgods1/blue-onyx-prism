//! A fake toolchain for tests (no network, no Python): records commands and writes what the
//! real tools would.

use super::*;
use std::sync::atomic::AtomicUsize;

/// Writes what the real tools would: manifests for fetched resources, the venv's python, the
/// exporter's output. Records every command.
#[derive(Default)]
pub struct Fake {
    pub commands: Mutex<Vec<Cmd>>,
    pub fetched: Mutex<Vec<String>>,
    /// Fail the command whose display contains this.
    pub fail_on: Option<&'static str>,
    /// Block in the command whose display contains this until cancelled.
    pub hang_on: Option<&'static str>,
    /// Classes the fake exporter writes.
    pub classes: usize,
    pub fetch_fails: Option<&'static str>,
    pub runs: AtomicUsize,
}

impl Fake {
    pub fn new() -> Self {
        Self {
            classes: 80,
            ..Default::default()
        }
    }

    pub fn cmds(&self) -> Vec<String> {
        self.commands
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.display())
            .collect()
    }
}

pub fn arg_after(cmd: &Cmd, flag: &str) -> Option<PathBuf> {
    let i = cmd.args.iter().position(|a| a == flag)?;
    cmd.args.get(i + 1).map(PathBuf::from)
}

impl CommandRunner for Fake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.commands.lock().unwrap().push(cmd.clone());
        let d = cmd.display();
        if self.hang_on.is_some_and(|h| d.contains(h)) {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(10));
            }
            return Err(anyhow!(Cancelled));
        }
        if self.fail_on.is_some_and(|f| d.contains(f)) {
            on_line("error: something broke");
            bail!(
                "{} failed (exit status: 2); last output: error: something broke",
                d
            );
        }
        if d.contains(" venv ") {
            let env = PathBuf::from(cmd.args.last().unwrap());
            let windows = cmd.program.extension().is_some_and(|e| e == "exe");
            let py = if windows {
                env.join("Scripts/python.exe")
            } else {
                env.join("bin/python")
            };
            std::fs::create_dir_all(py.parent().unwrap()).unwrap();
            std::fs::write(py, b"#!python").unwrap();
        }
        if d.contains(" pip install ") {
            on_line("Resolved 48 packages in 3ms");
            on_line("Downloading torch (75.3MiB)");
            on_line(" Downloaded torch");
            on_line("Prepared 48 packages in 40s");
            on_line("Installed 48 packages in 2s");
        }
        if d.contains(SCRIPT_NAME) {
            let name = arg_after(cmd, "--name").unwrap().display().to_string();
            on_line(&format!("[export_model] {name}: exporting ONNX (opset 17)"));
            let out = arg_after(cmd, "--out-dir").unwrap();
            let names: String = (0..self.classes).map(|i| format!("  - c{i}\n")).collect();
            std::fs::write(out.join(format!("{name}.onnx")), vec![7u8; 4096]).unwrap();
            std::fs::write(out.join(format!("{name}.yaml")), format!("NAMES:\n{names}")).unwrap();
        }
        Ok(())
    }
}

impl Fetcher for Fake {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        _cancel: &AtomicBool,
    ) -> Result<()> {
        self.fetched.lock().unwrap().push(res.id.to_string());
        if self.fetch_fails == Some(res.id) {
            bail!("hash mismatch for {}", res.parts[0].file_name);
        }
        progress(res.size() / 2, res.size());
        std::fs::create_dir_all(dir).unwrap();
        let mut files = Vec::new();
        if res.id == catalog::UV_ID {
            std::fs::write(dir.join("uv"), b"uv").unwrap();
            std::fs::write(dir.join("uv.exe"), b"uv").unwrap();
            files.push("uv".to_string());
        } else if res.kind == ResourceKind::Tool {
            std::fs::create_dir_all(dir.join("engine")).unwrap();
            std::fs::create_dir_all(dir.join("configs")).unwrap();
            std::fs::write(dir.join("engine/__init__.py"), b"").unwrap();
            std::fs::write(dir.join("configs/a.yml"), b"").unwrap();
            files.push("engine/__init__.py".to_string());
            files.push("configs/a.yml".to_string());
        } else {
            for p in res.parts {
                std::fs::write(dir.join(p.file_name), b"weights").unwrap();
                files.push(p.file_name.to_string());
            }
        }
        let m = Manifest {
            id: res.id.to_string(),
            version: res.version.to_string(),
            sha256: vec![],
            files,
        };
        std::fs::write(
            manager::manifest_path(res, dir),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        Ok(())
    }
}
