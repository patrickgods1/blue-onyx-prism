//! The stage machine: [`run_export`] runs uv -> Python -> packages -> weights -> export ->
//! install over the two seams, reporting [`Progress`] and honoring cancellation. Installing the
//! outputs is all-or-nothing (previous files are restored on failure).

use super::*;

/// One export to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportJob {
    pub resource: &'static Resource,
    pub paths: Paths,
    pub platform: Platform,
    pub timeouts: Timeouts,
}

impl ExportJob {
    pub fn new(resource: &'static Resource, paths: Paths) -> Self {
        Self {
            resource,
            paths,
            platform: Platform::current(),
            timeouts: Timeouts::default(),
        }
    }
}

/// Name of the lock file written into `tools/` for `uv pip install -r`.
pub(super) const LOCKFILE_NAME: &str = "model-export-requirements.lock.txt";

/// Global uv flags: no user/project config, no colors, our cache.
fn uv_cmd(job: &ExportJob, stage: Stage, timeout: Duration, args: &[&std::ffi::OsStr]) -> Cmd {
    let p = &job.paths;
    let windows = job.platform.os == "windows";
    let mut all: Vec<OsString> = vec!["--no-config".into(), "--color".into(), "never".into()];
    all.push("--cache-dir".into());
    all.push(p.uv_cache.clone().into_os_string());
    all.extend(args.iter().map(|a| a.to_os_string()));
    Cmd {
        stage,
        program: p.uv_exe(windows),
        args: all,
        env: vec![
            (
                "UV_PYTHON_INSTALL_DIR".into(),
                p.python_installs.clone().into_os_string(),
            ),
            ("UV_NO_CONFIG".into(), "1".into()),
            ("UV_NO_MODIFY_PATH".into(), "1".into()),
        ],
        cwd: p.tools.clone(),
        timeout,
    }
}

/// Progress of `uv pip install` from its output: "Downloading torch (75.3MiB)", " Downloaded
/// torch", "Prepared 48 packages in 40s", "Installed 48 packages in 2s".
#[derive(Debug, Default)]
pub(super) struct PackagesProgress {
    downloading: usize,
    downloaded: usize,
}

impl PackagesProgress {
    pub(super) fn line(&mut self, line: &str) -> Option<(u8, String)> {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("Downloading ") {
            self.downloading += 1;
            return Some((self.percent(), format!("downloading {rest}")));
        }
        if let Some(rest) = l.strip_prefix("Downloaded ") {
            self.downloaded += 1;
            return Some((self.percent(), format!("downloaded {rest}")));
        }
        if l.starts_with("Prepared ") {
            return Some((90, l.to_string()));
        }
        if l.starts_with("Installed ") {
            return Some((100, l.to_string()));
        }
        None
    }

    fn percent(&self) -> u8 {
        let p = (self.downloaded.min(self.downloading).saturating_mul(85))
            .checked_div(self.downloading)
            .unwrap_or(0);
        u8::try_from(p).unwrap_or(85)
    }
}

fn python_line(report: &mut dyn FnMut(Progress<'_>), line: &str) {
    report(Progress::Line {
        stage: Stage::Python,
        line,
    });
}

fn mb(b: u64) -> String {
    format!("{:.1}", b as f64 / 1e6)
}

/// Run every stage of `job` (see the module docs). Progress and child output go to `report`.
pub(crate) fn run_export(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
) -> Result<ExportOutput, ExportError> {
    let mut stage: Option<Stage> = None;
    let result = run_stages(job, runner, fetcher, report, cancel, &mut stage);
    result.map_err(|e| {
        let cancelled = e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed);
        ExportError {
            stage,
            message: if cancelled {
                "cancelled".to_string()
            } else {
                format!("{e:#}")
            },
            cancelled,
        }
    })
}

fn run_stages(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
    current: &mut Option<Stage>,
) -> Result<ExportOutput> {
    let res = job.resource;
    let p = &job.paths;
    let (os, arch) = (job.platform.os, job.platform.arch);
    let windows = os == "windows";
    if res.kind != ResourceKind::ExportModel {
        bail!("{} is not an exportable model", res.id);
    }
    let name = res
        .model_name()
        .ok_or_else(|| anyhow!("{} has no model name", res.id))?;
    let (onnx_name, yaml_name) =
        catalog::export_files(res).ok_or_else(|| anyhow!("{} is not exportable", res.id))?;
    if let Some(why) = unsupported_reason(os, arch) {
        bail!(why);
    }
    let lock = package_lock(os, arch).ok_or_else(|| anyhow!("no package lock"))?;
    let recipe =
        catalog::export_recipe(res).ok_or_else(|| anyhow!("{} has no export recipe", res.id))?;
    let family = match res.provides {
        catalog::Provides::Model { family, .. } => family,
        _ => bail!("{} is not a model", res.id),
    };
    let check = |cancel: &AtomicBool| -> Result<()> {
        if cancel.load(Ordering::Relaxed) {
            Err(anyhow!(Cancelled))
        } else {
            Ok(())
        }
    };
    macro_rules! enter {
        ($stage:expr, $pct:expr, $detail:expr) => {{
            *current = Some($stage);
            report(Progress::Stage {
                stage: $stage,
                percent: $pct,
                detail: $detail,
            });
        }};
    }

    std::fs::create_dir_all(&p.tools).with_context(|| format!("creating {}", p.tools.display()))?;
    enter!(Stage::Uv, None, "checking the toolchain");
    let _lock = ExportLock::acquire(&p.lock_file(), cancel, job.timeouts.lock_wait, || {
        tracing::info!(
            "another process is exporting a model; waiting up to {} min",
            job.timeouts.lock_wait.as_secs() / 60
        );
    })?;

    // 1. uv
    let uv_res = catalog::uv_for(os, arch).ok_or_else(|| anyhow!("no uv for {os}/{arch}"))?;
    let uv = p.uv_exe(windows);
    if manager::is_installed(uv_res, &p.uv_dir) && uv.is_file() {
        enter!(Stage::Uv, Some(100), "uv is installed");
    } else {
        enter!(Stage::Uv, Some(0), uv_res.title);
        fetcher.fetch(
            uv_res,
            &p.uv_dir,
            &mut |b, t| {
                let pct = manager::percent(b, t);
                let d = format!("{} of {} MB", mb(b), mb(t));
                report(Progress::Stage {
                    stage: Stage::Uv,
                    percent: Some(pct),
                    detail: &d,
                });
            },
            cancel,
        )?;
        if !uv.is_file() {
            bail!("{} is missing after installing uv", uv.display());
        }
    }
    check(cancel)?;

    // 2 + 3. Python and the packages
    if env_ready(p, job.platform, &lock) {
        enter!(Stage::Python, Some(100), "reusing the Python environment");
        enter!(Stage::Packages, Some(100), "packages already installed");
    } else {
        let _ = std::fs::remove_file(p.env.join(ENV_MARKER));
        // The YOLO26-only toolchain of earlier versions (same tools, older names) is replaced.
        for d in p.legacy_dirs() {
            if let Err(e) = manager::remove_any(&d) {
                tracing::warn!("removing the old export toolchain: {e:#}");
            }
        }
        enter!(Stage::Python, Some(0), "uv python install 3.11");
        let mut py_args: Vec<&std::ffi::OsStr> = vec![
            "python".as_ref(),
            "install".as_ref(),
            PYTHON_VERSION.as_ref(),
            "--install-dir".as_ref(),
            p.python_installs.as_os_str(),
            "--no-bin".as_ref(),
        ];
        if windows {
            py_args.push("--no-registry".as_ref());
        }
        let cmd = uv_cmd(job, Stage::Python, job.timeouts.python, &py_args);
        runner.run(&cmd, &mut |l| python_line(report, l), cancel)?;
        check(cancel)?;
        enter!(Stage::Python, Some(60), "uv venv");
        let cmd = uv_cmd(
            job,
            Stage::Python,
            job.timeouts.python,
            &[
                "venv".as_ref(),
                "--python".as_ref(),
                PYTHON_VERSION.as_ref(),
                "--managed-python".as_ref(),
                "--clear".as_ref(),
                p.env.as_os_str(),
            ],
        );
        runner.run(&cmd, &mut |l| python_line(report, l), cancel)?;
        check(cancel)?;

        enter!(Stage::Packages, Some(0), lock.name);
        let lockfile = p.tools.join(LOCKFILE_NAME);
        std::fs::write(&lockfile, lock.text)
            .with_context(|| format!("writing {}", lockfile.display()))?;
        let python = p.env_python(windows);
        let mut args: Vec<&std::ffi::OsStr> = vec![
            "pip".as_ref(),
            "install".as_ref(),
            "--python".as_ref(),
            python.as_os_str(),
            "--require-hashes".as_ref(),
            "--no-deps".as_ref(),
            "--only-binary".as_ref(),
            ":all:".as_ref(),
            "-r".as_ref(),
            lockfile.as_os_str(),
        ];
        if lock.torch_cpu_index {
            args.push(std::ffi::OsStr::new("--torch-backend"));
            args.push(std::ffi::OsStr::new("cpu"));
        }
        let cmd = uv_cmd(job, Stage::Packages, job.timeouts.packages, &args);
        let mut pp = PackagesProgress::default();
        runner.run(
            &cmd,
            &mut |l| {
                report(Progress::Line {
                    stage: Stage::Packages,
                    line: l,
                });
                if let Some((pct, detail)) = pp.line(l) {
                    report(Progress::Stage {
                        stage: Stage::Packages,
                        percent: Some(pct),
                        detail: &detail,
                    });
                }
            },
            cancel,
        )?;
        check(cancel)?;
        if !python.is_file() {
            bail!(
                "{} is missing after creating the environment",
                python.display()
            );
        }
        // The wheels are installed (copied, hard-linked or cloned); the cache is not needed.
        let _ = std::fs::remove_dir_all(&p.uv_cache);
        let marker = serde_json::to_vec_pretty(&marker_for(&lock)).map_err(|e| anyhow!(e))?;
        std::fs::write(p.env.join(ENV_MARKER), marker)
            .with_context(|| format!("writing the marker in {}", p.env.display()))?;
        enter!(Stage::Packages, Some(100), "packages installed");
    }
    check(cancel)?;

    // 4. Weights (and the code a recipe needs)
    let weights_dir = p.weights_of(name);
    let mut downloads: Vec<(&'static Resource, PathBuf)> = vec![(res, weights_dir.clone())];
    for extra in catalog::export_extras(res) {
        let dir = if extra.id == catalog::DEIMV2_SRC_ID {
            p.deimv2_src.clone()
        } else {
            p.tools
                .join(extra.dest.strip_prefix("tools/").unwrap_or(extra.dest))
        };
        downloads.push((extra, dir));
    }
    let missing: Vec<&(&'static Resource, PathBuf)> = downloads
        .iter()
        .filter(|(r, dir)| !manager::is_installed(r, dir))
        .collect();
    if missing.is_empty() {
        enter!(Stage::Weights, Some(100), "weights already downloaded");
    } else {
        let total: u64 = missing.iter().map(|(r, _)| r.size()).sum();
        let mut done = 0u64;
        for (r, dir) in missing {
            let label = r.parts.first().map_or(r.title, |p| p.file_name);
            enter!(Stage::Weights, Some(manager::percent(done, total)), label);
            fetcher.fetch(
                r,
                dir,
                &mut |b, _| {
                    let all = done + b;
                    let pct = (all * 100).checked_div(total).unwrap_or(0).min(100) as u8;
                    let d = format!("{label} {} of {} MB", mb(all), mb(total));
                    report(Progress::Stage {
                        stage: Stage::Weights,
                        percent: Some(pct),
                        detail: &d,
                    });
                },
                cancel,
            )?;
            if !manager::is_installed(r, dir) {
                bail!(
                    "{} is not complete in {} after the download",
                    r.title,
                    dir.display()
                );
            }
            done += r.size();
            check(cancel)?;
        }
    }
    check(cancel)?;

    // 5. Export
    enter!(
        Stage::Export,
        Some(0),
        &format!("starting {}", recipe.exporter())
    );
    std::fs::create_dir_all(&p.work).with_context(|| format!("creating {}", p.work.display()))?;
    std::fs::create_dir_all(&p.yolo_config)
        .with_context(|| format!("creating {}", p.yolo_config.display()))?;
    let script = p.work.join(SCRIPT_NAME);
    std::fs::write(&script, EXPORT_SCRIPT)
        .with_context(|| format!("writing {}", script.display()))?;
    std::fs::create_dir_all(&p.models)
        .with_context(|| format!("creating {}", p.models.display()))?;
    let staging = p.staging(name);
    manager::remove_any(&staging)?;
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let work = p.work.join(name);
    manager::remove_any(&work)?;
    // Ultralytics takes the `.pt` file; the Hugging Face recipes the directory with
    // `model.safetensors` + `config.json`.
    let weights_arg = match recipe {
        catalog::ExportRecipe::Ultralytics => weights_dir.join(
            res.parts
                .first()
                .ok_or_else(|| anyhow!("{} has no weights", res.id))?
                .file_name,
        ),
        _ => weights_dir.clone(),
    };
    let mut args: Vec<OsString> = vec![
        "-u".into(),
        "-I".into(),
        script.clone().into_os_string(),
        "--recipe".into(),
        recipe.as_str().into(),
        "--name".into(),
        name.into(),
        "--family".into(),
        family.to_string().into(),
        "--weights".into(),
        weights_arg.into_os_string(),
        "--imgsz".into(),
        catalog::export_input_size(res).to_string().into(),
        "--no-openvino".into(),
        "--out-dir".into(),
        staging.clone().into_os_string(),
        "--work".into(),
        work.clone().into_os_string(),
    ];
    if recipe == catalog::ExportRecipe::DeimV2 {
        args.push("--source".into());
        args.push(p.deimv2_src.clone().into_os_string());
    }
    let cfg = |sub: &str| p.yolo_config.join(sub).into_os_string();
    let cmd = Cmd {
        stage: Stage::Export,
        program: p.env_python(windows),
        args,
        env: vec![
            ("YOLO_OFFLINE".into(), "1".into()),
            ("YOLO_AUTOINSTALL".into(), "false".into()),
            (
                "YOLO_CONFIG_DIR".into(),
                p.yolo_config.clone().into_os_string(),
            ),
            ("HF_HUB_OFFLINE".into(), "1".into()),
            ("TRANSFORMERS_OFFLINE".into(), "1".into()),
            ("HF_HUB_DISABLE_TELEMETRY".into(), "1".into()),
            ("HF_HOME".into(), cfg("huggingface")),
            ("TORCH_HOME".into(), cfg("torch")),
            ("MPLBACKEND".into(), "Agg".into()),
        ],
        cwd: p.work.clone(),
        timeout: job.timeouts.export,
    };
    let mut lines = 0usize;
    let export = runner.run(
        &cmd,
        &mut |l| {
            report(Progress::Line {
                stage: Stage::Export,
                line: l,
            });
            lines += 1;
            // No real progress from the exporters: creep towards 90% with the output.
            let pct = u8::try_from(lines.min(45) * 2).unwrap_or(90);
            report(Progress::Stage {
                stage: Stage::Export,
                percent: Some(pct),
                detail: l,
            });
        },
        cancel,
    );
    // The exporters' copies and intermediate files are not needed any more.
    let _ = manager::remove_any(&work);
    if let Err(e) = export {
        let _ = manager::remove_any(&staging);
        return Err(e);
    }
    check(cancel)?;

    // 6. Install
    enter!(Stage::Install, Some(0), "checking the output");
    let out = install_outputs(res, &staging, &p.models, &onnx_name, &yaml_name);
    let _ = manager::remove_any(&staging);
    let out = out?;
    enter!(Stage::Install, Some(100), "installed");
    Ok(out)
}

/// Check the staged `.onnx` / `.yaml`, move them into `models` and write the manifest.
pub(super) fn install_outputs(
    res: &Resource,
    staging: &Path,
    models: &Path,
    onnx_name: &str,
    yaml_name: &str,
) -> Result<ExportOutput> {
    let onnx = staging.join(onnx_name);
    let yaml = staging.join(yaml_name);
    let onnx_len = std::fs::metadata(&onnx)
        .with_context(|| format!("the exporter wrote no {onnx_name}"))?
        .len();
    if onnx_len < 1024 {
        bail!("{onnx_name} is only {onnx_len} bytes");
    }
    let text = std::fs::read_to_string(&yaml)
        .with_context(|| format!("the exporter wrote no {yaml_name}"))?;
    let classes = class_count(&text)?;
    if classes != 80 {
        bail!("{yaml_name} lists {classes} classes; the exportable COCO models have 80");
    }
    let shas = vec![manager::sha256_file(&onnx)?, manager::sha256_file(&yaml)?];
    let m = Manifest {
        id: res.id.to_string(),
        version: res.version.to_string(),
        sha256: shas,
        files: vec![onnx_name.to_string(), yaml_name.to_string()],
    };
    std::fs::create_dir_all(models).with_context(|| format!("creating {}", models.display()))?;
    let path = manager::manifest_path(res, models);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&m).map_err(|e| anyhow!(e))?)
        .with_context(|| format!("writing {}", tmp.display()))?;
    // `.yaml` first, `.onnx` second (the `.onnx` is what makes the export "present"), the
    // manifest last; on any failure everything moved so far is undone (RS-09).
    let mut tx = InstallTx::default();
    let result = (|| -> Result<()> {
        tx.put(&yaml, &models.join(yaml_name))?;
        tx.put(&onnx, &models.join(onnx_name))?;
        #[cfg(test)]
        if FAIL_MANIFEST_RENAME.with(std::cell::Cell::get) {
            bail!("test: manifest rename fails");
        }
        std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))
    })();
    if let Err(e) = result {
        tx.rollback();
        let _ = std::fs::remove_file(&tmp);
        return Err(e.context(format!(
            "installing the exported {onnx_name} into {} failed and was rolled back; check free \
             disk space and that no other program (antivirus, an editor) holds the files, then \
             export again",
            models.display()
        )));
    }
    tx.commit();
    Ok(ExportOutput {
        onnx: models.join(onnx_name),
        yaml: models.join(yaml_name),
        classes,
    })
}

#[cfg(test)]
thread_local! {
    /// Make [`install_outputs`] fail at its last step (RS-09 test).
    pub(super) static FAIL_MANIFEST_RENAME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Files moved into the models directory by [`install_outputs`], undone on failure: each
/// replaced file is first renamed aside (`.<name>.old-<pid>`) and restored by [`Self::rollback`].
#[derive(Default)]
struct InstallTx {
    /// (destination, the previous file renamed aside, if there was one)
    moved: Vec<(PathBuf, Option<PathBuf>)>,
}

impl InstallTx {
    /// Move `src` to `dst`, keeping a previous `dst` aside.
    fn put(&mut self, src: &Path, dst: &Path) -> Result<()> {
        let name = dst
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let parent = dst.parent().unwrap_or(Path::new("."));
        let aside = if std::fs::symlink_metadata(dst).is_ok() {
            let a = parent.join(format!(".{name}.old-{}", std::process::id()));
            let _ = std::fs::remove_file(&a);
            std::fs::rename(dst, &a).with_context(|| format!("replacing {}", dst.display()))?;
            Some(a)
        } else {
            None
        };
        if let Err(e) = std::fs::rename(src, dst) {
            if let Some(a) = &aside {
                let _ = std::fs::rename(a, dst);
            }
            return Err(e).with_context(|| format!("moving to {}", dst.display()));
        }
        self.moved.push((dst.to_path_buf(), aside));
        Ok(())
    }

    /// Remove what was moved in and restore the previous files, newest first.
    fn rollback(&mut self) {
        for (dst, aside) in self.moved.drain(..).rev() {
            let _ = std::fs::remove_file(&dst);
            if let Some(a) = aside
                && let Err(e) = std::fs::rename(&a, &dst)
            {
                tracing::warn!(
                    "could not restore {} from {}: {e}",
                    dst.display(),
                    a.display()
                );
            }
        }
    }

    /// Delete the previous files.
    fn commit(&mut self) {
        for (_, aside) in self.moved.drain(..) {
            if let Some(a) = aside {
                let _ = std::fs::remove_file(a);
            }
        }
    }
}

/// Number of names in a `NAMES:` class file.
pub(super) fn class_count(text: &str) -> Result<usize> {
    #[derive(Deserialize)]
    struct Names {
        #[serde(rename = "NAMES")]
        names: Vec<String>,
    }
    let n: Names = serde_yaml_ng::from_str(text).context("parsing the class file")?;
    Ok(n.names.len())
}
