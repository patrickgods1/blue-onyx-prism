//! What the import and auto-label jobs share: the training environment (`tools/train-env`, set up
//! by the export's code under its lock, [`setup`]), the verified downloads of their checkpoints
//! and footage ([`fetch_missing`], where each one goes: [`weights_dir`], [`dataset_dir`]) and
//! running a helper script in it with its `@<tag> {json}` events ([`run_script`]).

use super::capability::Accelerator;
use super::task::TaskReport;
use crate::resources::catalog::{self, Platform, Resource};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::resources::export::{Cancelled, CommandRunner, Fetcher, Progress, Stage};
use crate::resources::manager;
use anyhow::{Context, Result, anyhow, bail};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Fail with [`Cancelled`] when `cancel` is set.
pub fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(anyhow!(Cancelled))
    } else {
        Ok(())
    }
}

/// Where a labeler checkpoint is installed: `tools/train-weights/<key>` (RF-DETR's training base
/// pin keeps its training directory).
pub fn weights_dir(paths: &TrainPaths, res: &Resource) -> PathBuf {
    if res.id.starts_with(catalog::AUTOLABEL_PREFIX) {
        paths.weights.join(catalog::install_key(res))
    } else {
        paths.weights_dir(res)
    }
}

/// Where pinned footage is installed: `<data root>/train-data/<key>`.
pub fn dataset_dir(root: &Path, res: &Resource) -> PathBuf {
    root.join(catalog::IMPORT_DATA_DEST)
        .join(catalog::install_key(res))
}

/// One stage report of a task: stage `step` of `steps` at `pct` percent.
pub fn stage(
    report: &mut dyn FnMut(TaskReport<'_>),
    name: &str,
    step: usize,
    steps: usize,
    pct: u8,
    detail: &str,
) {
    report(TaskReport::Stage {
        stage: name,
        step,
        steps,
        percent: super::task::overall(step, steps, pct),
        detail,
    });
}

/// Total size of the resources of `list` that are not installed yet.
pub fn missing_bytes(list: &[(&'static Resource, PathBuf)]) -> u64 {
    list.iter()
        .filter(|(r, d)| !manager::is_installed(r, d))
        .map(|(r, _)| r.size())
        .sum()
}

/// Refuse a download over the large-download gate unless allowed (`--allow-large`, the page's
/// confirmation, or `allow_large_downloads`).
pub fn check_large(list: &[(&'static Resource, PathBuf)], allowed: bool) -> Result<()> {
    let bytes = missing_bytes(list);
    if bytes > 0 && catalog::is_large(bytes) && !allowed {
        let names: Vec<&str> = list
            .iter()
            .filter(|(r, d)| !manager::is_installed(r, d))
            .map(|(r, _)| r.title)
            .collect();
        bail!(
            "this needs a large download ({}: {}); rerun with --allow-large (Training page: tick \
             the download confirmation) or enable allow_large_downloads",
            catalog::format_size(bytes),
            names.join(", ")
        );
    }
    Ok(())
}

/// Download (SHA-256 checked, through the download manager) every resource of `list` that is not
/// installed, reporting stage `step` of `steps`.
pub fn fetch_missing(
    fetcher: &dyn Fetcher,
    list: &[(&'static Resource, PathBuf)],
    report: &mut dyn FnMut(TaskReport<'_>),
    (name, step, steps): (&str, usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    let missing: Vec<&(&'static Resource, PathBuf)> = list
        .iter()
        .filter(|(r, d)| !manager::is_installed(r, d))
        .collect();
    if missing.is_empty() {
        stage(report, name, step, steps, 100, "already downloaded");
        return Ok(());
    }
    let total: u64 = missing.iter().map(|(r, _)| r.size()).sum();
    let mut done = 0u64;
    for (r, dir) in missing {
        stage(
            report,
            name,
            step,
            steps,
            manager::percent(done, total),
            r.title,
        );
        fetcher.fetch(
            r,
            dir,
            &mut |b, _| {
                let d = format!(
                    "{} {} of {} MB",
                    r.title,
                    (done + b) / 1_000_000,
                    total / 1_000_000
                );
                stage(
                    report,
                    name,
                    step,
                    steps,
                    manager::percent(done + b, total),
                    &d,
                );
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
    Ok(())
}

/// Set up the training environment for `accel`'s lane (uv, Python, the lock's packages; the
/// export's code under its lock), reporting stage `step` of `steps`.
#[allow(
    clippy::too_many_arguments,
    reason = "the environment, the two seams, the report and its stage, the cancel flag"
)]
pub fn setup(
    paths: &TrainPaths,
    platform: Platform,
    accel: &Accelerator,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(TaskReport<'_>),
    (name, step, steps): (&str, usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    let any = catalog::AUTOLABEL_WEIGHTS
        .first()
        .ok_or_else(|| anyhow!("no labeler pins in this build"))?;
    let job = env::env_job(paths, any, platform);
    env::prepare_train_env(
        &job,
        accel.lane(),
        runner,
        fetcher,
        &mut |p| match p {
            Progress::Stage {
                stage: s,
                percent,
                detail,
            } => {
                // The export bar's share of these stages ends at its self-check (71%); the
                // last tenth of this stage is the script's own self-check.
                let pct = (u32::from(s.overall(percent)) * 100 / 71).min(100) * 9 / 10;
                let d = format!("{}: {detail}", s.label());
                stage(
                    report,
                    name,
                    step,
                    steps,
                    u8::try_from(pct.min(100)).unwrap_or(100),
                    &d,
                );
            }
            Progress::Line { line, .. } => report(TaskReport::Line(line)),
        },
        cancel,
    )
    .map_err(|e| {
        if e.cancelled {
            anyhow!(Cancelled)
        } else {
            anyhow!(
                "{}{}",
                e.stage
                    .map(|s| format!("while {}: ", s.label()))
                    .unwrap_or_default(),
                e.message
            )
        }
    })?;
    Ok(())
}

/// The JSON of an event line `@<tag> {json}` (None for other lines).
pub fn parse_event(line: &str, tag: &str) -> Option<serde_json::Value> {
    let rest = line.trim_start().strip_prefix('@')?.strip_prefix(tag)?;
    let json = rest.strip_prefix(' ')?;
    serde_json::from_str(json.trim()).ok()
}

/// What a script's event callback gets: the event and the task's report (for progress).
pub type OnEvent<'a> = dyn FnMut(&serde_json::Value, &mut dyn FnMut(TaskReport<'_>)) + 'a;

/// Run `script args` in the training environment (offline flags, its caches), passing each
/// event `@<tag> {json}` to `on_event` (with `report`, for progress) and every other line to
/// the log. A failure is explained
/// like the training's (out of memory, missing kernels, ...).
#[allow(
    clippy::too_many_arguments,
    reason = "the environment, the command and its two output channels"
)]
pub fn run_script(
    runner: &dyn CommandRunner,
    paths: &TrainPaths,
    platform: Platform,
    cwd: &Path,
    timeout: Duration,
    args: Vec<OsString>,
    tag: &str,
    on_event: &mut OnEvent<'_>,
    report: &mut dyn FnMut(TaskReport<'_>),
    cancel: &AtomicBool,
) -> Result<()> {
    std::fs::create_dir_all(cwd).with_context(|| format!("creating {}", cwd.display()))?;
    let cmd = env::train_python_cmd(paths, platform, Stage::Export, cwd, timeout, args);
    runner
        .run(
            &cmd,
            &mut |l| match parse_event(l, tag) {
                Some(ev) => on_event(&ev, report),
                None => report(TaskReport::Line(l)),
            },
            cancel,
        )
        .map_err(|e| {
            if e.downcast_ref::<Cancelled>().is_some() {
                return e;
            }
            if let Some(why) = e
                .downcast_ref::<crate::resources::export::ChildFailed>()
                .and_then(|c| gpu_failure(&c.lines))
            {
                return e.context(why);
            }
            env::explain_train_failure(e, paths, platform, task_name(tag))
        })
}

/// What the failure messages call the script of event tag `tag`.
fn task_name(tag: &str) -> &'static str {
    match tag {
        "autolabel" => "auto-label",
        "import" => "import",
        _ => "training",
    }
}

/// The likely cause of a failed helper script when it ran out of GPU memory (not when the host
/// ran out too: on Windows a CUDA allocation also fails when the system commit is exhausted;
/// [`super::job::host_out_of_memory_line`], left to the generic memory message).
fn gpu_failure(lines: &[String]) -> Option<String> {
    if super::job::host_out_of_memory_line(lines).is_some() {
        return None;
    }
    let l = lines.iter().rev().find(|l| {
        [
            "CUDA out of memory",
            "MPS backend out of memory",
            "torch.OutOfMemoryError",
        ]
        .iter()
        .any(|p| l.contains(p))
    })?;
    Some(format!(
        "the GPU ran out of memory ({}): use a smaller labeler (--gdino tiny, --vlm qwen3-vl-2b \
         or none) or close other programs that use the GPU",
        l.trim()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_parse_by_tag() {
        let v = parse_event("@import {\"event\":\"frame\",\"t\":1.5}", "import").unwrap();
        assert_eq!(v["event"], "frame");
        assert!(parse_event("@import {broken", "import").is_none());
        assert!(parse_event("@train {\"event\":\"x\"}", "import").is_none());
        assert!(parse_event("[import] hello", "import").is_none());
        assert!(parse_event("@importx {}", "import").is_none());
        assert!(
            gpu_failure(&[
                "x".into(),
                "RuntimeError: MPS backend out of memory (MPS)".into()
            ])
            .unwrap()
            .contains("smaller labeler")
        );
        // The host out of memory too (Windows commit): not the GPU's fault.
        assert_eq!(
            gpu_failure(&[
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 \
                 has a total capacity of 8.00 GiB of which 6.17 GiB is free."
                    .into(),
                "RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not \
                 enough memory: you tried to allocate 262144 bytes."
                    .into(),
            ]),
            None
        );
        assert_eq!(task_name("autolabel"), "auto-label");
        assert_eq!(task_name("import"), "import");
    }

    #[test]
    fn install_dirs_have_no_colons() {
        let root = Path::new("/d");
        let p = TrainPaths::new(root, root.join("models"));
        let g = catalog::autolabel_weights("mm-gdino-tiny").unwrap();
        assert_eq!(
            weights_dir(&p, g),
            root.join("tools/train-weights/mm-gdino-tiny")
        );
        let rf = catalog::base_weights(catalog::TrainFamily::Rfdetr, "large").unwrap();
        assert_eq!(
            weights_dir(&p, rf),
            root.join("tools/train-weights/rfdetr-large")
        );
        let m = catalog::import_dataset("meva-sample").unwrap();
        assert_eq!(dataset_dir(root, m), root.join("train-data/meva-sample"));
    }
}
