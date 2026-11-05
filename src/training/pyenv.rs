//! What the import, auto-label and overlap jobs share: the training environment
//! (`tools/train-env`, set up by the export's code under its lock, [`setup`]), the verified downloads
//! of their checkpoints and footage ([`fetch_missing`], where each one goes: [`weights_dir`],
//! [`dataset_dir`]), running a helper script in it with its `@<tag> {json}` events ([`run_script`]),
//! `scripts/autolabel.py`'s own check ([`self_check`]) and telling a cancel from a failure
//! ([`is_cancel`]).

use super::capability::Accelerator;
use super::task::TaskReport;
use crate::resources::catalog::{self, Platform, Resource};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::resources::export::{Cancelled, CommandRunner, Fetcher, Progress, Stage};
use crate::resources::{hf_token, manager};
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

/// Whether `e` is a cancel ([`Cancelled`] or the benchmark's).
pub fn is_cancel(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.is::<Cancelled>()) || crate::benchmark::is_cancelled(e)
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

/// The resources of `list` a fetch would download: not installed, and not every file on disk at
/// its pinned size either, which the download manager adopts by hash with no network.
fn to_download<'a>(
    list: &'a [(&'static Resource, PathBuf)],
) -> impl Iterator<Item = &'static Resource> + 'a {
    list.iter()
        .filter(|(r, d)| !manager::is_installed(r, d) && !hf_token::parts_present(r, d))
        .map(|(r, _)| *r)
}

/// Total size of the resources of `list` that have to be downloaded ([`to_download`]).
pub fn missing_bytes(list: &[(&'static Resource, PathBuf)]) -> u64 {
    to_download(list).map(Resource::size).sum()
}

/// Refuse a download over the large-download gate unless allowed (`--allow-large`, the page's
/// confirmation, or `allow_large_downloads`).
pub fn check_large(list: &[(&'static Resource, PathBuf)], allowed: bool) -> Result<()> {
    let bytes = missing_bytes(list);
    if bytes > 0 && catalog::is_large(bytes) && !allowed {
        let names: Vec<&str> = to_download(list).map(|r| r.title).collect();
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
                .and_then(|c| gpu_failure(&c.lines, tag))
            {
                return e.context(why);
            }
            env::explain_train_failure(e, paths, platform, task_name(tag))
        })
}

/// Run `scripts/autolabel.py --self-check` (the imports and its own tests), reporting nothing but
/// the log; an error when it fails or did not run its tests.
pub fn self_check(
    runner: &dyn CommandRunner,
    paths: &TrainPaths,
    platform: Platform,
    script: &Path,
    report: &mut dyn FnMut(TaskReport<'_>),
    cancel: &AtomicBool,
) -> Result<()> {
    let mut passed = None;
    run_script(
        runner,
        paths,
        platform,
        &paths.scripts,
        Duration::from_secs(15 * 60),
        vec![script.as_os_str().to_owned(), "--self-check".into()],
        "autolabel",
        &mut |ev, _| {
            if ev.get("event").and_then(serde_json::Value::as_str) == Some("self_test") {
                passed = ev.get("passed").and_then(serde_json::Value::as_u64);
            }
        },
        report,
        cancel,
    )
    .context("the auto-labeler's self-check failed")?;
    if passed.is_none() {
        bail!("the auto-labeler's self-test did not run; the script is damaged");
    }
    Ok(())
}

/// What the failure messages call the script of event tag `tag`.
fn task_name(tag: &str) -> &'static str {
    match tag {
        "autolabel" => "auto-label",
        "overlaps" => "overlap check",
        "import" => "import",
        _ => "training",
    }
}

/// The likely cause of a failed helper script when it ran out of GPU memory (not when the host
/// ran out too: on Windows a CUDA allocation also fails when the system commit is exhausted;
/// [`super::job::host_out_of_memory_line`], left to the generic memory message).
fn gpu_failure(lines: &[String], tag: &str) -> Option<String> {
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
    let fix = if tag == "overlaps" {
        "close other programs that use the GPU (the check needs about 500 MB of it) and run it \
         again; it keeps the finished pairs"
    } else {
        "use a smaller labeler (--gdino tiny, --vlm qwen3-vl-2b or none) or close other programs \
         that use the GPU"
    };
    Some(format!("the GPU ran out of memory ({}): {fix}", l.trim()))
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
            gpu_failure(
                &[
                    "x".into(),
                    "RuntimeError: MPS backend out of memory (MPS)".into()
                ],
                "autolabel"
            )
            .unwrap()
            .contains("smaller labeler")
        );
        assert!(
            gpu_failure(&["CUDA out of memory".into()], "overlaps")
                .unwrap()
                .contains("keeps the finished pairs")
        );
        // The host out of memory too (Windows commit): not the GPU's fault.
        assert_eq!(
            gpu_failure(
                &[
                    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU \
                     0 has a total capacity of 8.00 GiB of which 6.17 GiB is free."
                        .into(),
                    "RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: \
                     not enough memory: you tried to allocate 262144 bytes."
                        .into(),
                ],
                "autolabel"
            ),
            None
        );
        assert_eq!(task_name("autolabel"), "auto-label");
        assert_eq!(task_name("overlaps"), "overlap check");
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

    #[test]
    fn files_on_disk_at_their_pinned_size_are_not_a_download() {
        let part = |name: &'static str| catalog::Part {
            url: "https://example.invalid/x",
            sha256: "00",
            size: 4,
            file_name: name,
            archive: None,
            layout: catalog::Layout::File,
        };
        let parts: &'static [catalog::Part] =
            Box::leak(vec![part("a"), part("b")].into_boxed_slice());
        let res: &'static Resource = Box::leak(Box::new(Resource {
            id: "autolabel:adopt-test",
            kind: catalog::ResourceKind::ExportModel,
            version: "1",
            platform: None,
            parts,
            dest: catalog::TRAIN_WEIGHTS_DEST,
            provides: catalog::Provides::Tool { name: "adopt-test" },
            title: "fixture",
            description: "fixture",
            license: "MIT",
        }));
        let dir = std::env::temp_dir().join(format!("bop-pyenv-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let list = vec![(res, dir.clone())];
        assert_eq!(missing_bytes(&list), 8, "nothing on disk");
        std::fs::write(dir.join("a"), b"1234").unwrap();
        std::fs::write(dir.join("b"), b"123").unwrap();
        assert_eq!(missing_bytes(&list), 8, "a short file is not there");
        std::fs::write(dir.join("b"), b"1234").unwrap();
        assert_eq!(missing_bytes(&list), 0, "adopted by hash, no network");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
