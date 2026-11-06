//! Retry export: a failed run keeps `<run>/work`, and the best checkpoint in it can become the
//! model without training again ([`plan_retry`], `train_model.py --export-only`).
//!
//! Where each trainer leaves its best weights ([`best_checkpoint`]; the script's module docs):
//! - Ultralytics: `run/weights/best.pt`, else `last.pt`, written after every epoch;
//! - RF-DETR: `rfdetr-out/checkpoint_best_total.pth` once training ended, else the running best
//!   `checkpoint_best_ema.pth` / `checkpoint_best_regular.pth` beside `training_config.json`;
//! - transformers (RT-DETRv2, D-FINE): `hf-best/`, which the script saves at every new best epoch.
//!   Runs of older versions kept nothing on disk: nothing to retry.
//!
//! A run that failed while still training gets the best epoch so far, and the row says so. Resuming
//! the training itself is not offered: Convergence, the selection and the learning-rate bends all
//! live in the trainer's process.
//!
//! The rest is what a run leaves on disk: [`disk_use`] for the row's sizes, [`delete_run_files`]
//! for Remove.

use super::job::{ExportRetry, TrainJob};
use super::service::{RunRecord, RunState};
use super::{Capability, options_for, training_dir};
use crate::config::Config;
use crate::resources::catalog::{Platform, Trainer};
use crate::resources::export::train::TrainPaths;
use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The checkpoint folder of a run.
pub fn work_dir(run_dir: &Path) -> PathBuf {
    run_dir.join("work")
}

/// Disk an export needs beyond the checkpoint it reads: the ONNX twice (work and staging), a
/// comfortable bound for the largest family (RF-DETR L, about 500 MB).
pub const EXPORT_BYTES: u64 = 1 << 30;

/// The best checkpoint `trainer` left in `work` (see the module docs), if any.
pub fn best_checkpoint(trainer: Trainer, work: &Path) -> Option<PathBuf> {
    match trainer {
        Trainer::Ultralytics => {
            let w = work.join("run").join("weights");
            [w.join("best.pt"), w.join("last.pt")]
                .into_iter()
                .find(|p| p.is_file())
        }
        Trainer::RfDetr => {
            let out = work.join("rfdetr-out");
            if !out.join("training_config.json").is_file() {
                return None;
            }
            [
                "checkpoint_best_total.pth",
                "checkpoint_best_ema.pth",
                "checkpoint_best_regular.pth",
            ]
            .into_iter()
            .map(|n| out.join(n))
            .find(|p| p.is_file())
        }
        Trainer::Transformers => {
            let dir = work.join("hf-best");
            (dir.join("model.safetensors").is_file() && dir.join("config.json").is_file())
                .then_some(dir)
        }
    }
}

/// Why `r` cannot export again, or the checkpoint it can export. The reasons are for the CLI and
/// the route; the page simply shows no button.
pub fn retry_checkpoint(r: &RunRecord) -> Result<PathBuf, String> {
    if r.state != RunState::Failed {
        return Err(format!(
            "run {} is {}; only a failed run can export again",
            r.id,
            r.describe()
        ));
    }
    if let Some(o) = &r.output {
        return Err(format!(
            "run {} already installed {}: there is nothing to export",
            r.id,
            o.onnx.display()
        ));
    }
    if !r.run_dir.join("data").join("dataset.json").is_file() {
        return Err(format!(
            "run {} has no prepared dataset in {}: it failed before training; queue it again",
            r.id,
            r.run_dir.display()
        ));
    }
    let trainer = r.request.recipe.family.trainer();
    best_checkpoint(trainer, &work_dir(&r.run_dir)).ok_or_else(|| {
        format!(
            "run {} left no checkpoint in {}: it failed before its first validated epoch, its \
             files were removed, or a version that kept none for this family trained it; queue \
             the training again",
            r.id,
            work_dir(&r.run_dir).display()
        )
    })
}

/// The job that exports the failed run `r` again: its own name, folder, request and classes.
/// `cap` supplies the environment's PyTorch build (the export itself runs on the CPU).
pub fn plan_retry(config: &Config, r: &RunRecord, cap: &Capability) -> Result<TrainJob> {
    let checkpoint = retry_checkpoint(r).map_err(|e| anyhow!(e))?;
    let cap = &cap.for_gpu(r.request.gpu.as_deref())?;
    let Some(accel) = cap.accelerator.clone() else {
        bail!("{}", cap.summary());
    };
    if config
        .models
        .iter()
        .any(|m| m.effective_name().eq_ignore_ascii_case(&r.name))
    {
        bail!(
            "the config already has a model named {}: remove that model first, or train again \
             under another name",
            r.name
        );
    }
    let t = &config.training;
    let train_seconds = r
        .progress
        .as_ref()
        .and_then(|p| p.history.last().map(|h| h.elapsed_s));
    Ok(TrainJob {
        id: r.id.clone(),
        request: r.request.clone(),
        name: r.name.clone(),
        paths: TrainPaths::for_config(config),
        run_dir: r.run_dir.clone(),
        platform: Platform::current(),
        accelerator: accel,
        options: options_for(config, &r.request),
        root: config.data_root(),
        timeout: Duration::from_secs(u64::from(t.timeout_hours) * 3600),
        workers: t
            .workers
            .map_or(super::job::Workers::Auto, super::job::Workers::Fixed),
        keep_work: false,
        export_only: Some(ExportRetry {
            checkpoint,
            train_seconds,
            device: r.device.clone(),
        }),
    })
}

/// What `r` trained until it stopped, for the row beside Retry export. Empty when training ended
/// and only the export failed.
pub fn partial_note(r: &RunRecord) -> String {
    let Some(p) = r.progress.as_ref().filter(|p| p.phase.is_none()) else {
        return if r.progress.is_some() {
            String::new()
        } else {
            "it stopped before reporting an epoch; the export uses the checkpoint it left".into()
        };
    };
    let done = p.history.last().map_or(0, |h| h.epoch);
    match p.best.as_ref() {
        Some(b) => format!(
            "training stopped after epoch {done}: the export uses the best epoch so far ({})",
            b.epoch
        ),
        None => {
            format!("training stopped after epoch {done}: the export uses the best epoch so far")
        }
    }
}

/// Bytes deleting `dir` would free. Each file once, and on Unix only files with no link outside
/// `dir`: a dataset's images are hard links to the export they came from, and deleting them
/// frees nothing while the export stays. Windows counts every file.
pub fn disk_use(dir: &Path) -> u64 {
    #[cfg(unix)]
    let mut seen: std::collections::HashMap<(u64, u64), (u64, u64, u64)> =
        std::collections::HashMap::new();
    let mut total = 0u64;
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                todo.push(e.path());
            } else if md.is_file() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    let s = seen
                        .entry((md.dev(), md.ino()))
                        .or_insert((0, md.nlink(), md.len()));
                    s.0 += 1;
                }
                #[cfg(not(unix))]
                {
                    total = total.saturating_add(md.len());
                }
            }
        }
    }
    #[cfg(unix)]
    for (links_here, links, len) in seen.into_values() {
        if links_here >= links {
            total = total.saturating_add(len);
        }
    }
    total
}

/// Delete the folder of run `id`, which must be `<root>/<id>`: a record edited by hand or moved
/// with another training folder never deletes anything else.
pub fn delete_run_files(run_dir: &Path, id: &str, root: &Path) -> Result<()> {
    let inside = run_dir.parent() == Some(root);
    let named = !id.is_empty() && run_dir.file_name().is_some_and(|n| n == id);
    if !inside || !named {
        bail!(
            "{} is not this run's folder under the training folder {}; nothing was deleted, \
             delete it by hand if you no longer need it",
            run_dir.display(),
            root.display()
        );
    }
    crate::resources::manager::remove_any(run_dir)
}

/// [`delete_run_files`] under the training folder of `config`.
pub fn delete_files_of(config: &Config, r: &RunRecord) -> Result<()> {
    delete_run_files(&r.run_dir, &r.id, &training_dir(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::catalog::TrainFamily;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-retry-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(p: &Path, n: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![1u8; n]).unwrap();
    }

    #[test]
    fn each_trainer_has_its_best_checkpoint() {
        let w = tmp("ckpt");
        assert_eq!(best_checkpoint(Trainer::Ultralytics, &w), None);
        put(&w.join("run/weights/last.pt"), 1);
        assert_eq!(
            best_checkpoint(Trainer::Ultralytics, &w),
            Some(w.join("run/weights/last.pt"))
        );
        put(&w.join("run/weights/best.pt"), 1);
        assert_eq!(
            best_checkpoint(Trainer::Ultralytics, &w),
            Some(w.join("run/weights/best.pt"))
        );
        // RF-DETR: the finished best, else the running one; none without its config.
        put(&w.join("rfdetr-out/checkpoint_best_ema.pth"), 1);
        assert_eq!(best_checkpoint(Trainer::RfDetr, &w), None);
        put(&w.join("rfdetr-out/training_config.json"), 1);
        assert_eq!(
            best_checkpoint(Trainer::RfDetr, &w),
            Some(w.join("rfdetr-out/checkpoint_best_ema.pth"))
        );
        put(&w.join("rfdetr-out/checkpoint_best_total.pth"), 1);
        assert_eq!(
            best_checkpoint(Trainer::RfDetr, &w),
            Some(w.join("rfdetr-out/checkpoint_best_total.pth"))
        );
        // transformers: the whole folder, or nothing.
        put(&w.join("hf-best/config.json"), 1);
        assert_eq!(best_checkpoint(Trainer::Transformers, &w), None);
        put(&w.join("hf-best/model.safetensors"), 1);
        assert_eq!(
            best_checkpoint(Trainer::Transformers, &w),
            Some(w.join("hf-best"))
        );
        let _ = std::fs::remove_dir_all(&w);
    }

    fn record(run_dir: &Path, state: RunState) -> RunRecord {
        let recipe = crate::training::TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap();
        serde_json::from_value(serde_json::json!({
            "id": "r1", "name": "my-model", "created": "",
            "request": {"recipe": recipe, "datasets": []},
            "device": "mps", "state": state, "stage": "train", "step": 5, "steps": 8, "percent": 40,
            "detail": "", "run_dir": run_dir
        }))
        .unwrap()
    }

    #[test]
    fn only_a_failed_run_with_a_dataset_and_a_checkpoint_retries() {
        let d = tmp("eligible");
        let r = record(&d, RunState::Failed);
        assert!(
            retry_checkpoint(&r)
                .unwrap_err()
                .contains("no prepared dataset")
        );
        put(&d.join("data/dataset.json"), 2);
        assert!(
            retry_checkpoint(&r)
                .unwrap_err()
                .contains("left no checkpoint")
        );
        put(&d.join("work/run/weights/best.pt"), 2);
        assert_eq!(
            retry_checkpoint(&r).unwrap(),
            d.join("work/run/weights/best.pt")
        );
        assert!(retry_checkpoint(&record(&d, RunState::Cancelled)).is_err());
        assert!(retry_checkpoint(&record(&d, RunState::Running)).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_partial_note_says_when_training_did_not_end() {
        let d = tmp("note");
        let mut r = record(&d, RunState::Failed);
        assert!(partial_note(&r).contains("before reporting"));
        r.progress = serde_json::from_value(serde_json::json!({
            "phase": "export"
        }))
        .ok();
        assert_eq!(partial_note(&r), "");
        r.progress = serde_json::from_value(serde_json::json!({
            "history": [{"epoch": 7, "elapsed_s": 1.0, "metrics": {}}]
        }))
        .ok();
        assert!(
            partial_note(&r).contains("after epoch 7"),
            "{}",
            partial_note(&r)
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn disk_use_counts_what_deleting_frees() {
        let d = tmp("disk");
        put(&d.join("run/a.bin"), 1000);
        put(&d.join("run/sub/b.bin"), 500);
        // Two links inside: one file. A link outside: nothing freed.
        std::fs::hard_link(d.join("run/a.bin"), d.join("run/sub/a-link.bin")).unwrap();
        put(&d.join("export/c.jpg"), 300);
        std::fs::hard_link(d.join("export/c.jpg"), d.join("run/sub/c.jpg")).unwrap();
        let expected = if cfg!(unix) { 1500 } else { 2800 };
        assert_eq!(disk_use(&d.join("run")), expected);
        assert_eq!(disk_use(&d.join("missing")), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn delete_run_files_only_touches_the_runs_own_folder() {
        let root = tmp("delete");
        let run = root.join("r1");
        put(&run.join("work/x"), 1);
        put(&root.join("runs.json"), 1);
        // Another run's name, the root itself, a folder elsewhere: refused, nothing gone.
        assert!(delete_run_files(&run, "r2", &root).is_err());
        assert!(delete_run_files(&root, "training", &root).is_err());
        assert!(delete_run_files(&run, "", &root).is_err());
        assert!(delete_run_files(&run, "r1", &root.join("other")).is_err());
        assert!(run.join("work/x").is_file());
        delete_run_files(&run, "r1", &root).unwrap();
        assert!(!run.exists());
        assert!(root.join("runs.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }
}
