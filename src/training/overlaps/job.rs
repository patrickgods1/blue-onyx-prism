//! The overlap check's stage machine ([`OverlapsJob`], run on the training queue as a [`Task`]):
//! select the pairs -> set up the environment and RF-DETR L's weights -> zoom check -> write the
//! suggestions. See [`super`] for what it may change: suggestions, nothing else.
//!
//! A cancel or a failure of the script after some crops keeps them: the finished crops are written
//! like a finished run's (no cancel checks in the write stage) and a rerun skips the pairs with an
//! answer. A cancel returns `Ok` with a "stopped by Cancel" summary; a failure returns its error, with
//! how many crops were kept.

use super::select::{self, Selection};
use super::write::{self, Written};
use super::zoom::{self, CropResult};
use super::{NEEDED_COMMIT_GB, OverlapsRequest, rfdetr_weights};
use crate::backend::memory::Commit;
use crate::collect::labels::bulk::Wanted;
use crate::resources::catalog::Platform;
use crate::resources::export::train::{self as env, TrainPaths};
use crate::training::capability::Accelerator;
use crate::training::pyenv::{self, check, is_cancel, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Stages of a run.
pub const STAGES: [&str; 4] = [
    "selecting pairs",
    "setting up the environment",
    "zoom check",
    "writing suggestions",
];

/// A planned run ([`super::plan_overlaps`]).
#[derive(Clone)]
pub struct OverlapsJob {
    pub id: String,
    pub request: OverlapsRequest,
    pub dataset_dir: PathBuf,
    /// `collect.classes`, all of them: the zoom must see a truck even when trucks are hidden, or a
    /// car pair would lose its detection.
    pub classes: Vec<String>,
    /// `collect.hidden_classes`: their pairs are not checked.
    pub hidden: Vec<String>,
    pub paths: TrainPaths,
    pub platform: Platform,
    pub accelerator: Accelerator,
    pub timeout: Duration,
    /// `<training dir>/overlaps/<id>`: the crop list and the results (deleted at the end).
    pub work: PathBuf,
    /// The free commit now ([`crate::system_info::commit`]; tests fake it).
    pub free_commit: fn() -> Option<Commit>,
}

impl std::fmt::Debug for OverlapsJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlapsJob")
            .field("id", &self.id)
            .field("request", &self.request)
            .field("dataset_dir", &self.dataset_dir)
            .finish_non_exhaustive()
    }
}

/// The report of a finished run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OverlapsReport {
    #[serde(flatten)]
    pub selection: Selection,
    #[serde(flatten)]
    pub written: Written,
    /// The training device.
    pub device: Option<String>,
    pub crops_done: usize,
    pub seconds: f64,
    pub warnings: Vec<String>,
    /// Stopped by Cancel during the zoom check: only `crops_done` of `crops` were run.
    pub cancelled: bool,
}

/// The refusal when `free` cannot hold the check; None off Windows (no fixed limit).
pub fn commit_problem(free: Option<Commit>) -> Option<String> {
    let needed = NEEDED_COMMIT_GB * 1024.0 * 1024.0 * 1024.0;
    let have = free?.available as f64;
    (have < needed).then(|| {
        format!(
            "the overlap check needs about {NEEDED_COMMIT_GB} GB of free commit for PyTorch and \
             RF-DETR L, and {:.1} GB is free: close programs or enlarge the paging file, then run \
             it again (nothing was changed)",
            have / (1024.0 * 1024.0 * 1024.0)
        )
    })
}

impl OverlapsJob {
    fn step(&self, report: &mut dyn FnMut(TaskReport<'_>), n: usize, pct: u8, detail: &str) {
        let name = STAGES.get(n.saturating_sub(1)).copied().unwrap_or("");
        stage(report, name, n, STAGES.len(), pct, detail);
    }

    /// Stage 1: the work of this run ([`select::select`]).
    pub fn select(&self) -> Result<Selection> {
        select::select(
            &self.dataset_dir,
            &self.request.frames,
            &Wanted {
                classes: &self.classes,
                hidden: &self.hidden,
            },
        )
    }

    /// Stage 2: the environment, the script's own check and the weights; the script and the weights
    /// file.
    fn prepare(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<(PathBuf, PathBuf)> {
        pyenv::setup(
            &self.paths,
            self.platform,
            &self.accelerator,
            ctx.runner,
            ctx.fetcher,
            report,
            (STAGES[1], 2, STAGES.len()),
            cancel,
        )?;
        let (_, script) = env::write_task_scripts(&self.paths)?;
        self.step(report, 2, 90, "checking the auto-labeler");
        pyenv::self_check(
            ctx.runner,
            &self.paths,
            self.platform,
            &script,
            report,
            cancel,
        )?;
        check(cancel)?;
        let weights = [rfdetr_weights(&self.paths)?];
        pyenv::check_large(&weights, self.request.allow_large)?;
        pyenv::fetch_missing(
            ctx.fetcher,
            &weights,
            report,
            (STAGES[1], 2, STAGES.len()),
            cancel,
        )?;
        let [(res, dir)] = weights;
        let file = res
            .parts
            .first()
            .ok_or_else(|| anyhow!("RF-DETR L has no file"))?
            .file_name;
        Ok((script, dir.join(file)))
    }

    /// Stage 3: run the script. `Ok` (whether a cancel stopped it, the failure that stopped it): the
    /// finished crops are in `out` either way.
    fn run_zoom(
        &self,
        script: &std::path::Path,
        weights: &std::path::Path,
        selection: &Selection,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<(bool, Option<anyhow::Error>, PathBuf)> {
        if let Some(why) = commit_problem((self.free_commit)()) {
            bail!(why);
        }
        std::fs::create_dir_all(&self.work)
            .with_context(|| format!("creating {}", self.work.display()))?;
        let list = self.work.join("crops.jsonl");
        std::fs::write(&list, zoom::crop_list(&selection.frames)?)
            .with_context(|| format!("writing {}", list.display()))?;
        let out = self.work.join("zoom.jsonl");
        let mut args = vec![script.as_os_str().to_owned()];
        args.extend(zoom::args(
            &list,
            &out,
            weights,
            &self.classes,
            &self.accelerator.device_arg(),
        )?);
        let total = selection.crops;
        let mut done_ev = false;
        let ran = pyenv::run_script(
            ctx.runner,
            &self.paths,
            self.platform,
            &self.work,
            self.timeout,
            args,
            zoom::TAG,
            &mut |ev, report| match ev.get("event").and_then(serde_json::Value::as_str) {
                Some("crop") => {
                    let i = ev
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    let pct = u8::try_from(i * 100 / u64::try_from(total.max(1)).unwrap_or(1))
                        .unwrap_or(100);
                    stage(
                        report,
                        STAGES[2],
                        3,
                        STAGES.len(),
                        pct.min(99),
                        &format!("{i} of {total} crops"),
                    );
                }
                Some("model") => stage(report, STAGES[2], 3, STAGES.len(), 0, "RF-DETR L loaded"),
                Some("done") => done_ev = true,
                _ => {}
            },
            report,
            cancel,
        );
        Ok(match ran {
            Err(e) if is_cancel(&e) => (true, None, out),
            Err(e) => (false, Some(e.context("the zoom check failed")), out),
            Ok(()) if cancel.load(Ordering::Relaxed) => (true, None, out),
            Ok(()) if !done_ev => (
                false,
                Some(anyhow!(
                    "the zoom check finished without its summary; the script is damaged"
                )),
                out,
            ),
            Ok(()) => (false, None, out),
        })
    }

    fn stages(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let started = Instant::now();
        let mut rep = OverlapsReport {
            device: Some(self.accelerator.device_arg()),
            ..OverlapsReport::default()
        };
        // 1. Pairs.
        self.step(report, 1, 0, &self.request.frames.describe());
        rep.selection = self.select()?;
        let s = &rep.selection;
        self.step(
            report,
            1,
            100,
            &format!("{} pairs on {} frames", s.pairs, s.frame_count),
        );
        if s.pairs == 0 {
            let summary = format!(
                "nothing to check: no open overlap without an answer on these frames ({} pairs \
                 answered before, {} frames skipped or checked by a reviewer); add again=1 to look \
                 at answered pairs afresh",
                s.answered, s.passed_over
            );
            return Ok(TaskOutcome {
                summary,
                result: serde_json::to_value(&rep)?,
            });
        }
        check(cancel)?;
        let _lock =
            crate::resources::export::train::TrainLock::try_acquire(&self.paths.train_lock_file())?;
        // 2 and 3. A run of classes RF-DETR cannot see needs no model.
        let mut stopped = false;
        let mut failed = None;
        let mut crops: BTreeMap<usize, CropResult> = BTreeMap::new();
        if rep.selection.crops > 0 {
            let (script, weights) = self.prepare(ctx, report, cancel)?;
            check(cancel)?;
            let (stop, failure, out) =
                self.run_zoom(&script, &weights, &rep.selection, ctx, report, cancel)?;
            stopped = stop;
            failed = failure;
            crops = zoom::read_results(&out, stopped || failed.is_some(), &mut rep.warnings)?;
        }
        rep.crops_done = crops.len();
        if crops.is_empty() {
            if stopped {
                return Err(anyhow!(crate::resources::export::Cancelled));
            }
            if let Some(e) = failed {
                return Err(e);
            }
        }
        // 4. Write (no cancel checks: a stopped run writes what it has).
        let wanted = Wanted {
            classes: &self.classes,
            hidden: &self.hidden,
        };
        let frames = std::mem::take(&mut rep.selection.frames);
        rep.written = write::write_frames(
            &self.dataset_dir,
            &frames,
            &crops,
            &wanted,
            self.request.frames.again(),
            &mut rep.warnings,
            &mut |i, n| {
                let pct = u8::try_from(i * 100 / n.max(1)).unwrap_or(100);
                self.step(report, 4, pct, &format!("{i} of {n} frames"));
            },
        );
        rep.cancelled = stopped;
        rep.seconds = (started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
        let w = &rep.written;
        self.step(
            report,
            4,
            100,
            &format!("{} frames gained suggestions", w.frames),
        );
        let n = crate::training::import::preview::thousands;
        let mut summary = if stopped {
            format!(
                "stopped by Cancel after {} of {} crops; those were saved and a rerun does the \
                 rest; ",
                n(rep.crops_done),
                n(rep.selection.crops),
            )
        } else {
            String::new()
        };
        summary.push_str(&format!(
            "{} pairs checked on {} frames: {} drops suggested ({} with a warning, {} from the \
             witnesses alone), {} pairs that look like two objects, {} left for you without a \
             suggestion, on {} frames. No box was changed: confirm them on the Review tab (Auto: \
             overlapping boxes, Status: any, Frames: all collected)",
            n(rep.selection.pairs),
            n(rep.selection.frame_count),
            n(w.drops),
            n(w.weak),
            n(w.low_confidence),
            n(w.two_objects),
            n(w.unclear),
            n(w.frames),
        ));
        if !rep.warnings.is_empty() {
            summary.push_str(&format!("; {} warnings", rep.warnings.len()));
        }
        if let Some(e) = failed {
            return Err(e.context(format!(
                "the {} finished crops were saved and a rerun does the rest",
                n(rep.crops_done)
            )));
        }
        Ok(TaskOutcome {
            summary,
            result: serde_json::to_value(&rep)?,
        })
    }
}

impl Task for OverlapsJob {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> TaskKind {
        TaskKind::Overlaps
    }
    fn title(&self) -> String {
        format!("overlap check ({})", self.request.frames.describe())
    }
    fn steps(&self) -> usize {
        STAGES.len()
    }
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        // Until this returns, a storage sweep leaves `work` alone; after a crash it removes it
        // ([`crate::storage::sweep`]).
        let _busy = crate::storage::Busy::hold(&self.work);
        let r = self.stages(ctx, report, cancel);
        let _ = crate::resources::manager::remove_any(&self.work);
        r
    }
}

#[cfg(test)]
mod tests;
