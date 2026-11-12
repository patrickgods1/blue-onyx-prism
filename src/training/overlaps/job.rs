//! The overlap check's stage machine ([`OverlapsJob`], run on the training queue as a [`Task`]):
//! select the pairs -> set up the environment and RF-DETR L's weights -> zoom check -> write the
//! suggestions. See [`super`] for what it may change: suggestions, nothing else.
//!
//! A cancel or a failure of the script after some crops keeps them: the finished crops are written
//! like a finished run's (no cancel checks in the write stage) and a rerun skips the pairs with an
//! answer. A cancel returns `Ok` with a "stopped by Cancel" summary; a failure returns its error, with
//! how many crops were kept.

use super::OverlapsRequest;
use super::select::{self, Selection};
use super::write::{self, Written};
use super::zoom::{self, CropResult};
use crate::backend::memory::Commit;
use crate::collect::labels::bulk::Wanted;
use crate::resources::catalog::Platform;
use crate::resources::export::train::TrainPaths;
use crate::training::capability::Accelerator;
use crate::training::pyenv::{check, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
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
    zoom::commit_problem("the overlap check", free)
        .map(|why| format!("{why}, then run it again (nothing was changed)"))
}

impl OverlapsJob {
    fn step(&self, report: &mut dyn FnMut(TaskReport<'_>), n: usize, pct: u8, detail: &str) {
        let name = STAGES.get(n.saturating_sub(1)).copied().unwrap_or("");
        stage(report, name, n, STAGES.len(), pct, detail);
    }

    fn helper(&self) -> zoom::Helper<'_> {
        zoom::Helper {
            paths: &self.paths,
            platform: self.platform,
            accelerator: &self.accelerator,
            timeout: self.timeout,
        }
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
        zoom::prepare(
            &self.helper(),
            self.request.allow_large,
            ctx,
            report,
            (STAGES[1], 2, STAGES.len()),
            cancel,
        )
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
        let ran = zoom::run(
            &self.helper(),
            &self.work,
            script,
            weights,
            &self.classes,
            &selection.frames,
            selection.crops,
            ctx,
            report,
            (STAGES[2], 3, STAGES.len()),
            cancel,
        )?;
        Ok((ran.stopped, ran.failed, ran.out))
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
