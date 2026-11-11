//! The dataset check's stage machine ([`RefineJob`], run on the training queue as a [`Task`]):
//! select the reviewed frames once (one label snapshot for every fold) -> per fold: export the
//! frames with the fold held out, train a throwaway model on the rest, check that it never saw a
//! fold frame, run it on the fold's frames, write the suggestions, delete the fold's folder -> the
//! report. See [`super`] for what it may change: suggestions and the report, nothing else.
//!
//! Steps: 1 selecting, then one per fold, then the report (K + 2; K may be lowered to the camera
//! days, and the reports then carry the lower count).
//!
//! **Fold training** reuses the training run as it is ([`run_training`]): a copy of the planned
//! [`TrainJob`] per fold with its run folder, dataset and install folder inside the work folder,
//! [`NoHooks`] (registers nothing, benchmarks nothing), and no record in `runs.json` (only the
//! queue writes those, and this calls the stage machine directly). It takes the training lock
//! itself, per fold: this job must not hold it.
//!
//! **Cancel** between or inside folds keeps the suggestions of the finished folds (a fold is
//! written whole or not at all) and ends with a partial report and summary (`Ok`; a cancel before
//! any fold finished is a plain cancel). **Pause** suspends the fold's training processes; no new
//! process starts while paused (the queue's runner), but this process's own work (the export, the
//! prediction, the writes) runs on until it starts the next one. **A failed fold** stops the job;
//! the folds before it keep their suggestions and the report says how far it got.
//!
//! The work folder (`<training dir>/refine/<id>`) is held busy for the whole run, so a storage sweep
//! leaves it alone, and removed at the end however the run ends.

use super::folds::Group;
use super::judge::{self, Context};
use super::predict::{FoldModel, FoldPredictor, LocalPredictor};
use super::report::{RecallTally, RefineReport};
use super::{RefineRequest, Selected, fold_model_name, recipe_info, select, write};
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::export::{
    self, BenchSplit, ClassFilter, ExportFrame, ExportRequest, Format, Selection, Split,
    SplitConfig,
};
use crate::config::Config;
use crate::resources::export::Cancelled;
use crate::resources::manager;
use crate::training::autolabel::relabel::Image;
use crate::training::job::{self, NoHooks, TrainJob, TrainReport, TrainStage};
use crate::training::pyenv::{check, is_cancel, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Context as _, Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Name of the first stage.
pub const SELECTING: &str = "selecting the reviewed frames";
/// Name of the last stage.
pub const REPORTING: &str = "writing the report";

/// The start of the sentence a failed training run adds to its message about its kept checkpoint
/// and `train retry-export <id>` (`job::kept_note`). A fold has no run record to retry: the job
/// drops that sentence and says what to do instead.
const KEPT_NOTE: &str = ". Its best checkpoint is kept (";

/// A planned check ([`super::plan_refine`]).
#[derive(Clone)]
pub struct RefineJob {
    pub id: String,
    pub request: RefineRequest,
    /// The folder under review.
    pub dataset_dir: PathBuf,
    /// `collect.classes`: the export's class list and the fold models' classes.
    pub classes: Vec<String>,
    /// `collect.hidden_classes`: not judged.
    pub hidden: Vec<String>,
    /// `<training dir>/refine/<id>`: one folder per fold (export, dataset, checkpoints, model);
    /// deleted at the end.
    pub work: PathBuf,
    /// The training run every fold copies ([`super::super::plan_job`]): recipe, GPU, paths,
    /// platform, timeout, workers.
    pub train: TrainJob,
    /// Where the recipe came from ([`super::choose_recipe`]).
    pub recipe_from: String,
    /// Runs a fold model over its frames (None: [`LocalPredictor`] on `config`, its own runtimes;
    /// the service passes the registry's).
    pub predictor: Option<Arc<dyn FoldPredictor>>,
    /// A copy of the config for [`LocalPredictor`]; never saved.
    pub config: Box<Config>,
    /// Fewest frames a check runs on ([`super::MIN_FRAMES`]; tests lower it).
    pub min_frames: usize,
}

impl std::fmt::Debug for RefineJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefineJob")
            .field("id", &self.id)
            .field("request", &self.request)
            .field("dataset_dir", &self.dataset_dir)
            .field("work", &self.work)
            .finish_non_exhaustive()
    }
}

/// The result of a finished (or stopped) check: the report, where it is, and the first frames
/// written.
#[derive(Debug, Clone, Serialize)]
pub struct RefineResult {
    #[serde(flatten)]
    pub report: RefineReport,
    /// `<review folder>/refine/report.json`; None when it could not be written (a warning says
    /// why).
    pub report_path: Option<PathBuf>,
    pub sample: Vec<String>,
    pub seconds: f64,
}

/// Why the folds stopped early.
enum Stop {
    Cancelled,
    Failed(anyhow::Error),
}

/// What a fold calls the end of its training run: the model files go into the fold's folder.
const FOLD_MODEL: &str = "preparing the fold model";

/// Steps of a check with `folds` folds: selecting, one per fold, the report.
const fn steps_for(folds: usize) -> usize {
    folds + 2
}

/// What a fold's training run shows of its stage `stage` at `percent` with `detail` (None:
/// nothing). A fold model is never added to the config nor benchmarked ([`NoHooks`] does neither),
/// so those two stages are not shown at all, and the install is [`FOLD_MODEL`] without the fold
/// folder's path. Ordinary training runs keep [`TrainStage::label`]; only this adapter rewords.
fn fold_stage_text(stage: TrainStage, percent: u8, detail: &str) -> Option<String> {
    match stage {
        TrainStage::Register | TrainStage::Benchmark => None,
        TrainStage::Install if percent >= 100 => Some(format!("{FOLD_MODEL}: done")),
        TrainStage::Install => Some(format!("{FOLD_MODEL}: {detail}")),
        s => Some(format!("{}: {detail}", s.label())),
    }
}

/// What a fold's failure message names as the stage that failed ([`fold_stage_text`]'s words).
fn fold_stage_label(stage: TrainStage) -> &'static str {
    match stage {
        TrainStage::Install | TrainStage::Register | TrainStage::Benchmark => FOLD_MODEL,
        s => s.label(),
    }
}

/// The training run's failure message without its retry advice ([`KEPT_NOTE`]).
fn without_kept_note(message: &str) -> &str {
    message.find(KEPT_NOTE).map_or(message, |at| &message[..at])
}

/// The flat names in `<data>/train.txt` and `valid.txt` (the training dataset's image lines,
/// `./images/s<source>-<flat name>`), back to flat names as the export wrote them.
fn trained_names(data: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for list in ["train.txt", "valid.txt"] {
        let path = data.join(list);
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "reading {} to check the fold model never saw its frames",
                path.display()
            )
        })?;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let name = line.rsplit(['/', '\\']).next().unwrap_or(line);
            out.insert(strip_source(name).to_string());
        }
    }
    Ok(out)
}

/// `name` without the dataset's `s<digits>-` source prefix.
fn strip_source(name: &str) -> &str {
    let Some(rest) = name.strip_prefix('s') else {
        return name;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    match rest.get(digits..).and_then(|r| r.strip_prefix('-')) {
        Some(r) if digits > 0 => r,
        _ => name,
    }
}

/// A flat name as the training dataset spells its image (the dataset's file-name rule: letters,
/// digits, `-`, `_` and `.`, others `_`, at most 120 bytes with the `s0-` prefix), without the
/// prefix.
fn dataset_spelling(flat: &str) -> String {
    let mut s: String = format!("s0-{flat}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    s.truncate(120);
    strip_source(&s).to_string()
}

/// Most frames of a check's sample (the first frames written).
const SAMPLE: usize = 20;

/// Count what writing a fold did (`w`) into the report and the sample.
fn count_written(rep: &mut RefineReport, sample: &mut Vec<String>, w: &write::Written) {
    rep.frames_checked += w.checked;
    rep.frames_written += w.frames;
    rep.loose += w.loose;
    rep.found.add += w.found.add;
    rep.found.verify += w.found.verify;
    rep.found.relabel += w.found.relabel;
    rep.written.add += w.written.add;
    rep.written.verify += w.written.verify;
    rep.written.relabel += w.written.relabel;
    let room = SAMPLE.saturating_sub(sample.len());
    sample.extend(w.sample.iter().take(room).cloned());
}

/// The hard check before predicting: no frame of the fold is in the fold model's training or
/// validation lists. One that is would make the check meaningless, so the job fails loudly.
pub fn check_unseen(data: &Path, fold: &[&ExportFrame]) -> Result<()> {
    let trained = trained_names(data)?;
    let leaked: Vec<&str> = fold
        .iter()
        .filter(|f| trained.contains(&dataset_spelling(&f.name)) || trained.contains(&f.name))
        .map(|f| f.name.as_str())
        .collect();
    if !leaked.is_empty() {
        bail!(
            "{} frame(s) of the held-out fold are in the fold model's training or validation \
             lists ({}{}): the check would judge frames the model trained on, so it stops here. \
             This is a bug in the dataset assembly; report it with this message",
            leaked.len(),
            leaked
                .iter()
                .take(5)
                .copied()
                .collect::<Vec<_>>()
                .join(", "),
            if leaked.len() > 5 { ", ..." } else { "" }
        );
    }
    Ok(())
}

impl RefineJob {
    fn step(
        report: &mut dyn FnMut(TaskReport<'_>),
        name: &str,
        n: usize,
        steps: usize,
        pct: u8,
        detail: &str,
    ) {
        stage(report, name, n, steps, pct, detail);
    }

    /// Stage 1 ([`select`]).
    pub fn select(&self) -> Selected {
        select(
            &self.dataset_dir,
            &self.classes,
            &self.hidden,
            &self.request.frames,
            self.request.folds,
            self.min_frames,
        )
    }

    /// The classes the judge looks at: the class list without the hidden ones.
    fn wanted(&self) -> Wanted<'_> {
        Wanted {
            classes: &self.classes,
            hidden: &self.hidden,
        }
    }

    /// What every frame of the check shares in the judge: the fold models' input size, the
    /// selection's training boxes per class, this check's id.
    fn judge_context<'a>(&'a self, wanted: &'a Wanted<'a>, s: &'a Selected) -> Context<'a> {
        Context {
            wanted,
            imgsz: self.train.request.recipe.imgsz,
            class_boxes: &s.class_boxes,
            run: &self.id,
        }
    }

    /// The report of a check over the selection `s` before any fold ran.
    fn start_report(&self, s: &Selected) -> RefineReport {
        let k = s.k;
        let recipe = &self.train.request.recipe;
        let mut rep = RefineReport::new(&self.id, k, recipe_info(recipe, &self.recipe_from));
        rep.camera_days = s.groups.len();
        rep.frames_selected = s.frames.len();
        rep.left_out = s.left_out.clone();
        rep.warnings.clone_from(&s.warnings);
        if k < self.request.folds {
            rep.warnings.push(format!(
                "{} folds asked for, {} camera days: {k} folds",
                self.request.folds,
                s.groups.len()
            ));
        }
        rep
    }

    fn predictor(&self) -> Arc<dyn FoldPredictor> {
        self.predictor
            .clone()
            .unwrap_or_else(|| Arc::new(LocalPredictor::new((*self.config).clone(), None)))
    }

    /// One fold (0-based `i` of `s.k`): export, train, check, predict, write, clean up. Its
    /// suggestions are written only once the fold model's results are complete.
    #[allow(clippy::too_many_arguments)]
    fn run_fold(
        &self,
        i: usize,
        s: &Selected,
        jctx: &Context<'_>,
        predictor: &dyn FoldPredictor,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
        tally: &mut RecallTally,
        rep: &mut RefineReport,
        sample: &mut Vec<String>,
    ) -> Result<()> {
        let k = s.k;
        let n = i + 1;
        let steps = steps_for(k);
        let at = i + 2;
        let name = format!("checking fold {n} of {k}");
        let fold_dir = self.work.join(format!("fold-{n}"));
        let fold_of = |f: &ExportFrame| -> Option<usize> {
            let g: Group = (f.rel.camera.clone(), f.rel.date.clone());
            s.folds.get(&g).copied()
        };
        let fold: Vec<&ExportFrame> = s.frames.iter().filter(|f| fold_of(f) == Some(i)).collect();
        // (a) Export with the fold held out.
        Self::step(
            report,
            &name,
            at,
            steps,
            0,
            &format!(
                "exporting: {} frames held out, {} to train on",
                fold.len(),
                s.frames.len() - fold.len()
            ),
        );
        manager::remove_any(&fold_dir)?;
        std::fs::create_dir_all(&fold_dir)
            .with_context(|| format!("creating {}", fold_dir.display()))?;
        let export_dir = fold_dir.join("export");
        let selection = Selection {
            frames: s
                .frames
                .iter()
                .map(|f| ExportFrame {
                    split: if fold_of(f) == Some(i) {
                        Split::Val
                    } else {
                        Split::Train
                    },
                    ..f.clone()
                })
                .collect(),
            ..Selection::default()
        };
        let req = ExportRequest {
            formats: vec![Format::Yolo],
            out_dir: export_dir.clone(),
            split: SplitConfig::default(),
            benchmark_split: BenchSplit::Val,
            include_unreviewed: false,
            hidden_classes: Vec::new(),
        };
        let exported = export::run_selected(
            &self.dataset_dir,
            &self.classes,
            &req,
            &ClassFilter::default(),
            selection,
        )
        .with_context(|| format!("fold {n} of {k}: exporting its frames"))?;
        if exported.out_dir != export_dir {
            bail!(
                "fold {n} of {k}: the export went to {} instead of {}; the work folder was not \
                 empty",
                exported.out_dir.display(),
                export_dir.display()
            );
        }
        check(cancel)?;
        // (b) Train on the other folds.
        let mut tj = self.train.clone();
        tj.id = format!("{}-fold{n}", self.id);
        tj.name = fold_model_name(n);
        tj.run_dir.clone_from(&fold_dir);
        tj.request.datasets = vec![export_dir];
        tj.request.install_dir = Some(fold_dir.join("model"));
        tj.keep_work = false;
        tj.export_only = None;
        let trained = job::run_training(
            &tj,
            ctx.runner,
            ctx.fetcher,
            &NoHooks,
            &mut |p| match p {
                TrainReport::Stage {
                    stage: ts,
                    percent,
                    detail,
                } => {
                    let Some(d) = fold_stage_text(ts, percent, detail) else {
                        return;
                    };
                    let pct = 5 + u32::from(ts.overall(percent)) * 80 / 100;
                    Self::step(
                        report,
                        &name,
                        at,
                        steps,
                        u8::try_from(pct).unwrap_or(85),
                        &d,
                    );
                }
                TrainReport::Line(l) => report(TaskReport::Line(l)),
                _ => {}
            },
            cancel,
        );
        let out = match trained {
            Ok(o) => o.output,
            Err(e) if e.cancelled || cancel.load(Ordering::Relaxed) => {
                return Err(anyhow!(Cancelled));
            }
            Err(e) => {
                let during = e
                    .stage
                    .map(|st| format!(" while {}", fold_stage_label(st)))
                    .unwrap_or_default();
                bail!(
                    "fold {n} of {k}: training its model failed{during}: {}. A fold model is \
                     throwaway (there is nothing to retry or export): fix the cause, then run \
                     the check again",
                    without_kept_note(&e.message)
                );
            }
        };
        // (c) The model never saw a frame of its fold.
        check_unseen(&fold_dir.join("data"), &fold).with_context(|| format!("fold {n} of {k}"))?;
        check(cancel)?;
        // (d) Predict.
        Self::step(
            report,
            &name,
            at,
            steps,
            86,
            &format!("running the fold model on its {} frames", fold.len()),
        );
        let images: Vec<Image> = fold
            .iter()
            .map(|f| Image {
                frame: f.rel.jpg(),
                path: f.image.clone(),
                width: f.width,
                height: f.height,
            })
            .collect();
        let model = FoldModel {
            name: out.name,
            onnx: out.onnx,
            yaml: out.yaml,
            family: out.family,
            classes: out.classes,
        };
        let found = predictor
            .predict(&model, &images, judge::MATCH_MIN_SCORE, cancel)
            .with_context(|| {
                format!(
                    "fold {n} of {k}: running the fold model {} on its frames",
                    model.onnx.display()
                )
            })?;
        if found.cancelled || cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(Cancelled));
        }
        // (e, f) Judge and write, under the label write lock (no cancel checks: a fold is written
        // whole).
        let w = write::write_fold(
            &self.dataset_dir,
            &fold,
            &found,
            jctx,
            tally,
            &mut rep.warnings,
            &mut |d, total| {
                let pct = 95 + d * 5 / total.max(1);
                Self::step(
                    report,
                    &name,
                    at,
                    steps,
                    u8::try_from(pct).unwrap_or(99),
                    &format!("writing suggestions: {d} of {total} frames"),
                );
            },
        );
        count_written(rep, sample, &w);
        Self::step(
            report,
            &name,
            at,
            steps,
            100,
            &format!("{} frames checked, {} changed", w.checked, w.frames),
        );
        // (g) The fold's export, dataset, checkpoints and model go.
        if let Err(e) = manager::remove_any(&fold_dir) {
            tracing::warn!(
                "dataset check: could not remove {}: {e:#}",
                fold_dir.display()
            );
        }
        Ok(())
    }

    fn stages(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let started = Instant::now();
        // 1. One label snapshot for every fold.
        Self::step(
            report,
            SELECTING,
            1,
            self.steps(),
            0,
            &self.request.frames.describe(),
        );
        let s = self.select();
        if let Some(why) = &s.refused {
            bail!("{why}");
        }
        let k = s.k;
        // From here on the count the folds were lowered to ([`Task::steps`] has the asked one).
        let steps = steps_for(k);
        let mut rep = self.start_report(&s);
        Self::step(
            report,
            SELECTING,
            1,
            steps,
            100,
            &format!(
                "{} frames on {} camera days, {k} folds",
                s.frames.len(),
                s.groups.len()
            ),
        );
        check(cancel)?;
        // 2. The folds.
        let wanted = self.wanted();
        let jctx = self.judge_context(&wanted, &s);
        let predictor = self.predictor();
        let mut tally = RecallTally::default();
        let mut sample = Vec::new();
        let mut stop = None;
        for i in 0..k {
            if cancel.load(Ordering::Relaxed) {
                stop = Some(Stop::Cancelled);
                break;
            }
            let r = self.run_fold(
                i,
                &s,
                &jctx,
                predictor.as_ref(),
                ctx,
                report,
                cancel,
                &mut tally,
                &mut rep,
                &mut sample,
            );
            match r {
                Ok(()) => rep.folds_done += 1,
                Err(e) if is_cancel(&e) || cancel.load(Ordering::Relaxed) => {
                    stop = Some(Stop::Cancelled);
                    break;
                }
                Err(e) => {
                    stop = Some(Stop::Failed(e));
                    break;
                }
            }
        }
        if rep.folds_done == 0 {
            match stop {
                Some(Stop::Cancelled) => return Err(anyhow!(Cancelled)),
                Some(Stop::Failed(e)) => return Err(e),
                None => {}
            }
        }
        // 3. The report (also of a stopped check: what its finished folds found).
        Self::step(report, REPORTING, steps, steps, 0, "");
        rep.cancelled = matches!(stop, Some(Stop::Cancelled));
        if let Some(Stop::Failed(e)) = &stop {
            rep.warnings.push(format!("the check stopped: {e:#}"));
        }
        rep.finish(&tally);
        let report_path = match rep.write(&self.dataset_dir) {
            Ok(p) => Some(p),
            Err(e) => {
                rep.warnings.push(format!("{e:#}"));
                None
            }
        };
        Self::step(report, REPORTING, steps, steps, 100, "done");
        let n = crate::training::import::preview::thousands;
        let w = &rep.written;
        let f = &rep.found;
        let mut summary = match &stop {
            Some(Stop::Cancelled) => format!(
                "stopped by Cancel after {} of {k} folds; their suggestions were saved and a \
                 rerun checks every fold again; ",
                rep.folds_done
            ),
            _ => String::new(),
        };
        summary.push_str(&format!(
            "{} frames checked by {} fold models: {} missing boxes, {} labels to verify, {} other \
             classes suggested ({} new), {} loose labels; {} frames changed. No box was changed: \
             confirm them on the Review tab",
            n(rep.frames_checked),
            rep.folds_done,
            n(f.add),
            n(f.verify),
            n(f.relabel),
            n(w.total()),
            n(rep.loose),
            n(rep.frames_written),
        ));
        if let Some(p) = &report_path {
            summary.push_str(&format!("; report: {}", p.display()));
        }
        if !rep.warnings.is_empty() {
            summary.push_str(&format!("; {} warnings", rep.warnings.len()));
        }
        let result = RefineResult {
            report: rep,
            report_path,
            sample,
            seconds: (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        };
        if let Some(Stop::Failed(e)) = stop {
            return Err(e.context(format!(
                "the suggestions of the {} finished fold(s) were saved{}",
                result.report.folds_done,
                result
                    .report_path
                    .as_ref()
                    .map(|p| format!(" and the report written to {}", p.display()))
                    .unwrap_or_default()
            )));
        }
        Ok(TaskOutcome {
            summary,
            result: serde_json::to_value(&result)?,
        })
    }
}

impl Task for RefineJob {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> TaskKind {
        TaskKind::Refine
    }
    fn title(&self) -> String {
        format!(
            "dataset check ({} folds x {} epochs, {})",
            self.request.folds,
            self.request.epochs,
            self.request.frames.describe()
        )
    }
    /// The steps of the asked fold count. The folds may be lowered to the camera days, which only
    /// the selection at run time knows (planning reads no frames, and the label snapshot must be
    /// the one the folds use), so this is the queued record's count until then: every stage
    /// report carries its own `steps` and the queue takes it from there (the first report still
    /// this count, every one after the selection [`steps_for`] of the lowered count).
    fn steps(&self) -> usize {
        steps_for(self.request.folds)
    }
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        // Until this returns, a storage sweep leaves `work` alone; after a crash it removes it
        // ([`crate::storage::sweep`]).
        let busy = crate::storage::Busy::hold(&self.work);
        let r = self.stages(ctx, report, cancel);
        if let Err(e) = manager::remove_any(&self.work) {
            tracing::warn!(
                "dataset check: could not remove its work folder {}: {e:#}",
                self.work.display()
            );
        }
        // The marker beside the work folder goes with the hold; then an empty `refine/` parent.
        drop(busy);
        if let Some(parent) = self.work.parent() {
            let _ = std::fs::remove_dir(parent);
        }
        r
    }
}

#[cfg(test)]
mod tests;
