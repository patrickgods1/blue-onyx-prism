//! The dataset check's stage machine ([`RefineJob`], run on the training queue as a [`Task`]):
//! select the reviewed frames once (one label snapshot for every fold) -> per fold: export the
//! frames with the fold held out, train a throwaway model on the rest, check that it never saw a
//! fold frame, run it on the fold's frames, write the suggestions, delete the fold's folder -> the
//! report. See [`super`] for what it may change: suggestions and the report, nothing else.
//!
//! Steps: 1 selecting, then one per fold, then the report (K + 2; K may be lowered to the camera
//! days, and the reports then carry the lower count). The selection also makes the check's static
//! spots ([`super::Selected::spots`]) from the same snapshot; every fold's judge uses that one table.
//!
//! **Fold training** reuses the training run as it is ([`job::run_training_holding`]): a copy of
//! the planned [`TrainJob`] per fold with its run folder, dataset and install folder inside the work
//! folder, [`NoHooks`] (registers nothing, benchmarks nothing), and no record in `runs.json` (only
//! the queue writes those, and this calls the stage machine directly). The check takes the training
//! lock once, after the selection, and holds it to the last fold's writes (like the auto-label and
//! overlap jobs hold it for their GPU work); each fold's training uses that hold, so no other
//! process trains or labels between a fold's training and its prediction or between two folds.
//!
//! **Frames gone meanwhile.** The selection is one snapshot; each fold leaves out the frames whose
//! image or label file went since ([`RefineJob::fold_frames`]), and a frame the fold model cannot
//! read gets no result ([`super::predict::skipping_unreadable`]): both are counted in a warning, the
//! fold goes on. A folder under review that is not available stops the check with the Review
//! page's words.
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

use super::judge::{self, Context};
use super::predict::{FoldModel, FoldPredictor, LocalPredictor};
use super::report::{RecallTally, RefineReport};
use super::{RefineRequest, Selected, fold_model_name, recipe_info, select, write};
use crate::collect::labels::Frame;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::export::{
    self, BenchSplit, ClassFilter, ExportFrame, ExportRequest, Format, Selection, Split,
    SplitConfig,
};
use crate::config::Config;
use crate::resources::export::Cancelled;
use crate::resources::export::train::TrainLock;
use crate::resources::manager;
use crate::training::autolabel::relabel::{Detections, Image};
use crate::training::dataset;
use crate::training::job::{self, NoHooks, TrainJob, TrainReport, TrainStage};
use crate::training::pyenv::{check, is_cancel, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Context as _, Result, anyhow, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
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

/// The image lines of `<data>/train.txt` and `valid.txt` (`./images/s<source>-<flat name>`, a
/// line per repeat), each once.
fn trained_lines(data: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for list in ["train.txt", "valid.txt"] {
        let path = data.join(list);
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "reading {} to check the fold model never saw its frames",
                path.display()
            )
        })?;
        out.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string),
        );
    }
    Ok(out)
}

/// The flat names of trained `lines`, back to flat names as the export wrote them.
fn trained_names(lines: &BTreeSet<String>) -> BTreeSet<String> {
    lines
        .iter()
        .map(|line| {
            let name = line.rsplit(['/', '\\']).next().unwrap_or(line);
            strip_source(name).to_string()
        })
        .collect()
}

/// The fold frames among the sources of the images the lists `lines` name, by the dataset's own
/// record of where each image came from ([`dataset::IMAGE_SOURCES_FILE`] in `data`): a source is
/// an image of the fold's export, whose file name is the frame's flat name. Exact whatever the
/// dataset named the image (its 120-byte cut, a `-2` for a name taken twice). A listed image the
/// record does not name, or no record, is an error: the check cannot be made.
fn by_provenance<'f>(
    data: &Path,
    lines: &BTreeSet<String>,
    fold: &[&'f ExportFrame],
) -> Result<Vec<&'f ExportFrame>> {
    let sources = dataset::read_image_sources(data)
        .context("reading where the fold model's training images came from")?;
    let mut trained: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        let name = line.rsplit(['/', '\\']).next().unwrap_or(line);
        let Some(src) = sources.get(name) else {
            bail!(
                "{name} in the fold model's training lists is not in {}: the check cannot tell \
                 which frame it is",
                data.join(dataset::IMAGE_SOURCES_FILE).display()
            );
        };
        if let Some(n) = src.file_name() {
            trained.insert(n.to_string_lossy().into_owned());
        }
    }
    Ok(fold
        .iter()
        .filter(|f| trained.contains(&f.name))
        .copied()
        .collect())
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
    rep.adds_at_spots += w.add_at_spots;
    rep.written.add += w.written.add;
    rep.written.verify += w.written.verify;
    rep.written.relabel += w.written.relabel;
    let room = SAMPLE.saturating_sub(sample.len());
    sample.extend(w.sample.iter().take(room).cloned());
}

/// What the folds of a check add up: the report, the recall tally, the sample, and how many
/// per-frame write failures the report names ([`write::MAX_FAILURES`] over the whole check; the
/// rest are counted in one line, so `report.json` and the task record stay small).
#[derive(Debug)]
pub struct Acc {
    pub rep: RefineReport,
    pub tally: RecallTally,
    pub sample: Vec<String>,
    /// Write failures named in `rep.warnings`.
    pub failures_shown: usize,
    /// Write failures not named.
    pub failures_more: usize,
}

impl Acc {
    pub fn new(rep: RefineReport) -> Self {
        Self {
            rep,
            tally: RecallTally::default(),
            sample: Vec::new(),
            failures_shown: 0,
            failures_more: 0,
        }
    }

    /// Name the write failures of `w` in the report while fewer than [`write::MAX_FAILURES`] are,
    /// count the others.
    pub fn note_failures(&mut self, w: &write::Written) {
        for f in &w.failures {
            if self.failures_shown < write::MAX_FAILURES {
                self.rep.warnings.push(f.clone());
                self.failures_shown += 1;
            } else {
                self.failures_more += 1;
            }
        }
        self.failures_more += w.failed.saturating_sub(w.failures.len());
    }

    /// The one line for the failures not named, when there are any (at the end of a check).
    pub fn close_failures(&mut self) {
        if self.failures_more > 0 {
            self.rep.warnings.push(format!(
                "... and {} more frames whose suggestions could not be written",
                self.failures_more
            ));
            self.failures_more = 0;
        }
    }
}

/// The hard check before predicting: no frame of the fold is in the fold model's training or
/// validation lists (`<data>/train.txt`, `valid.txt`): by name, then by the source the dataset
/// recorded for every image they list ([`by_provenance`]: the dataset's own names can differ from
/// the export's). One that is would make the check meaningless, so the job fails loudly.
pub fn check_unseen(data: &Path, fold: &[&ExportFrame]) -> Result<()> {
    let lines = trained_lines(data)?;
    let trained = trained_names(&lines);
    let mut leaked: Vec<&str> = fold
        .iter()
        .filter(|f| trained.contains(&dataset_spelling(&f.name)) || trained.contains(&f.name))
        .map(|f| f.name.as_str())
        .collect();
    if leaked.is_empty() {
        leaked = by_provenance(data, &lines, fold)?
            .iter()
            .map(|f| f.name.as_str())
            .collect();
    }
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
            self.config.collect.size_floor,
        )
    }

    /// The classes the judge looks at: the class list without the hidden ones.
    fn wanted(&self) -> Wanted<'_> {
        Wanted {
            classes: &self.classes,
            hidden: &self.hidden,
        }
    }

    /// What every frame of a fold shares in the judge: the fold models' input size, the training
    /// boxes per class of what this fold's model trained on (`class_boxes`,
    /// [`super::class_boxes`]), this check's id, and the check's static spots (`spots`, made once
    /// from the selection: [`Selected::spots`]).
    fn judge_context<'a>(
        &'a self,
        wanted: &'a Wanted<'a>,
        class_boxes: &'a BTreeMap<String, usize>,
        spots: &'a crate::collect::labels::statics::Table,
    ) -> Context<'a> {
        Context {
            wanted,
            imgsz: self.train.request.recipe.imgsz,
            class_boxes,
            run: &self.id,
            spots: Some(spots),
            floor: self.config.collect.size_floor,
        }
    }

    /// The frames of fold `i` of `s` still on disk, as the fold uses them: (held out, trained on,
    /// gone). The selection is one snapshot for every fold, and a fold may start an hour after it:
    /// a frame whose image or label file was deleted or moved since is left out of both sides
    /// (the export would fail on it), and counted.
    pub fn fold_frames<'s>(
        &self,
        s: &'s Selected,
        i: usize,
    ) -> (Vec<&'s ExportFrame>, Vec<&'s ExportFrame>, usize) {
        let present = |f: &&ExportFrame| {
            f.image.is_file()
                && Frame::resolve(&self.dataset_dir, &f.rel.jpg())
                    .is_some_and(|fr| fr.labels.is_file())
        };
        let (held, train) = s.fold_frames(i);
        let all = held.len() + train.len();
        let held: Vec<&ExportFrame> = held.into_iter().filter(present).collect();
        let train: Vec<&ExportFrame> = train.into_iter().filter(present).collect();
        let gone = all - held.len() - train.len();
        (held, train, gone)
    }

    /// The report of a check over the selection `s` before any fold ran.
    fn start_report(&self, s: &Selected) -> RefineReport {
        let k = s.k;
        let recipe = &self.train.request.recipe;
        let mut rep = RefineReport::new(&self.id, k, recipe_info(recipe, &self.recipe_from));
        rep.camera_days = s.groups.len();
        rep.frames_selected = s.frames.len();
        rep.static_spots = s.spots.len();
        rep.left_out = s.left_out.clone();
        rep.warnings.clone_from(&s.warnings);
        if k < self.request.folds {
            rep.warnings.push(format!(
                "{} folds asked for, {} camera days: {k} folds",
                self.request.folds,
                s.groups.len()
            ));
        }
        rep.warnings
            .extend(s.fold_warnings(&self.classes, &self.hidden));
        rep
    }

    /// Write fold `n` of `k`'s suggestions for its `held` frames from `found` (judged with `jctx`)
    /// and count them into `acc`; an error when every frame with a result failed to write (then
    /// nothing of the fold was saved, and it does not count as done). The real fold and the replay
    /// test share it.
    #[allow(clippy::too_many_arguments)]
    fn write_fold_into(
        &self,
        n: usize,
        k: usize,
        held: &[&ExportFrame],
        found: &Detections,
        jctx: &Context<'_>,
        acc: &mut Acc,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<write::Written> {
        let w = write::write_fold(
            &self.dataset_dir,
            held,
            found,
            jctx,
            &mut acc.tally,
            progress,
        );
        acc.note_failures(&w);
        if w.missing > 0 && !found.cancelled {
            acc.rep.warnings.push(format!(
                "fold {n} of {k}: {} frame(s) got no result from the fold model (the image is \
                 gone or cannot be read); they were not checked",
                w.missing
            ));
        }
        if w.all_failed() {
            bail!(
                "fold {n} of {k}: none of its {} frames could be written ({}); nothing of this \
                 fold was saved",
                w.attempted,
                w.failures.first().map_or("no reason given", String::as_str)
            );
        }
        count_written(&mut acc.rep, &mut acc.sample, &w);
        Ok(w)
    }

    fn predictor(&self) -> Arc<dyn FoldPredictor> {
        self.predictor
            .clone()
            .unwrap_or_else(|| Arc::new(LocalPredictor::new((*self.config).clone(), None)))
    }

    /// One fold (0-based `i` of `s.k`): export, train, check, predict, write, clean up. Its
    /// suggestions are written only once the fold model's results are complete. `lock` is the
    /// training lock the check holds for its whole run (the fold's training uses it).
    #[allow(clippy::too_many_arguments)]
    fn run_fold(
        &self,
        i: usize,
        s: &Selected,
        lock: &TrainLock,
        predictor: &dyn FoldPredictor,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
        acc: &mut Acc,
    ) -> Result<()> {
        let k = s.k;
        let n = i + 1;
        let steps = steps_for(k);
        let at = i + 2;
        let name = format!("checking fold {n} of {k}");
        let fold_dir = self.work.join(format!("fold-{n}"));
        // The folder under review may have gone since the last fold (an unplugged drive): that is
        // no fold of gone frames.
        if let Some(why) = self.config.collect.review_folder_problem(&self.dataset_dir) {
            bail!("fold {n} of {k}: {why}");
        }
        let (fold, train, gone) = self.fold_frames(s, i);
        if gone > 0 {
            acc.rep.warnings.push(format!(
                "fold {n} of {k}: {gone} selected frame(s) are gone (deleted or moved since the \
                 check started) and were left out of its training and its check"
            ));
        }
        if fold.is_empty() {
            acc.rep.warnings.push(format!(
                "fold {n} of {k}: every frame it holds out is gone; nothing to check"
            ));
            return Ok(());
        }
        if train.is_empty() {
            bail!("fold {n} of {k}: every frame its model would train on is gone");
        }
        let wanted = self.wanted();
        let class_boxes = super::class_boxes(train.iter().copied(), &wanted);
        let jctx = self.judge_context(&wanted, &class_boxes, &s.spots);
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
                train.len()
            ),
        );
        manager::remove_any(&fold_dir)?;
        std::fs::create_dir_all(&fold_dir)
            .with_context(|| format!("creating {}", fold_dir.display()))?;
        let export_dir = fold_dir.join("export");
        let rels =
            |v: &[&ExportFrame]| -> BTreeSet<String> { v.iter().map(|f| f.rel.jpg()).collect() };
        let (held_rels, train_rels) = (rels(&fold), rels(&train));
        let selection = Selection {
            // The selection's order, gone frames left out.
            frames: s
                .frames
                .iter()
                .filter_map(|f| {
                    let rel = f.rel.jpg();
                    let split = if held_rels.contains(&rel) {
                        Split::Val
                    } else if train_rels.contains(&rel) {
                        Split::Train
                    } else {
                        return None;
                    };
                    Some(ExportFrame { split, ..f.clone() })
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
            // The fold models train as a run would on an export of today's settings.
            size_floor: self.config.collect.size_floor,
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
        let trained = job::run_training_holding(
            &tj,
            lock,
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
        let w = self.write_fold_into(n, k, &fold, &found, &jctx, acc, &mut |d, total| {
            let pct = 95 + d * 5 / total.max(1);
            Self::step(
                report,
                &name,
                at,
                steps,
                u8::try_from(pct).unwrap_or(99),
                &format!("writing suggestions: {d} of {total} frames"),
            );
        })?;
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
        // An unavailable folder (an unplugged drive) is its own answer, never an empty dataset.
        if let Some(why) = self.config.collect.review_folder_problem(&self.dataset_dir) {
            bail!("{why}");
        }
        let s = self.select();
        if let Some(why) = &s.refused {
            bail!("{why}");
        }
        // The GPU is this check's from the first fold's training to the last fold's writes: held
        // across the folds, so no other process (the service or a `train` command) trains or
        // labels between a fold's training and its prediction, or takes the next fold's turn.
        let lock = TrainLock::try_acquire(&self.train.paths.train_lock_file())?;
        let k = s.k;
        // From here on the count the folds were lowered to ([`Task::steps`] has the asked one).
        let steps = steps_for(k);
        let mut acc = Acc::new(self.start_report(&s));
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
        let predictor = self.predictor();
        let mut stop = None;
        for i in 0..k {
            if cancel.load(Ordering::Relaxed) {
                stop = Some(Stop::Cancelled);
                break;
            }
            let r = self.run_fold(
                i,
                &s,
                &lock,
                predictor.as_ref(),
                ctx,
                report,
                cancel,
                &mut acc,
            );
            match r {
                Ok(()) => acc.rep.folds_done += 1,
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
        drop(lock);
        if acc.rep.folds_done == 0 {
            match stop {
                Some(Stop::Cancelled) => return Err(anyhow!(Cancelled)),
                Some(Stop::Failed(e)) => return Err(e),
                None => {}
            }
        }
        // 3. The report (also of a stopped check: what its finished folds found).
        Self::step(report, REPORTING, steps, steps, 0, "");
        acc.close_failures();
        let Acc {
            mut rep,
            tally,
            sample,
            ..
        } = acc;
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
        let at_spots = if rep.adds_at_spots > 0 {
            format!(" ({} at static spots)", n(rep.adds_at_spots))
        } else {
            String::new()
        };
        summary.push_str(&format!(
            "{} frames checked by {} fold models; suggested: {} missing boxes{at_spots}, {} labels \
             to verify, {} other classes ({} of these suggestions new since the last check, on {} \
             frames written); {} loose labels counted. No box was changed: confirm the \
             suggestions on the Review tab",
            n(rep.frames_checked),
            rep.folds_done,
            n(f.add),
            n(f.verify),
            n(f.relabel),
            n(w.total()),
            n(rep.frames_written),
            n(rep.loose),
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
            "dataset check ({} folds x {} {}, {})",
            self.request.folds,
            self.request.epochs,
            if self.request.epochs == 1 {
                "epoch"
            } else {
                "epochs"
            },
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
