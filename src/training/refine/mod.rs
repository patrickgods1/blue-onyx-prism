//! The dataset check (`blue-onyx-prism train refine`): a cross-check of the reviewed frames of the
//! folder under review by short throwaway fold models. The camera days of the reviewed frames are
//! split into K folds ([`folds::assign`]); for each fold a model trains on the other folds' frames
//! (the existing training run, [`crate::training::job::run_training`], in the job's work folder),
//! runs on the fold's frames it never saw ([`predict`]), and [`judge`] compares what it found with
//! the labels: missing boxes (`add`), labels no model found (`verify`), another class (`relabel`),
//! loose labels (counted); an `add` at a static spot of the selection (the same object labelled at
//! the place on other days, [`crate::collect::labels::statics`], [`Selected::spots`]) is made from
//! a lower score. The suggestions go on the label files for a reviewer to confirm
//! ([`crate::collect::labels::refine`]); the recall per camera, scene, class and size goes to
//! `<review folder>/refine/report.json` ([`report`]).
//!
//! **It changes nothing else**: no box, frame status, config entry, models folder file,
//! `runs.json` or `benchmark.json`. A fold model is installed in the job's work folder, registered
//! nowhere and benchmarked never ([`crate::training::job::NoHooks`]), and the whole work folder
//! (`<training dir>/refine/<id>`) is deleted at the end.
//!
//! Each stage is a file:
//!
//! - [`folds`]: whole camera days per fold, pure;
//! - [`job`]: the stage machine ([`RefineJob`], a task of the training queue);
//! - [`predict`]: a fold model over its frames, in this process ([`predict::FoldPredictor`]);
//! - [`judge`]: one frame's labels against the detections, pure (and [`judge::tighten`], the
//!   decision on a loose label from two looks);
//! - [`write`]: the suggestions, re-judged under the label write lock;
//! - [`tighten`]: the zoom stage after the last fold, a closer look at the loose labels by RF-DETR
//!   L on a crop of each (the overlap check's script), proposing better-fitting boxes;
//! - [`report`]: the report and its Wilson intervals.
//!
//! A route plans a check like the overlap check: [`plan_refine`] then the queue's `enqueue_task`;
//! [`preview`] answers the page's (and `--dry-run`'s) question of what a check would do, without a
//! GPU, a download or a write.
//!
//! **A quick check** ([`quick`], [`RefineRequest::model`]) judges the reviewed frames a model the
//! user trained here never trained or validated on, with that model instead of fold models: no
//! training, the same judge and writer, the same promise of changing nothing but suggestions and
//! the report.

pub mod folds;
pub mod job;
pub mod judge;
pub mod predict;
pub mod quick;
pub mod report;
pub mod tighten;
pub mod write;

pub use job::RefineJob;
pub use report::Mode;

use super::capability::Capability;
use super::dataset::SceneFilter;
use super::job::TrainRequest;
use super::progress::EpochPoint;
use super::recipe::TrainRecipe;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::export::{self, ClassFilter, ExportFrame, Selection, SplitConfig};
use crate::collect::labels::statics;
use crate::config::Config;
use crate::resources::catalog::{Platform, TrainFamily};
use crate::resources::export::train::{self as env, TrainPaths};
use anyhow::{Result, bail};
use report::{LeftOut, RecipeInfo};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Default number of folds: each model trains on three quarters of the camera days.
pub const DEFAULT_FOLDS: usize = 4;
/// Folds allowed (more train longer for a small gain in training data).
pub const FOLDS: std::ops::RangeInclusive<usize> = 2..=8;
/// Default epochs of a fold model: enough for a short fine-tune to find the common objects.
pub const DEFAULT_EPOCHS: u32 = 8;
/// Epochs allowed per fold model.
pub const EPOCHS: std::ops::RangeInclusive<u32> = 1..=50;
/// Fewest selected frames a check runs on: each fold model trains on the others, and fewer frames
/// teach it too little for its misses to say anything about the labels.
pub const MIN_FRAMES: usize = 40;
/// Seed of the fold assignment: fixed, so a rerun over the same frames uses the same folds.
pub const FOLD_SEED: u64 = 0;
/// The default recipe when no training run finished: RF-DETR nano, the model that fits a UHD 630.
pub const DEFAULT_FAMILY: TrainFamily = TrainFamily::Rfdetr;
pub const DEFAULT_SIZE: &str = "nano";
/// Cost of a fold beyond its training epochs, seconds: the environment self-check, the fold's
/// dataset export, the wait for memory and the ONNX export with its check, loading the fold model,
/// its detections on the held-out frames and the writes.
///
/// Measured (November 2026, GTX 1070 Ti, RF-DETR nano at 384 px, batch 8, 8 epochs, 851 frames
/// in 4 folds: 638 training and about 213 held-out frames per fold; two checks of 4,204 s and
/// 4,096 s): a fold took about 1,010 s, of which the 8 epochs (with their validation) took about
/// 576 s (72 s each) and everything else about 430 s. The detections are in this one figure: the
/// measurement does not separate them from the other steps, so no per-frame cost is claimed.
pub const FOLD_OVERHEAD_S: f64 = 430.0;

/// Which reviewed frames a check looks at (`--frames "scene=ir,camera=patio"`): every reviewed
/// frame of the folder under review by default. A scene or camera narrows the whole check: the
/// fold models train on those frames only.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefineFrames {
    /// `day`, `night`, `ir`; empty = every scene.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scenes: Vec<String>,
    /// Camera folder names; empty = every camera.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<String>,
}

impl RefineFrames {
    /// `scene=ir,camera=patio,camera=front` (a key may repeat; empty = every frame).
    pub fn parse(filter: &str) -> Result<Self> {
        let mut out = Self::default();
        for part in filter.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let Some((k, v)) = part.split_once('=') else {
                bail!(
                    "'{part}' in --frames is not key=value; use scene=<day|night|ir> or \
                     camera=<name>, comma-separated"
                );
            };
            let v = v.trim();
            match k.trim() {
                "scene" => {
                    if !crate::collect::layout::SCENES.contains(&v) {
                        bail!("scene '{v}' in --frames is not day, night or ir");
                    }
                    out.scenes.push(v.to_string());
                }
                "camera" if !v.is_empty() => out.cameras.push(v.to_string()),
                "camera" => bail!("camera= in --frames needs a camera folder name"),
                other => bail!(
                    "'{other}' in --frames is not a filter of the dataset check; use scene= or \
                     camera= (every reviewed frame by default)"
                ),
            }
        }
        out.scenes.sort();
        out.scenes.dedup();
        out.cameras.sort();
        out.cameras.dedup();
        Ok(out)
    }

    /// Whether `f` is in the filter.
    pub fn keeps(&self, f: &ExportFrame) -> bool {
        (self.scenes.is_empty() || self.scenes.contains(&f.rel.scene))
            && (self.cameras.is_empty() || self.cameras.contains(&f.rel.camera))
    }

    /// `all reviewed frames`, `scene ir, camera patio`.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.scenes.is_empty() {
            parts.push(format!("scene {}", self.scenes.join("+")));
        }
        if !self.cameras.is_empty() {
            parts.push(format!("camera {}", self.cameras.join("+")));
        }
        if parts.is_empty() {
            "all reviewed frames".into()
        } else {
            format!("reviewed frames of {}", parts.join(", "))
        }
    }
}

/// What to check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefineRequest {
    /// Folds ([`FOLDS`]); lowered to the number of camera days when there are fewer.
    #[serde(default = "default_folds")]
    pub folds: usize,
    /// Epochs of each fold model ([`EPOCHS`]); every epoch runs (no early stopping).
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default)]
    pub frames: RefineFrames,
    /// GPU (`cuda:N` of [`Capability::cuda_options`]; None = the automatic pick).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// Downloads over the large-download limit are allowed (the training environment, the base
    /// weights).
    #[serde(default)]
    pub allow_large: bool,
    /// A closer look at the loose labels after the last fold ([`tighten`]), proposing
    /// better-fitting boxes. None: on when RF-DETR L's weights are downloaded
    /// ([`tighten_weights_ready`]), else off with a warning; `Some(true)` asks for it (the weights
    /// downloaded with the check like the overlap check's), `Some(false)` never. A planned job
    /// holds the answer ([`plan_refine`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tighten: Option<bool>,
    /// A quick check with this configured model ([`quick`]): it judges the reviewed frames it never
    /// trained or validated on, and nothing trains (`folds` and `epochs` are not used). None: the
    /// fold check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Longest model name a request may carry (a config name; more is no name of the config).
pub const MAX_MODEL_NAME: usize = 255;

fn default_folds() -> usize {
    DEFAULT_FOLDS
}

fn default_epochs() -> u32 {
    DEFAULT_EPOCHS
}

impl Default for RefineRequest {
    fn default() -> Self {
        Self {
            folds: DEFAULT_FOLDS,
            epochs: DEFAULT_EPOCHS,
            frames: RefineFrames::default(),
            gpu: None,
            allow_large: false,
            tighten: None,
            model: None,
        }
    }
}

/// Whether the closer look's weights (RF-DETR L, the overlap check's model) are on disk: then a
/// check looks closer at its loose labels unless told not to.
pub fn tighten_weights_ready(config: &Config) -> bool {
    let paths = TrainPaths::for_config(config);
    crate::training::overlaps::rfdetr_weights(&paths)
        .is_ok_and(|w| super::pyenv::missing_bytes(&[w]) == 0)
}

/// Whether a check of `request` on `config` looks closer at its loose labels: the request's
/// answer, else [`tighten_weights_ready`].
pub fn tighten_on(config: &Config, request: &RefineRequest) -> bool {
    request
        .tighten
        .unwrap_or_else(|| tighten_weights_ready(config))
}

impl RefineRequest {
    /// The bounds of every field.
    pub fn validate(&self) -> Result<()> {
        if !FOLDS.contains(&self.folds) {
            bail!(
                "folds must be between {} and {}, got {}",
                FOLDS.start(),
                FOLDS.end(),
                self.folds
            );
        }
        if !EPOCHS.contains(&self.epochs) {
            bail!(
                "epochs must be between {} and {}, got {}",
                EPOCHS.start(),
                EPOCHS.end(),
                self.epochs
            );
        }
        if let Some(m) = &self.model {
            let m = m.trim();
            if m.is_empty() {
                bail!("the quick check needs the name of a configured model");
            }
            if m.len() > MAX_MODEL_NAME || m.chars().any(char::is_control) {
                bail!(
                    "a model name is up to {MAX_MODEL_NAME} characters, without control characters"
                );
            }
        }
        Ok(())
    }

    /// The quick check's model, trimmed; None for the fold check.
    pub fn quick_model(&self) -> Option<&str> {
        self.model.as_deref().map(str::trim)
    }
}

/// The recipe of the newest finished (Done) training run in the `runs.json` text `text`, by its
/// `created` time; None when there is none or the file does not parse. Read only: `runs.json` is
/// never written, moved or repaired here.
pub fn newest_done_recipe(text: &str) -> Option<TrainRecipe> {
    newest_done_run(text).map(|(r, _)| r)
}

/// [`newest_done_recipe`] with the run's whole record (an entry of `runs.json`, as JSON).
fn newest_done_run(text: &str) -> Option<(TrainRecipe, serde_json::Value)> {
    let list: Vec<serde_json::Value> = serde_json::from_str(text).ok()?;
    list.into_iter()
        .filter(|r| r.get("state").and_then(serde_json::Value::as_str) == Some("done"))
        .filter_map(|r| {
            let created = r.get("created")?.as_str()?.to_string();
            let recipe: TrainRecipe =
                serde_json::from_value(r.get("request")?.get("recipe")?.clone()).ok()?;
            Some((created, recipe, r))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, recipe, r)| (recipe, r))
}

/// The fold models' recipe ([`choose_recipe`]) and, when it is the newest finished run's, that
/// run's record (for its measured pace, [`pace_of`]).
struct Chosen {
    recipe: TrainRecipe,
    from: String,
    run: Option<serde_json::Value>,
}

/// The fold models' recipe and where it came from: the newest finished training run's model, input
/// size, batch and multi-scale choice (a batch it finished with fits the GPU), else RF-DETR nano at
/// the GPU's suggested batch; `epochs` epochs with early stopping off (patience 0: every epoch
/// runs, the best raw epoch is exported), the default learning rate and no augmentation.
pub fn choose_recipe(
    config: &Config,
    epochs: u32,
    cap: &Capability,
) -> Result<(TrainRecipe, String)> {
    let c = choose(config, epochs, cap)?;
    Ok((c.recipe, c.from))
}

/// [`choose_recipe`], keeping the run record the recipe came from.
fn choose(config: &Config, epochs: u32, cap: &Capability) -> Result<Chosen> {
    let usable =
        |r: &TrainRecipe| cap.family(r.family).is_none_or(|f| f.available) && r.validate().is_ok();
    let previous = std::fs::read_to_string(super::runs_file(config))
        .ok()
        .and_then(|t| newest_done_run(&t))
        .filter(|(p, _)| usable(p))
        .and_then(|(p, run)| {
            let mut r = TrainRecipe::new(p.family, &p.size).ok()?;
            r.imgsz = p.imgsz;
            r.batch = p.batch;
            r.multi_scale = p.multi_scale;
            r.base_weights = p.base_weights;
            Some((r, run))
        });
    let run = previous.as_ref().map(|(_, run)| run.clone());
    let (mut r, from) = match previous.map(|(r, _)| r) {
        Some(r) => {
            let from = format!("the newest finished training run ({})", r.label());
            (r, from)
        }
        None => {
            let mut r = TrainRecipe::new(DEFAULT_FAMILY, DEFAULT_SIZE)?;
            r.batch = cap
                .family(DEFAULT_FAMILY)
                .and_then(|f| f.sizes.iter().find(|s| s.size == DEFAULT_SIZE))
                .and_then(|s| s.suggested_batch)
                .unwrap_or(r.batch)
                .clamp(1, 16);
            (
                r,
                "the default, RF-DETR nano (no training run has finished)".to_string(),
            )
        }
    };
    r.epochs = epochs;
    r.patience = 0;
    r.patience_recommended = false;
    Ok(Chosen {
        recipe: r,
        from,
        run,
    })
}

/// What a finished training run measured on its GPU ([`pace_of`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pace {
    /// Seconds of one epoch (its training and its validation) per image the epoch went through
    /// (training and validation images).
    pub s_per_image: f64,
}

impl Pace {
    /// Seconds of `epochs` epochs over `frames` images (a fold model's training frames, which its
    /// run splits into training and validation).
    pub fn train_s(self, frames: usize, epochs: u32) -> f64 {
        self.s_per_image * frames as f64 * f64::from(epochs)
    }
}

/// The GPU the run record `run` trained on: `progress.settings.gpu`, else the name that starts its
/// `device` (`NVIDIA GeForce GTX 1070 Ti (compute capability 6.1, ...)`).
fn run_gpu(run: &serde_json::Value) -> Option<String> {
    let settings = run
        .pointer("/progress/settings/gpu")
        .and_then(serde_json::Value::as_str);
    let device = || {
        let d = run.get("device")?.as_str()?;
        Some(d.split_once(" (").map_or(d, |(name, _)| name))
    };
    settings
        .or_else(device)
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(str::to_string)
}

/// The median of `v` (sorted here); None when empty.
fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => v.get(n / 2).copied(),
        _ => Some((v.get(n / 2 - 1)? + v.get(n / 2)?) / 2.0),
    }
}

/// Seconds per epoch of a run's `history` (its `history.jsonl` rows): the median time between
/// two consecutive epochs, so a pause or a slow first epoch does not count (pauses are not
/// recorded, and the clock runs through them); with fewer than two such pairs, the last row's
/// time over its epoch. None for no row with a time.
fn epoch_s_of(history: &[EpochPoint]) -> Option<f64> {
    let steps: Vec<f64> = history
        .windows(2)
        .filter_map(|w| match w {
            [a, b] if b.epoch == a.epoch + 1 && b.elapsed_s > a.elapsed_s && a.elapsed_s > 0.0 => {
                Some(b.elapsed_s - a.elapsed_s)
            }
            _ => None,
        })
        .collect();
    if steps.len() >= 2 {
        return median(steps);
    }
    let last = history
        .iter()
        .rfind(|p| p.epoch > 0 && p.elapsed_s.is_finite() && p.elapsed_s > 0.0)?;
    Some(last.elapsed_s / f64::from(last.epoch))
}

/// The pace the finished run `run` (an entry of `runs.json`) measured, when it trained on the GPU
/// named `gpu` (another GPU, or one it does not name: None). One epoch's time comes from its
/// `history` ([`epoch_s_of`]), else from the record (`progress.elapsed_s` over `progress.epoch`);
/// the images of an epoch are `dataset.train_images` (repeats from balancing included) plus
/// `dataset.valid_images`. None when anything is missing or not positive.
pub fn pace_of(run: &serde_json::Value, history: &[EpochPoint], gpu: &str) -> Option<Pace> {
    if run_gpu(run).as_deref() != Some(gpu.trim()) {
        return None;
    }
    let epoch_s = epoch_s_of(history).or_else(|| {
        let elapsed = run.pointer("/progress/elapsed_s")?.as_f64()?;
        let epoch = run.pointer("/progress/epoch")?.as_u64()?;
        (epoch > 0).then(|| elapsed / epoch as f64)
    })?;
    let images = run.pointer("/dataset/train_images")?.as_u64()?
        + run
            .pointer("/dataset/valid_images")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
    let s_per_image = epoch_s / images as f64;
    (images > 0 && s_per_image.is_finite() && s_per_image > 0.0).then_some(Pace { s_per_image })
}

/// [`pace_of`] of the run record `run`, its history read from `<run_dir>/history.jsonl` (read only).
fn measured_pace(run: &serde_json::Value, gpu: &str) -> Option<Pace> {
    let history = run
        .get("run_dir")
        .and_then(serde_json::Value::as_str)
        .map(|d| {
            super::service::history::read(&super::service::history::history_path(Path::new(d)))
        })
        .unwrap_or_default();
    pace_of(run, &history, gpu)
}

/// Where a check's time estimate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeBasis {
    /// The pace the newest finished training run (whose recipe the folds use) measured on this
    /// GPU ([`pace_of`]).
    LastRun,
    /// The generic training estimate ([`super::estimate::estimate`]), or a quick check's guess
    /// ([`quick::QUICK_S_PER_FRAME`]): a rough guess.
    Guess,
    /// A quick check: what the last quick check with the same model measured per frame
    /// ([`report::RefineReport::predict_s_per_frame`]).
    LastCheck,
}

impl TimeBasis {
    /// `from your last training run`, `a rough guess`, `from the last quick check with this model`.
    pub const fn describe(self) -> &'static str {
        match self {
            TimeBasis::LastRun => "from your last training run",
            TimeBasis::Guess => "a rough guess",
            TimeBasis::LastCheck => "from the last quick check with this model",
        }
    }
}

/// Seconds of a check whose folds train on `train_frames` frames each, `epochs` epochs: per fold
/// its training (`pace`'s when there is one, else `guess` of the frame count, 0 when that has
/// none either) plus [`FOLD_OVERHEAD_S`].
pub fn check_seconds(
    train_frames: &[usize],
    epochs: u32,
    pace: Option<Pace>,
    guess: impl Fn(usize) -> Option<f64>,
) -> f64 {
    train_frames
        .iter()
        .map(|&n| {
            let train = match pace {
                Some(p) => p.train_s(n, epochs),
                None => guess(n).unwrap_or(0.0),
            };
            train + FOLD_OVERHEAD_S
        })
        .sum()
}

/// [`RecipeInfo`] of `r`.
pub fn recipe_info(r: &TrainRecipe, from: &str) -> RecipeInfo {
    RecipeInfo {
        model: r.label(),
        imgsz: r.imgsz,
        epochs: r.epochs,
        batch: r.batch,
        from: from.to_string(),
    }
}

/// What a check selects: one label snapshot for every fold.
#[derive(Debug, Default)]
pub struct Selected {
    /// The frames to check, in the export's order.
    pub frames: Vec<ExportFrame>,
    pub left_out: LeftOut,
    /// Frames per (camera, day).
    pub groups: BTreeMap<folds::Group, usize>,
    /// The fold (0-based) of each group; empty when refused.
    pub folds: BTreeMap<folds::Group, usize>,
    /// Folds after lowering to the groups.
    pub k: usize,
    /// Training boxes per wanted class (the list's spelling), ignore regions left out.
    pub class_boxes: BTreeMap<String, usize>,
    /// What the selection could not look at, beyond the per-frame counts of `left_out`.
    pub warnings: Vec<String>,
    /// Why a check would not run on this; None = it can.
    pub refused: Option<String>,
    /// Open suggestions of the folds' kinds a check of these frames replaces, the open `tighten`
    /// a check with the closer look may replace, and the frames holding them ([`open_to_replace`]).
    pub open: OpenCounts,
    /// The static spots of `frames` ([`spot_table`]): made once from this label snapshot, every
    /// fold's judge uses it ([`judge::Context::spots`]).
    pub spots: statics::Table,
}

/// The static spots of a check's `frames`, per camera ([`statics::observations`]: the training
/// boxes of wanted classes that are no ignore region, normalised; [`statics::Table::build`]).
pub fn spot_table(frames: &[ExportFrame], wanted: &Wanted<'_>) -> statics::Table {
    statics::Table::build(statics::observations(frames, wanted))
}

/// The open suggestions on a check's frames that it replaces ([`open_to_replace`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenCounts {
    /// Of the kinds the folds write ([`write::KINDS`]): every one is replaced.
    pub folds: usize,
    /// Frames holding them.
    pub fold_frames: usize,
    /// Open `tighten`: a check with the closer look replaces those of the labels it answers, at
    /// most all of them (which, only the look knows).
    pub tightens: usize,
    /// Frames holding either.
    pub any_frames: usize,
}

impl OpenCounts {
    /// (open suggestions of the folds' kinds, open `tighten` the check may replace, frames holding
    /// them) of a check with the closer look (`look`) or without (no `tighten` is touched).
    pub fn for_check(&self, look: bool) -> (usize, usize, usize) {
        if look {
            (self.folds, self.tightens, self.any_frames)
        } else {
            (self.folds, 0, self.fold_frames)
        }
    }
}

/// The open suggestions a check of `frames` replaces, and the frames holding them: unanswered
/// ones that a reviewer would see now ([`refine::live`] with `wanted`, as the check and the Review
/// tab judge them) of the kinds the folds write ([`write::KINDS`]), and the open `tighten` a check
/// with the closer look may replace. Stale ones are replaced too but nobody sees them, so they are
/// not counted.
pub fn open_to_replace(frames: &[ExportFrame], wanted: &Wanted<'_>) -> OpenCounts {
    let mut out = OpenCounts::default();
    for f in frames {
        let live = crate::collect::labels::refine::live(&f.labels, Some(wanted));
        let folds = live
            .iter()
            .filter(|l| write::KINDS.contains(&l.suggestion.kind))
            .count();
        let tightens = live
            .iter()
            .filter(|l| write::TIGHTEN_KINDS.contains(&l.suggestion.kind))
            .count();
        out.folds += folds;
        out.fold_frames += usize::from(folds > 0);
        out.tightens += tightens;
        out.any_frames += usize::from(folds + tightens > 0);
    }
    out
}

/// `n` `one`, or `n` `many`.
fn counted(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The warning of a check that replaces `open` open suggestions of the folds' kinds and up to
/// `tightens` open proposed boxes of the closer look on `frames` frames ([`open_to_replace`],
/// [`OpenCounts::for_check`]): the preview, the CLI and the start's confirmation say it.
pub fn replace_warning(open: usize, tightens: usize, frames: usize) -> String {
    replace_warning_by(open, tightens, frames, "what its fold models find")
}

/// [`replace_warning`] of a check whose suggestions come from `finder` (`what its fold models
/// find`, `what <model> finds`).
pub fn replace_warning_by(open: usize, tightens: usize, frames: usize, finder: &str) -> String {
    let boxes = format!(
        "up to {} for loose labels",
        counted(tightens, "open proposed box", "open proposed boxes")
    );
    let what = match (open, tightens) {
        (_, 0) => format!(
            "the {} on {} with {finder}",
            counted(open, "open suggestion", "open suggestions"),
            counted(frames, "frame", "frames"),
        ),
        (0, _) => format!(
            "{boxes} on {} with what its closer look finds",
            counted(frames, "frame", "frames"),
        ),
        _ => format!(
            "the {} on {} with {finder}, and {boxes} with what its closer look finds",
            counted(open, "open suggestion", "open suggestions"),
            counted(frames, "frame", "frames"),
        ),
    };
    format!(
        "this check replaces {what}; answered ones stay. Answer them on the Review tab first, or \
         run it to recompute them"
    )
}

/// Whether fold models of `family` trained `epochs` epochs are too short to judge labels: under
/// half of the family's [`super::recipe::Recommended::min_epochs`] (the epochs before which no
/// training run may stop: still settling). With every family's 10, 1 to 4 epochs. Measured: on the
/// same 851 frames, 4 folds of 8 epochs found 123 missing boxes and 2 folds of 1 epoch none.
pub fn weak_epochs(family: TrainFamily, epochs: u32) -> bool {
    epochs.saturating_mul(2) < super::recipe::recommended(family).min_epochs
}

/// The warning of [`weak_epochs`]; `open` = the open suggestions the check replaces.
pub fn weak_epochs_warning(family: TrainFamily, epochs: u32, open: usize) -> String {
    let min = super::recipe::recommended(family).min_epochs;
    format!(
        "{} per fold model is under half the {min} epochs {} training takes before it may stop: \
         fold models this short find little (on one dataset, fold models of 1 epoch found no \
         missing box where fold models of 8 epochs found 123), so the check says little about \
         the labels{}",
        counted(epochs as usize, "epoch", "epochs"),
        family.title(),
        if open > 0 {
            ", and the open suggestions it replaces likely came from stronger fold models"
        } else {
            ""
        }
    )
}

/// Most selection warnings kept in a report.
const MAX_WARNINGS: usize = 20;

/// Sort the export selection's warnings into `left_out` (the per-frame ones, by the wording of
/// [`export::reviewed_frames_filtered`]) and the rest.
fn sort_warnings(sel: &Selection, left_out: &mut LeftOut) -> Vec<String> {
    let mut rest = Vec::new();
    for w in &sel.warnings {
        if w.ends_with("reviewed but has pending boxes; left out") {
            left_out.pending += 1;
        } else if w.contains("are not in the class list; left out") {
            left_out.off_class_list += 1;
        } else if w.ends_with("; left out") {
            left_out.unreadable += 1;
            if rest.len() < MAX_WARNINGS {
                rest.push(w.clone());
            }
        } else if rest.len() < MAX_WARNINGS {
            rest.push(w.clone());
        }
    }
    left_out.lost = sel.lost_frames;
    left_out.unreadable_folders = sel.unreadable_folders;
    rest
}

/// Training boxes per wanted class (the list's spelling) of `frames`, ignore regions left out: with
/// the frames a fold model trains on, what [`judge::Context::class_boxes`] holds for it. The boxes
/// under the size floor count: the fold exports apply no floor, so the fold models train on them.
pub fn class_boxes<'a>(
    frames: impl IntoIterator<Item = &'a ExportFrame>,
    wanted: &Wanted<'_>,
) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for f in frames {
        for b in f.labels.training().filter(|b| !b.is_ignore()) {
            if let Some(c) = wanted.class_of(&b.label) {
                *out.entry(c.to_string()).or_default() += 1;
            }
        }
    }
    out
}

impl Selected {
    /// The frames of fold `i` (0-based): (held out, trained on), in the selection's order.
    pub fn fold_frames(&self, i: usize) -> (Vec<&ExportFrame>, Vec<&ExportFrame>) {
        self.frames.iter().partition(|f| {
            self.folds
                .get(&(f.rel.camera.clone(), f.rel.date.clone()))
                .is_some_and(|&k| k == i)
        })
    }

    /// What the selection's folds warn about (the preview and the report say it before any fold
    /// runs):
    ///
    /// - a fold whose model trains on clearly fewer frames than the fold holds out (under 90% of
    ///   them; two even folds differ by a frame and are no warning): a camera day that large
    ///   dominates the check, its fold model learns from less than it is asked about, so its
    ///   misses say more about the model than about the labels;
    /// - a class with [`judge::MIN_CLASS_BOXES`] training boxes or more in the selection but fewer
    ///   in what a fold model trains on: that fold's labels of it get no `verify`.
    pub fn fold_warnings(&self, classes: &[String], hidden: &[String]) -> Vec<String> {
        let wanted = Wanted { classes, hidden };
        let mut out = Vec::new();
        for i in 0..self.k {
            let n = i + 1;
            let (held, train) = self.fold_frames(i);
            if train.len() * 10 < held.len() * 9 {
                out.push(format!(
                    "fold {n} of {k} holds out {h} frames and its model trains on only {t}: a \
                     camera day this large dominates the check, and what its model misses says \
                     more about the model than about the labels; review frames of other camera \
                     days to balance it",
                    k = self.k,
                    h = held.len(),
                    t = train.len(),
                ));
            }
            let fold = class_boxes(train, &wanted);
            let rare: Vec<String> = self
                .class_boxes
                .iter()
                .filter(|(_, all)| **all >= judge::MIN_CLASS_BOXES)
                .filter_map(|(c, _)| {
                    let k = fold.get(c).copied().unwrap_or(0);
                    (k < judge::MIN_CLASS_BOXES).then(|| format!("{c} ({k})"))
                })
                .collect();
            if !rare.is_empty() {
                out.push(format!(
                    "fold {n} of {k} trains on fewer than {min} boxes of {list}: its frames' \
                     labels of these classes get no verify suggestion",
                    k = self.k,
                    min = judge::MIN_CLASS_BOXES,
                    list = rare.join(", "),
                ));
            }
        }
        out
    }
}

/// Select the frames of a check of `dataset_dir` (read only) and fold them. No size floor goes on
/// the frames ([`ExportFrame::floor`]): the fold exports apply none, so the fold models train on
/// every label; only the judge treats boxes under the floor as ignore regions.
pub fn select(
    dataset_dir: &Path,
    classes: &[String],
    hidden: &[String],
    filter: &RefineFrames,
    k: usize,
    min_frames: usize,
) -> Selected {
    let sel = export::reviewed_frames_filtered(
        dataset_dir,
        classes,
        &SplitConfig::default(),
        &ClassFilter::default(),
    );
    let mut out = Selected::default();
    out.warnings = sort_warnings(&sel, &mut out.left_out);
    let (frames, filtered): (Vec<ExportFrame>, Vec<ExportFrame>) =
        sel.frames.into_iter().partition(|f| filter.keeps(f));
    out.left_out.filtered = filtered.len();
    out.frames = frames;
    let wanted = Wanted { classes, hidden };
    for f in &out.frames {
        *out.groups
            .entry((f.rel.camera.clone(), f.rel.date.clone()))
            .or_default() += 1;
    }
    out.class_boxes = class_boxes(&out.frames, &wanted);
    out.open = open_to_replace(&out.frames, &wanted);
    out.spots = spot_table(&out.frames, &wanted);
    // Folded even when refused below, so a preview shows what a check would do.
    match folds::assign(&out.groups, k, FOLD_SEED) {
        Ok(f) => {
            out.k = folds::effective_folds(k, out.groups.len());
            out.folds = f;
        }
        Err(e) => out.refused = Some(format!("{e:#}")),
    }
    if out.frames.len() < min_frames {
        out.refused = Some(format!(
            "the dataset check needs at least {min_frames} reviewed frames that can be exported, \
             and {} has {} ({} left out: {} with pending boxes, {} with labels off the class list, \
             {} by the frame filter); review more frames on the Review tab first",
            dataset_dir.display(),
            out.frames.len(),
            out.left_out.frames(),
            out.left_out.pending,
            out.left_out.off_class_list,
            out.left_out.filtered,
        ));
    }
    out
}

/// One fold of a [`RefinePreview`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldPreview {
    /// 1-based.
    pub fold: usize,
    pub camera_days: usize,
    /// Frames checked in this fold (its model trains on the others).
    pub frames: usize,
}

/// What a check would do ([`preview`]; `train refine --dry-run`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinePreview {
    pub dataset_dir: PathBuf,
    /// [`RefineFrames::describe`].
    pub filter: String,
    /// Frames the check would judge.
    pub frames: usize,
    pub left_out: LeftOut,
    pub camera_days: usize,
    pub folds: Vec<FoldPreview>,
    pub recipe: RecipeInfo,
    /// Rough time of the whole check, seconds; None without a GPU to train on.
    pub seconds: Option<f64>,
    /// Where `seconds` came from; None with no `seconds`. With `tighten` the seconds include a
    /// rough figure for the closer look ([`tighten::estimate_seconds`]), whatever the basis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_basis: Option<TimeBasis>,
    /// The check would look closer at its loose labels ([`tighten_on`]).
    #[serde(default)]
    pub tighten: bool,
    pub warnings: Vec<String>,
    /// Open suggestions on the selected frames that the check replaces with what its fold models
    /// find ([`open_to_replace`]): a start asks first (`POST /v1/refine/check` `replace_open`).
    #[serde(default)]
    pub open_suggestions: usize,
    /// With the closer look (`tighten`), the open `tighten` on the selected frames it may replace
    /// (those of the labels it answers: at most this many); 0 without it, which touches none. A
    /// start asks first for these too.
    #[serde(default)]
    pub open_tightens: usize,
    /// Frames holding them (both kinds counted).
    #[serde(default)]
    pub open_frames: usize,
    /// Why the check would be refused; None = it can run.
    pub refused: Option<String>,
    /// The folder under review is not available (an unplugged drive): `refused` says so, and
    /// nothing was counted. Never an empty dataset.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub folder_missing: bool,
    /// Fold models, or a quick check with a model ([`quick`]): then `folds` is empty, `recipe` is
    /// the model's own, `frames` the reviewed frames it never saw and `open_*` count only those.
    #[serde(default)]
    pub mode: Mode,
    /// A quick check's model: its run, input size, the frames skipped as trained or validated on,
    /// the classes it lacks, the device it is configured for. Only `name` when its frames could
    /// not be determined (`refused` says why).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<report::ModelInfo>,
}

impl RefinePreview {
    /// [`replace_warning`] in this check's words (a quick check names its model).
    pub fn replace_text(&self) -> String {
        match (&self.mode, &self.model) {
            (Mode::Model, Some(m)) => replace_warning_by(
                self.open_suggestions,
                self.open_tightens,
                self.open_frames,
                &format!("what {} finds", m.name),
            ),
            _ => replace_warning(self.open_suggestions, self.open_tightens, self.open_frames),
        }
    }
}

/// What a check of `request` on `config` would do: no GPU, login or download needed, nothing
/// written. Errors only on a bad request.
pub fn preview(
    config: &Config,
    request: &RefineRequest,
    cap: &Capability,
) -> Result<RefinePreview> {
    request.validate()?;
    if let Some(name) = request.quick_model() {
        return quick::preview(config, request, cap, name);
    }
    let dir = config.collect.resolved_review_dir(&config.data_root());
    // An unavailable folder (an unplugged drive) is its own answer, never an empty dataset.
    let missing = config.collect.review_folder_problem(&dir);
    let mut s = match &missing {
        Some(_) => Selected::default(),
        None => select(
            &dir,
            &config.collect.classes,
            &config.collect.hidden_classes,
            &request.frames,
            request.folds,
            MIN_FRAMES,
        ),
    };
    if missing.is_some() {
        s.refused.clone_from(&missing);
    }
    let Chosen {
        recipe,
        from,
        run: recipe_run,
    } = choose(config, request.epochs, cap)?;
    let tighten = tighten_on(config, request);
    let (open_suggestions, open_tightens, open_frames) = s.open.for_check(tighten);
    // What a start asks about comes first.
    let mut warnings = Vec::new();
    if open_suggestions + open_tightens > 0 {
        warnings.push(replace_warning(
            open_suggestions,
            open_tightens,
            open_frames,
        ));
    }
    if weak_epochs(recipe.family, request.epochs) {
        warnings.push(weak_epochs_warning(
            recipe.family,
            request.epochs,
            open_suggestions,
        ));
    }
    warnings.extend(s.warnings.iter().cloned());
    warnings.extend(s.fold_warnings(&config.collect.classes, &config.collect.hidden_classes));
    let mut folds: Vec<FoldPreview> = (0..s.k)
        .map(|i| FoldPreview {
            fold: i + 1,
            camera_days: 0,
            frames: 0,
        })
        .collect();
    for (g, f) in &s.folds {
        if let Some(p) = folds.get_mut(*f) {
            p.camera_days += 1;
            p.frames += s.groups.get(g).copied().unwrap_or(0);
        }
    }
    if s.k > 0 && s.k < request.folds {
        warnings.push(format!(
            "{} folds asked for, {} camera days: {} folds",
            request.folds,
            s.groups.len(),
            s.k
        ));
    }
    let accelerator = cap.resolve_gpu(request.gpu.as_deref()).ok().flatten();
    let (seconds, time_basis) = match &accelerator {
        Some(a) => {
            let total = s.frames.len();
            // Each fold model trains on every frame but its fold's.
            let train: Vec<usize> = folds
                .iter()
                .map(|p| total.saturating_sub(p.frames))
                .collect();
            // The run the recipe came from, when it trained on this GPU, says how fast this
            // machine trains it; else the generic estimate.
            let pace = recipe_run
                .as_ref()
                .and_then(|run| measured_pace(run, a.name()));
            let guess = |n: usize| {
                super::estimate::estimate(
                    recipe.family,
                    &recipe.size,
                    recipe.imgsz,
                    recipe.batch,
                    n,
                    recipe.epochs,
                    a.target(),
                )
                .map(|e| e.total_s)
            };
            let mut secs = check_seconds(&train, recipe.epochs, pace, guess);
            if tighten {
                secs += tighten::estimate_seconds(total);
            }
            let basis = if pace.is_some() {
                TimeBasis::LastRun
            } else {
                TimeBasis::Guess
            };
            if folds.is_empty() {
                (None, None)
            } else {
                (Some(secs), Some(basis))
            }
        }
        None => {
            warnings.push(format!(
                "the check trains fold models in the training environment, which needs an \
                 NVIDIA GPU or Apple silicon: {}",
                cap.summary()
            ));
            (None, None)
        }
    };
    if let Some(a) = &accelerator {
        let paths = TrainPaths::for_config(config);
        if !env::train_env_ready(&paths, Platform::current(), a.lane())
            && !(request.allow_large || config.allow_large_downloads)
        {
            let (dl, _) = a.lane().estimate();
            warnings.push(format!(
                "the training environment is not set up yet (about {} download): allow large \
                 downloads to run the check",
                crate::resources::catalog::format_size(dl)
            ));
        }
    }
    let rare: Vec<String> = s
        .class_boxes
        .iter()
        .filter(|(_, n)| **n < judge::MIN_CLASS_BOXES)
        .map(|(c, n)| format!("{c} ({n})"))
        .collect();
    if !rare.is_empty() {
        warnings.push(format!(
            "fewer than {} training boxes: {}; their labels get no verify suggestion",
            judge::MIN_CLASS_BOXES,
            rare.join(", ")
        ));
    }
    // Off by default only for want of the weights: say so, and how to get them.
    if !tighten && request.tighten.is_none() {
        warnings.push(tighten::skipped_warning());
    }
    Ok(RefinePreview {
        dataset_dir: dir,
        filter: request.frames.describe(),
        frames: s.frames.len(),
        left_out: s.left_out,
        camera_days: s.groups.len(),
        folds,
        recipe: recipe_info(&recipe, &from),
        seconds,
        time_basis,
        tighten,
        warnings,
        open_suggestions,
        open_tightens,
        open_frames,
        refused: s.refused,
        folder_missing: missing.is_some(),
        mode: Mode::Folds,
        model: None,
    })
}

/// The warnings `train refine` prints before it starts (it is an explicit command: it asks
/// nothing): [`replace_warning`] when the check replaces open suggestions,
/// [`weak_epochs_warning`] when its fold models (of `family`; None for a quick check, which trains
/// none) are short, a quick check's frames of camera days its model trained on
/// ([`quick::same_day_warning`]), and [`tighten::skipped_warning`] when it runs without the closer
/// look for want of the weights.
pub fn start_warnings(p: &RefinePreview, family: Option<TrainFamily>) -> Vec<String> {
    let mut out = Vec::new();
    if p.open_suggestions + p.open_tightens > 0 {
        out.push(p.replace_text());
    }
    if let Some(family) = family
        && weak_epochs(family, p.recipe.epochs)
    {
        out.push(weak_epochs_warning(
            family,
            p.recipe.epochs,
            p.open_suggestions,
        ));
    }
    if let Some(m) = p.model.as_ref().filter(|m| m.same_day > 0) {
        out.push(quick::same_day_warning(m.same_day, p.frames, &m.name));
    }
    let skipped = tighten::skipped_warning();
    if p.warnings.contains(&skipped) {
        out.push(skipped);
    }
    out
}

/// The `--dry-run` text of `p`.
pub fn preview_text(p: &RefinePreview) -> String {
    let quick = p.mode == Mode::Model;
    let mut t = match (&p.model, quick) {
        (Some(m), true) => {
            let mut t = format!(
                "quick dataset check of {} in {} with {}: {} frames it never saw on {} camera \
                 days\n",
                p.filter,
                p.dataset_dir.display(),
                m.name,
                p.frames,
                p.camera_days
            );
            if !m.run.is_empty() {
                t.push_str(&format!(
                    "model: {} ({} at {} px, training run {}{}); skipped: {} frames it trained \
                     on, {} it validated on\n",
                    m.name,
                    m.recipe,
                    m.imgsz,
                    m.run,
                    if m.device.is_empty() {
                        String::new()
                    } else {
                        format!(", runs on {}", m.device)
                    },
                    m.trained_on,
                    m.validated_on
                ));
                t.push_str(if m.by_source {
                    "frames matched by name and by the run's record of its image sources\n"
                } else {
                    "frames matched by name (the run has no record of its image sources)\n"
                });
            }
            t
        }
        _ => format!(
            "dataset check of {} in {}: {} frames on {} camera days\n",
            p.filter,
            p.dataset_dir.display(),
            p.frames,
            p.camera_days
        ),
    };
    let l = &p.left_out;
    if l.frames() + l.unreadable_folders > 0 {
        t.push_str(&format!(
            "left out: {} with pending boxes, {} with labels off the class list, {} with a file \
             gone, {} unreadable, {} by the frame filter{}\n",
            l.pending,
            l.off_class_list,
            l.lost,
            l.unreadable,
            l.filtered,
            if l.unreadable_folders > 0 {
                format!("; {} folders could not be listed", l.unreadable_folders)
            } else {
                String::new()
            }
        ));
    }
    let r = &p.recipe;
    if !quick {
        t.push_str(&format!(
            "{} folds, each a {} model at {} px trained {} epochs (batch {}; {})\n",
            p.folds.len(),
            r.model,
            r.imgsz,
            r.epochs,
            r.batch,
            r.from
        ));
    }
    for f in &p.folds {
        t.push_str(&format!(
            "  fold {}: {} frames on {} camera days\n",
            f.fold, f.frames, f.camera_days
        ));
    }
    t.push_str(match (p.tighten, quick) {
        (true, false) => {
            "loose labels: a closer look by RF-DETR L after the last fold proposes better-fitting \
             boxes\n"
        }
        (true, true) => {
            "loose labels: a closer look by RF-DETR L after the model's pass proposes \
             better-fitting boxes\n"
        }
        _ => "loose labels: counted only (no closer look)\n",
    });
    if let Some(s) = p.seconds {
        t.push_str(&format!(
            "about {} {}({}){}\n",
            super::estimate::format_duration(s),
            if quick { "" } else { "on this GPU " },
            p.time_basis.unwrap_or(TimeBasis::Guess).describe(),
            if p.tighten {
                format!("; {}", tighten::time_line())
            } else {
                String::new()
            }
        ));
    }
    for w in &p.warnings {
        t.push_str(&format!("warning: {w}\n"));
    }
    match &p.refused {
        Some(r) => t.push_str(&format!("refused: {r}\n")),
        None => t.push_str("nothing was run or written (--dry-run)\n"),
    }
    t
}

/// The name of fold `k`'s model (1-based).
pub fn fold_model_name(k: usize) -> String {
    format!("refine-fold-{k}")
}

/// Plan a check of `request` on `config` and `cap`: fails fast on a bad request, no GPU to train
/// on, the training environment or base weights to download without `allow_large` (or
/// `allow_large_downloads`). The frames are read when it runs. Work folder:
/// `<training dir>/refine/<id>`.
///
/// With a `model` it plans a quick check instead ([`quick::plan`]): no GPU needed unless the
/// closer look runs, and the frames are counted now for the title.
pub fn plan_refine(config: &Config, request: RefineRequest, cap: &Capability) -> Result<RefineJob> {
    let mut request = request;
    request.validate()?;
    if let Some(name) = request.quick_model().map(str::to_string) {
        return quick::plan(config, request, cap, &name);
    }
    let paths = TrainPaths::for_config(config);
    let platform = Platform::current();
    let allow_large = request.allow_large || config.allow_large_downloads;
    let Some(accelerator) = cap.resolve_gpu(request.gpu.as_deref())? else {
        bail!(
            "the dataset check trains fold models in the training environment, which needs an \
             NVIDIA GPU or Apple silicon: {}",
            cap.summary()
        );
    };
    if !env::train_env_ready(&paths, platform, accelerator.lane()) && !allow_large {
        let (dl, _) = accelerator.lane().estimate();
        bail!(
            "the dataset check trains in the training environment (about {} download); rerun \
             with --allow-large (or enable allow_large_downloads)",
            crate::resources::catalog::format_size(dl)
        );
    }
    let (recipe, from) = choose_recipe(config, request.epochs, cap)?;
    let base = recipe.base()?;
    super::pyenv::check_large(&[(base, paths.weights_dir(base))], allow_large)?;
    // The closer look: on by default only when its weights are here; asked for, they come with
    // the check under the same gate as the overlap check's.
    let tighten = tighten_on(config, &request);
    if tighten {
        super::pyenv::check_large(
            &[crate::training::overlaps::rfdetr_weights(&paths)?],
            allow_large,
        )?;
    }
    request.tighten = Some(tighten);
    request.allow_large = allow_large;
    let id = super::service::new_run_id();
    let work = super::training_dir(config).join("refine").join(&id);
    let first = work.join("fold-1");
    let train_request = TrainRequest {
        recipe,
        datasets: vec![first.join("export")],
        include_public: false,
        scene: SceneFilter::ALL_SCENES,
        classes: None,
        rename: BTreeMap::new(),
        benchmark_sets: Vec::new(),
        benchmark_devices: Vec::new(),
        allow_large,
        install_dir: Some(first.join("model")),
        gpu: request.gpu.clone(),
        excluded: Default::default(),
        balance: Default::default(),
        compare_group: None,
        train_seed: None,
    };
    let mut train = super::plan_job(config, train_request, cap, Some(fold_model_name(1)))?;
    train.run_dir = first;
    train.id = format!("{id}-fold1");
    Ok(RefineJob {
        dataset_dir: config.collect.resolved_review_dir(&config.data_root()),
        classes: config.collect.classes.clone(),
        hidden: config.collect.hidden_classes.clone(),
        work,
        train: Some(train),
        quick: None,
        recipe_from: from,
        predictor: None,
        config: Box::new(config.clone()),
        min_frames: MIN_FRAMES,
        free_commit: crate::system_info::commit,
        id,
        request,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_request_parses_and_checks_its_bounds() {
        let r = RefineRequest::default();
        assert_eq!((r.folds, r.epochs), (4, 8));
        r.validate().unwrap();
        for bad in [
            RefineRequest {
                folds: 1,
                ..r.clone()
            },
            RefineRequest {
                folds: 9,
                ..r.clone()
            },
            RefineRequest {
                epochs: 0,
                ..r.clone()
            },
            RefineRequest {
                epochs: 51,
                ..r.clone()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        let back: RefineRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(back, r);
        let f = RefineFrames::parse(" scene=ir, camera=patio,camera=front,scene=ir ").unwrap();
        assert_eq!(f.scenes, ["ir"]);
        assert_eq!(f.cameras, ["front", "patio"]);
        assert_eq!(
            f.describe(),
            "reviewed frames of scene ir, camera front+patio"
        );
        assert_eq!(
            RefineFrames::parse("").unwrap().describe(),
            "all reviewed frames"
        );
        for bad in ["scene=dusk", "status=any", "camera=", "patio"] {
            assert!(RefineFrames::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_recipe_is_the_newest_finished_runs_or_rf_detr_nano() {
        let rec = |family, size: &str, imgsz: u32, batch: u32| {
            let mut r = TrainRecipe::new(family, size).unwrap();
            r.imgsz = imgsz;
            r.batch = batch;
            r.epochs = 60;
            r.patience = 12;
            r
        };
        let runs = serde_json::json!([
            {"id": "a", "created": "2026-10-01T00:00:00Z", "state": "done",
             "request": {"recipe": rec(TrainFamily::Yolo26, "n", 640, 8)}},
            {"id": "b", "created": "2026-10-05T00:00:00Z", "state": "done",
             "request": {"recipe": rec(TrainFamily::Rfdetr, "small", 448, 2)}},
            {"id": "c", "created": "2026-10-09T00:00:00Z", "state": "failed",
             "request": {"recipe": rec(TrainFamily::Yolo11, "s", 640, 8)}},
            {"id": "d", "created": "2026-10-08T00:00:00Z", "state": "done",
             "request": {"recipe": "broken"}}
        ]);
        let r = newest_done_recipe(&runs.to_string()).unwrap();
        assert_eq!(
            (r.family, r.size.as_str(), r.imgsz, r.batch),
            (TrainFamily::Rfdetr, "small", 448, 2)
        );
        assert!(newest_done_recipe("not json").is_none());
        assert!(newest_done_recipe("[]").is_none());
        // On a config: no runs.json -> RF-DETR nano; with one -> its model, input and batch, the
        // check's epochs and no early stopping.
        let root = std::env::temp_dir().join(format!("bop-refine-recipe-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let cap = super::super::capability::host_for_tests();
        let (r, from) = choose_recipe(&c, 8, &cap).unwrap();
        assert_eq!((r.family, r.size.as_str()), (TrainFamily::Rfdetr, "nano"));
        assert_eq!(
            (r.epochs, r.patience, r.patience_recommended),
            (8, 0, false)
        );
        assert!(!r.early_stopping() && r.anneal_epochs() == 0);
        // Patience 0: the convergence rule never stops the run, even on a small validation split.
        assert_eq!(r.convergence_patience(5).0, 0);
        assert!(from.contains("default"), "{from}");
        r.validate().unwrap();
        let runs_file = super::super::runs_file(&c);
        std::fs::create_dir_all(runs_file.parent().unwrap()).unwrap();
        std::fs::write(&runs_file, runs.to_string()).unwrap();
        let before = std::fs::read(&runs_file).unwrap();
        let (r, from) = choose_recipe(&c, 3, &cap).unwrap();
        assert_eq!(r.label(), "rfdetr-small");
        assert_eq!((r.imgsz, r.batch, r.epochs, r.patience), (448, 2, 3, 0));
        assert!(from.contains("rfdetr-small"), "{from}");
        assert_eq!(std::fs::read(&runs_file).unwrap(), before, "read only");
        let _ = std::fs::remove_dir_all(&root);
    }

    const GTX: &str = "NVIDIA GeForce GTX 1070 Ti";

    /// A `runs.json` entry shaped like the real ones (RF-DETR nano at 384 px, batch 8, balanced to
    /// 1,331 training images per epoch with 83 validation images, stopped after 51 epochs on a
    /// GTX 1070 Ti; the record holds no history, `history.jsonl` does).
    fn real_run(run_dir: &str) -> serde_json::Value {
        let mut recipe = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
        recipe.imgsz = 384;
        recipe.batch = 8;
        recipe.epochs = 100;
        recipe.patience = 10;
        serde_json::json!({
            "format": 1,
            "id": "20261111-020504-2fff",
            "name": "rfdetr-nano-s2",
            "created": "2026-11-11T02:05:06Z",
            "request": {"recipe": recipe},
            "device": format!("{GTX} (compute capability 6.1, 7 GB, PyTorch cu126)"),
            "state": "done",
            "progress": {
                "device": "cuda:0", "epoch": 51, "epochs": 100, "batch": 168, "batches": 168,
                "elapsed_s": 7707.1700769, "eta_s": null, "phase": "export", "history": [],
                "best": {"metric": "f2", "epoch": 34, "value": 0.67},
                "batch_size": 8, "imgsz": 384,
                "settings": {"lr_schedule": "cosine", "select_metric": "f2", "epochs": 100,
                             "patience": 10, "batch": 8, "workers": 0, "seed": 1,
                             "scene": "all", "gpu": GTX}
            },
            "dataset": {"classes": ["person", "car"], "train_images": 1331, "train_frames": 564,
                        "valid_images": 83, "heldout_images": 169},
            "finished": "2026-11-11T06:20:49Z",
            "run_dir": run_dir
        })
    }

    /// History rows of `epochs` epochs at `step` seconds each after a slower first one, with a
    /// pause of `pause` seconds before epoch 20 (the clock runs through pauses).
    fn rows(epochs: u32, step: f64, pause: f64) -> Vec<EpochPoint> {
        (1..=epochs)
            .map(|e| EpochPoint {
                epoch: e,
                elapsed_s: 155.5 + f64::from(e - 1) * step + if e >= 20 { pause } else { 0.0 },
                metrics: BTreeMap::new(),
            })
            .collect()
    }

    #[test]
    fn the_pace_is_the_last_runs_median_epoch_over_its_images() {
        let run = real_run("");
        let images = 1331.0 + 83.0;
        // From the history: the median epoch, untouched by an hour's pause or the first epoch.
        let p = pace_of(&run, &rows(51, 150.45, 3600.0), GTX).unwrap();
        assert!((p.s_per_image - 150.45 / images).abs() < 1e-9, "{p:?}");
        // Without a history: the record's elapsed time over its epochs (pause and all).
        let p = pace_of(&run, &[], GTX).unwrap();
        assert!(
            (p.s_per_image - 7_707.170_076_9 / 51.0 / images).abs() < 1e-9,
            "{p:?}"
        );
        // One history row: its time over its epoch.
        let one = [EpochPoint {
            epoch: 4,
            elapsed_s: 600.0,
            metrics: BTreeMap::new(),
        }];
        let p = pace_of(&run, &one, GTX).unwrap();
        assert!((p.s_per_image - 150.0 / images).abs() < 1e-9);
        // Another GPU, or none named: nothing measured here.
        assert!(pace_of(&run, &[], "NVIDIA GeForce RTX 3060").is_none());
        let mut nameless = run.clone();
        nameless["progress"]["settings"]["gpu"] = serde_json::Value::Null;
        nameless["device"] = serde_json::Value::Null;
        assert!(pace_of(&nameless, &[], GTX).is_none());
        // The device names the GPU when the settings do not (records before RunSettings).
        let mut old = run.clone();
        old["progress"].as_object_mut().unwrap().remove("settings");
        assert!(pace_of(&old, &[], GTX).is_some());
        // No epoch, no images: none.
        let mut fresh = run.clone();
        fresh["progress"]["epoch"] = 0.into();
        assert!(pace_of(&fresh, &[], GTX).is_none());
        let mut empty = run.clone();
        empty["dataset"]["train_images"] = 0.into();
        empty["dataset"]["valid_images"] = 0.into();
        assert!(pace_of(&empty, &rows(51, 150.45, 0.0), GTX).is_none());
        let mut no_dataset = run;
        no_dataset.as_object_mut().unwrap().remove("dataset");
        assert!(pace_of(&no_dataset, &[], GTX).is_none());
    }

    #[test]
    fn the_measured_check_takes_about_what_two_real_checks_took() {
        // 851 frames in 4 folds (213, 213, 213, 212 held out): each fold model trains on the rest,
        // 8 epochs, at the last run's pace (150.45 s per epoch of 1,414 images).
        let held = [213_usize, 213, 213, 212];
        let train: Vec<usize> = held.iter().map(|h| 851 - h).collect();
        assert_eq!(train.iter().sum::<usize>(), 3 * 851);
        let pace = pace_of(&real_run(""), &rows(51, 150.45, 0.0), GTX).unwrap();
        let none = |_: usize| -> Option<f64> { panic!("the guess is not used with a pace") };
        let s = check_seconds(&train, 8, Some(pace), none);
        let want = 150.45 / 1414.0 * 2553.0 * 8.0 + 4.0 * FOLD_OVERHEAD_S;
        assert!((s - want).abs() < 1e-6, "{s} vs {want}");
        // About 65 min; the two real checks took 4,204 s and 4,096 s (70 and 68 min).
        assert!((3850.0..3950.0).contains(&s), "{s}");
        assert_eq!(super::super::estimate::format_duration(s), "65 min");
        // The generic estimate when nothing was measured: its training time per fold plus the
        // fixed cost; a fold it cannot estimate counts the fixed cost only.
        let s = check_seconds(&train, 8, None, |n| Some(n as f64 * 0.5));
        assert!((s - (2553.0 * 0.5 + 4.0 * FOLD_OVERHEAD_S)).abs() < 1e-9);
        assert!((check_seconds(&train, 8, None, |_| None) - 4.0 * FOLD_OVERHEAD_S).abs() < 1e-9);
        assert!(check_seconds(&[], 8, Some(pace), |_| None).abs() < 1e-9);
    }

    #[test]
    fn the_preview_says_whether_its_time_was_measured() {
        let root = std::env::temp_dir().join(format!("bop-refine-pace-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let cap = super::super::capability::host_for_tests();
        let gpu = cap.resolve_gpu(None).unwrap().unwrap();
        let req = RefineRequest::default();
        // No runs.json: the generic estimate (no reviewed frame: no fold, no time either).
        let p = preview(&c, &req, &cap).unwrap();
        assert_eq!((p.seconds, p.time_basis), (None, None));
        // A finished run on this GPU with its history file: its recipe and its pace.
        let run_dir = root.join("training").join("20261111-020504-2fff");
        std::fs::create_dir_all(&run_dir).unwrap();
        let lines: String = rows(51, 150.45, 0.0)
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect();
        std::fs::write(
            super::super::service::history::history_path(&run_dir),
            lines,
        )
        .unwrap();
        let mut run = real_run(run_dir.to_str().unwrap());
        run["progress"]["settings"]["gpu"] = gpu.name().into();
        let runs_file = super::super::runs_file(&c);
        std::fs::create_dir_all(runs_file.parent().unwrap()).unwrap();
        std::fs::write(&runs_file, serde_json::json!([run]).to_string()).unwrap();
        let chosen = choose(&c, 8, &cap).unwrap();
        assert_eq!(chosen.recipe.label(), "rfdetr-nano");
        let pace = measured_pace(chosen.run.as_ref().unwrap(), gpu.name()).unwrap();
        assert!(
            (pace.s_per_image - 150.45 / 1414.0).abs() < 1e-9,
            "{pace:?}"
        );
        // The same record from another GPU: not this machine's pace.
        assert_ne!(gpu.name(), GTX);
        assert!(measured_pace(chosen.run.as_ref().unwrap(), GTX).is_none());
        let text = |basis| {
            preview_text(&RefinePreview {
                dataset_dir: root.clone(),
                filter: "all reviewed frames".into(),
                frames: 851,
                left_out: LeftOut::default(),
                camera_days: 12,
                folds: Vec::new(),
                recipe: recipe_info(&chosen.recipe, &chosen.from),
                seconds: Some(3893.0),
                time_basis: basis,
                tighten: false,
                warnings: Vec::new(),
                open_suggestions: 0,
                open_tightens: 0,
                open_frames: 0,
                refused: None,
                folder_missing: false,
                mode: Mode::Folds,
                model: None,
            })
        };
        assert!(
            text(Some(TimeBasis::LastRun))
                .contains("about 65 min on this GPU (from your last training run)")
        );
        assert!(text(Some(TimeBasis::Guess)).contains("(a rough guess)"));
        assert!(text(None).contains("(a rough guess)"));
        assert_eq!(
            serde_json::to_value(TimeBasis::LastRun).unwrap(),
            serde_json::json!("last_run")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    use crate::collect::labels::refine::{Answer, Kind, Suggestion};
    use crate::collect::labels::{BoxState, LabelBox, LabelFile, Source, Status};
    use crate::collect::layout::Rel;

    /// A frame of `camera` on `date` in memory, with `labels`.
    fn frame(camera: &str, date: &str, stem: &str, labels: LabelFile) -> ExportFrame {
        ExportFrame {
            rel: Rel {
                scene: "day".into(),
                camera: camera.into(),
                date: date.into(),
                stem: stem.into(),
            },
            name: format!("{camera}-{stem}.jpg"),
            image: PathBuf::from(format!("{stem}.jpg")),
            width: 320,
            height: 240,
            buckets: Vec::new(),
            split: export::Split::Train,
            labels,
            floor: None,
        }
    }

    /// `n` frames of a camera day of `camera` in fold `fold` of `s`.
    fn day_in_fold(s: &mut Selected, camera: &str, n: usize, fold: usize) {
        let group = (camera.to_string(), "2026-11-10".to_string());
        for i in 0..n {
            s.frames.push(frame(
                camera,
                "2026-11-10",
                &format!("f{i:04}"),
                LabelFile::default(),
            ));
        }
        s.groups.insert(group.clone(), n);
        s.folds.insert(group, fold);
    }

    fn held_warnings(s: &Selected) -> Vec<String> {
        s.fold_warnings(&[], &[])
            .into_iter()
            .filter(|w| w.contains("holds out"))
            .collect()
    }

    #[test]
    fn two_even_folds_are_no_warning_and_a_dominant_day_still_is() {
        // The real check: 851 frames in two folds of 426 and 425. Each fold model trains on the
        // other fold; one frame less is no warning.
        let mut s = Selected {
            k: 2,
            ..Selected::default()
        };
        day_in_fold(&mut s, "alpha", 426, 0);
        day_in_fold(&mut s, "bravo", 425, 1);
        assert_eq!(s.fold_frames(0).0.len(), 426);
        assert_eq!(s.fold_frames(0).1.len(), 425);
        assert!(held_warnings(&s).is_empty(), "{:?}", held_warnings(&s));
        // One camera day of 55 frames against 9 others: still a warning, for that fold only.
        let mut s = Selected {
            k: 2,
            ..Selected::default()
        };
        day_in_fold(&mut s, "big", 55, 0);
        day_in_fold(&mut s, "small", 9, 1);
        let w = held_warnings(&s);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].starts_with("fold 1 of 2 holds out 55 frames and its model trains on only 9"),
            "{w:?}"
        );
        // The edge: training on exactly 90% of the held-out frames is fine, one fewer is not.
        for (train, warns) in [(90, false), (89, true)] {
            let mut s = Selected {
                k: 2,
                ..Selected::default()
            };
            day_in_fold(&mut s, "held", 100, 0);
            day_in_fold(&mut s, "rest", train, 1);
            assert_eq!(!held_warnings(&s).is_empty(), warns, "{train}");
        }
    }

    fn boxed(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox,
            source: Source::Auto,
            score: Some(0.8),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    const CAR: [f32; 4] = [10.0, 10.0, 60.0, 50.0];
    const FAR: [f32; 4] = [150.0, 100.0, 200.0, 160.0];

    fn suggestion(kind: Kind, target: Option<[f32; 4]>, bbox: Option<[f32; 4]>) -> Suggestion {
        Suggestion {
            kind,
            target,
            target_label: target.map(|_| "car".into()),
            bbox,
            label: Some("car".into()),
            score: Some(0.7),
            why: "a fold model disagreed".into(),
            spot: None,
            answer: None,
            run: "r1".into(),
        }
    }

    fn reviewed(refine: Vec<Suggestion>) -> LabelFile {
        LabelFile {
            status: Status::Reviewed,
            reviewed_at: Some("2026-11-10T00:00:00Z".into()),
            boxes: vec![boxed("car", CAR)],
            refine,
            ..LabelFile::default()
        }
    }

    #[test]
    fn the_open_suggestions_a_check_replaces_are_its_kinds_unanswered_and_live() {
        let classes: Vec<String> = ["car", "dog", "person"].map(String::from).to_vec();
        let hidden = vec!["person".to_string()];
        let wanted = Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let add = || suggestion(Kind::Add, None, Some(FAR));
        let verify = || suggestion(Kind::Verify, Some(CAR), None);
        let relabel = || {
            let mut s = suggestion(Kind::Relabel, Some(CAR), None);
            s.label = Some("dog".into());
            s
        };
        let answered = || {
            let mut s = add();
            s.answer = Some(Answer::Dismissed);
            s
        };
        // Live, of the closer look's kind: counted apart (only a check with the look touches it).
        let tighten = suggestion(Kind::Tighten, Some(CAR), Some([12.0, 12.0, 58.0, 48.0]));
        // Stale (its box is not there) or of a hidden class: nobody sees them.
        let stale = suggestion(Kind::Verify, Some([1.0, 1.0, 9.0, 9.0]), None);
        let mut hidden_add = add();
        hidden_add.label = Some("person".into());
        let frames = [
            frame("a", "d1", "f1", reviewed(vec![add(), verify()])),
            frame("a", "d1", "f2", reviewed(vec![relabel(), answered()])),
            frame("b", "d1", "f3", reviewed(vec![answered(), tighten.clone()])),
            frame("b", "d1", "f4", reviewed(vec![stale, hidden_add])),
            frame("b", "d2", "f5", reviewed(vec![])),
        ];
        let open = open_to_replace(&frames, &wanted);
        assert_eq!(
            open,
            OpenCounts {
                folds: 3,
                fold_frames: 2,
                tightens: 1,
                any_frames: 3
            }
        );
        // Without the look the tighten is not touched, so not counted; with it, it is.
        assert_eq!(open.for_check(false), (3, 0, 2));
        assert_eq!(open.for_check(true), (3, 1, 3));
        assert_eq!(
            crate::collect::labels::refine::live(&frames[2].labels, Some(&wanted)).len(),
            1
        );
        // An answered or stale tighten is not counted.
        let mut done = tighten.clone();
        done.answer = Some(Answer::Dismissed);
        let gone = suggestion(Kind::Tighten, Some([1.0, 1.0, 9.0, 9.0]), Some(FAR));
        let more = [frame("c", "d1", "f6", reviewed(vec![done, gone]))];
        assert_eq!(open_to_replace(&more, &wanted), OpenCounts::default());
        assert_eq!(open_to_replace(&[], &wanted), OpenCounts::default());
    }

    #[test]
    fn short_fold_models_are_weak_for_every_family() {
        for family in [
            TrainFamily::Yolo26,
            TrainFamily::Yolo11,
            TrainFamily::Yolov8,
            TrainFamily::Rfdetr,
            TrainFamily::Rtdetrv2,
            TrainFamily::Dfine,
        ] {
            let min = super::super::recipe::recommended(family).min_epochs;
            for e in *EPOCHS.start()..=*EPOCHS.end() {
                assert_eq!(weak_epochs(family, e), e * 2 < min, "{family:?} {e}");
            }
            // The real case (1 epoch) is weak, the default (8) is not.
            assert!(weak_epochs(family, 1) && !weak_epochs(family, DEFAULT_EPOCHS));
        }
        assert!(weak_epochs(TrainFamily::Rfdetr, 4) && !weak_epochs(TrainFamily::Rfdetr, 5));
        assert_eq!(
            weak_epochs_warning(TrainFamily::Rfdetr, 1, 0),
            "1 epoch per fold model is under half the 10 epochs RF-DETR training takes before it \
             may stop: fold models this short find little (on one dataset, fold models of 1 epoch \
             found no missing box where fold models of 8 epochs found 123), so the check says \
             little about the labels"
        );
        let w = weak_epochs_warning(TrainFamily::Yolo26, 3, 5);
        assert!(w.starts_with("3 epochs per fold model"), "{w}");
        assert!(
            w.ends_with(
                ", and the open suggestions it replaces likely came from stronger fold models"
            ),
            "{w}"
        );
    }

    #[test]
    fn the_replace_warning_counts_suggestions_and_frames() {
        assert_eq!(
            replace_warning(139, 0, 52),
            "this check replaces the 139 open suggestions on 52 frames with what its fold models \
             find; answered ones stay. Answer them on the Review tab first, or run it to recompute \
             them"
        );
        assert!(
            replace_warning(1, 0, 1)
                .starts_with("this check replaces the 1 open suggestion on 1 frame with")
        );
        // With the closer look: its open proposed boxes, as a bound.
        assert_eq!(
            replace_warning(139, 6, 55),
            "this check replaces the 139 open suggestions on 55 frames with what its fold models \
             find, and up to 6 open proposed boxes for loose labels with what its closer look \
             finds; answered ones stay. Answer them on the Review tab first, or run it to \
             recompute them"
        );
        assert_eq!(
            replace_warning(0, 1, 1),
            "this check replaces up to 1 open proposed box for loose labels on 1 frame with what \
             its closer look finds; answered ones stay. Answer them on the Review tab first, or \
             run it to recompute them"
        );
    }

    #[test]
    fn the_preview_counts_the_open_suggestions_and_warns_first() {
        use crate::collect::labels::bulk::test_support::put_frame_at;
        let root = std::env::temp_dir().join(format!("bop-refine-open-{}", uuid::Uuid::new_v4()));
        let mut c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let dataset = root.join("dataset");
        c.collect.dir = Some(dataset.clone());
        let cap = super::super::capability::host_for_tests();
        let put = |stem: &str, refine: Vec<Suggestion>| {
            let rel = put_frame_at(
                &dataset,
                "patio",
                stem,
                (320, 240),
                None,
                vec![boxed("car", CAR)],
            );
            let path = crate::collect::labels::Frame::resolve(&dataset, &rel)
                .unwrap()
                .labels;
            let mut f = crate::collect::labels::read(&path).unwrap().unwrap();
            f.status = Status::Reviewed;
            f.reviewed_at = Some("2026-11-10T00:00:00Z".into());
            f.refine = refine;
            crate::collect::labels::write(&path, &f).unwrap();
        };
        put("a1", vec![]);
        // Nothing open: no warning about it, and 8 epochs are not weak.
        let p = preview(&c, &RefineRequest::default(), &cap).unwrap();
        assert_eq!((p.open_suggestions, p.open_frames), (0, 0));
        assert!(
            !p.warnings
                .iter()
                .any(|w| w.contains("open suggestion") || w.contains("per fold model")),
            "{:?}",
            p.warnings
        );
        put(
            "a2",
            vec![
                suggestion(Kind::Add, None, Some(FAR)),
                suggestion(Kind::Verify, Some(CAR), None),
            ],
        );
        put("a3", vec![suggestion(Kind::Add, None, Some(FAR))]);
        let p = preview(&c, &RefineRequest::default(), &cap).unwrap();
        assert_eq!((p.open_suggestions, p.open_frames), (3, 2));
        assert_eq!(p.warnings[0], replace_warning(3, 0, 2), "{:?}", p.warnings);
        assert!(
            p.refused.is_some(),
            "3 frames are too few, and still counted"
        );
        let text = preview_text(&p);
        assert!(
            text.contains(&format!("warning: {}\n", replace_warning(3, 0, 2))),
            "{text}"
        );
        // The frame filter narrows what is replaced.
        let elsewhere = RefineRequest {
            frames: RefineFrames::parse("camera=front").unwrap(),
            ..RefineRequest::default()
        };
        let p = preview(&c, &elsewhere, &cap).unwrap();
        assert_eq!((p.open_suggestions, p.open_frames), (0, 0));
        // One epoch: the second warning, which says the open ones came from stronger models.
        let short = RefineRequest {
            epochs: 1,
            ..RefineRequest::default()
        };
        let p = preview(&c, &short, &cap).unwrap();
        let family = choose_recipe(&c, 1, &cap).unwrap().0.family;
        assert_eq!(
            p.warnings[1],
            weak_epochs_warning(family, 1, 3),
            "{:?}",
            p.warnings
        );
        assert!(preview_text(&p).contains("warning: 1 epoch per fold model"));
        // RF-DETR L's weights are not here: no closer look, said once, last.
        assert!(!p.tighten);
        assert_eq!(p.warnings.last(), Some(&tighten::skipped_warning()));
        assert_eq!(
            start_warnings(&p, Some(family)),
            vec![
                replace_warning(3, 0, 2),
                weak_epochs_warning(family, 1, 3),
                tighten::skipped_warning()
            ]
        );
        assert!(preview_text(&p).contains("loose labels: counted only (no closer look)"));
        // Asked for, or told not to: no such warning.
        for asked in [Some(true), Some(false)] {
            let p = preview(
                &c,
                &RefineRequest {
                    tighten: asked,
                    ..short.clone()
                },
                &cap,
            )
            .unwrap();
            assert_eq!(p.tighten, asked == Some(true));
            assert!(
                !p.warnings.contains(&tighten::skipped_warning()),
                "{:?}",
                p.warnings
            );
        }
        // An open better box of an earlier look: counted (as a bound) and warned about only when
        // the check takes the closer look, which may replace it; without the look it is untouched.
        put(
            "a4",
            vec![suggestion(
                Kind::Tighten,
                Some(CAR),
                Some([12.0, 12.0, 58.0, 48.0]),
            )],
        );
        let with = |tighten: Option<bool>| {
            preview(
                &c,
                &RefineRequest {
                    tighten,
                    ..RefineRequest::default()
                },
                &cap,
            )
            .unwrap()
        };
        let p = with(Some(false));
        assert_eq!(
            (p.open_suggestions, p.open_tightens, p.open_frames),
            (3, 0, 2)
        );
        assert_eq!(p.warnings[0], replace_warning(3, 0, 2));
        let p = with(Some(true));
        assert_eq!(
            (p.open_suggestions, p.open_tightens, p.open_frames),
            (3, 1, 3)
        );
        assert_eq!(p.warnings[0], replace_warning(3, 1, 3), "{:?}", p.warnings);
        assert_eq!(
            start_warnings(&p, Some(family))[0],
            replace_warning(3, 1, 3)
        );
        assert!(
            preview_text(&p).contains("loose labels: a closer look by RF-DETR L"),
            "{}",
            preview_text(&p)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_closer_look_is_on_by_default_only_with_its_weights_and_its_time_is_said() {
        let root =
            std::env::temp_dir().join(format!("bop-refine-tighten-{}", uuid::Uuid::new_v4()));
        let c = Config {
            download_dir: Some(root.clone()),
            allow_large_downloads: true,
            ..Config::default()
        };
        let req = RefineRequest::default();
        assert!(!tighten_weights_ready(&c));
        assert!(!tighten_on(&c, &req));
        assert!(tighten_on(
            &c,
            &RefineRequest {
                tighten: Some(true),
                ..req.clone()
            }
        ));
        // The weights on disk (every part at its pinned size): on unless told not to.
        let paths = TrainPaths::for_config(&c);
        let (res, dir) = crate::training::overlaps::rfdetr_weights(&paths).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        for p in res.parts {
            let f = std::fs::File::create(dir.join(p.file_name)).unwrap();
            f.set_len(p.size).unwrap();
        }
        assert!(tighten_weights_ready(&c));
        assert!(tighten_on(&c, &req));
        assert!(!tighten_on(
            &c,
            &RefineRequest {
                tighten: Some(false),
                ..req.clone()
            }
        ));
        // The time line bounds the look per loose label (no promise of minutes); the basis stays
        // the folds'.
        let text = |tighten: bool| {
            preview_text(&RefinePreview {
                dataset_dir: root.clone(),
                filter: "all reviewed frames".into(),
                frames: 851,
                left_out: LeftOut::default(),
                camera_days: 12,
                folds: Vec::new(),
                recipe: RecipeInfo {
                    model: "rfdetr-nano".into(),
                    imgsz: 384,
                    epochs: 8,
                    batch: 8,
                    from: "the default".into(),
                },
                seconds: Some(4100.0),
                time_basis: Some(TimeBasis::LastRun),
                tighten,
                warnings: Vec::new(),
                open_suggestions: 0,
                open_tightens: 0,
                open_frames: 0,
                refused: None,
                folder_missing: false,
                mode: Mode::Folds,
                model: None,
            })
        };
        let on = text(true);
        assert!(
            on.contains(&format!(
                "(from your last training run); {}\n",
                tighten::time_line()
            )),
            "{on}"
        );
        assert!(
            on.contains("loose labels: a closer look by RF-DETR L"),
            "{on}"
        );
        assert!(!text(false).contains("closer look adds"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
