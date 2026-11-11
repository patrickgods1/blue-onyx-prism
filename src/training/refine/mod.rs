//! The dataset check (`blue-onyx-prism train refine`): a cross-check of the reviewed frames of the
//! folder under review by short throwaway fold models. The camera days of the reviewed frames are
//! split into K folds ([`folds::assign`]); for each fold a model trains on the other folds' frames
//! (the existing training run, [`crate::training::job::run_training`], in the job's work folder),
//! runs on the fold's frames it never saw ([`predict`]), and [`judge`] compares what it found with
//! the labels: missing boxes (`add`), labels no model found (`verify`), another class (`relabel`),
//! loose labels (counted). The suggestions go on the label files for a reviewer to confirm
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
//! - [`judge`]: one frame's labels against the detections, pure;
//! - [`write`]: the suggestions, re-judged under the label write lock;
//! - [`report`]: the report and its Wilson intervals.
//!
//! A route plans a check like the overlap check: [`plan_refine`] then the queue's `enqueue_task`;
//! [`preview`] answers the page's (and `--dry-run`'s) question of what a check would do, without a
//! GPU, a download or a write.

pub mod folds;
pub mod job;
pub mod judge;
pub mod predict;
pub mod report;
pub mod write;

pub use job::RefineJob;

use super::capability::Capability;
use super::dataset::SceneFilter;
use super::job::TrainRequest;
use super::recipe::TrainRecipe;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::export::{self, ClassFilter, ExportFrame, Selection, SplitConfig};
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
/// Rough fixed cost of a fold (self-check, dataset, ONNX export and its check), seconds.
pub const FOLD_OVERHEAD_S: f64 = 120.0;
/// Rough cost of predicting one frame in this process, seconds.
pub const PREDICT_S_PER_FRAME: f64 = 0.15;

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
}

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
        }
    }
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
        Ok(())
    }
}

/// The recipe of the newest finished (Done) training run in the `runs.json` text `text`, by its
/// `created` time; None when there is none or the file does not parse. Read only: `runs.json` is
/// never written, moved or repaired here.
pub fn newest_done_recipe(text: &str) -> Option<TrainRecipe> {
    let list: Vec<serde_json::Value> = serde_json::from_str(text).ok()?;
    list.iter()
        .filter(|r| r.get("state").and_then(serde_json::Value::as_str) == Some("done"))
        .filter_map(|r| {
            let created = r.get("created")?.as_str()?.to_string();
            let recipe: TrainRecipe =
                serde_json::from_value(r.get("request")?.get("recipe")?.clone()).ok()?;
            Some((created, recipe))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, r)| r)
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
    let usable =
        |r: &TrainRecipe| cap.family(r.family).is_none_or(|f| f.available) && r.validate().is_ok();
    let previous = std::fs::read_to_string(super::runs_file(config))
        .ok()
        .and_then(|t| newest_done_recipe(&t))
        .filter(usable)
        .and_then(|p| {
            let mut r = TrainRecipe::new(p.family, &p.size).ok()?;
            r.imgsz = p.imgsz;
            r.batch = p.batch;
            r.multi_scale = p.multi_scale;
            r.base_weights = p.base_weights;
            Some(r)
        });
    let (mut r, from) = match previous {
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
    Ok((r, from))
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

/// Select the frames of a check of `dataset_dir` (read only) and fold them.
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
        for b in f.labels.training().filter(|b| !b.is_ignore()) {
            if let Some(c) = wanted.class_of(&b.label) {
                *out.class_boxes.entry(c.to_string()).or_default() += 1;
            }
        }
    }
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
    pub warnings: Vec<String>,
    /// Why the check would be refused; None = it can run.
    pub refused: Option<String>,
}

/// What a check of `request` on `config` would do: no GPU, login or download needed, nothing
/// written. Errors only on a bad request.
pub fn preview(
    config: &Config,
    request: &RefineRequest,
    cap: &Capability,
) -> Result<RefinePreview> {
    request.validate()?;
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let s = select(
        &dir,
        &config.collect.classes,
        &config.collect.hidden_classes,
        &request.frames,
        request.folds,
        MIN_FRAMES,
    );
    let (recipe, from) = choose_recipe(config, request.epochs, cap)?;
    let mut warnings = s.warnings.clone();
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
    let seconds = match &accelerator {
        Some(a) => {
            let total = s.frames.len();
            let mut secs = 0.0;
            for p in &folds {
                let train = total.saturating_sub(p.frames);
                let est = super::estimate::estimate(
                    recipe.family,
                    &recipe.size,
                    recipe.imgsz,
                    recipe.batch,
                    train,
                    recipe.epochs,
                    a.target(),
                );
                secs += est.map_or(0.0, |e| e.total_s)
                    + FOLD_OVERHEAD_S
                    + p.frames as f64 * PREDICT_S_PER_FRAME;
            }
            (!folds.is_empty()).then_some(secs)
        }
        None => {
            warnings.push(format!(
                "the check trains fold models in the training environment, which needs an \
                 NVIDIA GPU or Apple silicon: {}",
                cap.summary()
            ));
            None
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
    Ok(RefinePreview {
        dataset_dir: dir,
        filter: request.frames.describe(),
        frames: s.frames.len(),
        left_out: s.left_out,
        camera_days: s.groups.len(),
        folds,
        recipe: recipe_info(&recipe, &from),
        seconds,
        warnings,
        refused: s.refused,
    })
}

/// The `--dry-run` text of `p`.
pub fn preview_text(p: &RefinePreview) -> String {
    let mut t = format!(
        "dataset check of {} in {}: {} frames on {} camera days\n",
        p.filter,
        p.dataset_dir.display(),
        p.frames,
        p.camera_days
    );
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
    t.push_str(&format!(
        "{} folds, each a {} model at {} px trained {} epochs (batch {}; {})\n",
        p.folds.len(),
        r.model,
        r.imgsz,
        r.epochs,
        r.batch,
        r.from
    ));
    for f in &p.folds {
        t.push_str(&format!(
            "  fold {}: {} frames on {} camera days\n",
            f.fold, f.frames, f.camera_days
        ));
    }
    if let Some(s) = p.seconds {
        t.push_str(&format!(
            "about {} on this GPU (rough)\n",
            super::estimate::format_duration(s)
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
pub fn plan_refine(config: &Config, request: RefineRequest, cap: &Capability) -> Result<RefineJob> {
    let mut request = request;
    request.validate()?;
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
        train,
        recipe_from: from,
        predictor: None,
        config: Box::new(config.clone()),
        min_frames: MIN_FRAMES,
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
}
