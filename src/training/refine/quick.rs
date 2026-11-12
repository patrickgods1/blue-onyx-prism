//! The quick check (`train refine --with-model <name>`, the Refine tab's "Quick check with a model
//! you trained"): instead of training throwaway fold models, a model the user already trained here
//! judges the reviewed frames it never trained or validated on, through the same predictor, judge
//! and writer as a fold ([`super::predict`], [`super::judge`], [`super::write`]). It needs no
//! training at all (seconds to minutes instead of about an hour), and no training environment unless
//! the closer look at loose labels runs.
//!
//! **Which frames.** The model's `<stem>.provenance.json` (beside its model file) names its training
//! run; the run's `data/train.txt` lists the images it trained on and `data/valid.txt` those it
//! validated on (its checkpoint was chosen on them, so they are not unseen either); held-out frames
//! (`data/heldout/`) and frames reviewed since are unseen and are checked. A frame is "seen" when
//! the dataset's name of any listed image is the frame's flat name as the dataset spells it (its
//! file-name rule with each source prefix the lists use, the 120-byte cut included, so a name the
//! cut made ambiguous leaves out every frame it may be), and, where the run recorded where each image
//! came from ([`crate::training::dataset::IMAGE_SOURCES_FILE`], newer runs), when the source's file
//! name is the frame's flat name. A listed image that record does not name refuses the check.
//!
//! **Refused, never guessed** ([`seen_of`], [`select`], [`check_weights`]): a model without the
//! record (a stock or downloaded model: what it trained on is unknown), a record that does not read
//! or names no run, a model file that is missing, a run folder that is gone, lists that do not
//! read, lists that name no reviewed frame of the folder at all (a run on another folder or on
//! renamed copies; a model that truly never saw a frame of the folder is refused too, the safe
//! side), and at preview and plan time a model file whose SHA-256 is not the record's
//! `onnx_sha256` (replaced since; a record without a hash is warned about, not refused). A
//! model judging frames it trained on finds every label "right", so a check that cannot leave them
//! out says nothing. Listed images without this program's frame names (public sets aside) are
//! counted and warned about ([`Seen::odd_names`]). The run folder is the run's `run_dir` in
//! `runs.json` of the training folder, else `<training dir>/<run id>`.
//!
//! **What the model has.** Its classes are the run's (the record's `classes`, else the run's
//! `data/classes.txt`) in the class list's spelling ([`judge::ModelJudge`]): a wanted class it does
//! not have gets no `verify` and no `relabel`, and is not in the recall; its detections of classes
//! off the list are not used at all (as a fold model's). The `verify` guard
//! ([`judge::MIN_CLASS_BOXES`]) counts the boxes the run trained on per class (its
//! `dataset.json` `stats.train_boxes`, repeats counted once); without that file no label gets a
//! `verify`. The input size is the run's recipe's.
//!
//! **What it cannot tell you.** A frame it never saw may still be of a camera day it trained on:
//! near copies of what it learned, so it finds those labels more easily than a fold model blind to
//! the whole day would ([`QuickSelected::same_day`], warned about). It is one model at its own
//! input size, not several short ones, so the recall tables measure that model on new frames, not
//! the recipe.

use super::judge::{self, ModelJudge};
use super::report::{LeftOut, ModelInfo, RecipeInfo, RefineReport};
use super::{
    Mode, RefineFrames, RefinePreview, RefineRequest, Selected, TimeBasis, open_to_replace,
    replace_warning_by, tighten, tighten_on,
};
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::export::{self, ExportFrame};
use crate::config::{Config, ModelConfig};
use crate::resources::catalog::Platform;
use crate::resources::export::train::{self as env, TrainPaths};
use crate::training::capability::{Accelerator, Capability};
use crate::training::recipe::TrainRecipe;
use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Fewest unseen frames a quick check runs on. A fold check needs [`super::MIN_FRAMES`] because
/// its models learn from the selection; a quick check's model is trained already, so any frame it
/// never saw can be judged, and one is enough to run (the preview says how few that is).
pub const MIN_UNSEEN_FRAMES: usize = 1;
/// The guess of a quick check's fixed cost, seconds, when no quick check with the model measured it
/// ([`super::report::RefineReport::predict_load_s`]): loading and compiling the model on its
/// device (an OpenVINO GPU compile without a cache takes the better part of a minute), the
/// selection walk and the report. Not measured.
pub const QUICK_FIXED_S: f64 = 60.0;
/// The guess of a quick check's cost per frame, seconds: decoding the frame, the model at its input
/// size on a slow device (an RF-DETR nano at 384 px takes about 0.11 s on a UHD 630), the judge and
/// the write. Not measured; a GPU is several times faster.
pub const QUICK_S_PER_FRAME: f64 = 0.25;

/// The model of a quick check and its training run, as its records say ([`locate`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Located {
    /// The configured model's name.
    pub name: String,
    /// `<stem>.provenance.json`.
    pub record: PathBuf,
    /// Its training run's id.
    pub run: String,
    /// The run's folder; `data/` in it holds the lists.
    pub run_dir: PathBuf,
    /// `rfdetr-nano`, ... (`unknown` when the recipe does not read).
    pub recipe: String,
    pub recipe_full: Option<TrainRecipe>,
    pub imgsz: u32,
    /// The run's classes (the record's spelling).
    pub classes: Vec<String>,
    /// The model file ([`Config::data_path`] of the entry's `path`); it exists when located.
    pub model_file: PathBuf,
    /// The SHA-256 of the ONNX its run installed (the record's `onnx_sha256`); None when the record
    /// has none.
    pub onnx_sha256: Option<String>,
    /// When the run made it (the record's `created`), for the newest-first order.
    pub created: String,
}

/// What a model's training run shows it saw ([`seen_of`]).
#[derive(Debug, Clone, Default)]
pub struct Seen {
    /// The model and its run; None only in [`Seen::default`].
    pub located: Option<Located>,
    /// Image names of `train.txt` / `valid.txt` as the dataset wrote them (`s<N>-<name>`).
    train: BTreeSet<String>,
    valid: BTreeSet<String>,
    /// The source prefixes (`s0`, `s1`, ...) those names use.
    prefixes: BTreeSet<String>,
    /// The file names of the sources of those images ([`crate::training::dataset::IMAGE_SOURCES_FILE`]).
    train_sources: BTreeSet<String>,
    valid_sources: BTreeSet<String>,
    /// The run recorded its images' sources: frames are matched by source too.
    pub by_source: bool,
    /// Training boxes per class of the run (`dataset.json` `stats.train_boxes`, the run's class
    /// names); None when it does not read.
    pub train_boxes: Option<BTreeMap<String, usize>>,
    /// Camera days (camera, `YYYY-MM-DD`) of the listed images that name a dataset frame.
    days: BTreeSet<(String, String)>,
    /// Listed images (`train.txt` and `valid.txt`, each name once) that carry no frame name of
    /// this program ([`odd_name`]): a dataset made by hand or by another tool, or renamed copies.
    /// Images of the run's public sets (its `dataset.json` sources) are not counted.
    pub odd_names: usize,
}

/// How a model saw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Saw {
    /// In its run's `train.txt`.
    Trained,
    /// In its run's `valid.txt`.
    Validated,
}

/// The dataset's file-name rule ([`crate::training::dataset`]: letters, digits, `-`, `_` and `.`,
/// others `_`, at most 120 bytes) applied to `<prefix>-<flat>`.
fn spelled(prefix: &str, flat: &str) -> String {
    let mut s: String = format!("{prefix}-{flat}")
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
    s
}

/// The `s<digits>` prefix of a dataset image name, and the rest.
fn split_prefix(name: &str) -> Option<(&str, &str)> {
    let (p, rest) = name.split_once('-')?;
    let digits = p.strip_prefix('s')?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some((p, rest))
}

impl Seen {
    /// How the model saw `f`, if at all (see the module docs).
    pub fn saw(&self, f: &ExportFrame) -> Option<Saw> {
        let hit = |names: &BTreeSet<String>, sources: &BTreeSet<String>| {
            sources.contains(&f.name)
                || self
                    .prefixes
                    .iter()
                    .any(|p| names.contains(&spelled(p, &f.name)))
        };
        if hit(&self.train, &self.train_sources) {
            Some(Saw::Trained)
        } else if hit(&self.valid, &self.valid_sources) {
            Some(Saw::Validated)
        } else {
            None
        }
    }

    /// Whether the run trained or validated on any frame of `camera` on `date`.
    pub fn saw_day(&self, camera: &str, date: &str) -> bool {
        self.days.contains(&(camera.to_string(), date.to_string()))
    }

    /// Image names in the run's lists (`train.txt` and `valid.txt`, each name once per list).
    pub fn listed(&self) -> usize {
        self.train.len() + self.valid.len()
    }
}

/// Whether the dataset image `name` (of a list, with its `s<N>-` prefix) carries no frame name of
/// this program: no source prefix, or after it no flat frame name ([`export::parse_flat_name`])
/// unless the dataset's 120-byte cut ([`spelled`]) took its end; `source` is the file name the
/// run's record of image sources gives it, when there is one (a frame name there counts).
fn odd_name(name: &str, source: Option<&str>) -> bool {
    if source.is_some_and(|s| export::parse_flat_name(s).is_some()) {
        return false;
    }
    match split_prefix(name) {
        Some((_, rest)) => export::parse_flat_name(rest).is_none() && name.len() < 120,
        None => true,
    }
}

/// The warning of a run whose lists hold `n` of `of` images with no frame name of this program.
pub fn odd_names_warning(n: usize, of: usize, l: &Located) -> String {
    format!(
        "{n} of the {of} images in run {}'s lists do not carry this program's frame names (a \
         dataset folder made by hand or by another tool, or renamed copies): they cannot be \
         matched to frames here, so if any is a copy of a frame of this folder, {} is judged on \
         it as if it never saw it",
        l.run, l.name
    )
}

/// `name` of the config's models, as the model names are compared ([`crate::registry::normalize_name`]).
pub fn find_entry<'a>(config: &'a Config, name: &str) -> Option<&'a ModelConfig> {
    let key = crate::registry::normalize_name(name);
    config
        .models
        .iter()
        .find(|m| crate::registry::normalize_name(&m.effective_name()) == key)
}

/// The training record beside `m`'s model file.
pub fn record_path(config: &Config, m: &ModelConfig) -> PathBuf {
    let path = config.data_path(&m.path);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}.provenance.json"))
}

/// Why a quick check refuses: what is missing and why it matters.
fn refusal(name: &str, what: String) -> anyhow::Error {
    anyhow!(
        "{what}, so the frames {name} trained and validated on cannot be left out; a model judging \
         frames it trained on finds every label right, so the quick check refuses. Run the full \
         check (fold models), or choose a model trained on the Train tab whose run folder is still \
         there"
    )
}

/// A plain folder name (a run id), no path.
fn plain(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && !id.contains(['/', '\\'])
        && Path::new(id).file_name().and_then(|n| n.to_str()) == Some(id)
}

/// The `run_dir` of every run in `runs.json` of the training folder (read only, once per caller):
/// run id -> folder. Empty when the file is missing or does not read.
#[derive(Debug, Clone, Default)]
pub struct RunDirs(BTreeMap<String, PathBuf>);

impl RunDirs {
    /// `runs.json` of `config`'s training folder.
    pub fn read(config: &Config) -> Self {
        let list = std::fs::read(super::super::runs_file(config))
            .ok()
            .and_then(|b| serde_json::from_slice::<Vec<serde_json::Value>>(&b).ok())
            .unwrap_or_default();
        let mut out = BTreeMap::new();
        for r in list {
            if let (Some(id), Some(dir)) = (
                r.get("id").and_then(serde_json::Value::as_str),
                r.get("run_dir").and_then(serde_json::Value::as_str),
            ) {
                out.entry(id.to_string())
                    .or_insert_with(|| PathBuf::from(dir));
            }
        }
        Self(out)
    }
}

/// The folder of run `run`: its `run_dir` in `runs.json` (`dirs`), else `<training dir>/<run>`;
/// the first that holds `data/`.
fn run_folder(config: &Config, dirs: &RunDirs, run: &str) -> Option<PathBuf> {
    dirs.0
        .get(run)
        .cloned()
        .into_iter()
        .chain(std::iter::once(
            super::super::training_dir(config).join(run),
        ))
        .find(|d| d.join("data").is_dir())
}

/// The model `m` and its training run, from its records (no list read): refused ([`refusal`]) when
/// it has no record, the record does not read or names no run, or the run folder is gone; an
/// error when its model file is missing.
pub fn locate(config: &Config, m: &ModelConfig) -> Result<Located> {
    locate_in(config, m, &RunDirs::read(config))
}

/// [`locate`] with `runs.json` read already (`dirs`).
pub fn locate_in(config: &Config, m: &ModelConfig, dirs: &RunDirs) -> Result<Located> {
    #[derive(Deserialize)]
    struct Record {
        #[serde(default)]
        run_id: String,
        #[serde(default)]
        recipe: Option<serde_json::Value>,
        #[serde(default)]
        classes: Vec<String>,
        #[serde(default)]
        created: String,
        #[serde(default)]
        onnx_sha256: String,
    }
    let name = m.effective_name();
    let record = record_path(config, m);
    let text = match std::fs::read(&record) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(refusal(
                &name,
                format!(
                    "{name} has no training record ({} is missing): it is not a model trained here \
                     (a stock or downloaded model), and what it trained on is unknown",
                    record.display()
                ),
            ));
        }
        Err(e) => {
            return Err(refusal(
                &name,
                format!("{} could not be read ({e})", record.display()),
            ));
        }
    };
    let r: Record = serde_json::from_slice(&text)
        .map_err(|e| refusal(&name, format!("{} does not read ({e})", record.display())))?;
    if !plain(&r.run_id) {
        return Err(refusal(
            &name,
            format!("{} names no training run", record.display()),
        ));
    }
    let model_file = config.data_path(&m.path);
    if !model_file.is_file() {
        bail!(
            "{name}'s model file {} is missing (moved or deleted?), so there is nothing to run; \
             put it back or choose another model",
            model_file.display()
        );
    }
    let run_dir = run_folder(config, dirs, &r.run_id).ok_or_else(|| {
        refusal(
            &name,
            format!(
                "{name} was trained in run {}, whose folder is gone (neither the runs list's \
                 run_dir nor {} holds its data folder)",
                r.run_id,
                super::super::training_dir(config).join(&r.run_id).display()
            ),
        )
    })?;
    let recipe_full = r
        .recipe
        .clone()
        .and_then(|v| serde_json::from_value::<TrainRecipe>(v).ok());
    let imgsz = r
        .recipe
        .as_ref()
        .and_then(|v| v.get("imgsz")?.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            anyhow!(
                "{} names no input size (recipe.imgsz), which the check measures box sizes at; \
                 choose another model",
                record.display()
            )
        })?;
    let mut classes = r.classes;
    if classes.is_empty() {
        classes = std::fs::read_to_string(run_dir.join("data").join("classes.txt"))
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
    }
    if classes.is_empty() {
        return Err(anyhow!(
            "neither {} nor its run's data/classes.txt names the classes {name} detects; choose \
             another model",
            record.display()
        ));
    }
    Ok(Located {
        recipe: recipe_full
            .as_ref()
            .map_or_else(|| "unknown".to_string(), TrainRecipe::label),
        recipe_full,
        name,
        record,
        run: r.run_id,
        run_dir,
        imgsz,
        classes,
        model_file,
        onnx_sha256: Some(r.onnx_sha256.trim().to_ascii_lowercase()).filter(|h| !h.is_empty()),
        created: r.created,
    })
}

/// Whether the model file of `l` is the one its training run produced: its SHA-256 against the
/// record's `onnx_sha256` (a hash of the whole file: run at plan and preview time only, never per
/// `GET /v1/refine`). `Ok(None)`: the same file; `Ok(Some(warning))`: it could not be checked (the
/// record has no hash, or the entry names no `.onnx`, which the hash is of); `Err`: another file,
/// or it does not read.
pub fn check_weights(l: &Located) -> Result<Option<String>> {
    let Some(want) = &l.onnx_sha256 else {
        return Ok(Some(format!(
            "{} names no hash of the model file, so the check cannot tell that {} is the file \
             training run {} produced",
            l.record.display(),
            l.model_file.display(),
            l.run
        )));
    };
    let is_onnx = l
        .model_file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    if !is_onnx {
        return Ok(Some(format!(
            "{} is no .onnx file, and the record's hash is of the ONNX training run {} \
             installed, so the check cannot tell that it is that model",
            l.model_file.display(),
            l.run
        )));
    }
    let got = crate::resources::manager::sha256_file(&l.model_file)
        .with_context(|| format!("reading {} to check it", l.model_file.display()))?;
    if got.eq_ignore_ascii_case(want) {
        Ok(None)
    } else {
        bail!(
            "{} is not the file training run {} produced (replaced since?): the frames it \
             trained on are unknown, so the quick check refuses. Run the full check (fold \
             models), or train the model again",
            l.model_file.display(),
            l.run
        )
    }
}

/// The image names of a list file (`./images/<name>` lines, each once).
fn list_names(path: &Path) -> Result<BTreeSet<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{} ({e})", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| l.rsplit(['/', '\\']).next().unwrap_or(l).to_string())
        .collect())
}

/// What model `m` trained and validated on ([`locate`], then the run's lists, its record of image
/// sources when it has one, and its training box counts). Refused as [`locate`] is, and when a list
/// does not read or the source record misses a listed image.
pub fn seen_of(config: &Config, m: &ModelConfig) -> Result<Seen> {
    let located = locate(config, m)?;
    let name = located.name.clone();
    let data = located.run_dir.join("data");
    let train = list_names(&data.join("train.txt")).map_err(|e| {
        refusal(
            &name,
            format!("its run's training list does not read: {e:#}"),
        )
    })?;
    let valid = list_names(&data.join("valid.txt")).map_err(|e| {
        refusal(
            &name,
            format!("its run's validation list does not read: {e:#}"),
        )
    })?;
    let mut out = Seen {
        by_source: false,
        ..Seen::default()
    };
    let sources_file = data.join(crate::training::dataset::IMAGE_SOURCES_FILE);
    let mut source_names: BTreeMap<String, String> = BTreeMap::new();
    if sources_file.is_file() {
        let sources = crate::training::dataset::read_image_sources(&data)
            .map_err(|e| refusal(&name, format!("{e:#}")))?;
        for (n, src) in &sources {
            if let Some(f) = src.file_name() {
                source_names.insert(n.clone(), f.to_string_lossy().into_owned());
            }
        }
        for (names, into) in [
            (&train, &mut out.train_sources),
            (&valid, &mut out.valid_sources),
        ] {
            for n in names {
                let Some(src) = sources.get(n) else {
                    return Err(refusal(
                        &name,
                        format!(
                            "{n} of its run's lists is not in {}: the check cannot tell which \
                             frame it is",
                            sources_file.display()
                        ),
                    ));
                };
                if let Some(f) = src.file_name() {
                    into.insert(f.to_string_lossy().into_owned());
                }
            }
        }
        out.by_source = true;
    }
    for n in train.iter().chain(&valid) {
        if let Some((p, rest)) = split_prefix(n) {
            out.prefixes.insert(p.to_string());
            if let Some(r) = export::parse_flat_name(rest) {
                out.days.insert((r.camera, r.date));
            }
        }
    }
    for s in out.train_sources.iter().chain(&out.valid_sources) {
        if let Some(r) = export::parse_flat_name(s) {
            out.days.insert((r.camera, r.date));
        }
    }
    // The run's public sets (`s<i>-` is its i-th source in dataset.json) carry their own names.
    let public: BTreeSet<String> = crate::training::dataset::read_summary_sources(&data)
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind != "review")
        .map(|(i, _)| format!("s{i}"))
        .collect();
    out.odd_names = train
        .iter()
        .chain(&valid)
        .filter(|n| !split_prefix(n).is_some_and(|(p, _)| public.contains(p)))
        .filter(|n| odd_name(n, source_names.get(*n).map(String::as_str)))
        .count();
    out.train = train;
    out.valid = valid;
    out.train_boxes = read_train_boxes(&data);
    out.located = Some(located);
    Ok(out)
}

/// `stats.train_boxes` of the run's `dataset.json`; None when it does not read.
fn read_train_boxes(data: &Path) -> Option<BTreeMap<String, usize>> {
    #[derive(Deserialize)]
    struct Stats {
        train_boxes: BTreeMap<String, usize>,
    }
    #[derive(Deserialize)]
    struct Summary {
        stats: Stats,
    }
    let b = std::fs::read(data.join("dataset.json")).ok()?;
    serde_json::from_slice::<Summary>(&b)
        .ok()
        .map(|s| s.stats.train_boxes)
}

/// The model of `seen` as the judge needs it: its name and its classes in the list's spelling.
pub fn model_judge(seen: &Seen, wanted: &Wanted<'_>) -> ModelJudge {
    let Some(l) = &seen.located else {
        return ModelJudge::default();
    };
    ModelJudge {
        name: l.name.clone(),
        classes: l
            .classes
            .iter()
            .filter_map(|c| wanted.class_of(c))
            .map(str::to_string)
            .collect(),
    }
}

/// The wanted classes (hidden ones not) the model of `judge` does not have, in the list's order.
pub fn classes_missing(judge: &ModelJudge, wanted: &Wanted<'_>) -> Vec<String> {
    wanted
        .classes
        .iter()
        .filter(|c| {
            wanted
                .class_of(c)
                .is_some_and(|w| !judge.classes.contains(w))
        })
        .cloned()
        .collect()
}

/// Training boxes per wanted class (the list's spelling) of the model's run: what
/// [`judge::Context::class_boxes`] holds for a quick check. Empty when the run's counts do not read.
pub fn model_class_boxes(seen: &Seen, wanted: &Wanted<'_>) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (c, n) in seen.train_boxes.iter().flatten() {
        if let Some(w) = wanted.class_of(c) {
            *out.entry(w.to_string()).or_default() += n;
        }
    }
    out
}

/// What a quick check selects ([`select`]).
#[derive(Debug, Default)]
pub struct QuickSelected {
    /// `frames`: the reviewed frames of the filter the model never saw, in the export's order;
    /// `groups` their camera days; `open` the open suggestions on them; `spots` made from every
    /// reviewed frame of the filter (the reviewer's labels are the evidence, which the model's
    /// training does not concern); `class_boxes` what the model's run trained on.
    pub s: Selected,
    /// Frames of the filter the model trained on: skipped.
    pub trained_on: usize,
    /// Frames of the filter it validated on: skipped.
    pub validated_on: usize,
    /// Of `s.frames`, those of a camera day it trained or validated on.
    pub same_day: usize,
}

/// Select the frames of a quick check of `dataset_dir` (read only) with the model that saw `seen`:
/// the fold check's selection ([`super::select`]), without the frames the model trained or
/// validated on. Refused with fewer than [`MIN_UNSEEN_FRAMES`], and when the run's lists are not
/// empty but name no exportable reviewed frame of the folder, the frame filter aside.
pub fn select(
    dataset_dir: &Path,
    classes: &[String],
    hidden: &[String],
    filter: &RefineFrames,
    seen: &Seen,
) -> QuickSelected {
    // No folds: their count and refusal are of no use here.
    let (mut s, filtered) = super::select_parts(dataset_dir, classes, hidden, filter, 2, 0);
    s.refused = None;
    s.folds.clear();
    s.k = 0;
    let wanted = Wanted { classes, hidden };
    let mut out = QuickSelected::default();
    let mut days: BTreeSet<(String, String)> = BTreeSet::new();
    let mut unseen = Vec::new();
    for f in std::mem::take(&mut s.frames) {
        match seen.saw(&f) {
            Some(saw) => {
                match saw {
                    Saw::Trained => out.trained_on += 1,
                    Saw::Validated => out.validated_on += 1,
                }
                days.insert((f.rel.camera.clone(), f.rel.date.clone()));
            }
            None => unseen.push(f),
        }
    }
    out.same_day = unseen
        .iter()
        .filter(|f| {
            days.contains(&(f.rel.camera.clone(), f.rel.date.clone()))
                || seen.saw_day(&f.rel.camera, &f.rel.date)
        })
        .count();
    s.groups.clear();
    for f in &unseen {
        *s.groups
            .entry((f.rel.camera.clone(), f.rel.date.clone()))
            .or_default() += 1;
    }
    s.open = open_to_replace(&unseen, &wanted);
    s.class_boxes = model_class_boxes(seen, &wanted);
    s.frames = unseen;
    let name = seen
        .located
        .as_ref()
        .map_or("the model", |l| l.name.as_str());
    // Lists that name no reviewed frame of the folder at all (the frame filter aside): the run
    // trained on another folder or on renamed copies, and every frame it saw would pass for new.
    let found_any =
        out.trained_on + out.validated_on > 0 || filtered.iter().any(|f| seen.saw(f).is_some());
    if seen.listed() > 0 && !found_any {
        s.refused = Some(format!(
            "none of the {} images {name} trained or validated on is a reviewed frame of {}: it \
             trained on another folder or on renamed copies, so the check cannot tell which \
             frames it saw. A model that never saw a frame of this folder is refused the same \
             way (the safe side); run the full check (fold models)",
            seen.listed(),
            dataset_dir.display(),
        ));
    } else if s.frames.len() < MIN_UNSEEN_FRAMES {
        s.refused = Some(format!(
            "{name} trained or validated on every reviewed frame of {} that can be exported ({} \
             trained on, {} validated on; {} left out: {} with pending boxes, {} with labels off \
             the class list, {} by the frame filter), so there is no frame it never saw: review \
             new frames, or run the full check",
            dataset_dir.display(),
            out.trained_on,
            out.validated_on,
            s.left_out.frames(),
            s.left_out.pending,
            s.left_out.off_class_list,
            s.left_out.filtered,
        ));
    }
    out.s = s;
    out
}

/// The entry a quick check runs its model with: the configured one (its family, class file, device
/// and GPU precision), disabled, with nothing that would hide a detection the judge needs: no
/// confidence threshold of its own nor per device (the check asks for everything from
/// [`judge::MATCH_MIN_SCORE`]; a model's own 0.5 would turn every faint find into a `verify`), and
/// no object filter. A copy, never saved.
pub fn predict_entry(m: &ModelConfig) -> ModelConfig {
    ModelConfig {
        enabled: false,
        confidence_threshold: None,
        device_thresholds: BTreeMap::new(),
        object_filter: None,
        ..m.clone()
    }
}

/// Where the closer look runs: the training environment and its GPU.
#[derive(Debug, Clone)]
pub struct LookEnv {
    pub paths: TrainPaths,
    pub platform: Platform,
    pub accelerator: Accelerator,
    pub timeout: Duration,
}

/// A planned quick check's model ([`super::RefineJob::quick`]).
#[derive(Debug, Clone)]
pub struct Quick {
    /// The model's config entry as configured ([`predict_entry`] is applied when it runs).
    pub entry: ModelConfig,
    /// The model as the judge words it (its classes as planned).
    pub judge: ModelJudge,
    /// Its input size.
    pub imgsz: u32,
    /// Frames it never saw when planned (the title's count; the run selects again).
    pub unseen: usize,
    /// Where the closer look runs; None without it.
    pub look: Option<LookEnv>,
    /// Why the model file could not be checked against its record when planned
    /// ([`check_weights`]); the report repeats it.
    pub weights_warning: Option<String>,
}

/// Whether a quick check of `request` looks closer at its loose labels, and the warning when it
/// does not for want of a GPU: the closer look runs RF-DETR L in the training environment, so it
/// needs a GPU to train on (an NVIDIA GPU or Apple silicon) even though the model's pass does not.
/// Asked for (`tighten: true`) without one: the refusal (`Err`).
pub fn look_choice(
    config: &Config,
    request: &RefineRequest,
    cap: &Capability,
) -> std::result::Result<(bool, Option<String>), String> {
    let wanted = tighten_on(config, request);
    if !wanted {
        return Ok((false, None));
    }
    match cap.resolve_gpu(request.gpu.as_deref()) {
        Ok(Some(_)) => Ok((true, None)),
        Ok(None) if request.tighten == Some(true) => Err(format!(
            "the closer look at loose labels runs RF-DETR L in the training environment, which \
             needs an NVIDIA GPU or Apple silicon: {}; run the quick check without it",
            cap.summary()
        )),
        Ok(None) => Ok((
            false,
            Some(format!(
                "loose labels are counted only: the closer look needs an NVIDIA GPU or Apple \
                 silicon ({}); the model itself runs on its own device",
                cap.summary()
            )),
        )),
        Err(e) => Err(format!("{e:#}")),
    }
}

/// The time a quick check over `frames` frames takes, seconds, and where it came from: the last
/// quick check with the same model (`last`, the quick report in the folder under review) measured
/// the model's load ([`RefineReport::predict_load_s`]) and its time per frame
/// ([`RefineReport::predict_s_per_frame`]), so `load + per frame x frames`; else the documented
/// guess ([`QUICK_FIXED_S`] + [`QUICK_S_PER_FRAME`] x frames). A report without both (an earlier
/// one, whose time per frame held the load) is no measure.
pub fn quick_seconds(frames: usize, name: &str, last: Option<&RefineReport>) -> (f64, TimeBasis) {
    #[allow(clippy::cast_precision_loss, reason = "a count of frames")]
    let n = frames as f64;
    let measured = last
        .filter(|r| r.mode == Mode::Model)
        .filter(|r| {
            r.model.as_ref().is_some_and(|m| {
                crate::registry::normalize_name(&m.name) == crate::registry::normalize_name(name)
            })
        })
        .and_then(|r| Some((r.predict_load_s?, r.predict_s_per_frame?)))
        .filter(|(load, per)| load.is_finite() && *load >= 0.0 && per.is_finite() && *per > 0.0);
    match measured {
        Some((load, per)) => (load + per * n, TimeBasis::LastCheck),
        None => (QUICK_FIXED_S + QUICK_S_PER_FRAME * n, TimeBasis::Guess),
    }
}

/// The quick check's report in the folder under review, when one reads (for its measured pace).
fn last_report(dir: &Path) -> Option<RefineReport> {
    let b = std::fs::read(super::report::quick_report_path(dir)).ok()?;
    serde_json::from_slice(&b).ok()
}

/// The warning of a check whose frames include `n` of camera days the model trained on.
pub fn same_day_warning(n: usize, of: usize, name: &str) -> String {
    format!(
        "{n} of the {of} frames {name} never saw are of camera days it trained or validated on: \
         near copies of what it learned, so it finds their labels more easily than a model blind \
         to the whole day would, and a quiet result there says less (the full check holds out \
         whole camera days)"
    )
}

/// The warning of a model without some wanted classes.
pub fn missing_classes_warning(name: &str, missing: &[String]) -> String {
    format!(
        "{name} does not detect {}: those labels get no verify and no other-class suggestion and \
         are left out of the recall",
        missing.join(", ")
    )
}

/// What the preview's `model` says of a model that was refused before its frames were known.
fn bare_info(name: &str) -> ModelInfo {
    ModelInfo {
        name: name.to_string(),
        ..ModelInfo::default()
    }
}

/// The [`ModelInfo`] of a quick check by the model of `seen` over `qs`. (The odd-name count is a
/// warning of the preview and the report, [`odd_names_warning`].)
pub fn model_info(seen: &Seen, qs: &QuickSelected, missing: Vec<String>) -> ModelInfo {
    let l = seen.located.as_ref();
    ModelInfo {
        name: l.map(|l| l.name.clone()).unwrap_or_default(),
        run: l.map(|l| l.run.clone()).unwrap_or_default(),
        recipe: l.map(|l| l.recipe.clone()).unwrap_or_default(),
        imgsz: l.map_or(0, |l| l.imgsz),
        trained_on: qs.trained_on,
        validated_on: qs.validated_on,
        same_day: qs.same_day,
        classes_missing: missing,
        by_source: seen.by_source,
        device: String::new(),
    }
}

/// [`RecipeInfo`] of the model of `l` (its own recipe: no fold trains).
pub fn recipe_info(l: &Located) -> RecipeInfo {
    RecipeInfo {
        model: l.recipe.clone(),
        imgsz: l.imgsz,
        epochs: l.recipe_full.as_ref().map_or(0, |r| r.epochs),
        batch: l.recipe_full.as_ref().map_or(0, |r| r.batch),
        from: format!("the model {}, trained in run {}", l.name, l.run),
    }
}

/// [`super::preview`] of a quick check with the model named `name`: no GPU, download or write. An
/// error only for a bad request or a name no configured model has; a model whose frames cannot be
/// determined is `refused` with the reason.
pub fn preview(
    config: &Config,
    request: &RefineRequest,
    cap: &Capability,
    name: &str,
) -> Result<RefinePreview> {
    let entry = find_entry(config, name).ok_or_else(|| no_model(config, name))?;
    let name = entry.effective_name();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let missing_folder = config.collect.review_folder_problem(&dir);
    let classes = &config.collect.classes;
    let hidden = &config.collect.hidden_classes;
    let wanted = Wanted { classes, hidden };
    let mut p = RefinePreview {
        dataset_dir: dir.clone(),
        filter: request.frames.describe(),
        frames: 0,
        left_out: LeftOut::default(),
        camera_days: 0,
        folds: Vec::new(),
        recipe: RecipeInfo {
            model: String::new(),
            imgsz: 0,
            epochs: 0,
            batch: 0,
            from: format!("the model {name}"),
        },
        seconds: None,
        time_basis: None,
        tighten: false,
        warnings: Vec::new(),
        open_suggestions: 0,
        open_tightens: 0,
        open_frames: 0,
        refused: None,
        folder_missing: missing_folder.is_some(),
        mode: Mode::Model,
        model: Some(bare_info(&name)),
    };
    if let Some(why) = missing_folder {
        p.refused = Some(why);
        return Ok(p);
    }
    let seen = match seen_of(config, entry) {
        Ok(s) => s,
        Err(e) => {
            p.refused = Some(format!("{e:#}"));
            return Ok(p);
        }
    };
    let Some(located) = seen.located.clone() else {
        p.refused = Some(format!("{name}: no training record"));
        return Ok(p);
    };
    // The file against its record, as a start checks it (one hash of the model file).
    let weights_warning = match check_weights(&located) {
        Ok(w) => w,
        Err(e) => {
            p.refused = Some(format!("{e:#}"));
            None
        }
    };
    p.recipe = recipe_info(&located);
    let qs = select(&dir, classes, hidden, &request.frames, &seen);
    let judge = model_judge(&seen, &wanted);
    let missing = classes_missing(&judge, &wanted);
    let mut info = model_info(&seen, &qs, missing.clone());
    info.device = config
        .device_spec_for(entry)
        .map(|d| d.to_string())
        .unwrap_or_default();
    let (look, look_warning) = match look_choice(config, request, cap) {
        Ok(c) => c,
        Err(why) => {
            p.refused.get_or_insert(why);
            (false, None)
        }
    };
    p.tighten = look;
    let (open, tightens, open_frames) = qs.s.open.for_check(look);
    (p.open_suggestions, p.open_tightens, p.open_frames) = (open, tightens, open_frames);
    if open + tightens > 0 {
        p.warnings.push(replace_warning_by(
            open,
            tightens,
            open_frames,
            &format!("what {name} finds"),
        ));
    }
    if qs.same_day > 0 {
        p.warnings
            .push(same_day_warning(qs.same_day, qs.s.frames.len(), &name));
    }
    if !missing.is_empty() {
        p.warnings.push(missing_classes_warning(&name, &missing));
    }
    if seen.train_boxes.is_none() {
        p.warnings.push(no_counts_warning(&located));
    } else {
        let rare: Vec<String> = judge
            .classes
            .iter()
            .map(|c| (c, qs.s.class_boxes.get(c).copied().unwrap_or(0)))
            .filter(|(_, n)| *n < judge::MIN_CLASS_BOXES)
            .map(|(c, n)| format!("{c} ({n})"))
            .collect();
        if !rare.is_empty() {
            p.warnings.push(format!(
                "{name} trained on fewer than {} boxes of {}: their labels get no verify \
                 suggestion",
                judge::MIN_CLASS_BOXES,
                rare.join(", ")
            ));
        }
    }
    if !seen.by_source {
        p.warnings.push(by_name_note(&located));
    }
    if seen.odd_names > 0 {
        p.warnings
            .push(odd_names_warning(seen.odd_names, seen.listed(), &located));
    }
    if let Some(w) = weights_warning {
        p.warnings.push(w);
    }
    p.warnings.extend(qs.s.warnings.iter().cloned());
    if let Some(w) = look_warning {
        p.warnings.push(w);
    } else if !look && request.tighten.is_none() {
        p.warnings.push(tighten::skipped_warning());
    }
    if !qs.s.frames.is_empty() {
        let last = last_report(&dir);
        let (mut secs, basis) = quick_seconds(qs.s.frames.len(), &name, last.as_ref());
        if look {
            secs += tighten::estimate_seconds(qs.s.frames.len());
        }
        p.seconds = Some(secs);
        p.time_basis = Some(basis);
    }
    p.frames = qs.s.frames.len();
    p.camera_days = qs.s.groups.len();
    p.left_out = qs.s.left_out.clone();
    p.model = Some(info);
    if p.refused.is_none() {
        p.refused = qs.s.refused;
    }
    Ok(p)
}

/// The warning of a run without its training box counts.
pub fn no_counts_warning(l: &Located) -> String {
    format!(
        "the training box counts of run {} ({}) do not read, so no label gets a verify \
         suggestion (the check cannot tell which classes {} learned well enough to miss)",
        l.run,
        l.run_dir.join("data").join("dataset.json").display(),
        l.name
    )
}

/// The note on a run of an older version, matched by name only.
pub fn by_name_note(l: &Located) -> String {
    format!(
        "run {} does not record where its images came from (runs of earlier versions), so frames \
         are matched by name only, the dataset's way of spelling them included (a name it cut \
         short leaves out every frame it may be)",
        l.run
    )
}

/// "No configured model named ...", with the choices.
pub fn no_model(config: &Config, name: &str) -> anyhow::Error {
    let names: Vec<String> = models(config).0.into_iter().map(|m| m.name).collect();
    anyhow!(
        "no configured model is named '{name}'; {}",
        if names.is_empty() {
            "no configured model was trained here (the quick check needs one with its training \
             record)"
                .to_string()
        } else {
            format!("models trained here: {}", names.join(", "))
        }
    )
}

/// A configured model a quick check can use (`GET /v1/refine` `capability.models`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuickModel {
    pub name: String,
    pub run: String,
    pub recipe: String,
    pub imgsz: u32,
    /// Serves requests now (an enabled entry).
    pub enabled: bool,
    /// When its run made it (the record's `created`), for the newest-first order.
    #[serde(skip)]
    created: String,
}

/// A configured model with a training record that a quick check cannot use, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Unusable {
    pub name: String,
    pub why: String,
}

/// The configured models with a training record: those a quick check can use ([`locate`] finds
/// their model file and their run, and its lists are there; newest first), and those it cannot,
/// with why. Models without a record (stock, downloaded) are in neither. Reads `runs.json` once and
/// each record once, never the lists or the model files' bytes (no hash: [`check_weights`] runs
/// when a check is previewed or planned).
pub fn models(config: &Config) -> (Vec<QuickModel>, Vec<Unusable>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    let mut dirs: Option<RunDirs> = None;
    for m in &config.models {
        let record = record_path(config, m);
        if !record.is_file() {
            continue;
        }
        let name = m.effective_name();
        let dirs = dirs.get_or_insert_with(|| RunDirs::read(config));
        let usable = locate_in(config, m, dirs).and_then(|l| {
            for list in ["train.txt", "valid.txt"] {
                let p = l.run_dir.join("data").join(list);
                if !p.is_file() {
                    return Err(refusal(&name, format!("{} is missing", p.display())));
                }
            }
            Ok(l)
        });
        match usable {
            Ok(l) => {
                ok.push(QuickModel {
                    name,
                    run: l.run,
                    recipe: l.recipe,
                    imgsz: l.imgsz,
                    enabled: m.enabled,
                    created: l.created,
                });
            }
            Err(e) => bad.push(Unusable {
                name,
                why: format!("{e:#}"),
            }),
        }
    }
    ok.sort_by(|a, b| b.created.cmp(&a.created).then(a.name.cmp(&b.name)));
    (ok, bad)
}

/// Plan the quick check of `request` (its `model` is `name`) on `config` and `cap`: fails fast on a
/// model that is not configured, whose frames cannot be determined ([`seen_of`]), with no frame it
/// never saw, or a closer look asked for that cannot run. The frames are selected again when it
/// runs.
pub fn plan(
    config: &Config,
    mut request: RefineRequest,
    cap: &Capability,
    name: &str,
) -> Result<super::RefineJob> {
    request.validate()?;
    let entry = find_entry(config, name).ok_or_else(|| no_model(config, name))?;
    let dir = config.collect.resolved_review_dir(&config.data_root());
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        bail!("{why}");
    }
    let seen = seen_of(config, entry)?;
    let located = seen
        .located
        .clone()
        .context("the model's training record")?;
    let weights_warning = check_weights(&located)?;
    let classes = &config.collect.classes;
    let hidden = &config.collect.hidden_classes;
    let qs = select(&dir, classes, hidden, &request.frames, &seen);
    if let Some(why) = qs.s.refused {
        bail!("{why}");
    }
    let allow_large = request.allow_large || config.allow_large_downloads;
    let (look, _) = look_choice(config, &request, cap).map_err(|e| anyhow!(e))?;
    let look_env = if look {
        let paths = TrainPaths::for_config(config);
        let platform = Platform::current();
        let Some(accelerator) = cap.resolve_gpu(request.gpu.as_deref())? else {
            bail!(
                "the closer look needs an NVIDIA GPU or Apple silicon: {}",
                cap.summary()
            );
        };
        if !env::train_env_ready(&paths, platform, accelerator.lane()) && !allow_large {
            let (dl, _) = accelerator.lane().estimate();
            bail!(
                "the closer look at loose labels runs in the training environment (about {} \
                 download); rerun with --allow-large (or enable allow_large_downloads), or \
                 without the closer look (--no-tighten)",
                crate::resources::catalog::format_size(dl)
            );
        }
        super::super::pyenv::check_large(
            &[crate::training::overlaps::rfdetr_weights(&paths)?],
            allow_large,
        )?;
        Some(LookEnv {
            paths,
            platform,
            accelerator,
            timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        })
    } else {
        None
    };
    request.tighten = Some(look);
    request.allow_large = allow_large;
    request.model = Some(located.name.clone());
    let wanted = Wanted { classes, hidden };
    let id = super::super::service::new_run_id();
    Ok(super::RefineJob {
        dataset_dir: dir,
        classes: classes.clone(),
        hidden: hidden.clone(),
        work: super::super::training_dir(config).join("refine").join(&id),
        train: None,
        quick: Some(Quick {
            entry: entry.clone(),
            judge: model_judge(&seen, &wanted),
            imgsz: located.imgsz,
            unseen: qs.s.frames.len(),
            look: look_env,
            weights_warning,
        }),
        recipe_from: format!("the model {}, trained in run {}", located.name, located.run),
        predictor: None,
        config: Box::new(config.clone()),
        min_frames: MIN_UNSEEN_FRAMES,
        free_commit: crate::system_info::commit,
        id,
        request,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
pub(crate) mod tests;
